"""
Phase 8 — Admin Review API

Tasks 8.2–8.4: Spec review, experience review, staging queue, conflict resolve.

PREFIX: /approval

Endpoints (matching roadmap spec):
    GET  /approval/phones                               → phones ready for review
    GET  /approval/phone/{final_id}                     → full approval package
    GET  /approval/source-file/{type}/{id}              → proxy source file (stub)
    GET  /approval/evidence/{final_id}/{field_path}     → hover tooltip data
    POST /approval/spec/override                        → inline spec field edit
    POST /approval/resolve-conflict                     → manual conflict resolution
    POST /approval/spec/approve                         → set spec_human_approved=TRUE
    POST /approval/session/start                        → open review session
    POST /approval/session/close                        → close review session

    GET  /approval/experiences/{url_registry_id}        → all experiences for phone
    GET  /approval/experience/{experience_id}           → single experience entry
    POST /approval/experience/edit                      → edit experience_text or sentiment
    POST /approval/experience/suppress                  → mark is_suppressed=TRUE
    POST /approval/experience/restore                   → undo suppression
    POST /approval/experiences/approve                  → set experience_human_approved=TRUE

    GET  /approval/staging-queue                        → pending staging entries for phone
    POST /approval/staging/resolve                      → resolve one staging entry

Design notes:
    - All writes check pre-conditions (final_id exists, field not immutable, etc.)
    - The spec/override endpoint logs to admin_field_overrides AND patches final_json inline.
    - resolve-conflict writes human_override to merge_conflict_log then rechecks gate.
    - experience/edit writes admin_experience_overrides + patches phone_experiences.
    - Admin user identity comes from the X-Admin-User header (no auth in v1 — trusts caller).
    - staging/resolve is a best-effort synchronous write; the full atomic Postgres
      transaction described in the roadmap is implemented in Phase 10 (commit orchestrator).
"""

from __future__ import annotations

import asyncio
import copy
import logging
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, field_validator

from app.repositories.extraction_repository import (
    approve_experiences,
    approve_spec,
    close_admin_review_session,
    fetch_approval_package,
    fetch_evidence_for_field,
    fetch_experiences_for_phone,
    fetch_experience_by_id,
    fetch_flagged_conflicts_for_phone,
    fetch_pending_staging_entries,
    fetch_phones_ready_for_review,
    insert_admin_experience_override,
    insert_admin_field_override,
    insert_admin_review_session,
    resolve_staging_entry,
    set_experience_suppressed,
    update_conflict_resolution,
    update_experience_field,
    update_final_json_direct,
    update_gate_conditions,
)
from app.services.conflict_resolver import (
    _coerce_to_match,
    _count_nulls,
    _get_at_path,
    _parse_path,
    _set_at_path,
)
from app.services.validation_service import (
    run_pre_commit_validation,
    get_pre_commit_validation_status,
)
from app.services.commit_orchestrator import run_db_commit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CRIT-3 — API Key Authentication
# ---------------------------------------------------------------------------
# All approval endpoints require X-API-Key matching ADMIN_API_KEY from .env.
# The X-Admin-User header continues to function for audit logging after auth.

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _require_admin_auth(api_key: str | None = Security(_api_key_header)) -> None:
    """
    FastAPI dependency applied to all /approval routes.
    Checks the X-API-Key header against ADMIN_API_KEY from config.
    Rejects with HTTP 401 if missing or incorrect.
    """
    from app.core.config import ADMIN_API_KEY
    if not api_key or api_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: valid X-API-Key header required.",
        )


router = APIRouter(
    prefix="/approval",
    tags=["approval"],
    dependencies=[Depends(_require_admin_auth)],
)

# ---------------------------------------------------------------------------
# Header helper
# ---------------------------------------------------------------------------

_DEFAULT_ADMIN_USER = "unknown_admin"


def _admin_user(x_admin_user: str | None) -> str:
    return (x_admin_user or _DEFAULT_ADMIN_USER).strip() or _DEFAULT_ADMIN_USER


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class SessionStartRequest(BaseModel):
    final_id:      int
    initial_view:  str = "ui_view"

    @field_validator("initial_view")
    @classmethod
    def _valid_view(cls, v: str) -> str:
        if v not in ("ui_view", "json_view"):
            raise ValueError("initial_view must be 'ui_view' or 'json_view'")
        return v


class SessionCloseRequest(BaseModel):
    session_id: int
    outcome:    str

    @field_validator("outcome")
    @classmethod
    def _valid_outcome(cls, v: str) -> str:
        allowed = {"approved", "rejected", "deferred", "partial_edit"}
        if v not in allowed:
            raise ValueError(f"outcome must be one of {sorted(allowed)}")
        return v


class SpecOverrideRequest(BaseModel):
    session_id:           int
    final_id:             int
    field_path:           str
    new_value:            Any
    override_reason:      str | None = None
    resolves_conflict_id: int | None = None


class ResolveConflictRequest(BaseModel):
    session_id:     int
    conflict_id:    int
    final_id:       int
    resolution:     str       # 'kept_run_a' | 'kept_enrichment' | 'human_override'
    resolved_value: Any       # the value to write into final_json
    field_path:     str       # needed to patch final_json

    @field_validator("resolution")
    @classmethod
    def _valid_resolution(cls, v: str) -> str:
        allowed = {"kept_run_a", "kept_enrichment", "human_override"}
        if v not in allowed:
            raise ValueError(f"resolution must be one of {sorted(allowed)}")
        return v


class ApproveSpecRequest(BaseModel):
    final_id: int


class ExperienceEditRequest(BaseModel):
    session_id:      int
    experience_id:   int
    field:           str    # 'experience_text' | 'sentiment'
    new_value:       str
    override_reason: str | None = None

    @field_validator("field")
    @classmethod
    def _valid_field(cls, v: str) -> str:
        if v not in ("experience_text", "sentiment"):
            raise ValueError("field must be 'experience_text' or 'sentiment'")
        return v

    @field_validator("new_value")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("new_value must not be empty")
        return v

    @field_validator("new_value", mode="after")
    @classmethod
    def _normalise_sentiment(cls, v: str, info: Any) -> str:
        """
        P9-3: When field='sentiment', normalise the value so that 'positive',
        'POSITIVE', 'Positive' all produce 'Positive'. Validates against the
        allowed set to prevent DB enum constraint violations.
        """
        field_name = (info.data or {}).get("field", "")
        if field_name != "sentiment":
            return v
        normalised = v.strip().capitalize()
        allowed = {"Positive", "Negative", "Neutral", "Mixed"}
        if normalised not in allowed:
            raise ValueError(
                f"sentiment must be one of {sorted(allowed)} (got {v!r}). "
                f"Value is case-insensitive: 'positive' is accepted."
            )
        return normalised


class SuppressRequest(BaseModel):
    experience_id: int


class ApproveExperiencesRequest(BaseModel):
    final_id:        int
    url_registry_id: int


class StagingResolveRequest(BaseModel):
    staging_id:         int
    final_id:           int
    url_registry_id:    int
    resolution:         str          # 'inserted_new' | 'mapped_to_existing' | 'commit_as_null'
    resolved_lookup_id: int | None = None

    @field_validator("resolution")
    @classmethod
    def _valid_staging_resolution(cls, v: str) -> str:
        allowed = {"inserted_new", "mapped_to_existing", "commit_as_null"}
        if v not in allowed:
            raise ValueError(f"resolution must be one of {sorted(allowed)}")
        return v


# ---------------------------------------------------------------------------
# Task 8.2a — Phones ready for review
# ---------------------------------------------------------------------------

@router.get("/phones")
async def list_phones_for_review():
    """
    GET /approval/phones

    Returns all phones in the pipeline that have a final_merged_json row.
    Includes all five gate-condition columns and ready_for_commit flag.
    Frontend uses this to populate the admin queue list.
    """
    phones = await asyncio.to_thread(fetch_phones_ready_for_review)
    return {"phones": phones, "count": len(phones)}


# ---------------------------------------------------------------------------
# Task 8.2b — Full approval package
# ---------------------------------------------------------------------------

@router.get("/phone/{final_id}")
async def get_approval_package(final_id: int):
    """
    GET /approval/phone/{final_id}

    Returns the complete approval package for one phone:
        final_json, all gate condition flags, approver metadata.

    Also fetches flagged conflicts (for the conflict review panel) and
    pending staging entries (for the staging queue badge count).
    """
    row = await asyncio.to_thread(fetch_approval_package, final_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"final_id={final_id} not found.")

    normalized_id = row["normalized_id"]
    url_registry_id = row["url_registry_id"]

    flagged_conflicts, staging_entries = await asyncio.gather(
        asyncio.to_thread(fetch_flagged_conflicts_for_phone, normalized_id),
        asyncio.to_thread(fetch_pending_staging_entries, url_registry_id),
    )

    return {
        "approval_package":      row,
        "flagged_conflicts":     flagged_conflicts,
        "flagged_conflict_count": len(flagged_conflicts),
        "pending_staging":       staging_entries,
        "pending_staging_count": len(staging_entries),
    }


# ---------------------------------------------------------------------------
# Task 8.2c — Source file proxy (stub)
# ---------------------------------------------------------------------------

@router.get("/source-file/{source_type}/{source_id}")
async def get_source_file(source_type: str, source_id: int):
    """
    GET /approval/source-file/{source_type}/{source_id}

    Proxy endpoint for source file retrieval. Never exposes signed URLs directly.

    source_type: 'raw_scraped' | 'transcript'
    source_id:   raw_id or raw_transcript_id

    STUB: Returns a placeholder in v1. Full signed-URL proxy is a Phase 10 task.
    """
    return {
        "source_type": source_type,
        "source_id":   source_id,
        "status":      "stub — signed URL proxy not yet implemented.",
        "content":     None,
    }


# ---------------------------------------------------------------------------
# Task 8.2d — Evidence tooltip
# ---------------------------------------------------------------------------

def _derive_evidence_tooltip_fields(evidence: dict) -> tuple[str | None, str | None]:
    """
    Normalises Run A evidence_json for the tooltip API.

    Legacy Gemini path used key 'evidence'; LangExtract (spec_json_builder) uses
    'evidence_text'. Accept both. Derive source_label when only source_type + ids exist.
    """
    text = evidence.get("evidence")
    if text is None:
        text = evidence.get("evidence_text")

    label = evidence.get("source_label")
    if label:
        return text, label

    st = evidence.get("source_type")
    rid = evidence.get("raw_id")
    rtid = evidence.get("raw_transcript_id")
    if st == "scraped" and rid is not None:
        label = f"Scraped source (raw_id={rid})"
    elif st == "transcript" and rtid is not None:
        label = f"YouTube transcript (raw_transcript_id={rtid})"
    elif st == "scraped":
        label = "Scraped source"
    elif st == "transcript":
        label = "YouTube transcript"
    else:
        label = "Source recorded" if text else None

    return text, label


@router.get("/evidence/{final_id}/{field_path:path}")
async def get_field_evidence(final_id: int, field_path: str):
    """
    GET /approval/evidence/{final_id}/{field_path}

    Returns evidence_json entry for this field (hover tooltip data).
    field_path uses dot+bracket notation, e.g. 'camera_lenses[0].megapixels'

    If no evidence entry exists:
        Returns {evidence_text: null, source_label: 'No source recorded', confidence: null}
        Frontend renders greyed-out indicator — no crash.
    """
    row = await asyncio.to_thread(fetch_approval_package, final_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"final_id={final_id} not found.")

    # P8-8: pass final_id so fetch_evidence_for_field checks admin_field_overrides first
    evidence = await asyncio.to_thread(
        fetch_evidence_for_field, row["url_registry_id"], field_path, final_id
    )

    if evidence is None:
        return {
            "field_path":    field_path,
            "evidence_text": None,
            "source_type":   None,
            "source_label":  "No source recorded",
            "confidence":    None,
            "source_url":    None,
        }

    # Admin override branch returns a distinct shape (no LangExtract keys).
    if evidence.get("source_type") == "admin_override":
        return {
            "field_path":    field_path,
            "evidence_text": evidence.get("evidence"),
            "source_type":   evidence.get("source_type"),
            "source_label":  evidence.get("source_label"),
            "confidence":    evidence.get("confidence"),
            "source_url":    None,
        }

    ev_text, src_label = _derive_evidence_tooltip_fields(evidence)
    out: dict = {
        "field_path":    field_path,
        "evidence_text": ev_text,
        "source_type":   evidence.get("source_type"),
        "source_label":  src_label,
        "confidence":    evidence.get("confidence"),
        "source_url":    None,   # signed URL proxy — Phase 10
    }
    if "grounded" in evidence:
        out["grounded"] = evidence["grounded"]
    return out


# ---------------------------------------------------------------------------
# Task 8.2e — Inline spec field override
# ---------------------------------------------------------------------------

@router.post("/spec/override")
async def override_spec_field(
    req: SpecOverrideRequest,
    x_admin_user: str | None = Header(default=None),
):
    """
    POST /approval/spec/override

    Records an admin inline edit on a spec field:
      1. Fetches the current value from final_json (for previous_value audit).
      2. Inserts an admin_field_overrides row (immutable audit row).
      3. Patches final_merged_json.final_json with the new value (full blob replace).
      4. Recounts fields_remaining_null and updates it.

    evidence_quote is NOT a spec field — this endpoint only touches final_json keys.
    """
    admin = _admin_user(x_admin_user)

    row = await asyncio.to_thread(fetch_approval_package, req.final_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"final_id={req.final_id} not found.")

    url_registry_id: int = row["url_registry_id"]
    final_json: dict     = row.get("final_json") or {}

    previous_value = _get_at_path(final_json, req.field_path)

    # P8-6: validate the path is reachable before writing
    # A path is reachable if either:
    #   - the current value is non-None (path exists), OR
    #   - the parent container exists (path ends in a key not yet set but parent dict is there)
    # We use _get_at_path on the parent path as the check.
    # If the path has segments, strip the last to get the parent.
    path_parts = _parse_path(req.field_path)
    if len(path_parts) > 1:
        parent_path = req.field_path.rsplit(".", 1)[0] if "." in req.field_path else None
        # For bracket notation like 'variants[0].ram_capacity' — parent is 'variants[0]'
        # We check by attempting _get_at_path on all but the last segment
        parent_obj = final_json
        for part in path_parts[:-1]:
            if isinstance(part, int):
                if isinstance(parent_obj, list) and len(parent_obj) > part:
                    parent_obj = parent_obj[part]
                else:
                    parent_obj = None
                    break
            elif isinstance(parent_obj, dict):
                parent_obj = parent_obj.get(part)
            else:
                parent_obj = None
                break
        if parent_obj is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"field_path={req.field_path!r} is unreachable: parent container "
                    f"does not exist in final_json. Check path notation."
                ),
            )

    # P8-7: coerce new_value to match the existing field's type to prevent re-introducing
    # string-vs-numeric mismatches that Phase 7's _coerce_to_match already fixed.
    coerced_value = _coerce_to_match(previous_value, req.new_value)

    # Log the override (append-only audit row)
    override_payload = {
        "session_id":           req.session_id,
        "final_id":             req.final_id,
        "url_registry_id":      url_registry_id,
        "field_path":           req.field_path,
        "previous_value":       previous_value,
        "new_value":            coerced_value,
        "override_reason":      req.override_reason,
        "resolves_conflict_id": req.resolves_conflict_id,
    }
    override_id = await asyncio.to_thread(insert_admin_field_override, override_payload)

    # Patch final_json in-place and persist the entire blob
    updated_json = copy.deepcopy(final_json)
    _set_at_path(updated_json, req.field_path, coerced_value)

    # Count remaining nulls after the patch
    fields_remaining_null = _count_nulls(updated_json)

    await asyncio.to_thread(update_final_json_direct, req.final_id, updated_json)
    await asyncio.to_thread(
        update_gate_conditions,
        req.final_id,
        {"fields_remaining_null": fields_remaining_null},
    )

    logger.info(
        "spec/override: final_id=%d field=%r admin=%s override_id=%d coerced=%r",
        req.final_id, req.field_path, admin, override_id, coerced_value,
    )
    return {
        "override_id":           override_id,
        "field_path":            req.field_path,
        "previous_value":        previous_value,
        "new_value":             coerced_value,
        "fields_remaining_null": fields_remaining_null,
    }


# ---------------------------------------------------------------------------
# Task 8.2f — Manual conflict resolution
# ---------------------------------------------------------------------------

@router.post("/resolve-conflict")
async def resolve_conflict(
    req: ResolveConflictRequest,
    x_admin_user: str | None = Header(default=None),
):
    """
    POST /approval/resolve-conflict

    Resolves a flagged merge conflict with admin's chosen outcome:
      1. Updates merge_conflict_log with resolution='human_override'.
      2. Patches final_json with the resolved_value if resolution != 'kept_run_a'
         (kept_run_a means Run A value stays — no patch needed).
      3. Re-checks whether any flagged conflicts remain; updates
         has_unresolved_conflicts on final_merged_json accordingly.

    The admin_field_overrides row (if needed) is logged via spec/override —
    this endpoint handles the conflict log only.
    """
    admin = _admin_user(x_admin_user)

    await asyncio.to_thread(
        update_conflict_resolution,
        req.conflict_id,
        req.resolution,
        req.resolved_value,
        admin,
    )

    # Fetch the package to patch final_json and log the override
    row = await asyncio.to_thread(fetch_approval_package, req.final_id)
    still_has_conflicts: bool = False
    updated_null_count: int | None = None

    if row:
        final_json       = row.get("final_json") or {}
        normalized_id: int = row["normalized_id"]
        url_registry_id: int = row["url_registry_id"]

        if req.resolution != "kept_run_a":
            previous_value = _get_at_path(final_json, req.field_path)
            coerced_value  = _coerce_to_match(previous_value, req.resolved_value)

            # P9-Missing: validate field_path before writing, same as override_spec_field
            path_parts = _parse_path(req.field_path)
            if len(path_parts) > 1:
                parent_obj: Any = final_json
                for part in path_parts[:-1]:
                    if isinstance(part, int):
                        parent_obj = parent_obj[part] if isinstance(parent_obj, list) and len(parent_obj) > part else None
                    elif isinstance(parent_obj, dict):
                        parent_obj = parent_obj.get(part)
                    else:
                        parent_obj = None
                    if parent_obj is None:
                        break
                if parent_obj is None:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"field_path={req.field_path!r} is unreachable in final_json "
                            f"— conflict marked resolved in DB but final_json was NOT patched. "
                            f"Check path notation."
                        ),
                    )

            updated = copy.deepcopy(final_json)
            _set_at_path(updated, req.field_path, coerced_value)

            # P8-2: log to admin_field_overrides so amber indicator appears in UI
            override_payload = {
                "session_id":           req.session_id,
                "final_id":             req.final_id,
                "url_registry_id":      url_registry_id,
                "field_path":           req.field_path,
                "previous_value":       previous_value,
                "new_value":            coerced_value,
                "override_reason":      f"Conflict resolved: {req.resolution}",
                "resolves_conflict_id": req.conflict_id,
            }
            await asyncio.to_thread(insert_admin_field_override, override_payload)
            await asyncio.to_thread(update_final_json_direct, req.final_id, updated)

            # P8-4: recount nulls after the patch
            updated_null_count = _count_nulls(updated)

        # Re-check remaining flagged conflicts
        remaining = await asyncio.to_thread(
            fetch_flagged_conflicts_for_phone, normalized_id
        )
        still_has_conflicts = len(remaining) > 0

        gate_patch: dict = {"has_unresolved_conflicts": still_has_conflicts}
        if updated_null_count is not None:
            gate_patch["fields_remaining_null"] = updated_null_count
        await asyncio.to_thread(update_gate_conditions, req.final_id, gate_patch)

    logger.info(
        "resolve-conflict: conflict_id=%d resolution=%s admin=%s has_conflicts=%s",
        req.conflict_id, req.resolution, admin, still_has_conflicts,
    )
    return {
        "conflict_id":              req.conflict_id,
        "resolution":               req.resolution,
        "has_unresolved_conflicts": still_has_conflicts,
        "fields_remaining_null":    updated_null_count,
    }


# ---------------------------------------------------------------------------
# Task 8.2g — Spec approval
# ---------------------------------------------------------------------------

@router.post("/spec/approve")
async def approve_spec_endpoint(
    req: ApproveSpecRequest,
    x_admin_user: str | None = Header(default=None),
):
    """
    POST /approval/spec/approve

    Sets spec_human_approved=TRUE. Admin must not have any unresolved conflicts
    or pending staging values for the gate to unlock (checked client-side and
    re-verified by the commit orchestrator in Phase 10).

    This endpoint is permissive — it sets the flag unconditionally.
    Phase 10 will re-verify all five conditions before opening the DB transaction.
    """
    admin = _admin_user(x_admin_user)

    # P8-10: fetch package to warn when approving with open gate conditions
    pkg = await asyncio.to_thread(fetch_approval_package, req.final_id)
    if not pkg:
        raise HTTPException(status_code=404, detail=f"final_id={req.final_id} not found.")
    if pkg.get("has_unresolved_conflicts") or (pkg.get("pending_staging_values") or 0) > 0:
        logger.warning(
            "spec/approve: final_id=%d approved with open gate conditions — "
            "has_unresolved_conflicts=%s pending_staging_values=%d admin=%s",
            req.final_id,
            pkg.get("has_unresolved_conflicts"),
            pkg.get("pending_staging_values") or 0,
            admin,
        )

    await asyncio.to_thread(approve_spec, req.final_id, admin)
    logger.info("spec/approve: final_id=%d admin=%s", req.final_id, admin)
    return {"final_id": req.final_id, "spec_human_approved": True, "approved_by": admin}


# ---------------------------------------------------------------------------
# Task 8.2h — Review session management
# ---------------------------------------------------------------------------

@router.post("/session/start")
async def start_review_session(
    req: SessionStartRequest,
    x_admin_user: str | None = Header(default=None),
):
    """
    POST /approval/session/start

    Opens a new admin_review_sessions row. Returns session_id for use in
    all subsequent override and edit calls within this review window.
    """
    admin = _admin_user(x_admin_user)

    row = await asyncio.to_thread(fetch_approval_package, req.final_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"final_id={req.final_id} not found.")

    session_payload = {
        "final_id":        req.final_id,
        "url_registry_id": row["url_registry_id"],
        "admin_user":      admin,
        "initial_view":    req.initial_view,
    }
    session_id = await asyncio.to_thread(insert_admin_review_session, session_payload)
    logger.info("session/start: session_id=%d final_id=%d admin=%s", session_id, req.final_id, admin)
    return {"session_id": session_id, "final_id": req.final_id, "admin_user": admin}


@router.post("/session/close")
async def close_review_session(
    req: SessionCloseRequest,
    x_admin_user: str | None = Header(default=None),
):
    """
    POST /approval/session/close

    Closes the review session with an outcome.
    outcome: 'approved' | 'rejected' | 'deferred' | 'partial_edit'
    """
    admin = _admin_user(x_admin_user)
    await asyncio.to_thread(close_admin_review_session, req.session_id, req.outcome)
    logger.info(
        "session/close: session_id=%d outcome=%s admin=%s",
        req.session_id, req.outcome, admin,
    )
    return {"session_id": req.session_id, "outcome": req.outcome}


# ---------------------------------------------------------------------------
# Task 8.3 — Experience review endpoints
# ---------------------------------------------------------------------------

@router.get("/experiences/{url_registry_id}")
async def list_experiences(url_registry_id: int):
    """
    GET /approval/experiences/{url_registry_id}

    Returns all phone_experiences grouped by category_id.
    Colour-coding signals: admin_edited=True → amber, is_suppressed=True → grey.
    """
    experiences = await asyncio.to_thread(
        fetch_experiences_for_phone, url_registry_id
    )
    # P8-9: defaultdict moved to module top level
    grouped: dict[int, list] = defaultdict(list)
    for exp in experiences:
        grouped[exp["category_id"]].append(exp)

    return {
        "url_registry_id": url_registry_id,
        "total":           len(experiences),
        "by_category":     dict(grouped),
    }


@router.get("/experience/{experience_id}")
async def get_experience(experience_id: int):
    """
    GET /approval/experience/{experience_id}

    Returns one experience entry with full detail including evidence_quote.
    """
    exp = await asyncio.to_thread(fetch_experience_by_id, experience_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"experience_id={experience_id} not found.")
    return exp


@router.post("/experience/edit")
async def edit_experience(
    req: ExperienceEditRequest,
    x_admin_user: str | None = Header(default=None),
):
    """
    POST /approval/experience/edit

    Edits experience_text or sentiment on a phone_experiences row.
    evidence_quote is immutable — returns 400 if attempted.

    Actions:
      1. Fetches current value for previous_value audit.
      2. Inserts admin_experience_overrides row (append-only audit).
      3. Updates phone_experiences field + sets admin_edited=TRUE.
    """
    admin = _admin_user(x_admin_user)

    if req.field == "evidence_quote":
        raise HTTPException(
            status_code=400,
            detail="evidence_quote is immutable. It is a permanent audit artifact and cannot be edited.",
        )

    exp = await asyncio.to_thread(fetch_experience_by_id, req.experience_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"experience_id={req.experience_id} not found.")

    previous_value = exp.get(req.field)

    # Log override (audit)
    override_payload = {
        "session_id":      req.session_id,
        "experience_id":   req.experience_id,
        "url_registry_id": exp["url_registry_id"],
        "field_edited":    req.field,
        "previous_value":  str(previous_value) if previous_value is not None else None,
        "new_value":       req.new_value,
        "override_reason": req.override_reason,
    }
    exp_override_id = await asyncio.to_thread(
        insert_admin_experience_override, override_payload
    )

    # Apply the change
    await asyncio.to_thread(update_experience_field, req.experience_id, req.field, req.new_value)

    logger.info(
        "experience/edit: experience_id=%d field=%r admin=%s exp_override_id=%d",
        req.experience_id, req.field, admin, exp_override_id,
    )
    return {
        "exp_override_id": exp_override_id,
        "experience_id":   req.experience_id,
        "field":           req.field,
        "previous_value":  previous_value,
        "new_value":       req.new_value,
    }


@router.post("/experience/suppress")
async def suppress_experience(
    req: SuppressRequest,
    x_admin_user: str | None = Header(default=None),
):
    """
    POST /approval/experience/suppress

    Marks is_suppressed=TRUE. Suppressed entries are excluded from the DB commit
    and from vector embeddings. The row is never physically deleted (audit artifact).
    """
    admin = _admin_user(x_admin_user)
    exp = await asyncio.to_thread(fetch_experience_by_id, req.experience_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"experience_id={req.experience_id} not found.")

    await asyncio.to_thread(set_experience_suppressed, req.experience_id, True)
    logger.info(
        "experience/suppress: experience_id=%d admin=%s", req.experience_id, admin
    )
    return {"experience_id": req.experience_id, "is_suppressed": True}


@router.post("/experience/restore")
async def restore_experience(
    req: SuppressRequest,
    x_admin_user: str | None = Header(default=None),
):
    """
    POST /approval/experience/restore

    Sets is_suppressed=FALSE (undo suppression).
    """
    admin = _admin_user(x_admin_user)
    exp = await asyncio.to_thread(fetch_experience_by_id, req.experience_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"experience_id={req.experience_id} not found.")

    await asyncio.to_thread(set_experience_suppressed, req.experience_id, False)
    logger.info(
        "experience/restore: experience_id=%d admin=%s", req.experience_id, admin
    )
    return {"experience_id": req.experience_id, "is_suppressed": False}


@router.post("/experiences/approve")
async def approve_experiences_endpoint(
    req: ApproveExperiencesRequest,
    x_admin_user: str | None = Header(default=None),
):
    """
    POST /approval/experiences/approve

    Sets experience_human_approved=TRUE and experience_entries_reviewed=TRUE.
    These are commit gate conditions 4 and 5. Both are set together — admin
    cannot approve without having reviewed entries.
    """
    admin = _admin_user(x_admin_user)
    await asyncio.to_thread(
        approve_experiences, req.final_id, req.url_registry_id, admin
    )
    logger.info(
        "experiences/approve: final_id=%d url_registry_id=%d admin=%s",
        req.final_id, req.url_registry_id, admin,
    )
    return {
        "final_id":                    req.final_id,
        "experience_human_approved":   True,
        "experience_entries_reviewed": True,
        "approved_by":                 admin,
    }


# ---------------------------------------------------------------------------
# Task 8.4 — Staging queue endpoints
# ---------------------------------------------------------------------------

@router.get("/staging-queue")
async def get_staging_queue(url_registry_id: int):
    """
    GET /approval/staging-queue?url_registry_id={id}

    Returns all pending lookup_value_staging rows for this phone,
    grouped by target_lookup_table for the admin staging panel.
    """
    entries = await asyncio.to_thread(
        fetch_pending_staging_entries, url_registry_id
    )
    # P8-9: defaultdict moved to module top level
    grouped: dict[str, list] = defaultdict(list)
    for e in entries:
        grouped[e["target_lookup_table"]].append(e)

    return {
        "url_registry_id":  url_registry_id,
        "pending_count":    len(entries),
        "by_lookup_table":  dict(grouped),
    }


@router.post("/staging/resolve")
async def resolve_staging(
    req: StagingResolveRequest,
    x_admin_user: str | None = Header(default=None),
):
    """
    POST /approval/staging/resolve

    Resolves one staging entry and updates the pending_staging_values counter
    on final_merged_json.

    resolution options:
        'inserted_new'      — a new lookup row was inserted; resolved_lookup_id required
        'mapped_to_existing'— mapped to an existing lookup row; resolved_lookup_id required
        'commit_as_null'    — admin chose to commit the field as NULL (7-day stale path)

    The full atomic transaction described in the roadmap (jsonb_set patches on
    normalized_spec_json and final_merged_json) is implemented in Phase 10.
    This endpoint handles the status update and counter decrement only.
    """
    admin = _admin_user(x_admin_user)

    if req.resolution in ("inserted_new", "mapped_to_existing") and req.resolved_lookup_id is None:
        raise HTTPException(
            status_code=400,
            detail=f"resolved_lookup_id is required for resolution='{req.resolution}'.",
        )

    # Mark the staging entry resolved
    await asyncio.to_thread(
        resolve_staging_entry,
        req.staging_id,
        req.resolution,
        req.resolved_lookup_id,
    )

    # Recount pending staging entries to recalculate the gate counter
    remaining = await asyncio.to_thread(
        fetch_pending_staging_entries, req.url_registry_id
    )
    pending_count = len(remaining)

    await asyncio.to_thread(
        update_gate_conditions,
        req.final_id,
        {"pending_staging_values": pending_count},
    )

    logger.info(
        "staging/resolve: staging_id=%d resolution=%s remaining=%d admin=%s",
        req.staging_id, req.resolution, pending_count, admin,
    )
    return {
        "staging_id":            req.staging_id,
        "resolution":            req.resolution,
        "resolved_lookup_id":    req.resolved_lookup_id,
        "pending_staging_values": pending_count,
    }


# ===========================================================================
# Phase 9 — Pre-Commit Validation (Layer 3)
# ===========================================================================


class PreCommitValidateRequest(BaseModel):
    final_id: int


@router.post("/validate-pre-commit")
async def validate_pre_commit(
    req: PreCommitValidateRequest,
    x_admin_user: str | None = Header(default=None),
) -> dict:
    """
    POST /approval/validate-pre-commit

    Runs Phase 9 Layer 3 backend validation on the current state of final_json.
    This endpoint is called automatically by the commit orchestrator (Phase 10)
    before the DB transaction opens. It may also be called manually by the admin
    to preview validation state before clicking the commit button.

    Checks (in order):
      1. Full Pydantic model validation on post-admin-edit final_json.
      2. FK pre-resolution: every SCALAR_FK_MAP field must be NULL or in LOOKUP_CACHE.
      3. Five-condition gate re-read atomically from live DB.
      4. Run B final: all non-suppressed experiences have non-null text + confidence >= 0.50.
      5. Structural consistency: foldable brand / display count mismatch (warnings only).

    Returns:
      {
        "passed":        bool,
        "errors":        list[dict],   # hard failures — will block commit
        "warnings":      list[dict],   # advisory — will not block commit
        "commit_val_id": int,
      }

    HTTP 400 on passed=False. HTTP 200 on passed=True.
    Both responses include the full error/warning lists.
    """
    admin = _admin_user(x_admin_user)

    # Fetch package to get url_registry_id
    pkg = await asyncio.to_thread(fetch_approval_package, req.final_id)
    if not pkg:
        raise HTTPException(
            status_code=404,
            detail=f"final_id={req.final_id} not found.",
        )

    url_registry_id: int = pkg["url_registry_id"]

    result = await run_pre_commit_validation(
        final_id=req.final_id,
        url_registry_id=url_registry_id,
    )

    logger.info(
        "validate-pre-commit: final_id=%d passed=%s errors=%d admin=%s commit_val_id=%d",
        req.final_id, result["passed"], len(result["errors"]),
        admin, result["commit_val_id"],
    )

    if not result["passed"]:
        # 400 with the full error payload so the admin UI can surface specific failures
        raise HTTPException(
            status_code=400,
            detail=result,
        )

    return result


@router.get("/validate-pre-commit/{final_id}")
async def get_validate_pre_commit_status(
    final_id: int,
    x_admin_user: str | None = Header(default=None),
) -> dict:
    """
    GET /approval/validate-pre-commit/{final_id}

    Returns the most recent commit_validation_runs row for this final_id.
    Used by the admin UI to show the current Layer 3 state without re-running.
    Returns HTTP 404 if no validation has been run yet.
    """
    row = await get_pre_commit_validation_status(final_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No validation run found for final_id={final_id}.",
        )
    return row


# ===========================================================================
# PHASE 10 — DB Commit
# POST /approval/commit
# ===========================================================================


class CommitRequest(BaseModel):
    final_id: int
    session_id: int | None = None


@router.post("/commit")
async def commit_to_db(
    req: CommitRequest,
    x_admin_user: str | None = Header(default=None),
) -> dict:
    """
    POST /approval/commit

    Phase 10 — Atomic DB Commit.

    Orchestrates the full write of final_merged_json into mobile_specs schema
    and marks non-suppressed phone_experiences as committed.

    PRE-COMMIT (400 on failure):
      Runs Phase 9 Layer 3 backend validation (run_pre_commit_validation).
      If validation fails → 400 with errors payload. No writes occur.

    ON SUCCESS:
      Returns {
        "success":           True,
        "model_id":          int,       # mobile_specs.phones.model_id
        "commit_run_id":     int,       # pipeline.db_commit_runs PK
        "tables_written":    list[str], # all tables written in order
        "rows_inserted":     int,
        "unresolved_fields": list[str], # FK fields left NULL due to cache misses
      }
      url_registry.status is set to 'stored_mainDB'.
      Run C (inference_engine) is fired asynchronously.

    ON FAILURE:
      Returns HTTP 500 with error detail.
      pipeline.db_commit_runs row is updated to status='failed'.
      Safe to retry — all writes are idempotent (upsert semantics).
    """
    admin = _admin_user(x_admin_user)

    logger.info(
        "commit_to_db: final_id=%d session_id=%s admin=%s",
        req.final_id, req.session_id, admin,
    )

    try:
        result = await run_db_commit(
            final_id=req.final_id,
            session_id=req.session_id,
        )
    except ValueError as exc:
        # Pre-commit validation failure or bad data (not a server error)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(
            "commit_to_db: unexpected error final_id=%d: %s", req.final_id, exc
        )
        raise HTTPException(
            status_code=500,
            detail=f"Commit failed: {exc}",
        )

    logger.info(
        "commit_to_db: SUCCESS final_id=%d model_id=%d commit_run_id=%d "
        "rows_inserted=%d unresolved=%d admin=%s",
        req.final_id,
        result["model_id"],
        result["commit_run_id"],
        result["rows_inserted"],
        len(result["unresolved_fields"]),
        admin,
    )

    return result


# ===========================================================================
# Phase L6 — LangExtract Visualization Endpoint
# GET /approval/visualization/{run_type}/{run_id}
# ===========================================================================


@router.get("/visualization/{run_type}/{run_id}")
async def get_lx_visualization(run_type: str, run_id: int):
    """
    GET /approval/visualization/{run_type}/{run_id}

    Fetches the stored JSONL for a LangExtract run and returns an HTML page
    that visualises the annotated document (evidence highlights per extraction).

    run_type:  "spec"  → spec_extraction_runs.jsonl_path
               "exp"   → experience_extraction_runs.jsonl_path

    run_id:    PK of the run row (extraction_run_id or exp_run_id)

    Design — generate on demand, not at storage time:
      - JSONL is stored post-extraction by langextract_run_a / langextract_run_b.
      - This endpoint fetches the JSONL and renders HTML via lx.io.create_html_from_jsonl.
      - The HTML is returned as text/html, suitable for display in an <iframe>.
      - If jsonl_path is NULL (JSONL upload failed), returns 404 with a diagnostic message.

    Returns:
        text/html — rendered annotated document

    Raises:
        HTTP 400: If run_type is not "spec" or "exp".
        HTTP 404: If run_id not found or jsonl_path is NULL.
        HTTP 500: On storage fetch or HTML generation failure.
    """
    import langextract as lx
    import tempfile
    import os
    from fastapi.responses import HTMLResponse
    from app.repositories.extraction_repository import fetch_run_jsonl_path
    from app.services.storage_service import fetch_file_content

    if run_type not in ("spec", "exp"):
        raise HTTPException(
            status_code=400,
            detail=f"run_type must be 'spec' or 'exp', got {run_type!r}.",
        )

    # Fetch jsonl_path from DB
    try:
        jsonl_path = await asyncio.to_thread(fetch_run_jsonl_path, run_type, run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception(
            "get_lx_visualization: DB error for run_type=%s run_id=%d: %s",
            run_type, run_id, exc,
        )
        raise HTTPException(status_code=500, detail=f"DB error: {exc}")

    if jsonl_path is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No JSONL visualization available for {run_type}_run_id={run_id}. "
                "The JSONL upload failed at extraction time, or this run predates "
                "the LangExtract migration. Re-run extraction to generate visualization."
            ),
        )

    # Fetch JSONL content from Supabase Storage
    try:
        jsonl_content: str = await fetch_file_content(jsonl_path)
    except Exception as exc:
        logger.exception(
            "get_lx_visualization: storage fetch failed for path=%s: %s",
            jsonl_path, exc,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch JSONL from storage ({jsonl_path!r}): {exc}",
        )

    # Generate HTML on demand — write JSONL to tmpfile, call lx.io.create_html_from_jsonl
    def _render_html() -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_file = os.path.join(tmpdir, f"{run_type}_{run_id}.jsonl")
            with open(jsonl_file, "w", encoding="utf-8") as fh:
                fh.write(jsonl_content)
            html_str = lx.io.create_html_from_jsonl(jsonl_file)
            return html_str

    try:
        html = await asyncio.to_thread(_render_html)
    except Exception as exc:
        logger.exception(
            "get_lx_visualization: HTML generation failed for "
            "run_type=%s run_id=%d jsonl_path=%s: %s",
            run_type, run_id, jsonl_path, exc,
        )
        raise HTTPException(
            status_code=500,
            detail=f"HTML generation failed: {exc}",
        )

    logger.info(
        "get_lx_visualization: served run_type=%s run_id=%d (%d chars JSONL, %d chars HTML)",
        run_type, run_id, len(jsonl_content), len(html),
    )
    return HTMLResponse(content=html, status_code=200)
