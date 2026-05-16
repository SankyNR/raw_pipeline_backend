"""
Extraction Pipeline API Router (app/api/extraction.py)

Phase 1 — Pre-Extraction Validation:
    POST /extraction/validate
    GET  /extraction/validation-status/{canonical_url_id}

Phase 2 — Extraction:
    POST /extraction/run-a    — Spec extraction. Manual or auto mode.
    POST /extraction/run-b    — Experience extraction. Manual or auto mode.
    POST /extraction/run-all  — Both pipelines in parallel. Manual or auto mode.

Phase 4 — Normalisation:
    POST /extraction/normalise
    POST /extraction/cache/refresh
    schema_version

Phase 5 — Gap Analysis:
    POST /extraction/gaps

Phase 6 — Enrichment:
    POST /extraction/enrich

Phase 7 — Conflict Resolution:
    POST /extraction/resolve-conflicts

Phase 7.5 — Pre-UI Validation:
    POST /extraction/validate-pre-ui
    GET  /extraction/validation-status-pre-ui/{final_id}
"""

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.constants import GATE_ERROR_PREFIX as _GATE_ERROR_PREFIX
from app.services.validation_service import (
    run_pre_extraction_validation,
    get_validation_status,
    run_pre_ui_validation,
    get_pre_ui_validation_status,
)
from app.services.extraction_run_a import run_spec_extraction
from app.services.extraction_run_b import run_experience_extraction_batch
from app.services.normalizer import run_normalisation
from app.services.gap_analyzer import detect_missing_fields
from app.services.enrichment_orchestrator import run_enrichment
from app.services.conflict_resolver import (
    detect_and_resolve_conflicts,
    build_final_merged_json,
    promote_normalized_to_final,
)
from app.repositories.pipeline_run_repository import (
    create_pipeline_run,
    fetch_pipeline_run,
    fetch_recent_pipeline_runs,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/extraction", tags=["extraction"])

# FIX-11: Rate limiter — key by remote IP.
# Instantiated here so main.py can attach it to app.state.limiter.
limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ValidateRequest(BaseModel):
    """
    Request model for POST /extraction/validate.

    Accepts brand + model_name instead of canonical_url_id.
    The backend resolves canonical_url_id internally via the GSMArena anchor.
    Both fields are stripped of whitespace before use.
    """
    brand:      str
    model_name: str


class ValidationStatusResponse(BaseModel):
    validation_id: int
    canonical_url_id: int
    validated_at: str
    validated_by: str
    total_urls_registered: int
    urls_not_attempted: int
    urls_scraped_success: int
    official_url_scraped: bool | None
    gsmarena_scraped: bool | None
    youtube_search_done: bool
    transcript_available: bool
    can_proceed: bool
    blocking_reasons: list[dict]
    warnings: list[dict]
    unscraped_url_ids: list[int] = []


# ---------------------------------------------------------------------------
# Task 1.1 — POST /extraction/validate
# ---------------------------------------------------------------------------

@router.post("/validate", response_model=ValidationStatusResponse)
async def validate_phone(body: ValidateRequest):
    """
    Runs pre-extraction validation for the phone identified by brand + model_name.

    Resolves canonical_url_id (GSMArena anchor) internally, then runs the full
    validation check and writes a new pre_extraction_validation row.

    Checks:
    - All registered URLs have been attempted (no unattempted scrapes)
    - At least one source file was scraped successfully
    - A YouTube search has been run (success_zero or success_with_results)

    Blockers (any one prevents can_proceed):
      - urls_not_attempted > 0
      - urls_scraped_success == 0
      - youtube_search_done == False

    Warnings (surfaced, not blocking):
      - OEM official URL registered but scrape failed
      - No transcript fetched

    Returns the validation record. Multiple runs are allowed — latest is
    authoritative. Check can_proceed before triggering Run A.

    Raises:
        HTTP 400: If brand or model_name is empty.
        HTTP 404: If phone not found in url_registry (no GSMArena row).
        HTTP 500: On unexpected DB or service error.
    """
    from app.repositories.extraction_repository import fetch_canonical_id_by_brand_model

    brand, model_name = _normalize_brand_model(body.brand, body.model_name)

    # Resolve canonical_url_id from brand + model_name
    try:
        canonical_url_id = await asyncio.to_thread(
            fetch_canonical_id_by_brand_model, brand, model_name
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": "PHONE_NOT_FOUND", "message": str(exc)},
        )

    try:
        record = await run_pre_extraction_validation(
            canonical_url_id=canonical_url_id,
            validated_by="admin_manual",
        )
        return record
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": "PHONE_NOT_FOUND", "message": str(exc)},
        )
    except Exception as exc:
        logger.exception(
            "validate_phone: unexpected error for brand=%r model_name=%r",
            brand, model_name,
        )
        raise HTTPException(
            status_code=500,
            detail={"error": "INTERNAL_ERROR", "message": str(exc)},
        )


# ---------------------------------------------------------------------------
# Task 1.1 — GET /extraction/validation-status/{canonical_url_id}
# ---------------------------------------------------------------------------

@router.get(
    "/validation-status/{canonical_url_id}",
    response_model=ValidationStatusResponse,
)
async def get_phone_validation_status(canonical_url_id: int):
    """
    Returns the most recent validation record for this phone.

    Use this to check the current validation state without re-running
    the full validation logic.

    Raises:
        HTTP 404: If no validation has been run yet for this phone.
        HTTP 500: On unexpected DB error.
    """
    try:
        record = await get_validation_status(canonical_url_id)
    except Exception as exc:
        logger.exception(
            "get_phone_validation_status: error for canonical_url_id=%d",
            canonical_url_id,
        )
        raise HTTPException(status_code=500, detail=str(exc))

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No validation record found for canonical_url_id={canonical_url_id}. "
                "Run POST /extraction/validate first."
            ),
        )
    return record


# ---------------------------------------------------------------------------
# FIX-10: Unified extraction request / response models
# ---------------------------------------------------------------------------

class ExtractionRequest(BaseModel):
    """
    Unified request model for all three extraction endpoints.

    MODES — controlled by auto_resolve flag:

    AUTO mode (auto_resolve=True):
        Required:  brand, model_name
        Forbidden: raw_source_ids, raw_transcript_ids
        Behaviour: Backend resolves canonical_url_id and ALL available source IDs
                   automatically from the DB.

    MANUAL mode (auto_resolve=False):
        Required:  brand, model_name, raw_source_ids (for run-a and run-all),
                   raw_transcript_ids (for run-b)
        Behaviour: Backend resolves canonical_url_id from brand+model_name, then
                   validates that all provided IDs belong to that phone.

    NOTE: canonical_url_id is NEVER accepted from the caller. It is always resolved
    internally. This removes ambiguity and prevents callers from referencing the
    wrong anchor row.

    STRICT RULES (enforced in _validate_extraction_request helper):
        - auto_resolve=True  + raw_source_ids or raw_transcript_ids present → HTTP 400
        - auto_resolve=False + raw_source_ids empty (run-a, run-all)        → HTTP 400
        - auto_resolve=False + raw_transcript_ids empty (run-b)             → HTTP 400
        - brand or model_name missing or empty string                       → HTTP 400
        - brand/model_name pair not found in DB (no GSMArena row)           → HTTP 404
    """
    model_config = {"protected_namespaces": ()}

    # Mode toggle — required, no default to force explicit caller decision
    auto_resolve: bool

    # Both modes — always required
    brand:      str
    model_name: str

    # Manual mode only — must be empty/absent in auto mode
    raw_source_ids:    list[int] = []
    raw_transcript_ids: list[int] = []

    # Shared
    schema_version: str = "v2"


class RunAResponse(BaseModel):
    success: bool
    output_id: int | None
    run_id: int | None = None    # B6: None for failure path — -1 is not a valid DB ID
    pipeline_run_id: str | None = None  # Global tracking UUID (pipeline.pipeline_runs)
    message: str
    failed_source_ids: list[int] = []


class RunBResult(BaseModel):
    success: bool
    aggregation_run_id: int | None = None       # Stage 2 run ID (two-stage system)
    exp_run_id: int | None = None               # Legacy field — kept for compat, now unused
    pipeline_run_id: str | None = None
    experiences_extracted: int = 0              # Final Stage 2 output count
    new_transcripts: int = 0                    # Transcripts that ran Stage 1 fresh
    reused_transcripts: int = 0                 # Transcripts whose candidates were reused
    transcripts_processed: int = 0              # new_transcripts + reused_transcripts
    transcripts_failed: int = 0
    message: str
    error: str | None = None


class RunAllResponse(BaseModel):
    run_a: RunAResponse
    run_b: RunBResult
    both_succeeded: bool


def _normalize_brand_model(brand: str, model_name: str) -> tuple[str, str]:
    """
    Strips leading/trailing whitespace from brand and model_name.

    Does NOT change case. The DB stores brand/model_name in their canonical
    casing (e.g. "Samsung", "Galaxy S25 Ultra"). The admin UI populates dropdowns
    directly from the DB so casing will always match in normal use. Stripping
    whitespace guards against copy-paste or direct API calls with accidental spaces.

    Called at the top of every handler before any DB interaction.
    """
    return brand.strip(), model_name.strip()


def _validate_extraction_request(
    body: ExtractionRequest,
    requires_sources: bool = True,
    requires_transcripts: bool = False,
) -> None:
    """
    Validates ExtractionRequest for mode consistency. Raises HTTPException on failure.

    brand and model_name are always required — enforced here explicitly so the
    error message is clear even though Pydantic would also reject empty strings.

    Args:
        requires_sources:     True for run-a and run-all (scraped markdown needed).
        requires_transcripts: True for run-b only (transcript data needed).
    """
    # ── Both modes: brand + model_name must be non-empty strings ────────────
    if not body.brand or not body.brand.strip():
        raise HTTPException(
            status_code=400,
            detail={
                "error":   "INVALID_REQUEST",
                "message": "'brand' is required and must be a non-empty string.",
            },
        )
    if not body.model_name or not body.model_name.strip():
        raise HTTPException(
            status_code=400,
            detail={
                "error":   "INVALID_REQUEST",
                "message": "'model_name' is required and must be a non-empty string.",
            },
        )

    if body.auto_resolve:
        # ── AUTO MODE: raw ID lists must be absent ───────────────────────────
        if body.raw_source_ids or body.raw_transcript_ids:
            raise HTTPException(
                status_code=400,
                detail={
                    "error":   "INVALID_REQUEST",
                    "message": (
                        "auto_resolve=true is mutually exclusive with raw_source_ids "
                        "and raw_transcript_ids. Remove those fields or set "
                        "auto_resolve=false to provide IDs manually."
                    ),
                },
            )
    else:
        # ── MANUAL MODE: required ID lists must be provided ─────────────────
        if requires_sources and not body.raw_source_ids:
            raise HTTPException(
                status_code=400,
                detail={
                    "error":   "INVALID_REQUEST",
                    "message": (
                        "Manual mode requires 'raw_source_ids' for this endpoint. "
                        "Provide at least one raw_scraped_data.raw_id."
                    ),
                },
            )
        if requires_transcripts and not body.raw_transcript_ids:
            raise HTTPException(
                status_code=400,
                detail={
                    "error":   "INVALID_REQUEST",
                    "message": (
                        "Manual mode requires 'raw_transcript_ids' for /run-b. "
                        "Provide at least one youtube_raw_transcript_data.raw_transcript_id."
                    ),
                },
            )


# ---------------------------------------------------------------------------
# FIX-12 — POST /extraction/run-a  (unified manual + auto)
# ---------------------------------------------------------------------------

@router.post("/run-a", response_model=RunAResponse)
@limiter.limit("10/minute")
async def trigger_run_a(request: Request, body: ExtractionRequest):
    """
    POST /extraction/run-a — Spec extraction (Run A) for one phone.

    MODES:
      Auto   (auto_resolve=true):  Requires brand + model_name only.
                                   Backend resolves ALL source IDs from DB.
                                   Uses top3_transcript_ids (top 3 by channel_reliability_score DESC, file_size_bytes DESC).
      Manual (auto_resolve=false): Requires brand + model_name + raw_source_ids.
                                   Optional: raw_transcript_ids (first is used for Run A).
                                   Backend validates all IDs belong to this phone.

    Both modes:
      - Resolve canonical_url_id from brand + model_name (GSMArena anchor).
      - Run pre-extraction validation automatically (fresh check, writes new record).
      - Enforce the Phase 1 validation gate (can_proceed=true).

    Raises:
        HTTP 400: Mode validation failure or cross-phone IDs detected.
        HTTP 404: Phone not found in DB.
        HTTP 422: Pre-extraction validation failed (blocking reasons returned).
        HTTP 429: Rate limit exceeded (10/minute).
        HTTP 500: Extraction or storage error.
    """
    from app.repositories.extraction_repository import (
        fetch_canonical_id_by_brand_model,
        fetch_sources_for_run_phone,
        validate_source_ids_belong_to_phone,
    )

    _validate_extraction_request(body, requires_sources=True, requires_transcripts=False)

    brand, model_name = _normalize_brand_model(body.brand, body.model_name)

    # ── Step 1: Resolve canonical_url_id (both modes) ───────────────────────
    try:
        canonical_url_id = await asyncio.to_thread(
            fetch_canonical_id_by_brand_model, brand, model_name
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": "PHONE_NOT_FOUND", "message": str(exc)},
        )

    # ── Step 2: Resolve / validate source IDs (mode-specific) ───────────────
    if body.auto_resolve:
        try:
            resolved = await asyncio.to_thread(fetch_sources_for_run_phone, canonical_url_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=404,
                detail={"error": "PHONE_NOT_FOUND", "message": str(exc)},
            )

        raw_source_ids      = resolved["raw_source_ids"]
        top3_transcript_ids = resolved["top3_transcript_ids"]


        logger.info(
            "trigger_run_a [auto]: url_id=%d brand=%r model=%r sources=%s top3_transcripts=%s",
            canonical_url_id, brand, model_name, raw_source_ids, top3_transcript_ids,
        )
    else:
        # Manual mode: validate that all provided IDs belong to this phone
        raw_source_ids      = body.raw_source_ids
        top3_transcript_ids = body.raw_transcript_ids

        ownership = await asyncio.to_thread(
            validate_source_ids_belong_to_phone,
            canonical_url_id,
            raw_source_ids,
            body.raw_transcript_ids,
        )
        if not ownership["valid"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "error":   "FOREIGN_IDS_DETECTED",
                    "message": (
                        "One or more provided IDs do not belong to the resolved phone. "
                        "This request has been blocked to prevent data contamination."
                    ),
                    "foreign_source_ids":     ownership["foreign_source_ids"],
                    "foreign_transcript_ids": ownership["foreign_transcript_ids"],
                },
            )

        logger.info(
            "trigger_run_a [manual]: url_id=%d brand=%r model=%r sources=%s transcripts=%s",
            canonical_url_id, brand, model_name, raw_source_ids, top3_transcript_ids,
        )

    # ── Step 3: Fresh pre-extraction validation (auto-runs before every trigger) ──
    validation_record = await run_pre_extraction_validation(
        canonical_url_id=canonical_url_id,
        validated_by="auto_pre_run",
    )
    if not validation_record.get("can_proceed"):
        raise HTTPException(
            status_code=422,
            detail={
                "error":            "PRE_VALIDATION_FAILED",
                "message":          "Pre-extraction validation did not pass. See blocking_reasons.",
                "blocking_reasons": validation_record.get("blocking_reasons", []),
                "warnings":         validation_record.get("warnings", []),
            },
        )

    # ── Step 4: Run extraction ────────────────────────────────────────────
    # Create the pipeline_runs row BEFORE calling the service so we have the UUID.
    # If create_pipeline_run fails, pipeline_run_id stays None and _track() is a no-op.
    pipeline_run_id: str | None = None
    try:
        pipeline_run_id = await asyncio.to_thread(
            create_pipeline_run, canonical_url_id, "run_a"
        )
    except Exception as _pr_exc:
        logger.warning("trigger_run_a: could not create pipeline_runs row: %s", _pr_exc)

    try:
        result = await run_spec_extraction(
            url_registry_id=canonical_url_id,
            raw_source_ids=raw_source_ids,
            raw_transcript_ids=top3_transcript_ids,
            brand=brand,
            model_name=model_name,
            schema_version=body.schema_version,
            pipeline_run_id=pipeline_run_id,
        )
        return RunAResponse(**result, pipeline_run_id=pipeline_run_id)
    except ValueError as exc:
        if str(exc).startswith(_GATE_ERROR_PREFIX):
            raise HTTPException(
                status_code=422,
                detail={"error": "EXTRACTION_GATE_FAILED", "message": str(exc)},
            )
        raise HTTPException(
            status_code=404,
            detail={"error": "NOT_FOUND", "message": str(exc)},
        )
    except Exception as exc:
        logger.exception(
            "trigger_run_a: unexpected error for canonical_url_id=%s", canonical_url_id
        )
        raise HTTPException(
            status_code=500,
            detail={"error": "INTERNAL_ERROR", "message": str(exc)},
        )


# ---------------------------------------------------------------------------
# FIX-13 — POST /extraction/run-b  (NEW endpoint, manual + auto)
# ---------------------------------------------------------------------------

@router.post("/run-b", response_model=RunBResult)
@limiter.limit("10/minute")
async def trigger_run_b(request: Request, body: ExtractionRequest):
    """
    POST /extraction/run-b — Experience extraction (Run B) for one phone.

    Fires one Gemini call per transcript, parallel, semaphore-capped at 5.

    MODES:
      Auto   (auto_resolve=true):  Requires brand + model_name. Resolves ALL available
                                   transcripts for this phone (status='fetched_raw').
      Manual (auto_resolve=false): Requires brand + model_name + raw_transcript_ids.
                                   Backend validates all transcript IDs belong to this phone.

    Both modes:
      - Resolve canonical_url_id from brand + model_name (GSMArena anchor).
      - Run pre-extraction validation automatically (fresh check, writes new record).
      - Enforce the Phase 1 validation gate.

    Experience extraction supersedes all existing active phone_experiences rows
    (is_superseded=FALSE → TRUE) before inserting new ones. Gate check must pass first.

    Raises:
        HTTP 400: Mode validation failure, cross-phone IDs, or no transcripts available.
        HTTP 404: Phone not found in DB.
        HTTP 422: Pre-extraction validation failed (blocking reasons returned).
        HTTP 429: Rate limit exceeded (10/minute).
        HTTP 500: Extraction error.
    """
    from app.repositories.extraction_repository import (
        fetch_canonical_id_by_brand_model,
        fetch_sources_for_run_phone,
        validate_source_ids_belong_to_phone,
    )
    from app.repositories.pipeline_run_repository import create_pipeline_run

    _validate_extraction_request(body, requires_sources=False, requires_transcripts=True)

    brand, model_name = _normalize_brand_model(body.brand, body.model_name)

    # ── Step 1: Resolve canonical_url_id (both modes) ───────────────────────
    try:
        canonical_url_id = await asyncio.to_thread(
            fetch_canonical_id_by_brand_model, brand, model_name
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": "PHONE_NOT_FOUND", "message": str(exc)},
        )

    # ── Step 2: Resolve / validate transcript IDs (mode-specific) ─────────────
    if body.auto_resolve:
        try:
            resolved = await asyncio.to_thread(fetch_sources_for_run_phone, canonical_url_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=404,
                detail={"error": "PHONE_NOT_FOUND", "message": str(exc)},
            )

        raw_transcript_ids = resolved["raw_transcript_ids"]

        if not raw_transcript_ids:
            raise HTTPException(
                status_code=400,
                detail={
                    "error":   "NO_TRANSCRIPTS_AVAILABLE",
                    "message": (
                        f"No transcripts found for {brand!r} {model_name!r} "
                        "(status='fetched_raw'). Fetch transcripts before running Run B."
                    ),
                },
            )

        logger.info(
            "trigger_run_b [auto]: url_id=%d brand=%r model=%r transcripts=%s",
            canonical_url_id, brand, model_name, raw_transcript_ids,
        )
    else:
        # Manual mode: validate that all provided transcript IDs belong to this phone
        raw_transcript_ids = body.raw_transcript_ids

        ownership = await asyncio.to_thread(
            validate_source_ids_belong_to_phone,
            canonical_url_id,
            [],  # no source IDs for run-b
            raw_transcript_ids,
        )
        if not ownership["valid"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "error":   "FOREIGN_IDS_DETECTED",
                    "message": (
                        "One or more transcript IDs do not belong to the resolved phone. "
                        "This request has been blocked to prevent data contamination."
                    ),
                    "foreign_source_ids":     ownership["foreign_source_ids"],
                    "foreign_transcript_ids": ownership["foreign_transcript_ids"],
                },
            )

        logger.info(
            "trigger_run_b [manual]: url_id=%d brand=%r model=%r transcripts=%s",
            canonical_url_id, brand, model_name, raw_transcript_ids,
        )

    # ── Step 3: Fresh pre-extraction validation (auto-runs before every trigger) ──
    validation_record = await run_pre_extraction_validation(
        canonical_url_id=canonical_url_id,
        validated_by="auto_pre_run",
    )
    if not validation_record.get("can_proceed"):
        raise HTTPException(
            status_code=422,
            detail={
                "error":            "PRE_VALIDATION_FAILED",
                "message":          "Pre-extraction validation did not pass. See blocking_reasons.",
                "blocking_reasons": validation_record.get("blocking_reasons", []),
                "warnings":         validation_record.get("warnings", []),
            },
        )

    # ── Step 4: Run extraction ────────────────────────────────────────────
    pipeline_run_id: str | None = None
    try:
        pipeline_run_id = await asyncio.to_thread(
            create_pipeline_run, canonical_url_id, "run_b",
            len(raw_transcript_ids),
        )
    except Exception as _pr_exc:
        logger.warning("trigger_run_b: could not create pipeline_runs row: %s", _pr_exc)

    try:
        run_b_raw = await run_experience_extraction_batch(
            url_registry_id=canonical_url_id,
            raw_transcript_ids=raw_transcript_ids,
            brand=brand,
            model_name=model_name,
            schema_version=body.schema_version,
            pipeline_run_id=pipeline_run_id,
        )
    except ValueError as exc:
        if str(exc).startswith(_GATE_ERROR_PREFIX):
            raise HTTPException(
                status_code=422,
                detail={"error": "EXTRACTION_GATE_FAILED", "message": str(exc)},
            )
        raise HTTPException(
            status_code=404,
            detail={"error": "NOT_FOUND", "message": str(exc)},
        )
    except Exception as exc:
        logger.exception(
            "trigger_run_b: unexpected error for canonical_url_id=%s", canonical_url_id
        )
        raise HTTPException(
            status_code=500,
            detail={"error": "INTERNAL_ERROR", "message": str(exc)},
        )

    # run_b_raw is now a dict (two-stage system)
    if isinstance(run_b_raw, Exception):
        return RunBResult(
            success=False,
            pipeline_run_id=pipeline_run_id,
            message=f"Run B failed: {run_b_raw}",
            error=str(run_b_raw),
        )

    new_t    = run_b_raw.get("new_transcripts", 0)
    reused_t = run_b_raw.get("reused_transcripts", 0)
    return RunBResult(
        success=run_b_raw.get("success", False),
        aggregation_run_id=run_b_raw.get("aggregation_run_id"),
        pipeline_run_id=pipeline_run_id,
        experiences_extracted=run_b_raw.get("experiences_output", 0),
        new_transcripts=new_t,
        reused_transcripts=reused_t,
        transcripts_processed=new_t + reused_t,
        transcripts_failed=len(raw_transcript_ids) - (new_t + reused_t),
        message=run_b_raw.get("message", ""),
        error=run_b_raw.get("error"),
    )


# ---------------------------------------------------------------------------
# FIX-14 — POST /extraction/run-all  (unified manual + auto, single gate check)
# ---------------------------------------------------------------------------

@router.post("/run-all", response_model=RunAllResponse)
@limiter.limit("5/minute")
async def trigger_run_all(request: Request, body: ExtractionRequest):
    """
    POST /extraction/run-all — Run A + Run B in parallel for one phone.

    MODES:
      Auto   (auto_resolve=true):  Requires brand + model_name only. Resolves all IDs.
                                   Run A uses top3_transcript_ids. Run B uses all transcripts.
      Manual (auto_resolve=false): Requires brand + model_name + raw_source_ids.
                                   raw_transcript_ids is optional (Run B skipped if empty).
                                   Backend validates all IDs belong to this phone.

    Both modes:
      - Resolve canonical_url_id from brand + model_name (GSMArena anchor).
      - Run pre-extraction validation ONCE before launching both runs.
      - If the gate fails, a single HTTP 422 is returned before any run records are created.

    Run A and Run B are launched in parallel (asyncio.gather). Failure of one does NOT
    block the other. Both results are returned independently in RunAllResponse.

    Raises:
        HTTP 400: Mode validation failure or cross-phone IDs detected.
        HTTP 404: Phone not found in DB.
        HTTP 422: Pre-extraction validation failed (checked once, before any launch).
        HTTP 429: Rate limit exceeded (5/minute).
        HTTP 500: Unexpected orchestration error.
    """
    from app.repositories.extraction_repository import (
        fetch_canonical_id_by_brand_model,
        fetch_sources_for_run_phone,
        validate_source_ids_belong_to_phone,
    )
    from app.repositories.pipeline_run_repository import create_pipeline_run

    _validate_extraction_request(body, requires_sources=True, requires_transcripts=False)

    brand, model_name = _normalize_brand_model(body.brand, body.model_name)

    # ── Step 1: Resolve canonical_url_id (both modes) ───────────────────────
    try:
        canonical_url_id = await asyncio.to_thread(
            fetch_canonical_id_by_brand_model, brand, model_name
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": "PHONE_NOT_FOUND", "message": str(exc)},
        )

    # ── Step 2: Resolve / validate IDs (mode-specific) ───────────────────────
    if body.auto_resolve:
        try:
            resolved = await asyncio.to_thread(fetch_sources_for_run_phone, canonical_url_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=404,
                detail={"error": "PHONE_NOT_FOUND", "message": str(exc)},
            )

        raw_source_ids      = resolved["raw_source_ids"]
        raw_transcript_ids  = resolved["raw_transcript_ids"]
        top3_transcript_ids = resolved["top3_transcript_ids"]


        logger.info(
            "trigger_run_all [auto]: url_id=%d brand=%r model=%r "
            "sources=%s transcripts=%s top3_transcripts=%s",
            canonical_url_id, brand, model_name,
            raw_source_ids, raw_transcript_ids, top3_transcript_ids,
        )
    else:
        # Manual mode: validate all provided IDs belong to this phone
        raw_source_ids     = body.raw_source_ids
        raw_transcript_ids = body.raw_transcript_ids

        ownership = await asyncio.to_thread(
            validate_source_ids_belong_to_phone,
            canonical_url_id,
            raw_source_ids,
            raw_transcript_ids,
        )
        if not ownership["valid"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "error":   "FOREIGN_IDS_DETECTED",
                    "message": (
                        "One or more provided IDs do not belong to the resolved phone. "
                        "This request has been blocked to prevent data contamination."
                    ),
                    "foreign_source_ids":     ownership["foreign_source_ids"],
                    "foreign_transcript_ids": ownership["foreign_transcript_ids"],
                },
            )

        top3_transcript_ids = raw_transcript_ids[:3]

        if len(raw_transcript_ids) > 3:
            logger.warning(
                "trigger_run_all [manual]: %d transcript IDs provided — "
                "Run A uses top 3 only (IDs: %s). Run B uses all %d.",
                len(raw_transcript_ids),
                raw_transcript_ids[:3],
                len(raw_transcript_ids),
            )

        logger.info(
            "trigger_run_all [manual]: url_id=%d brand=%r model=%r "
            "sources=%s transcripts=%s top3_transcripts=%s",
            canonical_url_id, brand, model_name,
            raw_source_ids, raw_transcript_ids, top3_transcript_ids,
        )

    # ── Step 3: Single fresh pre-extraction validation BEFORE launching both runs ──
    validation_record = await run_pre_extraction_validation(
        canonical_url_id=canonical_url_id,
        validated_by="auto_pre_run",
    )
    if not validation_record.get("can_proceed"):
        raise HTTPException(
            status_code=422,
            detail={
                "error":            "PRE_VALIDATION_FAILED",
                "message":          "Pre-extraction validation did not pass. See blocking_reasons.",
                "blocking_reasons": validation_record.get("blocking_reasons", []),
                "warnings":         validation_record.get("warnings", []),
            },
        )

    # ── Step 4: Launch Run A + Run B in parallel ────────────────────────────────
    # Create one pipeline_runs row per run (run_a and run_b are independent).
    pipeline_run_id_a: str | None = None
    pipeline_run_id_b: str | None = None
    try:
        pipeline_run_id_a = await asyncio.to_thread(
            create_pipeline_run, canonical_url_id, "run_a"
        )
    except Exception as _pr_exc:
        logger.warning("trigger_run_all: could not create run_a pipeline_runs row: %s", _pr_exc)
    if raw_transcript_ids:
        try:
            pipeline_run_id_b = await asyncio.to_thread(
                create_pipeline_run, canonical_url_id, "run_b",
                len(raw_transcript_ids),
            )
        except Exception as _pr_exc:
            logger.warning("trigger_run_all: could not create run_b pipeline_runs row: %s", _pr_exc)

    run_a_coro = run_spec_extraction(
        url_registry_id=canonical_url_id,
        raw_source_ids=raw_source_ids,
        raw_transcript_ids=top3_transcript_ids,
        brand=brand,
        model_name=model_name,
        schema_version=body.schema_version,
        pipeline_run_id=pipeline_run_id_a,
    )

    if raw_transcript_ids:
        run_b_coro = run_experience_extraction_batch(
            url_registry_id=canonical_url_id,
            raw_transcript_ids=raw_transcript_ids,
            brand=brand,
            model_name=model_name,
            schema_version=body.schema_version,
            pipeline_run_id=pipeline_run_id_b,
        )
    else:
        async def _no_transcripts():
            return []
        run_b_coro = _no_transcripts()

    results = await asyncio.gather(run_a_coro, run_b_coro, return_exceptions=True)
    run_a_result, run_b_raw = results

    # Process Run A
    if isinstance(run_a_result, Exception):
        exc = run_a_result
        logger.exception(
            "trigger_run_all: Run A failed for url_id=%d: %s", canonical_url_id, exc
        )
        run_a_response = RunAResponse(
            success=False, output_id=None, run_id=None,
            message=f"Run A failed: {exc}", failed_source_ids=[],
        )
    else:
        run_a_response = RunAResponse(**run_a_result)

    # Process Run B — run_b_raw is now a dict (two-stage system)
    if isinstance(run_b_raw, Exception):
        run_b_response = RunBResult(
            success=False,
            message=f"Run B batch failed: {run_b_raw}",
            error=str(run_b_raw),
        )
    elif not run_b_raw:
        run_b_response = RunBResult(
            success=True,
            message="No transcripts provided — Run B skipped.",
        )
    else:
        new_t    = run_b_raw.get("new_transcripts", 0)
        reused_t = run_b_raw.get("reused_transcripts", 0)
        run_b_response = RunBResult(
            success=run_b_raw.get("success", False),
            aggregation_run_id=run_b_raw.get("aggregation_run_id"),
            experiences_extracted=run_b_raw.get("experiences_output", 0),
            new_transcripts=new_t,
            reused_transcripts=reused_t,
            transcripts_processed=new_t + reused_t,
            transcripts_failed=len(raw_transcript_ids) - (new_t + reused_t),
            message=run_b_raw.get("message", ""),
            error=run_b_raw.get("error"),
        )

    return RunAllResponse(
        run_a=run_a_response,
        run_b=run_b_response,
        both_succeeded=run_a_response.success and run_b_response.success,
    )


# ---------------------------------------------------------------------------
# GET /extraction/run-status/{run_id}  — Poll a pipeline run's live status
# ---------------------------------------------------------------------------

@router.get("/run-status/{run_id}")
async def get_run_status(run_id: str):
    """
    Returns the current status of a pipeline run by its UUID.

    Designed for Admin UI polling (1.5s interval while status='running').
    The UI should stop polling when status is 'completed', 'partial', or 'failed'.

    Computed fields:
        elapsed_seconds  — integer seconds since started_at (computed server-side)
        processed        — alias for processed_items
        total            — alias for total_items
        failed           — alias for failed_items

    Raises:
        HTTP 404: If run_id is not found in pipeline.pipeline_runs.
        HTTP 500: On unexpected DB error.
    """
    try:
        row = await asyncio.to_thread(fetch_pipeline_run, run_id)
    except Exception as exc:
        logger.exception("get_run_status: DB error for run_id=%s", run_id)
        raise HTTPException(status_code=500, detail=str(exc))

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"run_id={run_id!r} not found in pipeline.pipeline_runs.",
        )

    started_at = row.get("started_at")
    elapsed = 0
    if started_at:
        start = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
        elapsed = int((datetime.now(timezone.utc) - start).total_seconds())

    return {
        "run_id":          row["run_id"],
        "run_type":        row["run_type"],
        "status":          row["status"],
        "current_stage":   row.get("current_stage"),
        "current_step":    row.get("current_step"),
        "processed":       row.get("processed_items", 0),
        "total":           row.get("total_items"),
        "failed":          row.get("failed_items", 0),
        "elapsed_seconds": elapsed,
        "started_at":      row.get("started_at"),
        "completed_at":    row.get("completed_at"),
        "updated_at":      row.get("updated_at"),
        "error_summary":   row.get("error_summary") or [],
    }


# ---------------------------------------------------------------------------
# GET /extraction/recent-runs  — Run history for a phone
# ---------------------------------------------------------------------------

@router.get("/recent-runs")
async def get_recent_runs(
    url_registry_id: int = Query(..., description="pipeline.url_registry.url_id"),
    limit: int = Query(default=10, ge=1, le=50, description="Max rows to return"),
):
    """
    Returns the N most recent pipeline runs for the given phone, ordered
    by started_at DESC.

    Used by the Admin UI to show run history and let the admin re-attach
    to an in-progress run after a page refresh.

    Raises:
        HTTP 500: On unexpected DB error.
    """
    try:
        rows = await asyncio.to_thread(
            fetch_recent_pipeline_runs, url_registry_id, limit
        )
    except Exception as exc:
        logger.exception(
            "get_recent_runs: DB error for url_registry_id=%d", url_registry_id
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return {"url_registry_id": url_registry_id, "runs": rows}


# ---------------------------------------------------------------------------
# Phase 4 - POST /extraction/normalise
# ---------------------------------------------------------------------------

class NormaliseRequest(BaseModel):
    output_id: int


class NormaliseResponse(BaseModel):
    success: bool
    normalized_id: int | None = None
    issue_count: int = 0
    staging_count: int = 0
    remaining_nulls: int = 0
    ready_for_enrichment: bool = False
    message: str


@router.post("/normalise", response_model=NormaliseResponse)
@limiter.limit("20/minute")
async def trigger_normalise(request: Request, body: NormaliseRequest):
    """
    Triggers Phase 4 normalisation for a completed Run A output.

    Normalisation flow:
      1. Fetches partial_json from spec_extraction_output
      2. Resolves FK strings to lookup PKs via in-memory LOOKUP_CACHE
         (built at startup - zero DB queries per field during resolution)
      3. Strips unit suffixes: "5000mAh"->5000, "8GB"->8, "f/1.8"->1.8
      4. Coerces booleans: "true"/"yes"->True, "false"/"no"->False
      5. Validates numeric ranges - out-of-range values nulled + logged
      6. Discards RUN_C_CALCULATED_FIELDS if LLM included them
      7. Sends unresolvable values to lookup_value_staging for human review
      8. Writes normalized_spec_json to pipeline.normalized_spec

    Raises:
        HTTP 404: If output_id not found in spec_extraction_output.
        HTTP 500: On normalisation error.
    """
    try:
        result = await run_normalisation(output_id=body.output_id)
        return NormaliseResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception(
            "trigger_normalise: unexpected error for output_id=%d", body.output_id
        )
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/cache/refresh", status_code=200)
async def refresh_lookup_cache():
    """
    Forces a full rebuild of the in-memory LOOKUP_CACHE from all lookup tables.
    Use after seeding new lookup values so they are immediately available
    to the normaliser without a server restart.

    Returns:
        {"message": "LOOKUP_CACHE refreshed. N tables loaded."}
    """
    from app.services.normalizer import build_lookup_cache, LOOKUP_CACHE
    await build_lookup_cache()
    return {"message": f"LOOKUP_CACHE refreshed. {len(LOOKUP_CACHE)} tables loaded."}

# ---------------------------------------------------------------------------
# Phase 5 — POST /extraction/gaps
# ---------------------------------------------------------------------------

class GapAnalysisRequest(BaseModel):
    normalized_id: int


class GapAnalysisResponse(BaseModel):
    success: bool
    normalized_id: int
    gaps_found: int
    missing_field_ids: list[int]
    message: str


@router.post("/gaps", response_model=GapAnalysisResponse)
@limiter.limit("20/minute")
async def trigger_gap_analysis(request: Request, body: GapAnalysisRequest):
    """
    Triggers Phase 5 gap analysis on a completed normalisation output.

    Gap analysis flow:
      1. Fetches normalized_json from normalized_spec_json
      2. Walks the JSON for every null scalar or empty array
      3. For each gap, creates a pipeline.missing_fields_log row with:
           - field_type:  type_a (scalar) | type_b (junction table array)
           - priority:    high | medium (from FIELD_PRIORITY_MAP)
           - site_hint:   domain to target for enrichment search
      4. For type_b gaps on phones already committed to mainDB:
           Creates type_b_gap_candidates rows (what already exists vs what is missing)
      5. Returns the list of missing_field_id values created

    Prerequisites:
      - Phase 4 normalisation must have completed (normalized_id must exist)
      - url_registry row must exist for the phone

    Args:
        normalized_id: pipeline.normalized_spec_json.normalized_id to analyse.

    Returns:
        gaps_found (count), missing_field_ids (list of created log row IDs).

    Raises:
        HTTP 404: If normalized_id not found.
        HTTP 500: On analysis error.
    """
    try:
        missing_field_ids = await detect_missing_fields(
            normalized_id=body.normalized_id
        )
        return GapAnalysisResponse(
            success=True,
            normalized_id=body.normalized_id,
            gaps_found=len(missing_field_ids),
            missing_field_ids=missing_field_ids,
            message=(
                f"Gap analysis complete. {len(missing_field_ids)} missing fields logged."
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception(
            "trigger_gap_analysis: unexpected error for normalized_id=%d",
            body.normalized_id,
        )
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Phase 6 — Enrichment
# ---------------------------------------------------------------------------

class ResolveConflictsResponse(BaseModel):
    success:             bool
    final_id:            int
    flagged_count:       int
    auto_resolved_count: int
    fill_count:          int
    concordant_count:    int
    message:             str


class PromoteNormalizedRequest(BaseModel):
    normalized_id: int


class EnrichRequest(BaseModel):
    normalized_id: int
    brand: str
    model: str


class EnrichResponse(BaseModel):
    success: bool
    enrichment_run_id: int
    fields_targeted: int = 0       # E10 fix
    fields_resolved: int
    total_api_cost_inr: float = 0.0  # E10 fix
    message: str


@router.post("/enrich", response_model=EnrichResponse)
@limiter.limit("20/minute")
async def trigger_enrichment(request: Request, body: EnrichRequest) -> EnrichResponse:
    """
    Phase 6 — Enrichment.

    Fires one Gemini grounded search query per pending missing field for the
    given phone. Stores candidates in enrichment_field_candidates with both
    raw and adjusted confidence, and auto-selects where rules allow.

    Body:
        normalized_id  int  — pipeline.normalized_spec_json.normalized_id
        brand          str  — phone brand, e.g. "Samsung"
        model          str  — phone model, e.g. "Galaxy S25 Ultra"

    Returns:
        success (bool), enrichment_run_id, fields_resolved.

    Raises:
        HTTP 404: If normalized_id not found.
        HTTP 500: On unexpected error.
    """
    try:
        result = await run_enrichment(
            normalized_id=body.normalized_id,
            brand=body.brand,
            model=body.model,
        )
        return EnrichResponse(
            success=result["success"],
            enrichment_run_id=result["enrichment_run_id"],
            fields_targeted=result.get("fields_targeted", 0),     # E10 fix
            fields_resolved=result["fields_resolved"],
            total_api_cost_inr=result.get("total_api_cost_inr", 0.0),  # E10 fix
            message=(
                f"Enrichment complete. "
                f"{result['fields_resolved']}/{result.get('fields_targeted', '?')} "
                f"field(s) resolved — ₹{result.get('total_api_cost_inr', 0.0):.2f} "
                f"(run_id={result['enrichment_run_id']})."
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception(
            "trigger_enrichment: unexpected error for normalized_id=%d",
            body.normalized_id,
        )
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/promote-normalized", response_model=ResolveConflictsResponse)
@limiter.limit("20/minute")
async def trigger_promote_normalized(request: Request, body: PromoteNormalizedRequest):
    """
    Skip-enrichment fast-path.

    Promotes normalized_spec_json directly to final_merged_json without
    running gap analysis, enrichment, or conflict resolution.

    Use when you want to review and fill null fields manually in the Admin UI
    rather than running the enrichment pipeline.

    Body:
        normalized_id: pipeline.normalized_spec_json.normalized_id

    Returns:
        final_id, all conflict counts as zero.

    Raises:
        HTTP 404: If normalized_id not found.
        HTTP 500: On unexpected error.
    """
    logger.info(
        "trigger_promote_normalized: normalized_id=%d",
        body.normalized_id,
    )
    try:
        result = await promote_normalized_to_final(body.normalized_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception(
            "trigger_promote_normalized: unexpected error normalized_id=%d",
            body.normalized_id,
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return ResolveConflictsResponse(
        success=True,
        final_id=result["final_id"],
        flagged_count=0,
        auto_resolved_count=0,
        fill_count=0,
        concordant_count=0,
        message=(
            f"Normalized spec promoted directly to final_merged_json "
            f"(final_id={result['final_id']}). No enrichment was run."
        ),
    )


# ===========================================================================
# Phase 7 — Conflict Resolution
# ===========================================================================

class ResolveConflictsRequest(BaseModel):
    normalized_id:     int
    enrichment_run_id: int


@router.post("/resolve-conflicts", response_model=ResolveConflictsResponse)
@limiter.limit("20/minute")
async def trigger_conflict_resolution(request: Request, body: ResolveConflictsRequest):
    """
    Phase 7 — Conflict Resolution.

    1. Detects genuine conflicts between Run A normalized values and selected
       enrichment candidates.
    2. Auto-resolves conflicts where source tier or confidence is decisive.
    3. Flags ambiguous conflicts for admin review.
    4. Builds (or updates) the final_merged_json artifact.

    Body:
        normalized_id:     Normalized spec JSON row to process.
        enrichment_run_id: The enrichment run to compare against.

    Returns:
        final_id, flagged_count, auto_resolved_count, fill_count, concordant_count
    """
    logger.info(
        "trigger_conflict_resolution: normalized_id=%d enrichment_run_id=%d",
        body.normalized_id, body.enrichment_run_id,
    )
    try:
        conflict_result = await detect_and_resolve_conflicts(body.normalized_id,  body.enrichment_run_id)
    except Exception as exc:
        logger.exception(
            "trigger_conflict_resolution: conflict detection failed normalized_id=%d",
            body.normalized_id,
        )
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        final_id = await build_final_merged_json(body.normalized_id, body.enrichment_run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception(
            "trigger_conflict_resolution: build_final_merged_json failed after "
            "successful conflict detection. normalized_id=%d",
            body.normalized_id,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "PARTIAL_FAILURE",
                "message": (
                    "Conflict detection succeeded but final merge failed. "
                    "Re-run /resolve-conflicts — detection is idempotent. "
                    f"Error: {exc}"
                ),
            },
        )

    return ResolveConflictsResponse(
        success=True,
        final_id=final_id,
        flagged_count=conflict_result["flagged_count"],
        auto_resolved_count=conflict_result["auto_resolved_count"],
        fill_count=conflict_result["fill_count"],
        concordant_count=conflict_result["concordant_count"],
        message=(
            f"Conflict resolution complete. "
            f"auto_resolved={conflict_result['auto_resolved_count']} "
            f"fills={conflict_result['fill_count']} "
            f"flagged={conflict_result['flagged_count']} "
            f"concordant={conflict_result['concordant_count']} "
            f"(final_id={final_id})."
        ),
    )


# ===========================================================================
# Phase 7.5 — Pre-UI System Validation
# ===========================================================================

class PreUIValidationRequest(BaseModel):
    final_id:        int
    url_registry_id: int
    normalized_id:   int


class PreUIValidationResponse(BaseModel):
    passed:         bool
    pre_ui_val_id:  int
    error_count:    int
    warning_count:  int
    errors:         list[dict]
    warnings:       list[dict]
    message:        str


@router.post("/validate-pre-ui", response_model=PreUIValidationResponse)
async def trigger_pre_ui_validation(body: PreUIValidationRequest):
    """
    Phase 7.5 — Layer 1 Pre-UI System Validation.

    Runs structural and integrity checks on the final_merged_json before
    surfacing it to the admin gate. Uses Pydantic/Python only — zero API cost.

    Hard blocks (errors): schema violations, missing required fields, type errors.
    Non-blocking (warnings): missing evidence, low confidence, no Run B entries.

    Body:
        final_id:        The final_merged_json row to validate.
        url_registry_id: For evidence and experience checks.
        normalized_id:   For persisting pre_ui_validation_runs.
    """
    logger.info(
        "trigger_pre_ui_validation: final_id=%d normalized_id=%d",
        body.final_id, body.normalized_id,
    )
    try:
        result = await run_pre_ui_validation(
            final_id=body.final_id,
            url_registry_id=body.url_registry_id,
            normalized_id=body.normalized_id,
        )
        return PreUIValidationResponse(
            passed=result["passed"],
            pre_ui_val_id=result["pre_ui_val_id"],
            error_count=len(result["errors"]),
            warning_count=len(result["warnings"]),
            errors=result["errors"],
            warnings=result["warnings"],
            message=(
                "Validation passed — admin gate may open."
                if result["passed"]
                else (
                    f"Validation FAILED — {len(result['errors'])} hard error(s). "
                    f"Admin gate is blocked."
                )
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception(
            "trigger_pre_ui_validation: unexpected error final_id=%d",
            body.final_id,
        )
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/validation-status-pre-ui/{final_id}",
    response_model=dict,
)
async def get_pre_ui_validation_status_endpoint(final_id: int):
    """
    Phase 7.5 — Returns the latest pre_ui_validation_runs result for a final_id.

    Looks up the normalized_id via final_merged_json, then queries
    pre_ui_validation_runs for the most recent result.

    Returns 404 if the final_id does not exist.
    Returns {"status": "not_run"} if validation has not been run yet.
    """
    from app.repositories.extraction_repository import fetch_final_merged_json
    try:
        final_row = await asyncio.to_thread(fetch_final_merged_json, final_id)
        if final_row is None:
            raise HTTPException(
                status_code=404,
                detail=f"final_id={final_id} not found in final_merged_json.",
            )
        normalized_id = final_row["normalized_id"]
        result = await get_pre_ui_validation_status(normalized_id)
        if result is None:
            return {"status": "not_run", "final_id": final_id, "normalized_id": normalized_id}
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "get_pre_ui_validation_status_endpoint: error for final_id=%d", final_id
        )
        raise HTTPException(status_code=500, detail=str(exc))
