"""
Phase 1 — Pre-Extraction Validation Service (Task 1.1)

Checks that all registered URLs have been attempted, at least one succeeded,
and that a YouTube search has been run before allowing Run A to proceed.

Why this gate exists:
    Without it, Run A can fire on phones with no OEM data and no transcripts,
    producing low-quality partial JSON that wastes LLM cost and requires a re-run.

Blockers (any one prevents can_proceed):
    1. urls_not_attempted > 0         — unattempted URLs must be scraped first
    2. urls_scraped_success == 0      — at least one source file required
    3. youtube_search_done == False   — YouTube search is mandatory (even zero results)

Warnings (surfaced to admin but not blocking):
    - official_url_scraped == False   — OEM URL registered but scrape failed
    - transcript_available == False   — search ran but no transcript fetched
"""

import asyncio
import logging
from datetime import datetime, timezone

from app.repositories.extraction_repository import (
    fetch_url_registry_for_phone,
    fetch_youtube_search_log,
    fetch_transcript_availability,
    insert_pre_extraction_validation,
    fetch_latest_validation,
)

logger = logging.getLogger(__name__)

# Status values considered "successfully scraped"
_SUCCESS_STATUSES = {"scraped_raw", "stored_mainDB"}

# Status values considered "not yet attempted"
_NOT_ATTEMPTED_STATUSES = {"not_scraped"}

# YouTube search statuses that count as "done" (even zero results is done)
_YOUTUBE_DONE_STATUSES = {"success_zero", "success_with_results"}


async def run_pre_extraction_validation(
    canonical_url_id: int,
    validated_by: str = "admin_manual",
) -> dict:
    """
    Runs pre-extraction validation for the phone identified by canonical_url_id.

    CHECKS:
    1. Fetch ALL url_registry rows for this (brand, model_name).
       Count total_urls_registered, urls_not_attempted, urls_scraped_success.
    2. Determine official_url_scraped (NULL if no OEM URL in registry).
    3. Determine gsmarena_scraped (NULL if no GSMArena URL in registry).
    4. youtube_search_done: check youtube_search_log for canonical_url_id
       WHERE search_status IN ('success_zero', 'success_with_results').
    5. transcript_available: check youtube_video_url_registry for canonical_url_id
       WHERE status = 'fetched_raw'.

    BLOCKERS (only these three block can_proceed):
      - urls_not_attempted > 0
      - urls_scraped_success == 0
      - youtube_search_done == False

    official_url_scraped == False is a WARNING, not a blocker.

    Writes result to pipeline.pre_extraction_validation.
    Returns the inserted validation record dict.

    Raises:
        ValueError: If canonical_url_id does not exist in url_registry.
        RuntimeError: If DB write fails.
    """
    logger.info(
        "run_pre_extraction_validation: canonical_url_id=%d validated_by=%r",
        canonical_url_id, validated_by,
    )

    # -------------------------------------------------------------------------
    # Step 1 — Fetch all URL registry rows for this phone
    # -------------------------------------------------------------------------
    url_rows = await asyncio.to_thread(
        fetch_url_registry_for_phone, canonical_url_id
    )

    total_urls_registered = len(url_rows)
    urls_not_attempted = sum(
        1 for r in url_rows if r["status"] in _NOT_ATTEMPTED_STATUSES
    )
    urls_scraped_success = sum(
        1 for r in url_rows if r["status"] in _SUCCESS_STATUSES
    )

    # -------------------------------------------------------------------------
    # Step 2 — OEM URL status (None = not registered, bool = registered + checked)
    # -------------------------------------------------------------------------
    oem_rows = [r for r in url_rows if r["site_name"].endswith("_official")]
    if not oem_rows:
        official_url_scraped = None        # Not registered — no warning needed
    else:
        official_url_scraped = any(
            r["status"] in _SUCCESS_STATUSES for r in oem_rows
        )

    # -------------------------------------------------------------------------
    # Step 3 — GSMArena URL status
    # -------------------------------------------------------------------------
    gsmarena_rows = [r for r in url_rows if r["site_name"] == "gsmarena"]
    if not gsmarena_rows:
        gsmarena_scraped = None
    else:
        gsmarena_scraped = any(
            r["status"] in _SUCCESS_STATUSES for r in gsmarena_rows
        )

    # -------------------------------------------------------------------------
    # Step 4 — YouTube search status
    # -------------------------------------------------------------------------
    search_logs = await asyncio.to_thread(
        fetch_youtube_search_log, canonical_url_id
    )
    youtube_search_done = any(
        log["search_status"] in _YOUTUBE_DONE_STATUSES for log in search_logs
    )

    # -------------------------------------------------------------------------
    # Step 5 — Transcript availability
    # -------------------------------------------------------------------------
    transcript_rows = await asyncio.to_thread(
        fetch_transcript_availability, canonical_url_id
    )
    transcript_available = len(transcript_rows) > 0

    # -------------------------------------------------------------------------
    # Step 6 — Determine blockers and warnings
    # -------------------------------------------------------------------------
    blocking_reasons: list[dict] = []
    warnings: list[dict] = []

    if urls_not_attempted > 0:
        unattempted_ids = [
            r["url_id"] for r in url_rows
            if r["status"] in _NOT_ATTEMPTED_STATUSES
        ]
        blocking_reasons.append({
            "code":    "urls_not_attempted",
            "url_ids": unattempted_ids,
            "message": (
                f"{urls_not_attempted} URL(s) registered but not yet scraped. "
                f"url_ids: {unattempted_ids}"
            ),
        })

    if urls_scraped_success == 0:
        blocking_reasons.append({
            "code": "no_scraped_sources",
            "message": (
                "No source files have been successfully scraped. "
                "At least one scraped source is required before extraction."
            ),
        })

    if not youtube_search_done:
        blocking_reasons.append({
            "code": "no_youtube_search",
            "message": (
                "YouTube search has not been run for this phone. "
                "Run a YouTube search before extraction (even if zero results expected)."
            ),
        })

    # Warnings — not blocking
    if official_url_scraped is False:
        oem_ids = [r["url_id"] for r in oem_rows]
        warnings.append({
            "code": "oem_url_not_scraped",
            "message": (
                f"OEM official URL registered but scrape failed or not attempted. "
                f"url_ids: {oem_ids}. "
                "Extraction will proceed without OEM data — lower priority source."
            ),
        })

    if not transcript_available:
        warnings.append({
            "code": "no_transcript",
            "message": (
                "No fetched transcript found. "
                "Extraction will proceed from scraped markdown only. "
                "India-specific fields (charger_in_box, colour availability) "
                "may have lower coverage."
            ),
        })

    can_proceed = len(blocking_reasons) == 0

    # -------------------------------------------------------------------------
    # Step 7 — Write to DB and return
    # -------------------------------------------------------------------------
    unscraped_url_ids = [
        r["url_id"] for r in url_rows
        if r["status"] in _NOT_ATTEMPTED_STATUSES
    ]

    payload = {
        "canonical_url_id":      canonical_url_id,
        "validated_by":          validated_by,
        "total_urls_registered": total_urls_registered,
        "urls_not_attempted":    urls_not_attempted,
        "urls_scraped_success":  urls_scraped_success,
        "official_url_scraped":  official_url_scraped,
        "gsmarena_scraped":      gsmarena_scraped,
        "youtube_search_done":   youtube_search_done,
        "transcript_available":  transcript_available,
        "can_proceed":           can_proceed,
        "blocking_reasons":      blocking_reasons,
        "warnings":              warnings,
        "unscraped_url_ids":     unscraped_url_ids,
    }

    record = await asyncio.to_thread(insert_pre_extraction_validation, payload)

    logger.info(
        "run_pre_extraction_validation: canonical_url_id=%d can_proceed=%s "
        "blockers=%d warnings=%d",
        canonical_url_id, can_proceed,
        len(blocking_reasons), len(warnings),
    )
    return record


async def get_validation_status(canonical_url_id: int) -> dict | None:
    """
    Returns the most recent validation record for this phone, or None if
    no validation has been run yet.
    """
    return await asyncio.to_thread(fetch_latest_validation, canonical_url_id)


# ===========================================================================
# Phase 7.5 — Pre-UI System Validation (Task 7.5.1)
# ===========================================================================

from app.repositories.extraction_repository import (  # noqa: E402 — deferred to avoid circular
    fetch_final_merged_json,
    fetch_latest_pre_ui_validation,
    fetch_latest_spec_output_for_phone,
    fetch_phone_experience_count,
    fetch_normalized_spec,
    fetch_selected_enrichment_candidates,
    insert_pre_ui_validation_run,
)
from app.services.conflict_resolver import (  # noqa: E402
    KNOWN_SPEC_TOP_LEVEL_KEYS,
    REQUIRED_FIELDS,
    NUMERIC_FIELD_PATTERNS,
    BOOLEAN_FIELD_PATTERNS,
    _get_at_path,
    _count_nulls,
    _path_has_secondary_index,
)


async def run_pre_ui_validation(
    final_id: int,
    url_registry_id: int,
    normalized_id: int,
) -> dict:
    """
    Phase 7.5 — Layer 1 Pre-UI System Validation.
    Runs after conflict resolution, before admin gate opens.

    CHECKS:
      1. Schema structure: top-level keys must be a subset of KNOWN_SPEC_TOP_LEVEL_KEYS.
         Unexpected keys = hard block.
      2. Required fields present and non-null:
           basic.model_name, variants[0].ram_capacity, displays[0].panel_type
           variants array has >= 1 element, displays array >= 1, cameras.lenses >= 1.
      3. Type correctness: numeric fields must be int/float; booleans must be bool.
      4. Evidence integrity: warning only — non-null fields missing evidence_json
         entries are noted but do not block.
      5. Run B minimum: >= 1 phone_experiences row with is_suppressed=FALSE AND
         confidence >= 0.50. Warning if zero.
      6. Confidence floor: enrichment fields with confidence < 0.30 — warning only.

    Returns:
        {"passed": bool, "errors": list[dict], "warnings": list[dict],
         "pre_ui_val_id": int}
    """
    import re as _re
    from datetime import datetime, timezone

    logger.info(
        "run_pre_ui_validation: START final_id=%d normalized_id=%d",
        final_id, normalized_id,
    )

    # Fetch final_merged_json
    final_row = await asyncio.to_thread(fetch_final_merged_json, final_id)
    if final_row is None:
        raise ValueError(
            f"run_pre_ui_validation: final_id={final_id} not found in final_merged_json."
        )
    final_json: dict = final_row.get("final_json") or {}

    errors:   list[dict] = []
    warnings: list[dict] = []

    # ------------------------------------------------------------------
    # Check 1 — Schema structure: no unexpected top-level keys
    # ------------------------------------------------------------------
    for key in final_json.keys():
        if key not in KNOWN_SPEC_TOP_LEVEL_KEYS:
            errors.append({
                "check":   "unexpected_top_level_key",
                "field":   key,
                "message": (
                    f"Top-level key '{key}' is not a recognised spec section. "
                    f"Known keys: {sorted(KNOWN_SPEC_TOP_LEVEL_KEYS)}."
                ),
            })

    # ------------------------------------------------------------------
    # Check 2 — Required fields present and non-null
    # ------------------------------------------------------------------
    # Array minimum lengths
    for arr_key, min_len, label in [
        ("variants",         1, "variants"),
        ("displays",         1, "displays"),
    ]:
        arr = final_json.get(arr_key)
        if not isinstance(arr, list) or len(arr) < min_len:
            actual = len(arr) if isinstance(arr, list) else 0
            errors.append({
                "check":   "array_empty",
                "field":   arr_key,
                "message": (
                    f"{label} array has {actual} element(s); minimum required is {min_len}."
                ),
            })

    # C4 fix: camera_lenses is a top-level array — not nested under cameras.lenses
    camera_lenses = final_json.get("camera_lenses")
    if not isinstance(camera_lenses, list) or len(camera_lenses) < 1:
        actual_l = len(camera_lenses) if isinstance(camera_lenses, list) else 0
        errors.append({
            "check":   "array_empty",
            "field":   "camera_lenses",
            "message": (
                f"camera_lenses array has {actual_l} element(s); minimum required is 1."
            ),
        })

    # Required scalar fields
    for field_path, label in REQUIRED_FIELDS:
        value = _get_at_path(final_json, field_path)
        if value is None:
            errors.append({
                "check":   "required_field_null",
                "field":   field_path,
                "message": f"{label} is null or missing — required field.",
            })

    # ------------------------------------------------------------------
    # Check 3 — Type correctness (numeric + boolean)
    # C8: Fields on secondary array indices (displays[1+], lenses[2+]) are
    #     demoted to warnings — they are often incomplete for foldable phones.
    # ------------------------------------------------------------------
    def _check_type(
        obj: dict,
        wildcard_path: str,
        expected: type | tuple,
        type_label: str,
    ) -> tuple[list[dict], list[dict]]:
        """Expands wildcard paths and checks all matching values.
        Returns (hard_errors, warnings) split by array index.
        """
        _errors: list[dict] = []
        _warns:  list[dict] = []
        concrete_paths = _expand_wildcard_path(obj, wildcard_path)
        for path in concrete_paths:
            v = _get_at_path(obj, path)
            if v is None:
                continue  # null allowed
            if not isinstance(v, expected):
                entry = {
                    "check":   "type_error",
                    "field":   path,
                    "message": (
                        f"Expected {type_label}, got {type(v).__name__} "
                        f"with value {v!r}."
                    ),
                }
                # C8: secondary index → warning, primary → error
                if _path_has_secondary_index(path):
                    _warns.append(entry)
                else:
                    _errors.append(entry)
        return _errors, _warns

    for wpat in NUMERIC_FIELD_PATTERNS:
        _errs, _wrns = _check_type(final_json, wpat, (int, float), "numeric")
        errors.extend(_errs)
        warnings.extend(_wrns)

    for wpat in BOOLEAN_FIELD_PATTERNS:
        _errs, _wrns = _check_type(final_json, wpat, bool, "boolean")
        errors.extend(_errs)
        warnings.extend(_wrns)

    # ------------------------------------------------------------------
    # Check 4 — Evidence integrity (WARNING only)
    # ------------------------------------------------------------------
    # C1: Use renamed function fetch_latest_spec_output_for_phone
    ext_output = await asyncio.to_thread(
        fetch_latest_spec_output_for_phone, url_registry_id
    )
    evidence_json: dict = (ext_output or {}).get("evidence_json") or {}

    # Fetch enrichment candidates once — used by both Check 4 and Check 6.
    enrichment_candidates: list[dict] = []
    enrichment_run_id = final_row.get("enrichment_run_id")
    if enrichment_run_id is not None:
        enrichment_candidates = await asyncio.to_thread(
            fetch_selected_enrichment_candidates, enrichment_run_id
        )
    enriched_fields = {c["field_path"] for c in enrichment_candidates}

    # Collect all non-null leaf fields in final_json
    non_null_fields = _collect_non_null_paths(final_json)
    for field_path in non_null_fields:
        if field_path in evidence_json:
            continue  # Has Run A evidence
        if field_path in enriched_fields:
            continue  # Enrichment fill — evidence is in enrichment_field_candidates
        warnings.append({
            "check":   "missing_evidence",
            "field":   field_path,
            "message": (
                f"Value present but no evidence_json entry and not from enrichment. "
                f"Admin tooltip will show greyed-out indicator."
            ),
        })

    # ------------------------------------------------------------------
    # Check 5 — Run B minimum (WARNING only)
    # ------------------------------------------------------------------
    experience_count = await asyncio.to_thread(
        fetch_phone_experience_count, url_registry_id
    )
    if experience_count == 0:
        warnings.append({
            "check":   "run_b_empty",
            "message": (
                "No phone_experiences rows found with is_suppressed=FALSE "
                "AND confidence >= 0.50. Transcript may not have been available. "
                "Experience section will be empty in admin UI."
            ),
        })

    # Check 6 — Confidence floor (WARNING only)
    for cand in enrichment_candidates:
        conf = float(cand.get("confidence", 1.0))
        if conf < 0.30:
            warnings.append({
                "check":   "low_confidence",
                "field":   cand["field_path"],
                "message": (
                    f"Enrichment confidence {conf:.2f} is below the floor 0.30. "
                    f"Value may be unreliable. Admin should verify."
                ),
            })

    # ------------------------------------------------------------------
    # Persist result
    # ------------------------------------------------------------------
    passed = len(errors) == 0
    validated_at = datetime.now(timezone.utc).isoformat()

    pre_ui_val_id = await asyncio.to_thread(
        insert_pre_ui_validation_run,
        {
            "url_registry_id": url_registry_id,
            "normalized_id":   normalized_id,
            "status":          "passed" if passed else "failed",
            "errors":          errors,
            "error_count":     len(errors),
            "warnings":        warnings,
            "warning_count":   len(warnings),
            "validated_at":    validated_at,
        },
    )

    logger.info(
        "run_pre_ui_validation: COMPLETE final_id=%d passed=%s "
        "errors=%d warnings=%d pre_ui_val_id=%d",
        final_id, passed, len(errors), len(warnings), pre_ui_val_id,
    )

    return {
        "passed":         passed,
        "errors":         errors,
        "warnings":       warnings,
        "pre_ui_val_id":  pre_ui_val_id,
    }


async def get_pre_ui_validation_status(normalized_id: int) -> dict | None:
    """
    Returns the most recent pre_ui_validation_runs row for this normalized_id.
    Returns None if no validation has been run yet.
    """
    return await asyncio.to_thread(fetch_latest_pre_ui_validation, normalized_id)


# ---------------------------------------------------------------------------
# Private helpers for Phase 7.5
# ---------------------------------------------------------------------------

def _expand_wildcard_path(obj: dict, wildcard_path: str) -> list[str]:
    """
    Expands a wildcard path like 'displays[*].panel_type' into concrete paths
    like ['displays[0].panel_type', 'displays[1].panel_type'] based on the
    actual array lengths in obj.
    """
    import re as _re
    # Find array wildcard segments
    m = _re.search(r"(\w+)\[\*\]", wildcard_path)
    if not m:
        # No wildcard — return as-is
        return [wildcard_path]

    arr_key = m.group(1)
    prefix  = wildcard_path[:m.start()]       # e.g. "cameras."
    suffix  = wildcard_path[m.end():]          # e.g. ".megapixels"

    # Navigate to the array
    parent_path = prefix.rstrip(".")
    parent_obj  = _get_at_path(obj, parent_path) if parent_path else obj

    if not isinstance(parent_obj, dict):
        return []

    arr = parent_obj.get(arr_key)
    if not isinstance(arr, list):
        return []

    results = []
    for i in range(len(arr)):
        concrete = f"{prefix}{arr_key}[{i}]{suffix}"
        # Recurse to handle nested wildcards (e.g. "cameras.lenses[*].megapixels")
        results.extend(_expand_wildcard_path(obj, concrete))
    return results


def _collect_non_null_paths(obj: object, prefix: str = "") -> list[str]:
    """
    Flattens a nested dict/list into a list of dotted/bracket paths
    for all non-null leaf values.
    """
    paths: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            paths.extend(_collect_non_null_paths(v, path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            # Skip plain string items — these are junction array values
            # (e.g. ["HDR10", "LTE Band 1"]) and are intentionally ungrounded.
            # Counting them as leaf paths generates hundreds of spurious
            # missing_evidence warnings per flagship phone.
            if isinstance(v, str):
                continue
            path = f"{prefix}[{i}]"
            paths.extend(_collect_non_null_paths(v, path))
    else:
        if obj is not None:
            paths.append(prefix)
    return paths


# ===========================================================================
# Phase 9 — Pre-Commit Validation (Layer 3, Task 9.1)
# ===========================================================================

from app.repositories.extraction_repository import (  # noqa: E402 — deferred import
    fetch_gate_conditions_live,
    fetch_latest_commit_validation,
    fetch_non_suppressed_experiences_for_commit,
    fetch_brand_for_phone,
    insert_commit_validation_run,
)
from app.config.field_mapping import SCALAR_FK_MAP  # noqa: E402
from app.services.normalizer import LOOKUP_CACHE, clean_for_lookup  # noqa: E402
from app.services.conflict_resolver import (        # noqa: E402 — section-level deferred
    KNOWN_SPEC_TOP_LEVEL_KEYS,
    REQUIRED_FIELDS,
    NUMERIC_FIELD_PATTERNS,
    BOOLEAN_FIELD_PATTERNS,
    _path_has_secondary_index,
)

# S2-P1-2: _FOLDABLE_BRANDS removed — brand-based foldable detection was causing
# every standard Galaxy/Pixel/Moto device to generate spurious foldable_single_display
# warnings. Check 5 now uses display_type presence (Inner+Cover) as the signal.
# fetch_brand_for_phone is retained for other potential uses in this module.

# Experience confidence floor for committed entries
_COMMITTED_EXPERIENCE_CONFIDENCE_FLOOR = 0.50


async def run_pre_commit_validation(
    final_id: int,
    url_registry_id: int,
) -> dict:
    """
    Phase 9 — Layer 3 Backend Validation.
    Runs after admin approval, before the DB transaction opens.

    CHECKS:
      1. Full Pydantic model validation on current final_merged_json.final_json.
         Re-runs the same model as Layer 1 on post-admin-edit data.

      2. FK pre-resolution check:
         For every field in SCALAR_FK_MAP, verify the value exists in LOOKUP_CACHE.
         Values not in cache must be NULL (not raw strings).
         Raw string in an FK field → hard failure.

      3. Five-condition gate check (atomic, reads live DB — not cached booleans):
         has_unresolved_conflicts, pending_staging_values, spec_human_approved,
         experience_human_approved, experience_entries_reviewed.
         All must pass. Any failure → hard block.

      4. Run B final check:
         All non-suppressed phone_experiences rows must have:
           - non-null, non-empty experience_text
           - confidence >= 0.50
         Failure → hard block.

      5. Structural consistency (WARNING, not block):
         Foldable brand + only one display entry → warning
         Non-foldable brand + displays[1] present → warning

    ON FAILURE:
         INSERT commit_validation_runs with status='failed', errors=JSONB
         Returns {"passed": False, "errors": [...], "commit_val_id": int}
         Caller must NOT open the DB transaction.

    ON SUCCESS:
         INSERT commit_validation_runs with status='passed'
         Returns {"passed": True, "errors": [], "commit_val_id": int}
         Caller may open the DB transaction.
    """
    logger.info(
        "run_pre_commit_validation: START final_id=%d url_registry_id=%d",
        final_id, url_registry_id,
    )

    errors:   list[dict] = []
    warnings: list[dict] = []

    # Fetch the current final_json
    final_row = await asyncio.to_thread(fetch_final_merged_json, final_id)
    if final_row is None:
        raise ValueError(
            f"run_pre_commit_validation: final_id={final_id} not found."
        )
    final_json: dict = final_row.get("final_json") or {}

    # ------------------------------------------------------------------
    # Check 1 — Full Pydantic model validation on post-edit final_json
    # ------------------------------------------------------------------
    # Re-use the same structural checks as Layer 1:
    # unexpected top-level keys, required fields, numeric/boolean types
    for key in final_json.keys():
        if key not in KNOWN_SPEC_TOP_LEVEL_KEYS:
            errors.append({
                "check":   "unexpected_top_level_key",
                "field":   key,
                "message": (
                    f"Top-level key '{key}' is not a recognised spec section. "
                    f"This may have been introduced by a malformed admin edit."
                ),
            })

    for field_path, label in REQUIRED_FIELDS:
        value = _get_at_path(final_json, field_path)
        if value is None:
            errors.append({
                "check":   "required_field_null",
                "field":   field_path,
                "message": (
                    f"{label} is null or missing after admin edit. "
                    f"This field is required for commit."
                ),
            })

    for arr_key, min_len in [("variants", 1), ("displays", 1)]:
        arr = final_json.get(arr_key)
        if not isinstance(arr, list) or len(arr) < min_len:
            errors.append({
                "check":   "array_empty",
                "field":   arr_key,
                "message": f"{arr_key} must have >= {min_len} element(s).",
            })

    # C4 fix: camera_lenses is now a top-level array, not nested under cameras
    camera_lenses = final_json.get("camera_lenses")
    if not isinstance(camera_lenses, list) or len(camera_lenses) < 1:
        errors.append({
            "check":   "array_empty",
            "field":   "camera_lenses",
            "message": "camera_lenses must have >= 1 element.",
        })

    # Type checks (numeric, boolean)
    # P9-1: returns (hard_errors, warnings) split on _path_has_secondary_index,
    # matching the Layer 1 behaviour in _check_type (Phase 7.5).
    def _l3_type_check(
        obj: dict, wildcard_path: str, expected: type | tuple, type_label: str
    ) -> tuple[list[dict], list[dict]]:
        _errors: list[dict] = []
        _warns:  list[dict] = []
        for path in _expand_wildcard_path(obj, wildcard_path):
            v = _get_at_path(obj, path)
            if v is None:
                continue
            if not isinstance(v, expected):
                entry = {
                    "check":   "type_error",
                    "field":   path,
                    "message": (
                        f"Expected {type_label}, got {type(v).__name__} "
                        f"with value {v!r}. Admin edit may have introduced wrong type."
                    ),
                }
                # Secondary array index (foldable secondary display, tertiary lens)
                # are demoted to warnings, not hard failures.
                if _path_has_secondary_index(path):
                    _warns.append(entry)
                else:
                    _errors.append(entry)
        return _errors, _warns

    for wpat in NUMERIC_FIELD_PATTERNS:
        _errs, _wrns = _l3_type_check(final_json, wpat, (int, float), "numeric")
        errors.extend(_errs)
        warnings.extend(_wrns)
    for wpat in BOOLEAN_FIELD_PATTERNS:
        _errs, _wrns = _l3_type_check(final_json, wpat, bool, "boolean")
        errors.extend(_errs)
        warnings.extend(_wrns)

    # ------------------------------------------------------------------
    # Check 2 — FK pre-resolution check
    # ------------------------------------------------------------------
    if not LOOKUP_CACHE:
        logger.warning(
            "run_pre_commit_validation: LOOKUP_CACHE is empty — "
            "FK check skipped. build_lookup_cache() may not have run."
        )
    else:
        for wildcard_path, table_path in SCALAR_FK_MAP.items():
            concrete_paths = _expand_wildcard_path(final_json, wildcard_path)
            for field_path in concrete_paths:
                value = _get_at_path(final_json, field_path)
                if value is None:
                    continue  # NULL is valid — FK not resolved but not a raw string
                if not isinstance(value, str):
                    continue  # numeric/bool FK values not relevant to string check
                cleaned = clean_for_lookup(str(value))
                lookup = LOOKUP_CACHE.get(table_path, {})
                if cleaned not in lookup:
                    errors.append({
                        "check":   "fk_not_resolved",
                        "field":   field_path,
                        "value":   value,
                        "table":   table_path,
                        "message": (
                            f"Value {value!r} is not in LOOKUP_CACHE for "
                            f"{table_path} and is not NULL. "
                            f"Resolve via staging queue before committing."
                        ),
                    })

    # ------------------------------------------------------------------
    # Check 3 — Five-condition gate (live DB re-read)
    # ------------------------------------------------------------------
    gate = await asyncio.to_thread(fetch_gate_conditions_live, final_id)
    if gate is None:
        errors.append({
            "check":   "five_condition_gate",
            "condition": "final_id_not_found",
            "message": f"final_id={final_id} not found in final_merged_json.",
        })
    else:
        gate_conditions = [
            (
                "has_unresolved_conflicts",
                gate.get("has_unresolved_conflicts") is True,
                "has_unresolved_conflicts is TRUE — all flagged conflicts must be resolved.",
            ),
            (
                "pending_staging_values",
                (gate.get("pending_staging_values") or 0) > 0,
                (
                    f"pending_staging_values={gate.get('pending_staging_values')} — "
                    f"all staging entries must be resolved."
                ),
            ),
            (
                "spec_human_approved",
                not gate.get("spec_human_approved"),
                "spec_human_approved is FALSE — spec has not been approved.",
            ),
            (
                "experience_human_approved",
                not gate.get("experience_human_approved"),
                "experience_human_approved is FALSE — experiences have not been approved.",
            ),
            (
                "experience_entries_reviewed",
                not gate.get("experience_entries_reviewed"),
                "experience_entries_reviewed is FALSE — experience entries have not been reviewed.",
            ),
        ]
        for condition_name, is_failing, message in gate_conditions:
            if is_failing:
                errors.append({
                    "check":     "five_condition_gate",
                    "condition": condition_name,
                    "message":   message,
                })

    # ------------------------------------------------------------------
    # Check 4 — Run B final check
    # ------------------------------------------------------------------
    experiences = await asyncio.to_thread(
        fetch_non_suppressed_experiences_for_commit, url_registry_id
    )
    for exp in experiences:
        text = exp.get("experience_text")
        if not text or not str(text).strip():
            errors.append({
                "check":         "experience_text_null",
                "experience_id": exp["experience_id"],
                "message": (
                    f"Non-suppressed experience_id={exp['experience_id']} "
                    f"has null or empty experience_text. Edit or suppress before committing."
                ),
            })
        conf = float(exp.get("confidence") or 0.0)
        if conf < _COMMITTED_EXPERIENCE_CONFIDENCE_FLOOR:
            errors.append({
                "check":         "experience_low_confidence",
                "experience_id": exp["experience_id"],
                "confidence":    conf,
                "message": (
                    f"Non-suppressed experience_id={exp['experience_id']} "
                    f"has confidence={conf:.2f} below floor "
                    f"{_COMMITTED_EXPERIENCE_CONFIDENCE_FLOOR}. "
                    f"Suppress or re-run Run B."
                ),
            })

    # ------------------------------------------------------------------
    # Check 5 — Structural consistency (warnings only)
    # S2-P1-2: replaced brand-based (_FOLDABLE_BRANDS) detection with
    # display_type structure-based detection. Brand check was firing for
    # every standard Samsung/Pixel/Motorola phone.
    # ------------------------------------------------------------------
    displays = final_json.get("displays") or []
    display_count = len(displays)

    has_inner = any(
        str(d.get("display_type", "")).lower() == "inner"
        for d in displays if isinstance(d, dict)
    )
    has_cover = any(
        str(d.get("display_type", "")).lower() == "cover"
        for d in displays if isinstance(d, dict)
    )
    is_foldable_structure = has_inner and has_cover

    if is_foldable_structure and display_count < 2:
        warnings.append({
            "check":   "foldable_single_display",
            "message": (
                "Phone has Inner/Cover display_type entries but fewer than 2 display objects. "
                "Verify the display structure is complete."
            ),
        })

    if not is_foldable_structure and display_count > 1:
        warnings.append({
            "check":   "non_foldable_multiple_displays",
            "message": (
                f"Phone has {display_count} display entries but no Inner/Cover type detected. "
                "Verify this is not a data error."
            ),
        })

    # ------------------------------------------------------------------
    # Persist result
    # ------------------------------------------------------------------
    passed = len(errors) == 0
    validated_at = datetime.now(timezone.utc).isoformat()

    commit_val_id = await asyncio.to_thread(
        insert_commit_validation_run,
        {
            "final_id":        final_id,
            "url_registry_id": url_registry_id,
            "status":          "passed" if passed else "failed",
            "errors":          errors,
            "error_count":     len(errors),
            "warnings":        warnings,
            "warning_count":   len(warnings),
            "validated_at":    validated_at,
        },
    )

    logger.info(
        "run_pre_commit_validation: COMPLETE final_id=%d passed=%s "
        "errors=%d warnings=%d commit_val_id=%d",
        final_id, passed, len(errors), len(warnings), commit_val_id,
    )

    return {
        "passed":         passed,
        "errors":         errors,
        "warnings":       warnings,
        "commit_val_id":  commit_val_id,
    }


async def get_pre_commit_validation_status(final_id: int) -> dict | None:
    """
    Returns the most recent commit_validation_runs row for this final_id.
    Returns None if no validation has been run yet.
    Used by the admin UI to show the current Layer 3 state without re-running.
    P9-6 fix: delegates to repo layer instead of executing raw DB query here.
    """
    return await asyncio.to_thread(fetch_latest_commit_validation, final_id)
