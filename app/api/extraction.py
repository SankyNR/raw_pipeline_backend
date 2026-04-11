"""
Extraction Pipeline API Router (app/api/extraction.py)

Phase 1 — Pre-Extraction Validation:
    POST /extraction/validate
    GET  /extraction/validation-status/{canonical_url_id}

Phase 2 — Run A: Spec Extraction:
    POST /extraction/run-a

Phase 3 — Run A + Run B: Parallel Trigger:
    POST /extraction/run-all

Phase 4 — Normalisation:
    POST /extraction/normalise
    POST /extraction/cache/refresh

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

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.validation_service import (
    run_pre_extraction_validation,
    get_validation_status,
    run_pre_ui_validation,
    get_pre_ui_validation_status,
)
from app.services.langextract_run_a import run_spec_extraction_lx
from app.services.langextract_run_b import run_experience_extraction_batch
from app.services.normalizer import run_normalisation
from app.services.gap_analyzer import detect_missing_fields
from app.services.enrichment_orchestrator import run_enrichment
from app.services.conflict_resolver import (
    detect_and_resolve_conflicts,
    build_final_merged_json,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/extraction", tags=["extraction"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ValidateRequest(BaseModel):
    canonical_url_id: int


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
    Runs pre-extraction validation for the given canonical_url_id.

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
        HTTP 404: If canonical_url_id does not exist in url_registry.
        HTTP 500: On unexpected DB or service error.
    """
    try:
        record = await run_pre_extraction_validation(
            canonical_url_id=body.canonical_url_id,
            validated_by="admin_manual",
        )
        return record
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception(
            "validate_phone: unexpected error for canonical_url_id=%d",
            body.canonical_url_id,
        )
        raise HTTPException(status_code=500, detail=str(exc))


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
# Task 2.3 — POST /extraction/run-a
# ---------------------------------------------------------------------------

class RunARequest(BaseModel):
    canonical_url_id: int
    raw_source_ids: list[int]
    raw_transcript_id: int | None = None
    brand: str
    model_name: str
    schema_version: str = "v1"


class RunAResponse(BaseModel):
    success: bool
    output_id: int | None
    run_id: int | None = None    # B6: None for failure path — -1 is not a valid DB ID
    message: str
    failed_source_ids: list[int] = []


# Prefix used by run_spec_extraction gate check (A2) so the API can distinguish
# "validation not passed" (422) from "url_id not found" (404).
_GATE_ERROR_PREFIX = "Pre-extraction validation not passed"


@router.post("/run-a", response_model=RunAResponse)
async def trigger_run_a(body: RunARequest):
    """
    Triggers Run A (spec extraction) for the given phone.

    Run A flow:
    1. Gate check — verifies Phase 1 validation passed (can_proceed=true).
       This is enforced in run_spec_extraction, not delegated to the UI.
    2. Fetches all source files from Supabase Storage (scraped markdown).
    3. Optionally fetches the processed/translated transcript.
    4. Concatenates sources in priority order: OEM → GSMArena → Smartprix → transcript.
    5. Calls Gemini with the ECD + spec template to extract structured specs.
    6. Stores partial_json + evidence_json in spec_extraction_output.
    7. Records the extraction run in spec_extraction_runs.

    Prerequisite: POST /extraction/validate must have returned can_proceed=TRUE
    for this canonical_url_id. The validation gate IS enforced server-side in
    run_spec_extraction — calling this endpoint without a passing validation
    record will return HTTP 422.

    Args:
        canonical_url_id:  The url_registry.url_id anchor for this phone.
        raw_source_ids:    List of raw_scraped_data.raw_id values to include.
        raw_transcript_id: youtube_raw_transcript_data.raw_transcript_id, or null.
        brand:             Brand name (for logging and legacy fallback).
        model_name:        Model name (keyword-matched for foldable/flippable detection).
        schema_version:    Extraction schema version. Default 'v1'.

    Returns:
        Run result including output_id, run_id, field counts, failed_source_ids.

    Raises:
        HTTP 400: If raw_source_ids is empty and no transcript provided.
        HTTP 404: If canonical_url_id does not exist in url_registry.
        HTTP 422: If Phase 1 validation has not been run or can_proceed=false.
        HTTP 500: On Gemini error, storage fetch error, or DB error.
    """
    if not body.raw_source_ids and body.raw_transcript_id is None:
        raise HTTPException(
            status_code=400,
            detail="raw_source_ids cannot be empty when raw_transcript_id is null. "
                   "Provide at least one source file to extract from.",
        )

    try:
        result = await run_spec_extraction_lx(
            url_registry_id=body.canonical_url_id,
            raw_source_ids=body.raw_source_ids,
            raw_transcript_id=body.raw_transcript_id,
            brand=body.brand,
            model_name=body.model_name,
            schema_version=body.schema_version,
        )
        return result

    except ValueError as exc:
        # Gate check failure (A2) → 422 Unprocessable Entity
        # Missing url_registry_id → 404 Not Found
        if str(exc).startswith(_GATE_ERROR_PREFIX):
            raise HTTPException(status_code=422, detail=str(exc))
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception(
            "trigger_run_a: unexpected error for canonical_url_id=%d",
            body.canonical_url_id,
        )
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Task 3.3 — POST /extraction/run-all  (parallel Run A + Run B)
# ▶ CHANGED L6: accepts raw_transcript_ids: list[int] (plural) for multi-transcript
#   Run B. Keeps raw_transcript_id: int as deprecated alias that wraps to a list.
# ---------------------------------------------------------------------------

class RunAllRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    canonical_url_id: int
    raw_source_ids: list[int]
    raw_transcript_ids: list[int] = []   # NEW — primary (plural) field
    raw_transcript_id: int | None = None # DEPRECATED alias — wraps to [raw_transcript_id]
    brand: str
    model_name: str
    schema_version: str = "v1"


class RunBResult(BaseModel):
    success: bool
    exp_run_id: int | None = None
    experiences_extracted: int = 0
    experiences_filtered: int = 0
    transcripts_processed: int = 0
    transcripts_failed: int = 0
    message: str
    error: str | None = None


class RunAllResponse(BaseModel):
    run_a: RunAResponse
    run_b: RunBResult
    both_succeeded: bool


@router.post("/run-all", response_model=RunAllResponse)
async def trigger_run_all(body: RunAllRequest):
    """
    Triggers Run A (LangExtract spec extraction) and Run B (LangExtract experience
    extraction) in parallel via asyncio.gather.

    Run A and Run B are independent — failure of one does NOT block the other.
    Both results are returned independently.

    Run B accepts:
      raw_transcript_ids: list[int]  — NEW primary field (multi-transcript)
      raw_transcript_id:  int        — DEPRECATED alias (single transcript, still accepted)
      If both present, raw_transcript_ids is used. The deprecated alias is ignored.

    Both runs enforce the Phase 1 validation gate (can_proceed=true).
    Returns HTTP 422 if gate has not been passed.

    Raises:
        HTTP 422: If Phase 1 validation gate has not been passed.
        HTTP 500: On unexpected orchestration error.
    """
    # Resolve transcript ID list — new plural field takes precedence over deprecated alias
    transcript_ids: list[int]
    if body.raw_transcript_ids:
        transcript_ids = body.raw_transcript_ids
    elif body.raw_transcript_id is not None:
        transcript_ids = [body.raw_transcript_id]   # backward compat wrap
    else:
        transcript_ids = []

    # Fire Run A (LX) + Run B batch in parallel
    run_a_coro = run_spec_extraction_lx(
        url_registry_id=body.canonical_url_id,
        raw_source_ids=body.raw_source_ids,
        raw_transcript_id=transcript_ids[0] if transcript_ids else None,
        brand=body.brand,
        model_name=body.model_name,
        schema_version=body.schema_version,
    )

    if transcript_ids:
        run_b_coro = run_experience_extraction_batch(
            url_registry_id=body.canonical_url_id,
            raw_transcript_ids=transcript_ids,
            brand=body.brand,
            model_name=body.model_name,
            schema_version=body.schema_version,
        )
    else:
        async def _no_transcripts():
            return []
        run_b_coro = _no_transcripts()

    results = await asyncio.gather(run_a_coro, run_b_coro, return_exceptions=True)
    run_a_result, run_b_raw = results

    # -------------------------------------------------------------------------
    # Process Run A result
    # -------------------------------------------------------------------------
    if isinstance(run_a_result, Exception):
        exc = run_a_result
        if str(exc).startswith(_GATE_ERROR_PREFIX):
            raise HTTPException(status_code=422, detail=str(exc))
        run_a_response = RunAResponse(
            success=False,
            output_id=None,
            run_id=None,
            message=f"Run A failed: {exc}",
            failed_source_ids=[],
        )
    else:
        run_a_response = RunAResponse(**run_a_result)

    # -------------------------------------------------------------------------
    # Process Run B batch results
    # run_b_raw is list[dict | Exception], one per transcript
    # -------------------------------------------------------------------------
    if isinstance(run_b_raw, Exception):
        # Gate-level failure before any tasks launched
        exc = run_b_raw
        if str(exc).startswith(_GATE_ERROR_PREFIX):
            raise HTTPException(status_code=422, detail=str(exc))
        run_b_response = RunBResult(
            success=False,
            message=f"Run B batch failed to start: {exc}",
            error=str(exc),
        )
    elif not run_b_raw:
        run_b_response = RunBResult(
            success=True,
            message="No transcripts provided — Run B skipped.",
        )
    else:
        successes = [r for r in run_b_raw if isinstance(r, dict) and r.get("success")]
        failures = [r for r in run_b_raw if isinstance(r, Exception)]
        total_extracted = sum(r.get("experiences_extracted", 0) for r in successes)
        total_filtered = sum(r.get("experiences_filtered", 0) for r in successes)
        best_exp_run_id = successes[0].get("exp_run_id") if successes else None

        run_b_response = RunBResult(
            success=len(failures) == 0,
            exp_run_id=best_exp_run_id,
            experiences_extracted=total_extracted,
            experiences_filtered=total_filtered,
            transcripts_processed=len(successes),
            transcripts_failed=len(failures),
            message=(
                f"Run B (LX) complete. {len(successes)}/{len(run_b_raw)} transcripts "
                f"succeeded. {total_extracted} experiences extracted, "
                f"{total_filtered} below confidence threshold."
            ),
            error=str(failures[0]) if failures else None,
        )

    return RunAllResponse(
        run_a=run_a_response,
        run_b=run_b_response,
        both_succeeded=run_a_response.success and run_b_response.success,
    )


# ---------------------------------------------------------------------------
# Phase L6 — POST /extraction/run-phone  (auto-resolve + parallel launch)
# ---------------------------------------------------------------------------

class RunPhoneRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    canonical_url_id: int
    schema_version: str = "v1"


class RunPhoneRunBSummary(BaseModel):
    transcripts_processed: int
    transcripts_failed: int
    experiences_extracted: int
    experiences_filtered: int
    exp_run_ids: list[int]
    error: str | None = None


class RunPhoneResponse(BaseModel):
    success: bool
    brand: str
    model_name: str
    canonical_url_id: int
    raw_source_ids_used: list[int]
    raw_transcript_ids_used: list[int]
    best_transcript_id: int | None
    run_a: RunAResponse
    run_b: RunPhoneRunBSummary
    both_succeeded: bool
    message: str


@router.post("/run-phone", response_model=RunPhoneResponse)
async def trigger_run_phone(body: RunPhoneRequest):
    """
    POST /extraction/run-phone — primary extraction trigger.

    Accepts only canonical_url_id. Auto-resolves all source IDs:
      1. brand + model_name       ← url_registry anchor row
      2. raw_source_ids           ← all raw_scraped_data with
                                    status IN ('scraped_raw', 'stored_mainDB')
      3. raw_transcript_ids       ← all youtube_raw_transcript_data with
                                    status='fetched_raw' (via video_registry)
      4. best_transcript_id       ← highest channel_reliability_score,
                                    then highest word_count_processed

    Fires Run A (one LangExtract spec call) and Run B batch (one call per
    transcript, semaphore-capped at 5) simultaneously via asyncio.gather.

    Run A receives best_transcript_id as its transcript input.
    Run B batch receives ALL raw_transcript_ids for full experience coverage.

    Raises:
        HTTP 404: If canonical_url_id not found in url_registry.
        HTTP 400: If no scraped sources found (cannot run extraction).
        HTTP 422: If Phase 1 validation gate not passed.
        HTTP 500: On unexpected orchestration error.
    """
    from app.repositories.extraction_repository import fetch_sources_for_run_phone

    # Step 1 — Auto-resolve sources
    try:
        resolved = await asyncio.to_thread(
            fetch_sources_for_run_phone, body.canonical_url_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    brand = resolved["brand"]
    model_name = resolved["model_name"]
    raw_source_ids = resolved["raw_source_ids"]
    raw_transcript_ids = resolved["raw_transcript_ids"]
    best_transcript_id = resolved["best_transcript_id"]

    if not raw_source_ids:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No scraped sources found for canonical_url_id={body.canonical_url_id} "
                f"({brand} {model_name}). Run the scraper first and ensure status is "
                "'scraped_raw' or 'stored_mainDB'."
            ),
        )

    logger.info(
        "trigger_run_phone: canonical_url_id=%d brand=%r model=%r "
        "raw_source_ids=%s raw_transcript_ids=%s best_transcript_id=%s",
        body.canonical_url_id, brand, model_name,
        raw_source_ids, raw_transcript_ids, best_transcript_id,
    )

    # Step 2 — Launch Run A + Run B in parallel
    run_a_coro = run_spec_extraction_lx(
        url_registry_id=body.canonical_url_id,
        raw_source_ids=raw_source_ids,
        raw_transcript_id=best_transcript_id,
        brand=brand,
        model_name=model_name,
        schema_version=body.schema_version,
    )

    if raw_transcript_ids:
        run_b_coro = run_experience_extraction_batch(
            url_registry_id=body.canonical_url_id,
            raw_transcript_ids=raw_transcript_ids,
            brand=brand,
            model_name=model_name,
            schema_version=body.schema_version,
        )
    else:
        async def _no_transcripts():
            return []
        run_b_coro = _no_transcripts()

    results = await asyncio.gather(run_a_coro, run_b_coro, return_exceptions=True)
    run_a_result, run_b_raw = results

    # -------------------------------------------------------------------------
    # Process Run A result
    # -------------------------------------------------------------------------
    if isinstance(run_a_result, Exception):
        exc = run_a_result
        if str(exc).startswith(_GATE_ERROR_PREFIX):
            raise HTTPException(status_code=422, detail=str(exc))
        logger.exception(
            "trigger_run_phone: Run A failed for canonical_url_id=%d: %s",
            body.canonical_url_id, exc,
        )
        run_a_response = RunAResponse(
            success=False,
            output_id=None,
            run_id=None,
            message=f"Run A failed: {exc}",
            failed_source_ids=[],
        )
    else:
        run_a_response = RunAResponse(**run_a_result)

    # -------------------------------------------------------------------------
    # Process Run B batch results
    # -------------------------------------------------------------------------
    if isinstance(run_b_raw, Exception):
        exc = run_b_raw
        if str(exc).startswith(_GATE_ERROR_PREFIX):
            raise HTTPException(status_code=422, detail=str(exc))
        run_b_summary = RunPhoneRunBSummary(
            transcripts_processed=0,
            transcripts_failed=len(raw_transcript_ids),
            experiences_extracted=0,
            experiences_filtered=0,
            exp_run_ids=[],
            error=str(exc),
        )
    else:
        b_successes = [r for r in run_b_raw if isinstance(r, dict) and r.get("success")]
        b_failures = [r for r in run_b_raw if isinstance(r, Exception)]
        run_b_summary = RunPhoneRunBSummary(
            transcripts_processed=len(b_successes),
            transcripts_failed=len(b_failures),
            experiences_extracted=sum(r.get("experiences_extracted", 0) for r in b_successes),
            experiences_filtered=sum(r.get("experiences_filtered", 0) for r in b_successes),
            exp_run_ids=[r["exp_run_id"] for r in b_successes if r.get("exp_run_id")],
            error=str(b_failures[0]) if b_failures else None,
        )

    both_succeeded = run_a_response.success and (
        not raw_transcript_ids
        or (
            not isinstance(run_b_raw, Exception)
            and all(isinstance(r, dict) and r.get("success") for r in run_b_raw)
        )
    )

    return RunPhoneResponse(
        success=run_a_response.success,
        brand=brand,
        model_name=model_name,
        canonical_url_id=body.canonical_url_id,
        raw_source_ids_used=raw_source_ids,
        raw_transcript_ids_used=raw_transcript_ids,
        best_transcript_id=best_transcript_id,
        run_a=run_a_response,
        run_b=run_b_summary,
        both_succeeded=both_succeeded,
        message=(
            f"run-phone complete for {brand} {model_name}. "
            f"Run A: {'ok' if run_a_response.success else 'FAILED'}. "
            f"Run B: {run_b_summary.transcripts_processed}/{len(raw_transcript_ids)} "
            f"transcripts OK."
        ),
    )


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
async def trigger_normalise(body: NormaliseRequest):
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
async def trigger_gap_analysis(body: GapAnalysisRequest):
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
async def trigger_enrichment(body: EnrichRequest) -> EnrichResponse:
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


# ===========================================================================
# Phase 7 — Conflict Resolution
# ===========================================================================

class ResolveConflictsRequest(BaseModel):
    normalized_id:     int
    enrichment_run_id: int


class ResolveConflictsResponse(BaseModel):
    success:             bool
    final_id:            int
    flagged_count:       int
    auto_resolved_count: int
    fill_count:          int
    concordant_count:    int
    message:             str


@router.post("/resolve-conflicts", response_model=ResolveConflictsResponse)
async def trigger_conflict_resolution(body: ResolveConflictsRequest):
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
        # Step 1 — Detect and resolve conflicts
        conflict_result = await detect_and_resolve_conflicts(
            normalized_id=body.normalized_id,
            enrichment_run_id=body.enrichment_run_id,
        )

        # Step 2 — Build final merged JSON
        final_id = await build_final_merged_json(
            normalized_id=body.normalized_id,
            enrichment_run_id=body.enrichment_run_id,
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
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception(
            "trigger_conflict_resolution: unexpected error normalized_id=%d",
            body.normalized_id,
        )
        raise HTTPException(status_code=500, detail=str(exc))


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
