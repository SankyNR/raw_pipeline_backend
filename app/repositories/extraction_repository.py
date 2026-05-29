"""
Phase 1 — Extraction Pipeline Repository

All DB queries used by the extraction pipeline.
Starts with pre-extraction validation (Task 1.1).

Supabase client pattern: get_client().schema("pipeline").table(...).execute()
All queries are synchronous (Supabase Python client is sync).
Async callers use asyncio.to_thread() where needed.
"""

import datetime
import logging

from app.core.supabase_client import get_client

_logger = logging.getLogger(__name__)

def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def update_normalized_json_field(normalized_id: int, field_path: str, new_value: any) -> None:
    """
    Patches a single field inside normalized_spec_json.normalized_json using jsonb_set.
    Used by staging/resolve to keep normalized_json in sync with final_json after FK resolution.
    field_path uses dot notation: 'camera.megapixels', 'connectivity.bands_5g'
    """
    from app.services.conflict_resolver import _get_at_path, _set_at_path, _parse_path
    import copy

    client = get_client()
    row = (
        client
        .schema("pipeline")
        .table("normalized_spec_json")
        .select("normalized_json")
        .eq("normalized_id", normalized_id)
        .single()
        .execute()
    )
    if not row.data:
        return

    normalized_json = copy.deepcopy(row.data["normalized_json"] or {})
    current_val = _get_at_path(normalized_json, field_path)
    if isinstance(current_val, list):
        if new_value not in current_val:
            current_val.append(new_value)
    else:
        _set_at_path(normalized_json, field_path, new_value)

    client.schema("pipeline").table("normalized_spec_json").update(
        {"normalized_json": normalized_json}
    ).eq("normalized_id", normalized_id).execute()


# ---------------------------------------------------------------------------
# Task 1.1 — URL coverage queries
# ---------------------------------------------------------------------------

def fetch_url_registry_for_phone(canonical_url_id: int) -> list[dict]:
    """
    Fetches ALL url_registry rows for the same (brand, model_name) as the
    given canonical_url_id.

    Used to count total URLs registered and check scrape completion status.
    The canonical_url_id acts as the anchor — its brand + model_name
    determine the phone identity.

    Returns:
        List of url_registry row dicts. Never empty (canonical must exist).

    Raises:
        ValueError: If the canonical_url_id does not exist in url_registry.
    """
    # First resolve the brand + model_name from the canonical row
    anchor = (
        get_client()
        .schema("pipeline")
        .table("url_registry")
        .select("brand, model_name")
        .eq("url_id", canonical_url_id)
        .execute()
    )
    if not anchor.data:
        raise ValueError(
            f"fetch_url_registry_for_phone: canonical_url_id={canonical_url_id} "
            f"not found in pipeline.url_registry."
        )

    brand = anchor.data[0]["brand"]
    model_name = anchor.data[0]["model_name"]

    # Fetch all rows for this phone
    result = (
        get_client()
        .schema("pipeline")
        .table("url_registry")
        .select("url_id, site_name, status")
        .eq("brand", brand)
        .eq("model_name", model_name)
        .execute()
    )
    return result.data or []


def fetch_youtube_search_log(canonical_url_id: int) -> list[dict]:
    """
    Fetches all youtube_search_log rows for this canonical_url_id.

    A phone is considered youtube_search_done if at least one row exists
    with search_status IN ('success_zero', 'success_with_results').

    Returns:
        List of search_log row dicts. Empty list if no search has run.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("youtube_search_log")
        .select("search_log_id, search_status, searched_at")
        .eq("url_registry_id", canonical_url_id)
        .execute()
    )
    return result.data or []


def fetch_transcript_availability(canonical_url_id: int) -> list[dict]:
    """
    Checks youtube_video_url_registry for fetched transcripts linked to
    this canonical_url_id.

    A transcript is considered available if at least one video_registry row
    exists with status = 'fetched_raw'.

    Returns:
        List of video registry row dicts. Empty means no transcript fetched.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("youtube_video_url_registry")
        .select("video_registry_id, status")
        .eq("url_registry_id", canonical_url_id)
        .eq("status", "fetched_raw")
        .execute()
    )
    return result.data or []


# ---------------------------------------------------------------------------
# Task 1.1 — pre_extraction_validation writes
# ---------------------------------------------------------------------------

def insert_pre_extraction_validation(payload: dict) -> dict:
    """
    Inserts a new row into pipeline.pre_extraction_validation.

    Expected payload keys (all required):
        canonical_url_id      int
        validated_by          str
        total_urls_registered int
        urls_not_attempted    int
        urls_scraped_success  int
        official_url_scraped  bool | None
        gsmarena_scraped      bool | None
        youtube_search_done   bool
        transcript_available  bool
        can_proceed           bool
        blocking_reasons      list[dict]   — [{code, message, url_ids?}]
        warnings              list[dict]   — [{code, message}]
        unscraped_url_ids     list[int]    — url_ids of not-attempted URLs

    Returns:
        Newly inserted row as dict (includes validation_id, validated_at).

    Raises:
        RuntimeError: If insert returns no data.
    """
    # Pass Python lists directly — supabase-py serialises the full payload
    # to JSON itself. Calling json.dumps() here would cause double-serialisation,
    # storing a JSON string in the JSONB column instead of a JSONB array.
    db_payload = {
        "canonical_url_id":      payload["canonical_url_id"],
        "validated_by":          payload["validated_by"],
        "total_urls_registered": payload["total_urls_registered"],
        "urls_not_attempted":    payload["urls_not_attempted"],
        "urls_scraped_success":  payload["urls_scraped_success"],
        "official_url_scraped":  payload["official_url_scraped"],
        "gsmarena_scraped":      payload["gsmarena_scraped"],
        "youtube_search_done":   payload["youtube_search_done"],
        "transcript_available":  payload["transcript_available"],
        "can_proceed":           payload["can_proceed"],
        "blocking_reasons":      payload["blocking_reasons"],
        "warnings":              payload["warnings"],
        # 'unscraped_url_ids' is in-memory only � no DB column, excluded from insert.
    }

    result = (
        get_client()
        .schema("pipeline")
        .table("pre_extraction_validation")
        .insert(db_payload)
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            f"insert_pre_extraction_validation: no data returned for "
            f"canonical_url_id={payload['canonical_url_id']}."
        )
    return result.data[0]


def fetch_latest_validation(canonical_url_id: int) -> dict | None:
    """
    Fetches the most recent pre_extraction_validation row for this phone.
    Returns None if no validation has been run yet.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("pre_extraction_validation")
        .select("*")
        .eq("canonical_url_id", canonical_url_id)
        .order("validated_at", desc=True)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    return rows[0] if rows else None


def fetch_brand_model_by_url_registry_id(url_registry_id: int) -> tuple[str, str] | None:
    """
    Returns (brand, model_name) for the given url_registry_id, or None if not found.

    Used by run_normalisation() to pass brand + model_name into the pre-normalizer
    enrichment pass (Section 2) without the caller needing to supply them separately.

    Synchronous. Wrap with asyncio.to_thread() from async callers.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("url_registry")
        .select("brand, model_name")
        .eq("url_id", url_registry_id)
        .limit(1)
        .execute()
    )
    if not result.data:
        return None
    row = result.data[0]
    return row["brand"], row["model_name"]


def fetch_gap_enrichment_policies() -> dict[str, dict]:
    """
    Section 4 — Loads all rows from pipeline.gap_enrichment_policy.

    Returns a dict keyed by field_path for O(1) lookup during gap analysis.
    Each value is the full row dict:
      {policy, min_tier_id, threshold_count, threshold_op,
       comma_split, preferred_site_hint, notes, ...}

    Called once at app startup (via build_lookup_cache / gap analyzer warm-up).
    Synchronous — wrap with asyncio.to_thread() from async callers.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("gap_enrichment_policy")
        .select(
            "field_path, policy, min_tier_id, threshold_count, "
            "threshold_op, comma_split, preferred_site_hint, notes"
        )
        .execute()
    )
    rows = result.data or []
    return {row["field_path"]: row for row in rows}


def fetch_price_tiers() -> list[dict]:
    """
    Section 4 — Loads all rows from pipeline.lookup_price_tiers ordered by sort_order.

    Each row: {tier_id, tier_name, min_inr, max_inr, enrichment_cap, sort_order}
    Used to build _TIER_CACHE in gap_analyzer.py.

    Synchronous — wrap with asyncio.to_thread() from async callers.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("lookup_price_tiers")
        .select("tier_id, tier_name, min_inr, max_inr, enrichment_cap, sort_order")
        .order("sort_order")
        .execute()
    )
    return result.data or []




# ---------------------------------------------------------------------------
# Phase 2 — Run A source and transcript fetching
# ---------------------------------------------------------------------------

def fetch_raw_source_rows(raw_source_ids: list[int]) -> list[dict]:
    """
    Fetches raw_scraped_data rows for the given list of raw_ids.

    Returns:
        List of rows with: raw_id, site_name, markdown_path.
        Order is NOT guaranteed — callers sort by get_concat_order(site_name).

    Raises:
        RuntimeError: If any raw_id from the list is missing in the DB
                      (data integrity violation — run was registered with invalid IDs).
    """
    if not raw_source_ids:
        return []

    result = (
        get_client()
        .schema("pipeline")
        .table("raw_scraped_data")
        .select("raw_id, markdown_path, url_registry_id, url_registry(site_name)")
        .in_("raw_id", raw_source_ids)
        .execute()
    )

    rows = result.data or []

    found_ids = {r["raw_id"] for r in rows}
    missing = set(raw_source_ids) - found_ids
    if missing:
        raise RuntimeError(
            f"fetch_raw_source_rows: raw_ids not found in pipeline.raw_scraped_data: "
            f"{sorted(missing)}. Possible data integrity issue."
        )

    # Flatten the nested url_registry join into a top-level site_name key
    # so callers can use r["site_name"] as before.
    for r in rows:
        nested = r.pop("url_registry", None) or {}
        r["site_name"] = nested.get("site_name", "unknown")

    return rows


def fetch_transcript_row(raw_transcript_id: int) -> dict:
    """
    Fetches the youtube_raw_transcript_data row for the given raw_transcript_id.

    Returns the full row dict (includes translation_status, processed_transcript_path,
    translated_transcript_path for path selection logic).

    Raises:
        ValueError: If the raw_transcript_id does not exist.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("youtube_raw_transcript_data")
        .select(
            "raw_transcript_id, processed_transcript_path, "
            "translated_transcript_path, translation_status, language_code"
        )
        .eq("raw_transcript_id", raw_transcript_id)
        .execute()
    )

    if not result.data:
        raise ValueError(
            f"fetch_transcript_row: raw_transcript_id={raw_transcript_id} "
            f"not found in pipeline.youtube_raw_transcript_data."
        )
    return result.data[0]


# ---------------------------------------------------------------------------
# Phase 2 — spec_extraction_runs writes
# ---------------------------------------------------------------------------

def insert_extraction_run(payload: dict) -> int:
    """
    Inserts a new row into pipeline.spec_extraction_runs with status='running'.

    Expected payload keys:
        url_registry_id           int
        raw_source_ids            list[int]   — stored as JSONB
        raw_transcript_ids        list[int]   — JSONB array (v5: replaces single FK)
        model_used                str
        extraction_schema_version str         — default 'v2'
        status                    str         — 'running'

    All raw_transcript_ids are validated to belong to the given url_registry_id
    before insert. Any IDs that do not match the phone are silently removed
    and logged. This prevents cross-phone transcript contamination.

    Returns:
        extraction_run_id (int) of the newly inserted row.

    Raises:
        RuntimeError: If insert returns no data.
    """
    url_registry_id: int = payload["url_registry_id"]
    raw_transcript_ids: list[int] = payload.get("raw_transcript_ids") or []

    # Validate transcript IDs belong to this phone
    if raw_transcript_ids:
        try:
            transcript_result = (
                get_client()
                .schema("pipeline")
                .table("youtube_raw_transcript_data")
                .select("raw_transcript_id, video_registry_id")
                .in_("raw_transcript_id", raw_transcript_ids)
                .execute()
            )
            transcript_rows = transcript_result.data or []
            tid_to_vid = {
                r["raw_transcript_id"]: r["video_registry_id"]
                for r in transcript_rows
            }

            valid_ids: list[int] = []
            if tid_to_vid:
                vr_result = (
                    get_client()
                    .schema("pipeline")
                    .table("youtube_video_url_registry")
                    .select("video_registry_id, url_registry_id")
                    .in_("video_registry_id", list(tid_to_vid.values()))
                    .execute()
                )
                vid_to_owner = {
                    r["video_registry_id"]: r["url_registry_id"]
                    for r in (vr_result.data or [])
                }
                for tid, vid in tid_to_vid.items():
                    if vid_to_owner.get(vid) == url_registry_id:
                        valid_ids.append(tid)
                    else:
                        _logger.warning(
                            "insert_extraction_run: raw_transcript_id=%d does not "
                            "belong to url_registry_id=%d — excluded.",
                            tid, url_registry_id,
                        )
            raw_transcript_ids = valid_ids
        except Exception as exc:
            _logger.warning(
                "insert_extraction_run: secondary transcript ownership check failed (%s). "
                "Proceeding with original IDs — primary validation already passed.",
                exc,
            )
            # DO NOT clear raw_transcript_ids
            # Keep original IDs

    db_payload = {
        "url_registry_id":           url_registry_id,
        "raw_source_ids":            payload["raw_source_ids"],
        "raw_transcript_ids":        raw_transcript_ids,
        "model_used":                payload["model_used"],
        "extraction_schema_version": payload.get("extraction_schema_version", "v2"),
        "status":                    payload.get("status", "running"),
    }

    result = (
        get_client()
        .schema("pipeline")
        .table("spec_extraction_runs")
        .insert(db_payload)
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            f"insert_extraction_run: no data returned for "
            f"url_registry_id={url_registry_id}."
        )
    return result.data[0]["extraction_run_id"]


def update_extraction_run(run_id: int, updates: dict) -> None:
    """
    Updates a spec_extraction_runs row by extraction_run_id.

    Common update patterns:
        On success: {"status": "completed", "finished_at": "now()",
                     "input_token_count": N, "output_token_count": N}
        On failure: {"status": "failed", "error_message": str, "finished_at": "now()"}

    Note: "finished_at": "now()" is a special sentinel resolved here to an
    ISO timestamp string. A14 fix: datetime import moved to module top level.

    Raises:
        RuntimeError: If the run_id is not found (no rows updated).
    """
    # Replace "now()" sentinel with actual UTC timestamp string
    safe_updates = {}
    for k, v in updates.items():
        if v == "now()":
            safe_updates[k] = _now_iso()
        else:
            safe_updates[k] = v

    result = (
        get_client()
        .schema("pipeline")
        .table("spec_extraction_runs")
        .update(safe_updates)
        .eq("extraction_run_id", run_id)
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            f"update_extraction_run: run_id={run_id} not found or no update applied."
        )


def insert_extraction_output(payload: dict) -> int:
    """
    Inserts a new row into pipeline.spec_extraction_output.

    Expected payload keys:
        extraction_run_id  int
        url_registry_id    int
        partial_json       dict   — the LLM-extracted spec structure
        evidence_json      dict   — field_path → EvidenceEntry instance or plain dict
        null_field_count   int
        filled_field_count int

    A1 fix: evidence_json values may be live EvidenceEntry Pydantic model instances.
    supabase-py serialises the payload with standard json.dumps, which cannot handle
    Pydantic models (raises TypeError: Object of type EvidenceEntry is not JSON
    serializable). We normalise every value to a plain dict via model_dump() before
    passing the payload to Supabase.

    Returns:
        output_id (int) of the newly inserted row.

    Raises:
        RuntimeError: If insert returns no data.
    """
    # A1 — Normalise evidence_json: Pydantic EvidenceEntry → plain dict
    raw_evidence = payload["evidence_json"]
    serialisable_evidence = {
        k: (v.model_dump() if hasattr(v, "model_dump") else v)
        for k, v in raw_evidence.items()
    }

    db_payload = {
        "extraction_run_id":  payload["extraction_run_id"],
        "url_registry_id":    payload["url_registry_id"],
        "partial_json":       payload["partial_json"],
        "evidence_json":      serialisable_evidence,
        "null_field_count":   payload["null_field_count"],
        "filled_field_count": payload["filled_field_count"],
    }

    result = (
        get_client()
        .schema("pipeline")
        .table("spec_extraction_output")
        .insert(db_payload)
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            f"insert_extraction_output: no data returned for "
            f"extraction_run_id={payload['extraction_run_id']}."
        )
    return result.data[0]["output_id"]


# ---------------------------------------------------------------------------
# Phase 3 — Run B experience run writes
# ---------------------------------------------------------------------------

def fetch_experience_category_map() -> dict[str, int]:
    """
    Returns {category_name: category_id} from pipeline.lookup_experience_categories.

    B1 fix: phone_experiences.category_id is a FK to this table.
    The orchestrator calls this once before bulk insert, then maps each
    LLM category string → integer ID. Avoids per-row DB queries.

    Returns:
        {category_name: category_id} e.g. {"Overall": 1, "Thermal": 2, ...}

    Raises:
        RuntimeError: If the lookup table is empty (misconfigured DB).
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("lookup_experience_categories")
        .select("category_id, category_name")
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise RuntimeError(
            "fetch_experience_category_map: pipeline.lookup_experience_categories "
            "is empty. Seed the table before running Run B."
        )
    return {r["category_name"]: r["category_id"] for r in rows}


def insert_experience_run(payload: dict) -> int:
    """
    Inserts a new row into pipeline.experience_extraction_runs with status='running'.

    Expected payload keys:
        url_registry_id           int
        raw_transcript_id         int
        model_used                str
        extraction_schema_version str  — default 'v1'
        status                    str  — 'running'

    Returns:
        exp_run_id (int) of the newly inserted row.

    Raises:
        RuntimeError: If insert returns no data.
    """
    db_payload = {
        "url_registry_id":           payload["url_registry_id"],
        "raw_transcript_id":         payload["raw_transcript_id"],
        "model_used":                payload["model_used"],
        "extraction_schema_version": payload.get("extraction_schema_version", "v1"),
        "status":                    payload.get("status", "running"),
    }

    result = (
        get_client()
        .schema("pipeline")
        .table("experience_extraction_runs")
        .insert(db_payload)
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            f"insert_experience_run: no data returned for "
            f"url_registry_id={payload['url_registry_id']}."
        )
    return result.data[0]["exp_run_id"]


def update_experience_run(run_id: int, updates: dict) -> None:
    """
    Updates an experience_extraction_runs row by exp_run_id.

    Common update patterns:
        On success: {"status": "completed", "finished_at": "now()",
                     "experiences_extracted": N,
                     "input_token_count": N, "output_token_count": N}
        On failure: {"status": "failed", "error_message": str, "finished_at": "now()"}

    Raises:
        RuntimeError: If the run_id is not found (no rows updated).
    """
    safe_updates = {}
    for k, v in updates.items():
        if v == "now()":
            safe_updates[k] = _now_iso()
        else:
            safe_updates[k] = v

    result = (
        get_client()
        .schema("pipeline")
        .table("experience_extraction_runs")
        .update(safe_updates)
        .eq("exp_run_id", run_id)
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            f"update_experience_run: run_id={run_id} not found or no update applied."
        )


def bulk_insert_phone_experiences(rows: list[dict]) -> int:
    """
    Plain INSERT. Idempotency is handled by the supersede model —
    supersede_experiences() marks all existing active rows as is_superseded=TRUE
    before each Run B batch, and every run gets a fresh exp_run_id. Exact-text
    deduplication within active rows is enforced by the partial unique index
    idx_phone_exp_unique_active_text.

    Expected keys per row:
        url_registry_id  int
        exp_run_id       int
        experience_text  str
        sentiment        str   — "Positive"|"Negative"|"Neutral"|"Mixed"
        evidence_quote   str
        category_id:     int
        confidence       float
        raw_transcript_id int

    Returns:
        Number of rows inserted.

    Raises:
        RuntimeError: If insert returns no data.
    """
    if not rows:
        return 0

    result = (
        get_client()
        .schema("pipeline")
        .table("phone_experiences")
        .insert(rows)
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            f"bulk_insert_phone_experiences: insert returned no data for "
            f"exp_run_id={rows[0].get('exp_run_id')}."
        )
    if len(result.data) != len(rows):
        raise RuntimeError(
            f"bulk_insert_phone_experiences: expected {len(rows)} rows inserted, "
            f"got {len(result.data)}. Possible partial failure or constraint violation."
        )
    return len(result.data)



# ---------------------------------------------------------------------------
# Phase 4 — Normalisation DB layer
# ---------------------------------------------------------------------------

def fetch_run_a_outputs_for_phone(url_registry_id: int, limit: int = 20) -> list[dict]:
    """
    Returns recent spec_extraction_output rows for a phone, ordered newest first.
    Used by the Normalizer UI OutputSelector to let the admin pick which
    Run A output to normalise.

    Returns columns: output_id, extraction_run_id, null_field_count,
    filled_field_count, created_at (from spec_extraction_runs.finished_at).
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("spec_extraction_output")
        .select(
            "output_id, extraction_run_id, null_field_count, filled_field_count, "
            "spec_extraction_runs(status, finished_at, extraction_schema_version)"
        )
        .eq("url_registry_id", url_registry_id)
        .order("output_id", desc=True)
        .limit(limit)
        .execute()
    )
    rows = result.data or []
    # Flatten the nested join
    for row in rows:
        run = row.pop("spec_extraction_runs", None) or {}
        row["run_status"]            = run.get("status")
        row["finished_at"]           = run.get("finished_at")
        row["schema_version"]        = run.get("extraction_schema_version")
    return rows


def fetch_spec_extraction_output(output_id: int) -> dict:
    """
    Fetches a spec_extraction_output row by output_id.

    Raises:
        ValueError: If output_id not found.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("spec_extraction_output")
        .select(
            "output_id, url_registry_id, extraction_run_id, "
            "partial_json, null_field_count, filled_field_count"
        )
        .eq("output_id", output_id)
        .execute()
    )
    if not result.data:
        raise ValueError(
            f"fetch_spec_extraction_output: output_id={output_id} not found."
        )
    return result.data[0]


def insert_normalisation_run(payload: dict) -> int:
    """
    Inserts a new row into pipeline.normalization_runs with status='running'.

    Expected payload keys:
        output_id        int
        url_registry_id  int
        status           str  — 'running'

    Returns:
        normalization_run_id (int) of the newly inserted row.

    Raises:
        RuntimeError: If insert returns no data.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("normalization_runs")          # N1: American spelling matches SQL schema
        .insert(payload)
        .execute()
    )
    if not result.data:
        raise RuntimeError(
            f"insert_normalisation_run: no data returned for "
            f"output_id={payload.get('output_id')}."
        )
    return result.data[0]["normalization_run_id"]  # N2: correct PK column name


def update_normalisation_run(run_id: int, updates: dict) -> None:
    """
    Updates a normalization_runs row by normalization_run_id.

    Common update patterns:
        On success: {"status": "completed", "finished_at": "now()",
                     "issue_count": N, "issues_found": [...]}
        On failure: {"status": "failed", "error_message": str, "finished_at": "now()"}
        Note: staging_count is omitted — column removed from schema. Do NOT include it.

    Raises:
        RuntimeError: If run_id not found.
    """
    safe_updates = {}
    for k, v in updates.items():
        if v == "now()":
            safe_updates[k] = _now_iso()
        else:
            safe_updates[k] = v

    result = (
        get_client()
        .schema("pipeline")
        .table("normalization_runs")          # N1: American spelling matches SQL schema
        .update(safe_updates)
        .eq("normalization_run_id", run_id)   # N2: correct PK column name
        .execute()
    )
    if not result.data:
        raise RuntimeError(
            f"update_normalisation_run: normalization_run_id={run_id} not found."
        )


def upsert_normalized_spec(payload: dict) -> int:
    """
    Inserts or updates a row in pipeline.normalized_spec_json.
    Uses normalization_run_id as the unique conflict key.

    Expected payload keys (matching actual SQL schema columns):
        normalization_run_id  int
        url_registry_id       int
        normalized_json       dict   — N4: was normalized_spec_json
        remaining_null_count  int    — N4: was null_field_count
        ready_for_enrichment  bool
        ready_for_commit      bool

    Returns:
        normalized_id (int) of the upserted row.

    Raises:
        RuntimeError: If upsert returns no data.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("normalized_spec_json")                    # N3: correct table name
        .upsert(payload, on_conflict="normalization_run_id")  # N5: correct conflict col
        .execute()
    )
    if not result.data:
        raise RuntimeError(
            f"upsert_normalized_spec: no data returned for "
            f"normalization_run_id={payload.get('normalization_run_id')}."
        )
    return result.data[0]["normalized_id"]


def insert_lookup_value_staging(payload: dict) -> None:
    """
    Inserts a not-found lookup value into pipeline.lookup_value_staging
    for human review. Silently ignores conflicts (same value already staged).

    Expected payload keys (matching SQL schema columns):
        extracted_value      str   — N6: was raw_value
        target_lookup_table  str   — N6: was table_path
        field_path           str   — "variants[0].ram_type"
        url_registry_id      int
        source_stage         str   — N6: required NOT NULL; e.g. "normalization"

    NOTE: The SQL schema sets status DEFAULT 'pending_review'. Do NOT include
    'status' in the payload — let the DB default handle it. This function does
    NOT set status explicitly; fetch_pending_staging_count reads status='pending_review'
    which is always the default for newly inserted rows.

    If the (extracted_value, target_lookup_table) pair already exists, does nothing.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("lookup_value_staging")
        .upsert(
            payload,
            on_conflict="extracted_value,target_lookup_table",  # N6: correct columns
            ignore_duplicates=True,
        )
        .execute()
    )
    # No error on conflict — staging is idempotent by design

# ---------------------------------------------------------------------------
# Phase 5 — Gap Analysis DB layer
# ---------------------------------------------------------------------------

def fetch_normalized_spec(normalized_id: int) -> dict:
    """
    Fetches a normalized_spec_json row by normalized_id.

    Returns the full row including normalized_json, url_registry_id,
    remaining_null_count, ready_for_enrichment, and Section 4/5 fields
    (tier_id, new_chipset_detected, chipset_name_enriched).

    Raises:
        ValueError: If normalized_id not found.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("normalized_spec_json")
        .select(
            "normalized_id, normalization_run_id, url_registry_id, "
            "normalized_json, remaining_null_count, ready_for_enrichment, "
            "ready_for_commit, tier_id, new_chipset_detected, chipset_name_enriched"
        )
        .eq("normalized_id", normalized_id)
        .execute()
    )
    if not result.data:
        raise ValueError(
            f"fetch_normalized_spec: normalized_id={normalized_id} not found "
            f"in pipeline.normalized_spec_json."
        )
    return result.data[0]


def fetch_url_registry_status(url_registry_id: int) -> dict:
    """
    Fetches status and site_name for a url_registry row.
    Used by gap_analyzer to determine if the phone is already stored in mainDB
    (status='stored_mainDB') which activates Type B gap candidate creation.

    Returns:
        dict with at least {status, site_name}.

    Raises:
        ValueError: If url_registry_id not found.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("url_registry")
        .select("url_id, site_name, status, brand, model_name")
        .eq("url_id", url_registry_id)
        .execute()
    )
    if not result.data:
        raise ValueError(
            f"fetch_url_registry_status: url_registry_id={url_registry_id} not found."
        )
    return result.data[0]


def insert_missing_field_log(payload: dict) -> int | None:
    """
    Inserts a row into pipeline.missing_fields_log.
    Idempotent: ON CONFLICT (normalized_id, field_path) DO UPDATE SET
    field_path=EXCLUDED.field_path RETURNING missing_field_id.
    This ensures the existing ID is always returned on re-runs (G7).

    Expected payload keys (must match SQL schema exactly — G1, G4):
        normalized_id           int    — FK to pipeline.normalized_spec_json
        url_registry_id         int    — FK to pipeline.url_registry
        field_path              str    — concrete path e.g. "displays[0].panel_type"
        missing_type            str    — "type_a" | "type_b"
        priority                str    — "high" | "medium" | "low" | "skip"
        preferred_site_hint     str | None
        query_template_override str | None  — E3 fix: column exists in schema, default None
        is_flag_only            bool   — Section 4: True for flag_only policy rows (never enriched)

    Returns:
        missing_field_id (int) always — whether newly inserted or already existing.
        None only on unexpected DB error (logged by caller).
    """

    result = (
        get_client()
        .schema("pipeline")
        .table("missing_fields_log")
        .upsert(
            payload,
            # G4: correct unique constraint columns from SQL schema
            on_conflict="normalized_id,field_path",
            # G7: DO UPDATE so RETURNING always gives the row ID
            ignore_duplicates=False,
        )
        .execute()
    )
    if not result.data:
        return None
    return result.data[0]["missing_field_id"]


def insert_type_b_gap_candidates(candidates: list[dict]) -> None:
    """
    Bulk inserts rows into pipeline.type_b_gap_candidates.
    Idempotent: ON CONFLICT (missing_field_id, lookup_row_id) DO NOTHING.

    Each dict in candidates must match SQL schema columns (G2):
        missing_field_id  int    — FK to pipeline.missing_fields_log
        url_registry_id   int    — FK to pipeline.url_registry
        candidate_value   str    — canonical string value for the lookup PK
        lookup_table      str    — "schema.table.column" path
        lookup_row_id     int | None  — PK in lookup table
        inclusion_reason  str    — why this candidate was included

    Raises:
        RuntimeError: On any non-conflict DB error.
    """
    if not candidates:
        return
    (
        get_client()
        .schema("pipeline")
        .table("type_b_gap_candidates")
        .upsert(
            candidates,
            # G2: correct unique constraint columns from SQL schema
            on_conflict="missing_field_id,lookup_row_id",
            ignore_duplicates=True,
        )
        .execute()
    )


def fetch_junction_table_existing_pks(
    url_registry_id: int,
    generic_path: str,
    table_path: str,
) -> list[dict]:
    """
    G3 FIX: Returns committed lookup rows for a phone's junction fields.

    Junction tables in mobile_specs do NOT link to url_registry_id directly.
    They link through intermediate entity IDs (e.g. display_id, lens spec ID).
    Resolving url_registry_id → model_id → display_id → feature_ids requires
    multi-step joins that are specific to each junction table structure.

    This level of join complexity belongs in the commit orchestrator (Phase 8),
    which knows the full committed entity ID map for the phone. Attempting it
    here speculatively — before the phone is committed — produces stale or
    wrong candidates.

    CURRENT BEHAVIOUR (safe, correct):
        Returns [] always. Type B gap candidate creation is a no-op for now.
        Missing_fields_log rows are still created with missing_type='type_b',
        giving Phase 6 enrichment the correct signal to do targeted search.
        Candidate pre-population is deferred to Phase 8.

    Args:
        url_registry_id: The phone's url_registry primary key.
        generic_path:    Wildcard path e.g. "displays[*].display_features"
        table_path:      "schema.table.column" from JUNCTION_TABLE_FIELDS

    Returns:
        [] — always, pending Phase 8 implementation.
    """
    # Deferred: junction → intermediate entity → model_id join chain
    # not implementable here without Phase 8 entity ID map.
    return []


# ===========================================================================
# Phase 6 — Enrichment Repository Functions
# ===========================================================================


def insert_enrichment_run(payload: dict) -> int:
    """
    Creates a new row in pipeline.enrichment_runs.

    Expected payload keys (match SQL schema):
        normalized_id    int
        url_registry_id  int
        search_provider  str  — default 'gemini_grounding'
        status           str  — 'running'

    Returns:
        enrichment_run_id (int)
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("enrichment_runs")
        .insert(payload)
        .execute()
    )
    if not result.data:  # E2 fix
        raise RuntimeError(
            f"insert_enrichment_run: no data returned for "
            f"normalized_id={payload.get('normalized_id')}. "
            f"Check enrichment_runs schema and constraints."
        )
    return result.data[0]["enrichment_run_id"]


def update_enrichment_run(run_id: int, payload: dict) -> None:
    """
    Updates an enrichment_runs row with final summary data.

    Expected payload keys (all optional — pass only what changed):
        status             str    — 'completed' | 'partially_completed' | 'failed'
        fields_targeted    int
        fields_resolved    int
        total_api_cost_inr float
        finished_at        str    — ISO timestamp (defaults to DB NOW() if omitted)
        error_message      str | None
    """
    if "finished_at" not in payload:
        payload = {
            **payload,
            "finished_at": _now_iso(),
        }
    (
        get_client()
        .schema("pipeline")
        .table("enrichment_runs")
        .update(payload)
        .eq("enrichment_run_id", run_id)
        .execute()
    )


def fetch_gap_fields_for_enrichment(normalized_id: int) -> list[dict]:
    """
    Returns all missing_fields_log rows for a phone where enrichment
    has not yet been attempted (enrichment_attempted=FALSE).

    Section 5.1: Filters WHERE is_flag_only = FALSE so that flag_only policy
    rows (network bands etc.) are never sent to the enrichment orchestrator.

    Returns columns needed by the enrichment orchestrator:
        missing_field_id, field_path, missing_type, priority,
        preferred_site_hint, query_template_override, is_flag_only

    Ordered by priority: high first, then medium.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("missing_fields_log")
        .select(
            "missing_field_id, field_path, missing_type, priority, "
            "preferred_site_hint, query_template_override, is_flag_only"
        )
        .eq("normalized_id", normalized_id)
        .eq("enrichment_attempted", False)
        .eq("is_flag_only", False)          # Section 5.1: exclude flag-only rows
        .neq("priority", "skip")            # L7.2: exclude skip-priority fields from enrichment
        .execute()
    )
    rows = result.data or []
    # Sort: high priority first, then medium, then everything else
    _priority_order = {"high": 0, "medium": 1}
    return sorted(rows, key=lambda r: _priority_order.get(r.get("priority", "medium"), 2))



def insert_enrichment_search_query(payload: dict) -> int:
    """
    Logs one Gemini grounded API call in pipeline.enrichment_search_queries.

    Expected payload keys:
        enrichment_run_id  int
        missing_field_id   int
        query_text         str
        query_template_used str | None
        grounding_used     bool
        api_cost_inr       float  — 0.00 before call; updated after via update_enrichment_search_query

    Returns:
        query_id (int)
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("enrichment_search_queries")
        .insert(payload)
        .execute()
    )
    if not result.data:  # E2 fix
        raise RuntimeError(
            f"insert_enrichment_search_query: no data returned for "
            f"missing_field_id={payload.get('missing_field_id')}. "
            f"Check enrichment_search_queries schema and constraints."
        )
    return result.data[0]["query_id"]


def update_enrichment_search_query(query_id: int, payload: dict) -> None:
    """
    Updates an enrichment_search_queries row after the API call completes.

    Expected payload keys:
        http_status    int | None
        api_cost_inr   float
        error_message  str | None
    """
    (
        get_client()
        .schema("pipeline")
        .table("enrichment_search_queries")
        .update(payload)
        .eq("query_id", query_id)
        .execute()
    )


def insert_enrichment_candidate(payload: dict) -> int:
    """
    Inserts one row into pipeline.enrichment_field_candidates.

    Expected payload keys (match SQL schema):
        query_id          int | None   — None for chipset dedup copies
        missing_field_id  int
        enrichment_run_id int
        field_path        str
        extracted_value   any          — stored as JSONB; None → null
        raw_confidence    float
        confidence        float
        evidence_text     str | None
        source_url        str | None
        source_domain     str | None
        source_tier       str          — 'oem_official' | 'trusted_aggregator' | 'tech_media' | 'forum' | 'unknown'
        is_selected       bool

    Returns:
        candidate_id (int)
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("enrichment_field_candidates")
        .insert(payload)
        .execute()
    )
    if not result.data:  # E2 fix
        raise RuntimeError(
            f"insert_enrichment_candidate: no data returned for "
            f"missing_field_id={payload.get('missing_field_id')}, "
            f"field_path={payload.get('field_path')!r}. "
            f"Check enrichment_field_candidates schema and constraints."
        )
    return result.data[0]["candidate_id"]


def update_missing_field_log_attempted(
    missing_field_id: int,
    enrichment_succeeded: bool,
) -> None:
    """
    Marks a missing_fields_log row as attempted.

    Sets:
        enrichment_attempted  = TRUE
        enrichment_succeeded  = <enrichment_succeeded>
    """
    (
        get_client()
        .schema("pipeline")
        .table("missing_fields_log")
        .update({
            "enrichment_attempted": True,
            "enrichment_succeeded": enrichment_succeeded,
        })
        .eq("missing_field_id", missing_field_id)
        .execute()
    )


def fetch_chipset_enrichment_copy(
    chipset_name: str,
    field_path: str,
) -> dict | None:
    """
    Chipset deduplication query for Task 6.2.

    Finds an existing is_selected=TRUE enrichment_field_candidates row
    for the same field_path whose extraction came from a phone sharing
    the same chipset_name.

    E6 fix: uses PostgREST JSONB containment filter (.contains()) to push
    chipset_name comparison to the DB — avoids pulling full normalized_json
    blobs into Python for comparison.

    Returns:
        First matching candidate dict if found, else None.
        Useful keys: extracted_value, raw_confidence, evidence_text,
                     source_url, source_domain, source_tier
    """
    # Step 1: find normalized_ids of phones that have this chipset_name
    # Uses PostgREST JSONB containment — DB does the filter, not Python
    nsj_result = (
        get_client()
        .schema("pipeline")
        .table("normalized_spec_json")
        .select("normalized_id")
        .contains("normalized_json", {"chipset": {"chipset_name": chipset_name}})
        .execute()
    )
    if not nsj_result.data:
        return None

    matching_ids = [r["normalized_id"] for r in nsj_result.data]

    # Step 2: find successfully enriched missing_fields_log rows for these phones
    mfl_result = (
        get_client()
        .schema("pipeline")
        .table("missing_fields_log")
        .select("missing_field_id")
        .eq("field_path", field_path)
        .eq("enrichment_succeeded", True)
        .in_("normalized_id", matching_ids)
        .limit(10)
        .execute()
    )
    if not mfl_result.data:
        return None

    matched_field_ids = [r["missing_field_id"] for r in mfl_result.data]

    # Step 3: find the confirmed candidate
    cand_result = (
        get_client()
        .schema("pipeline")
        .table("enrichment_field_candidates")
        .select(
            "candidate_id, extracted_value, raw_confidence, evidence_text, "
            "source_url, source_domain, source_tier"
        )
        .in_("missing_field_id", matched_field_ids)
        .eq("is_selected", True)
        .limit(1)
        .execute()
    )
    if not cand_result.data:
        return None

    return cand_result.data[0]


def fetch_best_existing_candidate_confidence(
    missing_field_id: int,
) -> float | None:
    """
    Returns the highest adjusted confidence of any existing candidate for
    this missing_field_id. Used by _should_auto_select() for the delta check.

    Returns None if no existing candidates exist.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("enrichment_field_candidates")
        .select("confidence")
        .eq("missing_field_id", missing_field_id)
        .order("confidence", desc=True)
        .limit(1)
        .execute()
    )
    if not result.data:
        return None
    return float(result.data[0]["confidence"])


def deselect_all_candidates_for_field(missing_field_id: int) -> None:
    """
    E1 fix — sets is_selected=FALSE for ALL existing enrichment_field_candidates
    rows for this missing_field_id.

    Called immediately BEFORE inserting the new winning candidate when
    is_selected=True, to enforce the invariant: exactly ONE is_selected=TRUE
    row per field. Without this, Rule 2 (delta >= 0.20) would create multiple
    selected rows and corrupt Phase 7 conflict resolution.
    """
    (
        get_client()
        .schema("pipeline")
        .table("enrichment_field_candidates")
        .update({"is_selected": False})
        .eq("missing_field_id", missing_field_id)
        .eq("is_selected", True)  # Only touch rows that are currently selected
        .execute()
    )


# ===========================================================================
# Phase 7 — Conflict Resolution Repository Functions
# ===========================================================================


def fetch_latest_spec_output_for_phone(url_registry_id: int) -> dict | None:
    """
    Fetches the most recent spec_extraction_output row for a phone,
    identified by url_registry_id.

    Returns the row including evidence_json — used by Phase 7 conflict resolver
    to infer Run A per-field source tier.

    Returns None if no extraction output exists for this phone.

    NOTE: Distinct from fetch_spec_extraction_output(output_id) which fetches
    by PK for Phase 4 normalisation. Both functions exist and must not be renamed.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("spec_extraction_output")
        .select(
            "output_id, url_registry_id, partial_json, evidence_json, "
            "null_field_count, filled_field_count"
        )
        .eq("url_registry_id", url_registry_id)
        .order("output_id", desc=True)
        .limit(1)
        .execute()
    )
    if not result.data:
        return None
    return result.data[0]


def fetch_selected_enrichment_candidates(enrichment_run_id: int) -> list[dict]:
    """
    Returns all is_selected=TRUE enrichment_field_candidates for an enrichment run.

    Columns returned:
        candidate_id, missing_field_id, field_path, extracted_value,
        raw_confidence, confidence, source_tier, source_url, evidence_text
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("enrichment_field_candidates")
        .select(
            "candidate_id, missing_field_id, field_path, extracted_value, "
            "raw_confidence, confidence, source_tier, source_url, evidence_text"
        )
        .eq("enrichment_run_id", enrichment_run_id)
        .eq("is_selected", True)
        .execute()
    )
    return result.data or []


def insert_merge_conflict(payload: dict) -> int:
    """
    Inserts one row into pipeline.merge_conflict_log.

    Expected payload keys (must match SQL schema):
        url_registry_id         int
        normalized_id           int
        field_path              str
        run_a_value             any      — JSONB; None → null
        enrichment_value        any      — JSONB; None → null
        enrichment_candidate_id int | None
        resolution              str      — 'kept_run_a' | 'kept_enrichment' | 'flagged'
        resolved_by             str | None — 'auto_source_priority' | 'auto_confidence' | None
        resolution_note         str | None
        resolved_at             str | None — ISO timestamp for auto-resolutions

    Returns:
        conflict_id (int)
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("merge_conflict_log")
        .insert(payload)
        .execute()
    )
    if not result.data:
        raise RuntimeError(
            f"insert_merge_conflict: no data returned for "
            f"field_path={payload.get('field_path')!r}, "
            f"normalized_id={payload.get('normalized_id')}. "
            f"Check merge_conflict_log schema and constraints."
        )
    return result.data[0]["conflict_id"]


def fetch_flagged_conflict_count(normalized_id: int) -> int:
    """
    Returns the count of conflicts with resolution='flagged' for this phone.
    Used to compute final_merged_json.has_unresolved_conflicts.

    Note: 'pending' is also unresolved but should not exist after conflict_resolver runs.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("merge_conflict_log")
        .select("conflict_id", count="exact")
        .eq("normalized_id", normalized_id)
        .in_("resolution", ["pending", "flagged"])
        .execute()
    )
    return result.count or 0


def fetch_pending_staging_count(url_registry_id: int) -> int:
    """
    Returns the count of lookup_value_staging rows with status='pending_review'
    for this phone. Used to compute final_merged_json.pending_staging_values.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("lookup_value_staging")
        .select("staging_id", count="exact")
        .eq("url_registry_id", url_registry_id)
        .eq("status", "pending_review")
        .execute()
    )
    return result.count or 0


def upsert_final_merged_json(payload: dict) -> int:
    """
    Inserts or updates pipeline.final_merged_json.
    ON CONFLICT (url_registry_id) DO UPDATE — handles re-pipeline runs.

    Expected payload keys (must match SQL schema):
        url_registry_id             int  — UNIQUE constraint, conflict target
        normalized_id               int
        enrichment_run_id           int | None
        final_json                  dict — JSONB
        fields_remaining_null       int
        has_unresolved_conflicts    bool
        pending_staging_values      int
        spec_human_approved         bool — always False on insert
        experience_human_approved   bool — always False on insert
        experience_entries_reviewed bool — always False on insert
        ready_for_commit            bool — always False on insert

    Returns:
        final_id (int)
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("final_merged_json")
        .upsert(
            payload,
            on_conflict="url_registry_id",
            ignore_duplicates=False,
        )
        .execute()
    )
    if not result.data:
        raise RuntimeError(
            f"upsert_final_merged_json: no data returned for "
            f"url_registry_id={payload.get('url_registry_id')}, "
            f"normalized_id={payload.get('normalized_id')}. "
            f"Check final_merged_json schema and UNIQUE constraint."
        )
    return result.data[0]["final_id"]


def insert_pre_ui_validation_run(payload: dict) -> int:
    """
    Inserts a row into pipeline.pre_ui_validation_runs.

    Expected payload keys (must match SQL schema):
        url_registry_id  int
        normalized_id    int
        status           str  — 'passed' | 'failed'
        errors           list — JSONB; [] if passed
        error_count      int
        warnings         list — JSONB
        warning_count    int
        validated_at     str  — ISO timestamp

    Returns:
        pre_ui_val_id (int)
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("pre_ui_validation_runs")
        .insert(payload)
        .execute()
    )
    if not result.data:
        raise RuntimeError(
            f"insert_pre_ui_validation_run: no data returned for "
            f"normalized_id={payload.get('normalized_id')}. "
            f"Check pre_ui_validation_runs schema and constraints."
        )
    return result.data[0]["pre_ui_val_id"]


def fetch_final_merged_json(final_id: int) -> dict | None:
    """
    Fetches a final_merged_json row by final_id.
    Returns None if not found.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("final_merged_json")
        .select(
            "final_id, url_registry_id, normalized_id, enrichment_run_id, "
            "final_json, fields_remaining_null, has_unresolved_conflicts, "
            "pending_staging_values, spec_human_approved, experience_human_approved, "
            "experience_entries_reviewed, ready_for_commit"
        )
        .eq("final_id", final_id)
        .execute()
    )
    if not result.data:
        return None
    return result.data[0]


def fetch_latest_pre_ui_validation(normalized_id: int) -> dict | None:
    """
    Returns the most recent pre_ui_validation_runs row for this normalized_id.
    Returns None if no validation has been run.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("pre_ui_validation_runs")
        .select(
            "pre_ui_val_id, status, errors, error_count, "
            "warnings, warning_count, validated_at"
        )
        .eq("normalized_id", normalized_id)
        .order("pre_ui_val_id", desc=True)
        .limit(1)
        .execute()
    )
    if not result.data:
        return None
    return result.data[0]


def fetch_phone_experience_count(url_registry_id: int) -> int:
    """
    Returns the count of phone_experiences rows with is_suppressed=FALSE,
    is_superseded=FALSE, and confidence >= 0.50 for this phone.
    Used in Phase 7.5 Run B minimum check.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("phone_experiences")
        .select("experience_id", count="exact")
        .eq("url_registry_id", url_registry_id)
        .eq("is_suppressed", False)
        .eq("is_superseded", False)
        .gte("confidence", 0.50)
        .execute()
    )
    return result.count or 0


# ---------------------------------------------------------------------------
# Phase 7 — additional repository helpers
# ---------------------------------------------------------------------------


def fetch_conflict_resolution_map(normalized_id: int) -> dict[str, str]:
    """
    Returns a dict mapping field_path → resolution for auto-resolved conflicts
    for this normalized_id.

    Only returns rows with resolution in ('kept_run_a', 'kept_enrichment').
    Flagged conflicts are excluded — Run A value stays in final_json for those.
    Used by build_final_merged_json to determine which enrichment values to apply.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("merge_conflict_log")
        .select("field_path, resolution")
        .eq("normalized_id", normalized_id)
        .in_("resolution", ["kept_run_a", "kept_enrichment"])
        .execute()
    )
    return {
        row["field_path"]: row["resolution"]
        for row in (result.data or [])
    }


def delete_conflict_log_for_phone(normalized_id: int) -> None:
    """
    Deletes all AUTO-RESOLVED conflict rows for this normalized_id.

    Called at the START of detect_and_resolve_conflicts to ensure idempotency.
    Without this, re-running Phase 7 creates duplicate conflict rows for the same
    (normalized_id, field_path), inflating flagged counts and corrupting the admin UI.

    P8-1 fix: In Postgres, NULL != 'human_override' evaluates to NULL (not TRUE).
    PostgREST's .neq() therefore silently skips all flagged rows whose resolved_by
    is NULL — those rows survive re-runs and inflate has_unresolved_conflicts forever.

    Correct filter: delete rows where resolved_by IS NULL OR resolved_by != 'human_override'.
    This preserves admin-manually-resolved rows (resolved_by='human_override') while
    deleting all auto-resolved ('auto_source_priority', 'auto_confidence') and flagged
    (NULL) rows so they can be re-generated cleanly.
    """
    (
        get_client()
        .schema("pipeline")
        .table("merge_conflict_log")
        .delete()
        .eq("normalized_id", normalized_id)
        .or_("resolved_by.is.null,resolved_by.neq.human_override")
        .execute()
    )


def fetch_all_enrichment_candidates_for_run(
    enrichment_run_id: int,
) -> list[dict]:
    """
    Returns ALL enrichment_field_candidates for a run, regardless of is_selected.

    Used by build_final_merged_json for the FILL pass: when Run A is null,
    we use the highest-confidence candidate regardless of selection status.
    This avoids the C7 bug where unselected but usable candidates produce
    permanently null fields in final_merged_json.

    Columns returned:
        candidate_id, missing_field_id, field_path, extracted_value,
        raw_confidence, confidence, source_tier, is_selected
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("enrichment_field_candidates")
        .select(
            "candidate_id, missing_field_id, field_path, extracted_value, "
            "raw_confidence, confidence, source_tier, is_selected"
        )
        .eq("enrichment_run_id", enrichment_run_id)
        .order("confidence", desc=True)  # highest confidence first
        .execute()
    )
    return result.data or []


def fetch_oem_raw_ids_for_phone(url_registry_id: int) -> set[int]:
    """
    Returns the set of raw_id values from spec_extraction_runs for this phone
    that correspond to OEM official sources (site_name LIKE '%_official').

    Used by detect_and_resolve_conflicts to correctly identify when Run A
    drew from an OEM source, enabling Rule 2 (Run A OEM priority) to fire.

    Returns an empty set if no OEM scrapes exist or no extraction run found.
    """
    # Step 1: Get the most recent spec_extraction_run for this phone
    run_result = (
        get_client()
        .schema("pipeline")
        .table("spec_extraction_runs")
        .select("extraction_run_id, raw_source_ids")
        .eq("url_registry_id", url_registry_id)
        .order("extraction_run_id", desc=True)
        .limit(1)
        .execute()
    )
    if not run_result.data:
        return set()

    raw_source_ids: list[int] = run_result.data[0].get("raw_source_ids") or []
    if not raw_source_ids:
        return set()

    # Step 2: Filter to OEM official rows by site_name convention
    # raw_scraped_data has no site_name; join url_registry via url_registry_id FK.
    # Fetch all rows then filter client-side to those with site_name ending in _official.
    scraped_result = (
        get_client()
        .schema("pipeline")
        .table("raw_scraped_data")
        .select("raw_id, url_registry(site_name)")
        .in_("raw_id", raw_source_ids)
        .execute()
    )
    return {
        row["raw_id"]
        for row in (scraped_result.data or [])
        if (row.get("url_registry") or {}).get("site_name", "").endswith("_official")
    }


# ===========================================================================
# Phase 8 — Admin Review API Repository Functions
# ===========================================================================


# ---------------------------------------------------------------------------
# 8.1 — Phones ready for review
# ---------------------------------------------------------------------------

def fetch_phones_ready_for_review() -> list[dict]:
    """
    Returns all final_merged_json rows joined with url_registry for brand/model.
    Includes phones not yet committed (ready_for_commit=False) AND those already
    committed but not yet approved (so the admin can still access them).
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("final_merged_json")
        .select(
            "final_id, url_registry_id, normalized_id, "
            "fields_remaining_null, has_unresolved_conflicts, pending_staging_values, "
            "spec_human_approved, experience_human_approved, experience_entries_reviewed, "
            "ready_for_commit, created_at, updated_at, "
            "url_registry(brand, model_name)"
        )
        .execute()
    )
    rows = result.data or []
    # Flatten the nested url_registry join
    for row in rows:
        nested = row.pop("url_registry", None) or {}
        row["brand"]      = nested.get("brand", "Unknown")
        row["model_name"] = nested.get("model_name", "Unknown")
    return rows


def fetch_approval_package(final_id: int) -> dict | None:
    """
    Fetches the full approval package for a phone: final_merged_json row
    including all gate conditions and final_json.
    Returns None if not found.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("final_merged_json")
        .select(
            "final_id, url_registry_id, normalized_id, enrichment_run_id, "
            "final_json, fields_remaining_null, has_unresolved_conflicts, "
            "pending_staging_values, spec_human_approved, experience_human_approved, "
            "experience_entries_reviewed, ready_for_commit, "
            "spec_approved_by, spec_approved_at, "
            "experience_approved_by, experience_approved_at, "
            "created_at, updated_at"
        )
        .eq("final_id", final_id)
        .execute()
    )
    if not result.data:
        return None
    return result.data[0]


# ---------------------------------------------------------------------------
# 8.2 — Admin review sessions
# ---------------------------------------------------------------------------

def insert_admin_review_session(payload: dict) -> int:
    """
    Opens a new admin review session row.

    Expected payload keys:
        final_id         int
        url_registry_id  int
        admin_user       str
        initial_view     str  — 'ui_view' | 'json_view'

    Returns:
        session_id (int)
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("admin_review_sessions")
        .insert(payload)
        .execute()
    )
    if not result.data:
        raise RuntimeError(
            f"insert_admin_review_session: no data returned for "
            f"final_id={payload.get('final_id')!r}."
        )
    return result.data[0]["session_id"]


def close_admin_review_session(session_id: int, outcome: str) -> None:
    """
    Closes an open review session by setting closed_at and session_outcome.

    outcome must be one of: 'approved', 'rejected', 'deferred', 'partial_edit'
    """
    (
        get_client()
        .schema("pipeline")
        .table("admin_review_sessions")
        .update({
            "session_outcome": outcome,
            "closed_at":       _now_iso(),
        })
        .eq("session_id", session_id)
        .execute()
    )


# ---------------------------------------------------------------------------
# 8.3 — Evidence tooltip
# ---------------------------------------------------------------------------

def fetch_evidence_for_field(
    url_registry_id: int,
    field_path: str,
    final_id: int | None = None,
) -> dict | None:
    """
    Returns the evidence entry for a specific field.

    P8-8 fix: Checks admin_field_overrides first (most-recent override wins for
    source attribution). If an admin override exists for this field, returns
    synthetic evidence with source_type='admin_override' so the tooltip shows
    the correct source — not the original Run A source that no longer corresponds
    to the current value.

    Priority order:
        1. Most-recent admin_field_overrides row for (final_id, field_path)
           → source_type='admin_override'
        2. evidence_json[field_path] from spec_extraction_output
           → original Run A source
        3. None → greyed tooltip (no source recorded)
    """
    # Priority 1: check admin_field_overrides if final_id provided
    if final_id is not None:
        override_result = (
            get_client()
            .schema("pipeline")
            .table("admin_field_overrides")
            .select("override_id, session_id, new_value, override_reason, created_at")
            .eq("final_id", final_id)
            .eq("field_path", field_path)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if override_result.data:
            ov = override_result.data[0]
            # Fetch admin_user from the linked session
            session_result = (
                get_client()
                .schema("pipeline")
                .table("admin_review_sessions")
                .select("admin_user")
                .eq("session_id", ov["session_id"])
                .execute()
            )
            admin_user = (
                session_result.data[0]["admin_user"]
                if session_result.data
                else "unknown_admin"
            )
            return {
                "source_type":    "admin_override",
                "source_label":   f"Admin override ({admin_user})",
                "admin_user":     admin_user,
                "override_reason": ov.get("override_reason"),
                "new_value":      ov["new_value"],
                "override_id":    ov["override_id"],
                "evidence":       None,
                "confidence":     None,
            }

    # Priority 2: Run A evidence_json
    result = (
        get_client()
        .schema("pipeline")
        .table("spec_extraction_output")
        .select("evidence_json")
        .eq("url_registry_id", url_registry_id)
        .order("output_id", desc=True)
        .limit(1)
        .execute()
    )
    if not result.data:
        return None
    evidence_json: dict = result.data[0].get("evidence_json") or {}
    return evidence_json.get(field_path)


# ---------------------------------------------------------------------------
# 8.4 — Spec field override (inline admin edit)
# ---------------------------------------------------------------------------

def insert_admin_field_override(payload: dict) -> int:
    """
    Logs one admin spec field edit into admin_field_overrides.

    Expected payload keys (matching SQL schema):
        session_id              int
        final_id                int
        url_registry_id         int
        field_path              str
        previous_value          any   — JSONB; None if previously null
        new_value               any   — JSONB; must not be null
        override_reason         str | None
        resolves_conflict_id    int | None  — links to merge_conflict_log.conflict_id

    Returns:
        override_id (int)
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("admin_field_overrides")
        .insert(payload)
        .execute()
    )
    if not result.data:
        raise RuntimeError(
            f"insert_admin_field_override: no data returned for "
            f"final_id={payload.get('final_id')!r}, "
            f"field_path={payload.get('field_path')!r}."
        )
    return result.data[0]["override_id"]


# patch_final_json_field REMOVED (P8-3):
#   The previous implementation first wrote NULL to final_json then called a
#   non-existent RPC — guaranteed data loss on first call. The approved path for
#   all admin edits is update_final_json_direct below. If atomic jsonb_set is ever
#   needed, create the Postgres function pipeline_patch_final_json first, then wire
#   a new Python caller. Do not reintroduce this pattern without the RPC existing.


def update_final_json_direct(final_id: int, final_json: dict) -> None:
    """
    Replaces the entire final_json blob for a final_merged_json row.

    APPROVED PATH for all admin edits to final_json.
    Pattern: fetch → patch in Python memory via _set_at_path → write blob back.
    This is safe because admin edits are low-frequency (one at a time) and the
    blob is small enough that a full replace is not a performance concern.

    If atomic per-field jsonb_set is needed in future, implement the Postgres
    function pipeline_patch_final_json first, then write a new Python caller.
    """
    (
        get_client()
        .schema("pipeline")
        .table("final_merged_json")
        .update({"final_json": final_json})
        .eq("final_id", final_id)
        .execute()
    )


# ---------------------------------------------------------------------------
# 8.5 — Conflict resolution (human override)
# ---------------------------------------------------------------------------

def update_conflict_resolution(
    conflict_id: int,
    resolution: str,
    resolved_value: object,
    admin_user: str,
) -> None:
    """
    Records an admin-manual conflict resolution.

    Sets resolution='human_override', resolved_by='human_override',
    resolved_at=NOW(). Also stores the chosen value in resolution_note.

    resolution must be one of: 'kept_run_a', 'kept_enrichment', 'human_override'
    (the admin may supply a custom value different from both sides).
    """
    (
        get_client()
        .schema("pipeline")
        .table("merge_conflict_log")
        .update({
            "resolution":      resolution,
            "resolved_by":     "human_override",
            "resolution_note": (
                f"Admin ({admin_user}) manually resolved. "
                f"Chosen value: {resolved_value!r}"
            ),
            "resolved_at":     _now_iso(),
        })
        .eq("conflict_id", conflict_id)
        .execute()
    )


def fetch_flagged_conflicts_for_phone(normalized_id: int) -> list[dict]:
    """
    Returns all unresolved (flagged) conflicts for this phone.
    Used to populate the admin conflict review panel.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("merge_conflict_log")
        .select(
            "conflict_id, field_path, run_a_value, enrichment_value, "
            "enrichment_candidate_id, resolution, resolved_by, resolution_note, resolved_at"
        )
        .eq("normalized_id", normalized_id)
        .in_("resolution", ["pending", "flagged"])
        .execute()
    )
    return result.data or []


# ---------------------------------------------------------------------------
# 8.6 — Spec approval gate
# ---------------------------------------------------------------------------

def approve_spec(
    final_id: int,
    admin_user: str,
) -> None:
    """
    Sets spec_human_approved=TRUE and records approver identity.
    Also recomputes ready_for_commit.
    """
    now = _now_iso()
    (
        get_client()
        .schema("pipeline")
        .table("final_merged_json")
        .update({
            "spec_human_approved": True,
            "spec_approved_by":    admin_user,
            "spec_approved_at":    now,
        })
        .eq("final_id", final_id)
        .execute()
    )
    _recompute_ready_for_commit(final_id)


def _recompute_ready_for_commit(final_id: int) -> None:
    """
    Fetches current gate conditions and updates ready_for_commit.
    Called after any gate-condition-changing operation.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("final_merged_json")
        .select(
            "has_unresolved_conflicts, pending_staging_values, "
            "spec_human_approved, experience_human_approved, "
            "experience_entries_reviewed"
        )
        .eq("final_id", final_id)
        .execute()
    )
    if not result.data:
        return
    row = result.data[0]
    ready = (
        not row["has_unresolved_conflicts"]
        and row["pending_staging_values"] == 0
        and row["spec_human_approved"]
        and row["experience_human_approved"]
        and row["experience_entries_reviewed"]
    )
    (
        get_client()
        .schema("pipeline")
        .table("final_merged_json")
        .update({"ready_for_commit": ready})
        .eq("final_id", final_id)
        .execute()
    )


def update_gate_conditions(final_id: int, patch: dict) -> None:
    """
    Partial-updates gate condition columns on final_merged_json and recomputes
    ready_for_commit. patch may contain any subset of:
        has_unresolved_conflicts bool
        pending_staging_values   int
        spec_human_approved      bool
        experience_human_approved      bool
        experience_entries_reviewed    bool
    """
    if patch:
        (
            get_client()
            .schema("pipeline")
            .table("final_merged_json")
            .update(patch)
            .eq("final_id", final_id)
            .execute()
        )
    _recompute_ready_for_commit(final_id)


# ---------------------------------------------------------------------------
# 8.7 — Phone experiences (Run B)
# ---------------------------------------------------------------------------

def fetch_experiences_for_phone(url_registry_id: int) -> list[dict]:
    """
    Returns all active (non-superseded) phone_experiences rows for this phone,
    ordered by category_id. Includes admin_edited and is_suppressed flags for
    UI colour coding. Also joins youtube metadata (video_title, video_url,
    channel_name) via raw_transcript_id -> video_registry -> channel chain.

    IMPORTANT: Always filters is_superseded=FALSE. Superseded rows from
    previous Run B re-runs must never be surfaced to the admin UI.
    """
    client = get_client()

    # Step 1 -- base experiences (include raw_transcript_id for the join)
    base_result = (
        client
        .schema("pipeline")
        .table("phone_experiences")
        .select(
            "experience_id, exp_run_id, url_registry_id, category_id, "
            "experience_text, sentiment, evidence_quote, confidence, "
            "is_suppressed, is_verified, admin_edited, created_at, updated_at, "
            "raw_transcript_id, "
            "lookup_experience_categories(category_name)"
        )
        .eq("url_registry_id", url_registry_id)
        .eq("is_superseded", False)
        .order("category_id")
        .execute()
    )
    rows: list[dict] = list(base_result.data or [])

    # Step 2 -- collect unique raw_transcript_ids that are not null
    rtids = list({r["raw_transcript_id"] for r in rows if r.get("raw_transcript_id") is not None})
    video_meta: dict[int, dict] = {}  # raw_transcript_id -> {video_title, video_url, channel_name}

    if rtids:
        # transcript_id -> video_registry_id
        tr = (
            client
            .schema("pipeline")
            .table("youtube_raw_transcript_data")
            .select("raw_transcript_id, video_registry_id")
            .in_("raw_transcript_id", rtids)
            .execute()
        )
        rtid_to_vrid: dict[int, int] = {
            r["raw_transcript_id"]: r["video_registry_id"]
            for r in (tr.data or [])
        }

        vrids = list(set(rtid_to_vrid.values()))
        if vrids:
            # video_registry_id -> video_title, video_url, channel_id
            vr = (
                client
                .schema("pipeline")
                .table("youtube_video_url_registry")
                .select("video_registry_id, video_title, video_url, channel_id")
                .in_("video_registry_id", vrids)
                .execute()
            )
            vrid_to_info: dict[int, dict] = {
                r["video_registry_id"]: r for r in (vr.data or [])
            }

            # channel_id -> channel_name
            channel_ids = list({
                r["channel_id"] for r in vrid_to_info.values() if r.get("channel_id")
            })
            cid_to_name: dict[int, str] = {}
            if channel_ids:
                cr = (
                    client
                    .schema("pipeline")
                    .table("youtube_channels")
                    .select("channel_id, channel_name")
                    .in_("channel_id", channel_ids)
                    .execute()
                )
                cid_to_name = {r["channel_id"]: r["channel_name"] for r in (cr.data or [])}

            for rtid, vrid in rtid_to_vrid.items():
                info = vrid_to_info.get(vrid, {})
                channel_id = info.get("channel_id")
                video_meta[rtid] = {
                    "video_title":  info.get("video_title"),
                    "video_url":    info.get("video_url"),
                    "channel_name": cid_to_name.get(channel_id) if channel_id else None,
                }

    # Step 3 -- merge video metadata; pop raw_transcript_id (not part of public API)
    # Also flatten nested lookup_experience_categories join result.
    for row in rows:
        rtid = row.pop("raw_transcript_id", None)
        meta = video_meta.get(rtid) if rtid is not None else None
        row["video_title"]  = meta["video_title"]  if meta else None
        row["video_url"]    = meta["video_url"]    if meta else None
        row["channel_name"] = meta["channel_name"] if meta else None
        # Flatten the nested category join: {category_name: ...} -> flat string
        cat_nested = row.pop("lookup_experience_categories", None)
        if isinstance(cat_nested, dict):
            row["category_name"] = cat_nested.get("category_name")
        else:
            row["category_name"] = None

    return rows


def fetch_experience_by_id(experience_id: int) -> dict | None:
    """
    Fetches a single phone_experiences row by PK.
    Returns None if not found.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("phone_experiences")
        .select(
            "experience_id, exp_run_id, url_registry_id, category_id, "
            "experience_text, sentiment, evidence_quote, confidence, "
            "is_suppressed, is_verified, admin_edited, created_at, updated_at"
        )
        .eq("experience_id", experience_id)
        .execute()
    )
    if not result.data:
        return None
    return result.data[0]


def insert_admin_experience_override(payload: dict) -> int:
    """
    Appends one admin edit row to admin_experience_overrides.

    Expected payload keys:
        session_id       int
        experience_id    int
        url_registry_id  int
        field_edited     str  — 'experience_text' | 'sentiment'
        previous_value   str | None
        new_value        str
        override_reason  str | None

    Returns:
        exp_override_id (int)
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("admin_experience_overrides")
        .insert(payload)
        .execute()
    )
    if not result.data:
        raise RuntimeError(
            f"insert_admin_experience_override: no data returned for "
            f"experience_id={payload.get('experience_id')}."
        )
    return result.data[0]["exp_override_id"]


def update_experience_field(
    experience_id: int,
    field: str,
    new_value: str,
) -> None:
    """
    Updates experience_text or sentiment on a phone_experiences row.
    Also sets admin_edited=TRUE.
    field must be 'experience_text' or 'sentiment'.
    """
    (
        get_client()
        .schema("pipeline")
        .table("phone_experiences")
        .update({field: new_value, "admin_edited": True})
        .eq("experience_id", experience_id)
        .execute()
    )


def set_experience_suppressed(
    experience_id: int,
    suppressed: bool,
) -> None:
    """
    Sets is_suppressed on a phone_experiences row.
    True  = suppress (effectively delete from commit).
    False = restore.
    """
    (
        get_client()
        .schema("pipeline")
        .table("phone_experiences")
        .update({"is_suppressed": suppressed})
        .eq("experience_id", experience_id)
        .execute()
    )


def approve_experiences(
    final_id: int,
    url_registry_id: int,
    admin_user: str,
) -> None:
    """
    Sets experience_human_approved=TRUE and experience_entries_reviewed=TRUE.
    Also recomputes ready_for_commit.
    """
    now = _now_iso()
    (
        get_client()
        .schema("pipeline")
        .table("final_merged_json")
        .update({
            "experience_human_approved":   True,
            "experience_entries_reviewed": True,
            "experience_approved_by":      admin_user,
            "experience_approved_at":      now,
        })
        .eq("final_id", final_id)
        .execute()
    )
    _recompute_ready_for_commit(final_id)


def reset_experience_approval(final_id: int) -> None:
    """
    Resets experience_human_approved and experience_entries_reviewed to FALSE.
    Called when Run B is re-triggered for a phone that has already been approved.
    """
    (
        get_client()
        .schema("pipeline")
        .table("final_merged_json")
        .update({
            "experience_human_approved":   False,
            "experience_entries_reviewed": False,
        })
        .eq("final_id", final_id)
        .execute()
    )
    _recompute_ready_for_commit(final_id)


# ---------------------------------------------------------------------------
# 8.8 — Staging queue
# ---------------------------------------------------------------------------

def fetch_pending_staging_entries(url_registry_id: int) -> list[dict]:
    """
    Returns all pending lookup_value_staging rows for this phone,
    ordered by target_lookup_table for grouped display.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("lookup_value_staging")
        .select(
            "staging_id, extracted_value, target_lookup_table, "
            "field_path, source_stage, status, created_at"
        )
        .eq("url_registry_id", url_registry_id)
        .eq("status", "pending_review")
        .order("target_lookup_table")
        .execute()
    )
    return result.data or []


def resolve_staging_entry(
    staging_id: int,
    resolution: str,
    resolved_lookup_id: int | None,
) -> None:
    """
    Marks a staging entry as resolved.

    resolution: 'inserted_new' | 'mapped_to_existing' | 'commit_as_null'

    M2 fix: the column is 'reviewed_at' (not 'resolved_at') per pipeline schema.
    M3 fix: 'commit_as_null' is a first-class status in the DB constraint
            ('pending_review' | 'inserted_new' | 'mapped_to_existing' | 'rejected'
             | 'committed_as_null').
            The request value 'commit_as_null' is mapped to DB status 'committed_as_null'.

    resolved_lookup_id: the PK of the inserted/mapped lookup row (None for commit_as_null)
    """
    # Map the API resolution value to the exact DB CHECK constraint value
    _STATUS_MAP = {
        "commit_as_null": "committed_as_null",
    }
    db_status = _STATUS_MAP.get(resolution, resolution)

    update_payload: dict = {
        "status":             db_status,
        "resolved_lookup_id": resolved_lookup_id,
        "reviewed_at":        _now_iso(),
    }
    (
        get_client()
        .schema("pipeline")
        .table("lookup_value_staging")
        .update(update_payload)
        .eq("staging_id", staging_id)
        .execute()
    )


def reset_staging_entries(url_registry_id: int) -> int:
    """
    Revert all non-pending staging rows for a phone back to 'pending_review'.
    Clears resolution metadata. Returns the number of rows reset.
    """
    from app.core.supabase_client import get_client
    client = get_client()
    result = (
        client
        .schema("pipeline")
        .table("lookup_value_staging")
        .update({
            "status":                  "pending_review",
            "resolution":              None,
            "resolved_lookup_id":      None,
            "resolution_target_table": None,
            "new_row_data":            None,
        })
        .eq("url_registry_id", url_registry_id)
        .neq("status", "pending_review")
        .execute()
    )
    return len(result.data or [])


# ===========================================================================
# Phase 9 — Pre-Commit Validation Repository Functions
# ===========================================================================


def insert_commit_validation_run(payload: dict) -> int:
    """
    Inserts a commit_validation_runs row and returns commit_val_id.

    Expected payload keys:
        final_id         int
        url_registry_id  int
        status           str   — 'pending' | 'passed' | 'failed'
        errors           list  — JSONB; hard failures list
        error_count      int
        validated_at     str   — ISO-8601 UTC timestamp
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("commit_validation_runs")
        .insert(payload)
        .execute()
    )
    if not result.data:
        raise RuntimeError(
            f"insert_commit_validation_run: no data returned for "
            f"final_id={payload.get('final_id')!r}."
        )
    return result.data[0]["commit_val_id"]


def fetch_latest_commit_validation(final_id: int) -> dict | None:
    """
    Returns the most recent commit_validation_runs row for this final_id.
    Returns None if no validation run has been persisted yet.

    P9-6: Provides a proper repo-layer function so validation_service doesn't
    issue raw DB calls directly (breaks the repository pattern).
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("commit_validation_runs")
        .select(
            "commit_val_id, status, errors, error_count, "
            "warnings, warning_count, validated_at, created_at"
        )
        .eq("final_id", final_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not result.data:
        return None
    return result.data[0]


def fetch_gate_conditions_live(final_id: int) -> dict | None:
    """
    Re-fetches the five commit gate conditions from the live DB row.
    Used by Phase 9 to avoid relying on cached booleans.

    Returns:
        {
          has_unresolved_conflicts, pending_staging_values,
          spec_human_approved, experience_human_approved,
          experience_entries_reviewed
        }
    or None if final_id not found.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("final_merged_json")
        .select(
            "has_unresolved_conflicts, pending_staging_values, "
            "spec_human_approved, experience_human_approved, "
            "experience_entries_reviewed"
        )
        .eq("final_id", final_id)
        .execute()
    )
    if not result.data:
        return None
    return result.data[0]


def fetch_non_suppressed_experiences_for_commit(url_registry_id: int) -> list[dict]:
    """
    Returns all non-suppressed, non-superseded phone_experiences rows for the
    final Run B check. Only these rows will be committed.
    Returns experience_id, experience_text, confidence.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("phone_experiences")
        .select("experience_id, experience_text, confidence")
        .eq("url_registry_id", url_registry_id)
        .eq("is_suppressed", False)
        .eq("is_superseded", False)
        .execute()
    )
    return result.data or []


def fetch_brand_for_phone(url_registry_id: int) -> str | None:
    """
    Returns the brand string for this phone (used for foldable brand check).
    Returns None if not found.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("url_registry")
        .select("brand")
        .eq("url_id", url_registry_id)
        .limit(1)
        .execute()
    )
    if not result.data:
        return None
    return result.data[0].get("brand")


# ===========================================================================
# Phase 11 — Run C Inference Engine Repository Functions
# ===========================================================================


def fetch_active_inference_rules() -> list[dict]:
    """
    Returns all active rows from pipeline.lookup_inference_rules.
    Columns returned: rule_id, rule_name, category_id, positive_template,
    negative_template, neutral_template, thresholds, confidence_score,
    sentiment_positive, sentiment_negative, sentiment_neutral.

    Sorted by rule_id for deterministic evaluation order.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("lookup_inference_rules")
        .select(
            "rule_id, rule_name, category_id, "
            "positive_template, negative_template, neutral_template, "
            "thresholds, confidence_score, "
            "sentiment_positive, sentiment_negative, sentiment_neutral"
        )
        .eq("is_active", True)
        .order("rule_id")
        .execute()
    )
    return result.data or []


def upsert_phone_spec_inference(payload: dict) -> int:
    """
    Upserts a row into pipeline.phone_spec_inferences.

    Idempotency: UNIQUE CONSTRAINT uq_inference_phone_rule (model_id, rule_id).
    ON CONFLICT DO UPDATE: refreshes inference_text, sentiment, input_field_snapshot,
    confidence, is_suppressed, generated_at.

    Expected payload keys (must match phone_spec_inferences columns):
        url_registry_id       int
        model_id              int
        rule_id               int
        category_id           int
        inference_text        str
        sentiment             str    — 'Positive'|'Negative'|'Neutral'|'Mixed'
        input_field_snapshot  dict   — JSONB; exact field values used
        confidence            float  — 0.00–1.00
        is_suppressed         bool   — default False
        generated_at          str    — ISO-8601 UTC timestamp

    Returns:
        inference_id (int)
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("phone_spec_inferences")
        .upsert(
            payload,
            on_conflict="model_id,rule_id",
            ignore_duplicates=False,
        )
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise RuntimeError(
            f"upsert_phone_spec_inference: no row returned for "
            f"model_id={payload.get('model_id')} rule_id={payload.get('rule_id')}."
        )
    return rows[0]["inference_id"]


def fetch_inferences_for_phone(model_id: int) -> list[dict]:
    """
    Returns all non-suppressed phone_spec_inferences rows for this phone.
    Used by the embedding payload endpoint (Phase 12).

    Columns: inference_id, rule_id, category_id, inference_text,
             sentiment, confidence, generated_at.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("phone_spec_inferences")
        .select(
            "inference_id, rule_id, category_id, "
            "inference_text, sentiment, confidence, generated_at"
        )
        .eq("model_id", model_id)
        .eq("is_suppressed", False)
        .order("inference_id")
        .execute()
    )
    return result.data or []


# ---------------------------------------------------------------------------
# Phase 11 — M2: inference_runs audit table
# ---------------------------------------------------------------------------

def insert_inference_run(
    model_id: int,
    url_registry_id: int,
    triggered_by: str = "post_commit",
) -> int:
    """
    Inserts a running inference_runs row at the start of run_inference_engine.
    Returns inference_run_id for later update.

    triggered_by: 'post_commit' | 'admin_batch'
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("inference_runs")
        .insert({
            "model_id":        model_id,
            "url_registry_id": url_registry_id,
            "triggered_by":    triggered_by,
            "status":          "running",
        })
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise RuntimeError(
            f"insert_inference_run: no row returned for model_id={model_id}."
        )
    return rows[0]["inference_run_id"]


def update_inference_run(
    inference_run_id: int,
    rules_evaluated: int,
    rules_written: int,
    rules_skipped: int,
    errors: list[str],
) -> None:
    """
    Updates the inference_runs audit row with final outcome at end of run.

    Status logic:
      errors empty  AND rules_evaluated > 0  → 'completed'
      errors empty  AND rules_evaluated == 0 → 'completed' (no-op run)
      errors present AND rules_written > 0   → 'partial_failure'
      errors present AND rules_written == 0  → 'failed'
    """
    if errors and rules_written == 0:
        status = "failed"
    elif errors:
        status = "partial_failure"
    else:
        status = "completed"

    (
        get_client()
        .schema("pipeline")
        .table("inference_runs")
        .update({
            "rules_evaluated": rules_evaluated,
            "rules_written":   rules_written,
            "rules_skipped":   rules_skipped,
            "errors":          errors,
            "status":          status,
            "finished_at":     _now_iso(),
        })
        .eq("inference_run_id", inference_run_id)
        .execute()
    )




# ---------------------------------------------------------------------------
# Phase L6 — Auto-resolution helpers for POST /extraction/run-phone
# ---------------------------------------------------------------------------

def fetch_canonical_id_by_brand_model(brand: str, model_name: str) -> int:
    """
    Resolves a (brand, model_name) pair to a single canonical url_id.

    Uses the GSMArena row as the canonical anchor — it is always present,
    always unique per phone, and is the standard anchor used for YouTube
    enrichment and all downstream pipeline stages.

    Args:
        brand:      Brand name, e.g. "Samsung"  (case-sensitive, must match DB)
        model_name: Model name, e.g. "Galaxy S25 Ultra" (case-sensitive)

    Returns:
        url_id (int) of the GSMArena url_registry row for this phone.

    Raises:
        ValueError: If no GSMArena row is found for the given brand/model,
                    or if more than one row is found (data integrity violation).
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("url_registry")
        .select("url_id")
        .eq("brand", brand)
        .eq("model_name", model_name)
        .eq("site_name", "gsmarena")
        .execute()
    )

    rows = result.data or []

    if len(rows) == 0:
        raise ValueError(
            f"fetch_canonical_id_by_brand_model: no GSMArena anchor found for "
            f"brand={brand!r} model_name={model_name!r}. "
            f"Ensure the phone exists in pipeline.url_registry with site_name='gsmarena'."
        )

    if len(rows) > 1:
        url_ids = [r["url_id"] for r in rows]
        raise ValueError(
            f"fetch_canonical_id_by_brand_model: data integrity violation — "
            f"{len(rows)} GSMArena rows found for brand={brand!r} model_name={model_name!r}. "
            f"Expected exactly 1. url_ids found: {url_ids}"
        )

    return rows[0]["url_id"]


def fetch_sources_for_run_phone(
    canonical_url_id: int,
) -> dict:
    """
    Auto-resolves all source IDs needed for run-phone from brand/model.

    Collects:
        raw_source_ids       — all raw_scraped_data.raw_id rows for this phone
        top3_transcript_ids  — top 3 raw_transcript_ids ordered by
                               channel_reliability_score DESC, file_size_bytes DESC.
                               Fewer if fewer are available.
        raw_transcript_ids   — ALL available transcript IDs (for Run B)
        brand                — brand name from url_registry
        model_name           — model name from url_registry
        best_transcript_id   — highest-priority single ID (first of top3, or None)

    Returns:
        {
            "brand":                str,
            "model_name":           str,
            "raw_source_ids":       list[int],
            "top3_transcript_ids":  list[int],   # up to 3, ordered by reliability
            "raw_transcript_ids":   list[int],   # all available
            "best_transcript_id":   int | None,
        }

    Raises:
        ValueError: If canonical_url_id not found.
    """
    client = get_client()

    # Step 1 — Resolve brand + model_name
    anchor = (
        client
        .schema("pipeline")
        .table("url_registry")
        .select("brand, model_name")
        .eq("url_id", canonical_url_id)
        .execute()
    )
    if not anchor.data:
        raise ValueError(
            f"fetch_sources_for_run_phone: canonical_url_id={canonical_url_id} "
            "not found in pipeline.url_registry."
        )
    brand = anchor.data[0]["brand"]
    model_name = anchor.data[0]["model_name"]

    # Step 2 — Fetch ALL url_registry rows for this brand + model_name, so every
    # registered URL (GSMArena, OEM official site, etc.) contributes its scrape.
    # canonical_url_id is still the FK anchor for all other pipeline references;
    # only source file assembly is broadened here.
    all_url_result = (
        client
        .schema("pipeline")
        .table("url_registry")
        .select("url_id")
        .eq("brand", brand)
        .eq("model_name", model_name)
        .execute()
    )
    all_url_ids = [r["url_id"] for r in (all_url_result.data or [])]
    if not all_url_ids:
        all_url_ids = [canonical_url_id]  # Fallback — should never happen

    scraped_result = (
        client
        .schema("pipeline")
        .table("raw_scraped_data")
        .select("raw_id, status")
        .in_("url_registry_id", all_url_ids)
        .execute()
    )
    raw_source_ids = [r["raw_id"] for r in (scraped_result.data or [])]

    # Step 3 — Fetch fetched transcript IDs via youtube_video_url_registry
    # youtube_video_url_registry links url_registry_id → video_registry_id
    # youtube_raw_transcript_data links video_registry_id → raw_transcript_id
    video_result = (
        client
        .schema("pipeline")
        .table("youtube_video_url_registry")
        .select("video_registry_id, channel_id")
        .eq("url_registry_id", canonical_url_id)
        .eq("status", "fetched_raw")
        # L8.2: exclude comparison and group_review videos — their transcripts
        # produce contaminated Stage 1 candidates (competitor mentions, diluted
        # per-phone signals). video_type='unknown' (DB default for all existing
        # rows) is NOT excluded so legacy transcripts continue to work.
        .not_.in_("video_type", ["comparison", "group_review"])
        .execute()
    )
    video_rows = video_result.data or []
    video_ids = [r["video_registry_id"] for r in video_rows]
    # Build map: video_registry_id → channel_id for later reliability join
    vid_to_channel: dict[int, int | None] = {
        r["video_registry_id"]: r.get("channel_id")
        for r in video_rows
    }

    raw_transcript_ids: list[int] = []
    top3_transcript_ids: list[int] = []
    best_transcript_id: int | None = None

    if video_ids:
        transcript_result = (
            client
            .schema("pipeline")
            .table("youtube_raw_transcript_data")
            .select("raw_transcript_id, video_registry_id, file_size_bytes, fetched_at")
            .in_("video_registry_id", video_ids)
            .execute()
        )
        transcript_rows = transcript_result.data or []
        raw_transcript_ids = [r["raw_transcript_id"] for r in transcript_rows]

        # Fetch channel_reliability_score for all unique channel_ids
        channel_ids = list({
            cid for cid in vid_to_channel.values() if cid is not None
        })
        channel_reliability: dict[int, float] = {}
        if channel_ids:
            try:
                ch_result = (
                    client
                    .schema("pipeline")
                    .table("youtube_channels")
                    .select("channel_id, channel_reliability_score")
                    .in_("channel_id", channel_ids)
                    .execute()
                )
                for ch in (ch_result.data or []):
                    if ch.get("channel_reliability_score") is not None:
                        channel_reliability[ch["channel_id"]] = float(
                            ch["channel_reliability_score"]
                        )
            except Exception:
                pass  # Fall back to file_size_bytes only

        def _sort_key(r: dict) -> tuple:
            vid = r["video_registry_id"]
            cid = vid_to_channel.get(vid)
            reliability = channel_reliability.get(cid, 0.0) if cid else 0.0
            file_size = r.get("file_size_bytes") or 0
            # Descending: negate both for ascending sort
            return (-reliability, -file_size)

        sorted_rows = sorted(transcript_rows, key=_sort_key)
        top3_transcript_ids = [
            r["raw_transcript_id"] for r in sorted_rows[:3]
        ]
        best_transcript_id = top3_transcript_ids[0] if top3_transcript_ids else None

    return {
        "brand":               brand,
        "model_name":          model_name,
        "raw_source_ids":      raw_source_ids,
        "top3_transcript_ids": top3_transcript_ids,
        "raw_transcript_ids":  raw_transcript_ids,
        "best_transcript_id":  best_transcript_id,
    }


def supersede_experiences(url_registry_id: int) -> int:
    """
    Marks all current active phone_experiences rows for this phone as superseded.

    Called at the start of run_experience_extraction_batch() — AFTER gate
    validation passes — before inserting new rows. This ensures valid approved
    data is never lost if the gate fails.

    Only rows where is_superseded=FALSE are touched. Rows already superseded
    (from previous re-runs) are left unchanged.

    Returns:
        Count of rows marked as superseded (0 if no active rows existed).

    Raises:
        RuntimeError: On unexpected DB error.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("phone_experiences")
        .update({"is_superseded": True})
        .eq("url_registry_id", url_registry_id)
        .eq("is_superseded", False)
        .execute()
    )
    return len(result.data or [])


def validate_source_ids_belong_to_phone(
    canonical_url_id: int,
    raw_source_ids: list[int],
    raw_transcript_ids: list[int],
) -> dict:
    """
    Validates that all provided IDs belong to the phone identified by canonical_url_id.

    SECURITY INVARIANT: In manual mode the admin selects IDs from a UI checklist.
    This function ensures no ID from a different phone has been passed — either
    by mistake or by a direct API call bypassing the UI.

    For raw_source_ids:
        Checks raw_scraped_data.url_registry_id == canonical_url_id for every ID.

    For raw_transcript_ids:
        Resolves youtube_raw_transcript_data → video_registry_id →
        youtube_video_url_registry.url_registry_id and checks it equals
        canonical_url_id for every transcript ID.

    Returns:
        {
            "valid":                  bool,
            "foreign_source_ids":     list[int],  — raw_ids from wrong phone OR not found in DB
            "foreign_transcript_ids": list[int],  — transcript_ids from wrong phone
        }
        "valid" is True only when BOTH foreign lists are empty.

    Raises:
        Nothing. All DB errors are swallowed and treated as invalid (safe default).
    """
    client = get_client()
    foreign_source_ids: list[int] = []
    foreign_transcript_ids: list[int] = []

    # ── Check raw_source_ids ownership ──────────────────────────────────────
    if raw_source_ids:
        try:
            result = (
                client
                .schema("pipeline")
                .table("raw_scraped_data")
                .select("raw_id, url_registry_id")
                .in_("raw_id", raw_source_ids)
                .execute()
            )
            rows = result.data or []

            # IDs not returned by the DB do not exist — treat as foreign
            returned_ids = {row["raw_id"] for row in rows}
            missing_in_db = set(raw_source_ids) - returned_ids
            foreign_source_ids.extend(sorted(missing_in_db))

            # IDs that exist but belong to a different phone
            for row in rows:
                if row["url_registry_id"] != canonical_url_id:
                    foreign_source_ids.append(row["raw_id"])

        except Exception:
            # On any DB error treat ALL supplied IDs as unverified — safe default.
            foreign_source_ids = list(raw_source_ids)

    # ── Check raw_transcript_ids ownership ──────────────────────────────────
    if raw_transcript_ids:
        try:
            # Step 1: raw_transcript_id → video_registry_id
            transcript_result = (
                client
                .schema("pipeline")
                .table("youtube_raw_transcript_data")
                .select("raw_transcript_id, video_registry_id")
                .in_("raw_transcript_id", raw_transcript_ids)
                .execute()
            )
            transcript_rows = transcript_result.data or []

            # Map: raw_transcript_id → video_registry_id
            tid_to_vid: dict[int, int] = {
                r["raw_transcript_id"]: r["video_registry_id"]
                for r in transcript_rows
            }

            if tid_to_vid:
                # Step 2: video_registry_id → url_registry_id
                vr_result = (
                    client
                    .schema("pipeline")
                    .table("youtube_video_url_registry")
                    .select("video_registry_id, url_registry_id")
                    .in_("video_registry_id", list(tid_to_vid.values()))
                    .execute()
                )
                # Map: video_registry_id → url_registry_id
                vid_to_owner: dict[int, int] = {
                    r["video_registry_id"]: r["url_registry_id"]
                    for r in (vr_result.data or [])
                }

                for transcript_id, video_id in tid_to_vid.items():
                    owner = vid_to_owner.get(video_id)
                    if owner != canonical_url_id:
                        foreign_transcript_ids.append(transcript_id)

            # Transcripts whose raw_transcript_id was not found in DB at all
            missing_in_db = set(raw_transcript_ids) - set(tid_to_vid.keys())
            foreign_transcript_ids.extend(sorted(missing_in_db))

        except Exception:
            # On any DB error treat ALL supplied transcript IDs as unverified.
            foreign_transcript_ids = list(raw_transcript_ids)

    return {
        "valid":                  not foreign_source_ids and not foreign_transcript_ids,
        "foreign_source_ids":     foreign_source_ids,
        "foreign_transcript_ids": foreign_transcript_ids,
    }


def fetch_evidence_json_for_final(final_id: int) -> dict | None:
    """
    Fetches evidence_json for a phone identified by final_id.

    Traversal chain:
        final_merged_json.final_id
        → final_merged_json.url_registry_id
        → spec_extraction_output (most recent row by output_id for that url_registry_id)
        → spec_extraction_output.evidence_json (JSONB)

    Note: normalized_spec_json does NOT have an output_id column.
    The correct anchor is url_registry_id, not normalized_id.

    Returns:
        The evidence_json dict, or None if any step in the chain is missing.
    """
    client = get_client()

    # Step 1 — Resolve url_registry_id from final_id
    fmj_result = (
        client
        .schema("pipeline")
        .table("final_merged_json")
        .select("url_registry_id")
        .eq("final_id", final_id)
        .execute()
    )
    if not fmj_result.data:
        return None
    url_registry_id: int = fmj_result.data[0]["url_registry_id"]

    # Step 2 — Fetch the most recent evidence_json from spec_extraction_output
    eo_result = (
        client
        .schema("pipeline")
        .table("spec_extraction_output")
        .select("evidence_json")
        .eq("url_registry_id", url_registry_id)
        .order("output_id", desc=True)
        .limit(1)
        .execute()
    )
    if not eo_result.data:
        return None
    return eo_result.data[0].get("evidence_json")


# ===========================================================================
# Phase L8.2 — Two-Stage Experience System
# ===========================================================================


def fetch_existing_candidates_for_transcript(
    url_registry_id: int,
    raw_transcript_id: int,
) -> list[dict]:
    """
    Check whether Stage 1 has already been run for this (phone, transcript) pair.
    Returns all phone_experience_candidates rows for the pair.
    Empty list  → Stage 1 not yet run → run Stage 1.
    Non-empty   → candidates exist   → skip Stage 1, reuse.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("phone_experience_candidates")
        .select(
            "candidate_id, experience_text, sentiment, evidence_quote, "
            "category, confidence, raw_transcript_id"
        )
        .eq("url_registry_id", url_registry_id)
        .eq("raw_transcript_id", raw_transcript_id)
        .execute()
    )
    return result.data or []


def fetch_all_candidates_for_phone(url_registry_id: int) -> list[dict]:
    """
    Fetch ALL Stage 1 candidates for a phone.
    Fed to Stage 2 aggregation as the complete candidate set.
    Ordered by raw_transcript_id then candidate_id for deterministic input.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("phone_experience_candidates")
        .select(
            "candidate_id, raw_transcript_id, experience_text, sentiment, "
            "evidence_quote, category, confidence"
        )
        .eq("url_registry_id", url_registry_id)
        .order("raw_transcript_id")
        .order("candidate_id")
        .execute()
    )
    return result.data or []


def bulk_insert_experience_candidates(rows: list[dict]) -> int:
    """
    Bulk insert Stage 1 candidates into phone_experience_candidates.
    Expected keys per row:
        url_registry_id, raw_transcript_id, exp_run_id,
        experience_text, sentiment, evidence_quote (str|None),
        category, confidence (float)
    Returns count inserted.
    """
    if not rows:
        return 0
    result = (
        get_client()
        .schema("pipeline")
        .table("phone_experience_candidates")
        .insert(rows)
        .execute()
    )
    if not result.data:
        raise RuntimeError(
            f"bulk_insert_experience_candidates: insert returned no data "
            f"(exp_run_id={rows[0].get('exp_run_id')})."
        )
    return len(result.data)


def insert_aggregation_run(payload: dict) -> int:
    """
    Insert a new experience_aggregation_runs row (status='running').
    Expected keys: url_registry_id, model_used, schema_version,
        total_candidates_input, transcripts_input,
        new_transcripts_count, reused_transcripts_count, status.
    Returns aggregation_run_id.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("experience_aggregation_runs")
        .insert(payload)
        .execute()
    )
    if not result.data:
        raise RuntimeError(
            f"insert_aggregation_run: no data returned "
            f"(url_registry_id={payload.get('url_registry_id')})."
        )
    return result.data[0]["aggregation_run_id"]


def update_aggregation_run(run_id: int, updates: dict) -> None:
    """
    Update an experience_aggregation_runs row by aggregation_run_id.
    Pass "now()" as a string value for any timestamp field — this function
    resolves it to a real UTC ISO timestamp before inserting.
    """
    import datetime
    safe = {}
    for k, v in updates.items():
        if v == "now()":
            safe[k] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        else:
            safe[k] = v
    result = (
        get_client()
        .schema("pipeline")
        .table("experience_aggregation_runs")
        .update(safe)
        .eq("aggregation_run_id", run_id)
        .execute()
    )
    if not result.data:
        raise RuntimeError(
            f"update_aggregation_run: aggregation_run_id={run_id} not found."
        )


def bulk_insert_aggregated_experiences(rows: list[dict]) -> int:
    """
    Bulk insert Stage 2-aggregated experiences into phone_experiences.
    Expected keys per row:
        url_registry_id, aggregation_run_id, exp_run_id (None for Stage 2 rows),
        raw_transcript_id, category_id, experience_text, sentiment,
        evidence_quote (required — never None), confidence,
        source_transcript_count (int)
    Returns count inserted.
    """
    if not rows:
        return 0
    result = (
        get_client()
        .schema("pipeline")
        .table("phone_experiences")
        .insert(rows)
        .execute()
    )
    if not result.data:
        raise RuntimeError(
            f"bulk_insert_aggregated_experiences: insert returned no data "
            f"(aggregation_run_id={rows[0].get('aggregation_run_id')})."
        )
    return len(result.data)


# ---------------------------------------------------------------------------
# Section 6 — Staging split + metadata helpers
# ---------------------------------------------------------------------------

def update_staging_entry_metadata(
    staging_id: int,
    resolution_target_table: str | None,
    new_row_data: dict | None,
) -> None:
    """
    Persists admin-supplied deferred-insert metadata onto an existing
    lookup_value_staging row.

    Called by /staging/resolve when resolution='inserted_new' and the admin
    provides new_row_data (N4/N5/N6 alias or canonical insert paths).
    The commit-time Step 2.5 auto-resolver reads these columns back and
    performs the actual lookup-table insert.

    Args:
        staging_id:              PK of the staging row to update.
        resolution_target_table: Schema-qualified table name where the new row
                                 should be inserted, e.g.
                                 'mobile_specs.lookup_feature_aliases'.
        new_row_data:            Dict of column → value pairs for the new row.

    Raises:
        RuntimeError: If staging_id is not found (no rows updated).
    """
    updates: dict = {}
    if resolution_target_table is not None:
        updates["resolution_target_table"] = resolution_target_table
    if new_row_data is not None:
        updates["new_row_data"] = new_row_data

    if not updates:
        return  # Nothing to persist; caller passed both None

    result = (
        get_client()
        .schema("pipeline")
        .table("lookup_value_staging")
        .update(updates)
        .eq("staging_id", staging_id)
        .execute()
    )
    if not result.data:
        raise RuntimeError(
            f"update_staging_entry_metadata: staging_id={staging_id} not found "
            f"or no update applied."
        )

