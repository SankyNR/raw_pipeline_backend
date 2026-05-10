"""
Phase 4 — Normalisation

Task 4.1 — Lookup Cache Builder (build_lookup_cache):
    Queries every lookup table referenced in SCALAR_FK_MAP / ARRAY_FK_MAP at startup.
    Builds an in-memory {table_path: {cleaned_value: pk_id}} dict.
    Zero DB queries per phone during normalisation.

Task 4.2 — Normalisation Engine (run_normalisation):
    Full normalisation pass on spec_extraction_output.partial_json.
    Resolves strings to PKs, coerces types, strips units, validates ranges,
    sends unknowns to staging, and writes normalized_spec_json to the DB.

Design:
  - Supabase-py is synchronous; all DB calls wrapped in asyncio.to_thread().
  - LOOKUP_CACHE built at startup (main.py → await build_lookup_cache()).
  - [*] wildcards in field paths expanded to concrete indices at runtime.
  - RUN_C_CALCULATED_FIELDS discarded silently if LLM included them.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import re
from typing import Any

from Levenshtein import distance as levenshtein_distance

from app.config.field_mapping import (
    ARRAY_FK_MAP,
    NUMERIC_PRECISION_FIELDS,
    RUN_C_CALCULATED_FIELDS,
    SCALAR_FK_MAP,
)
from app.core.db_retry import db_call_with_retry  # #9 fix: retry on transient DB errors
from app.core.supabase_client import get_client
from app.repositories.extraction_repository import (
    fetch_spec_extraction_output,
    insert_normalisation_run,
    insert_lookup_value_staging,
    update_normalisation_run,
    upsert_normalized_spec,
)
from app.repositories.mobile_specs_repository import fetch_chipset_by_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Band prefix normalisation helper
# ---------------------------------------------------------------------------

def _normalize_band_value(value: str, field_path: str) -> str:
    """
    Deterministically enforces correct band prefixes before fuzzy lookup.
    4G bands require "B" prefix (B1, B28, B41).
    5G bands require "n" prefix (n78, n28, n41).
    Handles raw numbers, existing correct prefixes, and strips spaces.

    Called inside _resolve_array_fks for network.bands_4g and network.bands_5g
    items before they reach resolve_lookup_value().  This bypasses fuzzy
    matching for pure-numeric values, preventing wrong matches like
    "28" → "B8" (Levenshtein distance 1 on the cleaned string).
    """
    v = str(value).strip()
    if "bands_4g" in field_path:
        # Strip any existing prefix variants and re-apply "B"
        stripped = v.lstrip("Bbn").strip()
        if stripped.isdigit():
            return f"B{stripped}"
    elif "bands_5g" in field_path:
        # Strip any existing prefix variants and re-apply "n"
        stripped = v.lstrip("nNBb").strip()
        if stripped.isdigit():
            return f"n{stripped}"
    return v


# ---------------------------------------------------------------------------
# Change 3a — SIM configuration alias map
# Maps LLM phrasing variants to canonical lookup_sim_configurations values.
# Applied in _resolve_scalar_fks BEFORE resolve_lookup_value() to avoid
# sending known variants to staging.
# ---------------------------------------------------------------------------
_SIM_CONFIGURATION_ALIASES: dict[str, str] = {
    "Dual SIM (2 Nano SIMs)"         : "Dual SIM (Nano-SIM, dual stand-by)",
    "Dual SIM (Nano-SIM + Nano-SIM)" : "Dual SIM (Nano-SIM, dual stand-by)",
    "Dual Nano-SIM"                  : "Dual SIM (Nano-SIM, dual stand-by)",
    "Dual SIM (Nano, dual stand-by)" : "Dual SIM (Nano-SIM, dual stand-by)",
}


# ---------------------------------------------------------------------------
# Change 3b — Autofocus type alias map
# Maps LLM phrasing variants / compound names to canonical lookup_autofocus_types values.
# "Quad PDAF", "All-Pixel Focus", and "PDAF" are DISTINCT entries — do not merge.
# ---------------------------------------------------------------------------
_AUTOFOCUS_TYPE_ALIASES: dict[str, str] = {
    "Quad PDAF - All Pixel Focus" : "Quad PDAF",
    "Quad PDAF – All Pixel Focus" : "Quad PDAF",  # em-dash variant
    "All Pixel Focus"             : "All-Pixel Focus",
    "All-pixel Focus"             : "All-Pixel Focus",
}


# ---------------------------------------------------------------------------
# Change 3c — Wi-Fi Hotspot exclusion set
# "Wi-Fi Hotspot" has a dedicated boolean column (connectivity.wifi_hotspot).
# These values must be filtered out of the many-to-many junction table entirely
# before resolve_lookup_value() is called, so they do not reach staging either.
# ---------------------------------------------------------------------------
_WIFI_TECHNOLOGIES_EXCLUDE: frozenset[str] = frozenset({
    "Wi-Fi Hotspot",
    "Wi-Fi hotspot",
    "Hotspot",
    "WiFi Hotspot",
})


def _apply_brand_charging_fallback(data: dict) -> dict:
    """
    Step 8.5 — Deterministic proprietary_charging fill from brand.

    If charging.proprietary_charging is still null after normalisation and
    the brand is in _BRAND_CHARGING_FALLBACK, writes the canonical brand value
    directly into normalized_json. This avoids a wasted enrichment LLM call.

    For brands with None in the map (Google, Apple, Nothing) the field stays
    null — those brands have no proprietary charging name.
    """
    charging = data.get("charging")
    if not isinstance(charging, dict):
        return data
    if charging.get("proprietary_charging") is not None:
        return data  # already filled by Run A extraction — nothing to do

    brand_name: str = (data.get("brand") or {}).get("brand_name") or ""
    brand_key = brand_name.strip().title()
    if brand_key not in _BRAND_CHARGING_FALLBACK:
        return data  # unknown brand — leave for enrichment

    fill_value = _BRAND_CHARGING_FALLBACK[brand_key]
    charging["proprietary_charging"] = fill_value  # None is valid (means no proprietary name)
    return data

# ---------------------------------------------------------------------------
# Task 4.1 — In-memory lookup cache
# Built ONCE at startup; never reloaded mid-request.
# Structure: {table_path: {cleaned_value: pk_id}}
# Example:   {"mobile_specs.lookup_panel_types.panel_type": {"super amoled": 3}}
# ---------------------------------------------------------------------------
LOOKUP_CACHE: dict[str, dict[str, int]] = {}

# Reverse cache for corrected_value retrieval on fuzzy match
# Structure: {table_path: {cleaned_value: raw_canonical_value}}
_LOOKUP_CANONICAL: dict[str, dict[str, str]] = {}

# Levenshtein distance threshold — accept fuzzy if distance <= this value
_FUZZY_THRESHOLD = 2

# Lookup table paths where fuzzy matching is FORBIDDEN.
# These fields have values so similar to each other (distance 1–2)
# that a fuzzy match means committing the WRONG value, which is worse
# than sending to staging for human review.
_NO_FUZZY_TABLE_PATHS: frozenset[str] = frozenset({
    "mobile_specs.lookup_widevine_levels.level_name",              # L1/L2/L3 differ by 1 char
    "mobile_specs.lookup_video_certifications.certification_name", # HDR10 vs HDR10+ = dist 1
    "mobile_specs.lookup_audio_certifications.certification_name", # similar names
})

# Minimum string length to attempt fuzzy matching.
# Strings shorter than this are sent directly to staging if no exact match.
# Rationale: for a 2-char string like "L1", a distance of 2 would match
# almost anything. For 3-char strings like "OIS", distance 2 matches "EIS".
_FUZZY_MIN_LENGTH: int = 4


async def build_lookup_cache() -> None:
    """
    Queries all lookup tables referenced in SCALAR_FK_MAP and ARRAY_FK_MAP.
    Populates LOOKUP_CACHE and _LOOKUP_CANONICAL in-memory.

    Called once at app startup (in main.py).
    Re-callable via POST /extraction/cache/refresh to pick up new lookup rows.

    Format of each entry built:
        table_path = "mobile_specs.lookup_panel_types.panel_type"
        schema     = "mobile_specs"
        table      = "lookup_panel_types"
        column     = "panel_type"

    Uses supabase-py (sync) in a thread to avoid blocking the event loop.
    """
    all_table_paths: set[str] = (
        set(SCALAR_FK_MAP.values()) | set(ARRAY_FK_MAP.values())
    )
    logger.info(
        "build_lookup_cache: loading %d lookup tables...", len(all_table_paths)
    )

    for table_path in all_table_paths:
        try:
            await asyncio.to_thread(_load_one_table, table_path)
        except Exception as exc:
            logger.error(
                "build_lookup_cache: failed to load table_path=%r: %s",
                table_path, exc,
            )

    logger.info(
        "build_lookup_cache: complete. %d tables cached.", len(LOOKUP_CACHE)
    )


def _load_one_table(table_path: str) -> None:
    """Loads one lookup table into LOOKUP_CACHE and _LOOKUP_CANONICAL."""
    # table_path format: "schema.table.column"  (3 parts)
    parts = table_path.rsplit(".", 2)
    if len(parts) != 3:
        logger.warning(
            "_load_one_table: unexpected table_path format=%r — skipping.", table_path
        )
        return

    schema, table, column = parts

    # PK column name: by convention all lookup tables use <table>_id
    # e.g. lookup_panel_types → panel_type_id
    # We select both the pk and the value column.
    # The pk column is resolved dynamically by inspecting the result row keys.
    result = (
        get_client()
        .schema(schema)
        .table(table)
        .select("*")
        .execute()
    )
    rows = result.data or []
    if not rows:
        logger.warning(
            "_load_one_table: table=%s.%s returned 0 rows.", schema, table
        )
        return

    # Detect pk column name: the column that ends with "_id" and is not the value column
    pk_col: str | None = None
    for key in rows[0].keys():
        if key.endswith("_id") and key != column:
            pk_col = key
            break

    if pk_col is None:
        logger.warning(
            "_load_one_table: could not detect PK column in %s.%s — skipping.",
            schema, table,
        )
        return

    value_map: dict[str, int] = {}
    canonical_map: dict[str, str] = {}

    for row in rows:
        raw_val = row.get(column)
        pk_val = row.get(pk_col)
        if raw_val is None or pk_val is None:
            continue
        cleaned = clean_for_lookup(str(raw_val))
        value_map[cleaned] = int(pk_val)
        canonical_map[cleaned] = str(raw_val)

    LOOKUP_CACHE[table_path] = value_map
    _LOOKUP_CANONICAL[table_path] = canonical_map
    logger.debug(
        "_load_one_table: %s → %d entries cached.", table_path, len(value_map)
    )


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def clean_for_lookup(value: str) -> str:
    """
    Normalises a string for case-insensitive lookup matching.
    "Super-AMOLED" → "super amoled"
    "LPDDR 5X"     → "lpddr 5x"
    "f/1.8"        → "f/1.8"  (slash preserved — not treated as separator)
    """
    value = value.strip().lower()
    value = re.sub(r"[\s\-\_]+", " ", value)
    return value


def resolve_lookup_value(
    value: str,
    table_path: str,
) -> tuple[int | None, str | None, str]:
    """
    Resolves a string value to a lookup table PK using LOOKUP_CACHE.

    Resolution steps:
      1. Exact match on cleaned value
      2. Levenshtein fuzzy match (distance <= _FUZZY_THRESHOLD)
      3. not_found → caller sends value to staging

    Args:
        value:      Raw string from LLM output (e.g. "Super-AMOLED").
        table_path: "schema.table.column" key into LOOKUP_CACHE.

    Returns:
        (pk_id, corrected_value, resolution_type) where:
          - pk_id:           integer PK, or None if not found
          - corrected_value: canonical spelling from DB, or None
          - resolution_type: "exact" | "fuzzy" | "not_found"
    """
    cleaned = clean_for_lookup(str(value))
    cache = LOOKUP_CACHE.get(table_path, {})
    canonical = _LOOKUP_CANONICAL.get(table_path, {})

    # Step 1 — Exact match
    if cleaned in cache:
        return cache[cleaned], canonical.get(cleaned, value), "exact"

    # Step 2 — Fuzzy match (Levenshtein distance <= threshold)

    # Guard 1: Never fuzzy-match on fields where a wrong match is worse than staging.
    if table_path in _NO_FUZZY_TABLE_PATHS:
        return None, None, "not_found"

    # Guard 2: Short strings are too ambiguous for fuzzy matching.
    if len(cleaned) < _FUZZY_MIN_LENGTH:
        return None, None, "not_found"

    best_pk: int | None = None
    best_key: str | None = None
    best_dist = _FUZZY_THRESHOLD + 1  # exclusive upper bound

    for cached_key, pk in cache.items():
        d = levenshtein_distance(cleaned, cached_key)
        if d < best_dist:
            best_dist = d
            best_pk = pk
            best_key = cached_key

    if best_pk is not None and best_key is not None:
        return best_pk, canonical.get(best_key, best_key), "fuzzy"

    return None, None, "not_found"


# ---------------------------------------------------------------------------
# Unit stripping helpers
# ---------------------------------------------------------------------------

def _strip_numeric_suffix(value: Any) -> Any:
    """
    Strips common unit suffixes and returns the numeric portion.
    "5000mAh" → 5000   "8GB" → 8   "3nm" → 3   "f/1.8" → 1.8
    "6.7\"" → 6.7   "50MP" → 50   "120Hz" → 120   "7.5W" → 7.5

    Returns the original value unchanged if it cannot be parsed.
    """
    if not isinstance(value, str):
        return value
    # Remove trailing units: mAh, GB, MP, Hz, W, nm, ms, inch/", x (resolution sep)
    cleaned = re.sub(
        r"(?i)\s*(mah|gb|tb|mb|mp|hz|ghz|mhz|nm|ms|w|kg|g|mm|cm|in|\"|\binch\b|x)$",
        "",
        value.strip(),
    )
    # Handle "f/1.8" → 1.8
    m = re.match(r"^f/(\d+\.?\d*)$", cleaned, re.IGNORECASE)
    if m:
        return float(m.group(1))
    try:
        return int(cleaned)
    except ValueError:
        pass
    try:
        return float(cleaned)
    except ValueError:
        pass
    return value  # unchanged — not a recognised numeric pattern


def _coerce_boolean(value: Any) -> Any:
    """
    "true"/"yes"/"1" → True, "false"/"no"/"0" → False.
    Non-string values returned unchanged.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        l = value.strip().lower()
        if l in ("true", "yes", "1"):
            return True
        if l in ("false", "no", "0"):
            return False
    return value


# ---------------------------------------------------------------------------
# Range constraints — values outside these are nulled and an issue is logged
# ---------------------------------------------------------------------------
_RANGE_CONSTRAINTS: dict[str, tuple[float, float]] = {
    "charging.battery_capacity":        (500,   22000),
    "body.weight":                      (50,    500),
    "body.height":                      (50,    250),
    "body.width":                       (30,    150),
    "body.thickness":                   (3,     30),
    "displays[*].refresh_rate":         (24,    240),
    "displays[*].brightness_hbm":       (100,   9000),   # was brightness_nits (ghost)
    "displays[*].brightness_peak":      (100,   12000),   # new — modern phones reach 8000+ nits
    "displays[*].size_inch":            (3.0,   10.0),
    "camera_lenses[*].megapixels":      (0.1,   300),
    "camera_lenses[*].aperture":        (0.5,   16.0),
    # os_and_security.launch_os_version removed — field does not exist in v5 OsAndSecurityData
}

# Deterministic proprietary_charging value by brand.
# Written into normalized_json at Step 8.5 (after range validation, before null counting).
# None = brand genuinely has no proprietary charging name — field stays null.
_BRAND_CHARGING_FALLBACK: dict[str, str | None] = {
    "Motorola":   "TurboPower",
    "Samsung":    "Super Fast Charging",
    "Oneplus":    "SUPERVOOC",
    "Xiaomi":     "HyperCharge",
    "Redmi":      "HyperCharge",
    "Poco":       "HyperCharge",
    "Realme":     "SUPERVOOC",
    "Oppo":       "SuperVOOC",
    "Vivo":       "FlashCharge",
    "Iqoo":       "FlashCharge",
    "Google":     None,
    "Apple":      None,
    "Nothing":    None,
    "Honor":      "Honor SuperCharge",
}


def _check_range(field_path: str, value: Any) -> tuple[Any, str | None]:
    """
    Checks a numeric value against _RANGE_CONSTRAINTS.
    Returns (original_value, None) if in range or no constraint.
    Returns (None, issue_message) if out of range.

    Wildcard paths: checks generic key by replacing "[0]", "[1]" etc. with "[*]".
    """
    # Normalise concrete array indices → wildcard for dict lookup
    generic_path = re.sub(r"\[\d+\]", "[*]", field_path)
    constraint = _RANGE_CONSTRAINTS.get(generic_path)
    if constraint is None:
        return value, None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return value, None
    lo, hi = constraint
    if num < lo or num > hi:
        return None, (
            f"out_of_range: {value!r} is outside [{lo}, {hi}] for {field_path}"
        )
    return value, None


# ---------------------------------------------------------------------------
# Task 4.2 — Normalisation Engine
# ---------------------------------------------------------------------------

async def run_normalisation(output_id: int) -> dict:
    """
    Full normalisation pass on spec_extraction_output.partial_json.
    Produces normalized_spec_json and writes it to pipeline.normalized_spec.

    STEP ORDER:
      1. Fetch partial_json from spec_extraction_output
      2. Deep-copy → working normalized_json
      3. Discard RUN_C_CALCULATED_FIELDS if LLM included them
      4. Walk all scalar FK fields (SCALAR_FK_MAP):
           resolve_lookup_value() → exact | fuzzy | staging
      5. Walk all array FK fields (ARRAY_FK_MAP):
           resolve each element, deduplicate, sort alphabetically
      6. Unit suffix stripping on all remaining numeric-looking strings
      7. Boolean coercion ("true"/"yes"/"false"/"no")
      8. Range constraint validation → null + issue logged on violation
      9. INSERT/UPDATE pipeline.normalized_spec
     10. UPDATE normalisation_runs: completed, issue_count, finished_at

    STAGING:
        resolve_lookup_value() returns "not_found" →
        INSERT pipeline.lookup_value_staging (idempotent ON CONFLICT DO NOTHING)
        Field set to null in normalized_json.

    ISSUES list format:
        {"field": "displays[0].panel_type", "issue": "fuzzy_match (dist 1)",
         "raw_value": "Super-AMOLED", "corrected_value": "Super AMOLED"}

    Args:
        output_id:  pipeline.spec_extraction_output.output_id to normalise.

    Returns:
        {
            "success":              bool,
            "normalized_id":        int,
            "issue_count":          int,
            "remaining_nulls":      int,
            "ready_for_enrichment": bool,
            "message":              str,
        }
        Note: staging_count (values sent to lookup_value_staging) is intentionally
        excluded — the column was removed from normalisation_runs. Fetch the count
        directly from lookup_value_staging if needed.


    Raises:
        ValueError: If output_id not found.
        Re-raises all other exceptions after marking run as failed.
    """
    logger.info("run_normalisation: START output_id=%d", output_id)

    # N11 — Guard: LOOKUP_CACHE empty means build_lookup_cache() failed at startup.
    # Every FK field would go straight to staging — normalisation would produce all nulls.
    if not LOOKUP_CACHE:
        raise RuntimeError(
            "LOOKUP_CACHE is empty — build_lookup_cache() may have failed at startup. "
            "Call POST /extraction/cache/refresh before normalising."
        )

    # Step 1 — Fetch source data
    output_row = await db_call_with_retry(fetch_spec_extraction_output, output_id)
    url_registry_id: int = output_row["url_registry_id"]
    extraction_run_id: int = output_row["extraction_run_id"]

    # Create a normalisation run record
    # N7: extraction_run_id is NOT a column in normalization_runs — omitted from payload
    norm_run_id = await db_call_with_retry(
        insert_normalisation_run,
        {
            "output_id":       output_id,
            "url_registry_id": url_registry_id,
            "status":          "running",
        },
    )
    logger.info("run_normalisation: norm_run_id=%d created", norm_run_id)

    try:
        # Step 2 — Deep-copy partial_json as working dict
        partial_json: dict = output_row["partial_json"] or {}
        normalized_json: dict = copy.deepcopy(partial_json)

        issues: list[dict] = []
        staging_count = 0

        # Step 2.5 — Chipset deduplication (Change 6b)
        # Must run BEFORE gap analysis so the gap analyzer sees finalized chipset data.
        # If the chipset already exists in the DB, overwrite all chipset fields with
        # DB-canonical values and embed chipset_id so the commit orchestrator skips
        # the INSERT and links to the existing row directly.
        chipset_raw = normalized_json.get("chipset", {})
        chipset_name_raw: str | None = chipset_raw.get("chipset_name") if isinstance(chipset_raw, dict) else None
        if chipset_name_raw:
            existing_chipset = await asyncio.to_thread(fetch_chipset_by_name, chipset_name_raw)
            if existing_chipset:
                # Overwrite extracted values with DB-canonical values.
                # All fields — even those the LLM may have filled differently — are replaced.
                normalized_json["chipset"] = {
                    k: existing_chipset.get(k)
                    for k in (
                        "chipset_id",
                        "chipset_name",
                        "cpu_architecture",
                        "fabrication_node",
                        "number_of_cores",
                        "cpu_high_performance_cores",
                        "cpu_performance_cores",
                        "cpu_efficiency_cores",
                        "cpu_clock_speed",
                        "gpu_name",
                        "gpu_unit_count",
                        "gpu_unit_type",
                        "gpu_clock_speed",
                        "npu_details",
                        "npu_tops",
                    )
                }
                logger.info(
                    "run_normalisation: chipset %r found in DB (chipset_id=%s) — "
                    "overriding extracted chipset fields with DB-canonical values. "
                    "Chipset fields will not appear in gap analysis.",
                    chipset_name_raw, existing_chipset.get("chipset_id"),
                )
            else:
                logger.info(
                    "run_normalisation: chipset %r is NEW — extracted values retained. "
                    "Gap analysis will identify missing chipset fields for enrichment.",
                    chipset_name_raw,
                )

        # Step 3 — Discard RUN_C_CALCULATED_FIELDS
        normalized_json = _discard_run_c_fields(normalized_json, issues)

        # Step 4 — Scalar FK resolution
        normalized_json, scalar_issues, scalar_staging = await asyncio.to_thread(
            _resolve_scalar_fks, normalized_json, url_registry_id
        )
        issues.extend(scalar_issues)
        staging_count += scalar_staging

        # Step 5 — Array FK resolution
        normalized_json, array_issues, array_staging = await asyncio.to_thread(
            _resolve_array_fks, normalized_json, url_registry_id
        )
        issues.extend(array_issues)
        staging_count += array_staging

        # Step 6 — Unit suffix stripping
        normalized_json = _walk_and_strip_units(normalized_json)

        # Step 7 — Boolean coercion
        normalized_json = _walk_and_coerce_booleans(normalized_json)

        # Step 8 — Range constraint validation
        normalized_json, range_issues = _validate_ranges(normalized_json)
        issues.extend(range_issues)

        # Step 8.5 — Deterministic brand charging fallback
        # Writes proprietary_charging from _BRAND_CHARGING_FALLBACK if still null.
        # Runs after range validation so it never overwrites a valid extracted value.
        normalized_json = _apply_brand_charging_fallback(normalized_json)

        # BIS certification — always true for all phones in this database.
        # All phones are sold in India; BIS/MTCTE is mandatory.
        # LLM cannot reliably set this without source evidence, so normalizer enforces it.
        if "certifications" in normalized_json and isinstance(normalized_json["certifications"], dict):
            normalized_json["certifications"]["bis_certification"] = True

        # Count remaining nulls
        remaining_nulls = _count_nulls(normalized_json)
        ready_for_enrichment = remaining_nulls > 0
        ready_for_commit = remaining_nulls == 0

        # Step 9 — Persist normalized_spec_json
        # N4: Use correct column names matching SQL schema.
        # Removed: output_id, extraction_run_id, issues_json (not columns on this table).
        normalized_id = await db_call_with_retry(
            upsert_normalized_spec,
            {
                "normalization_run_id": norm_run_id,       # N4: was norm_run_id
                "url_registry_id":      url_registry_id,
                "normalized_json":       normalized_json,   # N4: was normalized_spec_json
                "remaining_null_count": remaining_nulls,   # N4: was null_field_count
                "ready_for_enrichment": ready_for_enrichment,
                "ready_for_commit":     ready_for_commit,
            },
        )

        # Step 10 — Mark run complete
        # N9: issues_found (full log) goes to the run record, not to normalized_spec_json.
        # staging_count is NOT a column in normalization_runs — staging events are
        # already captured individually in issues_found (each produces an issue entry).
        await db_call_with_retry(
            update_normalisation_run,
            norm_run_id,
            {
                "status":       "completed",
                "finished_at":  "now()",
                "issue_count":  len(issues),
                "issues_found": issues,
            },
        )

        logger.info(
            "run_normalisation: COMPLETE output_id=%d normalized_id=%d "
            "issues=%d staging=%d nulls=%d",
            output_id, normalized_id, len(issues), staging_count, remaining_nulls,
        )
        return {
            "success":              True,
            "normalized_id":        normalized_id,
            "issue_count":          len(issues),
            "staging_count":        staging_count,
            "remaining_nulls":      remaining_nulls,
            "ready_for_enrichment": ready_for_enrichment,
            "message": (
                f"Normalisation complete. {len(issues)} corrections, "
                f"{staging_count} sent to staging, {remaining_nulls} nulls remain."
            ),
        }

    except Exception as exc:
        logger.exception(
            "run_normalisation: FAILED norm_run_id=%d: %s", norm_run_id, exc
        )
        try:
            await asyncio.to_thread(
                update_normalisation_run,
                norm_run_id,
                {"status": "failed", "error_message": str(exc), "finished_at": "now()"},
            )
        except Exception as db_exc:
            logger.error(
                "run_normalisation: also failed to mark norm_run_id=%d failed: %s",
                norm_run_id, db_exc,
            )
        raise


# ---------------------------------------------------------------------------
# Internal normalisation helpers
# ---------------------------------------------------------------------------

def _discard_run_c_fields(data: dict, issues: list[dict]) -> dict:
    """
    Removes any fields from data that are in RUN_C_CALCULATED_FIELDS.
    These are never extracted by Run A — they are computed post-commit.
    Mutates data in place and returns it.
    """
    for field_path in RUN_C_CALCULATED_FIELDS:
        # field_path may contain [*] — resolve to any concrete index
        pattern = re.compile(
            r"^" + re.escape(field_path).replace(r"\[\*\]", r"\[\d+\]") + r"$"
        )
        removed = _remove_matching_paths(data, pattern, "")
        for removed_path in removed:
            issues.append({
                "field":   removed_path,
                "issue":   "run_c_field_discarded",
                "raw_value": None,
                "corrected_value": None,
            })
    return data


def _remove_matching_paths(data: Any, pattern: re.Pattern, prefix: str) -> list[str]:
    """
    Recursively removes dict keys whose full dotted path matches pattern.
    Returns list of removed paths.
    """
    removed: list[str] = []
    if isinstance(data, dict):
        keys_to_delete = []
        for key, val in data.items():
            full_path = f"{prefix}.{key}" if prefix else key
            if pattern.match(full_path):
                keys_to_delete.append(key)
                removed.append(full_path)
            else:
                removed.extend(_remove_matching_paths(val, pattern, full_path))
        for key in keys_to_delete:
            del data[key]
    elif isinstance(data, list):
        for i, item in enumerate(data):
            full_path = f"{prefix}[{i}]"
            removed.extend(_remove_matching_paths(item, pattern, full_path))
    return removed


def _resolve_scalar_fks(
    data: dict,
    url_registry_id: int,
) -> tuple[dict, list[dict], int]:
    """
    Resolves all SCALAR_FK_MAP fields in data to their lookup PKs.
    Returns (mutated_data, issues, staging_count).

    Change 3a/3b: Before calling resolve_lookup_value(), applies alias rewrites
    for SIM configuration (_SIM_CONFIGURATION_ALIASES) and autofocus type
    (_AUTOFOCUS_TYPE_ALIASES) so known variants never reach staging.
    """
    issues: list[dict] = []
    staging_count = 0

    for field_path, table_path in SCALAR_FK_MAP.items():
        # Expand [*] wildcard to concrete indices
        concrete_paths = _expand_wildcard_path(data, field_path)

        for concrete_path in concrete_paths:
            raw_value = _get_nested(data, concrete_path)
            if raw_value is None:
                continue  # null field — nothing to resolve

            # Change 3a: apply SIM config alias BEFORE lookup
            lookup_value = str(raw_value)
            if "sim_configuration" in concrete_path:
                lookup_value = _SIM_CONFIGURATION_ALIASES.get(lookup_value, lookup_value)

            # Change 3b: apply autofocus alias BEFORE lookup
            if "autofocus_type" in concrete_path:
                lookup_value = _AUTOFOCUS_TYPE_ALIASES.get(lookup_value, lookup_value)

            pk, corrected, resolution_type = resolve_lookup_value(
                lookup_value, table_path
            )

            if resolution_type == "exact":
                _set_nested(data, concrete_path, pk)

            elif resolution_type == "fuzzy":
                _set_nested(data, concrete_path, pk)
                issues.append({
                    "field":           concrete_path,
                    "issue":           f"fuzzy_match (dist {levenshtein_distance(clean_for_lookup(str(raw_value)), clean_for_lookup(corrected or ''))})",
                    "raw_value":       raw_value,
                    "corrected_value": corrected,
                })
                logger.debug(
                    "_resolve_scalar_fks: fuzzy %r → %r (pk=%s) at %s",
                    raw_value, corrected, pk, concrete_path,
                )

            else:  # not_found
                _set_nested(data, concrete_path, None)
                _send_to_staging(raw_value, table_path, concrete_path, url_registry_id)
                staging_count += 1
                issues.append({
                    "field":           concrete_path,
                    "issue":           "not_found_sent_to_staging",
                    "raw_value":       raw_value,
                    "corrected_value": None,
                })

    return data, issues, staging_count


def _resolve_array_fks(
    data: dict,
    url_registry_id: int,
) -> tuple[dict, list[dict], int]:
    """
    Resolves all ARRAY_FK_MAP fields in data.
    Each array element is a string resolved to its PK.
    Deduplicates by PK and sorts alphabetically by raw value after resolution.
    Returns (mutated_data, issues, staging_count).

    Change 3c: Skips any wifi_technology value found in _WIFI_TECHNOLOGIES_EXCLUDE
    so that "Wi-Fi Hotspot" etc. are silently dropped rather than sent to staging.
    """
    issues: list[dict] = []
    staging_count = 0

    for field_path, table_path in ARRAY_FK_MAP.items():
        concrete_paths = _expand_wildcard_path(data, field_path)

        for concrete_path in concrete_paths:
            raw_array = _get_nested(data, concrete_path)
            if not isinstance(raw_array, list) or not raw_array:
                continue

            resolved_pairs: list[tuple[str, int]] = []  # N10: (canonical_str, pk)
            seen_pks: set[int] = set()

            for raw_value in raw_array:
                if raw_value is None:
                    continue

                # Change 3c: silently drop Wi-Fi Hotspot from junction table
                if "wifi_technologies" in concrete_path and str(raw_value) in _WIFI_TECHNOLOGIES_EXCLUDE:
                    logger.debug(
                        "_resolve_array_fks: skipping excluded wifi tech %r at %s",
                        raw_value, concrete_path,
                    )
                    continue

                # Fix: enforce correct band prefix before fuzzy matching
                # so "28" becomes "B28" (4G) or "n28" (5G) before lookup.
                normalized_value = _normalize_band_value(str(raw_value), field_path)
                pk, corrected, resolution_type = resolve_lookup_value(
                    normalized_value, table_path
                )

                if resolution_type == "exact":
                    if pk not in seen_pks:
                        resolved_pairs.append((corrected or str(raw_value), pk))
                        seen_pks.add(pk)

                elif resolution_type == "fuzzy":
                    if pk not in seen_pks:
                        resolved_pairs.append((corrected or str(raw_value), pk))
                        seen_pks.add(pk)
                    issues.append({
                        "field":           concrete_path,
                        "issue":           f"fuzzy_match (dist {levenshtein_distance(clean_for_lookup(str(raw_value)), clean_for_lookup(corrected or ''))})",
                        "raw_value":       raw_value,
                        "corrected_value": corrected,
                    })

                else:  # not_found
                    _send_to_staging(raw_value, table_path, concrete_path, url_registry_id)
                    staging_count += 1
                    issues.append({
                        "field":           concrete_path,
                        "issue":           "not_found_sent_to_staging",
                        "raw_value":       raw_value,
                        "corrected_value": None,
                    })

            # N10: Sort alphabetically by canonical string, then extract PK list
            resolved_pairs.sort(key=lambda x: x[0].lower())
            _set_nested(data, concrete_path, [pk for _, pk in resolved_pairs])

    return data, issues, staging_count


def _send_to_staging(
    raw_value: Any,
    table_path: str,
    field_path: str,
    url_registry_id: int,
) -> None:
    """
    Inserts a value into pipeline.lookup_value_staging for human review.
    Silently ignores DB errors (staging is best-effort).

    N6 fix: column names corrected to match SQL schema:
      raw_value   → extracted_value
      table_path  → target_lookup_table
      source_stage added (NOT NULL, no default).

    #8 fix: After a successful staging insert, schedule a background LOOKUP_CACHE
    refresh so that if an admin resolves the staging row and adds a new lookup row
    during the same session, the next normalisation run picks it up automatically
    without requiring a manual POST /extraction/cache/refresh call.
    The refresh is fire-and-forget (asyncio.Task) — failures are logged but not raised.
    """
    try:
        insert_lookup_value_staging({
            "extracted_value":     str(raw_value),       # N6: was raw_value
            "target_lookup_table": table_path,           # N6: was table_path
            "field_path":          field_path,
            "url_registry_id":     url_registry_id,
            "source_stage":        "normalization",      # N6: required NOT NULL column
        })
        # S2-P1-6: background cache refresh removed — it caused N concurrent
        # build_lookup_cache() calls (one per staging insert), exhausting the
        # Supabase connection pool on phones with many unknown field values.
        # Cache refresh is handled by the admin workflow:
        #   admin resolves staging entry → POST /extraction/cache/refresh
    except Exception as exc:
        logger.warning(
            "_send_to_staging: failed to insert staging row "
            "value=%r table=%r field=%r: %s",
            raw_value, table_path, field_path, exc,
        )


def _walk_and_strip_units(data: Any) -> Any:
    """
    Recursively walks data. For every string value that looks like a
    numeric-with-unit, strips the unit and returns the number.
    Dicts and lists are traversed; all other types are unchanged.
    """
    if isinstance(data, dict):
        return {k: _walk_and_strip_units(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_walk_and_strip_units(v) for v in data]
    if isinstance(data, str):
        return _strip_numeric_suffix(data)
    return data


def _walk_and_coerce_booleans(data: Any) -> Any:
    """Recursively coerces "true"/"yes"/"false"/"no" strings to Python bools."""
    if isinstance(data, dict):
        return {k: _walk_and_coerce_booleans(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_walk_and_coerce_booleans(v) for v in data]
    if isinstance(data, str):
        return _coerce_boolean(data)
    return data


def _validate_ranges(data: Any, prefix: str = "") -> tuple[Any, list[dict]]:
    """
    Recursively walks data and validates numeric values against _RANGE_CONSTRAINTS.
    Out-of-range values → None + issue logged.
    Returns (mutated_data, issues).
    """
    issues: list[dict] = []
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            full_path = f"{prefix}.{k}" if prefix else k
            new_v, sub_issues = _validate_ranges(v, full_path)
            issues.extend(sub_issues)
            result[k] = new_v  # N8: new_v is already None on range violation; no double-append
        return result, issues
    if isinstance(data, list):
        new_list = []
        for i, item in enumerate(data):
            new_item, sub_issues = _validate_ranges(item, f"{prefix}[{i}]")
            new_list.append(new_item)
            issues.extend(sub_issues)
        return new_list, issues
    # Leaf value check
    nulled, issue_msg = _check_range(prefix, data)
    if issue_msg:
        issues.append({"field": prefix, "issue": issue_msg, "raw_value": data, "corrected_value": None})
        return None, issues
    return data, issues


def _count_nulls(data: Any) -> int:
    """Recursively counts None values in a nested dict/list structure."""
    if data is None:
        return 1
    if isinstance(data, dict):
        return sum(_count_nulls(v) for v in data.values())
    if isinstance(data, list):
        return sum(_count_nulls(v) for v in data)
    return 0


# ---------------------------------------------------------------------------
# Path navigation helpers  ([*] wildcard expansion + get/set nested)
# ---------------------------------------------------------------------------

def _expand_wildcard_path(data: dict, field_path: str) -> list[str]:
    """
    Expands a field_path containing [*] to all concrete paths that exist in data.
    "variants[*].ram_type" → ["variants[0].ram_type", "variants[1].ram_type"]
    Paths with no [*] are returned as-is (wrapped in a list).
    """
    if "[*]" not in field_path:
        return [field_path]

    # Split at first [*]
    before, _, after = field_path.partition("[*]")
    # Navigate to the array
    arr = _get_nested(data, before.rstrip("."))
    if not isinstance(arr, list):
        return []

    results: list[str] = []
    for i in range(len(arr)):
        inlined = f"{before.rstrip('.')}[{i}]{'.' + after.lstrip('.') if after else ''}"
        if "[*]" in inlined:
            # More wildcards remain — recurse
            results.extend(_expand_wildcard_path(data, inlined))
        else:
            results.append(inlined)
    return results


def _get_nested(data: Any, path: str) -> Any:
    """
    Gets a value from a nested dict/list using dot + bracket notation.
    "variants[0].ram_type" → data["variants"][0]["ram_type"]
    Returns None if any intermediate key/index is missing.
    """
    if not path:
        return data
    tokens = _tokenize_path(path)
    current = data
    for token in tokens:
        if current is None:
            return None
        if isinstance(token, int):
            if isinstance(current, list) and 0 <= token < len(current):
                current = current[token]
            else:
                return None
        else:
            if isinstance(current, dict):
                current = current.get(token)
            else:
                return None
    return current


def _set_nested(data: Any, path: str, value: Any) -> None:
    """
    Sets a value in a nested dict/list using dot + bracket notation.
    Creates intermediate dicts if missing. Does nothing on index-out-of-range.
    """
    tokens = _tokenize_path(path)
    current = data
    for token in tokens[:-1]:
        if isinstance(token, int):
            if isinstance(current, list) and 0 <= token < len(current):
                current = current[token]
            else:
                return
        else:
            if isinstance(current, dict):
                if token not in current:
                    current[token] = {}
                current = current[token]
            else:
                return
    last = tokens[-1]
    if isinstance(last, int):
        if isinstance(current, list) and 0 <= last < len(current):
            current[last] = value
    else:
        if isinstance(current, dict):
            current[last] = value


def _tokenize_path(path: str) -> list[str | int]:
    """
    Tokenises a dot+bracket path string into a list of keys/indices.
    "variants[0].ram_type" → ["variants", 0, "ram_type"]
    "displays[1].features[0]" → ["displays", 1, "features", 0]
    """
    tokens: list[str | int] = []
    # Split on "." and "[N]"
    parts = re.split(r"\.|\[(\d+)\]", path)
    for part in parts:
        if part is None or part == "":
            continue
        try:
            tokens.append(int(part))
        except ValueError:
            tokens.append(part)
    return tokens
