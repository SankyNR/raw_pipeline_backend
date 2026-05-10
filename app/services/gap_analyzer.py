"""
Phase 5 — Gap Analysis

Task 5.1 — Missing Fields Detector (detect_missing_fields)

What this does:
    Walks the normalized_spec_json from normalized_spec_json to find every null or
    empty-array field. For each null, creates a pipeline.missing_fields_log row with:
      - field_path          — dot + bracket notation path (e.g. "displays[0].panel_type")
      - missing_type        — "type_a" (scalar enrichment) or "type_b" (junction table gap)
      - priority            — "high" | "medium" | "skip" from FIELD_PRIORITY_MAP
      - preferred_site_hint — domain to target for grounded search (from FIELD_SITE_HINTS,
                              or dynamic chipset vendor resolution)

Type assignment:
    - field_path (wildcards resolved) in JUNCTION_TABLE_FIELDS → type_b
      IF phone has status='stored_mainDB' in url_registry: also creates
      type_b_gap_candidates rows via DB gap query.
    - all else → type_a

Site hint resolution:
    - FIELD_SITE_HINTS[generic_path] → static hint
    - chipset.npu_tops → resolved dynamically from chipset_name:
        Snapdragon → qualcomm.com
        Dimensity  → mediatek.com
        Apple      → apple.com
        default    → None

Design:
    - Supabase-py is synchronous; DB calls wrapped in asyncio.to_thread().
    - Wildcards in field paths resolved to concrete indices before logging.
    - Idempotent: ON CONFLICT (normalized_id, field_path) DO UPDATE — existing
      missing_field_id is returned on re-runs (no duplicate rows created).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from app.config.field_mapping import (
    DEFAULT_FIELD_PRIORITY,
    FIELD_PRIORITY_MAP,
    FIELD_SITE_HINTS,
    JUNCTION_TABLE_FIELDS,
)
from app.repositories.extraction_repository import (
    fetch_normalized_spec,
    fetch_url_registry_status,
    insert_missing_field_log,
    insert_type_b_gap_candidates,
    fetch_junction_table_existing_pks,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chipset vendor → site hint
# ---------------------------------------------------------------------------

_CHIPSET_VENDOR_HINTS: dict[str, str] = {
    "snapdragon": "qualcomm.com",
    "dimensity":  "mediatek.com",
    "apple":      "apple.com",
    "exynos":     "samsung.com",       # G6: Samsung Exynos
    "tensor":     "store.google.com",  # G6: Google Tensor (Pixel phones)
}



def _resolve_chipset_site_hint(chipset_name: str | None) -> str | None:
    """
    Resolves chipset.npu_tops site hint dynamically from chipset_name.
    "Snapdragon 8 Gen 3" → "qualcomm.com"
    """
    if not chipset_name:
        return None
    lower = chipset_name.lower()
    for vendor, hint in _CHIPSET_VENDOR_HINTS.items():
        if vendor in lower:
            return hint
    return None


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def _generic_path(concrete_path: str) -> str:
    """
    Converts a concrete array-index path to a wildcard path for map lookups.
    "displays[0].panel_type" → "displays[*].panel_type"
    """
    return re.sub(r"\[\d+\]", "[*]", concrete_path)


def _walk_nulls(data: Any, prefix: str = "") -> list[str]:
    """
    Recursively walks a nested dict/list. Returns a list of concrete dot-bracket
    paths for every null or empty-array leaf.

    Skips None branches mid-path (can't descend into null).
    Empty arrays [] are also recorded as gaps (junction table with 0 members).
    Non-null atomic values (int, float, bool, str) are NOT returned.
    """
    gaps: list[str] = []

    if isinstance(data, dict):
        for k, v in data.items():
            full_path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                gaps.extend(_walk_nulls(v, full_path))
            elif v is None:
                gaps.append(full_path)
            # non-null scalar → not a gap
    elif isinstance(data, list):
        if not data:
            # empty array → gap (only record parent path, not [0], [1]…)
            gaps.append(prefix)
        else:
            for i, item in enumerate(data):
                gaps.extend(_walk_nulls(item, f"{prefix}[{i}]"))

    return gaps


# ---------------------------------------------------------------------------
# Task 5.1 — detect_missing_fields (main public API)
# ---------------------------------------------------------------------------

async def detect_missing_fields(normalized_id: int) -> list[int]:
    """
    Detects null / empty-array fields in normalized_spec_json.
    Creates pipeline.missing_fields_log rows for each gap.
    For type_b fields with status='stored_mainDB': also creates
    type_b_gap_candidates via DB gap query.

    Args:
        normalized_id:  pipeline.normalized_spec_json.normalized_id

    Returns:
        List of missing_field_id values created.

    Raises:
        ValueError: If normalized_id not found.
    """
    logger.info("detect_missing_fields: START normalized_id=%d", normalized_id)

    # Fetch source data
    norm_row = await asyncio.to_thread(fetch_normalized_spec, normalized_id)
    normalized_json: dict = norm_row["normalized_json"] or {}
    url_registry_id: int = norm_row["url_registry_id"]

    # L7.3 Fix 5: extract brand for deterministic charging fill check
    brand: str = (normalized_json.get("brand") or {}).get("brand_name") or ""

    # Fetch url_registry status + chipset_name for dynamic hints
    registry_row = await asyncio.to_thread(fetch_url_registry_status, url_registry_id)
    registry_status: str = registry_row.get("status", "")
    chipset_name: str | None = (
        (normalized_json.get("chipset") or {}).get("chipset_name")
    )

    phone_is_stored = (registry_status == "stored_mainDB")

    # Walk the normalized JSON to find all null / empty-list paths
    gap_paths: list[str] = _walk_nulls(normalized_json)
    logger.info(
        "detect_missing_fields: normalized_id=%d found %d gaps",
        normalized_id, len(gap_paths),
    )

    created_ids: list[int] = []

    for concrete_path in gap_paths:
        generic = _generic_path(concrete_path)

        # L7.3 Fix 5 — Dependency filter: stylus_features requires has_stylus=True ---
        if generic == "body.stylus_features":
            has_stylus = (normalized_json.get("body") or {}).get("has_stylus")
            if has_stylus is False:
                continue  # Phone confirmed no stylus — stylus_features will always be null

        # L7.3 Fix 5 — Dependency filter: wireless charging sub-fields ---
        _WIRELESS_CHARGING_DEPS = {
            "charging.wireless_charging_power",
            "charging.wireless_charging_standard",
            "charging.reverse_wireless_charging",
            "charging.reverse_wireless_charging_power",
        }
        if generic in _WIRELESS_CHARGING_DEPS:
            wireless = (normalized_json.get("charging") or {}).get("wireless_charging")
            if wireless is False:
                continue  # Phone confirmed no wireless charging — sub-fields are N/A

        # --- Field type ---
        is_junction = generic in JUNCTION_TABLE_FIELDS
        field_type = "type_b" if is_junction else "type_a"

        # --- Priority ---
        priority = FIELD_PRIORITY_MAP.get(generic, DEFAULT_FIELD_PRIORITY)

        # G5: skip fields the pipeline has explicitly decided not to enrich
        if priority == "skip":
            continue

        # --- Site hint ---
        if generic == "chipset.npu_tops":
            site_hint = _resolve_chipset_site_hint(chipset_name)
        else:
            site_hint = FIELD_SITE_HINTS.get(generic)

        # --- Insert missing_fields_log row (idempotent) ---
        # G1: correct column names from SQL schema:
        #   field_type  → missing_type
        #   site_hint   → preferred_site_hint
        #   generic_path removed  (no such column; derivable from field_path at read time)
        #   status       removed  (no such column)
        log_payload = {
            "normalized_id":         normalized_id,
            "url_registry_id":       url_registry_id,
            "field_path":            concrete_path,
            "missing_type":          field_type,        # G1
            "priority":              priority,
            "preferred_site_hint":   site_hint,         # G1
            "query_template_override": None,            # E3 fix: column exists, always set explicitly
        }

        try:
            missing_field_id = await asyncio.to_thread(
                insert_missing_field_log, log_payload
            )
            if missing_field_id is not None:
                created_ids.append(missing_field_id)
        except Exception as exc:
            logger.error(
                "detect_missing_fields: failed to log gap %r for normalized_id=%d: %s",
                concrete_path, normalized_id, exc,
            )
            continue

        # --- Type B gap candidates (only if phone already committed to mainDB) ---
        if field_type == "type_b" and phone_is_stored and missing_field_id is not None:
            await _create_type_b_candidates(
                missing_field_id=missing_field_id,
                url_registry_id=url_registry_id,
                generic_path=generic,
            )

    logger.info(
        "detect_missing_fields: COMPLETE normalized_id=%d created %d log rows",
        normalized_id, len(created_ids),
    )
    return created_ids


# ---------------------------------------------------------------------------
# Internal: Type B gap candidate creation
# ---------------------------------------------------------------------------

async def _create_type_b_candidates(
    missing_field_id: int,
    url_registry_id: int,
    generic_path: str,
) -> None:
    """
    For Type B (junction table) gaps on phones already committed to mainDB:
    Queries what PKs already exist for this phone in the junction table,
    then creates type_b_gap_candidates rows so the enrichment layer can
    verify each candidate against the specific phone.

    This is a best-effort step — errors are logged and swallowed.
    """
    table_path = JUNCTION_TABLE_FIELDS.get(generic_path)
    if not table_path:
        return

    try:
        existing_rows: list[dict] = await asyncio.to_thread(
            fetch_junction_table_existing_pks,
            url_registry_id,
            generic_path,
            table_path,
        )
        if not existing_rows:
            return

        # G2: correct column names matching SQL schema for type_b_gap_candidates:
        #   normalized_id  removed (not a column)
        #   generic_path   → lookup_table
        #   lookup_pk      → lookup_row_id
        #   source         → inclusion_reason
        #   candidate_value added (canonical string for the PK)
        candidates = [
            {
                "missing_field_id": missing_field_id,
                "url_registry_id":  url_registry_id,
                "candidate_value":  row["canonical_value"],   # G2
                "lookup_table":     table_path,               # G2
                "lookup_row_id":    row["pk"],                 # G2
                "inclusion_reason": "existing_maindb",        # G2
            }
            for row in existing_rows
        ]
        await asyncio.to_thread(insert_type_b_gap_candidates, candidates)
        logger.debug(
            "_create_type_b_candidates: field=%r %d candidates inserted",
            generic_path, len(candidates),
        )

    except Exception as exc:
        logger.warning(
            "_create_type_b_candidates: failed for field=%r missing_field_id=%d: %s",
            generic_path, missing_field_id, exc,
        )
