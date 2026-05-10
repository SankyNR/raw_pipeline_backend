"""
Phase 7 — Conflict Resolution

Task 7.1 — detect_and_resolve_conflicts
Task 7.2 — build_final_merged_json

What this does:
    1. Compares each normalized_json (Run A) field value with the selected
       enrichment candidate for the same field.
    2. Classifies each comparison as: fill, concordant, auto-resolved, or flagged.
    3. Inserts a merge_conflict_log row for every genuine conflict.
    4. Merges normalized_json with fills and auto-resolved winners into
       final_merged_json — the commit-ready artifact.

A CONFLICT = both Run A AND enrichment have a non-null value AND they differ
             (after type coercion).
A FILL = Run A is null AND enrichment has a non-null value (no conflict row needed).
CONCORDANT = both have the same non-null value after coercion (no conflict row).

AUTO-RESOLUTION ORDER (applied when conflict detected):
    Rule 1: Enrichment source_tier='oem_official' AND enrichment confidence >= 0.85
            AND Run A source evidence is NOT oem_official
            → kept_enrichment, resolved_by='auto_source_priority'
    Rule 2: Run A raw_id is in oem_raw_ids AND enrichment source_tier is not oem_official
            → kept_run_a, resolved_by='auto_source_priority'
    Rule 3: Absolute confidence delta >= 0.20 between enrichment and implied Run A conf
            → keep higher confidence value, resolved_by='auto_confidence'
    Rule 4: All else → resolution='flagged', resolved_by=None (admin must decide)

Fixes applied (C1–C11):
    C1:  fetch_spec_extraction_output renamed → fetch_latest_spec_output_for_phone
    C2:  _fetch_conflict_resolution_map moved to repo (fetch_conflict_resolution_map)
    C3:  delete_conflict_log_for_phone called at start for idempotency
    C4:  _coerce_to_match applied before equality comparison
    C5:  oem_raw_ids fetched via fetch_oem_raw_ids_for_phone; Rule 2 can now fire
    C7:  FILL pass uses ALL candidates (not just is_selected=TRUE)
    C8:  Secondary array indices (>=1) demoted to warnings in validation
    C9:  _set_at_path logs warning on silent fail
    C10: import copy at module top level
"""

from __future__ import annotations

import asyncio
import copy
import logging
import re
from datetime import datetime, timezone
from typing import Any

from app.repositories.extraction_repository import (
    delete_conflict_log_for_phone,
    fetch_all_enrichment_candidates_for_run,
    fetch_conflict_resolution_map,
    fetch_flagged_conflict_count,
    fetch_latest_spec_output_for_phone,
    fetch_normalized_spec,
    fetch_oem_raw_ids_for_phone,
    fetch_pending_staging_count,
    fetch_selected_enrichment_candidates,
    insert_merge_conflict,
    upsert_final_merged_json,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Implied Run A confidence heuristics (no per-field confidence stored)
# ---------------------------------------------------------------------------

_IMPLIED_CONF_OEM:        float = 0.90   # scraped from OEM official source
_IMPLIED_CONF_SCRAPED:    float = 0.80   # scraped from aggregator/unknown source
_IMPLIED_CONF_TRANSCRIPT: float = 0.75   # YouTube transcript evidence
_IMPLIED_CONF_NO_EVIDENCE: float = 0.65  # value present but no evidence entry

# Confidence threshold for Rule 1 / Rule 2 OEM auto-resolution
_OEM_AUTO_THRESHOLD: float = 0.85

# Confidence delta threshold for Rule 3 auto-resolution
_DELTA_THRESHOLD: float = 0.20

# ---------------------------------------------------------------------------
# Top-level spec keys — used in Phase 7.5 schema structure check
# ---------------------------------------------------------------------------

KNOWN_SPEC_TOP_LEVEL_KEYS: frozenset[str] = frozenset({
    # v5 schema: 'basic' split into 'brand' + 'phone_identity'
    # C4 fix: 'cameras' split into 'camera_overview' + 'camera_lenses'
    #         'os_software' + 'security' merged into 'os_and_security'
    #         'ai' renamed to 'ai_capabilities'
    # C5 fix: 'video_capabilities' added — extracted by spec_template.yaml
    #         but has no mobile_specs destination table; stripped before commit.
    "brand",
    "phone_identity",
    "variants",
    "body",
    "displays",
    "chipset",
    "camera_overview",
    "camera_lenses",
    "charging",
    "network",
    "os_and_security",
    "connectivity",
    "audio",
    "sensors",
    "ai_capabilities",
    "certifications",
    "extra_features",
    "in_the_box",
    "video_capabilities",
})

# ---------------------------------------------------------------------------
# Required field checks — (dotted path, human label) — hard blocks
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: list[tuple[str, str]] = [
    ("phone_identity.model_name",  "phone_identity.model_name"),  # v5: was basic.model_name
    ("variants[0].ram_capacity",   "variants[0].ram_capacity"),
    ("displays[0].panel_type",     "displays[0].panel_type"),
]

# Numeric field paths (array-wildcard notation) — type-correctness check
# S2-P1-1 fix: all ghost v4 names removed, missing v5 fields added.
NUMERIC_FIELD_PATTERNS: list[str] = [
    # Displays
    "displays[*].size_inch",
    "displays[*].resolution_height_px",
    "displays[*].resolution_width_px",
    "displays[*].refresh_rate",            # was: refresh_rate_max_hz (ghost)
    "displays[*].brightness_hbm",          # was: brightness_nits (ghost)
    "displays[*].brightness_peak",
    "displays[*].pwm_frequency",
    "displays[*].screen_to_body_ratio",
    # Charging
    "charging.battery_capacity",
    "charging.charging_power",             # was: max_charging_speed_watt (ghost)
    "charging.wireless_charging_power",    # was: wireless_charging_speed_watt (ghost)
    # Camera lenses
    "camera_lenses[*].megapixels",
    "camera_lenses[*].aperture",
    "camera_lenses[*].sensor_size_denominator",
    "camera_lenses[*].pixel_size",
    "camera_lenses[*].fov",
    "camera_lenses[*].focal_length",
    # Chipset
    "chipset.fabrication_node",
    "chipset.cpu_clock_speed",
    "chipset.gpu_clock_speed",
    "chipset.npu_tops",
    # Variants
    "variants[*].ram_capacity",
    "variants[*].storage_capacity",
    "variants[*].launch_price",
    # Certifications
    "certifications.sar_head",
    "certifications.sar_body",
    # Body
    "body.height",
    "body.width",
    "body.thickness",
    "body.weight",
]

# Boolean field paths (array-wildcard notation) — type-correctness check
# S2-P1-1 fix: ghost paths removed, missing v5 boolean fields added.
BOOLEAN_FIELD_PATTERNS: list[str] = [
    # Connectivity
    "connectivity.nfc",
    "connectivity.uwb",
    "connectivity.ir_blaster",
    "connectivity.wifi_hotspot",
    # Network
    "network.esim_support",
    "network.volte",
    "network.vo5g",
    "network.vowifi",              # was: network.vo_wifi (typo fixed)
    # Charging
    "charging.wireless_charging",
    "charging.reverse_wireless_charging",  # was: reverse_charging (ghost)
    "charging.charger_in_box",
    # was: charging.pd_support        → removed: field not in ChargingData
    # was: connectivity.hdmi          → removed: not a field in ConnectivityData
    # was: connectivity.usb_otg       → removed: string in usb_features list, not bool
    # Body
    "body.has_stylus",
    # Variants
    "variants[*].expandable_storage",
    "variants[*].virtual_ram_availability",
    # Camera
    "camera_lenses[*].is_macro_capable",
    # Certifications
    "certifications.bis_certification",
    "certifications.widevine_support",
]

# Secondary array index threshold for C8: indices >= this are warnings, not errors
_SECONDARY_ARRAY_INDEX_THRESHOLD = 1


# ---------------------------------------------------------------------------
# Path utility helpers
# ---------------------------------------------------------------------------

def _parse_path(path: str) -> list[str | int]:
    """
    Parses a field path string into a list of keys/indices.
    'displays[0].panel_type' → ['displays', 0, 'panel_type']
    'basic.model_name'       → ['basic', 'model_name']
    """
    parts: list[str | int] = []
    for segment in re.split(r"\.", path):
        m = re.match(r"^(\w+)\[(\d+)\]$", segment)
        if m:
            parts.append(m.group(1))
            parts.append(int(m.group(2)))
        else:
            parts.append(segment)
    return parts


def _get_at_path(obj: Any, path: str) -> Any:
    """
    Reads a value from a nested dict/list using a dotted/bracket path.
    Returns None if any segment is missing or index is out of range.
    """
    parts = _parse_path(path)
    current = obj
    for part in parts:
        if current is None:
            return None
        if isinstance(part, int):
            if isinstance(current, list) and len(current) > part:
                current = current[part]
            else:
                return None
        else:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
    return current


def _set_at_path(obj: dict, path: str, value: Any) -> None:
    """
    Writes a value into a nested dict/list using a dotted/bracket path.
    Creates intermediate dicts as needed. Does NOT create missing list indices —
    the outer dict must already contain the array.

    C9: Logs a warning when navigation fails to make silent data loss visible.
    """
    parts = _parse_path(path)
    current = obj
    for i, part in enumerate(parts[:-1]):
        next_part = parts[i + 1]
        if isinstance(part, int):
            if isinstance(current, list) and len(current) > part:
                current = current[part]
            else:
                logger.warning(
                    "_set_at_path: list index %d out of range at segment %d of path=%r "
                    "— merge skipped for this field.",
                    part, i, path,
                )
                return
        else:
            if isinstance(current, dict):
                if part not in current:
                    current[part] = {} if not isinstance(next_part, int) else []
                current = current[part]
            else:
                logger.warning(
                    "_set_at_path: expected dict at segment %r (index %d) of path=%r, "
                    "got %s — merge skipped.",
                    part, i, path, type(current).__name__,
                )
                return
    last = parts[-1]
    if isinstance(last, int):
        if isinstance(current, list) and len(current) > last:
            current[last] = value
        else:
            logger.warning(
                "_set_at_path: list index %d out of range at final segment of path=%r "
                "— merge skipped.",
                last, path,
            )
    elif isinstance(current, dict):
        current[last] = value


def _count_nulls(obj: Any) -> int:
    """
    Recursively counts None values in a nested dict/list structure.
    Empty lists do not count as nulls (they are valid empty arrays).
    """
    if obj is None:
        return 1
    if isinstance(obj, dict):
        return sum(_count_nulls(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(_count_nulls(v) for v in obj)
    return 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# C4: Type coercion for conflict comparison
# ---------------------------------------------------------------------------

def _coerce_to_match(run_a_val: Any, enr_val: Any) -> Any:
    """
    Coerces enr_val to match run_a_val's Python type before comparison.

    Problem: JSONB extracted_value often comes back as a string ("5000", "true")
    while Run A's normalized value is a proper Python type (int 5000, bool True).
    Without coercion, str != int produces false conflicts.

    Coercion rules:
        run_a_val is bool → parse "true"/"1"/"yes" → True, else → False
        run_a_val is int  → int(float(str(enr_val)))  (handles "5000" and "5000.0")
        run_a_val is float → float(str(enr_val))
        all others → return enr_val unchanged

    Returns enr_val unchanged on any conversion error.
    """
    if run_a_val is None or enr_val is None:
        return enr_val
    try:
        if isinstance(run_a_val, bool):
            return str(enr_val).lower().strip() in ("true", "1", "yes")
        if isinstance(run_a_val, int):
            return int(float(str(enr_val)))
        if isinstance(run_a_val, float):
            return float(str(enr_val))
    except (ValueError, TypeError):
        pass
    return enr_val


# ---------------------------------------------------------------------------
# C5: Per-field Run A source tier inference (with OEM raw_id set)
# ---------------------------------------------------------------------------

def _infer_run_a_source_tier(
    field_path: str,
    evidence_json: dict,
    oem_raw_ids: set[int],
) -> str:
    """
    Infers the source tier of a Run A value from evidence_json and the
    pre-fetched set of OEM official raw_ids.

    evidence_json structure (per field):
        {"source_type": "scraped", "raw_id": 15, "evidence": "..."}
        {"source_type": "transcript", "raw_transcript_id": 7, "evidence": "..."}

    Resolution:
        source_type="scraped" AND raw_id in oem_raw_ids → 'oem_official'
        source_type="scraped" AND raw_id NOT in oem_raw_ids → 'trusted_aggregator'
        source_type="transcript"                            → 'transcript'
        Missing evidence entry                              → 'unknown'

    C5 fix: oem_raw_ids is now populated from raw_scraped_data site_name LIKE
    '%_official', allowing Rule 2 to correctly fire when Run A drew from OEM.
    """
    entry = evidence_json.get(field_path)
    if not entry:
        return "unknown"
    source_type = entry.get("source_type", "")
    if source_type == "transcript":
        return "transcript"
    if source_type == "scraped":
        raw_id = entry.get("raw_id")
        if raw_id is not None and raw_id in oem_raw_ids:
            return "oem_official"
        return "trusted_aggregator"
    return "unknown"


def _infer_run_a_confidence(
    field_path: str,
    evidence_json: dict,
    oem_raw_ids: set[int],
) -> float:
    """
    Returns the implied confidence for a Run A field based on evidence quality
    and source tier.
    """
    entry = evidence_json.get(field_path)
    if not entry:
        return _IMPLIED_CONF_NO_EVIDENCE
    source_type = entry.get("source_type", "")
    if source_type == "scraped":
        raw_id = entry.get("raw_id")
        if raw_id is not None and raw_id in oem_raw_ids:
            return _IMPLIED_CONF_OEM
        return _IMPLIED_CONF_SCRAPED
    if source_type == "transcript":
        return _IMPLIED_CONF_TRANSCRIPT
    return _IMPLIED_CONF_NO_EVIDENCE


# ---------------------------------------------------------------------------
# Auto-resolution logic
# ---------------------------------------------------------------------------

def _auto_resolve_conflict(
    run_a_source_tier: str,
    run_a_conf: float,
    enrichment_tier: str,
    enrichment_conf: float,
) -> tuple[str, str | None, str]:
    """
    Applies the four auto-resolution rules to a genuine conflict.

    Returns:
        (resolution, resolved_by, note)
        resolution:  'kept_run_a' | 'kept_enrichment' | 'flagged'
        resolved_by: 'auto_source_priority' | 'auto_confidence' | None
        note:        human-readable explanation
    """
    # Rule 1: Enrichment from OEM official + high confidence vs non-OEM Run A
    if (
        enrichment_tier == "oem_official"
        and enrichment_conf >= _OEM_AUTO_THRESHOLD
        and run_a_source_tier != "oem_official"
    ):
        return (
            "kept_enrichment",
            "auto_source_priority",
            (
                f"Enrichment from oem_official (conf={enrichment_conf:.2f}) "
                f"overrides Run A from {run_a_source_tier} (conf={run_a_conf:.2f})."
            ),
        )

    # Rule 2: Run A from OEM official, enrichment NOT oem_official  (C5 fix)
    if (
        run_a_source_tier == "oem_official"
        and enrichment_tier != "oem_official"
        and run_a_conf >= _OEM_AUTO_THRESHOLD
    ):
        return (
            "kept_run_a",
            "auto_source_priority",
            (
                f"Run A from oem_official (conf={run_a_conf:.2f}) "
                f"takes priority over enrichment from {enrichment_tier} "
                f"(conf={enrichment_conf:.2f})."
            ),
        )

    # Rule 3: Confidence delta >= 0.20
    delta = enrichment_conf - run_a_conf
    if abs(delta) >= _DELTA_THRESHOLD:
        if delta > 0:
            return (
                "kept_enrichment",
                "auto_confidence",
                (
                    f"Enrichment confidence {enrichment_conf:.2f} exceeds "
                    f"Run A {run_a_conf:.2f} by delta={delta:.2f} >= {_DELTA_THRESHOLD}."
                ),
            )
        else:
            return (
                "kept_run_a",
                "auto_confidence",
                (
                    f"Run A confidence {run_a_conf:.2f} exceeds "
                    f"enrichment {enrichment_conf:.2f} by delta={abs(delta):.2f} >= {_DELTA_THRESHOLD}."
                ),
            )

    # Rule 4: Flagged — admin must decide
    return (
        "flagged",
        None,
        (
            f"Cannot auto-resolve: Run A {run_a_source_tier} conf={run_a_conf:.2f} "
            f"vs enrichment {enrichment_tier} conf={enrichment_conf:.2f}. "
            f"Delta={abs(delta):.2f} < {_DELTA_THRESHOLD}. Admin review required."
        ),
    )


# ---------------------------------------------------------------------------
# C8: Path array-index helper
# ---------------------------------------------------------------------------

def _path_has_secondary_index(path: str) -> bool:
    """
    Returns True if any array index in path is >= _SECONDARY_ARRAY_INDEX_THRESHOLD.
    Used by Phase 7.5 to demote secondary display/lens checks to warnings.

    'displays[0].panel_type' → False (primary)
    'displays[1].resolution' → True  (secondary — treat as warning)
    'cameras.lenses[2].aperture' → True (tertiary lens — treat as warning)
    """
    for m in re.finditer(r"\[(\d+)\]", path):
        if int(m.group(1)) >= _SECONDARY_ARRAY_INDEX_THRESHOLD:
            return True
    return False


# ---------------------------------------------------------------------------
# Task 7.1 — detect_and_resolve_conflicts
# ---------------------------------------------------------------------------

async def detect_and_resolve_conflicts(
    normalized_id: int,
    enrichment_run_id: int,
) -> dict:
    """
    Compares Run A normalized values with selected enrichment candidates.

    C3 idempotency: Deletes existing conflict rows for this normalized_id before
    inserting the new batch, so re-runs don't accumulate duplicates.

    C4 type coercion: enr_value is coerced to match run_a_value's type before
    comparison to avoid false conflicts from string vs int mismatches.

    C5 OEM source detection: Fetches oem_raw_ids to correctly classify Run A
    source tier, enabling Rule 2 to fire when OEM evidence exists.

    Returns:
        {
            "flagged_count":       int,
            "auto_resolved_count": int,
            "fill_count":          int,
            "concordant_count":    int,
            "conflict_ids":        list[int],
        }
    """
    logger.info(
        "detect_and_resolve_conflicts: START normalized_id=%d enrichment_run_id=%d",
        normalized_id, enrichment_run_id,
    )

    # Fetch normalized spec (Run A output)
    norm_row = await asyncio.to_thread(fetch_normalized_spec, normalized_id)
    url_registry_id: int = norm_row["url_registry_id"]
    normalized_json: dict = norm_row.get("normalized_json") or {}

    # C1: Use renamed function (was fetch_spec_extraction_output)
    # C5: Fetch OEM raw_ids for source tier detection
    ext_output, oem_raw_ids = await asyncio.gather(
        asyncio.to_thread(fetch_latest_spec_output_for_phone, url_registry_id),
        asyncio.to_thread(fetch_oem_raw_ids_for_phone, url_registry_id),
    )
    evidence_json: dict = (ext_output or {}).get("evidence_json") or {}

    # Fetch all selected enrichment candidates (only selected used for conflict detection)
    candidates: list[dict] = await asyncio.to_thread(
        fetch_selected_enrichment_candidates, enrichment_run_id
    )

    now_iso = _now_iso()

    flagged_count       = 0
    auto_resolved_count = 0
    fill_count          = 0
    concordant_count    = 0
    conflict_ids: list[int] = []

    # #7 fix: build the full payload list BEFORE touching the DB.
    # Previously delete_conflict_log_for_phone ran at the START of this function.
    # If any insert_merge_conflict call failed mid-loop, the deleted rows were
    # permanently lost. Now: collect payloads → delete old rows → insert new batch.
    # A per-field insert failure only skips that field (same as before), but the
    # delete only happens after we've determined what to insert.
    pending_payloads: list[dict] = []

    for cand in candidates:
        field_path: str   = cand["field_path"]
        enr_value         = cand["extracted_value"]   # may be None
        enr_conf: float   = float(cand["confidence"])
        enr_tier: str     = cand.get("source_tier", "unknown")
        candidate_id: int = cand["candidate_id"]

        # Read Run A value from normalized_json
        run_a_value = _get_at_path(normalized_json, field_path)

        # Classify the comparison
        if run_a_value is None and enr_value is not None:
            # FILL — no conflict row needed
            fill_count += 1
            continue

        if run_a_value is None and enr_value is None:
            continue   # both null — nothing to do

        if enr_value is None:
            continue   # enrichment returned null — keep Run A, no conflict

        # C4: Coerce enrichment value to Run A's type before comparison
        enr_value_coerced = _coerce_to_match(run_a_value, enr_value)

        if run_a_value == enr_value_coerced:
            # CONCORDANT — same value after coercion, no conflict
            concordant_count += 1
            continue

        # GENUINE CONFLICT — both non-null and different after coercion
        run_a_source_tier = _infer_run_a_source_tier(field_path, evidence_json, oem_raw_ids)
        run_a_conf        = _infer_run_a_confidence(field_path, evidence_json, oem_raw_ids)

        resolution, resolved_by, note = _auto_resolve_conflict(
            run_a_source_tier, run_a_conf,
            enr_tier, enr_conf,
        )

        pending_payloads.append({
            "url_registry_id":         url_registry_id,
            "normalized_id":           normalized_id,
            "field_path":              field_path,
            "run_a_value":             run_a_value,
            "enrichment_value":        enr_value,         # store original (not coerced)
            "enrichment_candidate_id": candidate_id,
            "resolution":              resolution,
            "resolved_by":             resolved_by,
            "resolution_note":         note,
            "resolved_at":             now_iso if resolution != "flagged" else None,
            "_is_flagged":             resolution == "flagged",   # scratch flag, stripped below
        })

    # #7 fix: delete OLD conflict rows only after we've classified all candidates.
    # human_override rows are protected by delete_conflict_log_for_phone's filter.
    await asyncio.to_thread(delete_conflict_log_for_phone, normalized_id)

    # Insert the new conflict batch
    for conflict_payload in pending_payloads:
        is_flagged = conflict_payload.pop("_is_flagged")
        try:
            conflict_id = await asyncio.to_thread(insert_merge_conflict, conflict_payload)
            conflict_ids.append(conflict_id)
        except Exception as exc:
            logger.error(
                "detect_and_resolve_conflicts: failed to insert conflict "
                "field=%r normalized_id=%d: %s",
                conflict_payload.get("field_path"), normalized_id, exc,
            )
            continue

        if is_flagged:
            flagged_count += 1
        else:
            auto_resolved_count += 1

    logger.info(
        "detect_and_resolve_conflicts: COMPLETE normalized_id=%d "
        "fills=%d concordant=%d auto_resolved=%d flagged=%d",
        normalized_id, fill_count, concordant_count, auto_resolved_count, flagged_count,
    )

    return {
        "flagged_count":       flagged_count,
        "auto_resolved_count": auto_resolved_count,
        "fill_count":          fill_count,
        "concordant_count":    concordant_count,
        "conflict_ids":        conflict_ids,
    }


# ---------------------------------------------------------------------------
# Task 7.2 — build_final_merged_json
# ---------------------------------------------------------------------------

async def build_final_merged_json(
    normalized_id: int,
    enrichment_run_id: int | None,
) -> int:
    """
    Merges normalized_json (base) + enrichment fills / conflict winners into
    pipeline.final_merged_json.

    C7 fix: The FILL pass now uses ALL enrichment candidates (not just is_selected=TRUE).
    When Run A is null, the highest-confidence candidate is used regardless of selection
    status. Only for conflict cases (both non-null and different) does is_selected matter.

    Merge precedence:
        1. Start with normalized_json (Run A output after normalization)
        2. For each field where Run A is null: apply highest-confidence candidate
           (regardless of is_selected) — FILL
        3. For each field where Run A and enrichment both non-null and differ:
           apply the candidate winner if resolution='kept_enrichment'

    Returns:
        final_id (int)
    """
    logger.info(
        "build_final_merged_json: START normalized_id=%d enrichment_run_id=%r",
        normalized_id, enrichment_run_id,
    )

    # Fetch normalized spec (Run A base) — C10: copy already imported at top
    norm_row = await asyncio.to_thread(fetch_normalized_spec, normalized_id)
    url_registry_id: int = norm_row["url_registry_id"]
    final_json: dict = copy.deepcopy(norm_row.get("normalized_json") or {})

    if enrichment_run_id is not None:
        # C2: Use proper repo function (was lambda with get_client import)
        conflict_map, all_candidates = await asyncio.gather(
            asyncio.to_thread(fetch_conflict_resolution_map, normalized_id),
            asyncio.to_thread(fetch_all_enrichment_candidates_for_run, enrichment_run_id),
        )

        # Group candidates by field_path, keeping highest confidence per field
        # (all_candidates is already ordered by confidence DESC from the repo fn)
        best_per_field: dict[str, dict] = {}
        for cand in all_candidates:
            fp = cand["field_path"]
            if fp not in best_per_field:
                best_per_field[fp] = cand   # first = highest confidence

        for field_path, cand in best_per_field.items():
            enr_value = cand["extracted_value"]
            if enr_value is None:
                continue

            run_a_value = _get_at_path(final_json, field_path)

            if run_a_value is None:
                # C7: FILL — use best candidate regardless of is_selected
                _set_at_path(final_json, field_path, enr_value)
                logger.debug(
                    "build_final_merged_json: FILL field=%r value=%r (is_selected=%s)",
                    field_path, enr_value, cand.get("is_selected"),
                )
            else:
                # Conflict or concordant — only apply if explicitly kept_enrichment
                coerced = _coerce_to_match(run_a_value, enr_value)
                if run_a_value != coerced:
                    outcome = conflict_map.get(field_path)
                    if outcome == "kept_enrichment":
                        _set_at_path(final_json, field_path, enr_value)
                        logger.debug(
                            "build_final_merged_json: CONFLICT WINNER (enrichment) field=%r",
                            field_path,
                        )
                    # kept_run_a or flagged → leave Run A value unchanged

    # Count gate conditions
    fields_remaining_null = _count_nulls(final_json)
    flagged_count, pending_staging = await asyncio.gather(
        asyncio.to_thread(fetch_flagged_conflict_count, normalized_id),
        asyncio.to_thread(fetch_pending_staging_count, url_registry_id),
    )
    has_unresolved_conflicts = flagged_count > 0

    final_payload: dict[str, Any] = {
        "url_registry_id":             url_registry_id,
        "normalized_id":               normalized_id,
        "enrichment_run_id":           enrichment_run_id,
        "final_json":                  final_json,
        "fields_remaining_null":       fields_remaining_null,
        "has_unresolved_conflicts":    has_unresolved_conflicts,
        "pending_staging_values":      pending_staging,
        # Admin approval flags — always FALSE on build
        "spec_human_approved":         False,
        "experience_human_approved":   False,
        "experience_entries_reviewed": False,
        "ready_for_commit":            False,
    }

    final_id = await asyncio.to_thread(upsert_final_merged_json, final_payload)

    logger.info(
        "build_final_merged_json: COMPLETE final_id=%d normalized_id=%d "
        "fields_remaining_null=%d has_conflicts=%s pending_staging=%d",
        final_id, normalized_id,
        fields_remaining_null, has_unresolved_conflicts, pending_staging,
    )

    return final_id
