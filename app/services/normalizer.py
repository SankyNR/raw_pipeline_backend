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
    fetch_brand_model_by_url_registry_id,
    insert_normalisation_run,
    insert_lookup_value_staging,
    update_normalisation_run,
    upsert_normalized_spec,
)
from app.repositories.mobile_specs_repository import fetch_brand_id, fetch_chipset_by_name
from app.services.pre_normalizer_enrichment import run_pre_normalizer_enrichment

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
    "Dual SIM (Nano-SIM)"            : "Dual SIM (Nano-SIM, dual stand-by)",
    "Dual SIM (Nano-SIM, dual standby)": "Dual SIM (Nano-SIM, dual stand-by)",
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


# ---------------------------------------------------------------------------
# Box item alias map — maps LLM-extracted item_name / combined forms to
# canonical lookup_box_items.item_name strings before FK resolution.
# Keys must be lowercased. Applied before fuzzy matching so common naming
# variants never reach staging.
# ---------------------------------------------------------------------------
_BOX_ITEM_ALIASES: dict[str, str] = {
    # SIM ejector variants
    "sim tool":              "SIM Ejector",
    "sim ejector tool":      "SIM Ejector",
    "sim pin":               "SIM Ejector",
    "sim card tool":         "SIM Ejector",
    "sim tray tool":         "SIM Ejector",

    # Documentation variants
    "guides":                "Quick Start Guide",
    "guide":                 "Quick Start Guide",
    "booklet":               "Quick Start Guide",
    "quick start guide":     "Quick Start Guide",
    "getting started guide": "Quick Start Guide",
    "user manual":           "User Manual",
    "manual":                "User Manual",
    "instruction manual":    "User Manual",
    "documentation":         "Quick Start Guide",

    # Case / cover variants
    "protective cover":      "Phone Case",
    "protective case":       "Phone Case",
    "cover":                 "Phone Case",
    "case":                  "Phone Case",
    "silicone case":         "Phone Case",
    "back cover":            "Phone Case",
    "hard case":             "Hard Protective Case",
    "hard cover":            "Hard Protective Case",
    "soft case":             "Soft Protective Case",
    "soft cover":            "Soft Protective Case",
    "transparent case":      "Phone Case",
    "clear case":            "Phone Case",

    # Cable variants
    "usb cable":             "USB Cable (Type-C to Type-C)",
    "type-c cable":          "USB Cable (Type-C to Type-C)",
    "charging cable":        "USB Cable (Type-C to Type-C)",
    "data cable":            "USB Cable (Type-C to Type-C)",

    # Charger variants — any wattage maps to plain "Charger"
    "charger":               "Charger",
    "wall charger":          "Charger",
    "power adapter":         "Charger",
    "adapter":               "Charger",
    "charging adapter":      "Charger",
    "travel charger":        "Charger",
    "turbopower charger":    "Charger",
    "supervooc charger":     "Charger",
    "flashcharge charger":   "Charger",
    "supercharge charger":   "Charger",
}

# Front camera shape aliases — normalizes casing/punctuation variants to
# the canonical lookup_front_camera_shape values.
# Keys must be lowercased with all spaces/hyphens collapsed to a single space.
_FRONT_CAMERA_SHAPE_ALIASES: dict[str, str] = {
    # Punch-hole variants
    "punchhole":          "Punch-hole",
    "punch hole":         "Punch-hole",
    "punch-hole":         "Punch-hole",
    "punchole":           "Punch-hole",
    "punch hole cutout":  "Punch-hole",
    "hole punch":         "Punch-hole",
    # Notch variants
    "notch":              "Notch",
    "waterdrop notch":    "Notch",
    "dewdrop notch":      "Notch",
    "teardrop notch":     "Notch",
    "dot notch":          "Notch",
    # Pill variants
    "pill":               "Pill",
    "pill shape":         "Pill",
    "pill-shaped":        "Pill",
    # Dynamic Island
    "dynamic island":     "Dynamic Island",
    "dynamicisland":      "Dynamic Island",
    # Pop-up
    "popup":              "Pop-up",
    "pop up":             "Pop-up",
    "pop-up":             "Pop-up",
    "motorized popup":    "Pop-up",
    # Under-display
    "underdisplay":       "Under-display",
    "under display":      "Under-display",
    "under-display":      "Under-display",
    "udc":                "Under-display",
    # Bezel / no cutout
    "no cutout":          "Bezel",
    "thick bezel":        "Bezel",
    "bezel":              "Bezel",
}

# Wattage charger pattern — matches any string like "68W TurboPower Charger",
# "45W Charger", "120W SuperVOOC Charger", "Charger (65W)", etc.
# Used to map any wattage-specific charger name to plain "Charger".
_WATTAGE_CHARGER_RE = re.compile(
    r"(\d+\s*w\b.*charger|charger.*\d+\s*w\b)",
    re.IGNORECASE,
)

# Brand keywords + generic device names used to detect handset entries.
# The LLM may extract the phone model name, "Device", "Smartphone", etc.
# as a box item — all map to "Handset".
_HANDSET_KEYWORDS: frozenset[str] = frozenset({
    # Generic device terms
    "device", "smartphone", "phone", "mobile", "handset",
    "the phone", "the device", "the smartphone",
    # Brand names — phone model extracted as box item
    "motorola", "samsung", "oneplus", "xiaomi", "redmi", "poco",
    "realme", "oppo", "vivo", "iqoo", "google", "apple", "nothing",
    "honor", "tecno", "infinix", "lava", "itel", "iphone", "pixel",
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

# ---------------------------------------------------------------------------
# Section 2.1 — Alias cache for camera feature brand-aware resolution
# Structure: {cleaned_brand_alias: {brand_id_or_None: feature_id}}
# brand_id_or_None preserves brand-aware resolution for collision cases (N7).
# ---------------------------------------------------------------------------
ALIAS_CACHE: dict[str, dict[int | None, int]] = {}

# ---------------------------------------------------------------------------
# Section 3.1 — Chipset row cache
# Loaded once at startup via warm_chipset_rows().
# Structure: {chipset_id: full_row_dict}
# Replaces per-phone fetch_chipset_by_name DB calls — zero I/O at runtime.
# Table is small (~500 rows), low write frequency, high read frequency.
# ---------------------------------------------------------------------------
CHIPSET_ROW_CACHE: dict[int, dict] = {}


def warm_chipset_rows() -> None:
    """
    Section 3.1 — Loads all rows from mobile_specs.chipsets into CHIPSET_ROW_CACHE.

    Called once at app startup inside build_lookup_cache(), after standard lookup
    tables are loaded.  Re-called via POST /extraction/cache/refresh so admin
    chipset additions are picked up without an app restart.

    Synchronous — wrapped in asyncio.to_thread() by build_lookup_cache().
    """
    result = (
        get_client()
        .schema("mobile_specs")
        .table("chipsets")
        .select("*")
        .execute()
    )
    rows = result.data or []
    CHIPSET_ROW_CACHE.clear()
    for row in rows:
        cid = row.get("chipset_id")
        if cid is not None:
            CHIPSET_ROW_CACHE[int(cid)] = row
    logger.info(
        "warm_chipset_rows: loaded %d chipset rows into CHIPSET_ROW_CACHE.",
        len(CHIPSET_ROW_CACHE),
    )


def get_chipset_row(chipset_id: int) -> dict:
    """Returns the cached chipset row dict for chipset_id, or empty dict if not cached."""
    return CHIPSET_ROW_CACHE.get(chipset_id, {})

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

# Deterministic map for deriving RAM frequency from ram_type string.
# Applied right before the end of normalisation.
_RAM_TYPE_FREQUENCY_MAP: dict[str, int] = {
    "LPDDR5X": 8533,
    "LPDDR5": 6400,
    "LPDDR4X": 4266,
    "LPDDR4": 3200,
    "LPDDR3": 1866,
    "LPDDR2": 1066,
}


# ---------------------------------------------------------------------------
# CPU core tier classifier — deterministic correction of LLM misclassification
# ---------------------------------------------------------------------------
# LLM frequently promotes A7x cores to cpu_ultra_high_performance_cores when
# no Cortex-X exists (training bias). This classifier examines the core
# codename and identifies the correct tier (ULTRA / PERF / EFF).
# Covers all chipset families: ARM Cortex, Qualcomm proprietary, Apple custom,
# Samsung Exynos custom, MediaTek (uses ARM cores).

# ULTRA tier — prime / highest-performance cores
_TIER_ULTRA_PATTERNS: tuple[re.Pattern, ...] = (
    # ARM Cortex-X series (X1, X2, X3, X4, X925, X1-Extreme, etc.)
    re.compile(r"Cortex[-\s]?X\d*", re.IGNORECASE),
    # Qualcomm Oryon Phoenix L (Snapdragon 8 Elite Gen 4+)
    re.compile(r"Oryon\s+Phoenix\s+L", re.IGNORECASE),
    # Qualcomm Kryo Prime (legacy 200–490 series)
    re.compile(r"Kryo\s+Prime", re.IGNORECASE),
    # Apple performance cores (A14/M1 onwards)
    re.compile(r"\b(Firestorm|Avalanche|Everest|Coyote|Tempest-Prime)\b", re.IGNORECASE),
    # Apple generic "Performance" tier label
    re.compile(r"performance\s+core", re.IGNORECASE),
    # Samsung Exynos custom prime cores
    re.compile(r"\bMongoose\b", re.IGNORECASE),
    re.compile(r"Exynos\s+M\d+", re.IGNORECASE),
)

# PERF tier — big / performance cluster (not prime)
_TIER_PERF_PATTERNS: tuple[re.Pattern, ...] = (
    # ARM Cortex-A7x series (A72-A78, A710-A725, A78C, A78AE)
    re.compile(r"Cortex[-\s]?A7\d[A-Z]?\d?", re.IGNORECASE),
    # Qualcomm Oryon Phoenix M
    re.compile(r"Oryon\s+Phoenix\s+M", re.IGNORECASE),
    # Qualcomm Kryo Gold (legacy)
    re.compile(r"Kryo\s+Gold", re.IGNORECASE),
)

# EFF tier — efficiency / little cluster
_TIER_EFF_PATTERNS: tuple[re.Pattern, ...] = (
    # ARM Cortex-A5x series (A53, A55, A510, A520)
    re.compile(r"Cortex[-\s]?A5\d{1,2}", re.IGNORECASE),
    # ARM Cortex-A3x and legacy Cortex-A7 (NOT A7x — the bare A7 only)
    re.compile(r"Cortex[-\s]?A3\d", re.IGNORECASE),
    re.compile(r"Cortex[-\s]?A7(?![\d])", re.IGNORECASE),
    # Qualcomm Kryo Silver (legacy)
    re.compile(r"Kryo\s+Silver", re.IGNORECASE),
    # Apple efficiency cores
    re.compile(r"\b(Icestorm|Blizzard|Sawtooth|Tempest-Efficient)\b", re.IGNORECASE),
    # Apple generic "Efficiency" tier label
    re.compile(r"efficiency\s+core", re.IGNORECASE),
)


def _classify_core_tier(cluster_str: str) -> str | None:
    """
    Examines a CPU cluster string (e.g. "4x2.40 GHz Cortex-A78") and returns
    which tier slot the cluster belongs in: 'ULTRA', 'PERF', 'EFF', or None
    if no known core pattern matches.

    Priority: ULTRA patterns checked first (most specific), then PERF, then EFF.
    """
    for pat in _TIER_ULTRA_PATTERNS:
        if pat.search(cluster_str):
            return "ULTRA"
    for pat in _TIER_PERF_PATTERNS:
        if pat.search(cluster_str):
            return "PERF"
    for pat in _TIER_EFF_PATTERNS:
        if pat.search(cluster_str):
            return "EFF"
    return None


def _fix_cpu_core_classification(data: dict, issues: list[dict]) -> dict:
    """
    Step 3.4 — Deterministic CPU core classification correction.

    Examines each cluster slot (ULTRA / PERF / EFF) and identifies the actual
    tier of the core(s) in that slot via _classify_core_tier(). If a slot
    contains a core that belongs in a different tier, moves the value to the
    correct slot — but only if the target slot is empty (avoids overwriting
    correctly placed entries like Dimensity 9400's dual-ULTRA design).

    Examples handled:
      "Cortex-A78" in cpu_ultra_high_performance_cores when no Cortex-X exists
        → moved to cpu_high_performance_cores, ULTRA set to null
      "Kryo Gold" in cpu_ultra_high_performance_cores
        → moved to cpu_high_performance_cores
      "Firestorm" in cpu_efficiency_cores (Apple anomaly)
        → moved to cpu_ultra_high_performance_cores
    """
    chipset = data.get("chipset")
    if not isinstance(chipset, dict):
        return data

    slot_field_map = {
        "ULTRA": "cpu_ultra_high_performance_cores",
        "PERF":  "cpu_high_performance_cores",
        "EFF":   "cpu_efficiency_cores",
    }

    # Snapshot current slot contents
    slot_contents = {
        slot: chipset.get(field)
        for slot, field in slot_field_map.items()
    }

    # Identify misclassifications: (from_slot, to_slot, value)
    corrections: list[tuple[str, str, str]] = []
    for from_slot, value in slot_contents.items():
        if not isinstance(value, str) or not value.strip():
            continue
        actual_tier = _classify_core_tier(value)
        if actual_tier is None or actual_tier == from_slot:
            continue
        corrections.append((from_slot, actual_tier, value))

    # Apply corrections
    for from_slot, to_slot, value in corrections:
        from_field = slot_field_map[from_slot]
        to_field   = slot_field_map[to_slot]

        # Refuse to overwrite a non-empty target slot — could be a real
        # dual-tier design (e.g. Dimensity 9400: X925 + X4 both in ULTRA).
        # If from_slot ≠ to_slot AND target is filled → genuine conflict.
        if chipset.get(to_field) is not None:
            issues.append({
                "field": from_field,
                "issue": "core_classification_conflict",
                "raw_value": value,
                "corrected_value": (
                    f"would move to {to_field} but target already has "
                    f"{chipset.get(to_field)!r} — manual review required"
                ),
            })
            logger.warning(
                "_fix_cpu_core_classification: conflict — %r in %s should be %s "
                "but %s already has %r",
                value, from_field, to_field, to_field, chipset.get(to_field),
            )
            continue

        # Move the value
        chipset[to_field] = value
        chipset[from_field] = None
        issues.append({
            "field": from_field,
            "issue": f"core_misclassified_{from_slot}_to_{to_slot}",
            "raw_value": value,
            "corrected_value": f"moved to {to_field}",
        })
        logger.info(
            "_fix_cpu_core_classification: moved %r from %s → %s",
            value, from_field, to_field,
        )

    return data


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

    _sem = asyncio.Semaphore(10)  # cap concurrent Supabase connections to free-tier safe limit

    async def _load_safe(table_path: str) -> None:
        async with _sem:
            try:
                await asyncio.to_thread(_load_one_table, table_path)
            except Exception as exc:
                logger.error(
                    "build_lookup_cache: failed to load table_path=%r: %s",
                    table_path, exc,
                )

    await asyncio.gather(*[_load_safe(tp) for tp in all_table_paths])

    logger.info(
        "build_lookup_cache: complete. %d tables cached.", len(LOOKUP_CACHE)
    )

    # Section 2.1 — load the alias cache after all standard tables
    await _load_alias_cache()

    # Section 3.1 — warm chipset row cache (for DB-injection branch in Step 2.5)
    try:
        await asyncio.to_thread(warm_chipset_rows)
    except Exception as exc:
        logger.error(
            "build_lookup_cache: warm_chipset_rows failed: %s — "
            "chipset DB-injection will fall back to fetch_chipset_by_name.", exc
        )

    # Section 4 — warm gap analyzer caches (price tiers + enrichment policy)
    try:
        from app.services.gap_analyzer import warm_gap_caches
        await asyncio.to_thread(warm_gap_caches)
    except Exception as exc:
        logger.error(
            "build_lookup_cache: warm_gap_caches failed: %s — "
            "gap analyzer will fall back to legacy FIELD_PRIORITY_MAP.", exc
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


async def _load_alias_cache() -> None:
    """
    Section 2.1 — Load lookup_feature_aliases into ALIAS_CACHE preserving brand_id.

    Structure: {cleaned_brand_alias: {brand_id_or_None: feature_id}}
    brand_id_or_None allows brand-aware resolution so the same alias string
    can map to different canonicals per brand (N7 collision scenario).

    Called at the end of build_lookup_cache() and on cache/refresh.
    """
    def _do() -> None:
        client = get_client()
        result = (
            client.schema("mobile_specs")
            .table("lookup_feature_aliases")
            .select("brand_alias, brand_id, feature_id")
            .execute()
        )
        ALIAS_CACHE.clear()
        for row in (result.data or []):
            cleaned = clean_for_lookup(str(row["brand_alias"]))
            ALIAS_CACHE.setdefault(cleaned, {})[row["brand_id"]] = row["feature_id"]

    await asyncio.to_thread(_do)
    logger.info(
        "build_lookup_cache: loaded %d alias entries into ALIAS_CACHE",
        sum(len(v) for v in ALIAS_CACHE.values()),
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
# Section 2.2 — Brand-aware camera feature resolver
# ---------------------------------------------------------------------------

def resolve_camera_feature_value(
    raw_value: str,
    brand_id: int | None,
) -> tuple[int | None, str | None, str]:
    """
    Three-pass resolution for camera_features (Section 2.2 of roadmap):
      1. Brand-specific alias match  (brand_alias, brand_id)  → ALIAS_CACHE
      2. Brand-agnostic alias match  (brand_alias, NULL)      → ALIAS_CACHE
      3. Canonical name match → delegates to resolve_lookup_value()
         (includes Levenshtein fuzzy fallback)
      Falls through to not_found if all three passes fail — caller sends to staging.

    Args:
        raw_value: String extracted by the LLM (e.g. "Nightography").
        brand_id:  Integer PK from public.brands, or None if brand is unknown.

    Returns:
        (feature_id, corrected_name, resolution_type) where resolution_type is one of:
          'alias_brand_specific' | 'alias_global' | 'exact' | 'fuzzy' | 'not_found'
    """
    cleaned = clean_for_lookup(raw_value)

    # Pass 1: brand-specific alias match
    if cleaned in ALIAS_CACHE and brand_id is not None and brand_id in ALIAS_CACHE[cleaned]:
        return ALIAS_CACHE[cleaned][brand_id], raw_value, "alias_brand_specific"

    # Pass 2: brand-agnostic alias match
    if cleaned in ALIAS_CACHE and None in ALIAS_CACHE[cleaned]:
        return ALIAS_CACHE[cleaned][None], raw_value, "alias_global"

    # Pass 3: canonical name match (exact + fuzzy via standard resolver)
    return resolve_lookup_value(raw_value, "mobile_specs.lookup_camera_features.canonical")


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


# ---------------------------------------------------------------------------
# v1 normalizer step helpers (Steps 3.5, 4-prefill, 8.7b, 8.8, 8.9)
# ---------------------------------------------------------------------------

_REAR_CAMERA_COUNT_MAP: dict[int, str] = {
    1: "Single",
    2: "Dual",
    3: "Triple",
    4: "Quad",
}

_REAR_LENS_TYPES: frozenset[str] = frozenset({
    "main",
    "ultra-wide",
    "ultrawide",       # LLM variant without hyphen
    "ultra wide",      # LLM variant with space
    "telephoto",
    "periscope",
    "macro",
    "depth",
})


def _derive_rear_camera_setup(data: dict, issues: list[dict]) -> dict:
    """
    Step 3.5 — Rear camera lens count derivation.

    Always counts camera_lenses[] entries whose lens_type (lowercased) is a
    rear-facing type and writes the canonical setup name string into
    camera_overview.rear_camera_setup — OVERRIDING any LLM-provided value.

    Runs BEFORE Step 4 (scalar FK resolution) so the written string is
    resolved to a PK integer automatically in the same pass as all other
    SCALAR_FK_MAP fields.

    Rear lens types: Main, Ultra-wide (and variants), Telephoto, Periscope,
    Macro, Depth. Front-facing lens types are excluded by not being in
    _REAR_LENS_TYPES.

    5+ rear lenses → "Quad" (lookup_rear_camera_setup caps at Quad).
    0 rear lenses  → skip (leave null for gap analyzer).
    """
    camera_overview = data.get("camera_overview")
    if not isinstance(camera_overview, dict):
        return data

    lenses = data.get("camera_lenses") or []
    rear_count = sum(
        1 for lens in lenses
        if isinstance(lens, dict)
        and isinstance(lens.get("lens_type"), str)
        and lens["lens_type"].strip().lower() in _REAR_LENS_TYPES
    )

    if rear_count == 0:
        return data  # can't derive — leave null for gap analyzer

    setup_name = _REAR_CAMERA_COUNT_MAP.get(rear_count, "Quad")  # 5+ → Quad

    old_value = camera_overview.get("rear_camera_setup")
    camera_overview["rear_camera_setup"] = setup_name

    issues.append({
        "field":           "camera_overview.rear_camera_setup",
        "issue":           "rear_camera_setup_derived",
        "raw_value":       old_value,
        "corrected_value": setup_name,
    })
    logger.debug(
        "_derive_rear_camera_setup: %d rear lens(es) → %r (was %r)",
        rear_count, setup_name, old_value,
    )
    return data


def _prefill_widevine_defaults(data: dict, issues: list[dict]) -> dict:
    """
    Step 4 pre-fill — Widevine defaults.

    Sets certifications.widevine_support → True and
    certifications.widevine_level → "L1" when both are null.

    Runs BEFORE Step 4 (scalar FK resolution) so "L1" is resolved to its
    lookup PK in the same pass as all other FK fields.  The LLM is
    instructed to leave these null; normalizer fills them deterministically
    because 99%+ of phones ≥ 2017 support Widevine L1.
    Only writes when value is currently null — never overrides LLM output.
    """
    certs = data.get("certifications")
    if not isinstance(certs, dict):
        return data

    filled: list[str] = []
    if certs.get("widevine_support") is None:
        certs["widevine_support"] = True
        filled.append("widevine_support=True")
    if certs.get("widevine_level") is None:
        certs["widevine_level"] = "L1"
        filled.append("widevine_level=L1")

    if filled:
        issues.append({
            "field":           "certifications.widevine",
            "issue":           "widevine_prefilled",
            "raw_value":       None,
            "corrected_value": ", ".join(filled),
        })
        logger.debug("_prefill_widevine_defaults: %s", ", ".join(filled))
    return data


def _apply_simple_boolean_defaults(data: dict, issues: list[dict]) -> dict:
    """
    Step 8.7b — Simple boolean defaults for hardware presence/absence fields.

    Features that are always listed on spec sheets when present — silence means
    absent. Defaults applied only when the field is currently null.

    False defaults (absence of evidence = absent):
        connectivity.nfc         — hardware chip, always mentioned when present
        connectivity.uwb         — hardware chip, always mentioned when present
        connectivity.ir_blaster  — hardware feature, always mentioned when present
        network.esim_support     — physical eSIM always listed when supported
        network.vowifi           — listed when supported; India phones increasingly have it
        network.vo5g             — only true when source explicitly says VoNR/Vo5G
        charging.reverse_charging — wired reverse charging always listed when present
        body.has_stylus          — stylus support always listed when present

    True defaults:
        network.volte            — TRAI mandates VoLTE on all Indian 4G phones;
                                   absence of evidence means present, not absent
    """
    connectivity = data.get("connectivity")
    network = data.get("network")
    charging = data.get("charging")
    body = data.get("body")
    defaulted: list[str] = []

    if isinstance(connectivity, dict):
        for field in ("nfc", "uwb", "ir_blaster"):
            if connectivity.get(field) is None:
                connectivity[field] = False
                defaulted.append(f"connectivity.{field}=False")

    if isinstance(network, dict):
        for field in ("esim_support", "vowifi", "vo5g"):
            if network.get(field) is None:
                network[field] = False
                defaulted.append(f"network.{field}=False")
        if network.get("volte") is None:
            network["volte"] = True
            defaulted.append("network.volte=True")

    if isinstance(charging, dict):
        if charging.get("reverse_charging") is None:
            charging["reverse_charging"] = False
            defaulted.append("charging.reverse_charging=False")

    if isinstance(body, dict):
        if body.get("has_stylus") is None:
            body["has_stylus"] = False
            defaulted.append("body.has_stylus=False")

    if defaulted:
        issues.append({
            "field":           "boolean defaults",
            "issue":           "boolean_defaults_applied",
            "raw_value":       None,
            "corrected_value": ", ".join(defaulted),
        })
        logger.debug("_apply_simple_boolean_defaults: %s", ", ".join(defaulted))
    return data


# Tokens that identify a non-button hardware entry — any entry whose
# comma-separated tokens fuzzy-match any of these is dropped entirely.
# Short tokens (≤ 3 chars) use exact match only.
# 4–6 char tokens use Levenshtein ≤ 1.
# 7+ char tokens use Levenshtein ≤ 2.
_BUTTON_EXCLUDE_KEYWORDS: tuple[str, ...] = (
    "usb", "port", "typec", "type",
    "microphone", "mic",
    "speaker", "grille", "grill",
    "sim", "tray",
    "headphone", "jack",
    "charging", "connector",
)

# After cleanup, at least one entry must contain a token fuzzy-matching
# each of these. If missing → warning issue is raised (field is kept).
_BUTTON_PRIMARY_KEYWORDS: tuple[str, ...] = ("power", "volume")


def _derive_audio_positions(data: dict, issues: list[dict]) -> dict:
    """
    Step 8.8 — Microphone and speaker position derivation from count.

    When the LLM extracted counts but not position strings, derives canonical
    position strings from the count. Only fills null fields — never overrides
    an explicit LLM value.

    Microphone rules:
        count is null → default to 1 (every phone since touchscreen era has
                        at least one mic), set position "Bottom"
        count = 1     → "Bottom"
        count = 2     → "Top, Bottom"
        count ≥ 3     → leave positions null, raise manual-review issue

    Speaker rules:
        count = 1     → "Bottom"
        count ≥ 2     → "Bottom + Earpiece (Stereo)"
    """
    audio = data.get("audio")
    if not isinstance(audio, dict):
        return data

    # Change 1: Canonicalize explicit string matches
    explicit_positions = audio.get("speaker_positions")
    if explicit_positions:
        p = explicit_positions.lower()
        if p == "bottom":
            audio["speaker_positions"] = "Bottom"
        elif "stereo" in p or "earpiece" in p or ("top" in p and "bottom" in p):
            audio["speaker_positions"] = "Bottom + Earpiece (Stereo)"

    derived: list[str] = []

    # ── Microphone positions ──────────────────────────────────────────────
    if audio.get("microphone_positions") is None:
        mic_count = audio.get("microphone_count")

        # Default null mic count to 1 — every modern phone has at least one
        if mic_count is None:
            audio["microphone_count"] = 1
            mic_count = 1
            derived.append("microphone_count=1 (defaulted from null)")

        if mic_count == 1:
            audio["microphone_positions"] = "Bottom"
            derived.append("microphone_positions=Bottom")
        elif mic_count == 2:
            audio["microphone_positions"] = "Top, Bottom"
            derived.append("microphone_positions=Top, Bottom")
        elif isinstance(mic_count, int) and mic_count >= 3:
            # Cannot safely derive — requires manual review
            issues.append({
                "field":           "audio.microphone_positions",
                "issue":           f"microphone_count={mic_count} exceeds 2 — positions require manual entry",
                "raw_value":       mic_count,
                "corrected_value": None,
            })
            logger.debug(
                "_derive_audio_positions: mic_count=%d ≥ 3 — positions left null",
                mic_count,
            )

    # ── Speaker positions ─────────────────────────────────────────────────
    if audio.get("speaker_positions") is None:
        speaker_count = audio.get("speaker_count")
        if speaker_count == 1:
            audio["speaker_positions"] = "Bottom"
            derived.append("speaker_positions=Bottom")
        elif isinstance(speaker_count, int) and speaker_count >= 2:
            audio["speaker_positions"] = "Bottom + Earpiece (Stereo)"
            derived.append("speaker_positions=Bottom + Earpiece (Stereo)")

    if derived:
        issues.append({
            "field":           "audio positions",
            "issue":           "audio_positions_derived",
            "raw_value":       None,
            "corrected_value": ", ".join(derived),
        })
        logger.debug("_derive_audio_positions: %s", ", ".join(derived))
    return data


def _cleanup_buttons_text(data: dict) -> dict:
    """
    Step 8.9 — Buttons text cleanup.

    Splits body.buttons on commas. For each entry, strips the parenthetical
    position annotation, tokenizes on spaces/hyphens, and fuzzy-matches each
    token against _BUTTON_EXCLUDE_KEYWORDS. If any token hits → the entire
    entry is dropped. Physical button entries (Power, Volume, Action, etc.)
    are kept as-is.

    Fuzzy threshold by token length:
        ≤ 3 chars  → exact match only  (avoids false positives on "AI", "C")
        4–6 chars  → Levenshtein ≤ 1   (catches "USN" → "usb", "speakr" → "speaker")
        7+ chars   → Levenshtein ≤ 2   (catches "microphonr" → "microphone")

    After cleanup, raises a WARNING issue (never nulls the field) if "power"
    or "volume" tokens are absent — signals possible over-aggressive cleanup
    or a data quality problem for admin review.

    Logs all dropped entries in issues_found.
    """
    body = data.get("body")
    if not isinstance(body, dict):
        return data

    buttons = body.get("buttons")
    if not isinstance(buttons, str) or not buttons.strip():
        return data

    def _token_hits_excludes(token: str) -> bool:
        tlen = len(token)
        for kw in _BUTTON_EXCLUDE_KEYWORDS:
            if tlen <= 3:
                if token == kw:
                    return True
            elif tlen <= 6:
                if levenshtein_distance(token, kw) <= 1:
                    return True
            else:
                if levenshtein_distance(token, kw) <= 2:
                    return True
        return False

    def _tokenize_entry(entry: str) -> list[str]:
        # Strip parenthetical "(Right)", "(Bottom)" etc.
        stripped = re.sub(r"\(.*?\)", "", entry).lower()
        # Split on spaces and hyphens; drop empty and single-char tokens
        tokens = re.split(r"[\s\-/]+", stripped)
        return [t.strip(".,") for t in tokens if len(t) > 1]

    entries = [e.strip() for e in buttons.split(",") if e.strip()]
    kept: list[str] = []
    dropped: list[str] = []

    for entry in entries:
        tokens = _tokenize_entry(entry)
        if any(_token_hits_excludes(t) for t in tokens):
            dropped.append(entry)
        else:
            kept.append(entry)

    cleaned = ", ".join(kept) if kept else buttons  # fall back to original if all dropped

    if dropped:
        body["buttons"] = cleaned
        logger.debug(
            "_cleanup_buttons_text: dropped %d entries %r → %r",
            len(dropped), dropped, cleaned,
        )

    # Validate primaries are still present after cleanup
    all_kept_tokens: set[str] = set()
    for entry in kept:
        all_kept_tokens.update(_tokenize_entry(entry))

    missing_primaries = [
        p for p in _BUTTON_PRIMARY_KEYWORDS
        if not any(
            levenshtein_distance(t, p) <= (0 if len(t) <= 3 else 1 if len(t) <= 6 else 2)
            for t in all_kept_tokens
        )
    ]

    # Build a single combined issue entry if anything changed or is missing
    if dropped or missing_primaries:
        issue_parts: list[str] = []
        if dropped:
            issue_parts.append(f"dropped_entries={dropped}")
        if missing_primaries:
            issue_parts.append(f"missing_primary_buttons={missing_primaries} (WARNING: check manually)")
        logger.debug("_cleanup_buttons_text: %s", "; ".join(issue_parts))
    return data


def _derive_ram_frequency(data: dict, issues: list[dict]) -> dict:
    """
    Step 8.10 — Deterministic RAM frequency derivation from ram_type.
    """
    variants = data.get("variants")
    if not isinstance(variants, list):
        return data

    for i, variant in enumerate(variants):
        if not isinstance(variant, dict):
            continue
        freq = variant.get("ram_frequency")
        if freq is not None:
            continue
        rtype = variant.get("ram_type")
        if not isinstance(rtype, str) or not rtype.strip():
            continue

        clean_type = rtype.upper().replace(" ", "").replace("-", "")
        derived = None
        for key, val in _RAM_TYPE_FREQUENCY_MAP.items():
            if key in clean_type:
                derived = val
                break

        if derived is not None:
            variant["ram_frequency"] = derived
            issues.append({
                "field": f"variants[{i}].ram_frequency",
                "issue": "ram_frequency_derived_from_type",
                "raw_value": None,
                "corrected_value": derived,
            })
            logger.debug(
                "_derive_ram_frequency: derived %d MHz for variants[%d] from type %r",
                derived, i, rtype,
            )

    return data


def _apply_tier_gated_wireless_cascade(data: dict) -> dict:
    """
    Section 3.2 — Step 8.7: Tier-gated wireless charging cascade.

    Wireless charging default logic, gated by base variant launch_price:

    Case 1: wireless_charging IS NULL
        - base_variant_price < ₹30,000  → default False + full cascade nulls
        - base_variant_price ≥ ₹30,000  → leave null (gap analyzer will enrich)
        - price unknown (no base variant) → leave null (safe default)

    Case 2: wireless_charging IS FALSE (explicit LLM output)
        → always enforce cascade nulls (power/standard) + default reverse=False

    Case 3: wireless_charging IS TRUE
        → no action; gap analyzer handles missing power/standard fields.

    cpu_clock_speed / gpu_clock_speed are NOT touched — they now live in the phones
    table (Migration 49) and are written at commit time, not by this function.
    """
    charging = data.get("charging")
    if not isinstance(charging, dict):
        return data

    variants = data.get("variants") or []
    base_variant_price: int | None = next(
        (v.get("launch_price") for v in variants if v.get("is_base_variant") is True),
        None,
    )

    wc = charging.get("wireless_charging")

    if wc is None:
        # Only default to False for confirmed budget phones (< ₹30,000)
        if base_variant_price is not None and base_variant_price < 30000:
            charging["wireless_charging"]               = False
            charging["wireless_charging_power"]         = None
            charging["wireless_charging_standard"]      = None
            charging["reverse_wireless_charging"]       = False
            charging["reverse_wireless_charging_power"] = None
        # else: ≥ ₹30,000 or unknown price → leave null, gap analyzer enriches

    elif wc is False:
        # Explicit False from LLM → cascade to dependents
        charging["wireless_charging_power"]    = None
        charging["wireless_charging_standard"] = None
        # Default reverse to False only if currently null
        if charging.get("reverse_wireless_charging") is None:
            charging["reverse_wireless_charging"] = False
        if charging.get("reverse_wireless_charging") is False:
            charging["reverse_wireless_charging_power"] = None

    return data


def _query_charging_voltage_amperage(
    brand_name: str | None,
    charging_standard: str | None,
    wattage: float | None
) -> tuple[float | None, float | None]:
    """
    Calls the Supabase RPC 'get_charging_specs_by_brand_tech_wattage'
    to look up known voltage/amperage for a given combination.
    """
    if wattage is None or wattage <= 0 or not brand_name or not charging_standard:
        return None, None

    try:
        supabase = get_client()
        response = supabase.rpc(
            "get_charging_specs_by_brand_tech_wattage",
            {
                "p_brand": brand_name,
                "p_technology": charging_standard,
                "p_power": int(wattage)
            }
        ).execute()

        data = response.data
        if data and len(data) > 0:
            row = data[0]
            return row.get("charging_voltage"), row.get("charging_ampere")
    except Exception as e:
        logger.warning(
            "_query_charging_voltage_amperage DB call failed: %s", e
        )

    return None, None


async def _inject_charging_specs(data: dict, brand_name: str | None) -> dict:
    """
    Step 2.6 — Charging voltage/amperage injection.
    Looks up voltage and amperage when missing but wattage and standard are known.
    """
    charging = data.get("charging")
    if not isinstance(charging, dict):
        return data

    wattage = charging.get("charging_power")
    if wattage is None:
        return data

    v = charging.get("charging_voltage")
    a = charging.get("charging_ampere")

    if v is not None and a is not None:
        return data

    standard = charging.get("charging_standard")

    db_v, db_a = await asyncio.to_thread(
        _query_charging_voltage_amperage, brand_name, standard, wattage
    )

    if db_v is not None and v is None:
        charging["charging_voltage"] = float(db_v)
    if db_a is not None and a is None:
        charging["charging_ampere"] = float(db_a)

    return data


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

        _pre_brand: str | None = None

        # ── Section 2 — Pre-normalizer enrichment pass ───────────────────────
        # Runs before ALL normalisation steps so prices are clean integers and
        # chipset.chipset_name is populated before Step 2.5 fires.
        # Failure is non-fatal: issues are logged, pipeline continues unchanged.
        try:
            brand_model = await asyncio.to_thread(
                fetch_brand_model_by_url_registry_id, url_registry_id
            )
            if brand_model:
                _pre_brand, _pre_model = brand_model
                normalized_json, pre_issues = await run_pre_normalizer_enrichment(
                    normalized_json,
                    brand=_pre_brand,
                    model_name=_pre_model,
                    normalized_id=None,   # row not yet created at this point
                )
                issues.extend(pre_issues)
                logger.info(
                    "run_normalisation: pre-normalizer enrichment complete — "
                    "%d pre-issues for output_id=%d",
                    len(pre_issues), output_id,
                )
            else:
                logger.warning(
                    "run_normalisation: could not resolve brand/model for "
                    "url_registry_id=%d — pre-normalizer enrichment skipped.",
                    url_registry_id,
                )
        except Exception as _pre_exc:
            logger.warning(
                "run_normalisation: pre-normalizer enrichment raised unexpectedly "
                "(output_id=%d): %s — continuing without pre-enrichment.",
                output_id, _pre_exc,
            )

        # Step 2.5 — Chipset deduplication + DB-injection (Change 6b + Section 3.1)
        # Must run BEFORE gap analysis so the gap analyzer sees finalized chipset data.
        # If the chipset already exists in the DB:
        #   a) Overwrite all chipset fields with DB-canonical values.
        #   b) Inject additional detail fields from CHIPSET_ROW_CACHE (zero I/O).
        #      new_chipset_detected flag is left False (chip is known).
        # If NOT in DB: retain LLM-extracted values; capture name so we can set
        #   new_chipset_detected=True after Step 9 (when normalized_id exists).
        _new_chipset_name: str | None = None  # set below if chip not found in DB
        chipset_raw = normalized_json.get("chipset", {})
        chipset_name_raw: str | None = chipset_raw.get("chipset_name") if isinstance(chipset_raw, dict) else None
        if chipset_name_raw:
            # Use in-memory cache when warm; fall back to DB query on cache miss.
            existing_chipset: dict | None = None
            _cached_by_name = next(
                (row for row in CHIPSET_ROW_CACHE.values()
                 if row.get("chipset_name", "").strip().lower() == chipset_name_raw.strip().lower()),
                None,
            )
            if _cached_by_name:
                existing_chipset = _cached_by_name
            else:
                # Cache miss (cache empty or not yet warmed) — fall back to DB
                existing_chipset = await asyncio.to_thread(fetch_chipset_by_name, chipset_name_raw)

            if existing_chipset:
                chipset_id_from_db = existing_chipset.get("chipset_id")

                # Section 3.1 — DB-injection: build canonical chipset block.
                # cpu_clock_speed and gpu_clock_speed are intentionally excluded:
                # these are phone-specific (OEMs bin/underclock the same chipset)
                # and are now columns on mobile_specs.phones (Migration 49).
                # gpu_unit_count and gpu_unit_type are replaced by gpu_cores (Migration 50).
                _CHIPSET_CANONICAL_COLS = (
                    "chipset_id",
                    "chipset_name",
                    "cpu_architecture",
                    "fabrication_node",
                    "number_of_cores",
                    "cpu_ultra_high_performance_cores",
                    "cpu_high_performance_cores",
                    "cpu_efficiency_cores",
                    "gpu_name",
                    "gpu_cores",
                    "npu_details",
                    "npu_tops",
                )

                injected_block = {
                    k: existing_chipset.get(k)
                    for k in _CHIPSET_CANONICAL_COLS
                }

                # Also pull any extra detail fields from in-memory cache row
                if chipset_id_from_db and chipset_id_from_db in CHIPSET_ROW_CACHE:
                    cache_row = CHIPSET_ROW_CACHE[int(chipset_id_from_db)]
                    for col in _CHIPSET_CANONICAL_COLS:
                        if injected_block.get(col) is None and cache_row.get(col) is not None:
                            injected_block[col] = cache_row[col]

                normalized_json["chipset"] = injected_block
                logger.info(
                    "run_normalisation: chipset %r found in DB (chipset_id=%s) — "
                    "DB-canonical values injected (Migration 49: clock speeds now on phones table).",
                    chipset_name_raw, chipset_id_from_db,
                )
            else:
                # Chipset not in DB — LLM values retained; flag after Step 9
                _new_chipset_name = chipset_name_raw
                logger.info(
                    "run_normalisation: chipset %r is NEW — extracted values retained. "
                    "new_chipset_detected flag will be set after persist.",
                    chipset_name_raw,
                )

            # Step 2.5c — Defensive null-fill for chipset fields.
            # LLM may omit fields entirely when it has no data for them
            # (e.g. gpu_cores, npu_details, npu_tops, cpu_architecture).
            # Ensure every canonical chipset field exists as at least null so
            # gap analyzer picks them up as enrichable gaps.
            _CHIPSET_REQUIRED_FIELDS = (
                "chipset_name", "cpu_architecture", "fabrication_node",
                "number_of_cores", "cpu_ultra_high_performance_cores",
                "cpu_high_performance_cores", "cpu_efficiency_cores",
                "cpu_clock_speed", "gpu_clock_speed", "gpu_name", "gpu_cores",
                "npu_details", "npu_tops",
            )
            chipset_block = normalized_json.get("chipset")
            if isinstance(chipset_block, dict):
                for col in _CHIPSET_REQUIRED_FIELDS:
                    if col not in chipset_block:
                        chipset_block[col] = None
                        logger.debug(
                            "run_normalisation: null-filled missing chipset field %r",
                            col,
                        )

        # Step 2.6 — Charging voltage/amperage injection
        if _pre_brand:
            normalized_json = await _inject_charging_specs(normalized_json, _pre_brand)

        # Step 3 — Discard RUN_C_CALCULATED_FIELDS
        normalized_json = _discard_run_c_fields(normalized_json, issues)

        # Step 3.4 — CPU core classification correction
        # Fixes LLM misclassification (e.g. A78 placed in ULTRA when no Cortex-X exists).
        # Covers ARM Cortex, Qualcomm Oryon/Kryo, Apple, Samsung Exynos.
        normalized_json = _fix_cpu_core_classification(normalized_json, issues)

        # Step 3.5 — Rear camera setup derivation (v1)
        # Counts rear-facing lenses → writes "Single"/"Dual"/"Triple"/"Quad"
        # string before Step 4 FK-resolves it to an integer PK.
        normalized_json = _derive_rear_camera_setup(normalized_json, issues)

        # Step 4 pre-fill — Widevine defaults (v1)
        # Sets widevine_support=True / widevine_level="L1" when null, before
        # Step 4 FK-resolves "L1" → integer PK automatically.
        normalized_json = _prefill_widevine_defaults(normalized_json, issues)

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

        # Step 8.6 — BIS certification hardcode
        # Always true: all phones in this DB are sold in India (BIS/MTCTE mandatory).
        # LLM cannot set this reliably without source evidence — normalizer enforces it.
        if "certifications" in normalized_json and isinstance(normalized_json["certifications"], dict):
            normalized_json["certifications"]["bis_certification"] = True

        # Step 8.7 — Tier-gated wireless charging cascade (Section 3.2)
        # Only defaults wireless_charging=False for phones < ₹30,000.
        # Premium phones (≥ ₹30,000) keep null so gap analyzer can enrich.
        normalized_json = _apply_tier_gated_wireless_cascade(normalized_json)

        # Step 8.7b — Simple boolean defaults (v1)
        # nfc/uwb/ir_blaster/esim_support → False; volte → True for null fields.
        normalized_json = _apply_simple_boolean_defaults(normalized_json, issues)

        # Step 8.8 — Microphone + speaker position derivation (v1)
        # Derives position strings from extracted counts when positions are null.
        normalized_json = _derive_audio_positions(normalized_json, issues)

        # Step 8.9 — Buttons text cleanup (v1)
        # Normalizes whitespace and separators in body.buttons string.
        normalized_json = _cleanup_buttons_text(normalized_json)

        # Step 8.10 — RAM frequency derivation
        normalized_json = _derive_ram_frequency(normalized_json, issues)

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

        # Section 3.1 — Set new_chipset_detected flag now that normalized_id exists.
        # Fire-and-forget: a flag write failure never blocks the pipeline.
        if _new_chipset_name:
            try:
                from app.repositories.pipeline_run_repository import update_normalized_spec_flag
                await asyncio.to_thread(
                    update_normalized_spec_flag,
                    normalized_id,
                    new_chipset_detected=True,
                )
                logger.info(
                    "run_normalisation: new_chipset_detected=True set for "
                    "normalized_id=%d chipset=%r — admin must add to mobile_specs.chipsets.",
                    normalized_id, _new_chipset_name,
                )
            except Exception as _flag_exc:
                logger.warning(
                    "run_normalisation: failed to set new_chipset_detected for "
                    "normalized_id=%d: %s", normalized_id, _flag_exc
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

            # Change 2A: apply front_camera_shape alias BEFORE lookup
            # Handles casing/punctuation variants: "Punch Hole" → "Punch-hole", etc.
            if concrete_path == "camera_overview.front_camera_shape":
                alias_key = re.sub(r'[-\s]+', ' ', lookup_value.lower().strip())
                lookup_value = _FRONT_CAMERA_SHAPE_ALIASES.get(alias_key, lookup_value)

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

    Section 2.3: For camera_features, uses resolve_camera_feature_value() which
    performs three-pass alias-aware resolution (brand-specific → brand-agnostic →
    canonical) before falling back to staging. brand_id is fetched once for the
    whole call; failures fall back gracefully to brand-agnostic resolution.
    """
    issues: list[dict] = []
    staging_count = 0

    # Section 2.3 — fetch brand_id once for camera_features alias resolution.
    # _resolve_array_fks is called inside asyncio.to_thread so a sync DB call is safe.
    brand_name = (data.get("brand") or {}).get("brand_name") or ""
    brand_id: int | None = None
    if brand_name:
        try:
            brand_id = fetch_brand_id(brand_name)
        except Exception as exc:
            logger.warning(
                "_resolve_array_fks: failed to fetch brand_id for %r: %s. "
                "camera_features alias resolution will fall back to brand-agnostic + canonical only.",
                brand_name, exc,
            )
            brand_id = None

    for field_path, table_path in ARRAY_FK_MAP.items():
        concrete_paths = _expand_wildcard_path(data, field_path)

        for concrete_path in concrete_paths:
            raw_array = _get_nested(data, concrete_path)
            if not isinstance(raw_array, list) or not raw_array:
                continue

            resolved_pairs: list[tuple[str, int]] = []  # N10: (canonical_str, pk)
            seen_pks: set[int] = set()

            # Section 2.3 — camera_features uses brand-aware three-pass resolution.
            # All other arrays use the standard band-prefix-normalised path.
            is_camera_features = (concrete_path == "camera_features")

            for raw_value in raw_array:
                if raw_value is None:
                    continue

                # ── in_the_box structured object handling ──────────────────────────────
                # Extraction outputs objects: {quantity, item_name, item_specification}.
                # Build a lookup string from item_name + item_specification,
                # then apply aliases, wattage charger detection, and brand
                # keyword handset detection before FK resolution.
                if concrete_path == "in_the_box" and isinstance(raw_value, dict):
                    item_name = (raw_value.get("item_name") or "").strip()
                    item_spec  = (raw_value.get("item_specification") or "").strip()
                    if not item_name:
                        continue  # malformed entry — skip

                    # Build combined form: spec first, then item name
                    # "68W TurboPower" + "Charger" → "68W TurboPower Charger"
                    combined       = f"{item_spec} {item_name}".strip() if item_spec else item_name
                    combined_lower = combined.lower().strip()
                    item_lower     = item_name.lower().strip()

                    # Pass 1: wattage charger pattern — any "NNW ... Charger" → "Charger"
                    if _WATTAGE_CHARGER_RE.search(combined):
                        raw_value = "Charger"

                    # Pass 2: alias map — try combined form, then item_name alone
                    elif combined_lower in _BOX_ITEM_ALIASES:
                        raw_value = _BOX_ITEM_ALIASES[combined_lower]
                    elif item_lower in _BOX_ITEM_ALIASES:
                        raw_value = _BOX_ITEM_ALIASES[item_lower]

                    # Pass 3: brand/device keyword → Handset
                    elif any(kw in item_lower for kw in _HANDSET_KEYWORDS):
                        raw_value = "Handset"

                    # Pass 4: no match — use combined form for FK lookup
                    # (fuzzy may still catch it; if not → staging with clean string)
                    else:
                        raw_value = combined

                # Change 3c: silently drop Wi-Fi Hotspot from junction table
                if "wifi_technologies" in concrete_path and str(raw_value) in _WIFI_TECHNOLOGIES_EXCLUDE:
                    logger.debug(
                        "_resolve_array_fks: skipping excluded wifi tech %r at %s",
                        raw_value, concrete_path,
                    )
                    continue

                if is_camera_features:
                    # Section 2.3: alias-aware three-pass resolution
                    pk, corrected, resolution_type = resolve_camera_feature_value(
                        str(raw_value), brand_id
                    )
                else:
                    # Standard path: enforce band prefix then resolve
                    normalized_value = _normalize_band_value(str(raw_value), field_path)
                    pk, corrected, resolution_type = resolve_lookup_value(
                        normalized_value, table_path
                    )

                # resolution_type values that yield a valid pk:
                # 'exact', 'fuzzy', 'alias_brand_specific', 'alias_global'
                if resolution_type in ("exact", "alias_brand_specific", "alias_global"):
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
