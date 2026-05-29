# app/services/inference_rules_group_a.py
"""
Run C — Deterministic Inference Engine
Group A: Connectivity (India-specific) — 6 rules

Rules A1–A6. All rules in this group:
  - Read band data from mobile_specs.phone_network_bands + lookup_network_bands
  - Use India band sets from core/constants.py — never score on raw band count
  - Have confidence: high
  - Defer to RunB: None (reviewers never state 'covers 2 of Jio's 3 deployed bands')
  - Return the Section 8 rule return schema dict

Section 4 of Run_C_Inference_Engine_Spec.md.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.core.constants import (
    # 5G
    INDIA_5G_ALL_RELEVANT,
    JIO_5G_BANDS,
    AIRTEL_5G_BANDS,
    VI_5G_BANDS,
    BSNL_5G_BANDS,
    INDIA_5G_COVERAGE_BANDS,
    INDIA_5G_CA_BANDS,
    # 4G
    INDIA_4G_ALL_RELEVANT,
    INDIA_4G_PRIORITY_BANDS,
    INDIA_4G_COVERAGE_BANDS,
    JIO_4G_BANDS,
    AIRTEL_4G_BANDS,
    VI_4G_BANDS,
    BSNL_4G_BANDS,
)
from app.core.supabase_client import get_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal: fetch phone's 5G and 4G bands in one round-trip
# ---------------------------------------------------------------------------

async def _fetch_bands(model_id: int) -> tuple[set[str], set[str]]:
    """
    Returns (bands_5g, bands_4g) as sets of normalised band name strings.

    Normalisation: lowercase, strip spaces and hyphens.
    Examples: "N 78" → "n78",  "B 28" → "b28"

    Filters by network_type = '5G' or '4G' from lookup_network_bands.
    Satellite and other types are ignored.
    """
    client = get_client().schema("mobile_specs")
    res = await asyncio.to_thread(lambda: (
        client
        .table("phone_network_bands")
        .select("lookup_network_bands(band_name, network_type)")
        .eq("model_id", model_id)
        .execute()
    ))
    rows = res.data or []

    bands_5g: set[str] = set()
    bands_4g: set[str] = set()

    for row in rows:
        lookup = row.get("lookup_network_bands") or {}
        raw_name    = (lookup.get("band_name") or "").strip()
        network_type = (lookup.get("network_type") or "").upper()
        # Normalise: lowercase, no spaces, no hyphens
        name = raw_name.lower().replace(" ", "").replace("-", "")
        if not name:
            continue
        if network_type == "5G":
            bands_5g.add(name)
        elif network_type == "4G":
            bands_4g.add(name)

    return bands_5g, bands_4g


def _make_result(
    rule_key: str,
    structured_value: Any,
    inference_text: str,
    sentiment: str,
    confidence: str,
    input_snapshot: dict,
    conflict_flag: bool = False,
) -> dict:
    """Builds the standard Section 8 rule return schema."""
    return {
        "rule_key":               rule_key,
        "structured_value":       structured_value,
        "inference_text":         inference_text,
        "sentiment":              sentiment,          # Positive | Neutral | Negative
        "confidence":             confidence,         # high | medium | low
        "defers_to_runb_category": None,              # Group A: always None
        "input_field_snapshot":   input_snapshot,
        "conflict_flag":          conflict_flag,
    }


# ---------------------------------------------------------------------------
# A1. jio_5g_compatibility
# ---------------------------------------------------------------------------

async def rule_jio_5g_compatibility(model_id: int, url_registry_id: int) -> dict:
    """
    Intersects phone's 5G bands against JIO_5G_BANDS {n28, n78, n41, n5}.
    Gate: if n78 absent → 'incompatible'.

    Returns structured_value: 'excellent'|'very_good'|'good'|'basic'|'incompatible'
    Output column: jio_5g_tier
    """
    bands_5g, _ = await _fetch_bands(model_id)

    india_bands = bands_5g & INDIA_5G_ALL_RELEVANT
    snapshot = {
        "raw_5g_bands":    sorted(bands_5g),
        "india_5g_bands":  sorted(india_bands),
        "jio_5g_bands_req": sorted(JIO_5G_BANDS),
    }

    if "n78" not in india_bands:
        tier = "incompatible"
        text = (
            "This phone does not support n78 (3500 MHz), the make-or-break India 5G band. "
            "It will not connect to Jio 5G — or any Indian 5G network — regardless of how "
            "many 5G bands it lists. Those bands are US/Europe frequencies unused by Indian operators."
        )
        sentiment = "Negative"
    else:
        jio_covered = india_bands & JIO_5G_BANDS
        count = len(jio_covered)
        snapshot["jio_covered"] = sorted(jio_covered)

        if count == 4:
            tier = "excellent"
            text = (
                f"Supports all four Jio 5G bands (n28, n78, n41, n5). "
                f"Full coverage: n78 for urban 5G data, n41 for carrier aggregation in dense areas, "
                f"n28 and n5 for rural coverage and indoor penetration. No compromise."
            )
            sentiment = "Positive"
        elif count == 3:
            missing = sorted(JIO_5G_BANDS - jio_covered)
            tier = "very_good"
            text = (
                f"Supports 3 of 4 Jio 5G bands. Missing: {', '.join(missing)}. "
                f"Excellent Jio 5G compatibility — the missing band is a secondary booster."
            )
            sentiment = "Positive"
        elif count == 2:
            missing = sorted(JIO_5G_BANDS - jio_covered)
            tier = "good"
            text = (
                f"Supports 2 Jio 5G bands (n78 + one other). Missing: {', '.join(missing)}. "
                f"Will connect to Jio 5G in most areas but misses carrier aggregation or coverage band benefits."
            )
            sentiment = "Neutral"
        else:  # count == 1 (n78 only)
            missing = sorted(JIO_5G_BANDS - jio_covered)
            tier = "basic"
            text = (
                f"Supports only n78 (the primary Jio 5G band). Missing: {', '.join(missing)}. "
                f"Will connect to Jio 5G but without carrier aggregation (n41) or indoor/rural "
                f"coverage band (n28/n5). Urban speeds fine; rural coverage limited."
            )
            sentiment = "Neutral"

    snapshot["jio_5g_tier"] = tier
    return _make_result(
        rule_key         = "jio_5g_compatibility",
        structured_value = tier,
        inference_text   = text,
        sentiment        = sentiment,
        confidence       = "high",
        input_snapshot   = snapshot,
    )


# ---------------------------------------------------------------------------
# A2. airtel_5g_compatibility
# ---------------------------------------------------------------------------

async def rule_airtel_5g_compatibility(model_id: int, url_registry_id: int) -> dict:
    """
    Intersects phone's 5G bands against AIRTEL_5G_BANDS {n78, n1, n3, n40, n38, n8}.
    Gate: if n78 absent → 'incompatible'.

    Returns structured_value: 'excellent'|'good'|'basic'|'incompatible'
    Output column: airtel_5g_tier
    """
    bands_5g, _ = await _fetch_bands(model_id)

    india_bands = bands_5g & INDIA_5G_ALL_RELEVANT
    snapshot = {
        "raw_5g_bands":      sorted(bands_5g),
        "india_5g_bands":    sorted(india_bands),
        "airtel_5g_bands_req": sorted(AIRTEL_5G_BANDS),
    }

    if "n78" not in india_bands:
        tier = "incompatible"
        text = (
            "This phone does not support n78, the primary Airtel 5G band. "
            "It will not connect to Airtel 5G. Band count is irrelevant — "
            "n78 is the single gate for all four Indian 5G operators."
        )
        sentiment = "Negative"
    else:
        airtel_covered = india_bands & AIRTEL_5G_BANDS
        count = len(airtel_covered)
        snapshot["airtel_covered"] = sorted(airtel_covered)

        if count >= 4:
            tier = "excellent"
            text = (
                f"Supports {count} Airtel 5G bands ({', '.join(sorted(airtel_covered))}). "
                f"Full Airtel 5G compatibility: n78 for primary 5G, NSA anchor bands (n1/n3) "
                f"for network handoff stability, and CA bands (n40/n38) for urban throughput boosts."
            )
            sentiment = "Positive"
        elif count >= 2:
            missing = sorted(AIRTEL_5G_BANDS - airtel_covered)
            tier = "good"
            text = (
                f"Supports {count} Airtel 5G bands ({', '.join(sorted(airtel_covered))}). "
                f"Good Airtel 5G compatibility. Missing secondary bands: {', '.join(missing)}. "
                f"Will handle primary 5G coverage and one or more NSA anchors or CA bands."
            )
            sentiment = "Positive"
        else:  # count == 1 (n78 only)
            missing = sorted(AIRTEL_5G_BANDS - airtel_covered)
            tier = "basic"
            text = (
                f"Supports only n78 on Airtel 5G. Missing: {', '.join(missing)}. "
                f"Basic Airtel 5G — connects but without NSA anchor stability (n1/n3) "
                f"or carrier aggregation (n40/n38). May experience more frequent fallback to 4G."
            )
            sentiment = "Neutral"

    snapshot["airtel_5g_tier"] = tier
    return _make_result(
        rule_key         = "airtel_5g_compatibility",
        structured_value = tier,
        inference_text   = text,
        sentiment        = sentiment,
        confidence       = "high",
        input_snapshot   = snapshot,
    )


# ---------------------------------------------------------------------------
# A3. vi_5g_compatibility
# ---------------------------------------------------------------------------

async def rule_vi_5g_compatibility(model_id: int, url_registry_id: int) -> dict:
    """
    Intersects phone's 5G bands against VI_5G_BANDS {n78, n1, n3, n40}.
    Gate: if n78 absent → 'incompatible'.

    Note: Vi's 5G is NSA and limited to ~28 cities as of 2025.
    Returns structured_value: 'good'|'basic'|'incompatible'
    Output column: vi_5g_tier
    """
    bands_5g, _ = await _fetch_bands(model_id)

    india_bands = bands_5g & INDIA_5G_ALL_RELEVANT
    snapshot = {
        "raw_5g_bands":   sorted(bands_5g),
        "india_5g_bands": sorted(india_bands),
        "vi_5g_bands_req": sorted(VI_5G_BANDS),
    }

    if "n78" not in india_bands:
        tier = "incompatible"
        text = (
            "This phone does not support n78, which is required for Vi 5G. "
            "Vi's 5G network (NSA architecture) is built entirely on n78 as the primary carrier."
        )
        sentiment = "Negative"
    else:
        vi_covered = india_bands & VI_5G_BANDS
        count = len(vi_covered)
        snapshot["vi_covered"] = sorted(vi_covered)

        if count >= 3:
            tier = "good"
            text = (
                f"Supports {count} Vi 5G bands ({', '.join(sorted(vi_covered))}). "
                f"Good Vi 5G compatibility with n78 primary and NSA anchor band(s). "
                f"Note: Vi's 5G rollout is NSA architecture and currently covers approximately "
                f"28 cities as of 2025 — check local coverage before relying on 5G speeds."
            )
            sentiment = "Positive"
        else:
            # count >= 1 (n78 present, but only 1–2 bands)
            missing = sorted(VI_5G_BANDS - vi_covered)
            tier = "basic"
            text = (
                f"Supports {count} Vi 5G band(s) ({', '.join(sorted(vi_covered))}). "
                f"Will connect to Vi 5G where available. Missing NSA anchor bands: {', '.join(missing)}. "
                f"Vi's 5G is NSA architecture limited to ~28 cities (2025) — "
                f"4G fallback will be the primary experience outside those cities."
            )
            sentiment = "Neutral"

    snapshot["vi_5g_tier"] = tier
    return _make_result(
        rule_key         = "vi_5g_compatibility",
        structured_value = tier,
        inference_text   = text,
        sentiment        = sentiment,
        confidence       = "high",
        input_snapshot   = snapshot,
    )


# ---------------------------------------------------------------------------
# A4. bsnl_5g_compatibility
# ---------------------------------------------------------------------------

async def rule_bsnl_5g_compatibility(model_id: int, url_registry_id: int) -> dict:
    """
    BSNL special rule: n77 counts as n78-equivalent for BSNL only
    (3300–3800 MHz overlap). Gate: n78 OR n77 required.

    Intersects against BSNL_5G_BANDS {n28, n78, n77, n40, n1}.
    Returns structured_value: 'excellent'|'good'|'basic'|'incompatible'
    Output column: bsnl_5g_tier
    """
    bands_5g, _ = await _fetch_bands(model_id)

    india_bands = bands_5g & INDIA_5G_ALL_RELEVANT
    # For BSNL: n77 satisfies the n78 gate (3300–3800 overlap)
    effective_n78 = ("n78" in india_bands) or ("n77" in india_bands)

    snapshot = {
        "raw_5g_bands":     sorted(bands_5g),
        "india_5g_bands":   sorted(india_bands),
        "bsnl_5g_bands_req": sorted(BSNL_5G_BANDS),
        "effective_n78":    effective_n78,
        "has_n77":          "n77" in india_bands,
    }

    if not effective_n78:
        tier = "incompatible"
        text = (
            "This phone supports neither n78 nor n77, which are required for BSNL 5G. "
            "BSNL uses n77 (3300–3800 MHz, overlapping with n78) for its SA 5G network."
        )
        sentiment = "Negative"
    else:
        bsnl_covered = india_bands & BSNL_5G_BANDS
        count = len(bsnl_covered)
        snapshot["bsnl_covered"] = sorted(bsnl_covered)

        if count >= 3:
            tier = "excellent"
            text = (
                f"Supports {count} BSNL 5G bands ({', '.join(sorted(bsnl_covered))}). "
                f"Excellent BSNL 5G compatibility. "
                f"{'n77 (BSNL-equivalent of n78) present. ' if 'n77' in india_bands else ''}"
                f"BSNL runs SA (Standalone) 5G architecture — technically superior to NSA "
                f"(better latency, no 4G anchor dependency) though network rollout is at ~100k "
                f"towers as of 2025 and primarily covers rural/semi-urban areas."
            )
            sentiment = "Positive"
        elif count >= 2:
            missing = sorted(BSNL_5G_BANDS - bsnl_covered)
            tier = "good"
            text = (
                f"Supports {count} BSNL 5G bands ({', '.join(sorted(bsnl_covered))}). "
                f"Good BSNL 5G compatibility. Missing: {', '.join(missing)}. "
                f"BSNL SA 5G rollout ongoing (~100k towers, 2025)."
            )
            sentiment = "Positive"
        else:
            missing = sorted(BSNL_5G_BANDS - bsnl_covered)
            tier = "basic"
            text = (
                f"Supports {count} BSNL 5G band(s) ({', '.join(sorted(bsnl_covered))}). "
                f"Basic BSNL 5G. Missing: {', '.join(missing)}. "
                f"Will connect where BSNL SA 5G is deployed but without secondary band benefits."
            )
            sentiment = "Neutral"

    snapshot["bsnl_5g_tier"] = tier
    return _make_result(
        rule_key         = "bsnl_5g_compatibility",
        structured_value = tier,
        inference_text   = text,
        sentiment        = sentiment,
        confidence       = "high",
        input_snapshot   = snapshot,
    )


# ---------------------------------------------------------------------------
# A5. india_4g_band_coverage
# ---------------------------------------------------------------------------

async def rule_india_4g_band_coverage(model_id: int, url_registry_id: int) -> dict:
    """
    Intersects phone's 4G bands against INDIA_4G_ALL_RELEVANT, then scores
    against INDIA_4G_PRIORITY_BANDS and INDIA_4G_COVERAGE_BANDS.

    Also computes per-operator 4G band counts for snapshot.

    Returns two structured values:
      india_4g_coverage: 'comprehensive'|'solid'|'adequate'|'limited'
      has_low_band_4g:   bool  (low-band = rural/indoor coverage)

    Output columns: india_4g_coverage, has_low_band_4g
    """
    _, bands_4g = await _fetch_bands(model_id)

    india_4g = bands_4g & INDIA_4G_ALL_RELEVANT
    priority_covered  = india_4g & INDIA_4G_PRIORITY_BANDS
    coverage_covered  = india_4g & INDIA_4G_COVERAGE_BANDS
    count_priority    = len(priority_covered)
    has_low_band_4g   = len(coverage_covered) > 0

    # Per-operator 4G coverage for snapshot (traceability)
    jio_4g    = india_4g & JIO_4G_BANDS
    airtel_4g = india_4g & AIRTEL_4G_BANDS
    vi_4g     = india_4g & VI_4G_BANDS
    bsnl_4g   = india_4g & BSNL_4G_BANDS

    snapshot = {
        "raw_4g_bands":           sorted(bands_4g),
        "india_4g_bands":         sorted(india_4g),
        "priority_covered":       sorted(priority_covered),
        "coverage_bands_covered": sorted(coverage_covered),
        "count_priority":         count_priority,
        "has_low_band_4g":        has_low_band_4g,
        "per_operator_4g": {
            "jio":    {"covered": sorted(jio_4g),    "count": len(jio_4g)},
            "airtel": {"covered": sorted(airtel_4g), "count": len(airtel_4g)},
            "vi":     {"covered": sorted(vi_4g),     "count": len(vi_4g)},
            "bsnl":   {"covered": sorted(bsnl_4g),   "count": len(bsnl_4g)},
        },
    }

    # Tier assignment
    if count_priority >= 6:
        tier = "comprehensive"
        text = (
            f"Supports {count_priority} of {len(INDIA_4G_PRIORITY_BANDS)} priority India 4G bands "
            f"({', '.join(sorted(priority_covered))}). Comprehensive 4G coverage across all four "
            f"Indian operators. "
            f"{'Includes low-band 4G (' + ', '.join(sorted(coverage_covered)) + ') for rural/indoor penetration. ' if has_low_band_4g else ''}"
            f"Strong 4G fallback even in 5G dead zones."
        )
        sentiment = "Positive"
    elif count_priority >= 4:
        tier = "solid"
        missing = sorted(INDIA_4G_PRIORITY_BANDS - priority_covered)
        text = (
            f"Supports {count_priority} of {len(INDIA_4G_PRIORITY_BANDS)} priority India 4G bands. "
            f"Solid 4G coverage. Missing: {', '.join(missing)}. "
            f"{'Has low-band 4G for rural/indoor coverage. ' if has_low_band_4g else 'No low-band 4G — may struggle indoors or in rural areas. '}"
            f"Reliable 4G experience across major cities and operators."
        )
        sentiment = "Positive"
    elif count_priority >= 2:
        tier = "adequate"
        missing = sorted(INDIA_4G_PRIORITY_BANDS - priority_covered)
        text = (
            f"Supports {count_priority} of {len(INDIA_4G_PRIORITY_BANDS)} priority India 4G bands. "
            f"Adequate 4G coverage — works in urban areas on primary bands. "
            f"Missing: {', '.join(missing)}. "
            f"{'No low-band 4G — indoor and rural coverage will be weaker.' if not has_low_band_4g else 'Has some low-band 4G coverage.'}"
        )
        sentiment = "Neutral"
    else:
        tier = "limited"
        missing = sorted(INDIA_4G_PRIORITY_BANDS - priority_covered)
        text = (
            f"Supports only {count_priority} priority India 4G band(s). "
            f"Limited 4G coverage — likely missing critical bands for one or more operators. "
            f"Missing: {', '.join(missing)}. Verify compatibility with your operator before purchase."
        )
        sentiment = "Negative"

    snapshot["india_4g_coverage"] = tier
    return _make_result(
        rule_key         = "india_4g_band_coverage",
        structured_value = {"india_4g_coverage": tier, "has_low_band_4g": has_low_band_4g},
        inference_text   = text,
        sentiment        = sentiment,
        confidence       = "high",
        input_snapshot   = snapshot,
    )


# ---------------------------------------------------------------------------
# A6. sim_connectivity_profile
# ---------------------------------------------------------------------------

async def rule_sim_connectivity_profile(model_id: int, url_registry_id: int) -> dict:
    """
    Reads SIM count, eSIM support, SIM configuration, and expandable storage.
    Determines if microSD slot is dedicated (no sacrifice of SIM2 slot).

    Returns structured_value JSONB:
      {dual_sim: bool, esim: bool, storage_expandable_without_sacrifice: bool}
    Output column: sim_profile
    """
    ms_client = get_client().schema("mobile_specs")

    # Fetch network row (SIM info)
    net_res = await asyncio.to_thread(lambda: (
        ms_client
        .table("network")
        .select("number_of_sims, esim_support, sim_config_id, lookup_sim_configurations(configuration_name)")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    ))
    net_rows = net_res.data or []

    # Fetch base variant expandable storage
    var_res = await asyncio.to_thread(lambda: (
        ms_client
        .table("variant")
        .select("expandable_storage, storage_capacity")
        .eq("model_id", model_id)
        .eq("is_base_variant", True)
        .limit(1)
        .execute()
    ))
    var_rows = var_res.data or []

    # Parse
    num_sims         = 0
    esim_support     = False
    sim_config_name  = ""
    expandable       = False

    if net_rows:
        row              = net_rows[0]
        num_sims         = int(row.get("number_of_sims") or 0)
        esim_support     = bool(row.get("esim_support") or False)
        lookup           = row.get("lookup_sim_configurations") or {}
        sim_config_name  = (lookup.get("configuration_name") or "").lower()

    if var_rows:
        expandable = bool(var_rows[0].get("expandable_storage") or False)

    dual_sim     = num_sims >= 2
    # Hybrid SIM = shared SIM2/microSD slot — microSD sacrifices SIM2
    is_hybrid    = "hybrid" in sim_config_name
    # "Without sacrifice": expandable AND not hybrid (dedicated microSD slot)
    storage_without_sacrifice = expandable and not is_hybrid

    profile = {
        "dual_sim":                            dual_sim,
        "esim":                                esim_support,
        "storage_expandable_without_sacrifice": storage_without_sacrifice,
    }

    snapshot = {
        "number_of_sims":        num_sims,
        "esim_support":          esim_support,
        "sim_configuration":     sim_config_name,
        "expandable_storage":    expandable,
        "is_hybrid_sim":         is_hybrid,
        "sim_profile":           profile,
    }

    # Build narrative
    parts = []
    if dual_sim:
        if esim_support:
            parts.append("Dual SIM + eSIM — maximum operator flexibility, can run physical and virtual SIM simultaneously")
        else:
            parts.append("Dual physical SIM (Nano-SIM) — use two operators simultaneously")
    elif esim_support:
        parts.append("eSIM supported — switch operators digitally without physical SIM swap")
    else:
        parts.append("Single SIM only — one operator at a time")

    if is_hybrid:
        parts.append(
            "Hybrid SIM tray: SIM2 slot shared with microSD. "
            "You must choose between dual SIM and expandable storage — cannot have both simultaneously"
        )
        sentiment = "Neutral"
    elif expandable:
        parts.append("Dedicated microSD slot — expandable storage without sacrificing a SIM slot")
        sentiment = "Positive"
    else:
        parts.append("No microSD slot — storage is fixed at launch capacity")
        sentiment = "Neutral"

    text = ". ".join(parts) + "."
    # Upgrade sentiment if dual_sim + no hybrid sacrifice
    if dual_sim and storage_without_sacrifice:
        sentiment = "Positive"
    elif not dual_sim and not esim_support:
        sentiment = "Neutral"

    return _make_result(
        rule_key         = "sim_connectivity_profile",
        structured_value = profile,
        inference_text   = text,
        sentiment        = sentiment,
        confidence       = "high",
        input_snapshot   = snapshot,
    )


# ---------------------------------------------------------------------------
# Group A handler registry — imported by the orchestrator
# ---------------------------------------------------------------------------

GROUP_A_HANDLERS: dict[str, Any] = {
    "jio_5g_compatibility":    rule_jio_5g_compatibility,
    "airtel_5g_compatibility": rule_airtel_5g_compatibility,
    "vi_5g_compatibility":     rule_vi_5g_compatibility,
    "bsnl_5g_compatibility":   rule_bsnl_5g_compatibility,
    "india_4g_band_coverage":  rule_india_4g_band_coverage,
    "sim_connectivity_profile": rule_sim_connectivity_profile,
}
