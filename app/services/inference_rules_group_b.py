# app/services/inference_rules_group_b.py
"""
Run C — Deterministic Inference Engine
Group B: Performance & Gaming — 4 rules

Rules B1–B4:
  B1: chipset_tier            — lookup pipeline.chipset_tier_map
  B2: gaming_capability       — chipset + refresh_rate + vapor_chamber + RAM
  B3: memory_performance      — RAM tier + storage speed class
  B4: multitasking_longevity  — RAM + storage class + chipset vs segment peers

Confidence: B1=high(if found)/low(if missing), B2=medium, B3=high, B4=medium
Defers to RunB: Performance (B2, B4 narrative suppressed if RunB covers it)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.core.supabase_client import get_client
from app.services.inference_engine import classify_price_tier, percentile_to_tier

logger = logging.getLogger(__name__)

_MS  = lambda: get_client().schema("mobile_specs")
_PL  = lambda: get_client().schema("pipeline")

_RUNS_B_CATEGORY = "Performance"


def _make_result(
    rule_key: str,
    structured_value: Any,
    inference_text: str,
    sentiment: str,
    confidence: str,
    input_snapshot: dict,
    defers_to_runb: str | None = None,
    conflict_flag: bool = False,
) -> dict:
    return {
        "rule_key":                rule_key,
        "structured_value":        structured_value,
        "inference_text":          inference_text,
        "sentiment":               sentiment,
        "confidence":              confidence,
        "defers_to_runb_category": defers_to_runb,
        "input_field_snapshot":    input_snapshot,
        "conflict_flag":           conflict_flag,
    }


# ---------------------------------------------------------------------------
# B1. chipset_tier
# ---------------------------------------------------------------------------

async def rule_chipset_tier(model_id: int, url_registry_id: int) -> dict:
    """
    Looks up phone's chipset in pipeline.chipset_tier_map.
    Returns tier: 'flagship'|'upper_mid'|'mid'|'entry'
    confidence: 'high' if found, 'low' if missing (gap logged separately).
    Output column: chipset_tier
    """
    # Fetch chipset name via phones → chipsets join
    res = await asyncio.to_thread(lambda: (
        _MS()
        .table("phones")
        .select("chipset_id, chipsets(chipset_name)")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    ))
    rows = res.data or []
    if not rows:
        return _make_result(
            rule_key="chipset_tier", structured_value=None,
            inference_text="Unable to determine chipset tier — no phone record found.",
            sentiment="Neutral", confidence="low",
            input_snapshot={"model_id": model_id, "error": "no_phone_row"},
        )

    chipsets_join = rows[0].get("chipsets") or {}
    chipset_name  = (chipsets_join.get("chipset_name") or "").strip()

    snapshot = {"chipset_name": chipset_name}

    if not chipset_name:
        return _make_result(
            rule_key="chipset_tier", structured_value=None,
            inference_text="Chipset information unavailable — tier cannot be determined.",
            sentiment="Neutral", confidence="low",
            input_snapshot=snapshot,
        )

    # Lookup in pipeline.chipset_tier_map
    tier_res = await asyncio.to_thread(lambda: (
        _PL()
        .table("chipset_tier_map")
        .select("tier, antutu_band_low, antutu_band_high, notes")
        .eq("chipset_name", chipset_name)
        .limit(1)
        .execute()
    ))
    tier_rows = tier_res.data or []
    snapshot["chipset_name"] = chipset_name

    if not tier_rows:
        logger.warning("Run C B1: chipset_name=%r not found in chipset_tier_map (model_id=%d)", chipset_name, model_id)
        return _make_result(
            rule_key="chipset_tier", structured_value=None,
            inference_text=f"Chipset '{chipset_name}' not yet mapped to a tier. Performance tier unknown.",
            sentiment="Neutral", confidence="low",
            input_snapshot=snapshot,
        )

    tier_row = tier_rows[0]
    tier     = tier_row["tier"]
    snapshot.update({
        "tier": tier,
        "antutu_band": f"{tier_row.get('antutu_band_low', '?')}–{tier_row.get('antutu_band_high', '?')}",
        "notes": tier_row.get("notes", ""),
    })

    tier_labels = {
        "flagship":  ("the top tier", "Positive"),
        "upper_mid": ("upper-mid tier", "Positive"),
        "mid":       ("mid-range tier", "Neutral"),
        "entry":     ("entry-level tier", "Neutral"),
    }
    label, sentiment = tier_labels.get(tier, (tier, "Neutral"))

    tier_narratives = {
        "flagship": (
            f"Powered by the {chipset_name} — a current-generation flagship SoC. "
            f"Delivers peak performance for any workload: gaming, heavy multitasking, "
            f"video editing, and AI inference."
        ),
        "upper_mid": (
            f"Powered by the {chipset_name} — an upper-mid range SoC delivering "
            f"near-flagship performance at a lower cost. Handles demanding tasks "
            f"comfortably with only minor trade-offs versus flagship silicon."
        ),
        "mid": (
            f"Powered by the {chipset_name} — a mid-range SoC suitable for everyday "
            f"tasks, social media, and casual gaming. May show frame drops in "
            f"graphically intensive titles at maximum settings."
        ),
        "entry": (
            f"Powered by the {chipset_name} — an entry-level SoC optimised for "
            f"basic daily tasks. Not suited for demanding gaming or heavy multitasking. "
            f"Expect performance trade-offs compared to mid-range alternatives."
        ),
    }
    text = tier_narratives.get(tier, f"Chipset tier: {tier}.")

    return _make_result(
        rule_key="chipset_tier", structured_value=tier,
        inference_text=text, sentiment=sentiment, confidence="high",
        input_snapshot=snapshot,
    )


# ---------------------------------------------------------------------------
# B2. gaming_capability
# ---------------------------------------------------------------------------

async def rule_gaming_capability(model_id: int, url_registry_id: int) -> dict:
    """
    Signals: chipset_tier + refresh_rate + has_vapor_chamber + ram_capacity (base variant).
    Produces both gaming_tier_absolute and gaming_tier_in_segment.
    confidence: medium (gaming is multi-dimensional; no benchmark data available).
    Output columns: gaming_tier_absolute, gaming_tier_in_segment
    """
    # Chipset tier from B1 output already committed to inferred_specs
    tier_res = await asyncio.to_thread(lambda: (
        _PL()
        .table("inferred_specs")
        .select("chipset_tier")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    ))
    tier_rows = tier_res.data or []
    chipset_tier = (tier_rows[0].get("chipset_tier") if tier_rows else None) or "unknown"

    # Refresh rate
    disp_res = await asyncio.to_thread(lambda: (
        _MS()
        .table("phone_displays")
        .select("refresh_rate, display_position")
        .eq("model_id", model_id)
        .execute()
    ))
    disp_rows = disp_res.data or []
    refresh_rate = 60
    for d in disp_rows:
        if (d.get("display_position") or "Primary").lower() == "primary":
            refresh_rate = int(d.get("refresh_rate") or 60)
            break
    if disp_rows and refresh_rate == 60:
        refresh_rate = int(disp_rows[0].get("refresh_rate") or 60)

    # Vapor chamber from body_features.other_features
    body_res = await asyncio.to_thread(lambda: (
        _MS()
        .table("body_features")
        .select("other_features")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    ))
    body_rows = body_res.data or []
    other_feats  = (body_rows[0].get("other_features") or "") if body_rows else ""
    has_vapor    = "vapor" in other_feats.lower()

    # RAM (base variant)
    var_res = await asyncio.to_thread(lambda: (
        _MS()
        .table("variant")
        .select("ram_capacity")
        .eq("model_id", model_id)
        .eq("is_base_variant", True)
        .limit(1)
        .execute()
    ))
    var_rows = var_res.data or []
    ram_gb = int(var_rows[0].get("ram_capacity") or 4) if var_rows else 4

    snapshot = {
        "chipset_tier":    chipset_tier,
        "refresh_rate_hz": refresh_rate,
        "has_vapor_chamber": has_vapor,
        "ram_gb":          ram_gb,
    }

    # Score: chipset contributes 60%, refresh 25%, vapor 10%, RAM 5%
    chip_score = {"flagship": 1.0, "upper_mid": 0.75, "mid": 0.45, "entry": 0.20}.get(chipset_tier, 0.30)
    ref_score  = min(1.0, (refresh_rate - 60) / (165 - 60)) if refresh_rate > 60 else 0.0
    vap_score  = 0.10 if has_vapor else 0.0
    ram_score  = min(1.0, max(0.0, (ram_gb - 4) / (16 - 4)))
    total      = 0.60 * chip_score + 0.25 * ref_score + 0.10 * vap_score + 0.05 * ram_score

    if total >= 0.75:
        tier_abs = "excellent"
        sentiment = "Positive"
    elif total >= 0.50:
        tier_abs = "good"
        sentiment = "Positive"
    elif total >= 0.30:
        tier_abs = "adequate"
        sentiment = "Neutral"
    else:
        tier_abs = "weak"
        sentiment = "Neutral"

    snapshot["gaming_score"] = round(total, 3)
    snapshot["gaming_tier_absolute"] = tier_abs

    tier_narratives = {
        "excellent": (
            f"Strong gaming hardware: {chipset_tier} SoC, {refresh_rate}Hz display"
            f"{', vapor chamber cooling' if has_vapor else ''}, {ram_gb}GB RAM. "
            f"Handles all major titles at high/ultra settings with sustained frame rates."
        ),
        "good": (
            f"Capable gaming phone: {chipset_tier} SoC, {refresh_rate}Hz display, {ram_gb}GB RAM. "
            f"Runs popular titles at high settings."
            f"{' Vapor chamber helps with sustained gaming sessions.' if has_vapor else ''}"
        ),
        "adequate": (
            f"Adequate for casual gaming: {chipset_tier} SoC, {refresh_rate}Hz display, {ram_gb}GB RAM. "
            f"Expect reduced settings or frame rates in demanding titles."
        ),
        "weak": (
            f"Limited gaming capability: {chipset_tier} SoC, {refresh_rate}Hz display, {ram_gb}GB RAM. "
            f"Suitable for lightweight games only. Demanding 3D titles will struggle."
        ),
    }
    text = tier_narratives.get(tier_abs, "")

    # ── Segment-relative gaming tier ─────────────────────────────────────────
    # Compare against peers already committed in the same price tier.
    # Uses classify_price_tier on launch_price directly (cannot read
    # inferred_specs.price_segment — B runs before H1 writes that column).
    # Guards at MIN_PEERS=4; falls back to absolute tier for early catalogue.
    _TIER_RANK_B2 = {"weak": 0, "adequate": 1, "good": 2, "excellent": 3}
    tier_in_seg = tier_abs
    try:
        lp_b2_res = await asyncio.to_thread(lambda: (
            _MS()
            .table("variant")
            .select("launch_price")
            .eq("model_id", model_id)
            .eq("is_base_variant", True)
            .limit(1)
            .execute()
        ))
        lp_b2_rows = lp_b2_res.data or []
        launch_price_b2 = float(lp_b2_rows[0]["launch_price"]) if lp_b2_rows and lp_b2_rows[0].get("launch_price") else None
        price_seg_b2 = classify_price_tier(launch_price_b2) if launch_price_b2 else None

        if price_seg_b2:
            peers_b2 = await asyncio.to_thread(lambda: (
                _PL()
                .table("inferred_specs")
                .select("gaming_tier_absolute")
                .eq("price_segment", price_seg_b2)
                .neq("model_id", model_id)
                .not_.is_("gaming_tier_absolute", "null")
                .execute()
            ))
            peer_ranks_b2 = [
                _TIER_RANK_B2.get(r.get("gaming_tier_absolute"), 1)
                for r in (peers_b2.data or [])
            ]
            if len(peer_ranks_b2) >= 4:
                this_rank_b2 = _TIER_RANK_B2.get(tier_abs, 1)
                beaten_b2 = sum(1 for pr in peer_ranks_b2 if pr <= this_rank_b2)
                pct_b2 = beaten_b2 / len(peer_ranks_b2)
                tier_in_seg = percentile_to_tier(pct_b2)
                snapshot["seg_peers"] = len(peer_ranks_b2)
                snapshot["seg_percentile"] = round(pct_b2, 3)
    except Exception as _exc_b2:
        logger.debug("B2: segment percentile computation failed: %s", _exc_b2)

    snapshot["gaming_tier_in_segment"] = tier_in_seg

    return _make_result(
        rule_key="gaming_capability",
        structured_value={
            "gaming_tier_absolute":   tier_abs,
            "gaming_tier_in_segment": tier_in_seg,
        },
        inference_text=text, sentiment=sentiment, confidence="medium",
        input_snapshot=snapshot, defers_to_runb=_RUNS_B_CATEGORY,
    )


# ---------------------------------------------------------------------------
# B3. memory_performance
# ---------------------------------------------------------------------------

async def rule_memory_performance(model_id: int, url_registry_id: int) -> dict:
    """
    RAM capacity (base variant, physical only — no virtual) + storage type speed class.
    Output columns: ram_tier_in_segment, storage_speed_class
    """
    # RAM (base variant, physical only)
    var_res = await asyncio.to_thread(lambda: (
        _MS()
        .table("variant")
        .select("ram_capacity, storage_type_id, lookup_storage_types(storage_type)")
        .eq("model_id", model_id)
        .eq("is_base_variant", True)
        .limit(1)
        .execute()
    ))
    var_rows = var_res.data or []
    if not var_rows:
        return _make_result(
            rule_key="memory_performance", structured_value=None,
            inference_text="Memory/storage data unavailable.",
            sentiment="Neutral", confidence="low",
            input_snapshot={"model_id": model_id},
        )

    row           = var_rows[0]
    ram_gb        = int(row.get("ram_capacity") or 4)
    storage_join  = row.get("lookup_storage_types") or {}
    storage_type  = (storage_join.get("storage_type") or "Unknown").strip()

    # Storage speed class
    _STORAGE_CLASS = {
        "UFS 4.1": "ultra_fast", "UFS 4.0": "ultra_fast", "NVMe PCIe 4.0": "ultra_fast",
        "NVMe PCIe 3.0": "fast",  "NVMe": "fast",
        "UFS 3.1": "fast",        "UFS 3.0": "fast",
        "UFS 2.2": "standard",    "UFS 2.1": "standard", "UFS 2.0": "standard",
        "eMMC 5.1": "slow",       "eMMC 5.0": "slow",    "eMMC 4.5": "slow",
    }
    storage_class = _STORAGE_CLASS.get(storage_type, "standard")

    # RAM tier
    if ram_gb >= 12:
        ram_tier = "high"
    elif ram_gb >= 8:
        ram_tier = "comfortable"
    elif ram_gb >= 6:
        ram_tier = "adequate"
    else:
        ram_tier = "limited"

    snapshot = {
        "ram_gb":          ram_gb,
        "ram_tier":        ram_tier,
        "storage_type":    storage_type,
        "storage_class":   storage_class,
    }

    # Sentiment
    score = (
        {"high": 1.0, "comfortable": 0.75, "adequate": 0.50, "limited": 0.25}.get(ram_tier, 0.5)
        + {"ultra_fast": 1.0, "fast": 0.75, "standard": 0.50, "slow": 0.25}.get(storage_class, 0.5)
    ) / 2.0
    sentiment = "Positive" if score >= 0.65 else ("Neutral" if score >= 0.45 else "Negative")

    ram_texts = {
        "high":        f"{ram_gb}GB RAM — keeps many apps open simultaneously without reloading",
        "comfortable": f"{ram_gb}GB RAM — comfortable multitasking for most users",
        "adequate":    f"{ram_gb}GB RAM — adequate for everyday use, may reload background apps",
        "limited":     f"{ram_gb}GB RAM — limited RAM; aggressive background app killing expected",
    }
    stor_texts = {
        "ultra_fast": f"{storage_type} storage (ultra-fast): near-instant app loads, quick file transfers",
        "fast":       f"{storage_type} storage (fast): good app load times and file performance",
        "standard":   f"{storage_type} storage (standard): adequate for daily tasks",
        "slow":       f"eMMC storage (slow): noticeably slower app load times versus UFS alternatives",
    }
    text = f"{ram_texts[ram_tier]}. {stor_texts[storage_class]}."

    return _make_result(
        rule_key="memory_performance",
        structured_value={"ram_tier_in_segment": ram_tier, "storage_speed_class": storage_class},
        inference_text=text, sentiment=sentiment, confidence="high",
        input_snapshot=snapshot,
    )


# ---------------------------------------------------------------------------
# B4. multitasking_longevity
# ---------------------------------------------------------------------------

async def rule_multitasking_longevity(model_id: int, url_registry_id: int) -> dict:
    """
    Combined RAM + storage speed class + chipset_tier → multitasking_longevity tier.
    This is the "will this phone still feel snappy in 2–3 years?" signal.
    Output column: multitasking_longevity
    """
    # Read from already-committed B1/B3 inferred outputs
    inf_res = await asyncio.to_thread(lambda: (
        _PL()
        .table("inferred_specs")
        .select("chipset_tier, ram_tier_in_segment, storage_speed_class")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    ))
    inf_rows = inf_res.data or []
    if not inf_rows:
        return _make_result(
            rule_key="multitasking_longevity", structured_value=None,
            inference_text="Insufficient data to evaluate multitasking longevity.",
            sentiment="Neutral", confidence="low",
            input_snapshot={"model_id": model_id},
        )

    row           = inf_rows[0]
    chipset_tier  = row.get("chipset_tier") or "unknown"
    ram_tier      = row.get("ram_tier_in_segment") or "adequate"
    storage_class = row.get("storage_speed_class") or "standard"

    chip_score = {"flagship": 1.0, "upper_mid": 0.75, "mid": 0.50, "entry": 0.20}.get(chipset_tier, 0.30)
    ram_score  = {"high": 1.0, "comfortable": 0.75, "adequate": 0.50, "limited": 0.20}.get(ram_tier, 0.50)
    stor_score = {"ultra_fast": 1.0, "fast": 0.75, "standard": 0.50, "slow": 0.20}.get(storage_class, 0.50)

    combined = 0.50 * chip_score + 0.35 * ram_score + 0.15 * stor_score

    if combined >= 0.75:
        longevity = "excellent"
        sentiment = "Positive"
        text = (
            f"Strong longevity profile — {chipset_tier} SoC, {ram_tier} RAM, {storage_class} storage. "
            f"Likely to remain responsive for 4+ years with software updates."
        )
    elif combined >= 0.55:
        longevity = "good"
        sentiment = "Positive"
        text = (
            f"Good longevity — {chipset_tier} SoC, {ram_tier} RAM, {storage_class} storage. "
            f"Should stay snappy for 3 years of typical use."
        )
    elif combined >= 0.35:
        longevity = "fair"
        sentiment = "Neutral"
        text = (
            f"Fair longevity — {chipset_tier} SoC, {ram_tier} RAM. "
            f"May show slowdowns after 2–3 years of OS updates and heavier apps."
        )
    else:
        longevity = "limited"
        sentiment = "Negative"
        text = (
            f"Limited longevity — {chipset_tier} SoC, {ram_tier} RAM. "
            f"Likely to feel sluggish within 1–2 years as apps and OS requirements grow."
        )

    snapshot = {
        "chipset_tier": chipset_tier, "ram_tier": ram_tier,
        "storage_class": storage_class, "combined_score": round(combined, 3),
        "multitasking_longevity": longevity,
    }

    return _make_result(
        rule_key="multitasking_longevity", structured_value=longevity,
        inference_text=text, sentiment=sentiment, confidence="medium",
        input_snapshot=snapshot, defers_to_runb=_RUNS_B_CATEGORY,
    )


# ---------------------------------------------------------------------------
# Group B handler registry
# ---------------------------------------------------------------------------

GROUP_B_HANDLERS: dict[str, Any] = {
    "chipset_tier":            rule_chipset_tier,
    "gaming_capability":       rule_gaming_capability,
    "memory_performance":      rule_memory_performance,
    "multitasking_longevity":  rule_multitasking_longevity,
}
