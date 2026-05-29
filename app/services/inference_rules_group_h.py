# app/services/inference_rules_group_h.py
"""
Run C — Deterministic Inference Engine
Group H: Derived Positioning — 6 rules (RUN LAST, consume A–G output)

These rules do NOT read raw mobile_specs. They consume structured outputs from
pipeline.inferred_specs (written by Groups A–G) plus variant.launch_price.

Execution order enforced by orchestrator:
    H1 → H2 → H3 → H4 → H5 → H6

H1 must always run first (price_segment required by H2–H6 and F1 rare_for_segment).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.core.constants import PRICE_TIERS
from app.core.supabase_client import get_client
from app.services.inference_engine import classify_price_tier

logger = logging.getLogger(__name__)

_MS = lambda: get_client().schema("mobile_specs")
_PL = lambda: get_client().schema("pipeline")

# Ordered tier list (low→high) for comparison arithmetic
_TIER_ORDER = [
    "ULTRA_BUDGET", "ENTRY", "BUDGET",
    "LOWER_MIDRANGE", "UPPER_MIDRANGE",
    "PREMIUM_MIDRANGE", "FLAGSHIP", "ULTRA_FLAGSHIP",
]

# Expected chipset tier per price segment for H2
_SEGMENT_EXPECTED_CHIPSET = {
    "ULTRA_FLAGSHIP":   "flagship",
    "FLAGSHIP":         "flagship",
    "PREMIUM_MIDRANGE": "upper_mid",
    "UPPER_MIDRANGE":   "mid",
    "LOWER_MIDRANGE":   "mid",
    "BUDGET":           "mid",
    "ENTRY":            "entry",
    "ULTRA_BUDGET":     "entry",
}

_CHIPSET_RANK = {"flagship": 4, "upper_mid": 3, "mid": 2, "entry": 1}


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


async def _fetch_launch_price(model_id: int) -> float | None:
    res = await asyncio.to_thread(lambda: (
        _MS()
        .table("variant")
        .select("launch_price")
        .eq("model_id", model_id)
        .eq("is_base_variant", True)
        .limit(1)
        .execute()
    ))
    rows = res.data or []
    if rows and rows[0].get("launch_price") is not None:
        return float(rows[0]["launch_price"])
    return None


async def _fetch_inferred(model_id: int) -> dict:
    """Reads the full inferred_specs row written by Groups A–G."""
    res = await asyncio.to_thread(lambda: (
        _PL()
        .table("inferred_specs")
        .select("*")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    ))
    rows = res.data or []
    return rows[0] if rows else {}


# ---------------------------------------------------------------------------
# H1. price_segment
# ---------------------------------------------------------------------------

async def rule_price_segment(model_id: int, url_registry_id: int) -> dict:
    """
    H1: Classify base variant launch price into canonical PRICE_TIERS segment.
    Output column: price_segment  TEXT
    """
    price = await _fetch_launch_price(model_id)
    snapshot = {"launch_price_inr": price}

    if price is None:
        return _make_result(
            rule_key="price_segment", structured_value=None,
            inference_text="Launch price unavailable — cannot determine price segment.",
            sentiment="Neutral", confidence="low",
            input_snapshot=snapshot,
        )

    segment = classify_price_tier(price)

    # Build human-readable label
    tier_labels = {
        "ULTRA_FLAGSHIP":   f"Ultra-flagship (above ₹1,20,000)",
        "FLAGSHIP":         f"Flagship (₹80,000–₹1,20,000)",
        "PREMIUM_MIDRANGE": f"Premium mid-range / Flagship killer (₹50,000–₹80,000)",
        "UPPER_MIDRANGE":   f"Upper mid-range (₹35,000–₹50,000)",
        "LOWER_MIDRANGE":   f"Lower mid-range (₹25,000–₹35,000)",
        "BUDGET":           f"Budget (₹15,000–₹25,000)",
        "ENTRY":            f"Entry-level (₹5,000–₹15,000)",
        "ULTRA_BUDGET":     f"Ultra-budget (below ₹5,000)",
    }
    label = tier_labels.get(segment, segment)
    text  = f"Priced at ₹{int(price):,} — {label} segment."
    snapshot["price_segment"] = segment

    return _make_result(
        rule_key="price_segment", structured_value=segment,
        inference_text=text, sentiment="Neutral", confidence="high",
        input_snapshot=snapshot,
    )


# ---------------------------------------------------------------------------
# H2. price_performance_verdict
# ---------------------------------------------------------------------------

async def rule_price_performance_verdict(model_id: int, url_registry_id: int) -> dict:
    """
    H2: Chipset tier vs price segment → over_spec | well_rounded | brand_premium.
    over_spec    → chipset one+ tier above segment expectation (flagship killer)
    well_rounded → chipset matches segment expectation
    brand_premium→ chipset tier below segment expectation (paying for brand/design)
    Output column: price_performance  TEXT
    """
    inf = await _fetch_inferred(model_id)
    chipset_tier   = inf.get("chipset_tier")
    price_segment  = inf.get("price_segment")

    snapshot = {"chipset_tier": chipset_tier, "price_segment": price_segment}

    if not chipset_tier or not price_segment:
        return _make_result(
            rule_key="price_performance_verdict", structured_value=None,
            inference_text="Cannot compute price-performance — chipset_tier or price_segment missing.",
            sentiment="Neutral", confidence="low",
            input_snapshot=snapshot,
        )

    expected       = _SEGMENT_EXPECTED_CHIPSET.get(price_segment, "mid")
    actual_rank    = _CHIPSET_RANK.get(chipset_tier, 2)
    expected_rank  = _CHIPSET_RANK.get(expected, 2)
    delta          = actual_rank - expected_rank

    snapshot.update({"expected_chipset": expected, "delta": delta})

    if delta >= 1:
        verdict   = "over_spec"
        sentiment = "Positive"
        text = (
            f"Strong price-performance: the {chipset_tier} chipset is above the "
            f"typical {expected} tier expected at {price_segment.replace('_', ' ').lower()} prices. "
            f"More silicon for your money — a flagship-killer pattern popular with Indian buyers."
        )
    elif delta == 0:
        verdict   = "well_rounded"
        sentiment = "Neutral"
        text = (
            f"Well-rounded for its price: the {chipset_tier} chipset matches expectations "
            f"for the {price_segment.replace('_', ' ').lower()} segment. "
            f"Value comes from balance across specs, not a single dominant advantage."
        )
    else:
        verdict   = "brand_premium"
        sentiment = "Neutral"
        text = (
            f"Brand premium pricing: the {chipset_tier} chipset is below the "
            f"{expected} tier typically seen at this price. "
            f"You're paying for brand, design, or ecosystem — not raw silicon performance."
        )

    snapshot["price_performance"] = verdict
    return _make_result(
        rule_key="price_performance_verdict", structured_value=verdict,
        inference_text=text, sentiment=sentiment, confidence="medium",
        input_snapshot=snapshot,
    )


# ---------------------------------------------------------------------------
# H3. use_case_fitness
# ---------------------------------------------------------------------------

async def rule_use_case_fitness(model_id: int, url_registry_id: int) -> dict:
    """
    H3: Per-activity fitness map from A–G inferred outputs.
    gaming, photography, battery, multimedia, productivity, portability
    Each → 'excellent' | 'good' | 'adequate' | 'weak'
    Output column: use_case_fitness  JSONB
    """
    inf = await _fetch_inferred(model_id)

    def _tier_to_score(tier: str | None) -> float:
        return {"excellent": 1.0, "very_good": 0.85, "good": 0.70,
                "adequate": 0.50, "fair": 0.40, "limited": 0.25,
                "weak": 0.15, "basic": 0.30, "premium": 0.90,
                "flagship": 1.0, "upper_mid": 0.75, "mid": 0.50, "entry": 0.25,
                "standard": 0.50, "comfortable": 0.65, "high": 0.80}.get(str(tier or ""), 0.50)

    def _score_to_tier(s: float) -> str:
        if s >= 0.80: return "excellent"
        if s >= 0.60: return "good"
        if s >= 0.40: return "adequate"
        return "weak"

    # Gaming: B2 absolute + chipset gate
    gaming_abs   = str(inf.get("gaming_tier_absolute") or "adequate")
    gaming_fit   = _score_to_tier(_tier_to_score(gaming_abs))

    # Photography: E1 main_camera_hw + versatility (low confidence always)
    main_hw      = str(inf.get("main_camera_hw") or "standard")
    cam_vers     = inf.get("camera_versatility") or {}
    has_tele     = bool(cam_vers.get("has_telephoto"))
    has_uw       = bool(cam_vers.get("has_ultrawide"))
    cam_score    = (_tier_to_score(main_hw) * 0.6
                    + (0.2 if has_tele else 0)
                    + (0.2 if has_uw  else 0))
    photo_fit    = _score_to_tier(cam_score)

    # Battery: D1 absolute
    batt_abs     = str(inf.get("endurance_tier_absolute") or "adequate")
    battery_fit  = _score_to_tier(_tier_to_score(batt_abs))

    # Multimedia: C5
    mm_tier      = str(inf.get("multimedia_tier") or "standard")
    mm_fit       = _score_to_tier(_tier_to_score(mm_tier))

    # Productivity: RAM tier + longevity + refresh class
    ram_tier     = str(inf.get("ram_tier_in_segment") or "adequate")
    longevity    = str(inf.get("multitasking_longevity") or "fair")
    refresh_cls  = str(inf.get("refresh_rate_class") or "standard")
    prod_score   = (
        _tier_to_score(ram_tier) * 0.45
        + _tier_to_score(longevity) * 0.35
        + (0.20 if refresh_cls in ("high", "ultra_high") else 0.10)
    )
    productivity_fit = _score_to_tier(prod_score)

    # Portability: F3
    port      = str(inf.get("portability") or "comfortable")
    port_map  = {
        "compact": "excellent", "comfortable": "good", "balanced": "good",
        "large_but_manageable": "adequate", "bulky": "adequate", "heavy_or_tall": "weak",
    }
    portability_fit = port_map.get(port, "adequate")

    fitness = {
        "gaming":       gaming_fit,
        "photography":  photo_fit,
        "battery":      battery_fit,
        "multimedia":   mm_fit,
        "productivity": productivity_fit,
        "portability":  portability_fit,
    }

    top    = [k for k, v in fitness.items() if v in ("excellent", "good")]
    bottom = [k for k, v in fitness.items() if v == "weak"]
    parts  = []
    if top:
        parts.append(f"Strongest in: {', '.join(top)}.")
    if bottom:
        parts.append(f"Weak at: {', '.join(bottom)}.")
    parts.append("Photography ratings are hardware estimates only — see reviewer scores for real-world quality.")
    text = " ".join(parts)

    sentiment = "Positive" if any(v in ("excellent", "good") for v in fitness.values()) else "Neutral"
    snapshot  = {"use_case_fitness": fitness}
    return _make_result(
        rule_key="use_case_fitness", structured_value=fitness,
        inference_text=text, sentiment=sentiment, confidence="medium",
        input_snapshot=snapshot,
    )


# ---------------------------------------------------------------------------
# H4. target_audience
# ---------------------------------------------------------------------------

async def rule_target_audience(model_id: int, url_registry_id: int) -> dict:
    """
    H4: Multi-label audience tags. Output column: target_audience  TEXT[]
    """
    inf     = await _fetch_inferred(model_id)
    fitness = inf.get("use_case_fitness") or {}
    segment = str(inf.get("price_segment") or "")
    pp      = str(inf.get("price_performance") or "")

    tags: list[str] = []
    _LOW_SEGS = {"ULTRA_BUDGET", "ENTRY"}
    _MID_LOW  = {"BUDGET", "LOWER_MIDRANGE"}
    _PRO_SEGS = {"LOWER_MIDRANGE", "UPPER_MIDRANGE", "PREMIUM_MIDRANGE", "FLAGSHIP", "ULTRA_FLAGSHIP"}

    if segment in _LOW_SEGS:
        tags.append("first_time_buyer")
    if fitness.get("gaming") in ("excellent", "good") and segment in (_MID_LOW | {"UPPER_MIDRANGE"}):
        tags.append("student_gamer")
    if fitness.get("gaming") in ("excellent", "good") and segment not in _LOW_SEGS:
        tags.append("mobile_gamer")
    if fitness.get("photography") in ("excellent", "good") and fitness.get("multimedia") in ("excellent", "good"):
        tags.append("casual_creator")
    if fitness.get("productivity") in ("excellent", "good") and segment in _PRO_SEGS:
        tags.append("working_professional")
    if fitness.get("battery") == "excellent":
        tags.append("battery_focused")
    if fitness.get("portability") in ("excellent", "good") and segment not in _LOW_SEGS:
        tags.append("minimalist_user")
    if pp == "over_spec":
        tags.append("value_seeker")
    if not tags:
        tags.append("general_user")

    tag_descs = {
        "first_time_buyer": "First-time smartphone buyer",
        "student_gamer":    "Student / casual gamer on a budget",
        "mobile_gamer":     "Mobile gamer",
        "casual_creator":   "Casual content creator",
        "working_professional": "Working professional",
        "battery_focused":  "Battery-life-first user",
        "minimalist_user":  "Minimalist / one-handed user",
        "value_seeker":     "Value seeker (flagship-killer buyer)",
        "general_user":     "General everyday user",
    }
    readable = [tag_descs.get(t, t) for t in tags]
    text     = f"Best suited for: {', '.join(readable)}."

    snapshot = {"tags": tags, "price_segment": segment, "price_performance": pp}
    return _make_result(
        rule_key="target_audience", structured_value=tags,
        inference_text=text, sentiment="Neutral", confidence="medium",
        input_snapshot=snapshot,
    )


# ---------------------------------------------------------------------------
# H5. strengths_and_compromises
# ---------------------------------------------------------------------------

# Segment rank for "expected minimum" logic
_SEG_RANK = {s: i for i, s in enumerate(_TIER_ORDER)}

# Thresholds: what counts as "expected" vs "compromise" per segment
# A weak camera in ENTRY is expected. A weak camera in UPPER_MIDRANGE is a compromise.
_COMPROMISE_THRESHOLDS = {
    # rule_key: (weak_value, min_segment_rank_to_flag)
    "jio_5g_tier":      ({"incompatible"}, 0),          # any segment — 5G compatibility matters
    "airtel_5g_tier":   ({"incompatible"}, 0),
    "india_4g_coverage":({"limited"}, 0),
    "chipset_tier":     ({"entry"}, _SEG_RANK["UPPER_MIDRANGE"]),
    "endurance_tier_absolute": ({"weak", "limited"}, _SEG_RANK["LOWER_MIDRANGE"]),
    "outdoor_visibility":      ({"limited", "poor"}, _SEG_RANK["UPPER_MIDRANGE"]),
    "display_durability":      ({"none"}, _SEG_RANK["UPPER_MIDRANGE"]),
    "ip_resistance":           ({"none"}, _SEG_RANK["FLAGSHIP"]),   # no IP in FLAGSHIP = compromise
    "charging_speed_tier":     ({"slow"}, _SEG_RANK["UPPER_MIDRANGE"]),
    "portability":             ({"heavy_or_tall"}, _SEG_RANK["PREMIUM_MIDRANGE"]),
    "ram_tier_in_segment":     ({"limited"}, _SEG_RANK["LOWER_MIDRANGE"]),
    "storage_speed_class":     ({"slow"}, _SEG_RANK["BUDGET"]),
    "panel_class":             ({"lcd", "other"}, _SEG_RANK["PREMIUM_MIDRANGE"]),
}

# Top-tier values per rule_key → strength signals
_STRENGTH_SIGNALS = {
    "jio_5g_tier":             {"excellent"},
    "airtel_5g_tier":          {"excellent"},
    "vi_5g_tier":              {"good", "excellent"},
    "bsnl_5g_tier":            {"excellent"},
    "india_4g_coverage":       {"comprehensive"},
    "chipset_tier":            {"flagship", "upper_mid"},
    "gaming_tier_absolute":    {"excellent"},
    "endurance_tier_absolute": {"excellent", "good"},
    "charging_speed_tier":     {"ultra_fast", "very_fast"},
    "outdoor_visibility":      {"exceptional", "excellent"},
    "display_durability":      {"elite", "premium"},
    "build_material_tier":     {"premium", "high_grade"},
    "ram_tier_in_segment":     {"high"},
    "storage_speed_class":     {"ultra_fast"},
    "panel_class":             {"oled"},
    "os_update_years":         {7, 6, 5},
    "multitasking_longevity":  {"excellent"},
    "portability":             {"compact"},
}


async def rule_strengths_and_compromises(model_id: int, url_registry_id: int) -> dict:
    """
    H5: Auto-populate strengths and compromises from all A–G structured outputs.
    Compromises are SEGMENT-RELATIVE — entry-tier weaknesses are never flagged.
    Output columns: strengths TEXT[], compromises TEXT[]
    """
    inf     = await _fetch_inferred(model_id)
    segment = str(inf.get("price_segment") or "BUDGET")
    seg_rank = _SEG_RANK.get(segment, 2)

    strengths:   list[str] = []
    compromises: list[str] = []

    # --- Strengths ---
    for key, top_vals in _STRENGTH_SIGNALS.items():
        val = inf.get(key)
        if val is None:
            continue
        if isinstance(val, dict):
            # e.g. ip_resistance JSONB
            val = val.get("ip_tier")
        if val in top_vals:
            label = key.replace("_", " ").replace("tier", "").replace("absolute", "").strip()
            strengths.append(label)

    # Special: rare_for_segment IP rating
    ip_prof = inf.get("ip_resistance") or {}
    if ip_prof.get("rare_for_segment"):
        strengths.append("IP68 water resistance (rare for this price segment)")

    # Special: eSIM + dual SIM
    sim_prof = inf.get("sim_profile") or {}
    if sim_prof.get("dual_sim") and sim_prof.get("esim"):
        strengths.append("Dual SIM + eSIM flexibility")

    # --- Compromises (segment-relative) ---
    for key, (weak_vals, min_rank) in _COMPROMISE_THRESHOLDS.items():
        if seg_rank < min_rank:
            continue   # weakness is expected at this price tier, skip
        val = inf.get(key)
        if val is None:
            continue
        if isinstance(val, dict):
            val = val.get("ip_tier") or val.get("included")
        if val in weak_vals:
            label = key.replace("_", " ").replace("tier", "").replace("absolute", "").strip()
            compromises.append(label)

    # Special: charger_in_box = False at LOWER_MIDRANGE+ is a compromise
    charger = inf.get("charger_in_box") or {}
    if charger.get("included") is False and seg_rank >= _SEG_RANK["LOWER_MIDRANGE"]:
        compromises.append("No charger in box")

    # Build narrative
    parts = []
    if strengths:
        parts.append(f"Key strengths: {', '.join(strengths[:5])}.")
    if compromises:
        parts.append(f"Compromises relative to {segment.replace('_', ' ').lower()} peers: {', '.join(compromises[:4])}.")
    if not strengths and not compromises:
        parts.append("No standout strengths or notable compromises at this price segment.")
    text = " ".join(parts)

    result = {"strengths": strengths, "compromises": compromises}
    sentiment = "Positive" if strengths and not compromises else ("Neutral" if strengths else "Negative")

    snapshot = {"segment": segment, "seg_rank": seg_rank,
                "strengths": strengths, "compromises": compromises}
    return _make_result(
        rule_key="strengths_and_compromises", structured_value=result,
        inference_text=text, sentiment=sentiment, confidence="medium",
        input_snapshot=snapshot,
    )


# ---------------------------------------------------------------------------
# H6. value_verdict
# ---------------------------------------------------------------------------

async def rule_value_verdict(model_id: int, url_registry_id: int) -> dict:
    """
    H6: Overall value verdict from H2, H3, H5.
    segment_leader | strong_value | competitive | compromised

    HONESTY RULE: Verdict is always segment-relative. Never oversell a segment
    win as an absolute win. Narrative must name the segment explicitly.
    Output column: value_verdict  TEXT
    """
    inf       = await _fetch_inferred(model_id)
    segment   = str(inf.get("price_segment") or "BUDGET")
    pp        = str(inf.get("price_performance") or "well_rounded")
    fitness   = inf.get("use_case_fitness") or {}
    strengths_data = inf.get("strengths") or []
    compromises_data = inf.get("compromises") or []

    seg_label = segment.replace("_", " ").lower()

    # Scoring
    fitness_good  = sum(1 for v in fitness.values() if v in ("excellent", "good"))
    fitness_total = len(fitness) or 1
    fitness_ratio = fitness_good / fitness_total

    strength_count   = len(strengths_data)
    compromise_count = len(compromises_data)

    pp_score = {"over_spec": 1.0, "well_rounded": 0.6, "brand_premium": 0.2}.get(pp, 0.5)
    combined = 0.40 * fitness_ratio + 0.35 * pp_score + 0.25 * max(0, (strength_count - compromise_count) / max(1, strength_count + compromise_count))

    if combined >= 0.72 and compromise_count == 0:
        verdict   = "segment_leader"
        sentiment = "Positive"
        text = (
            f"Segment leader in the {seg_label} tier. Outperforms most peers on primary use cases"
            f"{' with above-expected silicon for its price' if pp == 'over_spec' else ''}. "
            f"The verdict is relative: {seg_label} phones share a performance ceiling — "
            f"this is the best option in its class, not an absolute benchmark."
        )
    elif combined >= 0.52:
        verdict   = "strong_value"
        sentiment = "Positive"
        text = (
            f"Strong value in the {seg_label} segment — above-average on most dimensions. "
            f"{'Good price-performance from above-segment silicon. ' if pp == 'over_spec' else ''}"
            f"{'Minor compromises (' + ', '.join(compromises_data[:2]) + ') are acceptable at this price.' if compromises_data else 'No notable compromises.'}"
        )
    elif combined >= 0.35:
        verdict   = "competitive"
        sentiment = "Neutral"
        comps = (", ".join(compromises_data[:3])) if compromises_data else "none identified"
        text = (
            f"Competitive in the {seg_label} segment — holds its own but is not a standout. "
            f"Compromises: {comps}. "
            f"Consider alternatives if any weakness is a priority for your use case."
        )
    else:
        verdict   = "compromised"
        sentiment = "Negative"
        comps = (", ".join(compromises_data[:3])) if compromises_data else "multiple areas"
        text = (
            f"Compromised value in the {seg_label} segment — clear weaknesses relative to peers: {comps}. "
            f"Specific use-case fit may still justify the purchase, but it is not a general recommendation."
        )

    snapshot = {
        "segment": segment, "price_performance": pp,
        "fitness_ratio": round(fitness_ratio, 2),
        "strength_count": strength_count, "compromise_count": compromise_count,
        "combined_score": round(combined, 3), "value_verdict": verdict,
    }
    return _make_result(
        rule_key="value_verdict", structured_value=verdict,
        inference_text=text, sentiment=sentiment, confidence="medium",
        input_snapshot=snapshot,
    )


# ---------------------------------------------------------------------------
# Group H handler registry — ordered list enforced by orchestrator
# ---------------------------------------------------------------------------

GROUP_H_HANDLERS: dict[str, Any] = {
    "price_segment":              rule_price_segment,
    "price_performance_verdict":  rule_price_performance_verdict,
    "use_case_fitness":           rule_use_case_fitness,
    "target_audience":            rule_target_audience,
    "strengths_and_compromises":  rule_strengths_and_compromises,
    "value_verdict":              rule_value_verdict,
}

# Execution order is CRITICAL for Group H — each rule depends on previous output.
GROUP_H_EXECUTION_ORDER = [
    "price_segment",
    "price_performance_verdict",
    "use_case_fitness",
    "target_audience",
    "strengths_and_compromises",
    "value_verdict",
]
