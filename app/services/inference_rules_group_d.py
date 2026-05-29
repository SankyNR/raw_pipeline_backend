# app/services/inference_rules_group_d.py
"""
Run C — Deterministic Inference Engine
Group D: Battery & Charging — 3 rules

Rules D1–D3:
  D1: battery_endurance       — mAh + refresh penalty + adaptive exemption
  D2: charging_speed          — charging_power via saturating_curve
  D3: charger_in_box          — charger_in_box bool + wattage
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.core.supabase_client import get_client
from app.services.inference_curves import saturating_curve, threshold_curve
from app.services.inference_engine import classify_price_tier, percentile_to_tier

logger = logging.getLogger(__name__)

_MS = lambda: get_client().schema("mobile_specs")

_RUNB_BATTERY = "Battery"


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
# D1. battery_endurance
# ---------------------------------------------------------------------------

async def rule_battery_endurance(model_id: int, url_registry_id: int) -> dict:
    """
    D1: mAh endurance tier.
    High-refresh (>=90Hz) static penalty: downgrade one tier unless LTPO/adaptive.
    Output columns: endurance_tier_absolute, endurance_tier_in_segment
    """
    # Battery capacity
    chg_res = await asyncio.to_thread(lambda: (
        _MS()
        .table("charging_specs")
        .select("battery_capacity")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    ))
    chg_rows = chg_res.data or []
    if not chg_rows or chg_rows[0].get("battery_capacity") is None:
        return _make_result(
            rule_key="battery_endurance", structured_value=None,
            inference_text="Battery capacity data unavailable.",
            sentiment="Neutral", confidence="low",
            input_snapshot={"model_id": model_id}, defers_to_runb=_RUNB_BATTERY,
        )
    mah = float(chg_rows[0]["battery_capacity"])

    # Primary display refresh rate + display_id
    disp_res = await asyncio.to_thread(lambda: (
        _MS()
        .table("phone_displays")
        .select("display_id, refresh_rate, display_position")
        .eq("model_id", model_id)
        .execute()
    ))
    disp_rows = disp_res.data or []
    primary_refresh = 60.0
    primary_disp_id = None
    for d in disp_rows:
        if (d.get("display_position") or "Primary").lower() == "primary":
            primary_refresh = float(d.get("refresh_rate") or 60)
            primary_disp_id = d.get("display_id")
            break
    if disp_rows and primary_disp_id is None:
        primary_refresh = float(disp_rows[0].get("refresh_rate") or 60)
        primary_disp_id = disp_rows[0].get("display_id")

    high_refresh = primary_refresh >= 90

    # LTPO/adaptive check
    is_adaptive = False
    if high_refresh and primary_disp_id:
        feat_res = await asyncio.to_thread(lambda: (
            _MS()
            .table("phone_display_features")
            .select("lookup_display_features(feature_name)")
            .eq("display_id", primary_disp_id)
            .execute()
        ))
        for fr in (feat_res.data or []):
            lk = fr.get("lookup_display_features") or {}
            fn = (lk.get("feature_name") or "").lower()
            if "ltpo" in fn or "adaptive refresh" in fn:
                is_adaptive = True
                break

    # Base tier
    if mah >= 5500:
        base = "excellent"
    elif mah >= 5000:
        base = "good"
    elif mah >= 4500:
        base = "adequate"
    elif mah >= 4000:
        base = "limited"
    else:
        base = "weak"

    _ORDER = ["weak", "limited", "adequate", "good", "excellent"]

    # Downgrade for high static refresh
    final = base
    if high_refresh and not is_adaptive:
        idx   = _ORDER.index(base)
        final = _ORDER[max(0, idx - 1)]

    # Sentiment
    sentiment = "Positive" if final in ("excellent", "good") else ("Neutral" if final == "adequate" else "Negative")

    narratives = {
        "excellent": f"{int(mah)} mAh — exceptional battery. Expect 1.5–2 day battery life for most users.",
        "good":      f"{int(mah)} mAh — large battery. Comfortably lasts a full day for heavy users.",
        "adequate":  f"{int(mah)} mAh — adequate battery for typical daily use.",
        "limited":   f"{int(mah)} mAh — below average. Heavy users may need a mid-day charge.",
        "weak":      f"{int(mah)} mAh — small battery. Expect under-a-day life for moderate use.",
    }
    text = narratives[final]
    if high_refresh and not is_adaptive:
        text += f" Note: {int(primary_refresh)}Hz refresh rate slightly reduces estimated endurance."
    elif high_refresh and is_adaptive:
        text += f" LTPO adaptive refresh ({int(primary_refresh)}Hz max) intelligently drops to 1Hz when idle, minimising battery drain."

    snapshot = {
        "battery_mah":       int(mah),
        "refresh_hz":        int(primary_refresh),
        "high_refresh":      high_refresh,
        "is_adaptive":       is_adaptive,
        "base_tier":         base,
        "final_tier":        final,
    }

    # ── Segment-relative tier ────────────────────────────────────────────────
    # Compare this phone's endurance_tier against peers already committed in
    # the same price segment. Falls back to absolute tier when fewer than 4
    # peers exist (early catalogue, insufficient data for percentile).
    _TIER_RANK_D1 = {"weak": 0, "limited": 1, "adequate": 2, "good": 3, "excellent": 4}
    tier_in_seg = final
    try:
        lp_d1_res = await asyncio.to_thread(lambda: (
            _MS()
            .table("variant")
            .select("launch_price")
            .eq("model_id", model_id)
            .eq("is_base_variant", True)
            .limit(1)
            .execute()
        ))
        lp_d1_rows = lp_d1_res.data or []
        launch_price_d1 = float(lp_d1_rows[0]["launch_price"]) if lp_d1_rows and lp_d1_rows[0].get("launch_price") else None
        price_seg_d1 = classify_price_tier(launch_price_d1) if launch_price_d1 else None

        if price_seg_d1:
            peers_d1 = await asyncio.to_thread(lambda: (
                get_client().schema("pipeline")
                .table("inferred_specs")
                .select("endurance_tier_absolute")
                .eq("price_segment", price_seg_d1)
                .neq("model_id", model_id)
                .not_.is_("endurance_tier_absolute", "null")
                .execute()
            ))
            peer_ranks_d1 = [
                _TIER_RANK_D1.get(r.get("endurance_tier_absolute"), 2)
                for r in (peers_d1.data or [])
            ]
            if len(peer_ranks_d1) >= 4:
                this_rank_d1 = _TIER_RANK_D1.get(final, 2)
                beaten_d1 = sum(1 for pr in peer_ranks_d1 if pr <= this_rank_d1)
                pct_d1 = beaten_d1 / len(peer_ranks_d1)
                tier_in_seg = percentile_to_tier(pct_d1)
                snapshot["seg_peers"] = len(peer_ranks_d1)
                snapshot["seg_percentile"] = round(pct_d1, 3)
    except Exception as _exc_seg:
        logger.debug("D1: segment percentile computation failed: %s", _exc_seg)

    snapshot["endurance_tier_in_segment"] = tier_in_seg

    # ── Conflict check (spec §2.6) ────────────────────────────────────────────
    # If Run C says the battery is good/excellent but the majority of Run B
    # Battery-category entries are negative, flag for HITL review.
    # Run B observed reality (real reviewers) beats spec-sheet inference.
    conflict_flag = False
    if final in ("excellent", "good"):
        try:
            runb_bat = await asyncio.to_thread(lambda: (
                get_client().schema("pipeline")
                .table("phone_experiences")
                .select("sentiment, lookup_experience_categories(category_name)")
                .eq("url_registry_id", url_registry_id)
                .eq("is_suppressed", False)
                .eq("is_superseded", False)
                .execute()
            ))
            battery_entries = [
                r for r in (runb_bat.data or [])
                if (r.get("lookup_experience_categories") or {}).get("category_name", "").lower() == "battery"
            ]
            if battery_entries:
                neg_count = sum(
                    1 for e in battery_entries
                    if (e.get("sentiment") or "").lower() == "negative"
                )
                if neg_count > len(battery_entries) / 2:
                    conflict_flag = True
                    snapshot["conflict_reason"] = (
                        f"Run C tier='{final}' but {neg_count}/{len(battery_entries)} "
                        f"Run B Battery entries are Negative"
                    )
                    logger.warning(
                        "D1 battery_endurance CONFLICT — Run C tier=%r but %d/%d "
                        "Run B Battery entries are Negative for url_registry_id=%d model_id=%d",
                        final, neg_count, len(battery_entries), url_registry_id, model_id,
                    )
        except Exception as _exc_cf:
            logger.debug("D1: conflict check failed: %s", _exc_cf)

    return _make_result(
        rule_key="battery_endurance",
        structured_value={
            "endurance_tier_absolute":   final,
            "endurance_tier_in_segment": tier_in_seg,
        },
        inference_text=text, sentiment=sentiment, confidence="high",
        input_snapshot=snapshot, defers_to_runb=_RUNB_BATTERY,
        conflict_flag=conflict_flag,
    )


# ---------------------------------------------------------------------------
# D2. charging_speed
# ---------------------------------------------------------------------------

async def rule_charging_speed(model_id: int, url_registry_id: int) -> dict:
    """
    D2: charging_power via saturating_curve (knee=67W, ceiling=120W).
    Also checks wireless_charging and wireless_charging_power.
    Output column: charging_speed_tier
    """
    chg_res = await asyncio.to_thread(lambda: (
        _MS()
        .table("charging_specs")
        .select("charging_power, wireless_charging, wireless_charging_power")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    ))
    rows = chg_res.data or []
    if not rows:
        return _make_result(
            rule_key="charging_speed", structured_value=None,
            inference_text="Charging speed data unavailable.",
            sentiment="Neutral", confidence="low",
            input_snapshot={"model_id": model_id}, defers_to_runb=_RUNB_BATTERY,
        )

    row             = rows[0]
    watt            = float(row.get("charging_power") or 0)
    wireless        = bool(row.get("wireless_charging") or False)
    wireless_w      = float(row.get("wireless_charging_power") or 0)

    score = saturating_curve(watt, knee=67, ceiling=120)

    if watt >= 100:
        tier = "ultra_fast"; sentiment = "Positive"
        text = f"{int(watt)}W wired charging — ultra-fast. A full charge in approximately 20–30 minutes."
    elif watt >= 67:
        tier = "very_fast"; sentiment = "Positive"
        text = f"{int(watt)}W wired charging — very fast. 0–50% in roughly 20 minutes."
    elif watt >= 33:
        tier = "fast"; sentiment = "Positive"
        text = f"{int(watt)}W wired charging — fast charging. Full charge in approximately 60–75 minutes."
    elif watt >= 18:
        tier = "moderate"; sentiment = "Neutral"
        text = f"{int(watt)}W wired charging — moderate speed. Expect 90–120 minutes for a full charge."
    elif watt > 0:
        tier = "slow"; sentiment = "Negative"
        text = f"{int(watt)}W wired charging — slow. Full charge may take 2+ hours."
    else:
        tier = "unknown"; sentiment = "Neutral"
        text = "Charging speed not specified."

    if wireless and wireless_w >= 15:
        text += f" Wireless charging supported ({int(wireless_w)}W)."
    elif wireless:
        text += " Wireless charging supported."

    snapshot = {
        "wired_charging_watt":    watt,
        "wireless_charging":      wireless,
        "wireless_power_watt":    wireless_w,
        "score":                  round(score, 4),
        "charging_speed_tier":    tier,
    }
    return _make_result(
        rule_key="charging_speed", structured_value=tier,
        inference_text=text, sentiment=sentiment, confidence="high",
        input_snapshot=snapshot, defers_to_runb=_RUNB_BATTERY,
    )


# ---------------------------------------------------------------------------
# D3. charger_in_box
# ---------------------------------------------------------------------------

async def rule_charger_in_box(model_id: int, url_registry_id: int) -> dict:
    """
    D3: charger_in_box bool + wattage.
    Output column: charger_in_box (JSONB {included: bool, wattage: int|null})
    """
    chg_res = await asyncio.to_thread(lambda: (
        _MS()
        .table("charging_specs")
        .select("charger_in_box, charging_power")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    ))
    rows = chg_res.data or []
    if not rows:
        return _make_result(
            rule_key="charger_in_box", structured_value=None,
            inference_text="Charger data unavailable.",
            sentiment="Neutral", confidence="low",
            input_snapshot={"model_id": model_id},
        )

    row       = rows[0]
    included  = row.get("charger_in_box")  # may be True/False/None
    watt_raw  = row.get("charging_power")
    watt      = int(watt_raw) if watt_raw is not None else None

    structured = {"included": included, "wattage": watt}
    snapshot   = {"charger_in_box": included, "charging_power_watt": watt}

    if included is False:
        sentiment = "Negative"
        text = (
            "No charger included in the box — you'll need to source a compatible charger separately. "
            f"{'Supports ' + str(watt) + 'W charging; a compatible charger is recommended.' if watt else ''}"
        )
    elif watt and watt >= 65:
        sentiment = "Positive"
        text = f"{watt}W fast charger included in box. Ready to use out of the box at full speed."
    elif watt and watt >= 18:
        sentiment = "Positive"
        text = f"{watt}W charger included in box. Adequate for everyday charging."
    elif included is True:
        sentiment = "Neutral"
        text = "Charger included in box — wattage unspecified."
    else:
        # included is None — data missing
        sentiment = "Neutral"
        text = "Charger inclusion not confirmed in specs."

    return _make_result(
        rule_key="charger_in_box", structured_value=structured,
        inference_text=text, sentiment=sentiment, confidence="high",
        input_snapshot=snapshot,
    )


# ---------------------------------------------------------------------------
# Group D handler registry
# ---------------------------------------------------------------------------

GROUP_D_HANDLERS: dict[str, Any] = {
    "battery_endurance": rule_battery_endurance,
    "charging_speed":    rule_charging_speed,
    "charger_in_box":    rule_charger_in_box,
}
