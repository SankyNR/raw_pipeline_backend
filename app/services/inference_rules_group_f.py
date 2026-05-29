# app/services/inference_rules_group_f.py
"""
Run C — Deterministic Inference Engine
Group F: Build & Durability — 4 rules

Rules F1–F4:
  F1: water_dust_resistance  — IP rating + rare_for_segment flag
  F2: build_material_quality — body_features.build text + glass protection
  F3: portability            — weight_grams (threshold_curve inverted) + height_mm
  F4: display_durability     — glass protection name → tier

Confidence: F1=high, F2=medium, F3=high, F4=high
Defers to RunB: Build (F1, F2, F4). None (F3 — reviewers rarely state weight grams).

Column mapping after migrations:
  body_features.height    = phone height (long dimension), mm
  body_features.width     = phone width (short dimension), mm
  body_features.thickness = phone thickness, mm
  body_features.weight    = weight, grams

IP: phone_ip_ratings junction → lookup_ip_ratings.rating_name
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.core.supabase_client import get_client
from app.services.inference_curves import threshold_curve
from app.services.inference_engine import classify_price_tier  # Phase 4 helper — same pattern as G1/G2

logger = logging.getLogger(__name__)

_MS = lambda: get_client().schema("mobile_specs")

_RUNB_BUILD = "Build"

# IP rating tier map: rating_name → (tier, water_resist_mm, water_desc)
_IP_TIERS = {
    "IP68":  ("premium",   "30m",  "1.5m/30min water resistance"),
    "IP68K": ("premium",   "30m",  "IP68 + high-pressure water jets"),
    "IP69":  ("premium",   "high", "High-pressure/steam water resistance"),
    "IP69K": ("premium",   "high", "High-pressure/steam water resistance"),
    "IPX8":  ("good",      "30m",  "Water resistant — no dust rating"),
    "IP67":  ("good",      "1m",   "1m/30min water resistance"),
    "IP64":  ("moderate",  "N/A",  "Splash resistant, limited water jets"),
    "IP54":  ("moderate",  "N/A",  "Splash resistant (all directions)"),
    "IP53":  ("basic",     "N/A",  "Splash resistant from rain/spray angle"),
    "IP48":  ("basic",     "N/A",  "Water resistant — limited dust protection"),
    "IP4X":  ("basic",     "N/A",  "Dust protection — no water rating"),
}

# Glass protection tier map
_GLASS_TIERS = {
    # Top tier
    "Corning Gorilla Armor 2":        ("elite",    5),
    "Gorilla Glass Armor":            ("elite",    5),
    "Gorilla Glass Victus 2":         ("premium",  4),
    "Corning Gorilla Glass Victus 2": ("premium",  4),
    "Gorilla Glass Victus":           ("premium",  4),
    "Liquid Retina XDR":              ("premium",  4),
    "Ceramic Shield":                 ("premium",  4),
    "Ceramic Shield 2":               ("premium",  4),
    "Corning Gorilla Glass Ceramic":  ("premium",  4),
    "ProMotion Super Retina XDR OLED":("premium",  4),
    "Kunlun Glass":                   ("premium",  4),
    # Good tier
    "Gorilla Glass 7":                ("good",     3),
    "Gorilla Glass 6":                ("good",     3),
    "Dragon Crystal Glass":           ("good",     3),
    "Armor Glass":                    ("good",     3),
    # Standard tier
    "Gorilla Glass 5":                ("standard", 2),
    "Gorilla Glass 3":                ("standard", 2),
    "Panda Glass":                    ("standard", 2),
    "Tempered Glass":                 ("standard", 1),
    # None
    "No Protection":                  ("none",     0),
}


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
# F1. water_dust_resistance
# ---------------------------------------------------------------------------

async def rule_water_dust_resistance(model_id: int, url_registry_id: int) -> dict:
    """
    F1: IP rating from phone_ip_ratings junction.
    rare_for_segment: IP68 in BUDGET or lower → flag as rare for segment.
    Output column: ip_resistance (JSONB {ip_tier, rare_for_segment})
    """
    # Fetch IP rating
    ip_res = await asyncio.to_thread(lambda: (
        _MS()
        .table("phone_ip_ratings")
        .select("lookup_ip_ratings(rating_name, water_protection, dust_protection)")
        .eq("model_id", model_id)
        .execute()
    ))
    ip_rows = ip_res.data or []

    # Compute price_segment directly from launch_price.
    # CANNOT read from inferred_specs.price_segment — Group F runs before Group H1
    # writes that column. Same fix as G1/G2 (see Group G docstring).
    lp_res = await asyncio.to_thread(lambda: (
        get_client().schema("mobile_specs")
        .table("variant")
        .select("launch_price")
        .eq("model_id", model_id)
        .eq("is_base_variant", True)
        .limit(1)
        .execute()
    ))
    lp_rows = lp_res.data or []
    launch_price_f1 = float(lp_rows[0]["launch_price"]) if lp_rows and lp_rows[0].get("launch_price") else None
    price_segment = classify_price_tier(launch_price_f1) if launch_price_f1 else ""

    _BUDGET_OR_LOWER = {"BUDGET", "ENTRY", "ULTRA_BUDGET"}
    _RARE_RATINGS    = {"IP68", "IP68K", "IP69", "IP69K"}

    if not ip_rows:
        # No IP rating — no resistance claim
        profile = {"ip_tier": "none", "rare_for_segment": False}
        snapshot = {"ip_rating": None, "price_segment": price_segment}
        return _make_result(
            rule_key="water_dust_resistance", structured_value=profile,
            inference_text="No official IP water/dust resistance rating. Avoid exposing to rain or liquids.",
            sentiment="Negative", confidence="high",
            input_snapshot=snapshot, defers_to_runb=_RUNB_BUILD,
        )

    # Find the best (highest tier) IP rating
    best_rating = None
    best_tier_rank = -1
    _TIER_RANK = {"premium": 3, "good": 2, "moderate": 1, "basic": 0, "none": -1}

    for row in ip_rows:
        lk      = row.get("lookup_ip_ratings") or {}
        rname   = (lk.get("rating_name") or "").strip().upper()
        t, _, _ = _IP_TIERS.get(rname, ("none", "N/A", "No rating data"))
        if _TIER_RANK.get(t, -1) > best_tier_rank:
            best_rating    = rname
            best_tier_rank = _TIER_RANK.get(t, -1)

    if best_rating is None:
        best_rating = (ip_rows[0].get("lookup_ip_ratings") or {}).get("rating_name", "Unknown")

    ip_tier, _, water_desc = _IP_TIERS.get(best_rating, ("unknown", "N/A", "Unknown rating"))
    rare_for_segment = (
        ip_tier == "premium"
        and best_rating in _RARE_RATINGS
        and price_segment in _BUDGET_OR_LOWER
    )

    profile = {"ip_tier": ip_tier, "rare_for_segment": rare_for_segment}
    snapshot = {
        "ip_rating":         best_rating,
        "ip_tier":           ip_tier,
        "water_desc":        water_desc,
        "price_segment":     price_segment,
        "rare_for_segment":  rare_for_segment,
    }

    sentiment_map  = {"premium": "Positive", "good": "Positive", "moderate": "Neutral", "basic": "Neutral", "unknown": "Neutral"}
    sentiment      = sentiment_map.get(ip_tier, "Neutral")
    rare_note      = f" Notably, {best_rating} is rare to find in the {price_segment.lower().replace('_', '-')} segment — a standout feature." if rare_for_segment else ""

    narratives = {
        "premium":  f"Certified {best_rating} — {water_desc}. Safe for rain, splashes, and accidental submersion.{rare_note}",
        "good":     f"Certified {best_rating} — {water_desc}. Good protection against rain and splashes.{rare_note}",
        "moderate": f"Certified {best_rating} — splash resistant but not suitable for submersion.",
        "basic":    f"Certified {best_rating} — basic splash or dust protection only. Not water resistant.",
        "unknown":  f"IP rating {best_rating} — protection level unclear.",
    }
    text = narratives.get(ip_tier, f"IP rating: {best_rating}.")

    return _make_result(
        rule_key="water_dust_resistance", structured_value=profile,
        inference_text=text, sentiment=sentiment, confidence="high",
        input_snapshot=snapshot, defers_to_runb=_RUNB_BUILD,
    )


# ---------------------------------------------------------------------------
# F2. build_material_quality
# ---------------------------------------------------------------------------

async def rule_build_material_quality(model_id: int, url_registry_id: int) -> dict:
    """
    F2: body_features.build text + frame/back material parsing.
    confidence: medium (text-based detection of materials is imperfect).
    Output column: build_material_tier
    """
    body_res = await asyncio.to_thread(lambda: (
        _MS()
        .table("body_features")
        .select("build, other_features")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    ))
    body_rows = body_res.data or []
    if not body_rows:
        return _make_result(
            rule_key="build_material_quality", structured_value=None,
            inference_text="Build material data unavailable.",
            sentiment="Neutral", confidence="low",
            input_snapshot={"model_id": model_id}, defers_to_runb=_RUNB_BUILD,
        )

    build_text  = (body_rows[0].get("build") or "").lower()
    other_text  = (body_rows[0].get("other_features") or "").lower()
    combined    = build_text + " " + other_text

    # Frame material detection
    has_titanium = "titanium" in combined
    has_aluminum = "aluminum" in combined or "aluminium" in combined or "alloy" in combined
    has_stainless = "stainless" in combined

    # Back material
    has_glass_back    = "glass back" in combined or "glass rear" in combined or ("glass" in combined and "front" in combined)
    has_ceramic_back  = "ceramic back" in combined or "ceramic shell" in combined
    has_plastic_back  = "plastic" in combined or "polycarbonate" in combined

    # Tier
    if has_titanium:
        tier = "premium"; sentiment = "Positive"
        text = "Titanium frame — the most premium structural material in smartphones. Exceptional strength-to-weight ratio with a distinctive feel."
    elif has_stainless:
        tier = "premium"; sentiment = "Positive"
        text = "Stainless steel frame — premium, durable, and resistant to bending. Adds weight but provides a solid, high-end in-hand feel."
    elif has_aluminum and (has_glass_back or has_ceramic_back):
        tier = "high_grade"; sentiment = "Positive"
        back = "ceramic back" if has_ceramic_back else "glass back"
        text = f"Aluminium frame with {back} — premium construction combining structural rigidity with glass aesthetics."
    elif has_aluminum:
        tier = "mid_grade"; sentiment = "Neutral"
        text = "Aluminium frame — good rigidity and premium feel. More durable than plastic alternatives."
    elif has_plastic_back and has_aluminum:
        tier = "mid_grade"; sentiment = "Neutral"
        text = "Aluminium frame with polycarbonate back — practical and lightweight combination."
    else:
        tier = "plastic"; sentiment = "Neutral"
        text = "Polycarbonate (plastic) construction — lightweight and shatter-resistant, though less premium feeling."

    snapshot = {
        "build_text":        build_text[:200],
        "has_titanium":      has_titanium,
        "has_aluminum":      has_aluminum,
        "has_stainless":     has_stainless,
        "has_glass_back":    has_glass_back,
        "has_ceramic_back":  has_ceramic_back,
        "has_plastic_back":  has_plastic_back,
        "build_material_tier": tier,
    }
    return _make_result(
        rule_key="build_material_quality", structured_value=tier,
        inference_text=text, sentiment=sentiment, confidence="medium",
        input_snapshot=snapshot, defers_to_runb=_RUNB_BUILD,
    )


# ---------------------------------------------------------------------------
# F3. portability
# ---------------------------------------------------------------------------

async def rule_portability(model_id: int, url_registry_id: int) -> dict:
    """
    F3: weight + height → portability tier.
    weight: threshold_curve INVERTED (lower is better). Range 155g–215g.
    height: threshold_curve INVERTED (lower is better). Range 155mm–175mm.
    Output column: portability
    """
    body_res = await asyncio.to_thread(lambda: (
        _MS()
        .table("body_features")
        .select("height, width, thickness, weight")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    ))
    body_rows = body_res.data or []
    if not body_rows:
        return _make_result(
            rule_key="portability", structured_value=None,
            inference_text="Body dimensions unavailable.",
            sentiment="Neutral", confidence="low",
            input_snapshot={"model_id": model_id},
        )

    row       = body_rows[0]
    weight_g  = float(row.get("weight") or 200)
    height_mm = float(row.get("height") or 163)

    # Inverted threshold: lighter/shorter = higher score
    weight_score = 1.0 - threshold_curve(weight_g,  low=155, high=215)
    height_score = 1.0 - threshold_curve(height_mm, low=155, high=175)
    combined     = 0.60 * weight_score + 0.40 * height_score

    if combined >= 0.75:
        tier = "compact"; sentiment = "Positive"
        text = f"Compact and lightweight ({weight_g:.0f}g, {height_mm:.0f}mm tall) — easy one-handed use and pocketability."
    elif combined >= 0.50:
        tier = "comfortable"; sentiment = "Positive"
        text = f"Comfortable to handle ({weight_g:.0f}g, {height_mm:.0f}mm tall) — good balance for most hand sizes."
    elif combined >= 0.30:
        tier = "large_but_manageable"; sentiment = "Neutral"
        text = f"Larger device ({weight_g:.0f}g, {height_mm:.0f}mm tall) — some may find it unwieldy one-handed."
    else:
        tier = "heavy_or_tall"; sentiment = "Neutral"
        text = f"Large, heavy device ({weight_g:.0f}g, {height_mm:.0f}mm tall) — designed for screen real estate over portability. Pockets and one-handed use are challenging."

    snapshot = {
        "weight_grams":   weight_g,
        "height_mm":      height_mm,
        "weight_score":   round(weight_score, 3),
        "height_score":   round(height_score, 3),
        "combined_score": round(combined, 3),
        "portability":    tier,
    }
    return _make_result(
        rule_key="portability", structured_value=tier,
        inference_text=text, sentiment=sentiment, confidence="high",
        input_snapshot=snapshot,
    )


# ---------------------------------------------------------------------------
# F4. display_durability
# ---------------------------------------------------------------------------

async def rule_display_durability(model_id: int, url_registry_id: int) -> dict:
    """
    F4: Glass protection tier from lookup_glass_protection.
    Output column: display_durability
    """
    disp_res = await asyncio.to_thread(lambda: (
        _MS()
        .table("phone_displays")
        .select("display_position, lookup_glass_protection(protection_name)")
        .eq("model_id", model_id)
        .execute()
    ))
    rows = disp_res.data or []

    protection_name = None
    for r in rows:
        if (r.get("display_position") or "Primary").lower() == "primary":
            lk = r.get("lookup_glass_protection") or {}
            protection_name = (lk.get("protection_name") or "").strip()
            break
    if not protection_name and rows:
        lk = rows[0].get("lookup_glass_protection") or {}
        protection_name = (lk.get("protection_name") or "").strip()

    snapshot = {"glass_protection": protection_name}

    if not protection_name or protection_name == "No Protection":
        tier = "none"; sentiment = "Negative"
        text = "No display glass protection specified. Unprotected glass is significantly more susceptible to scratches and shattering from drops."
    else:
        tier, rank = _GLASS_TIERS.get(protection_name, ("standard", 2))
        if tier == "elite":
            sentiment = "Positive"
            text = f"{protection_name} — elite scratch and drop resistance. One of the toughest display glass solutions available."
        elif tier == "premium":
            sentiment = "Positive"
            text = f"{protection_name} — premium drop and scratch protection. Significantly more durable than standard tempered glass."
        elif tier == "good":
            sentiment = "Positive"
            text = f"{protection_name} — good drop and scratch resistance. Above-average durability for everyday use."
        elif tier == "standard":
            sentiment = "Neutral"
            text = f"{protection_name} — standard scratch resistance. Some protection versus unprotected glass."
        else:
            sentiment = "Neutral"
            text = f"{protection_name} — basic glass protection."

    snapshot["display_durability"] = tier
    return _make_result(
        rule_key="display_durability", structured_value=tier,
        inference_text=text, sentiment=sentiment, confidence="high",
        input_snapshot=snapshot, defers_to_runb=_RUNB_BUILD,
    )


# ---------------------------------------------------------------------------
# Group F handler registry
# ---------------------------------------------------------------------------

GROUP_F_HANDLERS: dict[str, Any] = {
    "water_dust_resistance": rule_water_dust_resistance,
    "build_material_quality": rule_build_material_quality,
    "portability":           rule_portability,
    "display_durability":    rule_display_durability,
}
