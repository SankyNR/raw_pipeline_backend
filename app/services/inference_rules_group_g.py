# app/services/inference_rules_group_g.py
"""
Run C — Deterministic Inference Engine
Group G: Software & Ecosystem — 5 rules

Rules G1–G5:
  G1: os_update_longevity        — brand_update_policy lookup
  G2: security_update_longevity  — brand_update_policy lookup
  G3: software_cleanliness       — brand prior (low confidence) + os_and_security data
  G4: ecosystem_depth            — cross-device features from cellular/connectivity features
  G5: ecosystem_profile          — ecosystem breadth and lock-in risk

Confidence: G1=high(if found)/low(miss), G2=high/low, G3=low, G4=medium, G5=medium
Defers to RunB: Software (G1, G2, G3, G4, G5)

IMPORTANT: G1/G2 cannot rely on inferred_specs.price_segment (H1 runs after Group G).
Instead they call classify_price_tier() on variant.launch_price directly.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.core.supabase_client import get_client
from app.services.inference_engine import classify_price_tier   # Phase 4 helper

logger = logging.getLogger(__name__)

_MS = lambda: get_client().schema("mobile_specs")
_PL = lambda: get_client().schema("pipeline")
_PB = lambda: get_client().schema("public")

_RUNB_SOFTWARE = "Software"


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


async def _fetch_brand_and_price(model_id: int) -> tuple[str, float | None]:
    """Returns (brand_name, launch_price_inr) for the phone."""
    res = await asyncio.to_thread(lambda: (
        _MS()
        .table("phones")
        .select("brand_id, public.brands(brand_name)")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    ))
    rows = res.data or []
    brand_name = ""
    if rows:
        bj = rows[0].get("brands") or {}  # Supabase returns cross-schema join without schema prefix
        brand_name = (bj.get("brand_name") or "").strip()

    # Price from base variant
    pr_res = await asyncio.to_thread(lambda: (
        _MS()
        .table("variant")
        .select("launch_price")
        .eq("model_id", model_id)
        .eq("is_base_variant", True)
        .limit(1)
        .execute()
    ))
    pr_rows = pr_res.data or []
    price = float(pr_rows[0]["launch_price"]) if pr_rows and pr_rows[0].get("launch_price") else None
    return brand_name, price


async def _fetch_brand_update_policy(brand: str, price_segment: str) -> dict | None:
    """
    Looks up pipeline.brand_update_policy with fallback to 'ALL'.
    Returns the row dict or None if no match.
    """
    # Try segment-specific first
    specific_res = await asyncio.to_thread(lambda: (
        _PL()
        .table("brand_update_policy")
        .select("os_update_years, security_update_years, notes")
        .eq("brand", brand)
        .eq("price_segment", price_segment)
        .limit(1)
        .execute()
    ))
    rows = specific_res.data or []
    if rows:
        return rows[0]

    # Fallback to ALL
    all_res = await asyncio.to_thread(lambda: (
        _PL()
        .table("brand_update_policy")
        .select("os_update_years, security_update_years, notes")
        .eq("brand", brand)
        .eq("price_segment", "ALL")
        .limit(1)
        .execute()
    ))
    rows = all_res.data or []
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# G1. os_update_longevity
# ---------------------------------------------------------------------------

async def rule_os_update_longevity(model_id: int, url_registry_id: int) -> dict:
    """
    G1: OS major version update years from brand_update_policy.
    Also reads mobile_specs.os_and_security.os_update_years as secondary signal.
    Output column: os_update_years
    """
    brand, price = await _fetch_brand_and_price(model_id)
    price_segment = classify_price_tier(price) if price else "UNKNOWN"

    # Secondary signal: os_and_security actual commit value
    oas_res = await asyncio.to_thread(lambda: (
        _MS()
        .table("os_and_security")
        .select("os_update_years, security_update_years")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    ))
    oas_rows = oas_res.data or []
    extracted_os_years = int(oas_rows[0].get("os_update_years") or 0) if oas_rows else 0

    policy_row = await _fetch_brand_update_policy(brand, price_segment) if brand else None

    snapshot = {
        "brand":              brand,
        "price_segment":      price_segment,
        "extracted_os_years": extracted_os_years,
        "policy_found":       policy_row is not None,
    }

    # Determine final years: use extracted value if specific and positive; else policy
    if extracted_os_years > 0:
        os_years = extracted_os_years
        confidence = "high"
        source = "extracted_spec"
    elif policy_row:
        os_years = int(policy_row.get("os_update_years") or 0)
        confidence = "high"
        source = "brand_policy"
    else:
        os_years = 0
        confidence = "low"
        source = "unknown"

    snapshot.update({"os_update_years": os_years, "source": source})

    if os_years == 0:
        sentiment = "Neutral"
        text = f"{brand} OS update commitment is not confirmed for this device."
    elif os_years >= 5:
        sentiment = "Positive"
        text = f"Committed to {os_years} major Android OS updates — long-term software support that ensures you receive the latest features and security improvements for years."
    elif os_years >= 3:
        sentiment = "Positive"
        text = f"{os_years} major OS updates committed — good software longevity. The phone will stay current for at least {os_years} Android versions."
    elif os_years >= 2:
        sentiment = "Neutral"
        text = f"{os_years} major OS updates — basic software support. Typical for budget/mid-range devices."
    else:
        sentiment = "Negative"
        text = f"Only {os_years} major OS update expected — limited software longevity. Consider whether the launch OS version is sufficient for your needs."

    return _make_result(
        rule_key="os_update_longevity", structured_value=os_years,
        inference_text=text, sentiment=sentiment, confidence=confidence,
        input_snapshot=snapshot, defers_to_runb=_RUNB_SOFTWARE,
    )


# ---------------------------------------------------------------------------
# G2. security_update_longevity
# ---------------------------------------------------------------------------

async def rule_security_update_longevity(model_id: int, url_registry_id: int) -> dict:
    """
    G2: Security patch years from brand_update_policy.
    Also reads mobile_specs.os_and_security.security_update_years.
    Output column: security_update_years
    """
    brand, price = await _fetch_brand_and_price(model_id)
    price_segment = classify_price_tier(price) if price else "UNKNOWN"

    oas_res = await asyncio.to_thread(lambda: (
        _MS()
        .table("os_and_security")
        .select("security_update_years")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    ))
    oas_rows = oas_res.data or []
    extracted_sec = int(oas_rows[0].get("security_update_years") or 0) if oas_rows else 0

    policy_row = await _fetch_brand_update_policy(brand, price_segment) if brand else None

    snapshot = {
        "brand":               brand,
        "price_segment":       price_segment,
        "extracted_sec_years": extracted_sec,
        "policy_found":        policy_row is not None,
    }

    if extracted_sec > 0:
        sec_years  = extracted_sec
        confidence = "high"
        source     = "extracted_spec"
    elif policy_row:
        sec_years  = int(policy_row.get("security_update_years") or 0)
        confidence = "high"
        source     = "brand_policy"
    else:
        sec_years  = 0
        confidence = "low"
        source     = "unknown"

    snapshot.update({"security_update_years": sec_years, "source": source})

    if sec_years == 0:
        sentiment = "Neutral"
        text = "Security patch commitment is not confirmed for this device."
    elif sec_years >= 5:
        sentiment = "Positive"
        text = (
            f"{sec_years} years of security patch updates — excellent protection commitment. "
            f"Monthly/quarterly security patches protect against newly discovered vulnerabilities throughout the device's useful life."
        )
    elif sec_years >= 3:
        sentiment = "Positive"
        text = f"{sec_years} years of security patches — solid protection for the expected ownership period."
    else:
        sentiment = "Neutral"
        text = f"{sec_years} years of security patches — basic coverage. Review whether this aligns with your expected ownership duration."

    return _make_result(
        rule_key="security_update_longevity", structured_value=sec_years,
        inference_text=text, sentiment=sentiment, confidence=confidence,
        input_snapshot=snapshot, defers_to_runb=_RUNB_SOFTWARE,
    )


# ---------------------------------------------------------------------------
# G3. software_cleanliness
# ---------------------------------------------------------------------------

# Brand prior: cleaner UIs get higher scores
# These are subjective priors based on reviewer consensus, not extracted data.
_CLEANLINESS_PRIOR = {
    "Google":    ("clean",     "Positive",  "Stock Android experience — no bloat, fastest updates, pure Google assistant integration."),
    "Nothing":   ("clean",     "Positive",  "Nothing OS — near-stock Android with minimal bloat. Unique dot-matrix aesthetic UI."),
    "CMF":       ("clean",     "Positive",  "CMF/Nothing OS 3 — near-stock. Lightweight and focused."),
    "Motorola":  ("near_clean","Positive",  "Near-stock Android (My UX) — minimal bloat with a few Moto Experiences additions."),
    "ASUS":      ("moderate",  "Neutral",   "ZenUI — feature-rich but more bloated than stock. Gaming-focused features (ROG) add value."),
    "Nokia":     ("clean",     "Positive",  "Nokia/HMD phones run near-stock Android. Android One programme phones have guaranteed updates."),
    "OnePlus":   ("moderate",  "Neutral",   "OxygenOS (now merged with ColorOS) — feature-rich but heavier than stock. Some bloatware."),
    "Samsung":   ("moderate",  "Neutral",   "One UI — feature-rich and polished but includes Samsung/carrier bloatware. Heavier than stock."),
    "Apple":     ("clean",     "Positive",  "iOS — tightly integrated, smooth, and controlled ecosystem with no third-party bloat."),
    "Xiaomi":    ("heavy",     "Neutral",   "HyperOS (formerly MIUI) — feature-heavy with ads in system apps on some variants. Highly customisable."),
    "POCO":      ("heavy",     "Neutral",   "HyperOS — same base as Xiaomi. Ads in system apps possible. Feature-rich but heavier."),
    "Redmi":     ("heavy",     "Neutral",   "HyperOS — heavy UI with system app ads possible on budget variants."),
    "Vivo":      ("heavy",     "Neutral",   "FuntouchOS — feature-heavy with bloatware. Many AI features but intrusive system ads."),
    "iQOO":      ("moderate",  "Neutral",   "OriginOS (FuntouchOS derivative) — gaming-focused. Less bloated than standard Vivo."),
    "realme":    ("heavy",     "Neutral",   "realme UI — bloatware-heavy. Improves with premium models but budget variants have system ads."),
    "OPPO":      ("heavy",     "Neutral",   "ColorOS — feature-rich but heavy. System ads and bloatware present on many models."),
}

_DEFAULT_PRIOR = ("moderate", "Neutral", "Software cleanliness data not available for this brand.")


async def rule_software_cleanliness(model_id: int, url_registry_id: int) -> dict:
    """
    G3: Brand-prior based UI cleanliness. confidence='low' always (brand prior only).
    Also reads ui_skin for additional context.
    Output column: software_cleanliness
    """
    brand, _ = await _fetch_brand_and_price(model_id)

    # UI skin
    oas_res = await asyncio.to_thread(lambda: (
        _MS()
        .table("os_and_security")
        .select("lookup_ui_skins(ui_skin_name)")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    ))
    oas_rows = oas_res.data or []
    ui_skin = ""
    if oas_rows:
        sk_join = oas_rows[0].get("lookup_ui_skins") or {}
        ui_skin = (sk_join.get("ui_skin_name") or "").strip()

    tier, sentiment, base_text = _CLEANLINESS_PRIOR.get(brand, _DEFAULT_PRIOR)

    if ui_skin:
        text = f"{ui_skin}: {base_text}"
    else:
        text = base_text

    snapshot = {"brand": brand, "ui_skin": ui_skin, "cleanliness_tier": tier}
    return _make_result(
        rule_key="software_cleanliness", structured_value=tier,
        inference_text=text, sentiment=sentiment, confidence="low",
        input_snapshot=snapshot, defers_to_runb=_RUNB_SOFTWARE,
    )


# ---------------------------------------------------------------------------
# G4. ecosystem_depth
# ---------------------------------------------------------------------------

async def rule_ecosystem_depth(model_id: int, url_registry_id: int) -> dict:
    """
    G4: Cross-device continuity features from cellular features + connectivity.
    Signals: 5G SA/NSA, carrier aggregation, Bluetooth version (for audio handoff),
    UWB (for spatial awareness). confidence: medium.
    Output column: ecosystem_depth
    """
    # Cellular features
    cel_res = await asyncio.to_thread(lambda: (
        _MS()
        .table("phone_cellular_features")
        .select("lookup_cellular_features(feature_name)")
        .eq("model_id", model_id)
        .execute()
    ))
    cel_feats = set()
    for r in (cel_res.data or []):
        lk = r.get("lookup_cellular_features") or {}
        fn = (lk.get("feature_name") or "").lower()
        if fn:
            cel_feats.add(fn)

    # Connectivity (NFC, UWB, Bluetooth version)
    conn_res = await asyncio.to_thread(lambda: (
        _MS()
        .table("connectivity")
        .select("nfc, uwb, lookup_bluetooth_versions(bluetooth_version)")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    ))
    conn_rows = conn_res.data or []
    nfc  = False; uwb = False; bt_version = "5.0"
    if conn_rows:
        r       = conn_rows[0]
        nfc     = bool(r.get("nfc") or False)
        uwb     = bool(r.get("uwb") or False)
        bv      = r.get("lookup_bluetooth_versions") or {}
        bt_version = (bv.get("bluetooth_version") or "5.0").strip()

    has_5g_sa = "5g sa" in cel_feats
    has_ca    = "carrier aggregation" in cel_feats
    has_en_dc = any("en-dc" in f or "dual connectivity" in f for f in cel_feats)

    # Ecosystem score: UWB + NFC + BT5.3+ + 5G SA are ecosystem enablers
    try:
        bt_val = float(bt_version)
    except ValueError:
        bt_val = 5.0

    score = (
        (0.25 if uwb else 0)
        + (0.20 if nfc else 0)
        + (0.20 if bt_val >= 5.3 else (0.10 if bt_val >= 5.0 else 0))
        + (0.20 if has_5g_sa else 0)
        + (0.15 if has_ca else 0)
    )

    if score >= 0.65:
        tier = "deep"; sentiment = "Positive"
    elif score >= 0.35:
        tier = "standard"; sentiment = "Neutral"
    else:
        tier = "basic"; sentiment = "Neutral"

    feats_present = []
    if uwb:      feats_present.append("UWB (Ultra-Wideband)")
    if nfc:      feats_present.append("NFC")
    if bt_val >= 5.3: feats_present.append(f"Bluetooth {bt_version}")
    if has_5g_sa: feats_present.append("5G SA")
    if has_ca:   feats_present.append("Carrier Aggregation")

    if tier == "deep":
        text = f"Rich ecosystem connectivity: {', '.join(feats_present)}. Supports advanced cross-device and spatial experiences (UWB for Find My, NFC payments, BT audio handoff)."
    elif tier == "standard":
        text = f"Standard ecosystem features: {', '.join(feats_present) or 'basic connectivity'}. Covers core use-cases."
    else:
        text = "Basic connectivity — limited ecosystem depth. Missing NFC, UWB, or modern Bluetooth for advanced cross-device features."

    snapshot = {
        "nfc": nfc, "uwb": uwb, "bt_version": bt_version,
        "has_5g_sa": has_5g_sa, "has_ca": has_ca,
        "cellular_features": sorted(cel_feats),
        "score": round(score, 3), "ecosystem_depth": tier,
    }
    return _make_result(
        rule_key="ecosystem_depth", structured_value=tier,
        inference_text=text, sentiment=sentiment, confidence="medium",
        input_snapshot=snapshot, defers_to_runb=_RUNB_SOFTWARE,
    )


# ---------------------------------------------------------------------------
# G5. ecosystem_profile
# ---------------------------------------------------------------------------

async def rule_ecosystem_profile(model_id: int, url_registry_id: int) -> dict:
    """
    G5: Breadth tier (open vs proprietary) + lock-in risk.
    Signals: USB standard (USB-C vs Lightning), NFC, BT open codecs.
    Output column: ecosystem_profile (JSONB {breadth_tier, lock_in_risk})
    """
    brand, _ = await _fetch_brand_and_price(model_id)

    conn_res = await asyncio.to_thread(lambda: (
        _MS()
        .table("connectivity")
        .select("nfc, lookup_usb_standards(usb_standard)")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    ))
    conn_rows = conn_res.data or []
    nfc = False
    usb_standard = ""
    if conn_rows:
        r            = conn_rows[0]
        nfc          = bool(r.get("nfc") or False)
        usb_join     = r.get("lookup_usb_standards") or {}
        usb_standard = (usb_join.get("usb_standard") or "").lower()

    has_usbc      = "usb-c" in usb_standard or "type-c" in usb_standard
    has_lightning = "lightning" in usb_standard

    # Lock-in risk: Apple = high (ecosystem lock), Chinese brands heavy ecosystem = medium
    _HIGH_LOCKIN   = {"Apple"}
    _MEDIUM_LOCKIN = {"Samsung", "Huawei", "Xiaomi", "vivo", "OPPO"}

    if brand in _HIGH_LOCKIN:
        lock_in = "high"
    elif brand in _MEDIUM_LOCKIN:
        lock_in = "medium"
    else:
        lock_in = "low"

    # Breadth tier
    if has_usbc and nfc:
        breadth = "open"
    elif has_usbc:
        breadth = "mostly_open"
    elif has_lightning:
        breadth = "proprietary"
    else:
        breadth = "standard"

    profile = {"breadth_tier": breadth, "lock_in_risk": lock_in}

    lock_texts = {
        "high":   "High ecosystem lock-in — moving to another platform requires replacing accessories and re-purchasing apps.",
        "medium": "Moderate ecosystem lock-in — brand services (cloud, wearables) are tightly integrated but USB-C maintains hardware openness.",
        "low":    "Low lock-in — standard USB-C, open app ecosystem (Android). Switch-friendly.",
    }
    breadth_texts = {
        "open":        "USB-C + NFC — universally compatible charging, data, and contactless payments.",
        "mostly_open": "USB-C charging and data — standard and widely compatible.",
        "proprietary": "Lightning connector — Apple proprietary. Requires Apple-specific cables and accessories.",
        "standard":    "Standard connectivity.",
    }

    text = f"{breadth_texts.get(breadth, '')} {lock_texts.get(lock_in, '')}"
    sentiment = "Positive" if lock_in == "low" and breadth in ("open", "mostly_open") else "Neutral"

    snapshot = {
        "brand": brand, "usb_standard": usb_standard, "nfc": nfc,
        "has_usbc": has_usbc, "has_lightning": has_lightning,
        "breadth_tier": breadth, "lock_in_risk": lock_in,
    }
    return _make_result(
        rule_key="ecosystem_profile", structured_value=profile,
        inference_text=text, sentiment=sentiment, confidence="medium",
        input_snapshot=snapshot, defers_to_runb=_RUNB_SOFTWARE,
    )


# ---------------------------------------------------------------------------
# Group G handler registry
# ---------------------------------------------------------------------------

GROUP_G_HANDLERS: dict[str, Any] = {
    "os_update_longevity":       rule_os_update_longevity,
    "security_update_longevity": rule_security_update_longevity,
    "software_cleanliness":      rule_software_cleanliness,
    "ecosystem_depth":           rule_ecosystem_depth,
    "ecosystem_profile":         rule_ecosystem_profile,
}
