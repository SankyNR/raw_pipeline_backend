# app/services/inference_rules_group_c.py
"""
Run C — Deterministic Inference Engine
Group C: Display & Multimedia — 5 rules

Rules C1–C5:
  C1: display_panel_quality   — panel_class + hdr_effective (OLED/LCD/HDR)
  C2: display_sharpness       — PPI via double_sigmoid
  C3: display_smoothness      — refresh_rate via saturating_curve + touch_sampling
  C4: outdoor_visibility      — brightness_hbm ONLY (never brightness_peak)
  C5: multimedia_experience   — speakers + audio features + display size + panel

Confidence: C1-C4=high, C5=medium
Defers to RunB: Display (C1–C4), None (C5)

CRITICAL: C4 uses brightness_hbm, NEVER brightness_peak.
Touch sampling is in lookup_display_features ("240Hz Touch Sampling" etc.)
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from app.core.supabase_client import get_client
from app.services.inference_curves import double_sigmoid, saturating_curve, threshold_curve

logger = logging.getLogger(__name__)

_MS = lambda: get_client().schema("mobile_specs")

_RUNB_DISPLAY = "Display"


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


async def _fetch_primary_display(model_id: int) -> dict:
    """Fetches the primary display row including panel_type join and protection join."""
    res = await asyncio.to_thread(lambda: (
        _MS()
        .table("phone_displays")
        .select(
            "display_id, display_position, panel_type_id, ppi, refresh_rate, "
            "brightness_hbm, brightness_peak, size_inch, protection_id, "
            "lookup_panel_types(panel_type), lookup_glass_protection(protection_name)"
        )
        .eq("model_id", model_id)
        .execute()
    ))
    rows = res.data or []
    if not rows:
        return {}
    # Prefer primary
    for r in rows:
        if (r.get("display_position") or "Primary").lower() == "primary":
            return r
    return rows[0]


async def _fetch_display_feature_names(display_id: int) -> set[str]:
    """Returns set of lower-cased feature names for a display."""
    res = await asyncio.to_thread(lambda: (
        _MS()
        .table("phone_display_features")
        .select("lookup_display_features(feature_name)")
        .eq("display_id", display_id)
        .execute()
    ))
    rows = res.data or []
    names = set()
    for r in rows:
        lk = r.get("lookup_display_features") or {}
        fn = (lk.get("feature_name") or "").strip().lower()
        if fn:
            names.add(fn)
    return names


# ---------------------------------------------------------------------------
# C1. display_panel_quality
# ---------------------------------------------------------------------------

# OLED panel substrings to match
_OLED_PATTERNS = {"oled", "amoled", "retina xdr", "super retina", "actua oled", "microled", "p-oled", "poled"}
_LCD_PATTERNS  = {"lcd", "ips", "tft", "pls"}


def _classify_panel(panel_type: str) -> str:
    """Returns 'oled' | 'lcd' | 'other' based on panel name."""
    pt = panel_type.lower()
    if any(p in pt for p in _OLED_PATTERNS):
        return "oled"
    if any(p in pt for p in _LCD_PATTERNS):
        return "lcd"
    return "other"


async def rule_display_panel_quality(model_id: int, url_registry_id: int) -> dict:
    """
    C1: panel_class (oled/lcd/other) + hdr_effective (bool).
    hdr_effective: TRUE if HDR10, HDR10+, or Dolby Vision in display features.
    Output columns: panel_class, hdr_effective
    """
    disp = await _fetch_primary_display(model_id)
    if not disp:
        return _make_result(
            rule_key="display_panel_quality", structured_value=None,
            inference_text="Display data unavailable.", sentiment="Neutral",
            confidence="low", input_snapshot={"model_id": model_id},
            defers_to_runb=_RUNB_DISPLAY,
        )

    pt_join    = disp.get("lookup_panel_types") or {}
    panel_type = (pt_join.get("panel_type") or "Unknown").strip()
    panel_class = _classify_panel(panel_type)
    display_id  = disp.get("display_id")

    hdr_effective = False
    feat_names: set[str] = set()
    if display_id:
        feat_names    = await _fetch_display_feature_names(display_id)
        hdr_effective = any(
            f in feat_names for f in {"hdr10", "hdr10+", "dolby vision"}
        )

    snapshot = {
        "panel_type":    panel_type,
        "panel_class":   panel_class,
        "hdr_effective": hdr_effective,
        "display_features": sorted(feat_names),
    }

    # Narrative
    is_ltpo = any("ltpo" in f or "adaptive refresh" in f for f in feat_names)
    hdr_str = "HDR10+/Dolby Vision certified" if hdr_effective else "no HDR certification"
    ltpo_str = " LTPO adaptive refresh rate for power-efficient smooth scrolling." if is_ltpo else ""

    if panel_class == "oled":
        sentiment = "Positive"
        text = (
            f"{panel_type} display — OLED technology delivers true blacks, infinite contrast, "
            f"and vibrant colours. {hdr_str}.{ltpo_str}"
        )
    elif panel_class == "lcd":
        sentiment = "Neutral"
        text = (
            f"{panel_type} display — LCD panel with solid brightness but lower contrast "
            f"and wider bezels versus OLED. {hdr_str}.{ltpo_str}"
        )
    else:
        sentiment = "Neutral"
        text = f"{panel_type} display. {hdr_str}."

    # ── HDR conflict check (spec §2.6) ───────────────────────────────────────
    # Spec says HDR-certified panel, but reviewers report HDR doesn’t work on
    # streaming platforms (e.g. Run B entry: “HDR doesn’t load on YouTube/Netflix”).
    # Flag for HITL review so admin can verify; Run B wins in the embedding.
    conflict_flag = False
    if hdr_effective:
        try:
            runb_hdr = await asyncio.to_thread(lambda: (
                get_client().schema("pipeline")
                .table("phone_experiences")
                .select("sentiment, inference_text, lookup_experience_categories(category_name)")
                .eq("url_registry_id", url_registry_id)
                .eq("is_suppressed", False)
                .eq("is_superseded", False)
                .execute()
            ))
            display_neg = [
                r for r in (runb_hdr.data or [])
                if (
                    (r.get("lookup_experience_categories") or {}).get("category_name", "").lower() == "display"
                    and (r.get("sentiment") or "").lower() == "negative"
                )
            ]
            hdr_negative = any(
                "hdr" in (r.get("inference_text") or "").lower()
                for r in display_neg
            )
            if hdr_negative:
                conflict_flag = True
                snapshot["conflict_reason"] = (
                    "Spec reports hdr_effective=True but Run B has negative "
                    "Display entries mentioning HDR (likely HDR not working on "
                    "streaming platforms)"
                )
                logger.warning(
                    "C1 display_panel_quality CONFLICT — spec hdr_effective=True but "
                    "Run B has negative HDR Display entries for "
                    "url_registry_id=%d model_id=%d",
                    url_registry_id, model_id,
                )
        except Exception as _exc_hdr:
            logger.debug("C1: HDR conflict check failed: %s", _exc_hdr)

    return _make_result(
        rule_key="display_panel_quality",
        structured_value={"panel_class": panel_class, "hdr_effective": hdr_effective},
        inference_text=text, sentiment=sentiment, confidence="high",
        input_snapshot=snapshot, defers_to_runb=_RUNB_DISPLAY,
        conflict_flag=conflict_flag,
    )


# ---------------------------------------------------------------------------
# C2. display_sharpness
# ---------------------------------------------------------------------------

async def rule_display_sharpness(model_id: int, url_registry_id: int) -> dict:
    """
    C2: PPI via double_sigmoid (spec Section 6).
    Canonical params: m1=265, s1=0.05, plateau_lo=330, plateau_hi=440, m2=490, s2=0.03
    Output column: display_sharpness_score (NUMERIC)
    """
    disp = await _fetch_primary_display(model_id)
    if not disp or disp.get("ppi") is None:
        return _make_result(
            rule_key="display_sharpness", structured_value=None,
            inference_text="PPI data unavailable — sharpness score cannot be computed.",
            sentiment="Neutral", confidence="low",
            input_snapshot={"model_id": model_id}, defers_to_runb=_RUNB_DISPLAY,
        )

    ppi   = float(disp["ppi"])
    score = double_sigmoid(ppi, m1=265, s1=0.05, plateau_lo=330, plateau_hi=440, m2=490, s2=0.03)
    score_rounded = round(score, 4)

    if ppi >= 440:
        label = "ultra-sharp"; sentiment = "Positive"
    elif ppi >= 330:
        label = "sharp"; sentiment = "Positive"
    elif ppi >= 265:
        label = "good"; sentiment = "Positive"
    elif ppi >= 200:
        label = "adequate"; sentiment = "Neutral"
    else:
        label = "basic"; sentiment = "Neutral"

    sharpness_map = {
        "ultra-sharp": f"Ultra-sharp {int(ppi)} PPI — pixel density exceeds the threshold of visual perception even at close range. Text and fine detail are razor-crisp.",
        "sharp":       f"Sharp {int(ppi)} PPI — comfortable sharpness for text, media, and gaming. Pixels are imperceptible at normal viewing distance.",
        "good":        f"{int(ppi)} PPI — good sharpness for everyday use. Text is clear; fine details are discernible at typical viewing distances.",
        "adequate":    f"{int(ppi)} PPI — adequate sharpness. Pixels may be visible when viewing close up, but comfortable for video and social media consumption.",
        "basic":       f"{int(ppi)} PPI — low pixel density. Pixelation visible on text and fine-detail imagery.",
    }
    text = sharpness_map.get(label, f"{int(ppi)} PPI — sharpness score {score_rounded}.")

    snapshot = {"ppi": int(ppi), "sharpness_score": score_rounded, "sharpness_label": label}
    return _make_result(
        rule_key="display_sharpness", structured_value=score_rounded,
        inference_text=text, sentiment=sentiment, confidence="high",
        input_snapshot=snapshot, defers_to_runb=_RUNB_DISPLAY,
    )


# ---------------------------------------------------------------------------
# C3. display_smoothness
# ---------------------------------------------------------------------------

_TOUCH_HZ_PATTERN = re.compile(r"(\d+)hz touch sampling", re.IGNORECASE)


async def rule_display_smoothness(model_id: int, url_registry_id: int) -> dict:
    """
    C3: refresh_rate via saturating_curve (knee=120, ceil=165) +
        touch_sampling_hz (extracted from display features feature_name strings).
    Output column: display_smoothness_score + refresh_rate_class
    """
    disp = await _fetch_primary_display(model_id)
    if not disp:
        return _make_result(
            rule_key="display_smoothness", structured_value=None,
            inference_text="Display data unavailable.", sentiment="Neutral",
            confidence="low", input_snapshot={"model_id": model_id},
            defers_to_runb=_RUNB_DISPLAY,
        )

    refresh_rate = float(disp.get("refresh_rate") or 60)
    display_id   = disp.get("display_id")
    feat_names   = await _fetch_display_feature_names(display_id) if display_id else set()

    # Extract touch sampling Hz from feature names like "360Hz Touch Sampling"
    touch_hz = 60  # default
    for feat in feat_names:
        m = _TOUCH_HZ_PATTERN.match(feat)
        if m:
            touch_hz = max(touch_hz, int(m.group(1)))

    is_adaptive = any("ltpo" in f or "adaptive refresh" in f for f in feat_names)

    rr_score    = saturating_curve(refresh_rate, knee=120, ceiling=165)
    touch_score = saturating_curve(touch_hz, knee=360, ceiling=720)
    # Combined: 75% refresh rate, 25% touch sampling
    smooth_score = round(0.75 * rr_score + 0.25 * touch_score, 4)

    # Refresh rate class
    if refresh_rate >= 144:
        rr_class = "ultra_high"
    elif refresh_rate >= 120:
        rr_class = "high"
    elif refresh_rate >= 90:
        rr_class = "elevated"
    else:
        rr_class = "standard"

    if rr_class in ("ultra_high", "high"):
        sentiment = "Positive"
    elif rr_class == "elevated":
        sentiment = "Positive" if touch_hz >= 240 else "Neutral"
    else:
        sentiment = "Neutral"

    adaptive_note = " LTPO adaptive rate saves battery by dropping to 1Hz when idle." if is_adaptive else ""
    touch_note    = f" {touch_hz}Hz touch sampling for ultra-responsive touch input." if touch_hz >= 240 else ""

    if rr_class == "ultra_high":
        text = (
            f"{int(refresh_rate)}Hz display — ultra-smooth scrolling, gaming, and animations. "
            f"Noticeably silkier than 60Hz panels.{adaptive_note}{touch_note}"
        )
    elif rr_class == "high":
        text = (
            f"{int(refresh_rate)}Hz display — smooth scrolling and fluid animations.{adaptive_note}{touch_note}"
        )
    elif rr_class == "elevated":
        text = (
            f"{int(refresh_rate)}Hz display — noticeably smoother than 60Hz for everyday scrolling.{touch_note}"
        )
    else:
        text = (
            f"{int(refresh_rate)}Hz display — standard refresh rate. Smooth for basic use; "
            f"gaming and fast scrolling may appear less fluid than high-refresh alternatives."
        )

    snapshot = {
        "refresh_rate_hz":    int(refresh_rate),
        "touch_sampling_hz":  touch_hz,
        "is_adaptive":        is_adaptive,
        "rr_score":           round(rr_score, 4),
        "touch_score":        round(touch_score, 4),
        "smoothness_score":   smooth_score,
        "refresh_rate_class": rr_class,
    }
    return _make_result(
        rule_key="display_smoothness",
        structured_value={"display_smoothness_score": smooth_score, "refresh_rate_class": rr_class},
        inference_text=text, sentiment=sentiment, confidence="high",
        input_snapshot=snapshot, defers_to_runb=_RUNB_DISPLAY,
    )


# ---------------------------------------------------------------------------
# C4. outdoor_visibility
# ---------------------------------------------------------------------------

async def rule_outdoor_visibility(model_id: int, url_registry_id: int) -> dict:
    """
    C4: Uses brightness_hbm ONLY. NEVER brightness_peak.
    Thresholds from spec: 1000 nits HBM = good, 1500 = excellent, 2000+ = exceptional.
    Output column: outdoor_visibility
    """
    disp = await _fetch_primary_display(model_id)
    hbm  = disp.get("brightness_hbm") if disp else None

    snapshot = {
        "brightness_hbm_nits": hbm,
        "note": "CRITICAL: outdoor_visibility uses brightness_hbm only, NEVER brightness_peak.",
    }

    if hbm is None:
        return _make_result(
            rule_key="outdoor_visibility", structured_value=None,
            inference_text="HBM brightness data unavailable — outdoor visibility cannot be rated.",
            sentiment="Neutral", confidence="low",
            input_snapshot=snapshot, defers_to_runb=_RUNB_DISPLAY,
        )

    hbm = float(hbm)

    if hbm >= 2000:
        tier = "exceptional"; sentiment = "Positive"
        text = (
            f"{int(hbm)}-nit HBM brightness — exceptional outdoor visibility. "
            f"Clearly readable in direct sunlight, on sandy beaches, and while navigating with GPS in a moving car."
        )
    elif hbm >= 1500:
        tier = "excellent"; sentiment = "Positive"
        text = (
            f"{int(hbm)}-nit HBM brightness — excellent outdoor visibility. "
            f"Comfortable reading in direct sunlight with auto-brightness engaged."
        )
    elif hbm >= 1000:
        tier = "good"; sentiment = "Positive"
        text = (
            f"{int(hbm)}-nit HBM brightness — good outdoor visibility. "
            f"Legible in most daylight conditions; slight strain in harsh direct sunlight."
        )
    elif hbm >= 600:
        tier = "adequate"; sentiment = "Neutral"
        text = (
            f"{int(hbm)}-nit HBM brightness — adequate for indoor/cloudy day use. "
            f"May be hard to read in bright outdoor sunlight."
        )
    else:
        tier = "limited"; sentiment = "Negative"
        text = (
            f"{int(hbm)}-nit HBM brightness — limited outdoor visibility. "
            f"Screen difficult to read outdoors in anything brighter than overcast conditions."
        )

    snapshot["outdoor_visibility"] = tier
    return _make_result(
        rule_key="outdoor_visibility", structured_value=tier,
        inference_text=text, sentiment=sentiment, confidence="high",
        input_snapshot=snapshot, defers_to_runb=_RUNB_DISPLAY,
    )


# ---------------------------------------------------------------------------
# C5. multimedia_experience
# ---------------------------------------------------------------------------

async def rule_multimedia_experience(model_id: int, url_registry_id: int) -> dict:
    """
    C5: speakers + Dolby Atmos/DTS in audio.audio_features + screen size (saturating_curve).
    Defers to RunB: None (reviewers rarely cover multimedia holistically).
    Output column: multimedia_tier
    """
    # Audio
    audio_res = await asyncio.to_thread(lambda: (
        _MS()
        .table("audio")
        .select("speaker_count, speaker_positions, audio_features")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    ))
    audio_rows = audio_res.data or []
    speaker_count = 0
    audio_feats   = ""
    if audio_rows:
        speaker_count = int(audio_rows[0].get("speaker_count") or 0)
        audio_feats   = (audio_rows[0].get("audio_features") or "").lower()

    has_dolby_atmos = "dolby atmos" in audio_feats
    has_dts         = "dts" in audio_feats
    has_hi_res      = "hi-res" in audio_feats or "hi res" in audio_feats

    # Display size
    disp = await _fetch_primary_display(model_id)
    size_inch = float(disp.get("size_inch") or 6.0) if disp else 6.0
    size_score = saturating_curve(size_inch, knee=6.5, ceiling=7.0)

    # Score
    stereo_score = min(1.0, speaker_count / 2.0)
    cert_score   = (0.5 if has_dolby_atmos else 0) + (0.3 if has_dts else 0) + (0.2 if has_hi_res else 0)
    cert_score   = min(1.0, cert_score)
    total        = 0.40 * stereo_score + 0.35 * cert_score + 0.25 * size_score

    if total >= 0.70:
        tier = "premium"; sentiment = "Positive"
    elif total >= 0.45:
        tier = "good"; sentiment = "Positive"
    elif total >= 0.25:
        tier = "standard"; sentiment = "Neutral"
    else:
        tier = "basic"; sentiment = "Neutral"

    # Build narrative
    audio_parts = []
    if speaker_count >= 2:
        audio_parts.append(f"stereo speakers ({speaker_count}-speaker setup)")
    elif speaker_count == 1:
        audio_parts.append("mono speaker")
    if has_dolby_atmos:
        audio_parts.append("Dolby Atmos")
    if has_dts:
        audio_parts.append("DTS audio")
    if has_hi_res:
        audio_parts.append("Hi-Res audio certified")
    audio_str = ", ".join(audio_parts) or "basic audio"

    screen_str = (
        f"large {size_inch:.1f}-inch screen" if size_inch >= 6.5
        else f"{size_inch:.1f}-inch screen"
    )

    if tier == "premium":
        text = f"Premium multimedia setup: {audio_str}, {screen_str}. Excellent for video consumption and gaming."
    elif tier == "good":
        text = f"Good multimedia experience: {audio_str}, {screen_str}."
    elif tier == "standard":
        text = f"Standard multimedia: {audio_str}, {screen_str}. Adequate for everyday media consumption."
    else:
        text = f"Basic multimedia: {audio_str}, {screen_str}. Limited for immersive media experiences."

    snapshot = {
        "speaker_count":     speaker_count,
        "has_dolby_atmos":   has_dolby_atmos,
        "has_dts":           has_dts,
        "has_hi_res":        has_hi_res,
        "size_inch":         size_inch,
        "size_score":        round(size_score, 3),
        "total_score":       round(total, 3),
        "multimedia_tier":   tier,
    }
    return _make_result(
        rule_key="multimedia_experience", structured_value=tier,
        inference_text=text, sentiment=sentiment, confidence="medium",
        input_snapshot=snapshot,
    )


# ---------------------------------------------------------------------------
# Group C handler registry
# ---------------------------------------------------------------------------

GROUP_C_HANDLERS: dict[str, Any] = {
    "display_panel_quality":    rule_display_panel_quality,
    "display_sharpness":        rule_display_sharpness,
    "display_smoothness":       rule_display_smoothness,
    "outdoor_visibility":       rule_outdoor_visibility,
    "multimedia_experience":    rule_multimedia_experience,
}
