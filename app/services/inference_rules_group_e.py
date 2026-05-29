# app/services/inference_rules_group_e.py
"""
Run C — Deterministic Inference Engine
Group E: Camera — 4 rules (ALL low confidence)

Rules E1–E4:
  E1: main_camera_hardware  — sensor MP + sensor_size_decimal + OIS
  E2: camera_versatility    — lens type set (ultrawide/telephoto/macro/depth counts)
  E3: zoom_capability       — optical_zoom from Telephoto/Periscope lens
  E4: video_capability      — max video resolution + stabilization + front 4K

CRITICAL: All Group E rules have confidence='low'.
Run B is the authoritative camera voice. Group E provides hard-filter signals only.
Defers to RunB: Camera (all E rules).

Lens type canonical names from lookup_lens_types:
  'Main', 'Ultra-wide', 'Telephoto', 'Periscope', 'Macro', 'Depth',
  'Front', 'Front (Cover Display)', 'Front (Inner Display)', 'Front (Secondary)'
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from app.core.supabase_client import get_client

logger = logging.getLogger(__name__)

_MS = lambda: get_client().schema("mobile_specs")

_RUNB_CAMERA = "Camera"

# Lens type sets for classification
_WIDE_TYPES      = {"main"}
_ULTRAWIDE_TYPES = {"ultra-wide"}
_TELE_TYPES      = {"telephoto", "periscope"}
_MACRO_TYPES     = {"macro"}
_FRONT_TYPES     = {"front", "front (cover display)", "front (inner display)", "front (secondary)"}


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


async def _fetch_all_lenses(model_id: int) -> list[dict]:
    """Returns all camera_lens_specs rows for model, with lens_type and stabilization."""
    res = await asyncio.to_thread(lambda: (
        _MS()
        .table("camera_lens_specs")
        .select(
            "lens_id, megapixels, sensor_size_decimal, aperture, "
            "optical_zoom_capacity, is_macro_capable, "
            "lookup_lens_types(lens_type)"
        )
        .eq("model_id", model_id)
        .execute()
    ))
    return res.data or []


async def _fetch_ois_lens_ids(model_id: int) -> set[int]:
    """Returns lens_ids that have OIS/Sensor-shift OIS/Gimbal OIS stabilization."""
    # Get all lens_ids for model
    lens_res = await asyncio.to_thread(lambda: (
        _MS()
        .table("camera_lens_specs")
        .select("lens_id")
        .eq("model_id", model_id)
        .execute()
    ))
    lens_ids = [r["lens_id"] for r in (lens_res.data or []) if r.get("lens_id")]
    if not lens_ids:
        return set()

    stab_res = await asyncio.to_thread(lambda: (
        _MS()
        .table("lens_stabilization")
        .select("lens_id, lookup_stabilization_types(stabilization_type)")
        .in_("lens_id", lens_ids)
        .execute()
    ))
    ois_ids: set[int] = set()
    for r in (stab_res.data or []):
        lk = r.get("lookup_stabilization_types") or {}
        st = (lk.get("stabilization_type") or "").lower()
        if "ois" in st:  # matches OIS, Sensor-shift OIS, Gimbal OIS
            ois_ids.add(r["lens_id"])
    return ois_ids


# ---------------------------------------------------------------------------
# E1. main_camera_hardware
# ---------------------------------------------------------------------------

async def rule_main_camera_hardware(model_id: int, url_registry_id: int) -> dict:
    """
    E1: Main lens (type='Main') megapixels + sensor_size_decimal + OIS.
    main_camera_hw: 'flagship'|'premium'|'standard'|'basic'
    """
    lenses  = await _fetch_all_lenses(model_id)
    ois_ids = await _fetch_ois_lens_ids(model_id)

    main_lens = None
    for l in lenses:
        lt = (l.get("lookup_lens_types") or {}).get("lens_type", "")
        if lt.lower() == "main":
            main_lens = l
            break
    if main_lens is None and lenses:
        # Fallback: largest sensor among rear lenses
        rear = [
            l for l in lenses
            if (l.get("lookup_lens_types") or {}).get("lens_type", "").lower() not in
               {t.lower() for t in _FRONT_TYPES}
        ]
        if rear:
            main_lens = max(rear, key=lambda x: float(x.get("sensor_size_decimal") or 0))

    if main_lens is None:
        return _make_result(
            rule_key="main_camera_hardware", structured_value=None,
            inference_text="Camera hardware data unavailable.",
            sentiment="Neutral", confidence="low",
            input_snapshot={"model_id": model_id}, defers_to_runb=_RUNB_CAMERA,
        )

    mp         = float(main_lens.get("megapixels") or 12)
    sensor_dec = float(main_lens.get("sensor_size_decimal") or 0.5)
    has_ois    = main_lens.get("lens_id") in ois_ids
    aperture   = main_lens.get("aperture")

    # Hardware tier (sensor size is the dominant factor)
    if sensor_dec >= 0.75:     # >= 1/1.33" sensor
        tier = "flagship"
    elif sensor_dec >= 0.50:   # >= 1/2.0" sensor
        tier = "premium"
    elif sensor_dec >= 0.33:   # >= 1/3.0" sensor
        tier = "standard"
    else:
        tier = "basic"

    sentiment = "Positive" if tier in ("flagship", "premium") else "Neutral"

    tier_texts = {
        "flagship": f"Large-sensor main camera ({mp:.0f}MP, 1/{1/sensor_dec:.2f}\" sensor{'+ OIS' if has_ois else ''}). Large sensor captures significantly more light for outstanding low-light and dynamic range performance.",
        "premium":  f"Good main camera ({mp:.0f}MP, 1/{1/sensor_dec:.2f}\" sensor{'+ OIS' if has_ois else ''}). Solid sensor size for well-lit and low-light photography.",
        "standard": f"Standard main camera ({mp:.0f}MP{'+ OIS' if has_ois else ''}). Produces good results in daylight; limited in low-light without post-processing.",
        "basic":    f"Basic main camera ({mp:.0f}MP{'+ OIS' if has_ois else ''}). Small sensor — performance dependent on software processing. Expect limitations in low light.",
    }
    text = tier_texts[tier]

    snapshot = {
        "lens_id":           main_lens.get("lens_id"),
        "megapixels":        mp,
        "sensor_size_decimal": sensor_dec,
        "aperture":          aperture,
        "has_ois":           has_ois,
        "main_camera_hw":    tier,
    }
    return _make_result(
        rule_key="main_camera_hardware", structured_value=tier,
        inference_text=text, sentiment=sentiment, confidence="low",
        input_snapshot=snapshot, defers_to_runb=_RUNB_CAMERA,
    )


# ---------------------------------------------------------------------------
# E2. camera_versatility
# ---------------------------------------------------------------------------

async def rule_camera_versatility(model_id: int, url_registry_id: int) -> dict:
    """
    E2: Counts ultrawide, telephoto, macro, depth lenses. Returns JSONB profile.
    Output column: camera_versatility
    """
    lenses = await _fetch_all_lenses(model_id)

    rear_lenses = [
        l for l in lenses
        if (l.get("lookup_lens_types") or {}).get("lens_type", "").lower() not in
           {t.lower() for t in _FRONT_TYPES}
    ]

    has_ultrawide = False
    has_telephoto = False
    has_periscope = False
    has_macro     = False
    has_depth     = False
    lens_count    = len(rear_lenses)

    for l in rear_lenses:
        lt = (l.get("lookup_lens_types") or {}).get("lens_type", "").lower()
        if lt in _ULTRAWIDE_TYPES:
            has_ultrawide = True
        if lt == "telephoto":
            has_telephoto = True
        if lt == "periscope":
            has_periscope = True
            has_telephoto = True  # periscope is a type of telephoto
        if lt in _MACRO_TYPES or bool(l.get("is_macro_capable")):
            has_macro = True
        if lt == "depth":
            has_depth = True

    profile = {
        "has_ultrawide": has_ultrawide,
        "has_telephoto": has_telephoto,
        "has_periscope": has_periscope,
        "has_macro":     has_macro,
        "lens_count":    lens_count,
    }

    versatility_score = (
        (1 if has_ultrawide else 0)
        + (1 if has_telephoto else 0)
        + (0.5 if has_macro else 0)
        + (0.3 * min(lens_count, 4) / 4)
    )

    if versatility_score >= 2.0:
        label = "versatile"; sentiment = "Positive"
        cameras_str = " + ".join(filter(None, [
            "main", "ultrawide" if has_ultrawide else None,
            "periscope telephoto" if has_periscope else ("telephoto" if has_telephoto else None),
            "macro" if has_macro else None,
        ]))
        text = f"Versatile {lens_count}-camera system ({cameras_str}). Covers wide, standard, telephoto, and closeup use-cases."
    elif versatility_score >= 1.0:
        label = "capable"; sentiment = "Positive"
        extras = []
        if has_ultrawide: extras.append("ultrawide")
        if has_telephoto: extras.append("telephoto")
        if has_macro: extras.append("macro")
        text = f"Capable {lens_count}-camera system with main + {' and '.join(extras) if extras else 'secondary lens'}. Good for typical photography needs."
    elif lens_count >= 2:
        label = "basic_multi"; sentiment = "Neutral"
        text = f"{lens_count}-camera system. Primary lens is most useful; secondary lens(es) may be depth/low-quality sensors."
    else:
        label = "single"; sentiment = "Neutral"
        text = "Single rear camera — no versatility benefit. Quality depends entirely on the main sensor."

    profile["versatility_label"] = label
    snapshot = {"lenses": rear_lenses, "profile": profile}

    return _make_result(
        rule_key="camera_versatility", structured_value=profile,
        inference_text=text, sentiment=sentiment, confidence="low",
        input_snapshot=snapshot, defers_to_runb=_RUNB_CAMERA,
    )


# ---------------------------------------------------------------------------
# E3. zoom_capability
# ---------------------------------------------------------------------------

async def rule_zoom_capability(model_id: int, url_registry_id: int) -> dict:
    """
    E3: Optical zoom from Telephoto or Periscope lens.
    Output column: zoom_capability
    """
    lenses = await _fetch_all_lenses(model_id)

    best_optical = 0.0
    has_periscope = False

    for l in lenses:
        lt  = (l.get("lookup_lens_types") or {}).get("lens_type", "").lower()
        opt = float(l.get("optical_zoom_capacity") or 0)
        if lt in ("telephoto", "periscope"):
            if opt > best_optical:
                best_optical = opt
            if lt == "periscope":
                has_periscope = True

    snapshot = {
        "best_optical_zoom": best_optical,
        "has_periscope":     has_periscope,
    }

    if best_optical == 0:
        tier = "none"; sentiment = "Neutral"
        text = "No optical zoom — only digital zoom available. Quality degrades significantly when zooming in."
    elif best_optical >= 10:
        tier = "exceptional"; sentiment = "Positive"
        text = f"{best_optical:.0f}x optical zoom{'(periscope)' if has_periscope else ''}. Exceptional zoom range for distant subjects, wildlife, and sports."
    elif best_optical >= 5:
        tier = "very_good"; sentiment = "Positive"
        text = f"{best_optical:.0f}x optical zoom{'(periscope)' if has_periscope else ''}. Very good zoom for portraits, events, and everyday telephoto shots."
    elif best_optical >= 3:
        tier = "good"; sentiment = "Positive"
        text = f"{best_optical:.0f}x optical zoom. Useful for portraits and moderate zoom shots without quality loss."
    else:
        tier = "basic"; sentiment = "Neutral"
        text = f"{best_optical:.1f}x optical zoom. Minimal zoom benefit — digital zoom will be required beyond this range."

    snapshot["zoom_capability"] = tier
    return _make_result(
        rule_key="zoom_capability", structured_value=tier,
        inference_text=text, sentiment=sentiment, confidence="low",
        input_snapshot=snapshot, defers_to_runb=_RUNB_CAMERA,
    )


# ---------------------------------------------------------------------------
# E4. video_capability
# ---------------------------------------------------------------------------

_4K_PATTERN = re.compile(r"4k|2160p|3840", re.IGNORECASE)


async def rule_video_capability(model_id: int, url_registry_id: int) -> dict:
    """
    E4: Max video resolution + has_stabilization (OIS/EIS on main lens) + front 4K.
    Output column: video_capability (JSONB)
    """
    # Video resolutions
    vid_res = await asyncio.to_thread(lambda: (
        _MS()
        .table("video_capabilities")
        .select("rear_video_resolutions, front_video_resolutions")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    ))
    vid_rows = vid_res.data or []
    rear_video  = (vid_rows[0].get("rear_video_resolutions")  or "") if vid_rows else ""
    front_video = (vid_rows[0].get("front_video_resolutions") or "") if vid_rows else ""

    # Max video resolution tier
    rear_lower = rear_video.lower()
    if "8k" in rear_lower or "4320" in rear_lower:
        max_video = "8K"
    elif _4K_PATTERN.search(rear_video):
        max_video = "4K"
    elif "1080" in rear_lower or "fhd" in rear_lower:
        max_video = "1080p"
    elif "720" in rear_lower or "hd" in rear_lower:
        max_video = "720p"
    elif rear_video:
        max_video = "other"
    else:
        max_video = "unknown"

    front_4k = bool(_4K_PATTERN.search(front_video))

    # Stabilization: check OIS on any rear lens
    ois_ids = await _fetch_ois_lens_ids(model_id)
    lenses  = await _fetch_all_lenses(model_id)
    rear_lenses = [
        l for l in lenses
        if (l.get("lookup_lens_types") or {}).get("lens_type", "").lower() not in
           {t.lower() for t in _FRONT_TYPES}
    ]
    has_stabilization = any(l.get("lens_id") in ois_ids for l in rear_lenses)

    capability = {
        "max_video":          max_video,
        "has_stabilization":  has_stabilization,
        "front_4k":           front_4k,
    }

    video_scores = {"8K": 1.0, "4K": 0.80, "1080p": 0.55, "720p": 0.30, "other": 0.20, "unknown": 0.0}
    score = video_scores.get(max_video, 0) + (0.15 if has_stabilization else 0) + (0.05 if front_4k else 0)
    sentiment = "Positive" if score >= 0.80 else ("Neutral" if score >= 0.50 else "Negative")

    stab_note = " OIS stabilisation for smooth handheld footage." if has_stabilization else " No OIS — handheld video may show shake."
    front_note = " Front camera supports 4K video." if front_4k else ""

    if max_video == "8K":
        text = f"8K video recording.{stab_note}{front_note}"
    elif max_video == "4K":
        text = f"4K video recording.{stab_note}{front_note}"
    elif max_video == "1080p":
        text = f"Full HD (1080p) video — no 4K support.{stab_note}{front_note}"
    elif max_video == "720p":
        text = f"HD (720p) video only — significantly below current standards.{stab_note}"
    else:
        text = "Video recording capability unknown."

    snapshot = {
        "rear_video_resolutions":  rear_video,
        "front_video_resolutions": front_video,
        "max_video":               max_video,
        "has_stabilization":       has_stabilization,
        "front_4k":                front_4k,
    }
    return _make_result(
        rule_key="video_capability", structured_value=capability,
        inference_text=text, sentiment=sentiment, confidence="low",
        input_snapshot=snapshot, defers_to_runb=_RUNB_CAMERA,
    )


# ---------------------------------------------------------------------------
# Group E handler registry
# ---------------------------------------------------------------------------

GROUP_E_HANDLERS: dict[str, Any] = {
    "main_camera_hardware": rule_main_camera_hardware,
    "camera_versatility":   rule_camera_versatility,
    "zoom_capability":      rule_zoom_capability,
    "video_capability":     rule_video_capability,
}
