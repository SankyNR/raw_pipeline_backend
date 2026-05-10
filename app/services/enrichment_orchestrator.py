"""
Phase 6 — Enrichment

Task 6.1 — Confidence Adjustment (adjust_confidence_for_source_url)
Task 6.2 — Enrichment Orchestrator     (run_enrichment)

What this does:
    For every phone coming out of gap analysis, fires one Gemini grounded
    search query per missing field in pipeline.missing_fields_log.
    Stores candidates in enrichment_field_candidates with both raw and
    adjusted confidence scores, then auto-selects the best candidate where
    the rules allow it.

Key design points:
    - Supabase-py is synchronous; all DB calls are wrapped in asyncio.to_thread().
    - call_gemini_grounded() includes its own 3× exponential backoff retry
      (2s → 4s → 8s). Per-field errors are caught and logged; the run continues.
    - Chipset-level deduplication: if chipset.npu_tops is already resolved for
      another phone with the same chipset_name, we copy that candidate instead
      of firing a new API call.
    - Cost tracking: ₹2.94 accumulated per grounded query, written to
      enrichment_runs.total_api_cost_inr at run completion.

Auto-selection rules (one is_selected=TRUE per field):
    1. source_tier='oem_official' AND confidence >= 0.85 → select immediately
    2. Delta vs best existing candidate for this field >= 0.20 → select higher
       (previous best is deselected before inserting the new one — E1 fix)
    3. All else → leave FALSE (human review in admin UI)
"""

from __future__ import annotations

import asyncio
import logging
import re  # E7 fix — top-level import, not inline
from typing import Any

from app.repositories.extraction_repository import (
    deselect_all_candidates_for_field,          # E1 fix
    fetch_best_existing_candidate_confidence,    # E8 fix — top-level
    fetch_chipset_enrichment_copy,
    fetch_gap_fields_for_enrichment,
    fetch_normalized_spec,                       # E8 + E11 fix — top-level
    insert_enrichment_candidate,
    insert_enrichment_run,
    insert_enrichment_search_query,
    update_enrichment_run,
    update_enrichment_search_query,
    update_missing_field_log_attempted,
)
from app.services.gemini_client import (
    GeminiNonRetryableError,
    GeminiRateLimitError,
    GeminiTransientError,
    call_gemini_grounded,
)
from app.utils.path_builder import slugify

logger = logging.getLogger(__name__)

def _is_valid_value(val: Any) -> bool:
    """
    Returns True only if val is a non-null, non-empty value worth inserting.
    Guards against Gemini returning None, empty string, or whitespace-only strings
    which would either violate the NOT NULL constraint or insert meaningless data.
    """
    if val is None:
        return False
    if isinstance(val, str) and val.strip() == "":
        return False
    return True


def _clean_extracted_value(val: Any) -> Any:
    """
    Strips double-quote wrapping that Gemini occasionally adds to string values.
    \"Wi-Fi 6\"  →  Wi-Fi 6
    Does NOT modify booleans, ints, floats, None, or lists.
    """
    if isinstance(val, str):
        return val.strip().strip('"')
    return val

# ---------------------------------------------------------------------------
# Cost constant
# ---------------------------------------------------------------------------

_GROUNDED_QUERY_COST_INR: float = 2.94  # paid tier, ₹ per grounded query

# Maximum fields processed per enrichment run.
# Prevents cost explosion on flagship phones with 60+ null fields after normalization.
# fetch_gap_fields_for_enrichment already returns rows ordered high→medium priority,
# so slicing preserves the most impactful fields.
_MAX_ENRICHMENT_FIELDS_PER_RUN: int = 20


# ---------------------------------------------------------------------------
# Task 6.1 — Source URL quality → confidence penalty
# ---------------------------------------------------------------------------

def adjust_confidence_for_source_url(
    raw_confidence: float,
    source_url: str | None,
    brand: str,
    model: str,
) -> float:
    """
    Applies source URL quality penalty to raw Gemini confidence.
    Returns adjusted confidence (stored as 'confidence' in
    enrichment_field_candidates). raw_confidence stored separately for audit.

    PENALTY RULES:
      URL contains brand_slug AND model_slug → no penalty  (factor 1.00)
      URL contains brand_slug but not model  → × 0.85
      URL contains neither                   → × 0.80
      source_url is None (parametric)        → × 0.85

    Uses slugify() from app/utils/path_builder.py.
    """
    if source_url is None:
        return round(raw_confidence * 0.85, 2)

    brand_slug = slugify(brand)
    model_slug = slugify(model)
    url_lower  = source_url.lower()

    has_brand = brand_slug in url_lower
    has_model = model_slug in url_lower

    if has_brand and has_model:
        factor = 1.00
    elif has_brand:
        factor = 0.85
    else:
        factor = 0.80

    return round(raw_confidence * factor, 2)


# ---------------------------------------------------------------------------
# Source tier classification
# ---------------------------------------------------------------------------

# Domain substring → source_tier
# Checked in order — first match wins.
_TIER_RULES: list[tuple[list[str], str]] = [
    # OEM official brand domains (phone manufacturers only — India-specific)
    (
        [
            "samsung.com/in", "apple.com/in", "oneplus.in", "mi.com/in",
            "in.nothing.tech", "realme.com/in", "oppo.com/in", "vivo.com/in", "motorola.in",
            "iqoo.com/in", "poco.in", "store.google.com",
        ],
        "oem_official",
    ),
    # S4-P1-8: chipset vendors moved from oem_official to their own tier.
    # They are authoritative for chipset.* fields ONLY, not phone-level fields
    # (battery, display brightness, software). Auto-selection is gated in _should_auto_select.
    (
        ["qualcomm.com", "mediatek.com"],
        "chipset_vendor",
    ),
    # Trusted aggregators / spec databases
    (
        [
            "gsmarena.com", "smartprix.com", "91mobiles.com",
            "notebookcheck.net", "devicespecifications.com",
            "kimovil.com", "phonearena.com",
        ],
        "trusted_aggregator",
    ),
    # Tech media / journalism
    (
        [
            "theverge.com", "androidauthority.com", "gadgets360.com", "dxomark.com",
            "techradar.com", "cnet.com", "wired.com", "androidcentral.com",
            "tomsguide.com", "digitaltrends.com", "indianexpress.com", "croma.com",
        ],
        "tech_media",
    ),
    # Forums / user-generated
    (
        ["reddit.com", "xda-developers.com", "forums.oneplus.com"],
        "forum",
    ),
]


def _determine_source_tier(source_domain: str | None) -> str:
    """
    Maps a domain string to one of: oem_official, trusted_aggregator,
    tech_media, forum, unknown.
    """
    if not source_domain:
        return "unknown"
    domain_lower = source_domain.lower()
    for domains, tier in _TIER_RULES:
        if any(d in domain_lower for d in domains):
            return tier
    return "unknown"


# ---------------------------------------------------------------------------
# Enrichment prompt builder
# ---------------------------------------------------------------------------

# Human-readable labels for known field paths. Used to form natural-language
# questions in the enrichment prompt. Falls back to raw field_path if not found.
_FIELD_LABELS: dict[str, str] = {
    # Certifications
    "certifications.sar_head":          "SAR (head) radiation level in W/kg (India, 0mm separation)",
    "certifications.sar_body":          "SAR (body) radiation level in W/kg (India, 0mm separation)",
    "certifications.widevine_level":    "Widevine DRM certification level (L1, L2, or L3)",
    "certifications.charger_in_box":    "whether a charger is included in the retail box sold in India",
    # Chipset
    "chipset.npu_tops":                 "NPU AI performance in TOPS (Tera Operations Per Second)",
    "chipset.fabrication_node":         "fabrication node size in nm",
    "chipset.cpu_high_performance_cores": "prime/performance core configuration (e.g. '1x Cortex-X4 @ 3.3GHz')",
    "chipset.cpu_performance_cores":    "mid-tier core configuration",
    "chipset.cpu_efficiency_cores":     "efficiency core configuration",
    # Network
    "network.vo5g":                     "VoNR / Voice over 5G support",
    "network.vowifi":                   "VoWiFi / Wi-Fi Calling support",  # S4-P1-6: was network.vo_wifi (ghost)
    # Charging — S4-P1-6: corrected to v5 schema field names
    "charging.pd_support":              "USB Power Delivery (PD) support for third-party charger compatibility",
    "charging.charging_power":          "maximum wired charging speed in watts",          # was max_charging_speed_watt
    "charging.wireless_charging_power": "maximum wireless charging speed in watts",       # was wireless_charging_speed_watt
    # Display — S4-P1-6: brightness_nits ghost removed; v5 paths used
    "displays[*].panel_type":           "display panel technology (e.g. LTPO OLED, AMOLED, IPS LCD)",
    "displays[*].refresh_rate":         "maximum display refresh rate in Hz",
    "displays[*].brightness_hbm":       "HBM (High Brightness Mode) brightness in nits",  # was brightness_nits (ghost)
    "displays[*].brightness_peak":      "peak brightness in nits",
    "displays[*].glass_protection":     "front glass protection type (e.g. Gorilla Glass Victus 2)",
    # Camera
    "camera_lenses[*].megapixels":      "camera sensor resolution in megapixels",
    "camera_lenses[*].aperture":        "lens aperture (e.g. f/1.8)",
    "camera_lenses[*].sensor_model":    "camera sensor model name",
    # Certifications — IP ratings
    "certifications.ip_ratings":        "IP water and dust resistance rating(s) (e.g. IP68, IP69K)",
    # OS & Security — S4-P1-6: launch_os_version is a ghost; correct v5 path is os_name
    "os_and_security.os_name":            "Android version at launch (e.g. Android 15)",  # was launch_os_version (ghost)
    "os_and_security.os_update_years":    "number of guaranteed major OS update years",
    "os_and_security.security_update_years": "number of guaranteed security patch years",

    # Body — N/A-aware labels for form-factor-specific fields
    "body.height_folded":   "folded height in mm — ONLY for foldable/flip phones. "
                            "If this phone is NOT foldable or flip-style, return null.",
    "body.width_folded":    "folded width in mm — ONLY for foldable/flip phones. "
                            "If this phone is NOT foldable or flip-style, return null.",
    "body.thickness_folded": "folded thickness in mm — ONLY for foldable/flip phones. "
                            "If this phone is NOT foldable or flip-style, return null.",
    "body.has_stylus":      "Does this phone include or officially support a stylus pen? "
                            "Answer true or false only.",
    "body.stylus_features": "stylus pen features — ONLY if this phone ships with or "
                            "officially supports a stylus. If it does not have a stylus, return null.",

    # Charging — boolean fields with explicit yes/no framing
    "charging.wireless_charging":           "Does this phone natively support wireless "
                                            "charging without any adapter or mod? "
                                            "Answer true or false only.",
    "charging.reverse_wireless_charging":   "Does this phone support reverse wireless "
                                            "charging (wirelessly powering other devices)? "
                                            "Answer true or false only.",
    "charging.reverse_charging":            "Does this phone support reverse wired charging "
                                            "(charging other devices via USB cable)? "
                                            "Answer true or false only.",
    "charging.charger_in_box":              "Is a charger included in the India retail box? "
                                            "Answer true or false only.",

    # Network — boolean fields with explicit yes/no framing
    "network.volte":   "Does this phone support VoLTE (Voice over LTE / 4G calling)? "
                       "Answer true or false only.",
    "network.vowifi":  "Does this phone support VoWiFi (Wi-Fi Calling)? "
                       "Answer true or false only.",

    # Connectivity — boolean fields
    "connectivity.ir_blaster":   "Does this phone have a built-in IR blaster? "
                                 "Answer true or false only.",
    "connectivity.uwb":          "Does this phone support UWB (Ultra-Wideband) connectivity? "
                                 "Answer true or false only.",
    "connectivity.wifi_hotspot": "Does this phone support Wi-Fi hotspot / mobile hotspot "
                                 "tethering? Answer true or false only.",
    "connectivity.nfc":          "Does this phone have NFC (Near Field Communication)? "
                                 "Answer true or false only.",

    # Audio
    "audio.microphone_count":     "microphone count (number of microphones on this phone)",
    "audio.audio_features":       "audio features including speaker setup, Dolby support, "
                                  "spatial audio, and any notable audio capabilities",
    "audio.microphone_positions": "microphone positions on the device (e.g. top, bottom, side)",
    "audio.has_3_5mm_jack":       "Does this phone have a 3.5mm headphone jack? "
                                  "Answer true or false only.",

    # Charging — text fields
    "charging.charger_technologies":            "charger technologies supported "
                                                "(e.g. USB Power Delivery, Quick Charge 5.0)",
    "charging.battery_and_charging_features":   "battery and charging features summary",
    "charging.reverse_wireless_charging_power": "reverse wireless charging power output in watts",
    "charging.wireless_charging_standard":      "wireless charging standard supported "
                                                "(e.g. Qi, Qi2, MagSafe)",

    # Network
    "network.sim_configuration": "SIM card slot configuration "
                                 "(e.g. Dual SIM nano+nano, Nano-SIM + eSIM)",

    # Connectivity
    "connectivity.wifi_standard":     "Wi-Fi generation or standard supported "
                                      "(e.g. Wi-Fi 6, Wi-Fi 5, Wi-Fi 6E)",
    "connectivity.wifi_technologies": "Wi-Fi technologies supported "
                                      "(e.g. OFDMA, MU-MIMO, Wi-Fi Direct, WPA3)",

    # Content
    "in_the_box": "items included in the India retail box",

    # Certifications
    "certifications.video_certifications":  "video streaming / display certifications "
                                            "(e.g. Dolby Vision, HDR10+, Netflix HDR)",
    "certifications.widevine_support":      "Does this phone support Widevine DRM? "
                                            "Answer true or false only.",
    "certifications.bis_certification":     "Does this phone have BIS (Bureau of Indian "
                                            "Standards) certification for India? "
                                            "Answer true or false only.",
    "certifications.other_certifications":  "other notable certifications this phone has "
                                            "(e.g. MIL-STD-810H, safety ratings)",

    # Video
    "video_capabilities.slow_motion":            "slow motion video recording capability "
                                                 "(fps and resolution, e.g. 960fps @ 720p)",
    "video_capabilities.rear_video_resolutions":  "rear camera maximum video recording resolution "
                                                 "(e.g. 4K @ 60fps, 8K @ 24fps)",
    "video_capabilities.front_video_resolutions": "front camera maximum video recording resolution",

    # OS
    "os_and_security.ui_skin":        "Android UI skin / launcher version "
                                      "(e.g. My UX, One UI 7, OxygenOS 15)",
    "os_and_security.unlock_methods": "available screen unlock methods "
                                      "(e.g. PIN, Pattern, Password, Face Unlock)",
}


def build_enrichment_prompt(
    brand: str,
    model: str,
    field_path: str,
    site_hint: str | None,
    template_override: str | None,
) -> str:
    """
    Builds a natural-language enrichment query for Gemini grounded search.

    If template_override is set it replaces the default question body entirely
    (used when missing_fields_log.query_template_override is populated).

    Returns a complete prompt string ready for call_gemini_grounded().
    """
    if template_override:
        try:
            question_body = template_override.format(brand=brand, model=model)
        except KeyError as exc:
            # S4-P1-5: template may contain literal {braces} for non-format placeholders
            # (e.g. "{mid-segment} price"). Fall back to the raw template as-is.
            logger.warning(
                "build_enrichment_prompt: template_override.format() KeyError for "
                "field_path=%r (missing key: %s). Using raw template as fallback.",
                field_path, exc,
            )
            question_body = template_override  # raw template — braces not substituted
    else:
        # Normalise array indices to wildcard for label lookup
        # E.g. "displays[0].panel_type" → "displays[*].panel_type"
        generic_key = re.sub(r"\[\d+\]", "[*]", field_path)  # E7 fix — uses top-level re
        label = _FIELD_LABELS.get(generic_key) or _FIELD_LABELS.get(field_path) or field_path
        question_body = f"What is the {label} for the {brand} {model} sold in India?"

    site_line = (
        f"If available, prefer results from {site_hint}.\n\n"
        if site_hint else ""
    )

    output_schema_hint = (
        '{"value": <extracted value or null>, '
        '"confidence": <float 0.0-1.0>, '
        '"evidence": "<verbatim phrase from source>"}'
    )

    return (
        f"{site_line}"
        f"{question_body}\n\n"
        f"Return ONLY valid JSON (no markdown, no explanation) matching:\n"
        f"{output_schema_hint}\n\n"
        f"Rules:\n"
        f"- If the value is genuinely unknown, return null for 'value' and set confidence <= 0.40.\n"
        f"- 'evidence' must be a verbatim phrase from the source, not a paraphrase.\n"
        f"- For India-specific fields (SAR, charger-in-box, price, availability), "
        f"prefer Indian market data over global spec sheets."
    )


# ---------------------------------------------------------------------------
# Task 6.2 — Enrichment Orchestrator
# ---------------------------------------------------------------------------

async def run_enrichment(
    normalized_id: int,
    brand: str,
    model: str,
) -> dict:
    """
    Batch enrichment session covering ALL missing fields for one phone.

    Returns:
        {
            "success":             bool,
            "enrichment_run_id":   int,
            "fields_targeted":     int,
            "fields_resolved":     int,   # fields where value is not None
            "total_api_cost_inr":  float,
        }

    Flow:
        1. Fetch normalized_spec_json ONCE → url_registry_id + chipset_name (E11 fix)
        2. INSERT enrichment_runs → enrichment_run_id
        3. Fetch all missing_fields_log rows where enrichment_attempted=FALSE
        4. Per field:
             a. Chipset dedup check (chipset.npu_tops only)
             b. Build prompt
             c. INSERT enrichment_search_queries → query_id
             d. call_gemini_grounded() with retry
             e. Classify source tier + adjust confidence (E5 fix)
             f. Deselect existing candidates if E1 rule applies (E1 fix)
             g. INSERT enrichment_field_candidates
             h. UPDATE missing_fields_log.enrichment_attempted
             i. UPDATE enrichment_search_queries with cost/status
        5. Zero-resolution → status='failed' (E4 fix)
        6. UPDATE enrichment_runs with final summary
    """
    logger.info(
        "run_enrichment: START normalized_id=%d brand=%r model=%r",
        normalized_id, brand, model,
    )

    # Step 1: Fetch normalized spec ONCE — extract url_registry_id + chipset_name
    # E11 fix: single fetch shared between run creation and chipset dedup
    norm_row = await asyncio.to_thread(fetch_normalized_spec, normalized_id)
    url_registry_id: int       = norm_row["url_registry_id"]
    norm_json: dict            = norm_row.get("normalized_json") or {}
    chipset_name_for_run: str | None = (
        (norm_json.get("chipset") or {}).get("chipset_name")
    )

    # Step 2: Create enrichment run record
    enrichment_run_id: int = await asyncio.to_thread(
        insert_enrichment_run,
        {
            "normalized_id":   normalized_id,
            "url_registry_id": url_registry_id,
            "search_provider": "gemini_grounding",
            "status":          "running",
        },
    )

    # Steps 3–6: wrapped in global crash guard
    # If anything between here and the final run update crashes unhandled,
    # the run record is marked 'failed' instead of staying 'running' forever.
    fields_targeted  = 0
    fields_resolved  = 0
    cost_accumulator = 0.0
    run_status       = "completed"
    try:
        # Step 3: Load all pending gap fields
        gap_fields: list[dict] = await asyncio.to_thread(
            fetch_gap_fields_for_enrichment, normalized_id
        )

        # Cap to prevent cost explosion on phones with many null fields.
        # fetch_gap_fields_for_enrichment orders by priority (high → medium),
        # so slicing preserves the most important fields.
        if len(gap_fields) > _MAX_ENRICHMENT_FIELDS_PER_RUN:
            logger.warning(
                "run_enrichment: normalized_id=%d has %d gap fields — "
                "capping to %d to control cost.",
                normalized_id, len(gap_fields), _MAX_ENRICHMENT_FIELDS_PER_RUN,
            )
            gap_fields = gap_fields[:_MAX_ENRICHMENT_FIELDS_PER_RUN]

        fields_targeted = len(gap_fields)

        # Step 4: Per-field processing
        for field_row in gap_fields:
            missing_field_id: int      = field_row["missing_field_id"]
            field_path:       str      = field_row["field_path"]
            site_hint:        str | None = field_row.get("preferred_site_hint")
            template_override: str | None = field_row.get("query_template_override")

            # -----------------------------------------------------------------
            # Chipset deduplication (chipset.npu_tops only)
            # E11 fix: chipset_name already fetched above — pass it directly
            # -----------------------------------------------------------------
            generic_path = re.sub(r"\[\d+\]", "[*]", field_path)  # E7 fix

            if generic_path == "chipset.npu_tops":
                copied = await _try_chipset_dedup(
                    chipset_name=chipset_name_for_run,
                    missing_field_id=missing_field_id,
                    field_path=field_path,
                    enrichment_run_id=enrichment_run_id,
                    brand=brand,
                    model=model,
                )
                if copied:
                    fields_resolved += 1
                    continue  # No API call needed

            # -----------------------------------------------------------------
            # Build prompt and fire grounded call
            # -----------------------------------------------------------------
            prompt = build_enrichment_prompt(brand, model, field_path, site_hint, template_override)

            # INSERT search query row (log before the API call)
            query_id: int = await asyncio.to_thread(
                insert_enrichment_search_query,
                {
                    "enrichment_run_id":   enrichment_run_id,
                    "missing_field_id":    missing_field_id,
                    "query_text":          prompt,
                    "query_template_used": template_override or "default",
                    "grounding_used":      True,
                    "api_cost_inr":        0.00,  # updated after call
                },
            )

            # Fire grounded call
            grounded_result: dict | None = None
            call_error: str | None = None
            http_status: int | None = None

            try:
                grounded_result = await call_gemini_grounded(
                    prompt=prompt,
                    output_schema={
                        "type": "object",
                        "properties": {
                            "value":      {},
                            "confidence": {"type": "number"},
                            "evidence":   {"type": "string"},
                        },
                    },
                    site_hint=site_hint,
                )
                http_status = 200
                cost_accumulator += _GROUNDED_QUERY_COST_INR

            except (GeminiRateLimitError, GeminiTransientError) as exc:
                # Already retried 3× inside call_gemini_grounded — give up for this field
                call_error = f"{type(exc).__name__}: {exc}"
                run_status = "partially_completed"
                logger.warning(
                    "run_enrichment: transient/rate-limit failure field=%r: %s",
                    field_path, call_error,
                )
            except GeminiNonRetryableError as exc:
                call_error = f"NonRetryable: {exc}"
                run_status = "partially_completed"
                logger.error(
                    "run_enrichment: non-retryable failure field=%r: %s",
                    field_path, call_error,
                )
            except Exception as exc:
                call_error = f"Unexpected: {exc}"
                run_status = "partially_completed"
                logger.error(
                    "run_enrichment: unexpected error field=%r: %s",
                    field_path, call_error, exc_info=True,
                )

            # Update the search query row regardless of outcome
            await asyncio.to_thread(
                update_enrichment_search_query,
                query_id,
                {
                    "http_status":   http_status,
                    "api_cost_inr":  _GROUNDED_QUERY_COST_INR if grounded_result else 0.00,
                    "error_message": call_error,
                },
            )

            if grounded_result is None:
                # Mark as attempted-but-failed
                await asyncio.to_thread(
                    update_missing_field_log_attempted, missing_field_id, False
                )
                continue

            # Extract response components
            raw_value     = _clean_extracted_value(grounded_result.get("value"))
            evidence_text = grounded_result.get("evidence") or None
            source_url    = grounded_result.get("source_url")
            source_domain = grounded_result.get("source_domain")

            # E5 fix — safe float conversion; default 0.5 on non-numeric Gemini output
            try:
                raw_confidence = float(grounded_result.get("confidence", 0.5))
            except (TypeError, ValueError):
                raw_confidence = 0.5
                logger.warning(
                    "run_enrichment: non-numeric confidence from Gemini "
                    "at field=%r, defaulting to 0.5", field_path,
                )
            raw_confidence = max(0.00, min(1.00, raw_confidence))

            # Classify source tier and adjust confidence
            source_tier   = _determine_source_tier(source_domain)
            adjusted_conf = adjust_confidence_for_source_url(
                raw_confidence, source_url, brand, model
            )

            # Auto-selection determination
            is_selected = await _should_auto_select(
                source_tier=source_tier,
                confidence=adjusted_conf,
                missing_field_id=missing_field_id,
            )

            # E1 fix — deselect all previous candidates BEFORE inserting this one
            # Prevents two is_selected=TRUE rows for the same field
            if is_selected:
                await asyncio.to_thread(
                    deselect_all_candidates_for_field, missing_field_id
                )

            # Skip candidate insert when Gemini returned null or empty value.
            # enrichment_field_candidates.extracted_value is NOT NULL — inserting
            # null violates the constraint. Empty strings and whitespace are
            # also meaningless. Mark as attempted-failed and move on.
            value_obtained = _is_valid_value(raw_value)
            if not value_obtained:
                await asyncio.to_thread(
                    update_missing_field_log_attempted, missing_field_id, False
                )
                continue

            # INSERT candidate (extracted_value is guaranteed non-null/non-empty here)
            await asyncio.to_thread(
                insert_enrichment_candidate,
                {
                    "query_id":          query_id,
                    "missing_field_id":  missing_field_id,
                    "enrichment_run_id": enrichment_run_id,
                    "field_path":        field_path,
                    "extracted_value":   raw_value,
                    "raw_confidence":    raw_confidence,
                    "confidence":        adjusted_conf,
                    "evidence_text":     evidence_text,
                    "source_url":        source_url,
                    "source_domain":     source_domain,
                    "source_tier":       source_tier,
                    "is_selected":       is_selected,
                },
            )

            # UPDATE missing_fields_log
            await asyncio.to_thread(
                update_missing_field_log_attempted, missing_field_id, True
            )
            fields_resolved += 1

        # E4 fix — total failure is never "success"
        if fields_resolved == 0 and fields_targeted > 0:
            run_status = "failed"

        # Step 6: Final run update
        total_cost = round(cost_accumulator, 2)
        await asyncio.to_thread(
            update_enrichment_run,
            enrichment_run_id,
            {
                "status":              run_status,
                "fields_targeted":     fields_targeted,
                "fields_resolved":     fields_resolved,
                "total_api_cost_inr":  total_cost,
            },
        )

    except Exception as crash_exc:
        run_status = "failed"
        logger.error(
            "run_enrichment: UNHANDLED CRASH normalized_id=%d enrichment_run_id=%d: %s",
            normalized_id, enrichment_run_id, crash_exc, exc_info=True,
        )
        try:
            await asyncio.to_thread(
                update_enrichment_run,
                enrichment_run_id,
                {
                    "status":             "failed",
                    "fields_targeted":    fields_targeted,
                    "fields_resolved":    fields_resolved,
                    "total_api_cost_inr": round(cost_accumulator, 2),
                    "error_message":      f"Unhandled crash: {crash_exc}",
                },
            )
        except Exception:
            logger.error(
                "run_enrichment: could not write failed status to DB for run_id=%d",
                enrichment_run_id,
            )
        raise  # re-raise so FastAPI returns 500

    # These lines are OUTSIDE the try/except — only execute on success/partial
    success = run_status in ("completed", "partially_completed")
    logger.info(
        "run_enrichment: COMPLETE normalized_id=%d run_id=%d "
        "targeted=%d resolved=%d cost=₹%.2f status=%s",
        normalized_id, enrichment_run_id,
        fields_targeted, fields_resolved, cost_accumulator, run_status,
    )

    return {
        "success":            success,
        "enrichment_run_id":  enrichment_run_id,
        "fields_targeted":    fields_targeted,
        "fields_resolved":    fields_resolved,
        "total_api_cost_inr": round(cost_accumulator, 2),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

async def _should_auto_select(
    source_tier: str,
    confidence: float,
    missing_field_id: int,
) -> bool:
    """
    Determines whether this candidate should be auto-selected.

    Rule 1: source_tier='oem_official' AND confidence >= 0.85 → True
    Rule 2: confidence delta vs. best existing candidate >= 0.20 → True
            (The caller deselects the previous best before inserting — E1 fix)
    Rule 3: All else → False

    Bug 1 fix: fetch_best_existing_candidate_confidence is a sync Supabase-py call.
    It is now wrapped in asyncio.to_thread to avoid blocking the event loop on
    every field that reaches Rule 2 (up to 20 calls per enrichment run).
    """
    # Rule 1: oem_official + confidence >= 0.85 → auto-select
    if source_tier == "oem_official" and confidence >= 0.85:
        return True

    # S4-P1-8: chipset_vendor is authoritative for chipset.* fields only.
    # For non-chipset fields (battery, display, software), chipset vendor
    # press-release data is not phone-level evidence — do not auto-select.
    if source_tier == "chipset_vendor" and confidence >= 0.85:
        # field_path is not available in _should_auto_select — but chipset dedup
        # (the primary chipset_vendor path) calls _try_chipset_dedup which inserts
        # with is_selected=True directly. All other chipset_vendor hits go through
        # Rule 2 below, capped by the delta guard. This rule is a safety fallback.
        return False  # chipset_vendor never auto-selects via Rule 1 for safety

    # Rule 2 — wrapped in to_thread so the sync DB call does not block the event loop
    try:
        best_existing = await asyncio.to_thread(
            fetch_best_existing_candidate_confidence, missing_field_id
        )
        if best_existing is not None and (confidence - best_existing) >= 0.20:
            return True
    except Exception:
        pass  # Best-effort — never block the insert on a delta-check failure

    return False


async def _try_chipset_dedup(
    chipset_name: str | None,
    missing_field_id: int,
    field_path: str,
    enrichment_run_id: int,
    brand: str,
    model: str,
) -> bool:
    """
    Chipset deduplication for chipset.npu_tops.

    E11 fix: chipset_name is passed in directly from the single normalized_spec
    fetch at run_enrichment start — no second DB round-trip here.

    Checks if another phone sharing the same chipset_name already has a
    confirmed (is_selected=TRUE) enrichment_field_candidates row for this
    field. If yes, copies that candidate for this phone → saves one API call.

    Returns True if a copy was made (caller should skip the API call).
    Returns False if deduplication did not apply (proceed normally).
    """
    if not chipset_name:
        return False

    # Query for existing confirmed candidate for this chipset + field
    existing: dict | None = await asyncio.to_thread(
        fetch_chipset_enrichment_copy, chipset_name, field_path
    )
    if not existing:
        return False

    # Re-adjust confidence for THIS phone's brand/model slugs
    adjusted_conf = adjust_confidence_for_source_url(
        float(existing["raw_confidence"]),
        existing.get("source_url"),
        brand,
        model,
    )

    # Deselect any existing candidates for this field before copying — E1 fix
    await asyncio.to_thread(deselect_all_candidates_for_field, missing_field_id)

    # Copy the existing candidate into a new row for this phone
    await asyncio.to_thread(
        insert_enrichment_candidate,
        {
            "query_id":          None,              # no query row for a copy
            "missing_field_id":  missing_field_id,
            "enrichment_run_id": enrichment_run_id,
            "field_path":        field_path,
            "extracted_value":   existing["extracted_value"],
            "raw_confidence":    float(existing["raw_confidence"]),
            "confidence":        adjusted_conf,
            "evidence_text":     existing.get("evidence_text"),
            "source_url":        existing.get("source_url"),
            "source_domain":     existing.get("source_domain"),
            "source_tier":       existing.get("source_tier", "unknown"),
            "is_selected":       True,
        },
    )
    await asyncio.to_thread(update_missing_field_log_attempted, missing_field_id, True)

    logger.info(
        "_try_chipset_dedup: copied npu_tops candidate for chipset=%r "
        "missing_field_id=%d — API call skipped",
        chipset_name, missing_field_id,
    )
    return True
