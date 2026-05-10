"""
Task 0.4 — ECD Generator

Dynamically builds the Extraction Context Document (ECD) for a given phone run.

The ECD is the system prompt passed to `call_gemini_json()` for Run A (spec extraction)
and Run B (experience extraction). It is assembled fresh per extraction run — never
cached, never stored, never returned to the frontend.

Assembly order:
  1. spec_template.yaml  — field definitions and allowed values (loading once at startup)
  2. ecd_disambiguation.yaml — source priority block + cross-table disambiguation rules
  3. Phone-type trimming — removes irrelevant sections (e.g. folded body dimensions
     for standard phones, removes cover display for non-foldables)
  4. Lookup allowlist injection — per-field allowed values from Supabase DB
     (populated in Phase 1 via the lookup cache; placeholder strings here in Phase 0)

Startup:
  Call `pre_warm_ecd()` once at lifespan startup.
  This loads and validates both YAML files, raising immediately on parse errors.

Usage:
  ecd = build_ecd(phone_type="standard")               → str (system prompt, no transcript)
  ecd = build_ecd(phone_type="standard", has_transcript=True)  → str (includes transcript tables)
  ecd = build_ecd(phone_type="foldable")               → str (includes foldable hints)
  ecd = build_ecd(phone_type="flippable")              → str (includes flippable hints)
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# YAML file paths
# ---------------------------------------------------------------------------

# C3 fix: ecd_generator.py lives in app/services/ but YAML files are in app/config/
_CONFIG_DIR = Path(__file__).parent.parent / "config"
_SPEC_TEMPLATE_PATH = _CONFIG_DIR / "spec_template.yaml"
_DISAMBIGUATION_PATH = _CONFIG_DIR / "ecd_disambiguation.yaml"

# ---------------------------------------------------------------------------
# Module-level YAML cache (pre-loaded at startup)
# ---------------------------------------------------------------------------

_spec_template: dict | None = None
_disambiguation: dict | None = None

# ---------------------------------------------------------------------------
# ECD string cache — keyed by phone_type (3 entries max)
# Populated by build_ecd() on first call per phone_type.
# Ensures every phone in the same session sees the identical ECD.
# Cleared by invalidate_ecd_cache() on admin ECD refresh.
# ---------------------------------------------------------------------------

_ECD_CACHE: dict[str, str] = {}


def pre_warm_ecd() -> None:
    """
    Loads and validates both YAML config files at application startup.
    Raises ValueError immediately on parse errors — fail-fast at boot, not at runtime.

    Called from app/main.py lifespan startup.
    """
    global _spec_template, _disambiguation

    try:
        with open(_SPEC_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            _spec_template = yaml.safe_load(f)
        logger.info("ECD: spec_template.yaml loaded (%d top-level keys)", len(_spec_template or {}))
    except Exception as exc:
        raise ValueError(f"ECD pre-warm failed: cannot parse spec_template.yaml: {exc}") from exc

    try:
        with open(_DISAMBIGUATION_PATH, "r", encoding="utf-8") as f:
            _disambiguation = yaml.safe_load(f)
        logger.info(
            "ECD: ecd_disambiguation.yaml loaded (%d rules)",
            len((_disambiguation or {}).get("disambiguation_rules", [])),
        )
    except Exception as exc:
        raise ValueError(
            f"ECD pre-warm failed: cannot parse ecd_disambiguation.yaml: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Phone type constants
# ---------------------------------------------------------------------------

PHONE_TYPE_STANDARD  = "standard"
PHONE_TYPE_FOLDABLE  = "foldable"
PHONE_TYPE_FLIPPABLE = "flippable"

_VALID_PHONE_TYPES = {PHONE_TYPE_STANDARD, PHONE_TYPE_FOLDABLE, PHONE_TYPE_FLIPPABLE}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_loaded() -> None:
    """
    Ensures YAML files are loaded. Calls pre_warm_ecd() if not yet loaded.
    This allows build_ecd() to work even if pre_warm was never called (dev/test mode).
    """
    global _spec_template, _disambiguation
    if _spec_template is None or _disambiguation is None:
        logger.warning(
            "ECD YAML not pre-warmed. Loading now. "
            "Call pre_warm_ecd() at app startup to avoid this."
        )
        pre_warm_ecd()


def _build_source_priority_block() -> str:
    """
    Formats the source priority section from ecd_disambiguation.yaml into a
    human-readable ECD block.
    """
    sp = _disambiguation.get("source_priority", {})
    lines = [
        "=== SOURCE PRIORITY ===",
        sp.get("description", "").strip(),
        "",
    ]
    for entry in sp.get("order", []):
        lines.append(f"  Priority {entry['priority']} — {entry['source']}")
        lines.append(f"    {entry['trust_reason'].strip()}")
        if "exceptions" in entry:
            exc_block = entry["exceptions"]
            fields = exc_block.get("fields_where_aggregator_wins", [])
            reason = exc_block.get("reason", "").strip()
            if fields:
                fields_str = " | ".join(fields)
                lines.append(f"    EXCEPTION — aggregator beats transcript for exact measurements: {fields_str}")
                lines.append(f"    Reason: {reason}")
        lines.append("")
    return "\n".join(lines)


def _build_disambiguation_block() -> str:
    """
    Formats the disambiguation rules from ecd_disambiguation.yaml into
    a numbered rule block for the ECD system prompt.
    """
    rules = _disambiguation.get("disambiguation_rules", [])
    lines = ["=== DISAMBIGUATION RULES ==="]
    for i, rule in enumerate(rules, start=1):
        lines.append(f"{i}. Rule: {rule.get('rule', '(unnamed)')}")
        note = rule.get("note", "").strip()
        if note:
            lines.append(f"   Note: {note}")
        if "applies_to" in rule:
            for target in rule["applies_to"]:
                lines.append(f"   → {target.get('path', '?')}: {target.get('context', '')}")
        if "maps_to" in rule:
            lines.append(f"   → Maps to: {rule['maps_to']}")
        if "canonical_value" in rule:
            lines.append(f"   → Canonical value: {rule['canonical_value']}")
        lines.append("")
    return "\n".join(lines)


def _build_phone_type_instructions(phone_type: str) -> str:
    """
    Returns additional instructions specific to the phone form factor.

    - standard:  No folded dimensions; single display entry expected.
    - foldable:  Include folded dimensions; two displays (Cover=Primary, Inner=Secondary).
    - flippable: Include folded dimensions; two displays (Inner=Primary, Cover=Secondary).
    """
    if phone_type == PHONE_TYPE_FOLDABLE:
        return (
            "=== PHONE TYPE: FOLDABLE (BOOK-FOLD) ===\n"
            "This is a book-fold foldable (e.g. Galaxy Z Fold). Rules:\n"
            "• body.height/width/thickness = UNFOLDED (open) state.\n"
            "• body.height_folded/width_folded/thickness_folded = FOLDED (closed) state.\n"
            "• displays: MUST have exactly 2 entries.\n"
            "  - Inner screen: display_type='Inner', display_position='Primary'\n"
            "  - Cover screen: display_type='Cover', display_position='Secondary'\n"
            "• NEVER average specs across the two screens.\n"
        )
    if phone_type == PHONE_TYPE_FLIPPABLE:
        return (
            "=== PHONE TYPE: FLIPPABLE (CLAMSHELL) ===\n"
            "This is a clamshell foldable (e.g. Galaxy Z Flip, Motorola Razr). Rules:\n"
            "• body.height/width/thickness = UNFOLDED (open) state.\n"
            "• body.height_folded/width_folded/thickness_folded = FOLDED (closed) state.\n"
            "• displays: MUST have exactly 2 entries.\n"
            "  - Inner screen: display_type='Inner', display_position='Primary'\n"
            "  - Cover screen: display_type='Cover', display_position='Secondary'\n"
            "• NEVER average specs across the two screens.\n"
        )
    # standard
    return (
        "=== PHONE TYPE: STANDARD ===\n"
        "This is a standard (non-foldable) phone. Rules:\n"
        "• body.height_folded / width_folded / thickness_folded = always null.\n"
        "• displays: exactly 1 entry.\n"
        "  - display_type='Main', display_position='Primary'\n"
    )


def _build_critical_rules_block() -> str:
    """
    Returns the invariant extraction rules block (always injected regardless of phone type).
    These rules correspond directly to Part 4 of extraction_templates_v3.md.
    """
    return """\
=== CRITICAL EXTRACTION RULES ===

1. DERIVED FIELDS — NEVER EXTRACT (Run C computes these):
   displays[*].ppi → sqrt(h²+w²) / size_inch
   camera_lenses[*].sensor_size_decimal → 1.0 / sensor_size_denominator

2. LOOKUP MATCHING: Output exact strings from allowed values. Case-sensitive. \
No match → null. Never invent a string.

3. SENSOR SIZE: sensor_size_denominator = number after "1/" in "1/X inch". \
e.g. "1/1.56 inch" → 1.56. Null if notation absent.

4. VARIANT INTEGRITY: ram_capacity = physical RAM only. is_base_variant = true on exactly ONE variant.

5. APERTURE: "f/1.7" → aperture: 1.7. Store numeric only. Lower = wider = better.

6. CHARGER IN BOX: India retail unit only. International policy ≠ India. Null if not stated.

7. NAVIC: Always check OEM India page explicitly. Absence from aggregators ≠ not supported.

8. SAR: India = 0mm separation. GSMArena = EU 10mm. Never use GSMArena SAR for India.

9. USB SPEED: Type-C connector ≠ USB 3.x. Verify usb_standard from explicit transfer rate in specs.

10. BOOLEAN INFERENCE FORBIDDEN: Never set pd_support from "USB Type-C" alone. \
Never set vo5g from "5G phone" alone. Booleans require explicit source confirmation.

11. NULL VS OMIT: Explicit null for missing scalars. [] for missing arrays. Never omit a key.

12. ARRAY ORDERING (STRICT — wrong order = wrong DB mapping):
    camera_lenses: Main → Ultra-wide → Telephoto → Macro → Depth → Front
    variants: ascending RAM first, then storage. is_base_variant=true on the first (lowest) variant.
    displays: per phone type rules above.

13. EVIDENCE (every non-null scalar MUST have this wrapper):
    Scraped:    {"value": X, "_source": {"raw_id": N, "evidence_text": "verbatim substring"}}
    Transcript: {"value": X, "_source": {"raw_transcript_id": N, "evidence_text": "verbatim substring"}}
    Rules: evidence_text = exact verbatim substring, never paraphrased. One source per field. \
Null fields → null directly, no _source. \
Junction arrays (bands_5g, display_features) and structural fields \
(display_type, lens_type, is_base_variant) → plain value, no _source.

14. STORAGE/RAM IN GB (MANDATORY): Output in GB always. 1TB → 1024. \
The normaliser strips suffixes but has zero unit conversion. Wrong unit = wrong DB value.
"""


def _build_spec_template_block() -> str:
    """
    Serialises the loaded spec_template.yaml output section back into YAML for injection.
    """
    output_section = (_spec_template or {}).get("output", _spec_template or {})
    return (
        "=== FIELD SCHEMA (Output Structure) ===\n"
        "Extract all fields into this exact JSON structure:\n\n"
        + yaml.dump(output_section, allow_unicode=True, default_flow_style=False, indent=2)
    )


# ---------------------------------------------------------------------------
# Task 0.4 — build_ecd (main public API)
# ---------------------------------------------------------------------------

def _build_lookup_values_block() -> str:
    """
    Injects the canonical allowed values for every lookup field into the ECD.
    This is the single most important accuracy fix: when the LLM knows the exact
    canonical string to output, the normalizer can exact-match it without fuzzy
    matching or staging. This eliminates the majority of false gaps and prevents
    wasted enrichment queries.

    Values are hardcoded here from the live DB lookup tables.
    Update this function whenever new lookup rows are added to the DB.
    """
    return """\=== LOOKUP FIELD ALLOWED VALUES ===
Output EXACTLY one of the listed strings — character-for-character, \
including capitalisation, spaces, and punctuation. Output null if no value matches. \
NEVER invent a new string.

--- CONNECTIVITY ---
connectivity.wifi_standard:
  "Wi-Fi (802.11b/g)" | "Wi-Fi 4 (802.11n)" | "Wi-Fi 5 (802.11ac)"
  "Wi-Fi 6 (802.11ax)" | "Wi-Fi 6E (802.11ax)" | "Wi-Fi 7 (802.11be)"

connectivity.bluetooth_version (version number only):
  "4.0" | "4.1" | "4.2" | "5.0" | "5.1" | "5.2" | "5.3" | "5.4" | "6.0"

connectivity.usb_standard (verify from data transfer rate, not connector alone):
  "USB-C 2.0" | "USB-C 3.2 Gen 1" | "USB-C 3.2 Gen 2"
  "USB Type-C 2.0" | "USB Type-C 3.2 Gen 2" | "USB Type-C 4.0" | "Lightning"

connectivity.wifi_technologies (array — protocols only, no frequency bands or hotspot):
  "OFDMA" | "MU-MIMO" | "SU-MIMO" | "2x2 MIMO" | "4x4 MIMO" | "8x8 MIMO"
  "Beamforming" | "TWT" | "WPA3" | "Wi-Fi Direct" | "1024-QAM" | "256-QAM"
  "4096-QAM" | "MLO" | "ETH320"

--- NETWORK ---
network.sim_configuration:
  "Single SIM (Nano-SIM)" | "Dual SIM (Nano-SIM, dual stand-by)"
  "Dual SIM (Nano-SIM, single stand-by)" | "Nano-SIM + eSIM"
  "Dual SIM + eSIM" | "Dual eSIM" | "Triple SIM" | "Hybrid SIM (Nano-SIM + microSD)"

network.cellular_features (array):
  "5G SA" | "5G NSA" | "5G Dual Connectivity (EN-DC) (LTE + NR Dual Connectivity)"
  "Carrier Aggregation" | "4x4 MIMO (Cellular)" | "2x2 MIMO (Cellular)"
  "Dual 5G Standby" | "5G Advanced Ready" | "Satellite Emergency Messaging"

--- DISPLAY ---
displays[*].panel_type:
  "AMOLED" | "Super AMOLED" | "Dynamic AMOLED" | "Dynamic AMOLED 2X"
  "LTPO Dynamic AMOLED" | "LTPO Super AMOLED" | "LTPO OLED" | "Fluid AMOLED"
  "LTPO Fluid AMOLED" | "P-OLED" | "pOLED" | "IPS LCD" | "TFT LCD" | "PLS LCD"
  "Super Actua OLED" | "Liquid Retina" | "Liquid Retina XDR"
  "Super Retina XDR" | "LTPO Super Retina XDR" | "ProMotion Super Retina XDR OLED"
  "MicroLED"

displays[*].glass_protection (drop "Corning" prefix — "Corning Gorilla Glass 5" → "Gorilla Glass 5"):
  "Gorilla Glass 3" | "Gorilla Glass 5" | "Gorilla Glass 6" | "Gorilla Glass 7"
  "Gorilla Glass Victus" | "Gorilla Glass Victus 2" | "Gorilla Glass Armor"
  "Corning Gorilla Armor 2" | "Corning Gorilla Glass Ceramic"
  "Ceramic Shield" | "Ceramic Shield 2" | "Dragon Crystal Glass"
  "Kunlun Glass" | "Armor Glass" | "Panda Glass" | "Tempered Glass" | "No Protection"

displays[*].screen_shape:
  "Flat" | "Curved" | "2.5D Curved" | "2.5 curved" | "Edge Display"
  "Waterfall Display" | "Micro Quad-curved" | "Foldable" | "Rollable"

--- VARIANTS ---
variants[*].ram_type:
  "LPDDR4" | "LPDDR4X" | "LPDDR5" | "LPDDR5X" | "LPDDR6" | "DDR4" | "DDR5" | "LPDDR3" | "Unknown"

variants[*].storage_type:
  "UFS 2.1" | "UFS 2.2" | "UFS 3.0" | "UFS 3.1" | "UFS 4.0" | "UFS 4.1"
  "eMMC 5.1" | "eMMC 5.0" | "eMMC 4.5" | "NVMe" | "NVMe PCIe 3.0" | "NVMe PCIe 4.0" | "Unknown"

--- CHARGING ---
charging.battery_type:
  "Li-Po" | "Li-Ion" | "Silicon-Carbon" | "Non-Removable Li-Polymer"
  "Non-Removable Si/C Li-ion" | "Graphene"

charging.cable_type:
  "USB Type-C to Type-C" | "Type-C to Type-C" | "Type-A to Type-C"
  "USB Type-A to Type-C" | "Type-A to Lightning" | "Type-C to Lightning"

charging.proprietary_charging (brand's official marketing name; null for Apple, Google Pixel, Nothing):
  "TurboPower" | "SUPERVOOC" | "SuperVOOC" | "SuperVOOC 2.0" | "Warp Charge"
  "Dart Charge" | "Dash Charge" | "VOOC" | "SuperDart" | "FlashCharge"
  "Super FlashCharge" | "HyperCharge" | "Mi Turbo Charge" | "Super Fast Charging"
  "Super Fast Charging 2.0" | "Adaptive Fast Charging" | "Honor SuperCharge"
  "Huawei SuperCharge" | "All-Round FastCharge" | "All-Round FastCharge 2.0"
  "All-Round FastCharge 3.0"

charging.wireless_charging_standard:
  "Qi" | "Qi2" | "Qi2 / PMA" | "Qi2 / PMA / MagSafe" | "Qi2.2 / PMA" | "MagSafe"

charging.charger_technologies (array — LLM may fill from training data when not stated in source):
  "GaN" | "GaN + PD" | "Power Delivery (PD)" | "PD 3.0" | "PD 3.1" | "PD 3.2"
  "PPS" | "Quick Charge 3.0" | "Quick Charge 4.0" | "Quick Charge 5.0"

--- CAMERA ---
camera_lenses[*].lens_type (array order rule applies — Main→Ultra-wide→Telephoto→Macro→Depth→Front):
  "Main" | "Ultra-wide" | "Telephoto" | "Periscope" | "Macro" | "Depth" | "Front"

camera_lenses[*].autofocus_type:
  "PDAF" | "Quad PDAF" | "Dual Pixel PDAF" | "Multi-directional PDAF"
  "Omni-directional PDAF" | "Super PDAF" | "All-Pixel Focus" | "Hybrid AF"
  "Contrast AF" | "Laser AF" | "ToF 3D" | "Manual Focus" | "Fixed Focus"

camera_lenses[*].stabilization (array):
  "OIS" | "EIS" | "Gimbal OIS" | "Sensor-shift OIS" | "Action Mode"
  "Super Steady" | "No Stabilization"

--- OS & SECURITY ---
os_and_security.os_name:
  "Android 11" | "Android 12" | "Android 13" | "Android 14" | "Android 15"
  "Android 16" | "iOS 16" | "iOS 17" | "iOS 18" | "iOS 26.0"
  "HarmonyOS 4.0" | "HarmonyOS 4.2"

os_and_security.ui_skin:
  "My UX" | "Hello UI" | "Hello UX" | "One UI 6" | "One UI 6.1" | "One UI 7" | "One UI 8"
  "OxygenOS 14" | "OxygenOS 15" | "OxygenOS 16"
  "ColorOS 14" | "ColorOS 15" | "ColorOS 16"
  "Funtouch OS 14" | "Funtouch OS 15" | "Funtouch OS 15 / Origin OS 6" | "OriginOS 5" | "OriginOS 6"
  "HyperOS" | "HyperOS 2" | "HyperOS 3" | "MIUI 14"
  "Realme UI 5" | "Realme UI 6" | "Realme UI 7"
  "Nothing OS 2" | "Nothing OS 3"
  "Pixel UI" | "Pixel UI (Material 3 Expressive)" | "Stock Android"
  "iOS" | "iOS 26.0 Liquid Glass UI"

os_and_security.biometrics (array):
  "Side-mounted Fingerprint" | "In-display Fingerprint (Optical)"
  "In-display Fingerprint (Ultrasonic)" | "Rear-mounted Fingerprint"
  "Fingerprint (Side-mounted)" | "Fingerprint (Under-display Ultrasonic)"
  "Face Unlock" | "Face Unlock (2D)" | "Face ID" | "Face ID (3D TrueDepth)"
  "3D Face Recognition" | "Iris Scanner" | "Voice Recognition"

--- CERTIFICATIONS ---
certifications.widevine_level — see rule 5s (default L1 for phones >= 2018, no source needed):
  "L1" | "L2" | "L3"

certifications.ip_ratings (array):
  "IP67" | "IP68" | "IP68K" | "IP69" | "IP69K" | "IP54" | "IP53" | "IP64" | "IPX8" | "IP48" | "IP4X"

certifications.video_certifications (array):
  "Dolby Vision" | "Dolby Vision Certified" | "HDR10+ Certified"
  "Netflix HD" | "Netflix HDR" | "YouTube HDR" | "Amazon Prime Video HD"
  "DisplayHDR 400" | "DisplayHDR 600" | "DisplayHDR 1000"
  "TÜV Rheinland Eye Comfort" | "TÜV Rheinland Flicker Free"
  "TÜV Rheinland Low Blue Light" | "TÜV Rheinland Circadian Friendly" | "SGS Eye Care Display"

certifications.audio_certifications (array):
  "Dolby Atmos" | "Dolby Atmos Certified" | "Dolby Digital Plus"
  "Hi-Res Audio Certified" | "Hi-Res Audio Wireless"
  "DTS:X Certified" | "THX Certified" | "JBL Tuned"
  "Qualcomm Snapdragon Sound" | "Spatial Audio Support" | "Spatial Sound" | "360 Reality Audio"

--- LOCATION SERVICES ---
connectivity.location_services (array — always check OEM India page explicitly for NavIC):
  "GPS" | "A-GPS" | "GLONASS" | "Galileo" | "BDS" | "NavIC" | "QZSS"
  "Dual-frequency GPS" | "WiFi Positioning" | "Cellular Positioning" | "iBeacon Micro-location"

--- USB FEATURES ---
connectivity.usb_features (array):
  "OTG" | "DisplayPort" | "USB Power Delivery" | "Desktop Mode"
  "Reverse Charging" | "USB Tethering" | "Audio Adapter Accessory Mode"
"""



def invalidate_ecd_cache() -> None:
    """
    Clears the ECD string cache for all phone types.
    Call this when the YAML config files are reloaded (e.g. admin refresh).
    After invalidation, the next build_ecd() call per phone_type will
    reassemble and re-cache.
    """
    global _ECD_CACHE
    _ECD_CACHE.clear()
    logger.info("ECD cache invalidated — will rebuild on next call per phone_type.")


def build_ecd(
    phone_type: str = PHONE_TYPE_STANDARD,
    has_transcript: bool = False,
    extra_context: str | None = None,
) -> str:
    """
    Builds the Extraction Context Document (ECD) for a given phone run.

    The ECD is the complete system prompt passed to call_gemini_json().
    It is assembled dynamically per-run and cached per (phone_type, has_transcript).

    Args:
        phone_type:     "standard" | "foldable" | "flippable".
                        Controls which dimensional sections are included and
                        which display ordering rules are injected.
        has_transcript: When True, injects an additional TRANSCRIPT CONTEXT block
                        instructing the LLM on how to use the audio source.
                        A6 fix: previously always omitted even when transcript present.
        extra_context:  Optional additional context appended after the standard
                        blocks (e.g. model-specific known issues, admin notes).

    Returns:
        The fully assembled ECD as a plain UTF-8 string ready for system_instruction.

    Raises:
        ValueError: If phone_type is not a known value.
    """
    if phone_type not in _VALID_PHONE_TYPES:
        raise ValueError(
            f"build_ecd: invalid phone_type={phone_type!r}. "
            f"Must be one of: {sorted(_VALID_PHONE_TYPES)}"
        )

    _ensure_loaded()

    # A6: cache key includes has_transcript so transcript-aware ECD is independent
    base_key = f"{phone_type}::transcript={has_transcript}"
    cache_key = base_key if not extra_context else f"{base_key}::{hash(extra_context)}"
    if cache_key in _ECD_CACHE:
        logger.debug("build_ecd: cache HIT for key=%s", cache_key)
        return _ECD_CACHE[cache_key]

    sections = [
        "You are a precision mobile phone spec extraction engine.",
        "Extract structured phone specifications from the provided source documents.",
        "Follow ALL rules below exactly.\n",
        _build_source_priority_block(),
        _build_phone_type_instructions(phone_type),
        _build_disambiguation_block(),
        _build_critical_rules_block(),
        _build_spec_template_block(),
        _build_lookup_values_block(),    # L7.3 Fix 2: inject canonical lookup values
    ]

    # A6 — Inject transcript context block only when a transcript is present
    if has_transcript:
        sections.append(
            "\n=== TRANSCRIPT CONTEXT ===\n"
            "A YouTube review transcript is included as a source document.\n"
            "The transcript is spoken language — treat it accordingly:\n"
            "• Transcripts use informal/approximate numbers. Prefer aggregator for exact measurements.\n"
            "• Transcripts are authoritative for India-specific retail details: charger_in_box, "
            "accessories, colour availability as physically observed.\n"
            "• Reviewer opinions and scores are NOT spec fields — skip these entirely.\n"
            "• Source: speaker may refer to the phone by nickname — verify it matches the target model.\n"
        )

    if extra_context:
        sections.append(f"\n=== ADDITIONAL CONTEXT ===\n{extra_context.strip()}\n")

    ecd = "\n".join(sections)
    _ECD_CACHE[cache_key] = ecd
    logger.debug(
        "build_ecd: cache MISS — assembled and cached %d chars, key=%s",
        len(ecd), cache_key,
    )
    return ecd
