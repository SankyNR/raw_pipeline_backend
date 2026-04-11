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
                lines.append(f"    EXCEPTION — aggregator beats transcript for:")
                for f in fields:
                    lines.append(f"      - {f}")
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
            "• body.length/breadth/height = UNFOLDED (open) state.\n"
            "• body.length_folded/breadth_folded/height_folded = FOLDED (closed) state.\n"
            "• displays: MUST have exactly 2 entries.\n"
            "  - Cover screen: display_type='Cover', display_position='Primary'\n"
            "  - Inner screen: display_type='Inner', display_position='Secondary'\n"
            "• NEVER average specs across the two screens.\n"
        )
    if phone_type == PHONE_TYPE_FLIPPABLE:
        return (
            "=== PHONE TYPE: FLIPPABLE (CLAMSHELL) ===\n"
            "This is a clamshell foldable (e.g. Galaxy Z Flip, Motorola Razr). Rules:\n"
            "• body.length/breadth/height = UNFOLDED (open) state.\n"
            "• body.length_folded/breadth_folded/height_folded = FOLDED (closed) state.\n"
            "• displays: MUST have exactly 2 entries.\n"
            "  - Inner screen: display_type='Inner', display_position='Primary'\n"
            "  - Cover screen: display_type='Cover', display_position='Secondary'\n"
            "• NEVER average specs across the two screens.\n"
        )
    # standard
    return (
        "=== PHONE TYPE: STANDARD ===\n"
        "This is a standard (non-foldable) phone. Rules:\n"
        "• body.length_folded / breadth_folded / height_folded = always null.\n"
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

1. DERIVED FIELDS — NEVER EXTRACT
   Run C computes these. Extracting them causes pipeline conflicts:
   • displays[*].ppi                       → computed: sqrt(h²+w²) / size_inch
   • camera_lenses[*].sensor_size_decimal  → computed: 1.0 / sensor_size_denominator

2. LOOKUP MATCHING
   • Always output exact strings from the allowed values (see FIELD SCHEMA below).
   • Case-sensitive: "LPDDR5X" not "lpddr5x", "Li-Po" not "Li-po".
   • If no match: output null. Never invent a new string value.

3. SENSOR SIZE
   • sensor_size_denominator = the number after "1/" in "1/X inch"
     e.g. "1/1.56 inch" → sensor_size_denominator: 1.56

4. VARIANT INTEGRITY
   • ram_capacity = PHYSICAL RAM only. Never sum physical + virtual RAM.
   • is_base_variant = true for exactly ONE variant per phone.

5. APERTURE
   • Lower aperture = BETTER (wider aperture, more light).
   • "f/1.7" → aperture: 1.7. Store the numeric value only.

6. CHARGER IN BOX — INDIA CRITICAL
   • Must reflect the INDIA RETAIL UNIT specifically.
   • International "no charger" news ≠ India unit policy.
   • If not explicitly stated in sources: output null.

7. NAVIC — INDIA PRIORITY
   • Always check all sources explicitly for NavIC support.
   • Absence from aggregators ≠ absence from phone.

8. SAR VALUES — INDIA STANDARD
   • India uses 0mm separation (tec.fptc.gov.in).
   • GSMArena SAR values use EU 10mm separation — NOT India values.

9. USB SPEED VS CONNECTOR
   • Type-C connector ≠ USB 3.x speed. Most budget phones use USB 2.0 over Type-C.
   • Verify usb_standard from the explicit data transfer rate in specs.

10. BOOLEAN INFERENCE FORBIDDEN
    • Do NOT set pd_support from "USB Type-C" alone.
    • Do NOT set vo5g from "5G phone" alone — requires explicit VoNR/Vo5G statement.
    • Only set booleans when the source explicitly confirms the feature.

11. NULL VS OMIT
    • Use explicit null for missing scalar fields.
    • Use [] for missing array/junction fields.
    • NEVER omit a key from the JSON output.

12. ARRAY ORDERING (STRICT)
    • camera_lenses MUST be ordered: Main → Ultra-wide → Telephoto → Macro → Depth
      The index position in the output array determines the FK mapping in the DB.
      Wrong order = wrong lens type committed. Verify against OEM specs.
    • variants MUST be ordered: ascending by RAM capacity first, then storage.
      Example: 6GB/128GB → 8GB/128GB → 8GB/256GB → 12GB/256GB.
      is_base_variant=true MUST be set on exactly the first (lowest) variant.
    • displays MUST follow the ordering in Rule [PHONE TYPE] above.

13. PER-FIELD EVIDENCE (REQUIRED)
    For EVERY non-null field you extract, wrap the value with evidence:
    {
      "value":       <the extracted value>,
      "evidence":    "exact sentence from source",
      "source_type": "scraped|transcript",
      "source_name": "<site_name from the --- SOURCE: <site_name> --- header>"
    }
    Fields with null values output: null (no evidence wrapper needed).
    Use source_type="scraped" for OEM/aggregator sources, "transcript" for YouTube.
    source_name MUST match the header of the section you drew the value from,
    e.g. "gsmarena", "samsung_official", "smartprix", or "transcript".

14. STORAGE AND RAM UNITS — ALWAYS OUTPUT IN GB (MANDATORY)
    The normaliser strips numeric suffixes AFTER you output them. If you output
    "1TB", it becomes integer 1. If you output "1024GB", it becomes integer 1024.
    1024 is correct. 1 is wrong. The pipeline has no unit conversion — YOU must convert.

    storage_capacity: ALWAYS in GB. Convert TB → GB if needed.
      Examples: 128GB → 128   |   256GB → 256   |   1TB → 1024   |   512GB → 512
    ram_capacity: ALWAYS in GB. Modern phones are always GB — no conversion needed.
      Examples: 6GB → 6   |   8GB → 8   |   12GB → 12
      If a source says MB (extremely rare): convert. 4096MB → 4.

    NEVER output raw TB values. NEVER omit the GB suffix (the normaliser needs it
    to confirm the number is storage, not a ppi or Hz value).
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
