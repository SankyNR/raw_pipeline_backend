"""
Phase 5 — extraction_run_a.py

Run A orchestrator (v5 — custom Gemini JSON extraction).

Replaces langextract_run_a.py. Uses call_gemini_json() via gemini_client.py
instead of lx.extract(). One Gemini call per extraction.

DESIGN
------
Non-negotiable contracts preserved:
  - Same gate check  (fetch_latest_validation → can_proceed)
  - Same DB insert order: run record → output record → update run
  - Same return dict shape: {success, output_id, run_id, message, failed_source_ids}
  - Same failed_source_ids tracking
  - Same field counting (null_count, filled_count)

Source assembly:
  Identical to langextract_run_a.assemble_run_a_input(), updated to accept
  raw_transcript_ids (list) instead of a single raw_transcript_id.
  Up to 3 transcript sections are appended in order (top-3 by reliability).

Evidence attribution:
  The assembled_source_string is passed directly to build_spec_json() for
  two-layer evidence: LLM-provided _source tags → char_start/char_end via str.find().

Timeouts:
  Run A: 180s (was 300s for LangExtract AFC chains).

asyncio safety:
  call_gemini_json() is async. DB calls (sync) are wrapped in asyncio.to_thread().
  The FastAPI event loop is never blocked.
"""

import asyncio
import functools
import logging

from app.services.ecd_generator import (
    PHONE_TYPE_FOLDABLE,
    PHONE_TYPE_FLIPPABLE,
    PHONE_TYPE_STANDARD,
    build_ecd,
)
from app.services.gemini_client import EXTRACTION_MODEL, call_gemini_json
from app.services.spec_json_builder import build_spec_json
from app.services.storage_service import fetch_file_content
from app.utils.path_builder import get_concat_order
from app.core.constants import GATE_ERROR_PREFIX as _GATE_ERROR_PREFIX, PipelineStage
from app.core.spec_canonicalizer import canonicalize_spec
from app.config.extraction_schema_run_a import RunAExtractionSchema, RunAExtractionSchemaSimple
from app.repositories.extraction_repository import (
    fetch_latest_validation,
    fetch_raw_source_rows,
    fetch_transcript_row,
    insert_extraction_run,
    insert_extraction_output,
    update_extraction_run,
)
from app.repositories.pipeline_run_repository import update_pipeline_run

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Global run tracking helper
# ---------------------------------------------------------------------------

async def _track(pipeline_run_id: str | None, **kwargs) -> None:
    """
    Fire-and-forget pipeline_runs status update.

    Spawned as a background task so it never blocks the extraction hot path.
    Swallows all exceptions — a tracking failure must never crash Run A.
    pipeline_run_id is the UUID from pipeline.pipeline_runs, NOT the integer
    extraction run_id from spec_extraction_runs.
    """
    if pipeline_run_id is None:
        return
    async def _do():
        try:
            await asyncio.to_thread(update_pipeline_run, pipeline_run_id, **kwargs)
        except Exception as exc:
            logger.debug("_track: update_pipeline_run failed silently: %s", exc)
    asyncio.create_task(_do())


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

_RUN_A_TIMEOUT_SECONDS: float = 180.0


# ---------------------------------------------------------------------------
# Phone type detection (model-name keyword based)
# ---------------------------------------------------------------------------

_FOLDABLE_KEYWORDS: frozenset[str] = frozenset({"fold", "find n"})
_FLIPPABLE_KEYWORDS: frozenset[str] = frozenset({"flip", "razr"})


def _detect_phone_type(brand: str, model_name: str) -> str:
    """
    Detects the phone form-factor from model_name keywords.

    Detection order:
      1. Flippable keywords: "flip", "razr"    → PHONE_TYPE_FLIPPABLE
      2. Foldable keywords:  "fold", "find n"  → PHONE_TYPE_FOLDABLE
      3. Default:                               → PHONE_TYPE_STANDARD
    """
    m = model_name.lower()
    if any(kw in m for kw in _FLIPPABLE_KEYWORDS):
        return PHONE_TYPE_FLIPPABLE
    if any(kw in m for kw in _FOLDABLE_KEYWORDS):
        return PHONE_TYPE_FOLDABLE
    return PHONE_TYPE_STANDARD


# ---------------------------------------------------------------------------
# Field counter
# ---------------------------------------------------------------------------

def _count_fields(obj, _path: str = "") -> tuple[int, int]:
    """
    Recursively counts null vs non-null leaf values in partial_json.
    Returns (null_count, filled_count).
    """
    null_count = 0
    filled_count = 0

    if isinstance(obj, dict):
        for v in obj.values():
            n, f = _count_fields(v)
            null_count += n
            filled_count += f
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, str):
                continue  # junction array values — not meaningful completeness signals
            n, f = _count_fields(item)
            null_count += n
            filled_count += f
    else:
        if obj is None:
            null_count += 1
        else:
            filled_count += 1

    return null_count, filled_count


# ---------------------------------------------------------------------------
# Source section formatters — XML tags with embedded authority metadata
# ---------------------------------------------------------------------------

# Site names that are OEM official India sources.
# Add new brands here as they are onboarded.
_OEM_OFFICIAL_SITE_NAMES: frozenset[str] = frozenset({
    "motorola_official",
    "samsung_official",
    "oneplus_official",
    "realme_official",
    "oppo_official",
    "vivo_official",
    "xiaomi_official",
    "apple_official",
    "google_official",
    "nothing_official",
    "iqoo_official",
    "tecno_official",
    "itel_official",
    "lava_official",
    "poco_official",
    "infinix_official",
})

# Site names that are aggregators (global specs, not India-specific).
_AGGREGATOR_SITE_NAMES: frozenset[str] = frozenset({
    "gsmarena",
    "gsm",
    "smartprix",
    "91mobiles",
    "devicespecifications",
    "nanoreview",
    "kimovil",
    "phonearena",
})


def _format_scraped_section(site_name: str, raw_id: int, content: str) -> str:
    """
    Wraps a scraped source in an XML tag with authority metadata.

    OEM official sources are tagged as authoritative for all India-specific
    fields. Aggregator sources are tagged with explicit restrictions on which
    fields they are forbidden to supply.

    The XML tag attributes are read by Gemini as structured constraints, not
    prose rules — this gives much stronger source isolation than markdown headers.

    NOTE: The combined string (including these XML tags) is still passed to
    build_spec_json() for char offset evidence attribution — str.find() works
    correctly against content inside XML tags.
    """
    if site_name in _OEM_OFFICIAL_SITE_NAMES:
        return (
            f'<source type="oem_official" raw_id="{raw_id}" site="{site_name}"\n'
            f'  authority="HIGHEST — authoritative for all India specs"\n'
            f'  variants="AUTHORITATIVE — extract ONLY variants listed in this section"\n'
            f'  sim_config="AUTHORITATIVE — always prefer this over aggregators"\n'
            f'  charger_in_box="AUTHORITATIVE — reflects India retail unit"\n'
            f'  use_for="all fields">\n'
            f'{content}\n'
            f'</source>'
        )
    elif site_name in _AGGREGATOR_SITE_NAMES:
        return (
            f'<source type="aggregator" raw_id="{raw_id}" site="{site_name}"\n'
            f'  authority="LOW — global specs, may not reflect India retail"\n'
            f'  restrictions="FORBIDDEN for: variants, sim_configuration, esim_support,\n'
            f'    charger_in_box. Use only when oem_official and transcripts are both\n'
            f'    silent on a field. Never use to add or override variant objects.">\n'
            f'{content}\n'
            f'</source>'
        )
    else:
        # Unknown site — treat as low authority
        return (
            f'<source type="other" raw_id="{raw_id}" site="{site_name}"\n'
            f'  authority="LOW — use only as last resort fallback">\n'
            f'{content}\n'
            f'</source>'
        )


def _format_transcript_section(
    raw_transcript_id: int,
    rank: int,
    content: str,
) -> str:
    """
    Wraps a YouTube transcript in an XML tag with authority metadata.

    Transcripts are India-based reviewers with a physical unit. They are
    authoritative for India retail details and can CONFIRM variant counts
    from the OEM source but cannot INTRODUCE new variants.

    rank=1 is the highest-reliability transcript (by channel reliability score).
    """
    return (
        f'<source type="transcript" raw_transcript_id="{raw_transcript_id}"\n'
        f'  transcript_rank="{rank}"\n'
        f'  authority="HIGH for India retail details"\n'
        f'  use_for="charger_in_box, accessories, india_launch_price,\n'
        f'    volte_vo5g_vowifi, color_availability, variant_price_confirmation"\n'
        f'  variants="CAN CONFIRM prices for variants already found in oem_official.\n'
        f'    CANNOT introduce new variant objects not present in oem_official.\n'
        f'    If transcript mentions a storage tier not in OEM source, ignore it."\n'
        f'  restrictions="Approximate numbers — defer to oem_official for exact\n'
        f'    measurements (dimensions, resolution, battery capacity).">\n'
        f'{content}\n'
        f'</source>'
    )


# ---------------------------------------------------------------------------
# Source assembler
# ---------------------------------------------------------------------------

async def assemble_run_a_input(
    raw_source_ids: list[int],
    raw_transcript_ids: list[int],
    brand: str,
) -> tuple[str, dict[str, int | list[int]], list[int], dict[int, str], dict[int, str]]:
    """
    Fetches all source files from Supabase Storage and concatenates them
    in a fixed order for Run A (v5: multi-transcript support).

    CONCATENATION ORDER:
      1. OEM official markdown  — wrapped in <source type="oem_official"> XML tag
      2. GSMArena markdown      — wrapped in <source type="aggregator"> XML tag
      3. Smartprix markdown     — wrapped in <source type="aggregator"> XML tag
      4. DeviceSpecifications   — wrapped in <source type="aggregator"> XML tag
      5. Any other aggregator   — wrapped in <source type="other"> XML tag
      99. Processed transcript(s) — wrapped in <source type="transcript" rank=N> XML tag;
                                    up to 3, in reliability rank order (rank=1 = best)

    XML source tags embed authority metadata and field-level restrictions directly
    on each source section. Gemini reads these as structured constraints, giving
    much stronger source isolation than plain markdown headers.

    The combined string (with XML tags) is passed to build_spec_json() unchanged —
    str.find() works correctly against content inside XML tags for char offsets.

    For each transcript: uses translated_transcript_path if
    youtube_raw_transcript_data.translation_status = 'translation_complete',
    otherwise falls back to processed_transcript_path.

    Returns:
        combined_content:     All files concatenated with section headers.
                              This string IS the assembled_source_string passed
                              to build_spec_json() for char offset computation.
        file_map:             {
                                "gsmarena": raw_id,
                                "samsung_official": raw_id,
                                "transcripts": [raw_transcript_id, ...]
                              }
        failed_source_ids:    raw_ids that failed to fetch.

    Raises:
        ValueError: If raw_source_ids is empty AND raw_transcript_ids is empty,
                    OR if two sources share the same site_name,
                    OR if ALL fetches (scraped + transcript) fail.
    """
    if not raw_source_ids and not raw_transcript_ids:
        raise ValueError(
            "assemble_run_a_input: raw_source_ids is empty and raw_transcript_ids is empty. "
            "Nothing to assemble."
        )

    # Step 1 — Fetch scraped rows, sort by concat priority
    source_rows = await asyncio.to_thread(fetch_raw_source_rows, raw_source_ids)
    source_rows.sort(key=lambda r: get_concat_order(r["site_name"]))

    # Guard against duplicate site_name (data integrity)
    seen_site_names: set[str] = set()
    for row in source_rows:
        sn = row["site_name"]
        if sn in seen_site_names:
            raise ValueError(
                f"assemble_run_a_input: duplicate site_name={sn!r} across "
                f"raw_source_ids={raw_source_ids}. Two rows share the same site_name — "
                "file_map would be corrupted. De-duplicate raw_source_ids before calling Run A."
            )
        seen_site_names.add(sn)

    # Step 2 — Fetch each scraped markdown file
    sections: list[str] = []
    file_map: dict[str, int | list[int]] = {}
    failed_source_ids: list[int] = []
    source_content_map: dict[int, str] = {}     # raw_id → raw file content (before XML wrap)
    transcript_content_map: dict[int, str] = {} # raw_transcript_id → raw transcript content

    for row in source_rows:
        raw_id = row["raw_id"]
        site_name = row["site_name"]
        markdown_path = row["markdown_path"]

        try:
            content = await fetch_file_content(markdown_path)
        except Exception as exc:
            logger.warning(
                "assemble_run_a_input: failed to fetch source raw_id=%d site=%r: %s",
                raw_id, site_name, exc,
            )
            failed_source_ids.append(raw_id)
            continue

        source_content_map[raw_id] = content  # capture raw content before XML wrapping
        file_map[site_name] = raw_id
        sections.append(_format_scraped_section(site_name, raw_id, content))
        logger.info(
            "assemble_run_a_input: fetched raw_id=%d site=%r (%d chars)",
            raw_id, site_name, len(content),
        )

    # Step 3 — Fetch each transcript in rank order (up to 3)
    fetched_transcript_ids: list[int] = []

    for raw_transcript_id in raw_transcript_ids:
        transcript_row = await asyncio.to_thread(
            fetch_transcript_row, raw_transcript_id
        )

        # v5 path selection: prefer translated path if translation is complete
        translation_status = transcript_row.get("translation_status")
        if translation_status == "translation_complete":
            transcript_path = (
                transcript_row.get("translated_transcript_path")
                or transcript_row.get("processed_transcript_path")
            )
        else:
            transcript_path = transcript_row.get("processed_transcript_path")

        if not transcript_path:
            logger.warning(
                "assemble_run_a_input: transcript row has no usable path "
                "(raw_transcript_id=%d, translation_status=%r) — skipping",
                raw_transcript_id, translation_status,
            )
            continue

        try:
            transcript_content = await fetch_file_content(transcript_path)
            transcript_content_map[raw_transcript_id] = transcript_content  # capture raw content before XML wrapping
            fetched_transcript_ids.append(raw_transcript_id)
            transcript_rank = len(fetched_transcript_ids)  # 1-indexed rank (1 = best)
            sections.append(
                _format_transcript_section(raw_transcript_id, transcript_rank, transcript_content)
            )
            logger.info(
                "assemble_run_a_input: fetched transcript raw_transcript_id=%d rank=%d (%d chars)",
                raw_transcript_id, transcript_rank, len(transcript_content),
            )
        except Exception as exc:
            # Transcript fetch failure is logged as error (high-value input).
            # Unlike scraped sources, we do NOT append to failed_source_ids —
            # those are raw_ids from raw_scraped_data; transcript IDs are separate.
            logger.error(
                "assemble_run_a_input: transcript fetch failed "
                "(raw_transcript_id=%d, path=%r): %s",
                raw_transcript_id, transcript_path, exc,
            )

    if fetched_transcript_ids:
        file_map["transcripts"] = fetched_transcript_ids

    # Only raise after all sources attempted
    if not sections:
        raise ValueError(
            "assemble_run_a_input: all source file fetches failed "
            f"(scraped raw_source_ids={raw_source_ids}, "
            f"transcript raw_transcript_ids={raw_transcript_ids})."
        )

    combined_content = "\n\n".join(sections)
    logger.info(
        "assemble_run_a_input: assembled %d source sections (%d chars). "
        "failed_source_ids=%s",
        len(sections), len(combined_content), failed_source_ids,
    )
    return combined_content, file_map, failed_source_ids, source_content_map, transcript_content_map


# ---------------------------------------------------------------------------
# Schema version → examples module mapping
# ---------------------------------------------------------------------------

_EXAMPLES_MODULE_MAP: dict[str, str] = {
    "v2": "app.config.extraction_examples_run_a_v2",
}

_DEFAULT_SCHEMA_VERSION = "v2"


@functools.lru_cache(maxsize=None)
def _load_run_a_examples(schema_version: str):
    """
    Loads RUN_A_EXAMPLES from the module for schema_version.
    Falls back to _DEFAULT_SCHEMA_VERSION with a warning if not found.

    Returns:
        (RUN_A_EXAMPLES, actual_schema_version_used)
    """
    import importlib
    module_path = _EXAMPLES_MODULE_MAP.get(schema_version)
    if module_path is None:
        logger.warning(
            "_load_run_a_examples: schema_version=%r not in _EXAMPLES_MODULE_MAP. "
            "Falling back to default version %r.",
            schema_version, _DEFAULT_SCHEMA_VERSION,
        )
        module_path = _EXAMPLES_MODULE_MAP[_DEFAULT_SCHEMA_VERSION]
        schema_version = _DEFAULT_SCHEMA_VERSION
    mod = importlib.import_module(module_path)
    return getattr(mod, "RUN_A_EXAMPLES"), schema_version


# ---------------------------------------------------------------------------
# Few-shot prompt block builder
# ---------------------------------------------------------------------------

def _build_few_shot_block(examples: list[dict]) -> str:
    """
    Formats a list of example dicts into a few-shot text block
    for inclusion in the system prompt.

    Fix 9: Accepts mixed key schemas in the examples list. Old examples may use
    'input_excerpt'/'expected_output'; new examples use 'input'/'output'.
    Both are now handled so that the few-shot block is never silently blank.
    Priority: 'input' > 'input_excerpt', 'output' > 'expected_output'.
    """
    if not examples:
        return ""

    import json
    blocks: list[str] = []
    for i, ex in enumerate(examples, 1):
        # Fix 9: accept either key variant for the input and output slots
        inp = ex.get("input") or ex.get("input_excerpt") or ""
        out = ex.get("output") or ex.get("expected_output") or {}
        blocks.append(
            f"EXAMPLE {i} INPUT:\n{inp}\n\n"
            f"EXAMPLE {i} OUTPUT:\n{json.dumps(out, separators=(',', ':'))}"
        )
    return "\n\n---\n\n".join(blocks)



# ---------------------------------------------------------------------------
# ECD / system prompt builder
# ---------------------------------------------------------------------------

# Static extraction rules appended to the ECD system prompt
_EXTRACTION_RULES = """\
CAMERA FEATURES: Distinguish per-lens features (stabilization, autofocus_type) from \
general camera features (Night Mode, RAW Capture). Populate both independently.

AI FEATURES: Only extract AI features explicitly named by the manufacturer. \
Do not infer AI capabilities from general ML/NPU mentions.

CELLULAR FEATURES: 5G SA and 5G NSA are distinct. Extract both if both are mentioned. \
VoLTE, VoWiFi, Vo5G are distinct boolean fields — extract each independently.

LOOKUP VALUES: Examples teach the exact lookup string values. Match them precisely. \
Case-sensitive: "LPDDR5X" not "lpddr5x". If no match: null, never invent a value.

STORAGE AND RAM — ALWAYS IN GB: storage_capacity and ram_capacity must be output in GB. \
Convert TB to GB (1 TB = 1024 GB). Never output TB values.

SAR VALUES — DO NOT EXTRACT:
  certifications.sar_head and certifications.sar_body: Leave both null.
  SAR values for India (0mm separation) are always fetched during enrichment
  from tec.fptc.gov.in. GSMArena SAR values use EU 10mm standard and are
  never valid for India. Do not extract from any source.

CHARGER IN BOX — INDIA UNIT: Must reflect India retail unit specifically. \
International "no charger" news ≠ India unit policy. If not stated, output null.

BAND FORMAT — ALWAYS PREFIX:
  bands_4g values must be prefixed with "B": output "B1", "B28", "B41" — never "1", "28", "41".
  bands_5g values must be prefixed with "n": output "n78", "n28", "n41" — never "78", "28", "41".
  Never output raw numbers without the prefix. Never output "Band 28" or "n 78" with a space.

SATELLITE BANDS (bands_satellite) — NTN/SAT CONNECTIVITY:
  Populate network.bands_satellite[] ONLY when the source explicitly mentions satellite
  connectivity, NTN (Non-Terrestrial Network), or satellite SOS/messaging support.
  Allowed values: "n253", "n254", "n255", "n256", "NB-IoT 23", "NTN".
  Always prefix numbered bands with "n": output "n253", never "253" or "Band n253".
  Output "NTN" only for generic satellite support with no specific band stated.
  If no satellite connectivity is mentioned: output [] (empty array).

FORBIDDEN — NEVER EXTRACT (computed by Run C):
  displays[*].ppi            → computed: sqrt(h²+w²) / size_inch
  camera_lenses[*].sensor_size_decimal → computed: 1.0 / sensor_size_denominator

VOLTE / VOWIFI / VO5G (ALL THREE REQUIRED):
  These are three distinct boolean fields. Extract each independently.
  volte   = VoLTE: 4G voice calling support. Look for "VoLTE", "4G VoLTE",
            "HD Voice", or "Voice over LTE". Default to null if not mentioned —
            do NOT assume true from "4G support".
  vowifi  = VoWiFi: Wi-Fi Calling. Look for "VoWiFi", "Wi-Fi Calling",
            "Voice over Wi-Fi".
  vo5g    = Vo5G / VoNR: only true if source explicitly states "VoNR", "Vo5G",
            or "Voice over 5G NR". 5G support alone does NOT imply Vo5G.
  All three go in network.volte, network.vowifi, network.vo5g respectively.

HEADPHONE JACK (has_3_5mm_jack):
  NOT NULL boolean — MUST be populated for every phone.
  Extract under audio.has_3_5mm_jack.
  true  = source confirms 3.5mm jack / headphone jack present.
  false = source states "no headphone jack", "no 3.5mm jack", or USB-C audio
          only with no 3.5mm port mentioned.
  If sources conflict: trust OEM spec page; transcript reviewer observation second.
  Never leave this field null — always resolve to true or false.

WIFI HOTSPOT (wifi_hotspot):
  Extract under connectivity.wifi_hotspot.
  Almost all modern smartphones support mobile hotspot / tethering.
  Look for "Mobile Hotspot", "Wi-Fi Hotspot", "Personal Hotspot", "Tethering".
  Default to true if phone supports Wi-Fi and no source explicitly denies hotspot.
  Only set false if source explicitly states hotspot is not supported.

WIFI STANDARD FORMAT:
  Output EXACTLY one canonical form (see LOOKUP FIELD ALLOWED VALUES above).
    "802.11a/b/g/n/ac" or "Wi-Fi 5"    → "Wi-Fi 5 (802.11ac)"
    "802.11a/b/g/n/ac/ax" or "Wi-Fi 6" → "Wi-Fi 6 (802.11ax)"
    "Wi-Fi 6E"                          → "Wi-Fi 6E (802.11ax)"
    "802.11be" or "Wi-Fi 7"             → "Wi-Fi 7 (802.11be)"
    "802.11a/b/g/n" or "Wi-Fi 4"        → "Wi-Fi 4 (802.11n)"
  Never output raw 802.11 notation.

BANDS_2G AND BANDS_3G:
  bands_2g: plain text, e.g. "GSM: B2/B3/B5/B8". Output null if not present.
  bands_3g: plain text, e.g. "UMTS: B1/B2/B4/B5/B8". Output null if not present.
  These are NOT junction arrays — do NOT split them into separate values.

FINGERPRINT SENSOR (sensors.fingerprint_sensor):
  Always specify the exact technology and mounting. Never write "In-display"
  or "Fingerprint on display" — these are ambiguous and technically imprecise.

  Use exactly one of these canonical values:
    "Under Display (Optical)"      — optical sensor behind the display panel
    "Under Display (Ultrasonic)"   — ultrasonic sensor behind the display panel
    "Side-mounted"                 — integrated into the power button on the frame
    "Rear-mounted"                 — on the back of the phone
    "None"                         — phone has no fingerprint sensor

  Decision rules:
    - "In-display", "on-screen", or "under-display" without type stated
      → check source for "ultrasonic" or "Qualcomm Ultrasonic":
          found    → "Under Display (Ultrasonic)"
          not found → "Under Display (Optical)"  (optical is far more common;
                       ultrasonic is almost exclusively Qualcomm flagship only)
    - "Side fingerprint" / "power button fingerprint" → "Side-mounted"
    - "Rear fingerprint"                              → "Rear-mounted"
    - No fingerprint sensor mentioned                 → "None"

  If sensors.fingerprint_sensor is in biometrics[] instead — still populate
  the sensors.fingerprint_sensor field with the canonical value above.

CHIPSET NAME — STRIP PART NUMBERS (5a):
  Extract the canonical marketing name only. Remove internal model/part number prefixes.
  Qualcomm: "Qualcomm SM7435-AB Snapdragon 7s Gen 2" → "Qualcomm Snapdragon 7s Gen 2"
  Qualcomm: "Qualcomm SM8650 Snapdragon 8 Gen 3"     → "Qualcomm Snapdragon 8 Gen 3"
  MediaTek: "MediaTek MT6985 Dimensity 9200"          → "MediaTek Dimensity 9200"
  MediaTek: "MediaTek MT6893 Dimensity 1200"          → "MediaTek Dimensity 1200"
  Apple:    "Apple A17 Pro"                           → "Apple A17 Pro" (no part number to strip)
  Rule: Remove "SM####-XX" (Qualcomm) or "MT####" (MediaTek) part numbers. Keep brand + marketing name.

CHIPSET — NAME AND CLOCK SPEEDS ONLY (5b):
  chipset_name: MUST extract.
  cpu_clock_speed: Extract if explicitly stated in source. GHz. Phone-specific.
  gpu_clock_speed: Extract if explicitly stated in source. MHz. Phone-specific.

  CRITICAL — OUTPUT FORMAT FOR ALL CHIPSET FIELDS:
    Every chipset field MUST appear in your output JSON, even if you have no
    data for it. For fields with no source data, output the field with value null.
    NEVER omit a field from the chipset object — always include it explicitly.

  Fields to leave with value null (NOT omit) when no source data is found:
    fabrication_node, number_of_cores,
    cpu_ultra_high_performance_cores, cpu_high_performance_cores,
    cpu_efficiency_cores, cpu_architecture, gpu_name, gpu_cores,
    npu_details, npu_tops.

  CORRECT output when fields are absent from source:
    "chipset": {
      "chipset_name": "Qualcomm Snapdragon 7s Gen 2",
      "cpu_clock_speed": 2.4,
      "gpu_clock_speed": null,
      "number_of_cores": 8,
      "fabrication_node": 4,
      "cpu_architecture": null,
      "cpu_ultra_high_performance_cores": null,
      "cpu_high_performance_cores": "4x2.40 GHz Cortex-A78",
      "cpu_efficiency_cores": "4x1.95 GHz Cortex-A55",
      "gpu_name": "Adreno 710",
      "gpu_cores": null,
      "npu_details": null,
      "npu_tops": null
    }

  WRONG output (fields omitted):
    "chipset": {
      "chipset_name": "Qualcomm Snapdragon 7s Gen 2",
      "cpu_clock_speed": 2.4,
      "gpu_name": "Adreno 710"
    }
    ↑ gpu_cores, npu_details, npu_tops, cpu_architecture etc. MUST be present as null.

CPU ARCHITECTURE — DO NOT EXTRACT (5c):
  chipset.cpu_architecture: Leave this field NULL in extraction output.
  Site data (GSMArena, OEM sites) does not mention ARMv9.2, ARMv8-A, etc.
  This field is sourced from chipset manufacturer datasheets during ENRICHMENT — not extraction.

CAMERA SENSOR TYPE — TECHNOLOGY NOT BRAND (5e):
  camera_lenses[*].sensor_type: Extract the image sensor TECHNOLOGY TYPE, not the manufacturer.
  Valid technology types:
    "CMOS"         — standard complementary metal-oxide-semiconductor sensor (most phones)
    "BSI CMOS"     — Back-Side Illuminated CMOS (improved low-light)
    "Stacked CMOS" — stacked BSI CMOS with separate logic layer (fastest readout, flagships)
    "RYYB"         — Huawei's alternative colour filter array (Red-Yellow-Yellow-Blue)
  WRONG (brands/product families, not sensor types):
    "Sony"     → sensor_model = "Sony Lytia 700C", sensor_type = "CMOS"
    "Samsung"  → sensor_model = "Samsung ISOCELL GN5", sensor_type = "BSI CMOS"
    "LYT" / "Lytia" → Sony's product family name, not a sensor technology type
    "ISOCELL"  → Samsung's product family name, not a sensor technology type
  The sensor manufacturer and model name always go in sensor_model.
  sensor_type must be the underlying silicon technology. If not explicitly stated, default to "CMOS".

OPTICAL ZOOM — TELEPHOTO AND PERISCOPE ONLY (5f):
  camera_lenses[*].optical_zoom_capacity: ONLY populate for Telephoto and Periscope lens types.
  For Main, Ultra-wide, and Front lenses: always set optical_zoom_capacity = null.
  "In-sensor zoom" or "lossless crop zoom" on the main camera is NOT optical zoom.
  Digital zoom is not optical zoom — never use digital zoom values here.


CHARGING VOLTAGE AND AMPERAGE — EXPLICIT ONLY (5h):
  charging.charging_voltage and charging.charging_ampere:
  Extract ONLY when explicitly stated by the manufacturer in the source data.
  DO NOT calculate, infer, or derive from wattage alone (W = V x A requires both values).
  DO NOT guess standard values (e.g. do not assume "5V/3A" for a 15W charger).
  If not explicitly stated: set null.

CPU CORE CLASSIFIER — HIERARCHICAL LOGIC (5i):
  Apply these rules in strict priority order. Stop at the first matching rule.
  Classify each core cluster by its ARM codename only — ignore cluster count or position.
  Three output slots:
    [ULTRA]  → cpu_ultra_high_performance_cores
    [PERF]   → cpu_high_performance_cores
    [EFF]    → cpu_efficiency_cores

  RULE 1 — ARM Cortex-X series → ALWAYS [ULTRA]:
    Any core starting with "Cortex-X" (X1, X2, X3, X4, X925, X1-Extreme etc.) → [ULTRA]
    No exceptions. These are always the highest-performance tier by definition.

  RULE 2 — ARM Cortex-A7x series → ALWAYS [PERF]:
    A72, A73, A75, A76, A77, A78, A710, A715, A720, A725, A78C, A78AE → [PERF]
    These are ALWAYS [PERF] regardless of what other cores are present.
    CRITICAL: A7x cores go to cpu_high_performance_cores. NEVER to cpu_ultra_high_performance_cores.
    If no Cortex-X core exists → cpu_ultra_high_performance_cores = null. Do not promote A7x.
    Example: "4x2.40 GHz Cortex-A78 & 4x1.95 GHz Cortex-A55"
      → cpu_high_performance_cores = "4x2.40 GHz Cortex-A78"
      → cpu_efficiency_cores = "4x1.95 GHz Cortex-A55"
      → cpu_ultra_high_performance_cores = null

  RULE 3 — ARM Cortex-A5x series → ALWAYS [EFF]:
    A53, A55, A510, A520 → [EFF] always.
    A35, A37, A7 → [EFF] always (legacy 32-bit efficiency cores).

  RULE 4 — Qualcomm Oryon (Snapdragon 8 Elite and newer):
    Oryon Phoenix L (2 prime cores) → [ULTRA]
    Oryon Phoenix M (6 performance cores) → [PERF]
    cpu_efficiency_cores = null (Oryon has no efficiency cores).

  RULE 5 — Apple:
    2 P-cores (performance) → [ULTRA]
    4 E-cores (efficiency) → [EFF]
    cpu_high_performance_cores = null (Apple uses two-tier only).

  RULE 6 — MediaTek All-Big (Dimensity 9300, 9400):
    cpu_efficiency_cores = null ALWAYS for these chips.
    Dimensity 9300: 4×Cortex-X4 → [ULTRA], 4×Cortex-A720 → [PERF]
    Dimensity 9400: 1×Cortex-X925 + 3×Cortex-X4 → [ULTRA], 4×Cortex-A720 → [PERF]

  RULE 7 — Qualcomm Kryo legacy (Kryo 200–490 series, Snapdragon ~2017–2021):
    Kryo Prime → [ULTRA], Kryo Gold → [PERF], Kryo Silver → [EFF]
    Kryo 5xx+ uses direct Cortex naming — apply Rules 1–3 instead.

  RULE 8 — Samsung Exynos Mongoose:
    Mongoose / Exynos M series → [ULTRA]
    Paired Cortex-A cores → apply Rules 2–3.

  RULE 9 — Fallback:
    If core name matches none of the above → leave field null. Admin fills via chipset DB.

  SELF-CHECK before finalising:
    If you placed any A7x core (A78, A715, A720, etc.) in cpu_ultra_high_performance_cores
    → MOVE it to cpu_high_performance_cores and set cpu_ultra_high_performance_cores = null.
    A7x NEVER goes to ULTRA. No exceptions.

CAMERA FEATURES — PHONE-LEVEL, LOOKUP NAMES ONLY (5j):
  camera_features (TOP-LEVEL field):
  Collect ALL camera features from ALL sources (OEM site, GSMArena, transcripts)
  into one deduplicated list at the root camera_features key.
  Use ONLY canonical feature names from the lookup_camera_features table provided.
  If a source feature name does not exactly match a lookup entry, map to the closest match.
  DO NOT add features to individual camera_lenses[*] objects — that field is removed.

PERFORMANCE BENCHMARKS — VERBAL CONVERSION AND ROUNDING (5k):
  performance_benchmarks: Extract benchmark scores from aggregator scraped data.
  Transcripts may give verbal approximations — convert to integers FIRST:
    "around 2 million AnTuTu"  → 2,000,000
    "around 2.3 million"       → 2,300,000
    "over half a million"      → 500,000
  After converting verbal scores to integers:
    If multiple sources: average all values, then round.
    AnTuTu: round to nearest 100,000.
    Geekbench: round to nearest 100.
    3DMark: round to nearest 10.
  Always record antutu_version / geekbench_version when available (v9, v10, v5, v6).
  cooling_system: extract if mentioned (e.g. "Vapor Chamber", "Graphite Sheet").

VIDEO RESOLUTION NORMALIZATION (5m):
  video_capabilities.rear_video_resolutions and front_video_resolutions:
  Normalize ALL resolution labels before outputting:
    UHD = 4K (both mean 3840x2160 at phone camera level)
    FHD = 1080p
    HD  = 720p
    8K  = 8K
  Combine entries at the same resolution+fps that only differ in aspect ratio (16:9 vs 20:9).
  Example: "UHD @30fps, UHD 20:9@30fps 3840x1728, FHD@30fps, FHD@60fps"
           → "4K@30fps, 1080p@30fps, 1080p@60fps"
  List highest resolution first. Do not duplicate the same resolution+fps combination.

SENSOR SIZE DENOMINATOR — EXACT FORMAT ONLY (5n):
  camera_lenses[*].sensor_size_denominator: Extract ONLY from explicit optical format
  notation "1/X.X\"" or "1/X.X inch" found verbatim in the source text.
    "1/1.56 inch" → sensor_size_denominator: 1.56
    "1/2.52\""   → sensor_size_denominator: 2.52
    "1/3.0\""    → sensor_size_denominator: 3.0
  FORBIDDEN SOURCES:
    • pixel size (µm) — "1.0µm pixel size" is NOT sensor size format. Never parse
      the number from a µm value as sensor_size_denominator.
    • Megapixels — MP count has no relation to sensor format.
    • Any number not preceded by "1/" in sensor format context.
  If the source does not contain "1/X" optical format for a lens: output null.
  Do NOT guess or approximate. null is correct when the data is absent.

REAR CAMERA SETUP:
  Count ONLY rear-facing lens types in camera_lenses[]:
  Valid rear types: Main | Ultra-wide | Telephoto | Periscope | Macro | Depth
  NEVER count any Front* lens type ("Front", "Front (Cover Display)", "Front (Inner Display)", "Front (Secondary)").
  1 rear lens → "Single" | 2 → "Dual" | 3 → "Triple" | 4 → "Quad"
  Count the lens objects with rear types, then map to word. Do not guess.

  CAMERA LENS TYPE — ALLOWED VALUES:
  Standard phones: "Main" | "Ultra-wide" | "Telephoto" | "Periscope" | "Macro" | "Depth" | "Front"| "Front (Secondary)"
  Foldable/flip phones only:
  "Front (Cover Display)" | "Front (Inner Display)" 

ESIM AND SIM CONFIGURATION — LOGICAL CONSISTENCY REQUIRED (5p):
  network.esim_support and network.sim_configuration MUST be logically consistent.
  They cannot contradict each other:

  Rule A — India OEM site is authoritative for BOTH fields:
    Always use the OEM official India site to determine SIM configuration.
    If OEM India site says "Dual SIM (2 Nano SIMs)" or "Dual SIM (Nano-SIM, dual stand-by)":
      → sim_configuration = "Dual SIM (Nano-SIM, dual stand-by)"
      → esim_support = false (regardless of what GSMArena says)
    GSMArena may list global/international variants that include eSIM not sold in India.
    If OEM India site is silent on eSIM: esim_support = false.

  Rule B — Consistency check:
    If sim_configuration resolves to any Dual Nano-SIM variant (no mention of eSIM):
      → esim_support MUST be false.
    If esim_support = true:
      → sim_configuration must include "eSIM" (e.g. "Nano-SIM + eSIM").
    Contradictory values (esim=true + Dual Nano-SIM config) are FORBIDDEN.
    If sources conflict, defer to OEM India site. Set esim_support = false when unsure.

AUDIO CODECS — EXPLICIT SOURCE ONLY, ALWAYS CHECK DEVICESPECIFICATIONS (5q):
  audio.audio_codecs: Extract ONLY codec names that are EXPLICITLY listed in the
  source documents. Do NOT infer, assume, or fill from training knowledge.

  DEVICESPECIFICATIONS SOURCE — MANDATORY CHECK:
    The <source type="other" site="devicespecifications"> section reliably lists
    supported audio formats and codecs under its "Audio" or "Multimedia" section.
    This section is always present and always lists codec data for every phone.
    You MUST extract audio codecs from this section when it is present in the input.
    Do NOT skip this source for audio_codecs. If devicespecifications lists
    "MP3, AAC, FLAC, WAV, ALAC, APE, OGG" → extract all of them.

  CORRECT: Source says "Supports LDAC, aptX Adaptive, AAC" → extract those.
  CORRECT: devicespecifications lists audio formats → extract them all.
  WRONG:   Source says nothing about codecs → do NOT invent them from training knowledge.

  If no source (including devicespecifications) lists audio codecs explicitly: output [].

  IMPORTANT — CERTIFICATIONS ARE NOT CODECS:
    Dolby Atmos, Hi-Res Audio, DTS, DTS:X, DTS HD are audio CERTIFICATIONS.
    They belong in certifications.audio_certifications ONLY.
    NEVER put them in audio.audio_codecs. NEVER put them in audio.audio_features.
    audio.audio_features is for speaker configuration notes only (e.g. "Stereo Speakers").
    If source mentions Dolby Atmos in the audio section → certifications.audio_certifications.

USB FEATURES — LOOKUP VALUES ONLY (5r-pre):
  connectivity.usb_features must contain ONLY values from the lookup_usb_features table:
    "OTG" | "DisplayPort" | "USB Power Delivery" | "Audio Adapter Accessory Mode" |
    "Reverse Charging" | "Desktop Mode" | "USB Tethering"

  NEVER extract generic capabilities as usb_features:
    "USB Charging" → this is not a feature, every USB port charges. Omit entirely.
    "USB Storage" → this is not a distinct feature. Omit entirely.
    "USB Data Transfer" → omit.
    "Mass Storage" → omit.
  Only extract values that exactly match the allowed list above.

WIFI TECHNOLOGIES — PROTOCOLS ONLY (5r):
  connectivity.wifi_technologies must contain ONLY technology protocol names
  from the lookup_wifi_technologies table (e.g., OFDMA, MU-MIMO, MLO, WPA3,
  Beamforming, TWT, Wi-Fi Direct, 4x4 MIMO, 1024-QAM, 4096-QAM, ETH320).

  NEVER include:
    • Frequency bands: "2.4GHz", "5GHz", "6GHz" are properties of wifi_standard,
      not technologies. Do not put them in this field.
    • "Wi-Fi Hotspot": this has a dedicated boolean column connectivity.wifi_hotspot.
      If the source mentions hotspot capability, set wifi_hotspot = true and do NOT
      add "Wi-Fi Hotspot" to wifi_technologies.

  If a source lists "2.4GHz & 5GHz": set wifi_hotspot from its own evidence;
  ignore the frequency bands for wifi_technologies.

VARIANTS — OEM OFFICIAL ONLY — XML SOURCE ANCHORED (5s):
  The input sources are wrapped in XML tags. Source authority is encoded in the tag:
    <source type="oem_official" ...>  — ONLY valid source for variants
    <source type="aggregator" ...>    — FORBIDDEN for variants (global tiers, not India)
    <source type="transcript" ...>    — can CONFIRM prices for OEM variants only

  PROCESS:
    Step 1 — Open the <source type="oem_official"> section.
    Step 2 — Count how many distinct RAM+Storage combinations are listed there.
    Step 3 — Output EXACTLY that many variant objects. Not one more.
    Step 4 — For each OEM variant, check transcripts to fill launch_price or
              ram_type if the OEM source is missing those details.
    Step 5 — If you output more variants than exist in the oem_official section,
              you have used aggregator data. Delete the excess variants.

  COUNT ANCHOR — HARD RULE:
    Count the variants in <source type="oem_official">. Call this N.
    Your variants[] array length MUST equal N.
    If the aggregator section shows 7 variants and OEM shows 2: output 2.
    If a transcript mentions a storage tier not in the OEM section: ignore it.

  SELF-CHECK before finalising variants[]:
    For every variant object: is its evidence_text from the oem_official raw_id?
    If any variant's evidence comes from an aggregator raw_id → delete that variant.

BODY BUTTONS — PHYSICAL CONTROLS ONLY (5t):
  body.buttons = physical buttons the user presses: power button, volume rocker,
  dedicated action button, camera shutter button.
  DO NOT include: USB-C port, SIM tray, speaker grille, microphone holes, headphone jack.
  These are openings/ports, not buttons. Example correct output:
  "Power button (Right), Volume rocker (Right)"

DISPLAY FEATURES — NO NON-FEATURE VALUES (5u):
  display_features must contain ONLY display capabilities and certifications from the lookup.
  DO NOT add:
    • IP ratings (IP67, IP68) — go in certifications.ip_ratings
    • Widevine levels (Widevine L1) — go in certifications.widevine_level
    • Camera cutout names ("Punch Hole", "Notch") — go in camera_overview.front_camera_shape
    • Marketing slogans or product names
  Only display technology features belong here (HDR10+, LTPO, Always-on Display, etc.)

EXTRA_FEATURES vs CAMERA_FEATURES — NO OVERLAP (5v):
  extra_features = phone-level UX gestures and OS features:
    Flip for DND, Pick Up to Silence, Three-Finger Screenshot, Ready For,
    Moto Connect, Gametime, Peek Display, Smart Water Touch, etc.
  camera_features = camera app modes and features:
    Night Vision, Portrait, Pro Mode, Macro Vision, Spot Color, Timelapse, etc.
  RULE: A feature that belongs in camera_features must NOT also appear in extra_features.
  No item should appear in both lists. Camera features go only in camera_features.

AUTOFOCUS SPECIFICITY (5w):
  Always pick the MOST SPECIFIC matching autofocus lookup value.
  If source says "Quad PDAF" → output "Quad PDAF", never degrade to "PDAF".
  If source says "Dual Pixel PDAF" → output "Dual Pixel PDAF", never "PDAF".
  Only output "PDAF" when source uses only the bare word "PDAF" with no qualifier.
  "Quad Pixel Technology" is a sensor architecture term, NOT an autofocus type → null.

MACRO AND DEPTH LENS — NO PHANTOM ENTRIES (5x):
  Only create a lens_type="Macro" entry if the source explicitly states a dedicated
  low-resolution macro sensor (typically 2–5 MP) as a separate physical lens.
  Only create a lens_type="Depth" entry if the source explicitly states a dedicated
  depth/ToF sensor with its own specifications.
  Motorola "Macro Vision" = ultra-wide with macro capability → set is_macro_capable=true
  on the Ultra-wide lens. Do NOT create a separate Macro lens entry.
  HARD STOP: After building camera_lenses, delete any entry where BOTH megapixels=null
  AND aperture=null. A real lens always has at least one of these. Null-null = hallucination.

FRONT CAMERA SHAPE vs POSITION — TWO DIFFERENT FIELDS (5y):
  front_camera_shape = cutout TYPE: "Punch-hole" | "Notch" | "Pill" | "Dynamic Island"
  front_camera_position = horizontal LOCATION: "Center" | "Left" | "Top Bezel"
  "Punch-hole" is a SHAPE, never a position.
  Center punch-hole camera → front_camera_shape="Punch-hole", front_camera_position="Center".

  FOLDABLE AND DUAL FRONT CAMERA RULES:
  Use the three specialised lens_type values ONLY for phones with multiple front cameras:
    "Front (Cover Display)" — selfie camera on the outer/cover panel of a flip or fold phone.
    "Front (Inner Display)" — selfie camera on the inner display of a book-fold phone.
    "Front (Secondary)"    — second selfie camera on the SAME display face (e.g. Xiaomi 14 CiVi
                             with dual punch-holes on the main display side by side).
  For all standard single-front-camera phones: use "Front" — never use the specialised types.

  PER-LENS front_camera_position: For each Front* lens, populate the per-lens
  front_camera_position field with the cutout location ("Center", "Left", "Top Bezel" etc.).
  This is SEPARATE from camera_overview.front_camera_position which holds the primary
  front camera's position for convenience.

  For foldables with Front (Cover Display) AND Front (Inner Display): the two cameras are
  on physically different display panels — extract each lens with its own position value.
  Leave camera_overview.front_camera_position as the position of the primary (Inner) camera.

AI CAPABILITIES — STRICT GROUNDING (5z):
  ai_system: Only set if source explicitly names a branded AI platform.
    Examples: "Galaxy AI", "Apple Intelligence", "Moto AI".
    "Hello UI" is a UI SKIN, NOT an AI system → ai_system = null.
  ai_features: Only explicitly named features from source text.
    Do NOT generate from training knowledge. If source is silent → ai_features = [].
  If ai_capabilities has nothing grounded → output ai_capabilities: {}

BIS AND WIDEVINE — DO NOT EXTRACT:
  certifications.bis_certification: Leave null. Normalizer sets true unconditionally.
  certifications.widevine_support: Leave null. Normalizer sets true for all phones >= 2017.
  certifications.widevine_level: Leave null. Normalizer sets "L1" for all phones >= 2017.
  Do not output evidence-free values for these fields. Null is the correct extraction output.

STRUCTURAL FIELDS — NO _source WRAPPER:
  The following fields must be plain values with NO _source wrapper:
  is_base_variant (boolean) | display_type | display_position | lens_type
  WRONG: {"value": true, "_source": {...}}
  CORRECT: true

SENSOR DEDUPLICATION — COMPASS:
  "Compass" and "E-Compass" are the same physical sensor.
  Always use "E-Compass" as the canonical name.
  Never output both in other_sensors. If source says "Compass" → map to "E-Compass".

DISPLAY FEATURES — EXCLUSIONS:
  Do NOT add values that duplicate dedicated numeric fields:
    refresh_rate has its own field → never add "144Hz refresh rate" to display_features
    colour_depth has its own field → never add "10 bit" or "10-bit color" to display_features
    pwm_frequency has its own field → never add "720Hz PWM" or "PWM Dimming" to display_features
    brightness_hbm and brightness_peak have own fields → never add nits values to display_features
  display_features is for qualitative capabilities (HDR10+, Always-on Display,
  Adaptive Refresh Rate, 100% DCI-P3, Game Mode, etc.) not numeric spec repetitions.

BIOMETRICS — USE EXACT LOOKUP VALUES:
  "In-display Fingerprint (Optical)" not "Fingerprint on display"
  "In-display Fingerprint (Ultrasonic)" not "Under-display ultrasonic fingerprint"
  "Side-mounted Fingerprint" not "Side fingerprint sensor"
  "Face Unlock" not "Face unlock" (capitalisation matters — lookup is case-sensitive)
  "Face ID" for Apple only
  Map all source descriptions to the closest exact lookup value.

UNLOCK METHODS — ALWAYS INCLUDE DEFAULTS:
  All Android phones support PIN, Pattern, Password, and Swipe by default.
  Always include: ["PIN", "Pattern", "Password", "Swipe"]
  Add biometric methods on top when explicitly supported: "Fingerprint", "Face Unlock"
  For iOS: ["Face ID", "Passcode"] or ["Touch ID", "Passcode"] — no pattern/swipe.
  Never output an empty unlock_methods array for Android phones.
  FORBIDDEN: "Smart Lock" is NOT in the lookup table — never include it.

SLOW MOTION RESOLUTIONS (video_capabilities.slow_motion_resolutions):
  ONLY populate this field when a resolution is captured at ≥120fps.
  Standard frame rates (24, 30, 60fps) are NOT slow motion — leave null.
  The field is a free-text string listing only the qualifying entries.

  Rule:
    For each resolution/fps pair in the source:
      fps < 120  → exclude entirely, do not write it
      fps ≥ 120  → include in the field
    If no resolution reaches 120fps → leave null.

  Format: "1080p@240fps, 720p@960fps"  (comma-separated, descending fps order)

  Examples:
    Source says "4K@30fps, 1080p@60fps, 1080p@120fps, 720p@960fps"
      → "1080p@120fps, 720p@960fps"                              ✅
    Source says "4K@30fps, 1080p@60fps" with no higher fps
      → null                                                      ✅
    Source says "1080p@30fps slow motion"
      → null  (30fps is not slow motion regardless of labelling)  ✅

VARIANT DEDUPLICATION:
  Two variants with identical ram_capacity + storage_capacity = same SKU.
  Keep only the lower launch_price. Delete the duplicate.

VARIANTS — RAM TYPE CONSISTENCY (variants[*].ram_type):
  RAM technology (LPDDR4X, LPDDR5, LPDDR5X) is a chipset-level decision,
  not a storage-tier decision. All variants of the same phone use identical
  RAM technology regardless of RAM/storage capacity.

  Rule:
    If ram_type is identified for ANY variant, apply the same value to ALL
    variants. Never leave ram_type null for one variant when another variant's
    type is known from the same source.

  Only use different values across variants if the source EXPLICITLY states
  different RAM technologies per tier (extremely rare — document it if seen).
  In that case, extract as stated and note the discrepancy in a comment.

VIDEO RESOLUTIONS — ONLY FROM SOURCE:
  Never infer video capabilities from camera megapixels or chipset tier.
  "4K@60fps" requires explicit source mention of 60fps at 4K — do not assume it.
  If source lists "UHD@30fps" only → rear_video_resolutions = "4K@30fps" (not "4K@30/60fps").

LTEPP IN LOCATION SERVICES:
  LTEPP = LTE Positioning Protocol (cellular-based positioning).
  Add to location_services when OEM India site lists it.
"""

_FOLDABLE_RULES = """\
FOLDABLE/FLIPPABLE PHONES:
  Extract two displays: display_index=0 (cover/outer), display_index=1 (inner/main).
  display_position: Inner display → "Primary". Cover display → "Secondary".
  body dimensions = UNFOLDED (open) state.
  Folded state → height_folded, width_folded, thickness_folded.
  NEVER average specs across the two screens.
"""

_STANDARD_RULES = """\
STANDARD PHONES:
  body.height_folded / width_folded / thickness_folded = always null.
  displays: exactly 1 entry (display_index=0, display_type='Main', display_position='Primary').
"""


def _build_system_prompt(
    brand: str,
    model_name: str,
    phone_type: str,
    has_transcript: bool,
    few_shot_block: str,
) -> str:
    """
    Assembles the system_prompt for call_gemini_json():
      1. ECD (build_ecd — dynamic per phone_type, transcript flag)
      2. Extraction rules
      3. Phone-type-specific display/folded rules
      4. Few-shot examples block
    """
    ecd = build_ecd(phone_type=phone_type, has_transcript=has_transcript)

    type_rules = (
        _FOLDABLE_RULES
        if phone_type in (PHONE_TYPE_FOLDABLE, PHONE_TYPE_FLIPPABLE)
        else _STANDARD_RULES
    )

    parts = [ecd, _EXTRACTION_RULES, type_rules]
    if few_shot_block:
        parts.append("FEW-SHOT EXAMPLES:\n\n" + few_shot_block)

    system_prompt = "\n\n".join(parts)

    logger.debug(
        "_build_system_prompt: %d chars (~%d tokens) for %r %r (phone_type=%r)",
        len(system_prompt), len(system_prompt) // 4, brand, model_name, phone_type,
    )
    return system_prompt


# ---------------------------------------------------------------------------
# Main orchestrator — run_spec_extraction()
# ---------------------------------------------------------------------------

async def run_spec_extraction(
    url_registry_id: int,
    raw_source_ids: list[int],
    raw_transcript_ids: list[int],
    brand: str,
    model_name: str,
    schema_version: str = "v2",
    pipeline_run_id: str | None = None,   # Global tracking — pipeline.pipeline_runs UUID
) -> dict:
    """
    Full Run A orchestration — spec extraction for one phone using Gemini JSON.

    Replaces langextract_run_a.run_spec_extraction_lx() for the v5 pathway.

    Flow:
      Step 0:  Gate check    — fetch_latest_validation → can_proceed
      Step 1:  Load examples for schema_version
      Step 2:  Detect phone_type from model_name
      Step 3:  insert_extraction_run (status='running', raw_transcript_ids=top3)
      Step 4:  assemble_run_a_input() → (assembled_source_string, file_map, failed_ids)
      Step 5:  Build system_prompt (ECD + rules + few-shot examples)
      Step 6:  asyncio.wait_for(call_gemini_json(...), timeout=60.0)
      Step 7:  build_spec_json(raw_output, ..., assembled_source_string)
               → (partial_json, evidence_json)
      Step 8:  _count_fields(partial_json) → (null_count, filled_count)
      Step 9:  insert_extraction_output({partial_json, evidence_json, counts})
      Step 10: update_extraction_run(status='completed', finished_at=NOW())
      Step 11: Return {success, output_id, run_id, message, failed_source_ids}

    On any exception between steps 3–10:
      update_extraction_run(status='failed', error_message=str(exc)), re-raise.

    Args:
        url_registry_id:     url_registry.url_id for this phone
        raw_source_ids:      raw_scraped_data.raw_id values to include
        raw_transcript_ids:  Up to 3 raw_transcript_ids (ranked by reliability)
        brand:               Brand name (used for ECD + logging)
        model_name:          Model name (used for phone_type detection + logging)
        schema_version:      Extraction schema version (default 'v2')

    Returns:
        {
            "success":           bool,
            "output_id":         int | None,
            "run_id":            int,
            "message":           str,
            "failed_source_ids": list[int],
        }

    Raises:
        ValueError: If gate check fails (can_proceed=False or no validation record).
        Re-raises all other exceptions after marking run as failed.
    """
    logger.info(
        "run_spec_extraction: START url_registry_id=%d brand=%r model=%r "
        "sources=%s transcripts=%s schema=%s",
        url_registry_id, brand, model_name,
        raw_source_ids, raw_transcript_ids, schema_version,
    )

    # -------------------------------------------------------------------------
    # Step 0 — Gate check (must run BEFORE insert to avoid stale run records)
    # -------------------------------------------------------------------------
    await _track(pipeline_run_id,
                 current_stage=PipelineStage.ASSEMBLING_SOURCES,
                 current_step="Checking extraction gate...")
    validation = await asyncio.to_thread(fetch_latest_validation, url_registry_id)
    if not validation or not validation.get("can_proceed"):
        raise ValueError(
            f"{_GATE_ERROR_PREFIX} for url_registry_id={url_registry_id}. "
            "Run POST /extraction/validate first and ensure can_proceed=true."
        )

    # -------------------------------------------------------------------------
    # Step 1 — Load examples (schema_version determines which module is used)
    # Run BEFORE insert so the DB records the actual version used after fallback.
    # -------------------------------------------------------------------------
    RUN_A_EXAMPLES, schema_version = _load_run_a_examples(schema_version)

    # -------------------------------------------------------------------------
    # Step 2 — Detect phone type
    # -------------------------------------------------------------------------
    phone_type = _detect_phone_type(brand, model_name)
    logger.info(
        "run_spec_extraction: url_registry_id=%d phone_type=%r",
        url_registry_id, phone_type,
    )

    # -------------------------------------------------------------------------
    # Step 3 — Create run record
    # -------------------------------------------------------------------------
    run_id = await asyncio.to_thread(
        insert_extraction_run,
        {
            "url_registry_id":           url_registry_id,
            "raw_source_ids":            raw_source_ids,
            "raw_transcript_ids":        raw_transcript_ids,
            "model_used":                EXTRACTION_MODEL,
            "extraction_schema_version": schema_version,
            "status":                    "running",
        },
    )
    logger.info("run_spec_extraction: run_id=%d created", run_id)

    failed_source_ids: list[int] = []

    try:
        # ---------------------------------------------------------------------
        # Step 4 — Assemble source content
        # Returns (combined_content, file_map, failed_ids, source_content_map, transcript_content_map).
        # source_content_map / transcript_content_map hold raw per-file content
        # (before XML wrapping) for accurate char-offset evidence attribution.
        # ---------------------------------------------------------------------
        await _track(pipeline_run_id,
                     current_stage=PipelineStage.ASSEMBLING_SOURCES,
                     current_step="Assembling source documents...")
        assembled_source_string, file_map, failed_source_ids, source_content_map, transcript_content_map = await assemble_run_a_input(
            raw_source_ids=raw_source_ids,
            raw_transcript_ids=raw_transcript_ids,
            brand=brand,
        )

        # ---------------------------------------------------------------------
        # Step 5 — Build system prompt (ECD + rules + few-shot)
        # ---------------------------------------------------------------------
        has_transcript = bool(file_map.get("transcripts"))
        examples_for_type = RUN_A_EXAMPLES.get(phone_type, RUN_A_EXAMPLES.get("standard", []))
        few_shot_block = _build_few_shot_block(examples_for_type)

        system_prompt = _build_system_prompt(
            brand=brand,
            model_name=model_name,
            phone_type=phone_type,
            has_transcript=has_transcript,
            few_shot_block=few_shot_block,
        )

        logger.info(
            "run_spec_extraction: run_id=%d launching call_gemini_json — "
            "system_prompt=%d chars, content=%d chars, total=%d chars",
            run_id,
            len(system_prompt),
            len(assembled_source_string),
            len(system_prompt) + len(assembled_source_string),
        )

        # ---------------------------------------------------------------------
        # Step 6 — Gemini extraction call with timeout
        # call_gemini_json() is already async — no to_thread() needed.
        # ---------------------------------------------------------------------
        await _track(pipeline_run_id,
                     current_stage=PipelineStage.GEMINI_EXTRACTION,
                     current_step="Sending to Gemini for extraction...")
        raw_output: dict = await asyncio.wait_for(
            call_gemini_json(
                system_prompt=system_prompt,
                user_content=assembled_source_string,
                output_schema=None,
                temperature=0.1,
            ),
            timeout=_RUN_A_TIMEOUT_SECONDS,
        )

        logger.info(
            "run_spec_extraction: run_id=%d call_gemini_json complete — "
            "%d top-level keys in output",
            run_id, len(raw_output),
        )

        # ---------------------------------------------------------------------
        # Step 7 — Build partial_json + evidence_json
        # spec_json_builder handles: first-seen-wins merge, indexed arrays,
        # extra_features/in_the_box append-all, Run C field stripping,
        # _source tag attribution, char offset computation.
        # ---------------------------------------------------------------------
        await _track(pipeline_run_id,
                     current_stage=PipelineStage.BUILDING_SPEC_JSON,
                     current_step="Building spec JSON with evidence...")
        partial_json, evidence_json = build_spec_json(
            raw_output=raw_output,
            raw_source_ids=raw_source_ids,
            raw_transcript_ids=raw_transcript_ids,
            assembled_source_string=assembled_source_string,
            source_content_map=source_content_map,
            transcript_content_map=transcript_content_map,
            url_registry_id=url_registry_id,
        )

        # ---------------------------------------------------------------------
        # Step 7b — Canonicalize field order (My_data_schema.xlsx order)
        # ---------------------------------------------------------------------
        partial_json = canonicalize_spec(partial_json)

        # ---------------------------------------------------------------------
        # Step 8 — Count null vs filled fields
        # ---------------------------------------------------------------------
        null_count, filled_count = _count_fields(partial_json)

        # ---------------------------------------------------------------------
        # Step 9 — Insert output record
        # ---------------------------------------------------------------------
        await _track(pipeline_run_id,
                     current_stage=PipelineStage.SAVING_OUTPUT,
                     current_step="Saving extraction output to database...")
        output_id = await asyncio.to_thread(
            insert_extraction_output,
            {
                "extraction_run_id":  run_id,
                "url_registry_id":    url_registry_id,
                "partial_json":       partial_json,
                "evidence_json":      evidence_json,
                "null_field_count":   null_count,
                "filled_field_count": filled_count,
            },
        )

        # ---------------------------------------------------------------------
        # Step 10 — Mark run complete
        # ---------------------------------------------------------------------
        await asyncio.to_thread(
            update_extraction_run,
            run_id,
            {
                "status":      "completed",
                "finished_at": "now()",
            },
        )

        logger.info(
            "run_spec_extraction: COMPLETE run_id=%d output_id=%d "
            "filled=%d null=%d failed_sources=%s",
            run_id, output_id, filled_count, null_count, failed_source_ids,
        )

        await _track(pipeline_run_id,
                     status="completed",
                     current_step="Done.",
                     completed_at=_now_iso())

        return {
            "success":           True,
            "output_id":         output_id,
            "run_id":            run_id,
            "message":           (
                f"Run A complete. {filled_count} fields filled, {null_count} null."
            ),
            "failed_source_ids": failed_source_ids,
        }

    except Exception as exc:
        # Mark run failed, then re-raise unchanged
        logger.exception(
            "run_spec_extraction: FAILED run_id=%d: %s", run_id, exc
        )
        await _track(pipeline_run_id,
                     status="failed",
                     current_step=f"Failed: {exc}",
                     completed_at=_now_iso(),
                     error_summary=[{"stage": "run_a", "message": str(exc)}])
        try:
            await asyncio.to_thread(
                update_extraction_run,
                run_id,
                {
                    "status":        "failed",
                    "error_message": str(exc),
                    "finished_at":   "now()",
                },
            )
        except Exception as db_exc:
            logger.error(
                "run_spec_extraction: also failed to mark run_id=%d as failed: %s",
                run_id, db_exc,
            )
        raise
