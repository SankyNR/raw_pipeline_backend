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
# Source assembler
# ---------------------------------------------------------------------------

async def assemble_run_a_input(
    raw_source_ids: list[int],
    raw_transcript_ids: list[int],
    brand: str,
) -> tuple[str, dict[str, int | list[int]], list[int]]:
    """
    Fetches all source files from Supabase Storage and concatenates them
    in a fixed order for Run A (v5: multi-transcript support).

    CONCATENATION ORDER:
      1. OEM official markdown  — structural anchor, clearest field layout
      2. GSMArena markdown
      3. Smartprix markdown
      4. DeviceSpecifications markdown
      5. Any other aggregator   — in fetch order
      99. Processed transcript(s) — contextual, always last; up to 3, in rank order

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

        file_map[site_name] = raw_id
        sections.append(
            f"--- SOURCE: {site_name} (raw_id={raw_id}) ---\n\n{content}"
        )
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
            fetched_transcript_ids.append(raw_transcript_id)
            sections.append(
                f"--- SOURCE: transcript (raw_transcript_id={raw_transcript_id}) ---\n\n"
                f"{transcript_content}"
            )
            logger.info(
                "assemble_run_a_input: fetched transcript raw_transcript_id=%d (%d chars)",
                raw_transcript_id, len(transcript_content),
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
    return combined_content, file_map, failed_source_ids


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

SAR VALUES — INDIA STANDARD: India uses 0 mm separation (tec.fptc.gov.in). \
GSMArena SAR values use EU 10 mm separation — NOT India values.

CHARGER IN BOX — INDIA UNIT: Must reflect India retail unit specifically. \
International "no charger" news ≠ India unit policy. If not stated, output null.

BAND FORMAT — ALWAYS PREFIX:
  bands_4g values must be prefixed with "B": output "B1", "B28", "B41" — never "1", "28", "41".
  bands_5g values must be prefixed with "n": output "n78", "n28", "n41" — never "78", "28", "41".
  Never output raw numbers without the prefix. Never output "Band 28" or "n 78" with a space.

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

FINGERPRINT SENSOR:
  Extract under sensors.fingerprint_sensor as free-text description.
  Examples: "In-display (Optical)", "Side-mounted", "Under-display (Ultrasonic)".
  Not a lookup field — output the type as described in source.

CHIPSET NAME — STRIP PART NUMBERS (5a):
  Extract the canonical marketing name only. Remove internal model/part number prefixes.
  Qualcomm: "Qualcomm SM7435-AB Snapdragon 7s Gen 2" → "Qualcomm Snapdragon 7s Gen 2"
  Qualcomm: "Qualcomm SM8650 Snapdragon 8 Gen 3"     → "Qualcomm Snapdragon 8 Gen 3"
  MediaTek: "MediaTek MT6985 Dimensity 9200"          → "MediaTek Dimensity 9200"
  MediaTek: "MediaTek MT6893 Dimensity 1200"          → "MediaTek Dimensity 1200"
  Apple:    "Apple A17 Pro"                           → "Apple A17 Pro" (no part number to strip)
  Rule: Remove "SM####-XX" (Qualcomm) or "MT####" (MediaTek) part numbers. Keep brand + marketing name.

CHIPSET — ALWAYS EXTRACT ALL FIELDS (5b):
  Always extract all available chipset fields from source data.
  The normalizer checks if the chipset already exists in the DB by name.
  If it exists, the DB values will override your extracted values before commit.
  If it does not exist, your extracted values create the new chipset record.
  Do NOT skip chipset extraction assuming it is already in the database.
  Extract all: chipset_name, fabrication_node, number_of_cores,
               cpu_high_performance_cores, cpu_efficiency_cores,
               cpu_performance_cores (null for dual-cluster),
               gpu_name, gpu_unit_count, gpu_unit_type, npu_details, npu_tops.

CPU ARCHITECTURE — DO NOT EXTRACT (5c):
  chipset.cpu_architecture: Leave this field NULL in extraction output.
  Site data (GSMArena, OEM sites) does not mention ARMv9.2, ARMv8-A, etc.
  This field is sourced from chipset manufacturer datasheets during ENRICHMENT — not extraction.

GPU CLOCK SPEED — DO NOT EXTRACT (5d):
  chipset.gpu_clock_speed: Leave this field NULL in extraction output.
  GPU clock speed for mobile chipsets is rarely published in consumer-facing specs.
  This field is sourced from chipset manufacturer technical documentation during ENRICHMENT.

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

SPEAKER POSITIONS — INFER FROM COUNT (5g):
  audio.speaker_positions: Must be consistent with audio.speaker_count.
  If speaker_count = 2 (stereo/dual speakers):
    Default positions = "Earpiece, Bottom"
    The earpiece doubles as a speaker in stereo configurations unless stated otherwise.
  If speaker_count = 1 (mono/single speaker):
    Default position = "Bottom"
  If source data explicitly states different positions, use those and override the default.

CHARGING VOLTAGE AND AMPERAGE — EXPLICIT ONLY (5h):
  charging.charging_voltage and charging.charging_ampere:
  Extract ONLY when explicitly stated by the manufacturer in the source data.
  DO NOT calculate, infer, or derive from wattage alone (W = V x A requires both values).
  DO NOT guess standard values (e.g. do not assume "5V/3A" for a 15W charger).
  If not explicitly stated: set null.

CPU CORE CLASSIFICATION (5i):
  chipset.cpu_high_performance_cores: The high-clock Performance cluster.
    Cortex-A78, A710, A720, X-series (X1, X2, X3, X4) = Performance cores.
  chipset.cpu_efficiency_cores: The low-power Efficiency cluster.
    Cortex-A55, A510, A520, A53 = Efficiency cores.
  chipset.cpu_performance_cores: The mid-tier cluster.
    ONLY for tri-cluster designs (Snapdragon 8 Gen 2+, Dimensity 9300+).
    NULL for all standard dual-cluster (big.LITTLE) chipsets.
  If you do not know which core type a specific model belongs to, leave the field null.
  It will be corrected in Admin UI during the first-time chipset setup.

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

REAR CAMERA SETUP — COUNT REAR LENSES ONLY (5o):
  camera_overview.rear_camera_setup counts ONLY rear-facing camera lenses:
    Valid rear lens types: Main, Ultra-wide, Telephoto, Periscope, Macro, Depth.
  NEVER count the Front lens — the front camera count goes in front_camera_setup.
  Examples:
    Main + Ultra-wide                     → "Dual"
    Main + Ultra-wide + Telephoto         → "Triple"
    Main + Ultra-wide + Telephoto + Macro → "Quad"
    Main only                             → "Single"
  If in doubt, count the lens_type entries in camera_lenses[] that are NOT "Front".

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

AUDIO CODECS — EXPLICIT SOURCE ONLY (5q):
  audio.audio_codecs: Extract ONLY codec names that are EXPLICITLY listed in the
  source documents. Do NOT infer, assume, or fill from training knowledge.

  CORRECT: Source says "Supports LDAC, aptX Adaptive, AAC" → extract those three.
  WRONG:   Source says nothing about codecs → do NOT output MP3, AAC, FLAC, etc.
           These are standard formats that all smartphones support, but they are
           not your data to add — they must come from the source.

  If the source does not list audio codecs explicitly: output [].
  The empty list is the correct answer when codec data is absent from sources.

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

VARIANTS — OEM INDIA ONLY — SOURCE-ANCHORED (5s):
  The input has source sections labeled "--- SOURCE: {site_name} (raw_id=N) ---".
  1. Find the OEM official India site section (motorola_official, samsung_official, etc.).
  2. Extract ONLY variants explicitly listed in THAT section with INR prices.
  3. Variants from sections labeled "gsmarena", "smartprix", or other aggregators → IGNORE.
  4. Output exactly the variants from the OEM India section — no more.
  SELF-CHECK: Does any variant object have evidence_text from a gsmarena/aggregator section?
  If yes → delete it. Only OEM India raw_id is valid for variants.

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

AI CAPABILITIES — STRICT GROUNDING (5z):
  ai_system: Only set if source explicitly names a branded AI platform.
    Examples: "Galaxy AI", "Apple Intelligence", "Moto AI".
    "Hello UI" is a UI SKIN, NOT an AI system → ai_system = null.
  ai_features: Only explicitly named features from source text.
    Do NOT generate from training knowledge. If source is silent → ai_features = [].
  If ai_capabilities has nothing grounded → output ai_capabilities: {}
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
        # Returns (combined_content, file_map, failed_ids).
        # combined_content is also the assembled_source_string for char offsets.
        # ---------------------------------------------------------------------
        await _track(pipeline_run_id,
                     current_stage=PipelineStage.ASSEMBLING_SOURCES,
                     current_step="Assembling source documents...")
        assembled_source_string, file_map, failed_source_ids = await assemble_run_a_input(
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
            url_registry_id=url_registry_id,
        )

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
