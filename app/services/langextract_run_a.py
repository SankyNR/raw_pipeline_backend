"""
Phase L4 — langextract_run_a.py

Full Run A orchestrator using LangExtract.

Replaces extraction_orchestrator.run_spec_extraction() for the LangExtract pathway.
The legacy orchestrator (extraction_orchestrator.py) has been deleted in Phase L7.

DESIGN
------
This module mirrors the exact flow of extraction_orchestrator.run_spec_extraction()
but replaces the Gemini JSON wrapper with lx.extract() and the evidence_utils
split with spec_json_builder.build_spec_json().

Non-negotiable contracts preserved:
  - Same gate check (fetch_latest_validation → can_proceed)
  - Same DB insert order: run record → output record → update run
  - Same return dict shape: {success, output_id, run_id, message, failed_source_ids}
  - Same failed_source_ids tracking
  - Same field counting (null_count, filled_count)

Source assembly is re-used directly from extraction_orchestrator.assemble_run_a_input()
to avoid duplicating that logic. All A-fixes (A7, A10, A11 etc.) are inherited.

Four-layer prompt (Section 7 of langextract_migration_v4.md):
  Layer 1 — BASE_TASK_DESCRIPTION  (invariant ~150 chars)
  Layer 2 — ECD via build_ecd()    (dynamic 3000–5000 chars)
  Layer 3 — REDUCED_YAML_RULES    (static ~600 chars)
  Layer 4 — RUN_SPECIFIC_RULES    (static ~300 chars, phone-type injected)

JSONL storage:
  After every successful lx.extract() call, the annotated JSONL is uploaded to:
    {brand_slug}/{model_slug}/extractions/spec_{run_id}.jsonl
  The path is persisted via update_spec_run_jsonl_path().
  HTML is NEVER stored — generated on demand by GET /approval/visualization/spec/{run_id}.

asyncio safety:
  lx.extract() is synchronous. All calls are wrapped in asyncio.to_thread().
  DB calls (synchronous) are also wrapped. FastAPI event loop is never blocked.
"""

import asyncio
import io
import logging
import os
import tempfile

import langextract as lx

from app.config.langextract_examples_run_a_v1 import RUN_A_EXAMPLES
from app.services.ecd_generator import (
    PHONE_TYPE_FOLDABLE,
    PHONE_TYPE_FLIPPABLE,
    PHONE_TYPE_STANDARD,
    build_ecd,
)
from app.services.spec_json_builder import build_spec_json
from app.services.storage_service import fetch_file_content, store_jsonl_content
from app.utils.path_builder import slugify, get_concat_order
from app.repositories.extraction_repository import (
    fetch_latest_validation,
    fetch_raw_source_rows,
    fetch_transcript_row,
    insert_extraction_run,
    insert_extraction_output,
    update_extraction_run,
    update_spec_run_jsonl_path,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# A8 — Model-name-based phone type detection (inlined from extraction_orchestrator)
# ---------------------------------------------------------------------------

_FOLDABLE_KEYWORDS: frozenset[str] = frozenset({"fold", "find n"})
_FLIPPABLE_KEYWORDS: frozenset[str] = frozenset({"flip", "razr"})


def _detect_phone_type(brand: str, model_name: str) -> str:
    """
    Detects the phone form-factor from model_name keywords rather than brand alone.

    Detection order:
      1. Flippable keywords: "flip", "razr"       → PHONE_TYPE_FLIPPABLE
      2. Foldable keywords:  "fold", "find n"     → PHONE_TYPE_FOLDABLE
      3. Default:                                  → PHONE_TYPE_STANDARD
    """
    m = model_name.lower()
    if any(kw in m for kw in _FLIPPABLE_KEYWORDS):
        return PHONE_TYPE_FLIPPABLE
    if any(kw in m for kw in _FOLDABLE_KEYWORDS):
        return PHONE_TYPE_FOLDABLE
    return PHONE_TYPE_STANDARD


# ---------------------------------------------------------------------------
# Field counter (inlined from extraction_orchestrator)
# ---------------------------------------------------------------------------

def _count_fields(obj, _path: str = "") -> tuple[int, int]:
    """
    Recursively counts null vs non-null leaf values in the partial_json dict.
    Returns (null_count, filled_count).

    Ignores nested dict/list scaffolding — only counts actual leaf values.
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
# Source assembler (inlined from extraction_orchestrator — Phase L7)
# ---------------------------------------------------------------------------

async def assemble_run_a_input(
    raw_source_ids: list[int],
    raw_transcript_id: int | None,
    brand: str,
) -> tuple[str, dict[str, int], list[int]]:
    """
    Fetches all source files from Supabase Storage and concatenates them
    in a fixed order for Run A.

    CONCATENATION ORDER (LLM schema comprehension quality):
      1. OEM official markdown  — structural anchor, clearest field layout
      2. GSMArena markdown
      3. Smartprix markdown
      4. DeviceSpecifications markdown
      5. Any other aggregator   — in fetch order
      99. Processed transcript  — contextual, always last

    For Hindi transcripts: uses translated_transcript_path if
    youtube_raw_transcript_data.translation_status = 'translation_complete',
    otherwise falls back to processed_transcript_path.

    A11 fix: Raises ValueError if two raw_scraped_data rows share the same site_name.
    A7 fix:  The "no sections" guard is checked after both scraped AND transcript fetches.

    Returns:
        combined_content:   all files concatenated with section headers
        file_map:           {"gsmarena": raw_id, "samsung_official": raw_id,
                             "transcript": raw_transcript_id}
        failed_source_ids:  list of raw_ids that failed to fetch (A10)

    Raises:
        ValueError: If raw_source_ids is empty AND raw_transcript_id is None,
                    OR if two sources share the same site_name (A11),
                    OR if ALL fetches (scraped + transcript) fail (A7).
    """
    if not raw_source_ids and raw_transcript_id is None:
        raise ValueError(
            "assemble_run_a_input: raw_source_ids is empty and raw_transcript_id is None. "
            "Nothing to assemble."
        )

    # Step 1 — Fetch rows and sort by concat priority
    source_rows = await asyncio.to_thread(fetch_raw_source_rows, raw_source_ids)
    source_rows.sort(key=lambda r: get_concat_order(r["site_name"]))

    # A11 — Guard against duplicate site_name
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

    # Step 2 — Fetch each markdown file
    sections: list[str] = []
    file_map: dict[str, int] = {}
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

    # Step 3 — Fetch transcript if provided
    if raw_transcript_id is not None:
        transcript_row = await asyncio.to_thread(
            fetch_transcript_row, raw_transcript_id
        )

        if transcript_row.get("translation_status") == "translation_complete":
            transcript_path = transcript_row.get("translated_transcript_path")
        else:
            transcript_path = transcript_row.get("processed_transcript_path")

        if transcript_path:
            try:
                transcript_content = await fetch_file_content(transcript_path)
                file_map["transcript"] = raw_transcript_id
                sections.append(
                    f"--- SOURCE: transcript (raw_transcript_id={raw_transcript_id}) ---\n\n"
                    f"{transcript_content}"
                )
                logger.info(
                    "assemble_run_a_input: fetched transcript raw_transcript_id=%d (%d chars)",
                    raw_transcript_id, len(transcript_content),
                )
            except Exception as exc:
                logger.warning(
                    "assemble_run_a_input: failed to fetch transcript "
                    "raw_transcript_id=%d: %s",
                    raw_transcript_id, exc,
                )
        else:
            logger.warning(
                "assemble_run_a_input: transcript row has no valid path "
                "(raw_transcript_id=%d, translation_status=%r)",
                raw_transcript_id, transcript_row.get("translation_status"),
            )

    # A7 — Only raise after all sources attempted
    if not sections:
        raise ValueError(
            "assemble_run_a_input: all source file fetches failed "
            f"(scraped raw_source_ids={raw_source_ids}, "
            f"transcript raw_transcript_id={raw_transcript_id})."
        )

    combined_content = "\n\n".join(sections)
    logger.info(
        "assemble_run_a_input: assembled %d source sections (%d chars). "
        "failed_source_ids=%s",
        len(sections), len(combined_content), failed_source_ids,
    )
    return combined_content, file_map, failed_source_ids


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXTRACTION_MODEL = "gemini-2.5-flash"  # change here to switch models pipeline-wide

# LangExtract chunk size.  25 000 chars keeps typical Firecrawl + OEM + transcript
# within a single chunk, preventing intra-chunk char_interval offsets.
_MAX_CHAR_BUFFER = 25_000

# Threshold for logging a "very long prompt" warning.
_PROMPT_CHAR_WARNING_THRESHOLD = 30_000

_GATE_ERROR_PREFIX = "Pre-extraction validation not passed"

# MED-3: Timeout for asyncio.wait_for() around lx.extract() to prevent indefinite
# event-loop stalls from hung Gemini calls. 300s = 5 minutes, generous for large inputs.
_LX_EXTRACT_TIMEOUT_SECONDS: int = 300


# ---------------------------------------------------------------------------
# Section 7 — Four-layer prompt
# ---------------------------------------------------------------------------

# Layer 1 — Base task description (invariant)
_LAYER_1_BASE_TASK = (
    "Extract ALL available mobile phone specifications from the provided source documents. "
    "Output one Extraction object per schema section. Use null for missing values. "
    "Do not guess. Do not derive calculated fields. "
    "Use verbatim text from the source as extraction_text."
)

# Layer 3 — Reduced YAML rules (static; enum lists removed since examples teach those)
_LAYER_3_REDUCED_YAML_RULES = """\
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

FORBIDDEN — NEVER EXTRACT (computed by Run C):
  displays[*].ppi            → computed: sqrt(h²+w²) / size_inch
  camera_lenses[*].sensor_size_decimal → computed: 1.0 / sensor_size_denominator
"""

# Layer 4 — Run-specific rules (phone-type instructions injected at build time)
_LAYER_4_FOLDABLE_RULES = """\
FOLDABLE/FLIPPABLE PHONES:
  Extract two displays: display_index=0 (cover/outer), display_index=1 (inner/main).
  body dimensions = UNFOLDED (open) state.
  Folded state → length_folded, breadth_folded, height_folded.
  NEVER average specs across the two screens.
"""

_LAYER_4_STANDARD_RULES = """\
STANDARD PHONES:
  body.length_folded / breadth_folded / height_folded = always null.
  displays: exactly 1 entry (display_index=0, display_type='Main').
"""


def _build_run_a_prompt(brand: str, model_name: str, has_transcript: bool) -> str:
    """
    Assembles the four-layer prompt_description for lx.extract().

    Layer 1: Base task statement
    Layer 2: Full ECD (build_ecd — dynamic per phone_type, transcript flag)
    Layer 3: Reduced YAML extraction rules
    Layer 4: Phone-type-specific display/folded dimension rules

    Total target: ~4,500–6,500 chars (~1,100–1,600 tokens).
    Warns if > _PROMPT_CHAR_WARNING_THRESHOLD.
    """
    phone_type = _detect_phone_type(brand, model_name)
    ecd = build_ecd(phone_type=phone_type, has_transcript=has_transcript)

    layer_4 = (
        _LAYER_4_FOLDABLE_RULES
        if phone_type in (PHONE_TYPE_FOLDABLE, PHONE_TYPE_FLIPPABLE)
        else _LAYER_4_STANDARD_RULES
    )

    prompt = "\n\n".join([
        _LAYER_1_BASE_TASK,
        ecd,
        _LAYER_3_REDUCED_YAML_RULES,
        layer_4,
    ])

    if len(prompt) > _PROMPT_CHAR_WARNING_THRESHOLD:
        logger.warning(
            "_build_run_a_prompt: prompt is long (%d chars, ~%d tokens). "
            "Consider trimming ECD or rules to keep within max_char_buffer.",
            len(prompt), len(prompt) // 4,
        )
    else:
        logger.debug("_build_run_a_prompt: %d chars (~%d tokens) for %r %r",
                     len(prompt), len(prompt) // 4, brand, model_name)

    return prompt


# ---------------------------------------------------------------------------
# JSONL storage helper
# ---------------------------------------------------------------------------

def _save_run_a_jsonl_sync(
    lx_result,
    brand: str,
    model_name: str,
    run_id: int,
) -> str:
    """
    Synchronous half of JSONL storage (runs inside asyncio.to_thread).

    lx.io.save_annotated_documents() writes to a tempfile; we read it back
    as a string and return it for the async wrapper to upload.

    Returns:
        jsonl_text — utf-8 string content of the JSONL file.

    Raises:
        RuntimeError on any IO failure.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_name = f"spec_{run_id}"
            lx.io.save_annotated_documents(
                [lx_result],
                output_name=run_name,
                output_dir=tmpdir,
            )
            jsonl_path_on_disk = os.path.join(tmpdir, f"{run_name}.jsonl")
            if not os.path.exists(jsonl_path_on_disk):
                # Some LangExtract versions append .jsonl automatically, some don't
                candidate = jsonl_path_on_disk + ".jsonl"
                if os.path.exists(candidate):
                    jsonl_path_on_disk = candidate
                else:
                    # List tmpdir to help debugging
                    files = os.listdir(tmpdir)
                    raise RuntimeError(
                        f"_save_run_a_jsonl_sync: expected JSONL at "
                        f"{jsonl_path_on_disk!r} — not found. "
                        f"tmpdir contents: {files}"
                    )
            with open(jsonl_path_on_disk, "r", encoding="utf-8") as fh:
                return fh.read()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"_save_run_a_jsonl_sync: failed to generate JSONL for run_id={run_id}: {exc}"
        ) from exc


async def _store_run_a_jsonl(
    lx_result,
    brand: str,
    model_name: str,
    run_id: int,
) -> str:
    """
    Generates JSONL from lx_result and uploads to Supabase Storage.

    Storage path: {brand_slug}/{model_slug}/extractions/spec_{run_id}.jsonl

    Returns the storage path (persisted to spec_extraction_runs.jsonl_path).

    Note: JSONL generation is synchronous (involves disk I/O) — runs in thread.
    Upload is async (store_jsonl_content). This is the correct split.
    """
    brand_slug = slugify(brand)
    model_slug = slugify(model_name)
    storage_path = f"{brand_slug}/{model_slug}/extractions/spec_{run_id}.jsonl"

    jsonl_text = await asyncio.to_thread(
        _save_run_a_jsonl_sync, lx_result, brand, model_name, run_id
    )

    await store_jsonl_content(storage_path, jsonl_text)
    logger.info(
        "_store_run_a_jsonl: uploaded %d chars to %s",
        len(jsonl_text), storage_path,
    )
    return storage_path


# ---------------------------------------------------------------------------
# Source section offset builder
# ---------------------------------------------------------------------------

def _build_source_section_offsets(
    combined_content: str,
    file_map: dict[str, int],
) -> dict[str, tuple[int, int]]:
    """
    Builds {site_name: (start_char, end_char)} offsets in combined_content.

    These offsets let spec_json_builder.build_spec_json() attribute each
    char_interval to the correct raw_id (OEM vs GSMArena vs transcript).

    The section header format (from assemble_run_a_input) is either:
        "--- SOURCE: {site_name} (raw_id={raw_id}) ---"
        "--- SOURCE: transcript (raw_transcript_id={id}) ---"

    We find each header position and compute the slice [header_end : next_header_start].

    Returns:
        {site_name: (content_start, content_end)}

    If a site_name from file_map isn't found in combined_content, logs a warning
    and omits that site — evidence attribution will fall back gracefully.
    """
    offsets: dict[str, tuple[int, int]] = {}

    # Build ordered list of (start_pos, site_name) by scanning for each header
    positions: list[tuple[int, str]] = []
    for site_name in file_map:
        if site_name == "transcript":
            # Header uses raw_transcript_id= not raw_id=
            search_prefix = f"--- SOURCE: transcript"
        else:
            search_prefix = f"--- SOURCE: {site_name}"

        idx = combined_content.find(search_prefix)
        if idx == -1:
            logger.warning(
                "_build_source_section_offsets: site_name=%r not found in combined_content. "
                "Evidence attribution for this source will fall back.",
                site_name,
            )
            continue

        # Advance past the header line (ends at \n\n)
        header_end = combined_content.find("\n\n", idx)
        content_start = header_end + 2 if header_end != -1 else idx + len(search_prefix)
        positions.append((content_start, site_name))

    # Sort by position in the merged string
    positions.sort(key=lambda t: t[0])

    # Assign end = start of next section (or end of string)
    for i, (content_start, site_name) in enumerate(positions):
        if i + 1 < len(positions):
            content_end = positions[i + 1][0] - 1
        else:
            content_end = len(combined_content) - 1
        offsets[site_name] = (content_start, content_end)

    logger.debug(
        "_build_source_section_offsets: resolved %d/%d site offsets: %s",
        len(offsets), len(file_map),
        {k: (v[0], v[1]) for k, v in offsets.items()},
    )
    return offsets


# ---------------------------------------------------------------------------
# Main orchestrator — run_spec_extraction_lx()
# ---------------------------------------------------------------------------

async def run_spec_extraction_lx(
    url_registry_id: int,
    raw_source_ids: list[int],
    raw_transcript_id: int | None,
    brand: str,
    model_name: str,
    schema_version: str = "v1",
) -> dict:
    """
    Full Run A orchestration — spec extraction for one phone using LangExtract.

    Replaces extraction_orchestrator.run_spec_extraction() for the LangExtract pathway.
    This is now the sole Run A implementation (extraction_orchestrator deleted in L7).

    Flow:
      0. Gate check    — fetch_latest_validation → can_proceed
      1. DB insert     — insert_extraction_run (status='running')
      2. Assemble      — assemble_run_a_input (local — inlined in Phase L7)
      3. Prompt        — _build_run_a_prompt (four-layer)
      4. Extract       — asyncio.to_thread(lx.extract, ...)
      5. Build JSON    — build_spec_json (from spec_json_builder)
      6. JSONL upload  — _store_run_a_jsonl → Supabase Storage
      7. DB output     — insert_extraction_output
      8. DB update     — update_extraction_run(completed) + update_spec_run_jsonl_path
      9. Return        — {success, output_id, run_id, message, failed_source_ids}

    On any exception between steps 1–8:
      - update_extraction_run(status='failed', error_message=...) is attempted
      - Exception is re-raised to the caller (FastAPI → 500 response)

    Args:
        url_registry_id:   url_registry.url_id for this phone
        raw_source_ids:    raw_scraped_data.raw_id values to include
        raw_transcript_id: youtube_raw_transcript_data.raw_transcript_id, or None
        brand:             Brand name (used for ECD + storage path)
        model_name:        Model name (used for phone_type detection + storage path)
        schema_version:    Extraction schema version (default 'v1')

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
        "run_spec_extraction_lx: START url_registry_id=%d brand=%r model=%r "
        "sources=%s transcript=%s schema=%s",
        url_registry_id, brand, model_name,
        raw_source_ids, raw_transcript_id, schema_version,
    )

    # -------------------------------------------------------------------------
    # Step 0 — Gate check (A2 fix: run before insert to avoid stale run records)
    # Must run BEFORE insert_extraction_run — do not create a record for blocked runs
    # -------------------------------------------------------------------------
    validation = await asyncio.to_thread(fetch_latest_validation, url_registry_id)
    if not validation or not validation.get("can_proceed"):
        raise ValueError(
            f"{_GATE_ERROR_PREFIX} for url_registry_id={url_registry_id}. "
            "Run POST /extraction/validate first and ensure can_proceed=true."
        )

    # -------------------------------------------------------------------------
    # Step 1 — Create run record
    # -------------------------------------------------------------------------
    run_id = await asyncio.to_thread(
        insert_extraction_run,
        {
            "url_registry_id":           url_registry_id,
            "raw_source_ids":            raw_source_ids,
            "raw_transcript_id":         raw_transcript_id,
            "model_used":                EXTRACTION_MODEL,
            "extraction_schema_version": schema_version,
            "status":                    "running",
        },
    )
    logger.info("run_spec_extraction_lx: run_id=%d created", run_id)

    failed_source_ids: list[int] = []

    try:
        # ---------------------------------------------------------------------
        # Step 2 — Assemble source content
        # assemble_run_a_input is now local (inlined Phase L7).
        # A-series fixes (A7 no-sections guard, A10 failed_ids, A11 dup guard) preserved.
        # Returns (combined_content, file_map, failed_ids).
        # ---------------------------------------------------------------------
        combined_content, file_map, failed_source_ids = await assemble_run_a_input(
            raw_source_ids=raw_source_ids,
            raw_transcript_id=raw_transcript_id,
            brand=brand,
        )

        # Build char-position offset map for evidence attribution in spec_json_builder
        source_section_offsets = _build_source_section_offsets(combined_content, file_map)

        # ---------------------------------------------------------------------
        # Step 3 — Build four-layer prompt
        # ---------------------------------------------------------------------
        has_transcript = raw_transcript_id is not None
        prompt = _build_run_a_prompt(
            brand=brand,
            model_name=model_name,
            has_transcript=has_transcript,
        )

        # Log total input size estimate
        total_chars = len(prompt) + len(combined_content)
        logger.info(
            "run_spec_extraction_lx: run_id=%d launching lx.extract — "
            "prompt=%d chars, content=%d chars, total=%d chars",
            run_id, len(prompt), len(combined_content), total_chars,
        )

        # ---------------------------------------------------------------------
        # Step 4 — LangExtract call (synchronous — wrapped in asyncio.to_thread)
        # lx.extract() uses threading internally; to_thread() prevents blocking
        # the FastAPI event loop. Two concurrent requests → two threads → safe.
        # ---------------------------------------------------------------------
        lx_result = await asyncio.wait_for(
            asyncio.to_thread(
                lx.extract,
                combined_content,       # positional: text_or_documents
                prompt_description=prompt,
                examples=RUN_A_EXAMPLES,
                model_id=EXTRACTION_MODEL,
                extraction_passes=1,
                max_workers=10,
                max_char_buffer=_MAX_CHAR_BUFFER,
            ),
            timeout=_LX_EXTRACT_TIMEOUT_SECONDS,
        )

        logger.info(
            "run_spec_extraction_lx: run_id=%d lx.extract complete — "
            "%d extractions returned",
            run_id, len(lx_result.extractions),
        )

        # Monitor ungrounded rate (char_interval=None) before build_spec_json
        ungrounded = sum(1 for e in lx_result.extractions if e.char_interval is None)
        if ungrounded:
            ungrounded_pct = (ungrounded / max(len(lx_result.extractions), 1)) * 100
            logger.warning(
                "run_spec_extraction_lx: run_id=%d — %d/%d extractions ungrounded "
                "(char_interval=None, %.0f%%). "
                "Review example quality if > 20%%.",
                run_id, ungrounded, len(lx_result.extractions), ungrounded_pct,
            )

        # ---------------------------------------------------------------------
        # Step 5 — Build partial_json + evidence_json
        # spec_json_builder handles: first-seen-wins merge, indexed arrays,
        # extra_feature/in_the_box append-all, Run C field stripping,
        # char_interval attribution, array completeness warnings.
        # ---------------------------------------------------------------------
        partial_json, evidence_json = build_spec_json(
            extractions=lx_result.extractions,
            source_text=combined_content,
            raw_source_ids=raw_source_ids,
            raw_transcript_id=raw_transcript_id,
            source_section_offsets=source_section_offsets,  # {site_name: (start, end)}
            raw_source_ids_by_site=file_map,               # {site_name: raw_id}
            url_registry_id=url_registry_id,
        )

        # ---------------------------------------------------------------------
        # Step 6 — Save JSONL for on-demand visualization
        # Failure is non-fatal: JSONL upload error logs a warning, run continues.
        # The run record will have jsonl_path=None if upload fails — visualization
        # tab will show "not available" in the UI (handled by L6 endpoint).
        # ---------------------------------------------------------------------
        jsonl_path: str | None = None
        try:
            jsonl_path = await _store_run_a_jsonl(
                lx_result=lx_result,
                brand=brand,
                model_name=model_name,
                run_id=run_id,
            )
        except Exception as jsonl_exc:
            logger.warning(
                "run_spec_extraction_lx: run_id=%d JSONL upload failed "
                "(non-fatal — run continues): %s",
                run_id, jsonl_exc,
            )

        # ---------------------------------------------------------------------
        # Step 7 — Count fields and insert output
        # ---------------------------------------------------------------------
        null_count, filled_count = _count_fields(partial_json)

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
        # Step 8 — Mark run complete + persist jsonl_path
        # Two separate DB calls: update_extraction_run for status/timestamps,
        # update_spec_run_jsonl_path for the JSONL path (separate column).
        # update_spec_run_jsonl_path is a no-op if jsonl_path is None.
        # ---------------------------------------------------------------------
        await asyncio.to_thread(
            update_extraction_run,
            run_id,
            {
                "status":      "completed",
                "finished_at": "now()",
            },
        )

        if jsonl_path is not None:
            await asyncio.to_thread(
                update_spec_run_jsonl_path, run_id, jsonl_path
            )

        logger.info(
            "run_spec_extraction_lx: COMPLETE run_id=%d output_id=%d "
            "filled=%d null=%d failed_sources=%s jsonl_path=%s",
            run_id, output_id, filled_count, null_count,
            failed_source_ids, jsonl_path,
        )
        return {
            "success":           True,
            "output_id":         output_id,
            "run_id":            run_id,
            "message":           (
                f"Run A (LX) complete. {filled_count} fields filled, "
                f"{null_count} null."
            ),
            "failed_source_ids": failed_source_ids,
        }

    except Exception as exc:
        # Mark run failed, then re-raise unchanged
        logger.exception(
            "run_spec_extraction_lx: FAILED run_id=%d: %s", run_id, exc
        )
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
                "run_spec_extraction_lx: also failed to mark run_id=%d as failed: %s",
                run_id, db_exc,
            )
        raise
