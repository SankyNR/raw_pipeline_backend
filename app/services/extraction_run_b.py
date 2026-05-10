"""
Phase 6 — extraction_run_b.py

Run B orchestrator (v5 — custom Gemini JSON extraction).

Replaces langextract_run_b.py. Uses call_gemini_json() with a Pydantic-backed
output schema instead of lx.extract(). One Gemini call per transcript.

DESIGN
------
  run_experience_extraction_batch()   ← public entry point
    Step 1: Gate check (once for the whole batch)
    Step 2: Supersede existing active rows via supersede_experiences()
    Step 3: Launch all _run_single_transcript() tasks via asyncio.gather
    Step 4: Log batch summary + return results

  _run_single_transcript()            ← semaphore-gated per-transcript task
    Step 1:  Load examples for schema_version
    Step 2:  insert_experience_run (status='running')
    Step 3:  fetch_transcript_row → resolve path
    Step 4:  fetch_file_content → truncate at _MAX_TRANSCRIPT_CHARS
    Step 5:  async with _RUN_B_SEMAPHORE: call_gemini_json (45s timeout)
    Step 6:  _parse_experiences() → (passing, filtered_count)
    Step 7:  fetch_experience_category_map()
    Step 8:  bulk_insert_phone_experiences()
    Step 9:  update_experience_run(status='completed')

Supersede order (CRITICAL):
  gate_check → supersede → batch_extract
  Supersede MUST NOT execute before gate validation.  Otherwise, approved data
  may be silently lost if the gate fails after supersede.

Semaphore:
  _RUN_B_SEMAPHORE = asyncio.Semaphore(5)
  Acquired only around call_gemini_json() — not around DB operations.

Timeouts:
  Run B: 45s per transcript (was 300s for LangExtract AFC chains).

Metrics:
  Run B uses list-based metrics: experiences_extracted and experiences_filtered.

asyncio safety:
  call_gemini_json() is async. DB calls (sync) are wrapped in asyncio.to_thread().
  The FastAPI event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, List

from pydantic import BaseModel

from app.services.gemini_client import EXTRACTION_MODEL, call_gemini_json
from app.services.storage_service import fetch_file_content
from app.core.constants import GATE_ERROR_PREFIX as _GATE_ERROR_PREFIX, PipelineStage
from app.repositories.extraction_repository import (
    fetch_latest_validation,
    fetch_transcript_row,
    fetch_experience_category_map,
    insert_experience_run,
    update_experience_run,
    bulk_insert_phone_experiences,
    supersede_experiences,
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
    Swallows all exceptions — a tracking failure must never crash Run B.
    pipeline_run_id is the UUID from pipeline.pipeline_runs.
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
# Constants
# ---------------------------------------------------------------------------

# Caps concurrent call_gemini_json() calls per FastAPI process.
# At 5: a phone with 10 transcripts launches 2 waves of 5, preventing 429s.
# Tune upward (to 8) only after 50+ batch runs with no rate-limit errors.
_RUN_B_SEMAPHORE: asyncio.Semaphore = asyncio.Semaphore(5)

# Caps concurrent Supabase Storage fetch calls.
# Prevents HTTP/2 connection pool exhaustion when 10 transcripts fetch simultaneously.
_RUN_B_STORAGE_SEMAPHORE: asyncio.Semaphore = asyncio.Semaphore(3)

# Max transcript characters passed to Gemini. Long transcripts risk token limits.
# Matches the legacy _MAX_TRANSCRIPT_CHARS in langextract_run_b.py.
_MAX_TRANSCRIPT_CHARS: int = 120_000

# Confidence threshold — entries below this are discarded before DB insert.
_MIN_CONFIDENCE: float = 0.50

# Per-transcript Gemini call timeout (seconds).
_RUN_B_TIMEOUT_SECONDS: float = 45.0


# ---------------------------------------------------------------------------
# Output schema for call_gemini_json()
# ---------------------------------------------------------------------------

class ExperienceItem(BaseModel):
    """
    Pydantic schema for a single extracted experience.

    This is the output_schema passed to call_gemini_json() for Run B.
    Gemini JSON mode enforces this structure via response_schema.

    Fields:
        experience_text:  1–3 sentence neutral third-person summary, present tense.
                          Written as: what would a user type to find phones with this quality?
        sentiment:        Positive | Negative | Neutral | Mixed
        evidence_quote:   Exact verbatim sentence(s) from the transcript. No paraphrasing.
                          Null if no clear verbatim evidence exists.
        category:         One of the 16 valid categories.
        confidence:       Float 0.00–1.00.
    """
    experience_text: str
    sentiment: str
    evidence_quote: str | None = None
    category: str
    confidence: float


class RunBExtractionSchema(BaseModel):
    """
    Top-level wrapper: Gemini returns a list of experiences.
    Using a wrapper model is required because call_gemini_json() needs a
    BaseModel subclass — it cannot accept a raw list type.
    """
    experiences: List[ExperienceItem]


class AggregatedExperienceItem(BaseModel):
    """One final experience produced by Stage 2 aggregation. All fields required."""
    experience_text: str
    sentiment: str
    evidence_quote: str                      # Required — never null in Stage 2
    category: str
    source_transcript_count: int             # Distinct candidates merged into this
    representative_raw_transcript_id: int    # Transcript that provided the evidence_quote


class RunBStage2Schema(BaseModel):
    """Top-level wrapper for Stage 2 aggregation output."""
    experiences: list[AggregatedExperienceItem]


# ---------------------------------------------------------------------------
# Valid categories
# ---------------------------------------------------------------------------

_VALID_CATEGORIES: frozenset[str] = frozenset({
    "Thermal", "Camera", "Battery Life", "Charging Speed", "Display",
    "Performance", "Audio", "Build Quality", "Software", "Gaming",
    "Call Quality", "Haptics", "Connectivity", "In the Box",
    "Durability", "Overall",
})

# Case-insensitive normalisation: "battery life" → "Battery Life"
# Using lowered map avoids .title() bug ("In The Box" ≠ "In the Box").
_CATEGORY_NORMALISE: dict[str, str] = {c.lower(): c for c in _VALID_CATEGORIES}


# ---------------------------------------------------------------------------
# System prompt (EXPERIENCE_SYSTEM_PROMPT)
# ---------------------------------------------------------------------------

EXPERIENCE_SYSTEM_PROMPT = """\
Extract subjective experiential observations from a YouTube phone review transcript.
This is Stage 1. Your output feeds Stage 2 which deduplicates across all transcripts.

AIM: 8–12 distinct observations. HARD MAXIMUM: 15. Never exceed 15.
Quality over quantity. If you find fewer than 8 worthy observations, output only those.

=== EXTRACT ONLY — real subjective experiences ===
Things the reviewer personally observed, felt, or experienced during actual use:
  "battery easily lasts a full day with heavy usage"
  "photos in low light are surprisingly sharp"
  "display becomes warm during extended gaming sessions"
  "speaker sounds loud but slightly tinny at maximum volume"

=== NEVER EXTRACT ===

RULE 1 — NO SPEC FACTS
A spec fact is an objective property of the hardware/software, true for every
unit of this phone regardless of how it is used.

DO NOT EXTRACT (spec facts):
  "phone has a 5000mAh battery"             → spec
  "supports Wi-Fi 6 and NFC"                → spec
  "has IP68 rating"                         → spec
  "comes with 68W charger in box"           → spec
  "supports LDAC and LHDC codecs"           → spec
  "has Gorilla Glass 5"                     → spec
  "144Hz refresh rate display"              → spec
  "runs Snapdragon 7s Gen 2"                → spec
  "supports 4K video at 30fps"              → spec

DO EXTRACT (subjective experience from a spec):
  "the IP68 rating gave real peace of mind using it in the rain" → EXTRACT
  "the 68W charging is noticeably fast — full charge in under an hour" → EXTRACT
  "the 144Hz display makes scrolling feel exceptionally fluid" → EXTRACT

RULE 2 — NO COMPARISONS WITH OTHER PHONES
Do not extract any statement that names a competing brand or model.

DO NOT EXTRACT:
  "battery is better than OnePlus"             → comparison
  "Moto beats Samsung in camera quality"       → comparison
  "OnePlus has a better vibration motor"       → about competitor — skip
  "display is brighter than the Nord CE4"      → comparison

DO EXTRACT (no competitor named):
  "display is bright enough to use in direct sunlight" → EXTRACT

RULE 3 — ATOMIC OBSERVATIONS ONLY
One entry = one distinct aspect of the phone's real-world experience.

DO NOT EXTRACT (multiple signals bundled):
  "the design, display, battery and camera all impressed me" → 4 signals, skip
  "solid phone with good connectivity and smooth performance" → too vague, skip

Camera has many sub-aspects — each is a separate observation:
  "portrait mode captures excellent background blur"      → EXTRACT (portrait)
  "low-light photos retain detail with minimal noise"     → EXTRACT (low light)
  "video stabilisation is smooth while walking outdoors"  → EXTRACT (video)

RULE 4 — NO GENERIC STATEMENTS
Drop statements under 10 words or with no concrete claim.

DO NOT EXTRACT:
  "overall nice phone"                → generic
  "performance is decent"             → vague
  "good for the price"                → no specific aspect

DO EXTRACT:
  "app launches feel instant with no stutter during typical daily use" → EXTRACT

=== EVIDENCE QUOTE ===
Verbatim sentence(s) from the transcript directly supporting the observation.
Copy exactly — do not paraphrase or construct a composite.
Set to null ONLY if no single sentence clearly supports the observation.

=== CATEGORIES ===
Thermal | Camera | Battery Life | Charging Speed | Display | Performance |
Audio | Build Quality | Software | Gaming | Call Quality | Haptics |
Connectivity | In the Box | Durability | Overall
"""


_STAGE2_TIMEOUT_SECONDS: float = 120.0
_STAGE2_MAX_INPUT_CANDIDATES: int = 200

_STAGE2_SYSTEM_PROMPT = """\
You receive Stage 1 candidate experiences extracted from multiple YouTube phone review
transcripts of the SAME phone. Aggregate, deduplicate, filter, and produce a final
high-quality set.

=== MANDATORY RULES ===

RULE 1 — SEMANTIC DEDUPLICATION
Merge candidates that express the same underlying observation, even if worded differently.
Produce ONE canonical experience per semantic cluster.

MERGE THESE (same idea, different words):
  [CANDIDATE] transcript=23: "Battery lasted 7 hours in my testing run"
  [CANDIDATE] transcript=24: "I got 7 to 8 hours of screen on time"
  [CANDIDATE] transcript=26: "Easily lasts all day with moderate usage"
  [CANDIDATE] transcript=29: "Over a day of battery with typical use"
  → ONE experience: "Battery provides 7–8 hours of screen-on time under typical usage."
  → source_transcript_count = 4

KEEP SEPARATE (genuinely different sub-aspects):
  "Portrait mode edge detection is precise and sharp"          → keep separate
  "Low-light photos retain detail with minimal noise"          → keep separate
  "Video stabilisation is smooth while walking outdoors"       → keep separate
  These are distinct experiential aspects of the camera — not duplicates.

RULE 2 — CANONICAL REWRITING
Write experience_text as a neutral, precise, present-tense third-person statement.
Use the most specific and informative phrasing from the candidate cluster.
Avoid vague phrases: "phone is good", "overall balanced", "quite decent".
Prefer specific observable claims grounded in the best evidence quote.

RULE 3 — EVIDENCE QUOTE (REQUIRED — NEVER NULL)
Select ONE verbatim quote from the input candidates.
NEVER generate a new quote. Copy it character-for-character from the input.
representative_raw_transcript_id must match the transcript that provided the quote.
If NO candidate in a cluster has an evidence_quote, DROP the entire cluster.

RULE 4 — DROP SPEC FACTS
Drop any candidate whose core content is an objective specification.
A spec fact is equally true for every unit of this phone regardless of usage.

DROP (spec facts):
  "Phone supports Wi-Fi 6"                  → DROP
  "Has IP68 rating"                         → DROP
  "Includes 68W charger in box"             → DROP
  "Supports LDAC codec"                     → DROP
  "Has Gorilla Glass 5"                     → DROP
  "Phone has 5000mAh battery"               → DROP
  "Supports 4K video recording"             → DROP
  "3 years of OS update guarantee"          → DROP

KEEP (spec fact as subjective experience):
  "IP68 protection gave real peace of mind using it in rain" → KEEP
  "Fast charging fills the battery in under an hour"         → KEEP

RULE 5 — DROP COMPARISONS
Drop any candidate naming another phone brand or model.
  "Better than OnePlus"                    → DROP
  "Motorola beats Samsung in battery"      → DROP
  "OnePlus has superior haptics"           → DROP (about competitor)

RULE 6 — DROP WEAK ENTRIES
Drop entries with fewer than 10 words in experience_text.
Drop non-atomic bundles (multiple unrelated aspects combined).

RULE 7 — CATEGORY CAP
Maximum 6 experiences per category.
If a category has more clusters than 6, keep the 6 most distinct and informative.

RULE 8 — TOTAL OUTPUT CAP
Produce 15–25 final experiences total.
Prioritise breadth of category coverage over deep redundancy in one category.
If fewer than 15 worthy observations exist, output only those — do not pad.

RULE 9 — source_transcript_count
Count distinct raw_transcript_ids from the input candidates merged into this cluster.
Minimum: 1. A single specific observation from one high-quality transcript is valid.

=== INPUT FORMAT ===
[CANDIDATE] transcript={raw_transcript_id} category={category} conf={confidence}
experience_text: {text}
evidence_quote: {verbatim quote or null}

=== OUTPUT FORMAT ===
Return JSON with key "experiences" (list). Each item must have:
  experience_text (str), sentiment (str), evidence_quote (str — NOT null),
  category (str), source_transcript_count (int ≥ 1),
  representative_raw_transcript_id (int)
"""


# ---------------------------------------------------------------------------
# Schema version → examples module mapping
# ---------------------------------------------------------------------------

_EXAMPLES_MODULE_MAP_B: dict[str, str] = {
    "v1": "app.config.extraction_examples_run_b_v1",
    # v2 uses dedicated Stage 1 examples created in L8.2 Phase 6.
    "v2": "app.config.extraction_examples_run_b_v2_stage1",
}

_DEFAULT_SCHEMA_VERSION_B = "v2"


@functools.lru_cache(maxsize=None)
def _load_run_b_examples(schema_version: str):
    """
    Loads RUN_B_EXAMPLES from the module for schema_version.
    Falls back to _DEFAULT_SCHEMA_VERSION_B with a warning if not found.
    Cached per schema_version (lru_cache eliminates redundant imports per batch).

    Returns:
        (RUN_B_EXAMPLES, actual_schema_version_used)
    """
    import importlib
    module_path = _EXAMPLES_MODULE_MAP_B.get(schema_version)
    if module_path is None:
        logger.warning(
            "_load_run_b_examples: schema_version=%r not in _EXAMPLES_MODULE_MAP_B. "
            "Falling back to %r.",
            schema_version, _DEFAULT_SCHEMA_VERSION_B,
        )
        module_path = _EXAMPLES_MODULE_MAP_B[_DEFAULT_SCHEMA_VERSION_B]
        schema_version = _DEFAULT_SCHEMA_VERSION_B
    mod = importlib.import_module(module_path)
    # v2 examples file uses RUN_B_STAGE1_EXAMPLES; v1 uses RUN_B_EXAMPLES
    examples = (
        getattr(mod, "RUN_B_STAGE1_EXAMPLES", None)
        or getattr(mod, "RUN_B_EXAMPLES", [])
    )
    return examples, schema_version


# ---------------------------------------------------------------------------
# Few-shot prompt block builder
# ---------------------------------------------------------------------------

def _build_few_shot_block(examples: list[dict]) -> str:
    """
    Formats a list of {input_excerpt, expected_output} example dicts into a
    few-shot text block for inclusion in the user_content prompt preamble.
    """
    if not examples:
        return ""

    import json
    blocks: list[str] = []
    for i, ex in enumerate(examples, 1):
        inp = ex.get("input_excerpt", ex.get("input", ""))
        out = ex.get("expected_output", ex.get("output", []))
        desc = ex.get("description", "")
        header = f"EXAMPLE {i}" + (f" ({desc})" if desc else "")
        blocks.append(
            f"{header} INPUT:\n{inp}\n\n"
            f"{header} OUTPUT:\n{json.dumps({'experiences': out}, indent=2)}"
        )
    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Experience parser: Gemini output → validated rows
# ---------------------------------------------------------------------------

def _parse_experiences(
    raw_output: dict,
    exp_run_id: int,
) -> tuple[list[dict], int]:
    """
    Converts the raw Gemini JSON output dict → validated experience rows.

    Post-processing pipeline:
      1. Extract experiences list from raw_output["experiences"]
      2. Confidence filter: entries < _MIN_CONFIDENCE are discarded
      3. Category normalisation: case-insensitive, remapped to canonical strings
      4. Exact-text deduplication on normalised experience_text (within this run)

    Args:
        raw_output:   Dict returned by call_gemini_json(output_schema=RunBExtractionSchema).
                      Expected shape: {"experiences": [...]}
        exp_run_id:   Used for structured log messages only.

    Returns:
        (passing, filtered_count)
        passing        — list of validated experience dicts (category as string, no category_id)
        filtered_count — number discarded below _MIN_CONFIDENCE
    """
    raw_list: list[Any] = raw_output.get("experiences", [])
    if not isinstance(raw_list, list):
        logger.warning(
            "_parse_experiences: exp_run_id=%d — raw_output['experiences'] is not a list "
            "(got %s). Treating as empty.",
            exp_run_id, type(raw_list).__name__,
        )
        raw_list = []

    logger.debug(
        "_parse_experiences: exp_run_id=%d — %d raw items from Gemini",
        exp_run_id, len(raw_list),
    )

    # Step 1+2 — Confidence filter
    passing: list[dict] = []
    filtered_count = 0

    for item in raw_list:
        if not isinstance(item, dict):
            filtered_count += 1
            continue

        try:
            confidence = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        item["confidence"] = confidence

        if confidence < _MIN_CONFIDENCE:
            filtered_count += 1
            continue

        # Step 3 — Category normalisation
        raw_cat = str(item.get("category") or "").lower().strip()
        normalised_cat = _CATEGORY_NORMALISE.get(raw_cat)
        if normalised_cat:
            item["category"] = normalised_cat
        else:
            logger.warning(
                "_parse_experiences: exp_run_id=%d — unknown category=%r, "
                "remapping to 'Overall'.",
                exp_run_id, item.get("category"),
            )
            item["category"] = "Overall"

        passing.append(item)

    # Step 4 — Exact-text deduplication on normalised experience_text
    seen_texts: set[str] = set()
    deduped: list[dict] = []
    for e in passing:
        key = str(e.get("experience_text") or "").lower().strip()
        if key and key not in seen_texts:
            seen_texts.add(key)
            deduped.append(e)

    duplicates_removed = len(passing) - len(deduped)
    if duplicates_removed:
        logger.info(
            "_parse_experiences: exp_run_id=%d — removed %d exact-text duplicates.",
            exp_run_id, duplicates_removed,
        )

    logger.info(
        "_parse_experiences: exp_run_id=%d — raw=%d filtered=%d deduped=%d passing=%d",
        exp_run_id, len(raw_list), filtered_count,
        duplicates_removed, len(deduped),
    )
    return deduped, filtered_count


# ---------------------------------------------------------------------------
# Single transcript extraction — Stage 1 (semaphore-gated)
# ---------------------------------------------------------------------------

async def _run_single_transcript_stage1(
    url_registry_id: int,
    raw_transcript_id: int,
    brand: str,
    model_name: str,
    schema_version: str,
    category_map: dict[str, int],
    examples: list[dict],
) -> dict:
    """
    Stage 1: extracts candidate experiences from one transcript.
    Stores candidates in phone_experience_candidates (NOT phone_experiences).
    Logs the run in experience_extraction_runs for audit.
    Returns: {"success": bool, "exp_run_id": int, "candidates_stored": int}
    """
    from app.repositories.extraction_repository import bulk_insert_experience_candidates

    exp_run_id = await asyncio.to_thread(
        insert_experience_run,
        {
            "url_registry_id":           url_registry_id,
            "raw_transcript_id":         raw_transcript_id,
            "model_used":                EXTRACTION_MODEL,
            "extraction_schema_version": schema_version,
            "status":                    "running",
        },
    )
    logger.info(
        "_run_single_transcript_stage1: exp_run_id=%d created "
        "(url_registry_id=%d raw_transcript_id=%d)",
        exp_run_id, url_registry_id, raw_transcript_id,
    )

    try:
        # Resolve transcript path
        transcript_row = await asyncio.to_thread(fetch_transcript_row, raw_transcript_id)
        translation_status = transcript_row.get("translation_status")
        if translation_status == "translation_complete":
            transcript_path = (
                transcript_row.get("translated_transcript_path")
                or transcript_row.get("processed_transcript_path")
            )
        else:
            transcript_path = transcript_row.get("processed_transcript_path")

        if not transcript_path:
            raise ValueError(
                f"_run_single_transcript_stage1: raw_transcript_id={raw_transcript_id} "
                f"has no usable path (translation_status={translation_status!r})."
            )

        async with _RUN_B_STORAGE_SEMAPHORE:
            transcript_content = await fetch_file_content(transcript_path)

        original_len = len(transcript_content)
        few_shot_block = _build_few_shot_block(examples)
        _reserved_chars = len(few_shot_block) + 500
        _effective_cap = max(_MAX_TRANSCRIPT_CHARS - _reserved_chars, 60_000)

        if len(transcript_content) > _effective_cap:
            logger.warning(
                "_run_single_transcript_stage1: exp_run_id=%d transcript truncated %d→%d",
                exp_run_id, len(transcript_content), _effective_cap,
            )
            transcript_content = transcript_content[:_effective_cap]

        if few_shot_block:
            user_content = (
                f"FEW-SHOT EXAMPLES:\n\n{few_shot_block}"
                f"\n\n---\n\nTRANSCRIPT TO EXTRACT:\n\n{transcript_content}"
            )
        else:
            user_content = transcript_content

        logger.info(
            "_run_single_transcript_stage1: exp_run_id=%d — "
            "launching Stage 1 call (%d chars original, %d chars sent)",
            exp_run_id, original_len, len(user_content),
        )

        async with _RUN_B_SEMAPHORE:
            raw_output: dict = await asyncio.wait_for(
                call_gemini_json(
                    system_prompt=EXPERIENCE_SYSTEM_PROMPT,
                    user_content=user_content,
                    output_schema=RunBExtractionSchema,
                    temperature=0.3,
                ),
                timeout=_RUN_B_TIMEOUT_SECONDS,
            )

        logger.info(
            "_run_single_transcript_stage1: exp_run_id=%d — Stage 1 call complete",
            exp_run_id,
        )

        passing, filtered_count = _parse_experiences(raw_output=raw_output, exp_run_id=exp_run_id)

        # Store in phone_experience_candidates — NOT phone_experiences
        candidate_rows = [
            {
                "url_registry_id":   url_registry_id,
                "raw_transcript_id": raw_transcript_id,
                "exp_run_id":        exp_run_id,
                "experience_text":   e.get("experience_text", ""),
                "sentiment":         str(e.get("sentiment") or "Neutral").strip().capitalize(),
                "evidence_quote":    e.get("evidence_quote") or None,
                "category":          e.get("category", "Overall"),
                "confidence":        float(e.get("confidence", 0.0)),
            }
            for e in passing
        ]

        candidates_stored = 0
        if candidate_rows:
            candidates_stored = await asyncio.to_thread(
                bulk_insert_experience_candidates, candidate_rows
            )

        await asyncio.to_thread(
            update_experience_run,
            exp_run_id,
            {
                "status":                "completed",
                "finished_at":           "now()",
                "experiences_extracted": candidates_stored,
            },
        )

        logger.info(
            "_run_single_transcript_stage1: COMPLETE exp_run_id=%d "
            "candidates_stored=%d filtered=%d",
            exp_run_id, candidates_stored, filtered_count,
        )

        return {"success": True, "exp_run_id": exp_run_id, "candidates_stored": candidates_stored}

    except Exception as exc:
        logger.exception(
            "_run_single_transcript_stage1: FAILED exp_run_id=%d raw_transcript_id=%d: %s",
            exp_run_id, raw_transcript_id, exc,
        )
        try:
            await asyncio.to_thread(
                update_experience_run,
                exp_run_id,
                {"status": "failed", "error_message": str(exc), "finished_at": "now()"},
            )
        except Exception as db_exc:
            logger.error(
                "_run_single_transcript_stage1: failed to mark exp_run_id=%d as failed: %s",
                exp_run_id, db_exc,
            )
        raise


# ---------------------------------------------------------------------------
# Stage 2 aggregation
# ---------------------------------------------------------------------------

async def _run_stage2_aggregation(
    url_registry_id: int,
    all_candidates: list[dict],
    brand: str,
    model_name: str,
    schema_version: str,
    new_transcripts_count: int,
    reused_transcripts_count: int,
) -> dict:
    """
    Stage 2: aggregates all Stage 1 candidates into the final experience set.
    One LLM call per phone per run-b trigger.

    Supersedes old phone_experiences AFTER Stage 2 LLM call succeeds.
    If Stage 2 fails for any reason, old phone_experiences are untouched.

    Returns: {"success": bool, "aggregation_run_id": int, "experiences_output": int, "message": str}
    """
    from app.repositories.extraction_repository import (
        insert_aggregation_run,
        update_aggregation_run,
        fetch_experience_category_map,
        bulk_insert_aggregated_experiences,
        supersede_experiences,
    )

    unique_transcript_count = len({c["raw_transcript_id"] for c in all_candidates})

    aggregation_run_id = await asyncio.to_thread(
        insert_aggregation_run,
        {
            "url_registry_id":          url_registry_id,
            "model_used":               EXTRACTION_MODEL,
            "schema_version":           schema_version,
            "total_candidates_input":   len(all_candidates),
            "transcripts_input":        unique_transcript_count,
            "new_transcripts_count":    new_transcripts_count,
            "reused_transcripts_count": reused_transcripts_count,
            "status":                   "running",
        },
    )
    logger.info(
        "_run_stage2_aggregation: aggregation_run_id=%d created "
        "(url_registry_id=%d, %d candidates from %d transcripts)",
        aggregation_run_id, url_registry_id, len(all_candidates), unique_transcript_count,
    )

    try:
        # Cap input size
        candidates_to_use = all_candidates[:_STAGE2_MAX_INPUT_CANDIDATES]
        if len(all_candidates) > _STAGE2_MAX_INPUT_CANDIDATES:
            logger.warning(
                "_run_stage2_aggregation: aggregation_run_id=%d — "
                "input capped %d→%d candidates",
                aggregation_run_id, len(all_candidates), _STAGE2_MAX_INPUT_CANDIDATES,
            )

        # Build labeled candidate list
        lines: list[str] = []
        for c in candidates_to_use:
            eq = c.get("evidence_quote") or "null"
            lines.append(
                f"[CANDIDATE] transcript={c['raw_transcript_id']} "
                f"category={c['category']} conf={c['confidence']}\n"
                f"experience_text: {c['experience_text']}\n"
                f"evidence_quote: {eq}\n"
            )
        user_content = (
            f"Phone: {brand} {model_name}\n"
            f"Total candidates: {len(candidates_to_use)} "
            f"from {unique_transcript_count} transcripts\n\n"
            + "\n".join(lines)
        )

        logger.info(
            "_run_stage2_aggregation: aggregation_run_id=%d — "
            "launching Stage 2 LLM call (%d candidates, %d chars)",
            aggregation_run_id, len(candidates_to_use), len(user_content),
        )

        async with _RUN_B_SEMAPHORE:
            raw_output: dict = await asyncio.wait_for(
                call_gemini_json(
                    system_prompt=_STAGE2_SYSTEM_PROMPT,
                    user_content=user_content,
                    output_schema=RunBStage2Schema,
                    temperature=0.1,
                ),
                timeout=_STAGE2_TIMEOUT_SECONDS,
            )

        logger.info(
            "_run_stage2_aggregation: aggregation_run_id=%d — Stage 2 call complete",
            aggregation_run_id,
        )

        raw_experiences = raw_output.get("experiences", [])
        if not isinstance(raw_experiences, list):
            raise ValueError(
                f"_run_stage2_aggregation: raw_output['experiences'] is not a list "
                f"(got {type(raw_experiences).__name__})"
            )

        category_map = await asyncio.to_thread(fetch_experience_category_map)
        fallback_category_id = category_map.get("Overall")
        if fallback_category_id is None:
            raise RuntimeError(
                "_run_stage2_aggregation: 'Overall' category missing from "
                "lookup_experience_categories. Seed this row before running Run B."
            )

        experience_rows: list[dict] = []
        dropped = 0

        for item in raw_experiences:
            if not isinstance(item, dict):
                dropped += 1
                continue

            # evidence_quote required in Stage 2 output
            eq = item.get("evidence_quote")
            if not eq or not str(eq).strip():
                logger.warning(
                    "_run_stage2_aggregation: aggregation_run_id=%d — "
                    "dropping experience with null/empty evidence_quote: %r",
                    aggregation_run_id, str(item.get("experience_text", ""))[:80],
                )
                dropped += 1
                continue

            # representative_raw_transcript_id required
            rep_tid = item.get("representative_raw_transcript_id")
            if rep_tid is None:
                logger.warning(
                    "_run_stage2_aggregation: aggregation_run_id=%d — "
                    "dropping experience missing representative_raw_transcript_id",
                    aggregation_run_id,
                )
                dropped += 1
                continue

            # Category → FK resolution
            raw_cat = str(item.get("category") or "").lower().strip()
            normalised_cat = _CATEGORY_NORMALISE.get(raw_cat)
            if normalised_cat:
                category_id = category_map.get(normalised_cat, fallback_category_id)
            else:
                logger.warning(
                    "_run_stage2_aggregation: aggregation_run_id=%d — "
                    "unknown category %r, remapping to 'Overall'",
                    aggregation_run_id, item.get("category"),
                )
                category_id = fallback_category_id

            source_count = max(1, int(item.get("source_transcript_count") or 1))

            raw_sentiment = str(item.get("sentiment") or "Neutral").strip().capitalize()
            if raw_sentiment not in {"Positive", "Negative", "Neutral", "Mixed"}:
                logger.warning(
                    "_run_stage2_aggregation: aggregation_run_id=%d — "
                    "unknown sentiment %r, remapping to 'Neutral'",
                    aggregation_run_id, item.get("sentiment"),
                )
                raw_sentiment = "Neutral"

            experience_rows.append({
                "url_registry_id":         url_registry_id,
                "aggregation_run_id":      aggregation_run_id,
                "exp_run_id":              None,   # Stage 2 rows always have None here
                "raw_transcript_id":       int(rep_tid),
                "category_id":             category_id,
                "experience_text":         str(item.get("experience_text", "")).strip(),
                "sentiment":               raw_sentiment,
                "evidence_quote":          str(eq).strip(),
                "confidence":              0.90,   # Post-filtered Stage 2 output — high fixed confidence
                "source_transcript_count": source_count,
            })

        if dropped:
            logger.info(
                "_run_stage2_aggregation: aggregation_run_id=%d — "
                "dropped %d experiences (missing evidence or transcript ID)",
                aggregation_run_id, dropped,
            )

        # Supersede AFTER Stage 2 succeeds — old data is safe until now
        superseded_count = await asyncio.to_thread(supersede_experiences, url_registry_id)
        logger.info(
            "_run_stage2_aggregation: aggregation_run_id=%d — "
            "superseded %d old phone_experiences rows",
            aggregation_run_id, superseded_count,
        )

        if experience_rows:
            await asyncio.to_thread(bulk_insert_aggregated_experiences, experience_rows)

        await asyncio.to_thread(
            update_aggregation_run,
            aggregation_run_id,
            {
                "status":                   "completed",
                "finished_at":              "now()",
                "total_experiences_output": len(experience_rows),
            },
        )

        logger.info(
            "_run_stage2_aggregation: COMPLETE aggregation_run_id=%d "
            "output=%d dropped=%d",
            aggregation_run_id, len(experience_rows), dropped,
        )

        return {
            "success":            True,
            "aggregation_run_id": aggregation_run_id,
            "experiences_output": len(experience_rows),
            "message": (
                f"Stage 2 complete. {len(experience_rows)} experiences aggregated "
                f"from {len(candidates_to_use)} candidates across "
                f"{unique_transcript_count} transcripts. {dropped} dropped."
            ),
        }

    except Exception as exc:
        logger.exception(
            "_run_stage2_aggregation: FAILED aggregation_run_id=%d: %s",
            aggregation_run_id, exc,
        )
        try:
            await asyncio.to_thread(
                update_aggregation_run,
                aggregation_run_id,
                {"status": "failed", "error_message": str(exc), "finished_at": "now()"},
            )
        except Exception as db_exc:
            logger.error(
                "_run_stage2_aggregation: also failed to mark aggregation_run_id=%d as failed: %s",
                aggregation_run_id, db_exc,
            )
        raise


# ---------------------------------------------------------------------------
# Batch entry point (public API) — two-stage orchestrator
# ---------------------------------------------------------------------------

async def run_experience_extraction_batch(
    url_registry_id: int,
    raw_transcript_ids: list[int],
    brand: str,
    model_name: str,
    schema_version: str = "v2",
    pipeline_run_id: str | None = None,
) -> dict:
    """
    Two-stage Run B orchestration. Same endpoint, transparent to the caller.

    First run:   all transcripts go through Stage 1 → Stage 2 → phone_experiences
    Incremental: only new transcripts run Stage 1; existing candidates reused in Stage 2

    Returns a dict (not a list — update RunBResult construction in extraction.py).
    """
    if not raw_transcript_ids:
        raise ValueError(
            "run_experience_extraction_batch: raw_transcript_ids is empty."
        )

    logger.info(
        "run_experience_extraction_batch: START url_registry_id=%d "
        "brand=%r model=%r transcripts=%s schema=%s",
        url_registry_id, brand, model_name, raw_transcript_ids, schema_version,
    )

    from app.repositories.extraction_repository import (
        fetch_existing_candidates_for_transcript,
        fetch_all_candidates_for_phone,
        bulk_insert_experience_candidates,
        fetch_experience_category_map,
    )

    # -------------------------------------------------------------------------
    # Step 1 — Gate check ONCE before anything else (and before supersede)
    # -------------------------------------------------------------------------
    validation = await asyncio.to_thread(fetch_latest_validation, url_registry_id)
    if not validation or not validation.get("can_proceed"):
        raise ValueError(
            f"{_GATE_ERROR_PREFIX} for url_registry_id={url_registry_id}. "
            "Run POST /extraction/validate first."
        )

    await _track(pipeline_run_id,
                 total_items=len(raw_transcript_ids),
                 current_stage=PipelineStage.EXPERIENCE_EXTRACTION,
                 current_step=f"Checking existing candidates for "
                              f"{len(raw_transcript_ids)} transcripts...")

    # -------------------------------------------------------------------------
    # Step 2 — Classify: need Stage 1 vs. reuse existing candidates
    # -------------------------------------------------------------------------
    transcripts_needing_stage1: list[int] = []
    transcripts_reusing: list[int] = []

    for tid in raw_transcript_ids:
        existing = await asyncio.to_thread(
            fetch_existing_candidates_for_transcript, url_registry_id, tid
        )
        if existing:
            transcripts_reusing.append(tid)
            logger.info(
                "run_experience_extraction_batch: transcript=%d — %d existing "
                "candidates found, Stage 1 skipped",
                tid, len(existing),
            )
        else:
            transcripts_needing_stage1.append(tid)

    logger.info(
        "run_experience_extraction_batch: url_registry_id=%d — "
        "Stage 1 needed: %d | reused: %d",
        url_registry_id, len(transcripts_needing_stage1), len(transcripts_reusing),
    )

    # -------------------------------------------------------------------------
    # Step 3 — Load Stage 1 examples and category map (once for all tasks)
    # -------------------------------------------------------------------------
    RUN_B_EXAMPLES, schema_version_used = _load_run_b_examples(schema_version)
    category_map = await asyncio.to_thread(fetch_experience_category_map)

    # -------------------------------------------------------------------------
    # Step 4 — Run Stage 1 in parallel for new transcripts only
    # -------------------------------------------------------------------------
    stage1_failed: list[int] = []

    if transcripts_needing_stage1:
        await _track(pipeline_run_id,
                     current_step=f"Running Stage 1 on "
                                  f"{len(transcripts_needing_stage1)} new transcripts...")

        tasks = [
            _run_single_transcript_stage1(
                url_registry_id=url_registry_id,
                raw_transcript_id=tid,
                brand=brand,
                model_name=model_name,
                schema_version=schema_version_used,
                category_map=category_map,
                examples=RUN_B_EXAMPLES,
            )
            for tid in transcripts_needing_stage1
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for tid, result in zip(transcripts_needing_stage1, results):
            if isinstance(result, Exception):
                logger.error(
                    "run_experience_extraction_batch: Stage 1 FAILED transcript=%d: %s",
                    tid, result,
                )
                stage1_failed.append(tid)
            else:
                logger.info(
                    "run_experience_extraction_batch: Stage 1 OK transcript=%d "
                    "candidates_stored=%d",
                    tid, result.get("candidates_stored", 0),
                )

    # -------------------------------------------------------------------------
    # Step 5 — Fetch ALL candidates (old + new) for Stage 2
    # -------------------------------------------------------------------------
    await _track(pipeline_run_id,
                 current_step="Fetching all candidates for Stage 2 aggregation...")

    all_candidates = await asyncio.to_thread(fetch_all_candidates_for_phone, url_registry_id)

    if not all_candidates:
        logger.warning(
            "run_experience_extraction_batch: url_registry_id=%d — "
            "no candidates available for Stage 2.",
            url_registry_id,
        )
        return {
            "success":            False,
            "aggregation_run_id": None,
            "experiences_output": 0,
            "new_transcripts":    len(transcripts_needing_stage1) - len(stage1_failed),
            "reused_transcripts": len(transcripts_reusing),
            "message":            "No candidates available for Stage 2.",
            "error":              "No candidates",
        }

    logger.info(
        "run_experience_extraction_batch: url_registry_id=%d — "
        "%d candidates from %d unique transcripts ready for Stage 2",
        url_registry_id,
        len(all_candidates),
        len({c["raw_transcript_id"] for c in all_candidates}),
    )

    # -------------------------------------------------------------------------
    # Step 6 — Stage 2 aggregation
    # -------------------------------------------------------------------------
    await _track(pipeline_run_id,
                 current_step="Running Stage 2 aggregation across all transcripts...")

    stage2_result = await _run_stage2_aggregation(
        url_registry_id=url_registry_id,
        all_candidates=all_candidates,
        brand=brand,
        model_name=model_name,
        schema_version=schema_version_used,
        new_transcripts_count=len(transcripts_needing_stage1) - len(stage1_failed),
        reused_transcripts_count=len(transcripts_reusing),
    )

    final_status = "completed" if stage2_result["success"] else "failed"
    await _track(pipeline_run_id,
                 status=final_status,
                 processed_items=len(transcripts_needing_stage1) - len(stage1_failed),
                 failed_items=len(stage1_failed),
                 current_step="Done.",
                 completed_at=_now_iso())

    logger.info(
        "run_experience_extraction_batch: COMPLETE url_registry_id=%d — "
        "Stage1 new=%d failed=%d reused=%d | Stage2 output=%d",
        url_registry_id,
        len(transcripts_needing_stage1),
        len(stage1_failed),
        len(transcripts_reusing),
        stage2_result.get("experiences_output", 0),
    )

    return {
        "success":            stage2_result["success"],
        "aggregation_run_id": stage2_result.get("aggregation_run_id"),
        "experiences_output": stage2_result.get("experiences_output", 0),
        "new_transcripts":    len(transcripts_needing_stage1) - len(stage1_failed),
        "reused_transcripts": len(transcripts_reusing),
        "message":            stage2_result["message"],
        "error":              stage2_result.get("error"),
    }

