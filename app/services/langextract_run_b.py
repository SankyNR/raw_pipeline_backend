"""
Phase L5 — langextract_run_b.py

Multi-transcript Run B orchestrator using LangExtract.

Replaces experience_orchestrator.run_experience_extraction() for the LangExtract pathway.
The legacy orchestrator (experience_orchestrator.py) is NOT deleted until Phase L7.
Both coexist. The API router switches to this file in Phase L6.

DESIGN
------
This module fires one lx.extract() call per transcript, fully parallel, capped by a
semaphore (_RUN_B_SEMAPHORE) to prevent flooding the Gemini API with concurrent calls.

  run_experience_extraction_batch()   ← public entry point
    └─ asyncio.gather(*[_run_single_transcript() for t in transcript_ids])
         └─ async with _RUN_B_SEMAPHORE:
              lx_result = await asyncio.to_thread(lx.extract, ...)

Post-extraction pipeline (mirrors experience_orchestrator.py exactly):
  - Attribute parsing from lx.extractions  → _parse_lx_experiences()
  - Min-confidence filter (0.50)
  - Category normalisation (case-insensitive, same map as legacy)
  - Exact-text deduplication on normalised experience_text
  - DB: insert_experience_run → bulk_insert_phone_experiences → update_experience_run
  - JSONL upload → Supabase Storage (non-fatal on failure)

Return shape per transcript:
  {
      "success":               bool,
      "exp_run_id":            int,
      "raw_transcript_id":     int,
      "experiences_extracted": int,
      "experiences_filtered":  int,
      "message":               str,
  }

Return shape from batch:
  list[dict] — one dict per transcript. On asyncio.gather(..., return_exceptions=True),
  failed transcripts produce exception objects instead of dicts — callers must inspect.

SEMAPHORE VALUE:
  Start at 5. A Galaxy S24 Ultra may have 22 transcripts → 22 concurrent thread launches
  without the semaphore. With 5, at most 5 Gemini calls are in-flight per process. Tune
  upward (to 8 or 10) if no 429s appear after 50+ batch runs. Never start at 10+ without
  testing.

THREADING SAFETY:
  lx.extract() is synchronous. Each call runs in its own asyncio.to_thread() worker.
  Multiple simultaneous calls against the same lx module are expected and tested by the
  Phase L5 concurrent-safety validation (Section 17 Q2 of migration doc). No global
  mutable state is written by lx.extract() between calls — each invocation returns a
  fresh result object. If thread-local state issues appear, serialise with a global Lock
  in addition to the semaphore.
"""

import asyncio
import logging
import os
import tempfile

import langextract as lx

from app.config.langextract_examples_run_b_v1 import RUN_B_EXAMPLES
from app.services.storage_service import fetch_file_content, store_jsonl_content
from app.repositories.extraction_repository import (
    fetch_latest_validation,
    fetch_transcript_row,
    fetch_experience_category_map,
    insert_experience_run,
    update_experience_run,
    bulk_insert_phone_experiences,
    update_exp_run_jsonl_path,
)
from app.utils.path_builder import slugify

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants (mirrored from experience_orchestrator.py)
# ---------------------------------------------------------------------------

EXTRACTION_MODEL = "gemini-2.5-flash"   # change here to switch models pipeline-wide

# Caps concurrent lx.extract() Gemini calls per FastAPI process.
# At 5: 22-transcript run takes ceil(22/5) = 5 "waves" ≈ 5× slower than uncapped
# but prevents 429 rate-limit errors and protects Gemini quota.
_RUN_B_SEMAPHORE = asyncio.Semaphore(5)

# Matches _MAX_TRANSCRIPT_CHARS in experience_orchestrator.py — intentional parity.
# Long transcripts risk silent intra-chunk splitting inside lx.extract().
_MAX_TRANSCRIPT_CHARS: int = 120_000

# Confidence threshold — entries below this are discarded before DB insert
_MIN_CONFIDENCE: float = 0.50

_GATE_ERROR_PREFIX = "Pre-extraction validation not passed"

# MED-3: Timeout for asyncio.wait_for() around lx.extract() to prevent indefinite
# event-loop stalls from hung Gemini calls. 300s = 5 minutes, generous for large inputs.
_LX_EXTRACT_TIMEOUT_SECONDS: int = 300


# ---------------------------------------------------------------------------
# Category constants (mirrored from experience_orchestrator.py)
# ---------------------------------------------------------------------------

_VALID_CATEGORIES: frozenset[str] = frozenset({
    "Thermal", "Camera", "Battery Life", "Charging Speed", "Display",
    "Performance", "Audio", "Build Quality", "Software", "Gaming",
    "Call Quality", "Haptics", "Connectivity", "In the Box",
    "Durability", "Overall",
})

# B4 fix: .title() is wrong for "In the Box" → "In The Box". Use lowered map.
_CATEGORY_NORMALISE: dict[str, str] = {c.lower(): c for c in _VALID_CATEGORIES}


# ---------------------------------------------------------------------------
# Run B Prompt
#
# Mirrors SYSTEM_PROMPT_RUN_B from experience_orchestrator.py.
# In the LangExtract pathway this becomes prompt_description rather than
# a Gemini system_instruction. The examples (RUN_B_EXAMPLES) teach the
# attribute names and value formats — the prompt_description sets context.
# ---------------------------------------------------------------------------

RUN_B_PROMPT = """\
Extract real-world user experience observations from this YouTube phone review transcript.
Output one Extraction per distinct experiential observation.
Use null for missing attribute values.

EXTRACT ONLY:
- Subjective, experiential statements — things the reviewer personally observed or felt.
- Examples: "gets warm after gaming", "battery is excellent", "display is stunning outdoors"

DO NOT EXTRACT:
- Objective specifications (battery mAh, screen size, resolution, processor speed, etc.)
- Sponsored content, affiliate links, channel promotion
- Meta-statements about the review itself ("subscribe", "check the link")
- Repeated observations — extract each observation ONCE

EXPERIENCE CATEGORIES (assign exactly one):
Thermal | Camera | Battery Life | Charging Speed | Display | Performance |
Audio | Build Quality | Software | Gaming | Call Quality | Haptics |
Connectivity | In the Box | Durability | Overall

ATTRIBUTES (required for every extraction):
  experience_text  — 1–3 sentence neutral third-person summary, present tense.
                     Write as: what would a user type to find phones with this quality?
                     Avoid reviewer-specific language (I, you, we).
  sentiment        — Positive | Negative | Neutral | Mixed
  evidence_quote   — Exact verbatim sentence(s) from the transcript. Do not paraphrase.
  category         — one of the categories above
  confidence       — float 0.00–1.00
                     1.00 = explicit, unambiguous, specific observation
                     0.75 = clear but slightly ambiguous evidence
                     0.50 = implied or borderline observation
                     < 0.50 = exclude entirely
"""


# ---------------------------------------------------------------------------
# LangExtract result → experience rows parser
# ---------------------------------------------------------------------------

def _parse_lx_experiences(
    lx_result,
    raw_transcript_id: int,
    url_registry_id: int,
    exp_run_id: int,
) -> tuple[list[dict], int]:
    """
    Converts lx_result.extractions into a list of experience dicts ready for
    bulk_insert_phone_experiences(), applying the same post-processing pipeline
    as experience_orchestrator.py:

      1. Extract attributes from each Extraction object
      2. Filter by confidence < _MIN_CONFIDENCE
      3. Normalise category (case-insensitive, same _CATEGORY_NORMALISE map)
      4. Deduplicate on normalised experience_text

    Returns:
        (passing, filtered_count)
        passing        — list of validated experience dicts (without category_id — added later)
        filtered_count — how many were dropped below _MIN_CONFIDENCE

    Note: category_id FK resolution happens in the caller (_run_single_transcript)
    after the category_map is fetched from DB. This function returns category strings.
    """
    raw_entries: list[dict] = []

    for extraction in lx_result.extractions:
        # extraction.attributes is the dict of field→value pairs
        attrs = extraction.attributes or {}
        raw_entries.append({
            "experience_text": attrs.get("experience_text"),
            "sentiment":       attrs.get("sentiment"),
            "evidence_quote":  attrs.get("evidence_quote"),
            "category":        attrs.get("category"),
            "confidence":      attrs.get("confidence"),
        })

    logger.debug(
        "_parse_lx_experiences: exp_run_id=%d — %d raw extractions",
        exp_run_id, len(raw_entries),
    )

    # Step 1 — Confidence filter
    passing: list[dict] = []
    filtered_count = 0

    for entry in raw_entries:
        try:
            confidence = float(entry.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        entry["confidence"] = confidence

        if confidence < _MIN_CONFIDENCE:
            filtered_count += 1
            continue

        # Step 2 — Category normalisation (B4 fix: lowered map avoids .title() bug)
        raw_cat = str(entry.get("category") or "").lower().strip()
        normalised_cat = _CATEGORY_NORMALISE.get(raw_cat)
        if normalised_cat:
            entry["category"] = normalised_cat
        else:
            logger.warning(
                "_parse_lx_experiences: exp_run_id=%d unknown category=%r "
                "— remapping to 'Overall'.",
                exp_run_id, entry.get("category"),
            )
            entry["category"] = "Overall"

        passing.append(entry)

    # Step 3 — Exact-text deduplication on normalised experience_text (B5 fix)
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
            "_parse_lx_experiences: exp_run_id=%d removed %d duplicate entries.",
            exp_run_id, duplicates_removed,
        )

    return deduped, filtered_count


# ---------------------------------------------------------------------------
# JSONL storage helper (mirrors Run A pattern, adapted for exp paths)
# ---------------------------------------------------------------------------

def _save_run_b_jsonl_sync(
    lx_result,
    brand: str,
    model_name: str,
    exp_run_id: int,
) -> str:
    """
    Synchronous half of JSONL storage (runs inside asyncio.to_thread).
    Uses lx.io.save_annotated_documents() → tempfile → read back as string.

    Returns:
        jsonl_text — utf-8 string content of the JSONL file.

    Raises:
        RuntimeError on any IO failure.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_name = f"exp_{exp_run_id}"
            lx.io.save_annotated_documents(
                [lx_result],
                output_name=run_name,
                output_dir=tmpdir,
            )
            jsonl_path_on_disk = os.path.join(tmpdir, f"{run_name}.jsonl")
            if not os.path.exists(jsonl_path_on_disk):
                candidate = jsonl_path_on_disk + ".jsonl"
                if os.path.exists(candidate):
                    jsonl_path_on_disk = candidate
                else:
                    files = os.listdir(tmpdir)
                    raise RuntimeError(
                        f"_save_run_b_jsonl_sync: expected JSONL at "
                        f"{jsonl_path_on_disk!r} — not found. "
                        f"tmpdir contents: {files}"
                    )
            with open(jsonl_path_on_disk, "r", encoding="utf-8") as fh:
                return fh.read()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"_save_run_b_jsonl_sync: failed to generate JSONL for exp_run_id={exp_run_id}: {exc}"
        ) from exc


async def _store_run_b_jsonl(
    lx_result,
    brand: str,
    model_name: str,
    exp_run_id: int,
    raw_transcript_id: int,
) -> str:
    """
    Generates JSONL and uploads to Supabase Storage.

    Storage path:
        {brand_slug}/{model_slug}/extractions/exp_{exp_run_id}.jsonl

    Returns the storage path (persisted to experience_extraction_runs.jsonl_path).
    """
    brand_slug = slugify(brand)
    model_slug = slugify(model_name)
    storage_path = f"{brand_slug}/{model_slug}/extractions/exp_{exp_run_id}.jsonl"

    jsonl_text = await asyncio.to_thread(
        _save_run_b_jsonl_sync, lx_result, brand, model_name, exp_run_id
    )

    await store_jsonl_content(storage_path, jsonl_text)
    logger.info(
        "_store_run_b_jsonl: exp_run_id=%d uploaded %d chars to %s",
        exp_run_id, len(jsonl_text), storage_path,
    )
    return storage_path


# ---------------------------------------------------------------------------
# Single transcript extraction (semaphore-gated)
# ---------------------------------------------------------------------------

async def _run_single_transcript(
    url_registry_id: int,
    raw_transcript_id: int,
    brand: str,
    model_name: str,
    schema_version: str,
) -> dict:
    """
    Extracts experiences from one transcript using LangExtract.

    Semaphore-gated via _RUN_B_SEMAPHORE to cap concurrent Gemini calls.
    The semaphore is acquired ONLY around the lx.extract() call — not around
    DB operations (which are fast and don't hit Gemini quota).

    Flow:
      0. Gate check     — (skipped here; validated once in batch before launch)
      1. DB insert      — insert_experience_run (status='running')
      2. Resolve path   — fetch_transcript_row, pick translated or processed path
      3. Fetch content  — fetch_file_content, truncate at _MAX_TRANSCRIPT_CHARS
      4. Semaphore lock — acquire _RUN_B_SEMAPHORE
      5. Extract        — asyncio.to_thread(lx.extract, ...)
      6. Semaphore release
      7. Parse          — _parse_lx_experiences (filter → normalise → dedup)
      8. JSONL upload   — _store_run_b_jsonl (non-fatal on failure)
      9. Category map   — fetch_experience_category_map
      10. DB insert     — bulk_insert_phone_experiences
      11. DB update     — update_experience_run(completed)

    Returns:
        {success, exp_run_id, raw_transcript_id, experiences_extracted,
         experiences_filtered, message}

    Raises:
        ValueError: If transcript has no usable path.
        RuntimeError: on catastrophic parse failures.
        Re-raises on any unhandled exception after marking run as failed.
    """
    # Step 1 — Create run record
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
        "_run_single_transcript: exp_run_id=%d created "
        "(url_registry_id=%d raw_transcript_id=%d)",
        exp_run_id, url_registry_id, raw_transcript_id,
    )

    try:
        # Step 2 — Resolve transcript path
        transcript_row = await asyncio.to_thread(fetch_transcript_row, raw_transcript_id)

        if transcript_row.get("translation_status") == "translation_complete":
            transcript_path = transcript_row.get("translated_transcript_path")
        else:
            transcript_path = transcript_row.get("processed_transcript_path")

        if not transcript_path:
            raise ValueError(
                f"_run_single_transcript: raw_transcript_id={raw_transcript_id} "
                f"has no usable transcript path "
                f"(translation_status={transcript_row.get('translation_status')!r})."
            )

        # Step 3 — Fetch content + truncate guard
        transcript_content = await fetch_file_content(transcript_path)
        original_len = len(transcript_content)
        if original_len > _MAX_TRANSCRIPT_CHARS:
            logger.warning(
                "_run_single_transcript: exp_run_id=%d transcript truncated "
                "%d→%d chars (cost/chunking guard).",
                exp_run_id, original_len, _MAX_TRANSCRIPT_CHARS,
            )
            transcript_content = transcript_content[:_MAX_TRANSCRIPT_CHARS]

        logger.info(
            "_run_single_transcript: exp_run_id=%d transcript ready "
            "(%d chars → %d sent)",
            exp_run_id, original_len, len(transcript_content),
        )

        # Steps 4–6 — Semaphore-gated lx.extract()
        async with _RUN_B_SEMAPHORE:
            logger.info(
                "_run_single_transcript: exp_run_id=%d — semaphore acquired, "
                "launching lx.extract()",
                exp_run_id,
            )
            lx_result = await asyncio.wait_for(
                asyncio.to_thread(
                    lx.extract,
                    transcript_content,          # positional: text_or_documents
                    prompt_description=RUN_B_PROMPT,
                    examples=RUN_B_EXAMPLES,
                    model_id=EXTRACTION_MODEL,
                    extraction_passes=1,
                    max_char_buffer=_MAX_TRANSCRIPT_CHARS,
                ),
                timeout=_LX_EXTRACT_TIMEOUT_SECONDS,
            )

        logger.info(
            "_run_single_transcript: exp_run_id=%d — lx.extract() returned "
            "%d extractions",
            exp_run_id, len(lx_result.extractions),
        )

        # Monitor ungrounded rate
        ungrounded = sum(1 for e in lx_result.extractions if e.char_interval is None)
        if ungrounded:
            logger.warning(
                "_run_single_transcript: exp_run_id=%d — %d/%d extractions "
                "ungrounded (char_interval=None, %.0f%%). "
                "Review example quality if > 20%%.",
                exp_run_id, ungrounded, max(len(lx_result.extractions), 1),
                (ungrounded / max(len(lx_result.extractions), 1)) * 100,
            )

        # Step 7 — Parse, filter, normalise, deduplicate
        passing, filtered_count = _parse_lx_experiences(
            lx_result=lx_result,
            raw_transcript_id=raw_transcript_id,
            url_registry_id=url_registry_id,
            exp_run_id=exp_run_id,
        )

        logger.info(
            "_run_single_transcript: exp_run_id=%d — raw=%d passing=%d filtered=%d",
            exp_run_id,
            len(lx_result.extractions),
            len(passing),
            filtered_count,
        )

        # Step 8 — JSONL upload (non-fatal)
        jsonl_path: str | None = None
        try:
            jsonl_path = await _store_run_b_jsonl(
                lx_result=lx_result,
                brand=brand,
                model_name=model_name,
                exp_run_id=exp_run_id,
                raw_transcript_id=raw_transcript_id,
            )
        except Exception as jsonl_exc:
            logger.warning(
                "_run_single_transcript: exp_run_id=%d JSONL upload failed "
                "(non-fatal): %s",
                exp_run_id, jsonl_exc,
            )

        # Steps 9–10 — Fetch category map and bulk insert
        if passing:
            category_map = await asyncio.to_thread(fetch_experience_category_map)
            fallback_category_id = category_map.get("Overall")

            experience_rows = [
                {
                    "url_registry_id":   url_registry_id,
                    "exp_run_id":        exp_run_id,
                    "experience_text":   e.get("experience_text", ""),
                    "sentiment":         e.get("sentiment", "Neutral"),
                    "evidence_quote":    e.get("evidence_quote") or None,  # B7: empty → NULL
                    "category_id":       category_map.get(               # B1: FK not string
                        e.get("category", ""), fallback_category_id
                    ),
                    "confidence":        float(e.get("confidence", 0.0)),
                    "raw_transcript_id": raw_transcript_id,
                }
                for e in passing
            ]
            await asyncio.to_thread(bulk_insert_phone_experiences, experience_rows)

        # Step 11 — Mark run complete
        await asyncio.to_thread(
            update_experience_run,
            exp_run_id,
            {
                "status":                "completed",
                "finished_at":           "now()",
                "experiences_extracted": len(passing),
            },
        )

        if jsonl_path is not None:
            await asyncio.to_thread(
                update_exp_run_jsonl_path, exp_run_id, jsonl_path
            )

        logger.info(
            "_run_single_transcript: COMPLETE exp_run_id=%d "
            "extracted=%d filtered=%d jsonl_path=%s",
            exp_run_id, len(passing), filtered_count, jsonl_path,
        )
        return {
            "success":               True,
            "exp_run_id":            exp_run_id,
            "raw_transcript_id":     raw_transcript_id,
            "experiences_extracted": len(passing),
            "experiences_filtered":  filtered_count,
            "message": (
                f"Run B (LX) complete. {len(passing)} experiences extracted, "
                f"{filtered_count} below confidence threshold."
            ),
        }

    except Exception as exc:
        # Mark run failed, then re-raise so asyncio.gather captures the exception
        logger.exception(
            "_run_single_transcript: FAILED exp_run_id=%d raw_transcript_id=%d: %s",
            exp_run_id, raw_transcript_id, exc,
        )
        try:
            await asyncio.to_thread(
                update_experience_run,
                exp_run_id,
                {
                    "status":        "failed",
                    "error_message": str(exc),
                    "finished_at":   "now()",
                },
            )
        except Exception as db_exc:
            logger.error(
                "_run_single_transcript: also failed to mark exp_run_id=%d as failed: %s",
                exp_run_id, db_exc,
            )
        raise


# ---------------------------------------------------------------------------
# Batch entry point (public API)
# ---------------------------------------------------------------------------

async def run_experience_extraction_batch(
    url_registry_id: int,
    raw_transcript_ids: list[int],
    brand: str,
    model_name: str,
    schema_version: str = "v1",
) -> list[dict]:
    """
    Parallel Run B: fires one lx.extract() per transcript, semaphore-capped.

    The gate check (fetch_latest_validation) is done ONCE here before launching
    all tasks. This avoids N identical DB reads for an N-transcript phone and
    ensures a blocked phone is rejected before any exp_run_id records are created.

    Args:
        url_registry_id:     url_registry.url_id for this phone.
        raw_transcript_ids:  List of youtube_raw_transcript_data.raw_transcript_id values.
                             Each becomes one lx.extract() call.
        brand:               Brand name (used for JSONL storage path + logging).
        model_name:          Model name (used for JSONL storage path + logging).
        schema_version:      Extraction schema version (default 'v1').

    Returns:
        list[dict] — one entry per transcript.
        On asyncio.gather(..., return_exceptions=True), failed transcripts
        produce exception objects instead of dicts. Callers must inspect each
        element:
            results = await run_experience_extraction_batch(...)
            for r in results:
                if isinstance(r, Exception):
                    # handle failure
                else:
                    # use r["experiences_extracted"] etc.

    Raises:
        ValueError: If gate check fails before any task is launched.
        ValueError: If raw_transcript_ids is empty.
    """
    if not raw_transcript_ids:
        raise ValueError(
            "run_experience_extraction_batch: raw_transcript_ids is empty. "
            "Nothing to extract."
        )

    logger.info(
        "run_experience_extraction_batch: START url_registry_id=%d "
        "brand=%r model=%r transcripts=%s",
        url_registry_id, brand, model_name, raw_transcript_ids,
    )

    # Gate check — once for the whole batch
    validation = await asyncio.to_thread(fetch_latest_validation, url_registry_id)
    if not validation or not validation.get("can_proceed"):
        raise ValueError(
            f"{_GATE_ERROR_PREFIX} for url_registry_id={url_registry_id}. "
            "Run POST /extraction/validate first and ensure can_proceed=true."
        )

    # Launch all transcript tasks simultaneously
    # asyncio.gather catches per-transcript exceptions when return_exceptions=True
    tasks = [
        _run_single_transcript(
            url_registry_id=url_registry_id,
            raw_transcript_id=tid,
            brand=brand,
            model_name=model_name,
            schema_version=schema_version,
        )
        for tid in raw_transcript_ids
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Log batch summary
    successes = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
    failures = len(results) - successes
    total_extracted = sum(
        r.get("experiences_extracted", 0)
        for r in results
        if isinstance(r, dict)
    )

    logger.info(
        "run_experience_extraction_batch: COMPLETE url_registry_id=%d — "
        "%d/%d transcripts succeeded, %d failed, %d total experiences extracted",
        url_registry_id, successes, len(results), failures, total_extracted,
    )

    return list(results)
