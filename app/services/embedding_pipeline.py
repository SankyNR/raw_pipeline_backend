"""
app/services/embedding_pipeline.py
===================================
Transistor — Semantic Embedding Pipeline (admin backend)

This module owns ALL embedding-specific logic so that embedding model
changes, dimension choices, task-type semantics, and normalization are
isolated in one place without touching the shared Gemini wrapper.

Phases built here incrementally:
  EM1  — call_gemini_embedding, l2_normalize, _hash_document
  EM3  — assemble_embedding_document, _SECTION_ORDER, _render_spec_tail, trim
  EM4  — run_embedding_for_model, run_embedding_safely, enqueue_or_first_embed
  EM5  — post-Run-C trigger (wired in run_c_orchestrator.py; local import)
  EM6  — queue worker (embedding_queue_worker.py)

Design invariants (enforced here):
  1. No DB calls in this module — all DB goes through the repository layer.
  2. asyncio.to_thread() wraps all sync DB work in the repo layer (not here).
  3. call_gemini_embedding is the ONLY LLM call in this pipeline.
  4. l2_normalize is called immediately after every embedding API return.
  5. Content-hash short-circuit fires before every embedding API call.
  6. [SPECS] tail is never trimmed.
  7. This module never raises into its callers (fire-and-forget contract).

See EMBEDDING_PIPELINE_SYSTEM_DESIGN.md and EMBEDDING_PIPELINE_DEV_ROADMAP.md
for full architecture and invariants.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field

import numpy as np

from app.core.constants import EMBEDDING_DIM, EMBEDDING_MODEL
from app.services.gemini_client import (
    GeminiNonRetryableError,
    GeminiRateLimitError,
    GeminiTransientError,
    _classify_gemini_exception,
    _retry_with_backoff,
    get_gemini_client,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EM1.1 — call_gemini_embedding
# ---------------------------------------------------------------------------

async def call_gemini_embedding(
    text: str,
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
    output_dimensionality: int = EMBEDDING_DIM,
    max_retries: int = 3,
) -> list[float]:
    """
    Single embedding call to gemini-embedding-001.

    Owns all embedding-specific wiring: model name, task_type semantics,
    output_dimensionality, and the response shape contract.

    task_type:
        'RETRIEVAL_DOCUMENT' — used at index time (default, used here).
        'RETRIEVAL_QUERY'    — used by the user backend for buyer query strings.

    output_dimensionality:
        768 (EMBEDDING_DIM) — Matryoshka truncation from the native 3072 output.
        gemini-embedding-001 is MRL-trained so the first 768 dims form a
        coherent sub-embedding; this stays within ivfflat's 2000-dim hard limit.

    Returns:
        Raw float vector as list[float] with len == output_dimensionality.
        CALLER MUST l2_normalize before storage — gemini-embedding-001 does NOT
        auto-normalize truncated dimensions. Without it, cosine rankings are
        silently wrong.

    Raises:
        GeminiRateLimitError, GeminiTransientError, GeminiNonRetryableError.
    """
    async def _call():
        try:
            response = await get_gemini_client().aio.models.embed_content(
                model=f"models/{EMBEDDING_MODEL}",
                contents=text,
                config={
                    "task_type":             task_type,
                    "output_dimensionality": output_dimensionality,
                },
            )

            if not response.embeddings or not response.embeddings[0].values:
                raise GeminiNonRetryableError(
                    "call_gemini_embedding: API returned empty embedding list."
                )

            values = list(response.embeddings[0].values)
            logger.debug(
                "call_gemini_embedding success | model=%s | task_type=%s | dim=%d | chars=%d",
                EMBEDDING_MODEL, task_type, len(values), len(text),
            )
            return values

        except (GeminiRateLimitError, GeminiTransientError, GeminiNonRetryableError):
            raise
        except Exception as exc:
            error_class = _classify_gemini_exception(exc)
            raise error_class(f"call_gemini_embedding error: {exc}") from exc

    return await _retry_with_backoff(
        _call, max_retries=max_retries, label="call_gemini_embedding"
    )


# ---------------------------------------------------------------------------
# EM1.2 — L2 normalization helper
# ---------------------------------------------------------------------------

def l2_normalize(vec: list[float] | np.ndarray) -> list[float]:
    """
    L2-normalize a vector. Returns a list[float] (JSON-serializable).

    gemini-embedding-001 does NOT auto-normalize Matryoshka-truncated output
    (output_dimensionality < 3072). This MUST be called on every raw vector
    returned by call_gemini_embedding before storage. Without it, cosine
    similarity rankings are silently wrong.

    Args:
        vec: Raw float vector from the embedding API (list or numpy array).

    Returns:
        Unit-norm vector as list[float].

    Raises:
        ValueError: If the input norm is 0.0 (degenerate / all-zero vector).
    """
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise ValueError(
            "l2_normalize: received a zero-norm vector. "
            "This indicates a degenerate embedding — do not store."
        )
    return (arr / norm).tolist()


# ---------------------------------------------------------------------------
# EM1.2 — Document hash helper
# ---------------------------------------------------------------------------

def _hash_document(text: str) -> str:
    """
    SHA-256 of the UTF-8-encoded document text.

    Used as the content-hash short-circuit key: if the assembled document hash
    matches the stored phone_embeddings.document_hash, the pipeline skips the
    Gemini call and writes a 'skipped_unchanged' audit row instead.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# EM3.1 — Section taxonomy
# ---------------------------------------------------------------------------

# Canonical display order for all sections in the embedding document.
# Matches the 16 rows in pipeline.lookup_experience_categories exactly.
_SECTION_ORDER: list[str] = [
    "Battery Life",
    "Camera",
    "Display",
    "Performance",
    "Gaming",
    "Thermal",
    "Software",
    "Audio",
    "Connectivity",
    "Charging Speed",
    "Build Quality",
    "Durability",
    "Haptics",
    "Call Quality",
    "In the Box",
    "Overall",
]

# Maps inference_entries.rule_key → embedding document section name.
# Only rules with emitted_to_embedding=TRUE appear in the document;
# the gating is done by run_c_engine, not here.
_RUNC_RULE_TO_SECTION: dict[str, str] = {
    "jio_5g_compatibility":        "Connectivity",
    "airtel_5g_compatibility":      "Connectivity",
    "vi_5g_compatibility":          "Connectivity",
    "bsnl_5g_compatibility":        "Connectivity",
    "india_4g_band_coverage":       "Connectivity",
    "sim_connectivity_profile":     "Connectivity",
    "charger_in_box":               "Charging Speed",
    "fast_charge_assessment":       "Charging Speed",
    "npu_capability":               "Performance",
    "gaming_tier":                  "Gaming",
    "value_for_money":              "Overall",
}


# ---------------------------------------------------------------------------
# EM3.2 — AssembledDocument result type
# ---------------------------------------------------------------------------

@dataclass
class AssembledDocument:
    """
    Immutable result of assemble_embedding_document().

    Passed directly to the orchestrator (EM4), which uses it for:
      - Content-hash short-circuit  (document_hash vs stored hash)
      - LLM API call                (text)
      - Audit row                   (run_b_count, run_c_count, trimmed_count)
      - DB storage                  (char_length, token_estimate)
    """
    text:            str
    document_hash:   str          # SHA-256 of text
    char_length:     int
    token_estimate:  int          # chars // 4 heuristic
    run_b_count:     int          # Run B entries included after trim
    run_c_count:     int          # Run C entries included (never trimmed)
    trimmed_count:   int          # Run B entries dropped by trim algorithm
    # section_name -> list of experience_id / inference entry id
    sections:        dict[str, list[int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# EM3.3 — Internal helpers
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """chars // 4 token budget estimate. Fast — called in trim loop."""
    return len(text) // 4


def _group_run_b_by_section(
    entries: list[dict],
) -> dict[str, list[dict]]:
    """
    Group Run B entries by their category_name.
    Preserves within-category ordering (confidence DESC from the repo).
    Only keeps sections present in _SECTION_ORDER.
    """
    grouped: dict[str, list[dict]] = {s: [] for s in _SECTION_ORDER}
    for entry in entries:
        cat = entry.get("category_name", "Overall")
        if cat in grouped:
            grouped[cat].append(entry)
    return grouped


def _group_run_c_by_section(
    entries: list[dict],
) -> dict[str, list[dict]]:
    """
    Group Run C inference entries by their document section using
    _RUNC_RULE_TO_SECTION. Entries whose rule_key has no mapping
    are silently discarded (they were not emitted for embedding).
    """
    grouped: dict[str, list[dict]] = {s: [] for s in _SECTION_ORDER}
    for entry in entries:
        section = _RUNC_RULE_TO_SECTION.get(entry.get("rule_key", ""))
        if section and section in grouped:
            grouped[section].append(entry)
    return grouped


def _build_body_text(
    run_b_by_section: dict[str, list[dict]],
    run_c_by_section: dict[str, list[dict]],
) -> str:
    """
    Render the Run B + Run C portion of the document (no [SPECS] tail).

    Section header rules:
      - Has Run B entries only or mixed   -> [SECTION NAME]
      - Has Run C entries only            -> [SECTION NAME — INFERENCE]

    Ordering within a section: Run B entries first, Run C second.
    Sections with zero entries in either input are omitted.
    """
    lines: list[str] = []
    for section_name in _SECTION_ORDER:
        run_b = run_b_by_section.get(section_name, [])
        run_c = run_c_by_section.get(section_name, [])
        if not run_b and not run_c:
            continue

        header = (
            f"[{section_name.upper()}]"
            if run_b
            else f"[{section_name.upper()} — INFERENCE]"
        )
        lines.append(header)

        for e in run_b:
            text = (e.get("experience_text") or "").strip()
            if text:
                lines.append(text)

        for e in run_c:
            text = (e.get("inference_text") or "").strip()
            if text:
                lines.append(text)

        lines.append("")  # blank separator between sections

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# EM3.3 — [SPECS] tail renderer
# ---------------------------------------------------------------------------

def _render_spec_tail(spec_inputs: dict) -> str:
    """
    Renders the [SPECS] tail (4-6 sentence human-readable summary).

    The tail is NEVER trimmed — it provides the minimal structured signal
    for every phone even when Run B/C have zero entries.

    Uses only keys from fetch_spec_summary_inputs(). Missing values
    are silently omitted; the tail degrades gracefully.
    """
    parts: list[str] = []

    # Identity line
    brand = (spec_inputs.get("brand_name") or "").strip()
    model = (spec_inputs.get("model_name") or "").strip()
    if brand or model:
        parts.append(f"{brand} {model}.".strip())

    # Chipset + memory
    chipset      = spec_inputs.get("chipset_name")
    ram_opts     = spec_inputs.get("ram_gb_options") or []
    storage_opts = spec_inputs.get("storage_gb_options") or []
    if chipset:
        seg = f"{chipset} chipset"
        if ram_opts:
            seg += " with " + "/".join(str(r) for r in ram_opts) + " GB RAM"
        if storage_opts:
            seg += " and " + "/".join(str(s) for s in storage_opts) + " GB storage"
        parts.append(seg + ".")

    # Battery + charging
    battery  = spec_inputs.get("battery_capacity")
    charging = spec_inputs.get("charging_power")
    if battery:
        seg = f"{battery} mAh battery"
        if charging:
            seg += f" with {charging}W charging"
        parts.append(seg + ".")

    # Primary camera
    mp       = spec_inputs.get("primary_camera_mp")
    aperture = spec_inputs.get("primary_camera_aperture")
    if mp is not None:
        seg = f"{mp:.0f} MP main camera"
        if aperture is not None:
            seg += f" at f/{aperture}"
        parts.append(seg + ".")

    # Display
    size    = spec_inputs.get("display_size_inch")
    panel   = spec_inputs.get("panel_class")
    refresh = spec_inputs.get("refresh_rate")
    if size is not None:
        seg = f"{size:.1f}-inch"
        if panel:
            seg += f" {panel}"
        seg += " display"
        if refresh:
            seg += f" at {refresh}Hz"
        parts.append(seg + ".")

    # 5G
    if spec_inputs.get("has_5g"):
        parts.append("Supports 5G across major Indian carriers.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# EM3.4 — Trim algorithm
# ---------------------------------------------------------------------------

def _trim_run_b_entries(
    entries: list[dict],
    run_c_by_section: dict[str, list[dict]],
    spec_tail_text: str,
    budget_target: int,
) -> tuple[list[dict], int]:
    """
    4-step trim algorithm. Operates on Run B entries only.
    Run C entries and the [SPECS] tail are NEVER trimmed.

    Returns:
        (surviving_entries, trimmed_count)

    Steps:
      1. Defense-in-depth: drop entries with confidence < 0.75 (should be
         pre-filtered by repo but guarded here too).
      2. Drop single-source entries (source_transcript_count == 1) before
         multi-source ones — they are less corroborated.
      3. Drop the lowest-confidence entry from the largest category,
         one at a time, until under budget.
      4. Stop when token_estimate <= budget_target OR entries exhausted.
    """
    trimmed_count = 0

    def _current_token_estimate(current: list[dict]) -> int:
        grouped = _group_run_b_by_section(current)
        body    = _build_body_text(grouped, run_c_by_section)
        full    = body + "\n[SPECS]\n" + spec_tail_text
        return _estimate_tokens(full)

    # ── Step 1: confidence floor ───────────────────────────────────────────
    before  = len(entries)
    entries = [e for e in entries if float(e.get("confidence", 0)) >= 0.75]
    trimmed_count += before - len(entries)

    if _current_token_estimate(entries) <= budget_target:
        return entries, trimmed_count

    # ── Step 2: drop single-source before multi-source ────────────────────
    while _current_token_estimate(entries) > budget_target:
        singles = [
            e for e in entries
            if (e.get("source_transcript_count") or 2) == 1
        ]
        if not singles:
            break
        # Drop lowest-confidence single-source entry
        drop = min(singles, key=lambda e: float(e.get("confidence", 0)))
        entries = [e for e in entries if e.get("experience_id") != drop.get("experience_id")]
        trimmed_count += 1

    if _current_token_estimate(entries) <= budget_target:
        return entries, trimmed_count

    # ── Step 3: longest-category-first, lowest-confidence-first ──────────
    while _current_token_estimate(entries) > budget_target and entries:
        by_section = _group_run_b_by_section(entries)
        non_empty  = {s: lst for s, lst in by_section.items() if lst}
        if not non_empty:
            break

        # Largest category (most entries)
        longest = max(non_empty, key=lambda s: len(non_empty[s]))

        # Lowest-confidence entry in that category
        candidates = [
            e for e in entries
            if e.get("category_name") == longest
        ]
        if not candidates:
            break
        drop = min(candidates, key=lambda e: float(e.get("confidence", 0)))
        entries = [e for e in entries if e.get("experience_id") != drop.get("experience_id")]
        trimmed_count += 1

    return entries, trimmed_count


# ---------------------------------------------------------------------------
# EM3.2 — assemble_embedding_document (main assembler)
# ---------------------------------------------------------------------------

async def assemble_embedding_document(
    model_id: int,
    url_registry_id: int | None = None,
) -> AssembledDocument:
    """
    Assemble the embedding input document for a phone.

    Fetches all three data classes in parallel (asyncio.gather), assembles
    the document, runs the trim algorithm if over token budget, and returns
    an AssembledDocument ready for hashing and the Gemini embedding call.

    This function NEVER calls the Gemini API — that is strictly EM4's job.

    Args:
        model_id:        mobile_specs.phones.model_id of the phone to embed.
        url_registry_id: Optional — hints which url_registry_id triggered this
                         embed (used for logging only at this layer).

    Returns:
        AssembledDocument with the full text, hash, counts, and section index.

    Raises:
        RuntimeError: If spec_inputs cannot determine brand+model (indicates
                      a missing phone row in mobile_specs.phones).
    """
    from app.core.constants import (
        EMBEDDING_TOKEN_BUDGET_TARGET,
        EMBEDDING_TOKEN_HARD_CEILING,
    )
    from app.repositories.embedding_inputs_repository import (
        fetch_run_b_for_embedding,
        fetch_run_c_for_embedding,
        fetch_spec_summary_inputs,
    )

    # ── Fetch all inputs in parallel ──────────────────────────────────────
    run_b_raw, run_c_raw, spec_inputs = await asyncio.gather(
        asyncio.to_thread(fetch_run_b_for_embedding, model_id),
        asyncio.to_thread(fetch_run_c_for_embedding, model_id),
        asyncio.to_thread(fetch_spec_summary_inputs, model_id),
    )

    logger.info(
        "assemble_embedding_document: model_id=%d url_registry_id=%s "
        "run_b=%d run_c=%d",
        model_id, url_registry_id, len(run_b_raw), len(run_c_raw),
    )

    # ── Group Run C by section (constant — never trimmed) ─────────────────
    run_c_by_section = _group_run_c_by_section(run_c_raw)

    # ── Render [SPECS] tail first (never trimmed) ─────────────────────────
    spec_tail_text = _render_spec_tail(spec_inputs)

    # ── Trim Run B if over budget ─────────────────────────────────────────
    surviving_run_b, trimmed_count = _trim_run_b_entries(
        list(run_b_raw),
        run_c_by_section,
        spec_tail_text,
        EMBEDDING_TOKEN_BUDGET_TARGET,
    )

    if trimmed_count:
        logger.warning(
            "assemble_embedding_document: model_id=%d trimmed %d Run B entries "
            "to fit token budget (%d).",
            model_id, trimmed_count, EMBEDDING_TOKEN_BUDGET_TARGET,
        )

    # ── Build final grouped data ───────────────────────────────────────────
    run_b_by_section = _group_run_b_by_section(surviving_run_b)

    # ── Render body + append [SPECS] tail ─────────────────────────────────
    body_text = _build_body_text(run_b_by_section, run_c_by_section)
    full_text = (
        body_text
        + ("\n" if body_text else "")
        + "[SPECS]\n"
        + spec_tail_text
    ).strip()

    # ── Build section index (section -> list of entry IDs) ────────────────
    sections: dict[str, list[int]] = {}
    for section_name in _SECTION_ORDER:
        ids: list[int] = []
        for e in run_b_by_section.get(section_name, []):
            eid = e.get("experience_id")
            if eid is not None:
                ids.append(int(eid))
        for e in run_c_by_section.get(section_name, []):
            eid = e.get("id")
            if eid is not None:
                ids.append(int(eid))
        if ids:
            sections[section_name] = ids

    # ── Compute hash + metrics ─────────────────────────────────────────────
    doc_hash       = _hash_document(full_text)
    char_length    = len(full_text)
    token_estimate = _estimate_tokens(full_text)

    # Defensive hard-ceiling guard.
    # Should never fire after the trim algorithm runs successfully, but is the
    # last line of defence against silent truncation by the embedding model.
    # If the document exceeds the hard ceiling, the model truncates from the
    # end — the [SPECS] tail goes first, then sections in reverse order — and
    # the truncation is undetectable after the fact. Log loudly so the audit
    # row reflects the truncation risk for this phone.
    if token_estimate > EMBEDDING_TOKEN_HARD_CEILING:
        logger.error(
            "assemble_embedding_document: model_id=%d HARD-CEILING BREACH "
            "tokens_est=%d > hard_ceiling=%d. Trim algorithm should have prevented "
            "this. The embedding model will silently truncate — the resulting "
            "vector may not represent the full document. Investigate.",
            model_id, token_estimate, EMBEDDING_TOKEN_HARD_CEILING,
        )

    logger.info(
        "assemble_embedding_document: model_id=%d hash=%s chars=%d "
        "tokens_est=%d run_b=%d run_c=%d trimmed=%d",
        model_id, doc_hash[:12], char_length, token_estimate,
        len(surviving_run_b), len(run_c_raw), trimmed_count,
    )

    return AssembledDocument(
        text           = full_text,
        document_hash  = doc_hash,
        char_length    = char_length,
        token_estimate = token_estimate,
        run_b_count    = len(surviving_run_b),
        run_c_count    = len(run_c_raw),
        trimmed_count  = trimmed_count,
        sections       = sections,
    )


# ---------------------------------------------------------------------------
# EM4 — Embedding Orchestrator
# ---------------------------------------------------------------------------

import datetime as _dt


def _now_iso() -> str:
    """UTC ISO timestamp for embedding_runs audit fields."""
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


async def run_embedding_for_model(
    model_id: int,
    triggered_by: str,
    url_registry_id: int | None = None,
) -> None:
    """
    Full embedding pipeline for one phone.

    Steps:
      1. Insert embedding_run row (status='running')
      2. assemble_embedding_document — fetch + assemble + trim
      3. Content-hash short-circuit: stored hash == new hash → 'skipped_unchanged'
      4. call_gemini_embedding(doc.text) → l2_normalize
      5. upsert_phone_embedding (with prior_* audit fields from existing row)
      6. update embedding_run → 'completed'

    Always closes the audit row before returning or re-raising.
    On failure: marks 'failed' then re-raises — run_embedding_safely catches it.

    Args:
        model_id:        mobile_specs.phones.model_id
        triggered_by:    'first_embed'|'run_b_updated'|'run_c_updated'
                         |'manual'|'model_change'
        url_registry_id: Source context (used in audit row, may be None).

    Raises:
        Any exception from the pipeline. The audit row is always closed first.
    """
    from app.core.constants import EMBEDDING_DIM, EMBEDDING_MODEL
    from app.repositories.embeddings_repository import (
        get_phone_embedding_row,
        insert_embedding_run,
        update_embedding_run,
        upsert_phone_embedding,
    )

    # ── Step 1: open audit row ────────────────────────────────────────────
    run_id = await asyncio.to_thread(
        insert_embedding_run,
        {
            "model_id":        model_id,
            "triggered_by":    triggered_by,
            "url_registry_id": url_registry_id,
            "status":          "running",
        },
    )
    logger.info(
        "run_embedding_for_model: START model_id=%d triggered_by=%s run_id=%d",
        model_id, triggered_by, run_id,
    )

    try:
        # ── Step 2: assemble document ─────────────────────────────────────
        doc = await assemble_embedding_document(model_id, url_registry_id)

        # ── Step 3: fetch existing row (needed for prior_* audit + hash check)
        existing = await asyncio.to_thread(get_phone_embedding_row, model_id)

        if existing and existing.get("document_hash") == doc.document_hash:
            # Hash unchanged — skip Gemini call
            await asyncio.to_thread(
                update_embedding_run,
                run_id,
                {
                    "status":      "skipped_unchanged",
                    "finished_at": _now_iso(),
                },
            )
            logger.info(
                "run_embedding_for_model: SKIP (unchanged) model_id=%d "
                "hash=%s run_id=%d",
                model_id, doc.document_hash[:12], run_id,
            )
            return

        # ── Step 4: call Gemini → normalize ──────────────────────────────
        raw_vec    = await call_gemini_embedding(doc.text)
        normalized = l2_normalize(raw_vec)

        # pgvector via PostgREST requires string format "[v1,v2,...]"
        embedding_str = "[" + ",".join(f"{v:.8f}" for v in normalized) + "]"

        # ── Step 5: upsert phone_embeddings ──────────────────────────────
        upsert_payload: dict = {
            "model_id":               model_id,
            "embedding":              embedding_str,
            "embedding_model":        EMBEDDING_MODEL,
            "embedding_dim":          EMBEDDING_DIM,
            "document_hash":          doc.document_hash,
            "document_char_length":   doc.char_length,
            "document_token_estimate": doc.token_estimate,
            "run_b_count":            doc.run_b_count,
            "run_c_count":            doc.run_c_count,
            "trimmed_entry_count":    doc.trimmed_count,
            "embedded_at":            _now_iso(),
        }
        await asyncio.to_thread(upsert_phone_embedding, upsert_payload)

        # ── Step 6: close audit row ───────────────────────────────────────
        run_patch: dict = {
            "status":              "completed",
            "finished_at":         _now_iso(),
            "embedding_model":     EMBEDDING_MODEL,
            "embedding_dim":       EMBEDDING_DIM,
            "new_document_hash":   doc.document_hash,
            "new_run_b_count":     doc.run_b_count,
            "new_run_c_count":     doc.run_c_count,
            "trimmed_entry_count": doc.trimmed_count,
        }
        # Populate prior_* fields from existing row if available
        if existing:
            run_patch["prior_document_hash"] = existing.get("document_hash")
            run_patch["prior_run_b_count"]   = existing.get("run_b_count")
            run_patch["prior_run_c_count"]   = existing.get("run_c_count")

        await asyncio.to_thread(update_embedding_run, run_id, run_patch)

        logger.info(
            "run_embedding_for_model: DONE model_id=%d run_id=%d hash=%s "
            "chars=%d tokens_est=%d run_b=%d run_c=%d trimmed=%d",
            model_id, run_id, doc.document_hash[:12],
            doc.char_length, doc.token_estimate,
            doc.run_b_count, doc.run_c_count, doc.trimmed_count,
        )

    except Exception as exc:
        # Always close the run row — never leave it stuck in 'running'
        try:
            await asyncio.to_thread(
                update_embedding_run,
                run_id,
                {
                    "status":        "failed",
                    "error_message": str(exc)[:1000],
                    "finished_at":   _now_iso(),
                },
            )
        except Exception as close_exc:
            logger.error(
                "run_embedding_for_model: could not close run row "
                "model_id=%d run_id=%d: %s",
                model_id, run_id, close_exc,
            )
        raise  # re-raise so run_embedding_safely can log it


async def run_embedding_safely(
    model_id: int,
    triggered_by: str,
    url_registry_id: int | None = None,
) -> None:
    """
    Fire-and-forget wrapper around run_embedding_for_model.

    Design invariant 7: this module NEVER raises into its callers.
    Catches ALL exceptions, emits an error log, and returns silently.
    The embedding_run audit row is always closed by run_embedding_for_model
    before any exception propagates here.

    Used as:
      asyncio.ensure_future(run_embedding_safely(...))   — first embed
      await run_embedding_safely(...)                    — queue worker
    """
    try:
        await run_embedding_for_model(model_id, triggered_by, url_registry_id)
    except Exception as exc:
        logger.error(
            "run_embedding_safely: UNHANDLED ERROR model_id=%d "
            "triggered_by=%s: %s",
            model_id, triggered_by, exc,
            exc_info=True,
        )


async def enqueue_or_first_embed(
    model_id: int,
    url_registry_id: int | None,
    reason: str,
) -> None:
    """
    EM5 trigger decision gate. Called by the post-Run-C hook.

    Decision logic:
      - No existing phone_embeddings row  →  first embed
        Schedules run_embedding_safely as a non-blocking asyncio task
        (fire-and-forget; does not block the commit response).
      - Existing row  →  re-embed enqueued
        Inserts into embedding_queue for the background worker (EM6).
        ON CONFLICT DO NOTHING if an active job already exists.

    Args:
        model_id:        Phone to embed/re-embed.
        url_registry_id: Source URL context for the trigger.
        reason:          Trigger reason string stored in embedding_queue.reason.
    """
    from app.repositories.embeddings_repository import (
        enqueue_reembed,
        get_phone_embedding_row,
    )

    existing = await asyncio.to_thread(get_phone_embedding_row, model_id)

    if existing is None:
        # First embed: schedule immediately as a background asyncio task
        logger.info(
            "enqueue_or_first_embed: FIRST EMBED model_id=%d url_registry_id=%s",
            model_id, url_registry_id,
        )
        asyncio.ensure_future(
            run_embedding_safely(model_id, "first_embed", url_registry_id)
        )
    else:
        # Re-embed: hand off to queue worker
        queue_id = await asyncio.to_thread(
            enqueue_reembed, model_id, url_registry_id, reason
        )
        if queue_id is not None:
            logger.info(
                "enqueue_or_first_embed: ENQUEUED model_id=%d queue_id=%d reason=%s",
                model_id, queue_id, reason,
            )
        else:
            logger.debug(
                "enqueue_or_first_embed: model_id=%d already has active queue job "
                "(skipped). reason=%s",
                model_id, reason,
            )

