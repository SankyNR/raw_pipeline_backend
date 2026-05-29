"""
app/api/embeddings.py
======================
Phase EM7 — Sutradhar Console Embedding API

Admin-facing endpoints for embedding pipeline visibility and manual control.
All reads go through the embeddings_repository layer (sync, asyncio.to_thread).
All mutations go through embedding_pipeline.enqueue_or_first_embed or
embeddings_repository.enqueue_reembed.

Endpoints:
  GET  /admin/embeddings/status/{model_id}          — full embedding state for one phone
  GET  /admin/embeddings/runs/{model_id}             — last N audit rows
  GET  /admin/embeddings/queue                       — active queue rows (pending + claimed)
  GET  /admin/embeddings/conflicts/{model_id}        — conflict_flag=TRUE inference entries
  POST /admin/embeddings/enqueue                     — manual re-embed for one phone
  POST /admin/embeddings/enqueue-all-for-model-change — bulk re-embed (model change gate)

Schemas exposed: embeddings (read via repo), pipeline (inference_entries).
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.supabase_client import get_client
from app.repositories.embeddings_repository import (
    enqueue_reembed,
    get_phone_embedding_row,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/embeddings", tags=["embeddings"])


# ---------------------------------------------------------------------------
# EM7.1 — GET /admin/embeddings/status/{model_id}
# Full embedding state for one phone: embedding row + active queue row
# ---------------------------------------------------------------------------

@router.get("/status/{model_id}")
async def get_embedding_status(model_id: int):
    """
    GET /admin/embeddings/status/{model_id}

    Returns the full embedding state for a phone:
      - phone_embeddings row (or null if never embedded)
      - Most recent embedding_runs row (or null)
      - Active embedding_queue row with status 'pending' or 'claimed' (or null)

    Never returns 404 — a phone that has not been embedded returns
    `{"embedding": null, "last_run": null, "active_queue_job": null}`.
    """
    # Fetch all three in parallel
    embedding_row, runs_rows, queue_rows = await asyncio.gather(
        asyncio.to_thread(get_phone_embedding_row, model_id),
        asyncio.to_thread(_fetch_recent_runs, model_id, limit=1),
        asyncio.to_thread(_fetch_active_queue_row, model_id),
    )

    # Strip the raw vector from the response (too large / not useful in UI)
    if embedding_row:
        embedding_row = {k: v for k, v in embedding_row.items() if k != "embedding"}

    return {
        "model_id":        model_id,
        "embedding":       embedding_row,
        "last_run":        runs_rows[0] if runs_rows else None,
        "active_queue_job": queue_rows,
    }


# ---------------------------------------------------------------------------
# EM7.2 — GET /admin/embeddings/runs/{model_id}
# Audit log — last N embedding_runs rows for one phone
# ---------------------------------------------------------------------------

@router.get("/runs/{model_id}")
async def get_embedding_runs(model_id: int, limit: int = 20, offset: int = 0):
    """
    GET /admin/embeddings/runs/{model_id}?limit=20&offset=0

    Returns the embedding_runs audit log for a phone, most recent first.
    Includes status, triggered_by, prior/new hash, run counts, and timestamps.
    """
    limit  = max(1, min(limit, 100))
    offset = max(0, offset)

    rows = await asyncio.to_thread(_fetch_recent_runs, model_id, limit=limit, offset=offset)
    return {
        "model_id":      model_id,
        "total_returned": len(rows),
        "limit":         limit,
        "offset":        offset,
        "runs":          rows,
    }


# ---------------------------------------------------------------------------
# EM7.3 — GET /admin/embeddings/queue
# Active queue — all pending + claimed rows across all phones
# ---------------------------------------------------------------------------

@router.get("/queue")
async def get_embedding_queue(limit: int = 50, offset: int = 0):
    """
    GET /admin/embeddings/queue?limit=50&offset=0

    Returns all pending and claimed embedding_queue rows across all phones.
    Ordered by queued_at ASC (oldest jobs first — mirrors worker FIFO order).

    Useful for the admin to see the re-embed backlog and any stuck jobs.
    """
    limit  = max(1, min(limit, 200))
    offset = max(0, offset)

    rows = await asyncio.to_thread(_fetch_queue_rows, limit=limit, offset=offset)
    return {
        "total_returned": len(rows),
        "limit":          limit,
        "offset":         offset,
        "queue":          rows,
    }


# ---------------------------------------------------------------------------
# EM7.4 — GET /admin/embeddings/conflicts/{model_id}
# conflict_flag=TRUE inference_entries for one phone (HITL review surface)
# ---------------------------------------------------------------------------

@router.get("/conflicts/{model_id}")
async def get_embedding_conflicts(model_id: int):
    """
    GET /admin/embeddings/conflicts/{model_id}

    Returns all inference_entries where conflict_flag=TRUE for this phone.
    These are Run C rules that fired but Run B already covers that category —
    the embedding pipeline suppressed the Run C entry (emitted_to_embedding=FALSE).

    The admin should verify the Run B entry is correct and (if needed) either:
      - Accept the Run B version (no action required)
      - Edit the Run B experience and trigger a manual re-embed

    Returns an empty list if no conflicts exist.
    """
    rows = await asyncio.to_thread(_fetch_conflict_entries, model_id)
    return {
        "model_id":       model_id,
        "conflict_count": len(rows),
        "conflicts":      rows,
    }


# ---------------------------------------------------------------------------
# EM7.5 — POST /admin/embeddings/enqueue
# Manual re-embed for one phone
# ---------------------------------------------------------------------------

class EnqueueRequest(BaseModel):
    model_id: int
    reason:   str = "manual"
    """
    Enqueue reason stored in embedding_queue.reason.
    One of: 'manual'|'run_b_updated'|'run_c_updated'|'model_change'
    Default: 'manual'
    """


@router.post("/enqueue")
async def enqueue_embedding(body: EnqueueRequest):
    """
    POST /admin/embeddings/enqueue

    Manually enqueue a re-embed job for one phone.

    Behaviour:
      - If no existing phone_embeddings row exists: schedules an immediate
        first-embed (asyncio background task).
      - If the phone is already embedded: inserts a pending queue row.
        ON CONFLICT DO NOTHING if an active job already exists for this phone.

    Returns the resulting queue_id (null if skipped due to existing active job
    or if first-embed was triggered directly).
    """
    from app.services.embedding_pipeline import enqueue_or_first_embed

    # Check if phone exists in mobile_specs
    phone_exists = await asyncio.to_thread(_phone_exists, body.model_id)
    if not phone_exists:
        raise HTTPException(
            status_code=404,
            detail=f"model_id={body.model_id} not found in mobile_specs.phones.",
        )

    valid_reasons = {"manual", "run_b_updated", "run_c_updated", "model_change", "first_embed"}
    if body.reason not in valid_reasons:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid reason '{body.reason}'. Must be one of: {sorted(valid_reasons)}",
        )

    # Pre-check embedding existence to determine the action message
    is_first_embed = await asyncio.to_thread(get_phone_embedding_row, body.model_id) is None

    await enqueue_or_first_embed(
        model_id=body.model_id,
        url_registry_id=None,   # manual trigger has no specific url context
        reason=body.reason,
    )

    logger.info(
        "enqueue_embedding: model_id=%d reason=%s is_first_embed=%s",
        body.model_id, body.reason, is_first_embed,
    )
    return {
        "model_id": body.model_id,
        "reason":   body.reason,
        "status":   "enqueued",
        "message":  (
            "First embed scheduled immediately as background task."
            if is_first_embed
            else "Re-embed job added to embedding_queue."
        ),
    }



# ---------------------------------------------------------------------------
# EM7.6 — POST /admin/embeddings/enqueue-all-for-model-change
# Bulk re-embed — model change gate (requires confirm=true)
# ---------------------------------------------------------------------------

class EnqueueAllRequest(BaseModel):
    confirm: bool = False
    """
    Safety gate — must be explicitly set to true to trigger bulk re-embed.
    This will enqueue a re-embed job for EVERY phone in mobile_specs.phones.
    """
    batch_size: int = 50
    """
    Max phones to enqueue per request call.
    Range: 1-200. Default: 50.
    Use with offset to paginate across the full catalogue.
    """
    offset: int = 0
    """
    Skip this many phones from the sorted model_id list before enqueuing.
    """


@router.post("/enqueue-all-for-model-change")
async def enqueue_all_for_model_change(body: EnqueueAllRequest):
    """
    POST /admin/embeddings/enqueue-all-for-model-change

    Bulk enqueue a re-embed for the entire catalogue due to a model change
    (e.g. upgrading from gemini-embedding-001 to a new model version).

    SAFETY GATE: body.confirm must be true or this returns 400.

    Paginates through mobile_specs.phones sorted by model_id.
    Call repeatedly with offset until total_remaining == 0.

    Returns:
      {
        "total_phones":    int,   — total phones in catalogue
        "enqueued":        int,   — jobs inserted this call
        "skipped":         int,   — phones already had active queue jobs
        "total_remaining": int,   — phones not yet processed
        "offset_next":     int,   — pass as offset in the next call
      }
    """
    if not body.confirm:
        raise HTTPException(
            status_code=400,
            detail=(
                "Safety gate: set confirm=true to proceed with bulk re-embed. "
                "This will enqueue a re-embed for every phone in the catalogue."
            ),
        )

    batch_size = max(1, min(body.batch_size, 200))
    offset     = max(0, body.offset)

    # Resolve all model_ids
    all_model_ids = await asyncio.to_thread(_fetch_all_model_ids)
    total_phones  = len(all_model_ids)

    batch         = all_model_ids[offset: offset + batch_size]
    total_remaining = max(0, total_phones - offset - len(batch))

    if not batch:
        return {
            "total_phones":    total_phones,
            "enqueued":        0,
            "skipped":         0,
            "total_remaining": total_remaining,
            "offset_next":     offset + batch_size,
        }

    enqueued = 0
    skipped  = 0

    for model_id in batch:
        queue_id = await asyncio.to_thread(
            enqueue_reembed, model_id, None, "model_change"
        )
        if queue_id is not None:
            enqueued += 1
        else:
            skipped += 1

    logger.info(
        "enqueue_all_for_model_change: enqueued=%d skipped=%d "
        "offset=%d batch_size=%d total_remaining=%d",
        enqueued, skipped, offset, batch_size, total_remaining,
    )
    return {
        "total_phones":    total_phones,
        "enqueued":        enqueued,
        "skipped":         skipped,
        "total_remaining": total_remaining,
        "offset_next":     offset + batch_size,
    }


# ---------------------------------------------------------------------------
# Private DB helpers (sync — called via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _fetch_recent_runs(model_id: int, limit: int = 20, offset: int = 0) -> list[dict]:
    """SELECT last N embedding_runs rows for model_id, newest first."""
    result = (
        get_client()
        .schema("embeddings")
        .table("embedding_runs")
        .select(
            "embedding_run_id, status, triggered_by, embedding_model, embedding_dim, "
            "new_document_hash, prior_document_hash, "
            "new_run_b_count, new_run_c_count, prior_run_b_count, prior_run_c_count, "
            "trimmed_entry_count, error_message, started_at, finished_at"
        )
        .eq("model_id", model_id)
        .order("embedding_run_id", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    return result.data or []


def _fetch_active_queue_row(model_id: int) -> dict | None:
    """SELECT the active (pending or claimed) embedding_queue row for model_id."""
    result = (
        get_client()
        .schema("embeddings")
        .table("embedding_queue")
        .select(
            "queue_id, status, reason, claimed_by, "
            "queued_at, claimed_at, url_registry_id"
        )
        .eq("model_id", model_id)
        .in_("status", ["pending", "claimed"])
        .order("queued_at", desc=False)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    return rows[0] if rows else None


def _fetch_queue_rows(limit: int = 50, offset: int = 0) -> list[dict]:
    """SELECT all active embedding_queue rows across all phones, FIFO order."""
    result = (
        get_client()
        .schema("embeddings")
        .table("embedding_queue")
        .select(
            "queue_id, model_id, url_registry_id, reason, "
            "status, claimed_by, queued_at, claimed_at"
        )
        .in_("status", ["pending", "claimed"])
        .order("queued_at", desc=False)
        .range(offset, offset + limit - 1)
        .execute()
    )
    return result.data or []


def _fetch_conflict_entries(model_id: int) -> list[dict]:
    """SELECT inference_entries where conflict_flag=TRUE for model_id."""
    result = (
        get_client()
        .schema("pipeline")
        .table("inference_entries")
        .select(
            "id, rule_key, inference_text, sentiment, confidence, "
            "defers_to_runb_category, emitted_to_embedding, "
            "input_field_snapshot, created_at"
        )
        .eq("model_id", model_id)
        .eq("conflict_flag", True)
        .order("rule_key")
        .execute()
    )
    return result.data or []


def _phone_exists(model_id: int) -> bool:
    """Check mobile_specs.phones has a row for model_id."""
    result = (
        get_client()
        .schema("mobile_specs")
        .table("phones")
        .select("model_id")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    )
    return bool(result.data)


def _fetch_all_model_ids() -> list[int]:
    """Return all model_ids from mobile_specs.phones, sorted ASC."""
    result = (
        get_client()
        .schema("mobile_specs")
        .table("phones")
        .select("model_id")
        .order("model_id")
        .execute()
    )
    return [r["model_id"] for r in (result.data or [])]
