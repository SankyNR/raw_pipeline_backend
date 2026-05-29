"""
app/repositories/embeddings_repository.py
==========================================
Phase EM2.1 — Embeddings Schema Repository

All DB access for the embedding pipeline.
Thin wrappers: no business logic, no imports from services.

Patterns (match the rest of this repo):
  - get_client().schema("embeddings").table(...) for embeddings schema
  - All functions are SYNCHRONOUS (Supabase Python client is sync)
  - Async callers (embedding_pipeline.py) wrap with asyncio.to_thread()
  - _now_iso() for timestamp fields instead of "now()" string

Three tables owned:
  embeddings.phone_embeddings  — durable per-phone vector store
  embeddings.embedding_runs    — append-only audit log
  embeddings.embedding_queue   — re-embed job queue
"""

import datetime
import logging

from app.core.supabase_client import get_client

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# phone_embeddings — read/write
# ---------------------------------------------------------------------------

def get_phone_embedding_row(model_id: int) -> dict | None:
    """
    SELECT the current phone_embeddings row for this model_id.

    Returns:
        Full row dict if a row exists, else None.
    """
    result = (
        get_client()
        .schema("embeddings")
        .table("phone_embeddings")
        .select("*")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    return rows[0] if rows else None


def upsert_phone_embedding(payload: dict) -> None:
    """
    INSERT … ON CONFLICT (model_id) DO UPDATE for phone_embeddings.

    The vector must already be L2-normalized before this call.
    Payload keys must match phone_embeddings column names exactly.

    Required keys:
        model_id, embedding, embedding_model, embedding_dim,
        document_hash, document_char_length,
        run_b_count, run_c_count, trimmed_entry_count, embedded_at

    Optional keys (pass None to clear):
        document_text, document_token_estimate
    """
    result = (
        get_client()
        .schema("embeddings")
        .table("phone_embeddings")
        .upsert(payload, on_conflict="model_id", ignore_duplicates=False)
        .execute()
    )
    if not result.data:
        raise RuntimeError(
            f"upsert_phone_embedding: no data returned for model_id={payload.get('model_id')}."
        )
    logger.debug(
        "upsert_phone_embedding: model_id=%d hash=%s",
        payload.get("model_id"), payload.get("document_hash", "")[:12],
    )


# ---------------------------------------------------------------------------
# embedding_runs — audit log
# ---------------------------------------------------------------------------

def insert_embedding_run(payload: dict) -> int:
    """
    INSERT a new row into embedding_runs with status='running'.

    Required keys:
        model_id       int
        triggered_by   str  — 'first_embed'|'run_b_updated'|'run_c_updated'
                             |'manual'|'model_change'
        status         str  — 'running' (initial state)

    Optional keys:
        url_registry_id int

    Returns:
        embedding_run_id (int) of the newly inserted row.

    Raises:
        RuntimeError if insert returns no data.
    """
    db_payload = {
        "model_id":       payload["model_id"],
        "triggered_by":   payload["triggered_by"],
        "status":         payload.get("status", "running"),
        "started_at":     _now_iso(),
    }
    if payload.get("url_registry_id") is not None:
        db_payload["url_registry_id"] = payload["url_registry_id"]

    result = (
        get_client()
        .schema("embeddings")
        .table("embedding_runs")
        .insert(db_payload)
        .execute()
    )
    if not result.data:
        raise RuntimeError(
            f"insert_embedding_run: no data returned for model_id={payload.get('model_id')}."
        )
    run_id = result.data[0]["embedding_run_id"]
    logger.debug(
        "insert_embedding_run: embedding_run_id=%d model_id=%d triggered_by=%s",
        run_id, payload.get("model_id"), payload.get("triggered_by"),
    )
    return run_id


def update_embedding_run(embedding_run_id: int, patch: dict) -> None:
    """
    UPDATE an embedding_runs row by embedding_run_id.

    Common patch patterns:
        Completed:  {"status": "completed", "new_document_hash": ...,
                     "new_run_b_count": N, "new_run_c_count": N,
                     "embedding_model": ..., "embedding_dim": N,
                     "trimmed_entry_count": N, "finished_at": _now_iso()}
        Skipped:    {"status": "skipped_unchanged"|"skipped_no_signal",
                     "finished_at": _now_iso()}
        Failed:     {"status": "failed", "error_message": str,
                     "finished_at": _now_iso()}

    Raises:
        RuntimeError if the row is not found.
    """
    result = (
        get_client()
        .schema("embeddings")
        .table("embedding_runs")
        .update(patch)
        .eq("embedding_run_id", embedding_run_id)
        .execute()
    )
    if not result.data:
        raise RuntimeError(
            f"update_embedding_run: embedding_run_id={embedding_run_id} not found "
            f"or no update applied."
        )
    logger.debug(
        "update_embedding_run: embedding_run_id=%d status=%s",
        embedding_run_id, patch.get("status", "?"),
    )


# ---------------------------------------------------------------------------
# embedding_queue — enqueue / claim / complete / reaper
# ---------------------------------------------------------------------------

def enqueue_reembed(
    model_id: int,
    url_registry_id: int | None,
    reason: str,
) -> int | None:
    """
    INSERT a pending re-embed job into embedding_queue.

    The partial UNIQUE index (model_id WHERE status IN ('pending','claimed'))
    is the SAFETY NET at the database level. PostgREST's upsert(ignore_duplicates)
    cannot target a *partial* unique index reliably, so we use a two-step
    approach:

      1. Pre-check the table for an active row (pending/claimed) for this model_id.
         If found, return None — the existing job will still run.
      2. Otherwise, plain INSERT. Race condition between (1) and (2) is caught
         by the partial unique index as a Postgres IntegrityError; we catch
         that and return None.

    Args:
        model_id:        Phone to re-embed.
        url_registry_id: Source URL context (may be None for model_change).
        reason:          'run_b_updated'|'run_c_updated'|'manual'|'model_change'

    Returns:
        queue_id if a new row was inserted, or None if already active.
    """
    client = get_client()

    # ── Step 1: pre-check for active job ──────────────────────────────────
    existing = (
        client
        .schema("embeddings")
        .table("embedding_queue")
        .select("queue_id")
        .eq("model_id", model_id)
        .in_("status", ["pending", "claimed"])
        .limit(1)
        .execute()
    )
    if existing.data:
        logger.debug(
            "enqueue_reembed: model_id=%d already has active queue row — skipped.",
            model_id,
        )
        return None

    # ── Step 2: insert new row ────────────────────────────────────────────
    db_payload: dict = {"model_id": model_id, "reason": reason}
    if url_registry_id is not None:
        db_payload["url_registry_id"] = url_registry_id

    try:
        result = (
            client
            .schema("embeddings")
            .table("embedding_queue")
            .insert(db_payload)
            .execute()
        )
    except Exception as exc:
        # Race condition safety net: another process inserted between our
        # pre-check and this insert. The partial unique index will raise
        # an IntegrityError surfaced as a string match.
        msg = str(exc).lower()
        if "duplicate" in msg or "unique" in msg or "conflict" in msg:
            logger.debug(
                "enqueue_reembed: race condition — another worker enqueued "
                "model_id=%d concurrently. Returning None.",
                model_id,
            )
            return None
        raise

    if not result.data:
        logger.warning(
            "enqueue_reembed: insert returned no data for model_id=%d.",
            model_id,
        )
        return None

    queue_id = result.data[0]["queue_id"]
    logger.debug(
        "enqueue_reembed: model_id=%d queue_id=%d reason=%s",
        model_id, queue_id, reason,
    )
    return queue_id


def claim_next_queue_row(worker_id: str | None = None) -> dict | None:
    """
    Atomically claim the next pending queue row (FIFO by queued_at).

    Uses SELECT … FOR UPDATE SKIP LOCKED via raw SQL because the Supabase
    PostgREST client cannot express FOR UPDATE. Falls back to a two-step
    SELECT + conditional UPDATE that is safe for single-worker deployments.

    For production multi-worker safety the RPC path (Postgres function) is
    the correct approach, but the default deployment is a single worker so
    the two-step is acceptable here.

    Returns:
        The claimed row as dict, or None if the queue is empty.
    """
    client = get_client()

    # Step 1: find the oldest pending row
    select_result = (
        client
        .schema("embeddings")
        .table("embedding_queue")
        .select("*")
        .eq("status", "pending")
        .order("queued_at", desc=False)
        .limit(1)
        .execute()
    )
    rows = select_result.data or []
    if not rows:
        return None

    row = rows[0]
    queue_id = row["queue_id"]

    # Step 2: claim it (UPDATE status='claimed')
    claim_payload: dict = {
        "status":     "claimed",
        "claimed_at": _now_iso(),
    }
    if worker_id:
        claim_payload["claimed_by"] = worker_id

    update_result = (
        client
        .schema("embeddings")
        .table("embedding_queue")
        .update(claim_payload)
        .eq("queue_id", queue_id)
        .eq("status", "pending")   # guard: only update if still pending
        .execute()
    )

    # If another worker claimed it first, update returns empty → skip
    if not update_result.data:
        logger.debug(
            "claim_next_queue_row: queue_id=%d was claimed by another worker — skipping.",
            queue_id,
        )
        return None

    claimed_row = update_result.data[0]
    logger.debug(
        "claim_next_queue_row: claimed queue_id=%d model_id=%d worker=%s",
        queue_id, claimed_row.get("model_id"), worker_id,
    )
    return claimed_row


def mark_queue_row_done(queue_id: int) -> None:
    """Mark an embedding_queue row as done after successful processing."""
    result = (
        get_client()
        .schema("embeddings")
        .table("embedding_queue")
        .update({"status": "done", "processed_at": _now_iso()})
        .eq("queue_id", queue_id)
        .execute()
    )
    if not result.data:
        logger.warning("mark_queue_row_done: queue_id=%d not found.", queue_id)
    else:
        logger.debug("mark_queue_row_done: queue_id=%d", queue_id)


def mark_queue_row_failed(queue_id: int, error_message: str) -> None:
    """Mark an embedding_queue row as failed with an error message (truncated to 1000 chars)."""
    result = (
        get_client()
        .schema("embeddings")
        .table("embedding_queue")
        .update({
            "status":        "failed",
            "processed_at":  _now_iso(),
            "error_message": error_message[:1000],
        })
        .eq("queue_id", queue_id)
        .execute()
    )
    if not result.data:
        logger.warning("mark_queue_row_failed: queue_id=%d not found.", queue_id)
    else:
        logger.debug("mark_queue_row_failed: queue_id=%d", queue_id)


def release_stale_claims(timeout_seconds: int) -> int:
    """
    Reset claimed rows that have been held longer than timeout_seconds back
    to 'pending' so the next worker tick can re-claim them.

    Called by the queue worker's reaper step on every poll tick.

    Implementation note: PostgREST's UPDATE returns rows only when
    Prefer: return=representation is set. The Supabase Python client's
    default Prefer behaviour varies across versions, so to get an accurate
    released count we:
      1. SELECT stale claimed rows by queue_id first (count them ourselves).
      2. UPDATE all of them in one statement.
    This avoids relying on .update().execute().data being populated.

    Returns:
        Count of rows released back to 'pending'.
    """
    client = get_client()
    cutoff = (
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(seconds=timeout_seconds)
    ).isoformat()

    # ── Step 1: identify stale claims (this is the source of truth for count) ──
    select_result = (
        client
        .schema("embeddings")
        .table("embedding_queue")
        .select("queue_id")
        .eq("status", "claimed")
        .lt("claimed_at", cutoff)
        .execute()
    )
    stale_ids = [r["queue_id"] for r in (select_result.data or [])]
    if not stale_ids:
        logger.debug("release_stale_claims: no stale claims found.")
        return 0

    # ── Step 2: release all of them in one UPDATE ───────────────────────────
    (
        client
        .schema("embeddings")
        .table("embedding_queue")
        .update({
            "status":     "pending",
            "claimed_at": None,
            "claimed_by": None,
        })
        .in_("queue_id", stale_ids)
        .execute()
    )
    released = len(stale_ids)
    logger.warning(
        "release_stale_claims: released %d stale claimed rows (timeout=%ds).",
        released, timeout_seconds,
    )
    return released
