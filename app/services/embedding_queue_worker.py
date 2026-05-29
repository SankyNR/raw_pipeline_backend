"""
app/services/embedding_queue_worker.py
=======================================
Phase EM6 — Embedding Queue Background Worker

Runs as a long-lived asyncio background task inside the FastAPI process.
Polls embeddings.embedding_queue on a fixed interval, claims one job per tick,
and runs the full embedding pipeline for that job.

Architecture:
  - Single-worker by default (one claim per tick).
  - Multi-worker safe: claim uses conditional UPDATE status='pending' guard
    (see embeddings_repository.claim_next_queue_row).
  - Reaper fires every tick before the claim attempt to reset stale claims.
  - Fire-and-forget: embedding pipeline errors mark the queue row 'failed'
    and are logged; they never crash the worker loop.

Lifecycle (FastAPI lifespan):
  - start_embedding_queue_worker()  — called in lifespan startup, returns task
  - stop_embedding_queue_worker()   — called in lifespan shutdown, cancels task

Design invariants:
  - One import from embedding_pipeline; no circular import risk (worker imports
    from pipeline, not the other way around).
  - Sleep is interruptible: asyncio.CancelledError propagates cleanly.
  - Worker logs each tick at DEBUG level; only notable events at INFO/WARNING.
"""

from __future__ import annotations

import asyncio
import logging
import socket

from app.core.constants import (
    QUEUE_CLAIM_TIMEOUT_SECONDS,
    QUEUE_WORKER_POLL_INTERVAL_SECONDS,
)
from app.repositories.embeddings_repository import (
    claim_next_queue_row,
    mark_queue_row_done,
    mark_queue_row_failed,
    release_stale_claims,
)
from app.services.embedding_pipeline import run_embedding_safely

logger = logging.getLogger(__name__)

# Unique worker identity per process (hostname + PID fragment).
_WORKER_ID: str = f"{socket.gethostname()}-emq"

# Module-level handle so lifespan can cancel the task.
_worker_task: asyncio.Task | None = None


# ---------------------------------------------------------------------------
# Core poll loop
# ---------------------------------------------------------------------------

async def _embedding_queue_poll_loop() -> None:
    """
    Infinite poll loop. Runs until CancelledError (clean shutdown).

    Each tick:
      1. Reaper: reset stale 'claimed' rows back to 'pending'.
      2. Claim: atomically grab the oldest 'pending' row.
      3. Process: run the full embedding pipeline for that row.
      4. Settle: mark the row 'done' or 'failed'.
      5. Sleep QUEUE_WORKER_POLL_INTERVAL_SECONDS before next tick.
    """
    logger.info(
        "embedding_queue_worker: START worker_id=%s poll_interval=%ds "
        "claim_timeout=%ds",
        _WORKER_ID,
        QUEUE_WORKER_POLL_INTERVAL_SECONDS,
        QUEUE_CLAIM_TIMEOUT_SECONDS,
    )

    while True:
        try:
            await _run_one_tick()
        except asyncio.CancelledError:
            # Clean shutdown — propagate so the task ends
            logger.info("embedding_queue_worker: STOP (CancelledError received)")
            raise
        except Exception as exc:
            # Unexpected error in the tick itself — log and keep running
            logger.error(
                "embedding_queue_worker: unexpected tick error: %s",
                exc, exc_info=True,
            )

        try:
            await asyncio.sleep(QUEUE_WORKER_POLL_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            logger.info("embedding_queue_worker: STOP (cancelled during sleep)")
            raise


async def _run_one_tick() -> None:
    """
    One poll tick: reaper → claim → process → settle.
    """
    # ── Step 1: Reaper — reset stale claimed rows ─────────────────────────
    released = await asyncio.to_thread(
        release_stale_claims, QUEUE_CLAIM_TIMEOUT_SECONDS
    )
    if released:
        logger.warning(
            "embedding_queue_worker: reaper released %d stale claims", released
        )

    # ── Step 2: Claim next pending row ────────────────────────────────────
    row = await asyncio.to_thread(claim_next_queue_row, _WORKER_ID)
    if row is None:
        logger.debug("embedding_queue_worker: queue empty — sleeping")
        return

    queue_id        = row["queue_id"]
    model_id        = row["model_id"]
    url_registry_id = row.get("url_registry_id")
    reason          = row.get("reason", "run_c_updated")

    logger.info(
        "embedding_queue_worker: CLAIMED queue_id=%d model_id=%d reason=%s",
        queue_id, model_id, reason,
    )

    # ── Step 3: Run the full embedding pipeline ───────────────────────────
    # run_embedding_safely NEVER raises (design invariant 7).
    # It also always closes its own embedding_runs audit row on failure.
    pipeline_error: str | None = None
    try:
        await run_embedding_safely(
            model_id=model_id,
            triggered_by=reason,
            url_registry_id=url_registry_id,
        )
    except Exception as exc:
        # Should never happen due to run_embedding_safely's invariant,
        # but guard defensively.
        pipeline_error = str(exc)[:1000]
        logger.error(
            "embedding_queue_worker: run_embedding_safely raised unexpectedly "
            "queue_id=%d model_id=%d: %s",
            queue_id, model_id, exc, exc_info=True,
        )

    # ── Step 4: Settle the queue row ──────────────────────────────────────
    if pipeline_error is None:
        await asyncio.to_thread(mark_queue_row_done, queue_id)
        logger.info(
            "embedding_queue_worker: DONE queue_id=%d model_id=%d",
            queue_id, model_id,
        )
    else:
        await asyncio.to_thread(mark_queue_row_failed, queue_id, pipeline_error)
        logger.error(
            "embedding_queue_worker: FAILED queue_id=%d model_id=%d error=%s",
            queue_id, model_id, pipeline_error,
        )


# ---------------------------------------------------------------------------
# Lifecycle — start / stop (called by FastAPI lifespan)
# ---------------------------------------------------------------------------

async def start_embedding_queue_worker() -> asyncio.Task:
    """
    Start the background poll loop as an asyncio Task.

    Called in the FastAPI lifespan startup block. Returns the Task handle
    so lifespan can cancel it on shutdown.

    Usage (in main.py lifespan):
        worker_task = await start_embedding_queue_worker()
        yield
        await stop_embedding_queue_worker(worker_task)
    """
    global _worker_task

    if _worker_task is not None and not _worker_task.done():
        logger.warning("embedding_queue_worker: start called but worker already running")
        return _worker_task

    _worker_task = asyncio.create_task(
        _embedding_queue_poll_loop(),
        name="embedding_queue_worker",
    )
    logger.info("embedding_queue_worker: task created task_name=%s", _worker_task.get_name())
    return _worker_task


async def stop_embedding_queue_worker(task: asyncio.Task | None = None) -> None:
    """
    Cancel the background poll loop and wait for it to finish.

    Called in the FastAPI lifespan shutdown block.

    Usage:
        await stop_embedding_queue_worker(worker_task)
    """
    global _worker_task

    target = task or _worker_task
    if target is None or target.done():
        logger.info("embedding_queue_worker: stop called but worker not running")
        return

    target.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(target), timeout=10.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass  # Expected — task was cancelled or timed out during wait

    _worker_task = None
    logger.info("embedding_queue_worker: STOPPED cleanly")
