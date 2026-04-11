"""YouTube transcript execution repository — open and close execution log rows in pipeline.youtube_transcript_executions."""

from datetime import datetime

from app.core.supabase_client import get_client


# ---------------------------------------------------------------------------
# Task 10.1 — create_transcript_execution()
# ---------------------------------------------------------------------------

async def create_transcript_execution(
    video_registry_id: int,
    started_at: datetime,
) -> int:
    """
    Inserts a new row into pipeline.youtube_transcript_executions with:
        video_registry_id, started_at
        success = False    ← safe default; updated on close
        finished_at = NULL
        language_code = NULL
        error_message = NULL
        duration_ms = NULL

    started_at is passed explicitly from the orchestrator — do NOT use DB DEFAULT NOW().
    Reason: the orchestrator uses datetime.utcnow() for duration calculation.
    If DB generates its own timestamp, clock divergence produces incorrect duration_ms.

    Returns the new transcript_exec_id.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("youtube_transcript_executions")
        .insert({
            "video_registry_id": video_registry_id,
            "started_at": started_at.isoformat(),
            "success": False,
        })
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            f"create_transcript_execution() failed — no row returned for "
            f"video_registry_id={video_registry_id}."
        )

    return result.data[0]["transcript_exec_id"]


# ---------------------------------------------------------------------------
# Task 10.2 — finish_transcript_execution()
# ---------------------------------------------------------------------------

async def finish_transcript_execution(
    exec_id: int,
    success: bool,
    started_at: datetime,
    finished_at: datetime,
    language_code: str | None = None,
    error_message: str | None = None,
) -> None:
    """
    Updates the transcript execution row: success, finished_at, duration_ms,
    language_code, error_message.

    duration_ms is computed here:
        duration_ms = int((finished_at - started_at).total_seconds() * 1000)

    Both timestamps must come from the orchestrator clock — do not re-fetch from DB.
    """
    duration_ms = int((finished_at - started_at).total_seconds() * 1000)

    result = (
        get_client()
        .schema("pipeline")
        .table("youtube_transcript_executions")
        .update({
            "success": success,
            "finished_at": finished_at.isoformat(),
            "duration_ms": duration_ms,
            "language_code": language_code,
            "error_message": error_message,
        })
        .eq("transcript_exec_id", exec_id)
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            f"finish_transcript_execution() failed — transcript_exec_id={exec_id} not found."
        )
