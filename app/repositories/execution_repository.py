"""
Phase 5 — Execution Logging Layer

Task 5.1 — create_execution(): inserts a new row into pipeline.scrape_executions.
Task 5.2 — finish_execution(): updates the row with outcome, timestamps, and duration.
"""

from datetime import datetime

from app.core.supabase_client import get_client


async def create_execution(
    url_registry_id: int,
    template_id: int,
    started_at: datetime,
) -> int:
    """
    Inserts a new row into pipeline.scrape_executions.

    started_at is passed explicitly from the orchestrator — do NOT rely on DB DEFAULT NOW().
    Reason: if the DB generates its own timestamp and the orchestrator uses datetime.utcnow(),
    the two clocks can diverge due to timezone differences or network latency, producing
    incorrect or negative duration_ms values. Both timestamps must come from the same clock.

    Returns the new execution_id (int).
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("scrape_executions")
        .insert(
            {
                "url_registry_id": url_registry_id,
                "template_id":     template_id,
                "started_at":      started_at.isoformat(),
            }
        )
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            f"create_execution() insert returned no data for "
            f"url_registry_id={url_registry_id}, template_id={template_id}."
        )

    return result.data[0]["execution_id"]


async def finish_execution(
    execution_id: int,
    success: bool,
    started_at: datetime,
    finished_at: datetime,
    error_message: str | None = None,
) -> None:
    """
    Updates the scrape_executions row with the final outcome.

    Sets:
        success        — True / False
        finished_at    — ISO timestamp from orchestrator clock
        duration_ms    — computed here from (finished_at - started_at).total_seconds() * 1000
        error_message  — None on success, populated on failure

    Both started_at and finished_at must come from the orchestrator — do not re-fetch
    from DB. This guarantees they are on the same clock, preventing incorrect durations.
    """
    duration_ms = int((finished_at - started_at).total_seconds() * 1000)

    payload: dict = {
        "success":       success,
        "finished_at":   finished_at.isoformat(),
        "duration_ms":   duration_ms,
        "error_message": error_message,
    }

    result = (
        get_client()
        .schema("pipeline")
        .table("scrape_executions")
        .update(payload)
        .eq("execution_id", execution_id)
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            f"finish_execution() update returned no data for execution_id={execution_id}. "
            f"The row may not exist."
        )
