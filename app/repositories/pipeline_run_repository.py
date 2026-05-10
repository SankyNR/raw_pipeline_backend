"""
app/repositories/pipeline_run_repository.py

CRUD operations for pipeline.pipeline_runs — the global run tracking table.

All functions are synchronous (designed for asyncio.to_thread() wrapping).
Never call these directly from async code — always use `await asyncio.to_thread(fn, ...)`.

The tracking design is fire-and-forget: a DB write failure here must NEVER
crash the extraction pipeline. The _track() helper in each orchestrator
swallows exceptions from these calls.
"""

from app.core.supabase_client import get_client


def create_pipeline_run(
    url_registry_id: int,
    run_type: str,
    total_items: int | None = None,
) -> str:
    """
    Inserts a new pipeline_runs row with status='running'.

    Args:
        url_registry_id: The pipeline.url_registry.url_id for this phone.
        run_type:        One of 'run_a', 'run_b', 'run_all', 'youtube_fetch',
                         'normalise', 'enrich', 'resolve_conflicts'.
        total_items:     Total count of discrete items (transcripts, videos, etc.)
                         for progress tracking. None for single-item runs like run_a.

    Returns:
        run_id (UUID string) of the newly created row.

    Raises:
        RuntimeError: If the insert returns no data (unexpected DB error).
    """
    payload: dict = {
        "url_registry_id": url_registry_id,
        "run_type":        run_type,
        "status":          "running",
    }
    if total_items is not None:
        payload["total_items"] = total_items

    result = (
        get_client()
        .schema("pipeline")
        .table("pipeline_runs")
        .insert(payload)
        .execute()
    )
    if not result.data:
        raise RuntimeError(
            f"create_pipeline_run: insert returned no data "
            f"(run_type={run_type!r}, url_registry_id={url_registry_id})"
        )
    return result.data[0]["run_id"]


def update_pipeline_run(
    run_id: str,
    *,
    current_stage: str | None = None,
    current_step: str | None = None,
    processed_items: int | None = None,
    failed_items: int | None = None,
    status: str | None = None,
    completed_at: str | None = None,
    error_summary: list | None = None,
) -> None:
    """
    Partial update on a pipeline_runs row. Only non-None keyword arguments are written.

    The updated_at column is automatically refreshed by the DB trigger
    (trg_pipeline_runs_updated_at) on every UPDATE.

    Args:
        run_id:                  UUID of the run to update.
        current_stage:           Stage name from PipelineStage (e.g. "gemini_extraction").
        current_step:            Human-readable step text for the UI status line.
        processed_items:         Absolute count (not delta) of successfully processed items.
        failed_items:            Absolute count (not delta) of failed items.
        status:                  One of 'running', 'completed', 'partial', 'failed'.
        completed_at:            ISO timestamp string (use _now_iso() from orchestrator).
        error_summary:           List of {stage, message, item_id?} dicts.

    Returns:
        None. Silently succeeds even if the row doesn't exist (idempotent).
    """
    payload: dict = {}

    if current_stage      is not None: payload["current_stage"]    = current_stage
    if current_step       is not None: payload["current_step"]     = current_step
    if processed_items    is not None: payload["processed_items"]  = processed_items
    if failed_items       is not None: payload["failed_items"]     = failed_items
    if status             is not None: payload["status"]           = status
    if completed_at       is not None: payload["completed_at"]     = completed_at
    if error_summary      is not None: payload["error_summary"]    = error_summary

    if not payload:
        return  # Nothing to update — skip the round trip

    (
        get_client()
        .schema("pipeline")
        .table("pipeline_runs")
        .update(payload)
        .eq("run_id", run_id)
        .execute()
    )


def fetch_pipeline_run(run_id: str) -> dict | None:
    """
    Fetches a single pipeline_runs row by run_id.

    Returns:
        The row as a dict, or None if not found.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("pipeline_runs")
        .select("*")
        .eq("run_id", run_id)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


def fetch_recent_pipeline_runs(url_registry_id: int, limit: int = 10) -> list[dict]:
    """
    Returns the N most recent pipeline runs for a phone, ordered by started_at DESC.

    Args:
        url_registry_id: The pipeline.url_registry.url_id to filter by.
        limit:           Max number of rows to return (default 10).

    Returns:
        List of dicts — may be empty if no runs have been triggered for this phone.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("pipeline_runs")
        .select(
            "run_id, run_type, status, current_stage, current_step, "
            "total_items, processed_items, failed_items, "
            "started_at, updated_at, completed_at, error_summary"
        )
        .eq("url_registry_id", url_registry_id)
        .order("started_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []
