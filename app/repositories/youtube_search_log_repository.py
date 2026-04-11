"""YouTube search log repository — open/close search log rows and update url_registry denorm columns."""

from app.core.supabase_client import get_client


# ---------------------------------------------------------------------------
# Task 3.1 — create_search_log()
# ---------------------------------------------------------------------------

async def create_search_log(url_registry_id: int, triggered_by: str) -> int:
    """
    Inserts a new row into pipeline.youtube_search_log with:
        url_registry_id, triggered_by
        search_status   = 'failed'  <- safe default; updated by close_search_log()
        videos_found    = 0
        channels_searched = 0
        searched_at     = NOW()     <- set by DB default

    Returns the new search_log_id.

    Why 'failed' as default: if the process crashes before close_search_log()
    is called, the log row correctly reflects an incomplete run rather than
    falsely showing success.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("youtube_search_log")
        .insert({
            "url_registry_id": url_registry_id,
            "triggered_by": triggered_by,
            "search_status": "failed",
            "videos_found": 0,
            "channels_searched": 0,
        })
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            f"create_search_log() failed — no row returned for "
            f"url_registry_id={url_registry_id}."
        )

    return result.data[0]["search_log_id"]


# ---------------------------------------------------------------------------
# Task 3.2 — close_search_log()
# ---------------------------------------------------------------------------

async def close_search_log(
    search_log_id: int,
    search_status: str,
    videos_found: int,
    channels_searched: int,
    error_message: str | None,
) -> None:
    """
    Updates the search log row with final outcome values.
    search_status must be one of: 'success_zero', 'success_with_results', 'failed'

    Raises RuntimeError if the row is not found (e.g., ID was never created).
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("youtube_search_log")
        .update({
            "search_status": search_status,
            "videos_found": videos_found,
            "channels_searched": channels_searched,
            "error_message": error_message,
        })
        .eq("search_log_id", search_log_id)
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            f"close_search_log() failed — search_log_id={search_log_id} not found."
        )


# ---------------------------------------------------------------------------
# Task 3.3 — update_url_registry_search_denorm()
# ---------------------------------------------------------------------------

async def update_url_registry_search_denorm(url_registry_id: int) -> None:
    """
    Atomically increments youtube_search_count and sets last_youtube_search_at = NOW()
    on the url_registry row via a server-side Postgres RPC function.

    Uses increment_youtube_search_count(_url_registry_id INT) — a RETURNS VOID
    plpgsql function that performs a single UPDATE statement server-side:

        UPDATE pipeline.url_registry
        SET
            last_youtube_search_at = NOW(),
            youtube_search_count   = youtube_search_count + 1
        WHERE url_id = _url_registry_id;

    Why this is safe over the previous read-modify-write pattern:
    - No Python-side SELECT then increment then UPDATE sequence.
    - The entire operation is a single atomic DB statement.
    - Two concurrent calls on the same url_id serialize at the DB level;
      no increment is ever lost.

    Previous bugs fixed:
    1. Race condition: two concurrent requests could both read the same count,
       both compute count+1, and write the same value — permanently losing
       one increment and drifting the count indefinitely.
    2. "now()" string literal: PostgREST passes strings as column values,
       so the column received the literal string "now()" not a real timestamp.
       NOW() is only evaluated inside Postgres — which this RPC does correctly.

    Requires: increment_youtube_search_count() Postgres function.
    See: migrations/add_increment_youtube_search_count.sql

    Raises RuntimeError if the RPC call returns an error.
    """
    try:
        get_client().rpc(
            "increment_youtube_search_count",
            {"_url_registry_id": url_registry_id},
        ).execute()
    except Exception as e:
        raise RuntimeError(
            f"update_url_registry_search_denorm() RPC failed for "
            f"url_registry_id={url_registry_id}: {e}"
        )
