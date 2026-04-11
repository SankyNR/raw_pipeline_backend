"""YouTube video registry repository — upsert, query, and status helpers for pipeline.youtube_video_url_registry."""

from app.core.supabase_client import get_client


# ---------------------------------------------------------------------------
# Task 4.1 — upsert_video()
# ---------------------------------------------------------------------------

async def upsert_video(
    url_registry_id: int,
    channel_id: int,
    yt_video_id: str,
    video_url: str,
    video_title: str | None,
    published_at: str | None,
) -> bool:
    """
    Inserts a new row into pipeline.youtube_video_url_registry.
    Uses ON CONFLICT (url_registry_id, yt_video_id) DO NOTHING.

    Returns True if a new row was inserted.
    Returns False if the row already existed (conflict, no update).

    Never overwrites an existing row — not its status, not its title, not anything.
    A video already at fetched_raw or failed must remain untouched by re-search.
    New rows default to status = 'not_fetched' (DB default).
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("youtube_video_url_registry")
        .upsert(
            {
                "url_registry_id": url_registry_id,
                "channel_id": channel_id,
                "yt_video_id": yt_video_id,
                "video_url": video_url,
                "video_title": video_title,
                "published_at": published_at,
            },
            on_conflict="url_registry_id,yt_video_id",
            ignore_duplicates=True,
        )
        .execute()
    )

    # supabase-py returns the inserted row(s) in result.data on success.
    # If the row already existed (conflict + DO NOTHING), result.data == [].
    return bool(result.data)


# ---------------------------------------------------------------------------
# Task 4.2 — get_videos_for_phone()
# ---------------------------------------------------------------------------

async def get_videos_for_phone(url_registry_id: int) -> list[dict]:
    """
    Returns all video rows for the given url_registry_id.
    JOINs youtube_channels to include: channel_name, channel_handle, language, yt_channel_id.

    Each row includes at minimum:
        video_registry_id, yt_video_id, video_url, video_title, published_at,
        status, channel_name, channel_handle, language, yt_channel_id, created_at

    Returns empty list if no videos registered for this phone.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("youtube_video_url_registry")
        .select(
            "video_registry_id, yt_video_id, video_url, video_title, "
            "published_at, status, created_at, "
            "youtube_channels(channel_name, channel_handle, language, yt_channel_id)"
        )
        .eq("url_registry_id", url_registry_id)
        .order("created_at", desc=True)
        .execute()
    )

    rows = result.data or []

    # Flatten the nested youtube_channels dict into the top-level row
    flattened = []
    for row in rows:
        channel = row.pop("youtube_channels", {}) or {}
        row.update(channel)
        flattened.append(row)

    return flattened


# ---------------------------------------------------------------------------
# Task 4.3 — get_video_row()
# ---------------------------------------------------------------------------

async def get_video_row(video_registry_id: int) -> dict:
    """
    Fetches a single video row by video_registry_id.
    JOINs youtube_channels to include: channel_name, language, yt_channel_id.

    Raises ValueError if not found.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("youtube_video_url_registry")
        .select(
            "*, youtube_channels(channel_name, channel_handle, language, yt_channel_id)"
        )
        .eq("video_registry_id", video_registry_id)
        .execute()
    )

    if not result.data:
        raise ValueError(
            f"No video row found for video_registry_id={video_registry_id}."
        )

    row = result.data[0]
    channel = row.pop("youtube_channels", {}) or {}
    row.update(channel)
    return row


# ---------------------------------------------------------------------------
# Task 4.4 — get_phone_info_for_video()
# ---------------------------------------------------------------------------

async def get_phone_info_for_video(video_registry_id: int) -> dict:
    """
    Resolves the brand and model_name for the phone linked to this video.
    Chain: youtube_video_url_registry → url_registry → brand, model_name.

    Returns dict with: brand, model_name.
    Raises ValueError if the chain is broken (video not found, or url_registry row missing).
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("youtube_video_url_registry")
        .select(
            "video_registry_id, "
            "url_registry(url_id, brand, model_name)"
        )
        .eq("video_registry_id", video_registry_id)
        .execute()
    )

    if not result.data:
        raise ValueError(
            f"get_phone_info_for_video() — no video row found for "
            f"video_registry_id={video_registry_id}."
        )

    url_registry = result.data[0].get("url_registry")
    if not url_registry:
        raise ValueError(
            f"get_phone_info_for_video() — url_registry chain broken for "
            f"video_registry_id={video_registry_id}. url_registry row is missing."
        )

    return {
        "brand": url_registry["brand"],
        "model_name": url_registry["model_name"],
    }


# ---------------------------------------------------------------------------
# Phase 11 — Task 11.1: Video Status Helpers
# ---------------------------------------------------------------------------

async def claim_video_for_fetching(video_registry_id: int) -> bool:
    """
    Atomically sets status = 'currently_fetching' ONLY IF
    current status != 'currently_fetching'.

    Must use a single UPDATE with WHERE clause — NOT a separate SELECT + UPDATE.
    The two-step approach has a race condition where two simultaneous requests
    can both read 'not_fetched' and both proceed to fetch the same video.

    Correct implementation:
        UPDATE pipeline.youtube_video_url_registry
        SET status = 'currently_fetching', updated_at = NOW()
        WHERE video_registry_id = {video_registry_id}
          AND status != 'currently_fetching'
        RETURNING video_registry_id

    Returns True if claimed (RETURNING returned a row).
    Returns False if already currently_fetching (no row returned — another pipeline run is active).

    Note: not_fetched, fetched_raw, and failed are all claimable.
    Re-fetching a fetched_raw or failed video is intentional.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("youtube_video_url_registry")
        .update({"status": "currently_fetching"})
        .eq("video_registry_id", video_registry_id)
        .neq("status", "currently_fetching")
        .execute()
    )
    # If the WHERE clause (AND status != 'currently_fetching') was false,
    # result.data == [] — another pipeline run already holds the claim.
    return bool(result.data)


async def set_video_fetched_raw(video_registry_id: int) -> None:
    """Sets status = 'fetched_raw'. Terminal success state for the YouTube pipeline."""
    result = (
        get_client()
        .schema("pipeline")
        .table("youtube_video_url_registry")
        .update({"status": "fetched_raw"})
        .eq("video_registry_id", video_registry_id)
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            f"set_video_fetched_raw() failed — video_registry_id={video_registry_id} not found or not updated."
        )


async def set_video_failed(video_registry_id: int) -> None:
    """
    Sets status = 'failed'. Terminal content failure.
    Used when the video has no captions, is private, or is age-restricted.
    Not retried automatically. Row is preserved for audit.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("youtube_video_url_registry")
        .update({"status": "failed"})
        .eq("video_registry_id", video_registry_id)
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            f"set_video_failed() failed — video_registry_id={video_registry_id} not found or not updated."
        )


async def set_video_not_fetched(video_registry_id: int) -> None:
    """
    Sets status = 'not_fetched'. Used on infrastructure failure to release
    the claim and allow retry. Never called for content failures.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("youtube_video_url_registry")
        .update({"status": "not_fetched"})
        .eq("video_registry_id", video_registry_id)
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            f"set_video_not_fetched() failed — video_registry_id={video_registry_id} not found or not updated."
        )
