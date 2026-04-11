"""YouTube channel repository — fetches active channels from pipeline.youtube_channels."""

from app.core.supabase_client import get_client


async def get_active_channels() -> list[dict]:
    """
    Fetches all rows from pipeline.youtube_channels where is_active = TRUE.

    Returns a list of dicts. Each dict includes:
        channel_id, yt_channel_id, channel_name, channel_handle,
        language, is_india_market

    Returns an empty list if no active channels exist — does not raise.
    The orchestrator is responsible for handling the zero-channel case.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("youtube_channels")
        .select(
            "channel_id, yt_channel_id, channel_name, channel_handle, "
            "language, is_india_market"
        )
        .eq("is_active", True)
        .execute()
    )

    return result.data or []
