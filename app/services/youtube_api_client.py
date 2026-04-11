"""YouTube Data API v3 client — search videos by channel using httpx."""

import httpx

from app.core.config import YOUTUBE_DATA_API_KEY

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
YOUTUBE_TIMEOUT = httpx.Timeout(30.0)


async def search_videos_for_channel(
    yt_channel_id: str,
    query: str,
    max_results: int = 5,
) -> list[dict]:
    """
    Calls YouTube Data API v3 search.list.
    Filters by channelId={yt_channel_id}, type=video, order=relevance.
    Query is: "{brand} {model_name} review" (constructed by the orchestrator).

    Returns list of dicts, each with:
        yt_video_id    (str, 11 chars)
        video_url      (str, https://www.youtube.com/watch?v={id})
        video_title    (str or None)
        published_at   (str ISO timestamp or None)

    Returns empty list if no results — this is NOT an error.

    Raises ValueError if:
      - YouTube API returns an error response (quota exceeded, invalid key, etc.)
      - Response structure is missing expected fields

    Raises httpx.TimeoutException if request times out.

    YouTube API quota is finite. If you get a quotaExceeded error, stop and
    surface it — do not retry automatically.
    """
    params = {
        "part": "snippet",
        "channelId": yt_channel_id,
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "order": "relevance",
        "videoDuration": "medium",   # excludes Shorts (<4min); adjust to "long" if deep-dives are needed
        "key": YOUTUBE_DATA_API_KEY,
    }

    async with httpx.AsyncClient(timeout=YOUTUBE_TIMEOUT) as client:
        response = await client.get(f"{YOUTUBE_API_BASE}/search", params=params)
        response.raise_for_status()
        data = response.json()

        if response.status_code != 200:
            raise ValueError(
                f"YouTube API HTTP error {response.status_code}: "
                f"{data.get('error', {}).get('message', 'Unknown error')}"
            )

        if "error" in data:
            raise ValueError(
                f"YouTube API error: {data['error'].get('message', str(data['error']))}"
            )

        items = data.get("items", [])
        results = []
        for item in items:
            video_id = item.get("id", {}).get("videoId")
            if not video_id or len(video_id) != 11:
                continue
            snippet = item.get("snippet", {})
            results.append({
                "yt_video_id": video_id,
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "video_title": snippet.get("title"),
                "published_at": snippet.get("publishedAt"),
            })

        return results


# ---------------------------------------------------------------------------
# Phase 2 — Task 2.1: fetch_video_descriptions()
# ---------------------------------------------------------------------------

async def fetch_video_descriptions(video_ids: list[str]) -> dict[str, str]:
    """
    Calls YouTube Data API v3 videos.list to fetch snippet descriptions
    for a batch of video IDs.

    Batches all IDs in a single request — costs 1 quota unit total regardless
    of how many video IDs are included (max 50 per call, well within our usage).

    Returns dict mapping yt_video_id → description string.
    Returns empty string for any video whose description is missing or blank.

    Returns empty dict on API error — do NOT raise. The caller (relevance filter)
    treats a missing description as an empty string and scores on title alone.
    Raising here would cause the entire per-channel search to abort for a
    non-critical enrichment step.

    Args:
        video_ids: list of 11-character YouTube video ID strings.
                   Must not be empty — check before calling.
    """
    if not video_ids:
        return {}

    params = {
        "part": "snippet",
        "id": ",".join(video_ids),
        "key": YOUTUBE_DATA_API_KEY,
    }

    try:
        async with httpx.AsyncClient(timeout=YOUTUBE_TIMEOUT) as client:
            response = await client.get(f"{YOUTUBE_API_BASE}/videos", params=params)
            response.raise_for_status()
            data = response.json()

        return {
            item["id"]: item.get("snippet", {}).get("description", "")
            for item in data.get("items", [])
        }
    except Exception:
        # Description fetch is best-effort — return empty dict, caller handles gracefully
        return {}
