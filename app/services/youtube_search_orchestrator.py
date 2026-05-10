import logging
import re

from app.repositories.url_registry_repository import get_gsmarena_anchor, check_connectivity_variants
from app.repositories.youtube_channel_repository import get_active_channels
from app.repositories.youtube_search_log_repository import (
    create_search_log,
    close_search_log,
    update_url_registry_search_denorm,
)
from app.repositories.youtube_video_registry_repository import upsert_video
from app.core.supabase_client import get_client
from app.services.youtube_api_client import search_videos_for_channel, fetch_video_descriptions
from app.services.video_relevance_filter import (
    extract_search_tokens, _normalize_text, _matches
)
from app.services.youtube_llm_filter import stage1_prefilter, llm_filter_videos

logger = logging.getLogger(__name__)


def prioritize_by_connectivity(
    videos: list[dict],
    descriptions: dict[str, str],
    target_connectivity: str | None,
    has_4g: bool,
    has_5g: bool,
) -> list[dict]:
    """
    Reorders videos to prioritize those matching the target connectivity (4G/5G).
    Only applies if BOTH 4G and 5G variants exist in the registry for this phone.
    """
    # 1. If not both variants exist, prioritization is unnecessary
    if not (has_4g and has_5g):
        return videos

    # 2. If no connectivity was requested (e.g. searching "Pixel 8"), return as-is
    if not target_connectivity:
        return videos

    preferred = []
    others = []

    for video in videos:
        # 3. Combine title and description for variant detection
        title = video.get("video_title") or ""
        description = descriptions.get(video.get("yt_video_id", ""), "")
        # Normalize text — same pipeline as score_video.
        # _normalize_text expands "5g" → "5 g" (digit→letter step), so we cannot use
        # _matches(text, "5g"). Instead use an inline pattern that accounts for the space.
        text = _normalize_text(f"{title} {description}")

        # 4. Match connectivity using space-aware pattern against normalized text
        if target_connectivity == "5g":
            conn_pattern = r'\b5\s+g\b'
        else:
            conn_pattern = r'\b4\s+g\b'
        if re.search(conn_pattern, text):
            preferred.append(video)
        else:
            others.append(video)

    # 5. Return ordered list (preferred first) or original list if no preferred found
    if preferred:
        return preferred + others
    else:
        return videos


_SIGNAL_RANK:     dict[str, int] = {"high": 0, "medium": 1, "low": 2}
_CONFIDENCE_RANK: dict[str, int] = {"high": 0, "medium": 1, "low": 2}

# Maximum transcripts to register per phone.
# Applied AFTER Stage 2 LLM filter, BEFORE upsert.
# Videos beyond the cap are never registered, fetched, or extracted.
# Extend with per-phone category logic once category tagging is added.
# Suggested values: flagship=12, midrange=10, budget=5.
_TRANSCRIPT_CAP_DEFAULT: int = 10

# Maximum Stage 1 survivors passed to Stage 2 LLM classification.
# Prevents cost explosion on popular phones with many raw search results.
_PRE_FILTER_CAP: int = 20


def _apply_transcript_cap(videos: list[dict], cap: int) -> list[dict]:
    """
    Sorts passing videos by LLM signal quality and keeps the top `cap` videos.
    Sort key: (signal_strength rank, confidence rank) — ascending = best first.
    Videos without _llm_* keys (e.g. no description available) default to medium/medium.
    Returns the full list unchanged if len(videos) <= cap.
    """
    if len(videos) <= cap:
        return videos

    def _sort_key(v: dict) -> tuple[int, int]:
        sig  = _SIGNAL_RANK.get(v.get("_llm_signal_strength", "medium"), 1)
        conf = _CONFIDENCE_RANK.get(v.get("_llm_confidence", "medium"), 1)
        return (sig, conf)

    sorted_videos = sorted(videos, key=_sort_key)
    capped = sorted_videos[:cap]
    logger.info(
        "_apply_transcript_cap: %d videos → capped to %d (dropped %d lower-signal videos)",
        len(videos), len(capped), len(videos) - len(capped),
    )
    return capped


async def run_youtube_search(brand: str, model_name: str, triggered_by: str) -> dict:
    """
    Full search flow for one phone.
    Returns dict with: success, search_log_id, videos_found,
                       new_videos_registered, channels_searched, message
    """
    search_log_id = None
    url_registry_id = None

    try:
        # 1. Resolve canonical GSMArena anchor — fail immediately if missing
        anchor = await get_gsmarena_anchor(brand, model_name)
        url_registry_id = anchor["url_id"]

        # 2. Open search log (safe default status = 'failed' until closed)
        search_log_id = await create_search_log(url_registry_id, triggered_by)

        # 3. Fetch all active channels
        channels = await get_active_channels()
        if not channels:
            raise ValueError(
                "No active channels in youtube_channels. Seed the table before running searches."
            )

        # 4. Search each channel — one channel failure does not abort the full run
        query = f"{brand} {model_name}"
        all_videos = []
        failed_channels = 0

        # Pre-calculate tokens and variant info outside the loop for performance
        tokens = extract_search_tokens(brand, model_name)
        target_conn = tokens["connectivity"]  # already "5g" or "4g" or None — do NOT normalize, _normalize_text("5g") → "5 g" (with space) which never matches
        variant_info = await check_connectivity_variants(brand, model_name)

        for channel in channels:
            try:
                videos = await search_videos_for_channel(channel["yt_channel_id"], query)

                if videos:
                    # Fetch descriptions in one batched API call (1 quota unit per channel batch)
                    video_ids = [v["yt_video_id"] for v in videos]
                    descriptions = await fetch_video_descriptions(video_ids)

                    # Stage 1: Pure presence check — no scoring, no penalties.
                    # Eliminates videos with no brand/number relationship to the target.
                    # Passes ~30/70 raw videos through to Stage 2.
                    videos = stage1_prefilter(videos, descriptions, brand, model_name)

                    # Stage 2 pre-filter cap: limit LLM classification calls on popular phones.
                    # Applied BEFORE Stage 2 to prevent cost explosion.
                    videos = videos[:_PRE_FILTER_CAP]

                    # Stage 2: LLM semantic verification of Stage 1 survivors.
                    # Handles description pollution, comparison videos, variant suffix
                    # disambiguation, and successor-framing that presence checks cannot resolve.
                    # All survivors classified in parallel via asyncio.gather.
                    videos = await llm_filter_videos(videos, descriptions, brand, model_name)

                    # Prioritize by connectivity variant (4G vs 5G) — runs on final filtered set
                    videos = prioritize_by_connectivity(
                        videos=videos,
                        descriptions=descriptions,
                        target_connectivity=target_conn,
                        has_4g=variant_info["has_4g"],
                        has_5g=variant_info["has_5g"],
                    )

                for v in videos:
                    v["channel_id"] = channel["channel_id"]
                all_videos.extend(videos)

            except Exception as e:
                failed_channels += 1
                logger.warning(
                    "Channel %r failed during search for %s %s: %s",
                    channel.get("channel_name"), brand, model_name, e
                )
                continue

        # 4.5. Deduplicate across channels (first occurrence wins), then apply transcript cap.
        # Dedup is needed because the same video may surface from multiple channel searches.
        # Cap is applied BEFORE upsert: videos beyond the cap are never registered and
        # therefore never fetched or extracted. Primary cost and review-load lever.
        seen_ids: set[str] = set()
        deduped: list[dict] = []
        for v in all_videos:
            vid_id = v.get("yt_video_id")
            if vid_id and vid_id not in seen_ids:
                seen_ids.add(vid_id)
                deduped.append(v)
        all_videos = deduped

        videos_found_total = len(all_videos)
        all_videos = _apply_transcript_cap(all_videos, _TRANSCRIPT_CAP_DEFAULT)

        # 5. Upsert all found videos — ON CONFLICT DO NOTHING, existing rows untouched.
        # Hard ceiling: never register more than _TRANSCRIPT_CAP_DEFAULT videos per phone
        # total, across all runs. Per-run cap (_apply_transcript_cap) already limits the
        # candidate list — this ceiling ensures re-runs never inflate the total count.
        existing_result = (
            get_client()
            .schema("pipeline")
            .table("youtube_video_url_registry")
            .select("video_registry_id", count="exact")
            .eq("url_registry_id", url_registry_id)
            .execute()
        )
        existing_count = existing_result.count or 0
        slots_remaining = max(0, _TRANSCRIPT_CAP_DEFAULT - existing_count)

        new_count = 0
        for v in all_videos[:slots_remaining]:
            inserted = await upsert_video(
                url_registry_id=url_registry_id,
                channel_id=v["channel_id"],
                yt_video_id=v["yt_video_id"],
                video_url=v["video_url"],
                video_title=v.get("video_title"),
                published_at=v.get("published_at"),
                video_type=v.get("_llm_video_type"),
            )
            if inserted:
                new_count += 1

        # 6. Determine terminal status — zero results is valid, not failure
        if failed_channels == len(channels):
            search_status = "failed"
        elif all_videos:
            search_status = "success_with_results"
        else:
            search_status = "success_zero"

        # 7. Close search log
        await close_search_log(
            search_log_id=search_log_id,
            search_status=search_status,
            videos_found=videos_found_total,
            channels_searched=len(channels),
            error_message=None,
        )

        # 8. Update denormalized fast-access columns on url_registry (only on success)
        if search_status != "failed":
            await update_url_registry_search_denorm(url_registry_id)

        return {
            "success": True,
            "search_log_id": search_log_id,
            "videos_found": videos_found_total,
            "new_videos_registered": new_count,
            "channels_searched": len(channels),
            "message": search_status,
        }

    except Exception as e:
        if search_log_id is not None:
            await close_search_log(
                search_log_id=search_log_id,
                search_status="failed",
                videos_found=0,
                channels_searched=0,
                error_message=str(e),
            )
        return {
            "success": False,
            "search_log_id": search_log_id,
            "videos_found": 0,
            "new_videos_registered": 0,
            "channels_searched": 0,
            "message": str(e),
        }
