"""
YouTube API Router.

Phase 7  (Task 7.1):  POST /admin/youtube/search
Phase 8  (Task 8.1):  GET  /admin/youtube/phones
Phase 8  (Task 8.2):  GET  /admin/youtube/videos
Phase 17 (Task 17.1): POST /admin/youtube/fetch-transcript
Phase 18 (Task 18.1): GET  /admin/youtube/transcript/{video_registry_id}/srt
Phase 5  (Task 5.1):  POST /admin/youtube/debug/score-video
Phase 20 (Task 20.1): POST /admin/youtube/fetch-transcripts-for-model
"""

import asyncio
import logging
import random

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.core.supabase_client import get_client as _get_client
from app.repositories.url_registry_repository import get_gsmarena_anchor
from app.repositories.youtube_raw_transcript_repository import get_raw_transcript_data
from app.repositories.youtube_video_registry_repository import (
    get_videos_for_phone,
    reset_stale_video_locks,
)
from app.repositories.pipeline_run_repository import (
    create_pipeline_run,
    update_pipeline_run,
)
from app.core.constants import PipelineStage
from app.services.youtube_search_orchestrator import run_youtube_search
from app.services.youtube_transcript_orchestrator import run_transcript_pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tracking helper (fire-and-forget, never blocks the batch loop)
# ---------------------------------------------------------------------------

async def _track(pipeline_run_id: str | None, **kwargs) -> None:
    """Non-blocking pipeline_runs update. Swallows all exceptions."""
    if pipeline_run_id is None:
        return
    async def _do():
        try:
            await asyncio.to_thread(update_pipeline_run, pipeline_run_id, **kwargs)
        except Exception as exc:
            logger.debug("_track: update_pipeline_run failed silently: %s", exc)
    asyncio.create_task(_do())

router = APIRouter(prefix="/admin", tags=["youtube"])

_VALID_TRIGGERED_BY = {"initial", "admin_manual", "scheduled"}

_VIDEO_STATUS_BADGE: dict[str, str] = {
    "not_fetched":        "gray",
    "currently_fetching": "yellow",
    "fetched_raw":        "green",
    "failed":             "red",
}

_SRT_SIGNED_URL_EXPIRY_SECONDS = 3600
_RAW_FILES_BUCKET = "raw_files"

# Delay between consecutive transcript pipeline calls to avoid YouTube rate-limiting.
# Applied as jitter: random.uniform(delay_seconds, delay_seconds * 1.5)
# Tune via the delay_seconds field in FetchTranscriptsForModelRequest without a code change.
_INTER_TRANSCRIPT_DELAY_SECONDS = 10
_INTER_TRANSCRIPT_DELAY_MAX_MULTIPLIER = 1.5

# ---------------------------------------------------------------------------
# S3-P1-4 / S3-P1-5 — Per-video retry helper with timeout
# ---------------------------------------------------------------------------
# Wraps run_transcript_pipeline with:
#   - asyncio.wait_for timeout (90s) to prevent a hung video from blocking the batch
#   - 1 retry for transient failures (timeout, connection, proxy, rate-limit)
# Non-transient failures (no transcript, parse error) are NOT retried.
# ---------------------------------------------------------------------------
_TRANSCRIPT_TIMEOUT_SECONDS = 90.0
_TRANSCRIPT_MAX_ATTEMPTS    = 2
_TRANSCRIPT_RETRY_DELAY     = 3.0


async def _run_with_retry(
    video_registry_id: int,
    proxy_session_id: str | None = None,
) -> dict:
    """
    Runs run_transcript_pipeline with a per-video timeout and one transient retry.

    Retries on: asyncio.TimeoutError, and exceptions containing 'timeout',
    'connection', 'proxy', '429', or 'rate' in their message.
    Non-transient failures are re-raised immediately without sleeping.
    """
    last_exc: Exception | None = None
    for attempt in range(_TRANSCRIPT_MAX_ATTEMPTS):
        try:
            return await asyncio.wait_for(
                run_transcript_pipeline(video_registry_id, proxy_session_id=proxy_session_id),
                timeout=_TRANSCRIPT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            last_exc = exc
            logger.warning(
                "_run_with_retry: video_registry_id=%d attempt %d/%d timed out after %.0fs",
                video_registry_id, attempt + 1, _TRANSCRIPT_MAX_ATTEMPTS,
                _TRANSCRIPT_TIMEOUT_SECONDS,
            )
            if attempt < _TRANSCRIPT_MAX_ATTEMPTS - 1:
                await asyncio.sleep(_TRANSCRIPT_RETRY_DELAY)
        except Exception as exc:
            err_str = str(exc).lower()
            is_transient = any(k in err_str for k in (
                "timeout", "connection", "proxy", "429", "rate"
            ))
            if is_transient and attempt < _TRANSCRIPT_MAX_ATTEMPTS - 1:
                last_exc = exc
                logger.warning(
                    "_run_with_retry: video_registry_id=%d attempt %d/%d transient error: %s",
                    video_registry_id, attempt + 1, _TRANSCRIPT_MAX_ATTEMPTS, exc,
                )
                await asyncio.sleep(_TRANSCRIPT_RETRY_DELAY)
            else:
                raise  # non-transient or exhausted — propagate immediately
    raise last_exc  # exhausted all attempts


# ---------------------------------------------------------------------------
# Phase 7 — Task 7.1: POST /admin/youtube/search
# ---------------------------------------------------------------------------

class YouTubeSearchRequest(BaseModel):
    model_config = {"protected_namespaces": ()}  # suppress false-positive on model_name field

    brand:        str
    model_name:   str
    triggered_by: str


@router.post("/youtube/search")
async def trigger_youtube_search(body: YouTubeSearchRequest):
    """
    Triggers a full YouTube video search for a single (brand, model_name) phone.

    Validates triggered_by against the allowed enum before running.

    Success (HTTP 200):
        {
            "success": true,
            "search_log_id": 12,
            "videos_found": 8,
            "new_videos_registered": 3,
            "channels_searched": 12,
            "message": "success_with_results"
        }

    Invalid triggered_by (HTTP 400):
        { "detail": "Invalid triggered_by value: '...'. Must be one of: initial, admin_manual, scheduled." }

    Any other failure (HTTP 500):
        { "detail": "<error message>" }
    """
    if body.triggered_by not in _VALID_TRIGGERED_BY:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid triggered_by value: {body.triggered_by!r}. "
                f"Must be one of: {', '.join(sorted(_VALID_TRIGGERED_BY))}."
            ),
        )

    try:
        result = await run_youtube_search(body.brand, body.model_name, body.triggered_by)

        # run_youtube_search() catches all errors and returns a dict with success=False.
        # It never re-raises — so map success=False to HTTP 500 here.
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])

        return result

    except HTTPException:
        raise  # 400 and 500 pass through unchanged


# ---------------------------------------------------------------------------
# Phase 8 — Task 8.1: GET /admin/youtube/phones
# ---------------------------------------------------------------------------

@router.get("/youtube/phones")
async def get_youtube_phones(brand: str = Query(..., description="Brand name to filter by")):
    """
    Returns all phones for the given brand (gsmarena anchor rows only) with
    their most recent YouTube search info and video counts.

    Response:
        {
          "phones": [
            {
              "brand": "Apple",
              "model_name": "iPhone 13",
              "url_registry_id": 42,
              "last_searched_at": "2025-01-15T10:30:00Z",
              "search_count": 3,
              "last_search_status": "success_with_results",
              "videos_total": 8,
              "videos_fetched_raw": 5,
              "videos_failed": 1,
              "videos_not_fetched": 2
            }
          ]
        }

    `last_searched_at` and `last_search_status` are null if the phone has never been searched.
    """
    try:
        result = (
            _get_client()
            .rpc("get_youtube_phones_for_brand", {"_brand": brand})
            .execute()
        )
    except Exception as e:
        logger.error("get_youtube_phones DB error (brand=%s): %s", brand, e)
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    return {"phones": result.data or []}


# ---------------------------------------------------------------------------
# Phase 8 — Task 8.2: GET /admin/youtube/videos
# ---------------------------------------------------------------------------

@router.get("/youtube/videos")
async def get_youtube_videos(
    brand: str  = Query(..., description="Brand name"),
    model: str  = Query(..., description="Model name"),
):
    """
    Returns all registered YouTube videos for the given phone, with badge_color per video.

    Response:
        {
          "videos": [
            {
              "video_registry_id": 101,
              "yt_video_id": "dQw4w9WgXcQ",
              "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
              "video_title": "iPhone 13 Review",
              "published_at": "2025-01-10T08:00:00Z",
              "status": "fetched_raw",
              "badge_color": "green",
              "channel_name": "Trakin Tech English",
              "channel_handle": "@TrakinTechEnglish",
              "language": "English"
            }
          ]
        }

    Returns 404 if no GSMArena anchor exists for this phone.
    Returns empty list if anchor exists but no videos are registered yet.
    """
    try:
        anchor = await get_gsmarena_anchor(brand, model)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("get_youtube_videos anchor error (brand=%s, model=%s): %s", brand, model, e)
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    try:
        videos = await get_videos_for_phone(anchor["url_id"])
    except Exception as e:
        logger.error("get_youtube_videos fetch error (url_registry_id=%s): %s", anchor["url_id"], e)
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    for v in videos:
        v["badge_color"] = _VIDEO_STATUS_BADGE.get(v.get("status", ""), "gray")

    return {"videos": videos}


# ---------------------------------------------------------------------------
# Phase 17 — Task 17.1: POST /admin/youtube/fetch-transcript
# ---------------------------------------------------------------------------

class FetchTranscriptRequest(BaseModel):
    video_registry_id: int


@router.post("/youtube/fetch-transcript")
async def trigger_fetch_transcript(body: FetchTranscriptRequest):
    """
    Triggers the full transcript pipeline for a single registered video.

    Success (HTTP 200):
        { "success": true, "exec_id": 88, "message": "Pipeline completed successfully" }

    Already in progress (HTTP 409):
        { "detail": "Pipeline already running for this video." }

    Any other failure (HTTP 500):
        { "detail": "<error message>" }
    """
    try:
        result = await run_transcript_pipeline(body.video_registry_id)

        # run_transcript_pipeline() catches all non-409 errors and returns
        # a dict with success=False. Map success=False to HTTP 500 here.
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])

        return result

    except HTTPException:
        raise  # 409 and 500 pass through unchanged


# ---------------------------------------------------------------------------
# Phase 18 — Task 18.1: GET /admin/youtube/transcript/{video_registry_id}/srt
# ---------------------------------------------------------------------------

@router.get("/youtube/transcript/{video_registry_id}/srt")
async def get_transcript_srt(video_registry_id: int):
    """
    Returns a signed Supabase Storage URL for the SRT file of a given video.
    Never streams file bytes directly — offloads bandwidth to Supabase Storage.

    Returns 404 if no transcript row exists for this video yet.

    Success (HTTP 200):
        {
          "signed_url": "https://...",
          "expires_in": 3600,
          "srt_path": "samsung/galaxy-s25/raw_transcripts/..."
        }
    """
    row = await get_raw_transcript_data(video_registry_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail="No transcript data found for this video. Run the pipeline first.",
        )

    srt_path = row["srt_path"]
    try:
        response = (
            _get_client()
            .storage
            .from_(_RAW_FILES_BUCKET)
            .create_signed_url(srt_path, _SRT_SIGNED_URL_EXPIRY_SECONDS)
        )
        # Support BOTH dict (v1/mock) and object (v2) responses
        signed_url = None
        if isinstance(response, dict):
            signed_url = response.get("signedURL") or response.get("signedUrl")
        elif hasattr(response, "data"):
            # supabase-py v2 returns a response object with a .data attribute
            data = response.data
            if isinstance(data, dict):
                signed_url = data.get("signedURL") or data.get("signedUrl")
            else:
                signed_url = getattr(data, "signedURL", None) or getattr(data, "signedUrl", None)

        if not signed_url:
            logger.error("Failed to extract signed URL from response: %s", response)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate signed URL. Response: {response}"
            )
    except HTTPException:
        raise  # 500 from above passed through
    except Exception as e:
        logger.error(
            "get_transcript_srt signed URL error (video_registry_id=%s, path=%s): %s",
            video_registry_id, srt_path, e,
        )
        raise HTTPException(status_code=500, detail=f"Storage error: {e}")

    return {
        "signed_url": signed_url,
        "expires_in": _SRT_SIGNED_URL_EXPIRY_SECONDS,
        "srt_path": srt_path,
    }


# ---------------------------------------------------------------------------
# Phase 5 — Task 5.1: POST /admin/youtube/debug/score-video
# ---------------------------------------------------------------------------

class ScoreVideoRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    brand:       str
    model_name:  str
    yt_video_id: str
    video_title: str


@router.post("/youtube/debug/score-video")
async def debug_score_video(body: ScoreVideoRequest):
    """
    Diagnostic endpoint — runs Stage 1 presence check on a single video without upserting.
    Fetches the video description live from YouTube, runs extract_search_tokens() and
    score_video() for inspection, then reports whether Stage 1 would pass the video.

    scorer_score is informational only — it is NOT used in the live pipeline.
    The live pipeline uses Stage 1 (presence check) + Stage 2 (LLM) instead.
    Do NOT use in production flows.
    """
    from app.services.youtube_api_client import fetch_video_descriptions
    from app.services.video_relevance_filter import extract_search_tokens, score_video
    from app.services.youtube_llm_filter import stage1_prefilter

    descriptions = await fetch_video_descriptions([body.yt_video_id])
    description = descriptions.get(body.yt_video_id, "")
    tokens = extract_search_tokens(body.brand, body.model_name)
    score = score_video(body.video_title, description, body.brand, body.model_name)

    # Run Stage 1 presence check on this single video
    video_dict = {"yt_video_id": body.yt_video_id, "video_title": body.video_title}
    desc_map   = {body.yt_video_id: description}
    passes_stage1 = bool(
        stage1_prefilter([video_dict], desc_map, body.brand, body.model_name)
    )

    return {
        "yt_video_id":          body.yt_video_id,
        "video_title":         body.video_title,
        "description_preview":  description[:300] if description else "",
        "tokens":              tokens,
        "scorer_score":         score,
        "passes_stage1":        passes_stage1,
        "stage1_method":        "presence_check (brand alias + model number in text)",
        "note": (
            "passes_stage1=true means this video goes to LLM Stage 2 verification. "
            "It does not mean the video will be stored \u2014 LLM may still reject it. "
            "scorer_score is informational only; it is NOT used in the live pipeline."
        ),
    }


# ---------------------------------------------------------------------------
# Phase 20 — Task 20.1: POST /admin/youtube/fetch-transcripts-for-model
# ---------------------------------------------------------------------------

class FetchTranscriptsForModelRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    brand: str
    model_name: str
    skip_already_fetched: bool = True
    """
    If True (default): skip videos with status='fetched_raw'. Only process not_fetched and failed.
    If False: run pipeline for ALL registered videos regardless of current status.
    """
    delay_seconds: float = _INTER_TRANSCRIPT_DELAY_SECONDS
    """
    Seconds to sleep between consecutive transcript pipeline calls.
    Defaults to _INTER_TRANSCRIPT_DELAY_SECONDS (10 s). Set to 0 to disable throttling.
    """


@router.post("/youtube/fetch-transcripts-for-model")
async def trigger_fetch_transcripts_for_model(body: FetchTranscriptsForModelRequest):
    """
    Triggers the full transcript pipeline for every registered video belonging
    to the given (brand, model_name) phone.

    Runs sequentially — one video at a time — to avoid rate-limiting against
    YouTube's transcript API and the Gemini LLM.

    skip_already_fetched=True (default):
        Only processes videos with status 'not_fetched' or 'failed'.
        Videos already at 'fetched_raw' are counted in skipped_already_fetched and skipped.

    skip_already_fetched=False:
        Runs the pipeline for ALL registered videos regardless of current status.

    Per-video outcome field (status):
        "ok"               — pipeline returned success=True
        "failed"           — pipeline returned success=False (graceful content failure)
        "skipped_in_progress" — HTTP 409 raised (another run is currently fetching this video)
        "error"            — unexpected exception; logged and skipped

    Returns 404 if no GSMArena anchor exists for (brand, model_name).
    """
    # Resolve url_registry_id via GSMArena anchor
    try:
        anchor = await get_gsmarena_anchor(body.brand, body.model_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    url_registry_id: int = anchor["url_id"]

    # Auto-recover any orphaned workers that crashed/timed out mid-fetch (>15 mins ago)
    try:
        stale_reset_count = await reset_stale_video_locks(url_registry_id, stale_minutes=15)
        if stale_reset_count > 0:
            logger.warning(
                "fetch_transcripts_for_model: recovered %d stale 'currently_fetching' locks "
                "for url_registry_id=%s", stale_reset_count, url_registry_id
            )
    except Exception as e:
        logger.error("Failed to reset stale video locks: %s", e)

    # Fetch all registered videos for this phone
    videos = await get_videos_for_phone(url_registry_id)
    total_videos = len(videos)

    # Optionally filter out already-fetched videos
    skipped_already_fetched = 0
    if body.skip_already_fetched:
        to_process = [v for v in videos if v.get("status") != "fetched_raw"]
        skipped_already_fetched = total_videos - len(to_process)
    else:
        to_process = list(videos)

    if not to_process:
        return {
            "brand":                   body.brand,
            "model_name":              body.model_name,
            "total_videos":            total_videos,
            "processed":               0,
            "succeeded":               0,
            "failed":                  0,
            "skipped_in_progress":     0,
            "skipped_already_fetched": skipped_already_fetched,
            "results":                 [],
        }

    logger.info(
        "fetch_transcripts_for_model: starting batch of %d videos "
        "(brand=%s, model=%s)",
        len(to_process), body.brand, body.model_name,
    )

    # Create a pipeline_runs tracking row for the entire batch.
    # If this fails, pipeline_run_id stays None and all _track() calls are no-ops.
    pipeline_run_id: str | None = None
    try:
        pipeline_run_id = await asyncio.to_thread(
            create_pipeline_run, url_registry_id, "youtube_fetch", len(to_process)
        )
        await _track(pipeline_run_id,
                     current_stage=PipelineStage.TRANSCRIPT_PIPELINE,
                     current_step=f"Starting batch of {len(to_process)} videos...")
    except Exception as _pr_exc:
        logger.warning("fetch_transcripts_for_model: could not create pipeline_runs row: %s", _pr_exc)

    succeeded = 0
    failed = 0
    skipped_in_progress = 0
    results: list[dict] = []

    last_index = len(to_process) - 1
    for idx, video in enumerate(to_process):
        vid_id    = video["video_registry_id"]
        yt_vid_id = video.get("yt_video_id", "")
        title     = video.get("video_title", "")

        try:
            # S3-P1-4: _run_with_retry adds a 90s timeout + 1 transient retry
            result = await _run_with_retry(vid_id, proxy_session_id=None)

            if result["success"]:
                succeeded += 1
                results.append({
                    "video_registry_id": vid_id,
                    "yt_video_id":       yt_vid_id,
                    "video_title":       title,
                    "status":            "ok",
                    "exec_id":           result.get("exec_id"),
                })
                await _track(pipeline_run_id,
                             processed_items=succeeded,
                             failed_items=failed,
                             current_step=f"Done {succeeded + failed}/{len(to_process)} — {title!r}: ok")
            else:
                # S3-P1-1: classify why the pipeline reported failure
                msg = result.get("message", "")
                msg_lower = msg.lower()
                if "no transcript" in msg_lower or "transcripts disabled" in msg_lower:
                    failure_reason = "no_transcript"
                elif "429" in msg_lower or "quota" in msg_lower or "rate" in msg_lower:
                    failure_reason = "rate_limited"
                elif "proxy" in msg_lower or "connection" in msg_lower:
                    failure_reason = "network_error"
                else:
                    failure_reason = "pipeline_error"

                failed += 1
                results.append({
                    "video_registry_id": vid_id,
                    "yt_video_id":       yt_vid_id,
                    "video_title":       title,
                    "status":            "failed",
                    "failure_reason":    failure_reason,
                    "message":           msg,
                })
                await _track(pipeline_run_id,
                             processed_items=succeeded,
                             failed_items=failed,
                             current_step=f"Done {succeeded + failed}/{len(to_process)} — {title!r}: failed ({failure_reason})")

        except HTTPException as exc:
            if exc.status_code == 409:
                skipped_in_progress += 1
                results.append({
                    "video_registry_id": vid_id,
                    "yt_video_id":       yt_vid_id,
                    "video_title":       title,
                    "status":            "skipped_in_progress",
                })
            else:
                # Re-raise unexpected HTTP errors (e.g. 500 from run_transcript_pipeline)
                raise

        except Exception as exc:
            # S3-P1-1: classify the unexpected exception type
            exc_str = str(exc).lower()
            if "timeout" in exc_str or isinstance(exc, asyncio.TimeoutError):
                failure_reason = "timeout"
            elif "no transcript" in exc_str or "transcripts disabled" in exc_str:
                failure_reason = "no_transcript"
            elif "429" in exc_str or "quota" in exc_str or "rate" in exc_str:
                failure_reason = "rate_limited"
            elif "proxy" in exc_str or "connection" in exc_str:
                failure_reason = "network_error"
            else:
                failure_reason = "pipeline_error"

            failed += 1
            logger.error(
                "fetch_transcripts_for_model: FAILED for video_registry_id=%d "
                "(yt_video_id=%s, title=%r, failure_reason=%s): %s",
                vid_id, yt_vid_id, title, failure_reason, exc, exc_info=True,
            )
            results.append({
                "video_registry_id": vid_id,
                "yt_video_id":       yt_vid_id,
                "video_title":       title,
                "status":            "error",
                "failure_reason":    failure_reason,
                "message":           str(exc),
            })
            await _track(pipeline_run_id,
                         processed_items=succeeded,
                         failed_items=failed,
                         current_step=f"Done {succeeded + failed}/{len(to_process)} — {title!r}: error ({failure_reason})")

        # Throttle between requests with jitter — skip after the last video.
        # Jitter breaks the deterministic timing signature that anti-bot systems detect.
        if idx < last_index and body.delay_seconds > 0:
            jitter_sleep = random.uniform(body.delay_seconds, body.delay_seconds * _INTER_TRANSCRIPT_DELAY_MAX_MULTIPLIER)
            logger.debug(
                "fetch_transcripts_for_model: sleeping %.1fs (jitter) before next video",
                jitter_sleep,
            )
            await asyncio.sleep(jitter_sleep)

    # S3-P1-3: compute overall_status so admin UI doesn't need to derive from counts
    overall_status = (
        "success"         if failed == 0 and succeeded > 0 else
        "partial_failure" if succeeded > 0 and failed > 0 else
        "failed"          if succeeded == 0 and failed > 0 else
        "no_op"  # all skipped or zero processed
    )

    # Finalise the pipeline_runs row
    from datetime import datetime, timezone
    _completed_at = datetime.now(timezone.utc).isoformat()
    final_db_status = (
        "completed" if overall_status == "success"
        else "partial" if overall_status == "partial_failure"
        else "failed" if overall_status == "failed"
        else "completed"  # no_op = completed with 0 items
    )
    await _track(pipeline_run_id,
                 status=final_db_status,
                 processed_items=succeeded,
                 failed_items=failed,
                 current_step="Done.",
                 completed_at=_completed_at)

    return {
        "overall_status":           overall_status,
        "pipeline_run_id":          pipeline_run_id,
        "brand":                    body.brand,
        "model_name":               body.model_name,
        "total_videos":             total_videos,
        "processed":                len(to_process),
        "succeeded":                succeeded,
        "failed":                   failed,
        "skipped_in_progress":      skipped_in_progress,
        "skipped_already_fetched":  skipped_already_fetched,
        "results":                  results,
    }
