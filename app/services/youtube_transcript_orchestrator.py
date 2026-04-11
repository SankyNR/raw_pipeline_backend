"""Full atomic transcript pipeline per video — fetch SRT, translate if needed, process to TXT."""

import logging
from datetime import datetime

from fastapi import HTTPException
from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled, VideoUnavailable

from app.repositories.youtube_execution_repository import (
    create_transcript_execution,
    finish_transcript_execution,
)
from app.repositories.youtube_raw_transcript_repository import insert_raw_transcript_data
from app.repositories.youtube_video_registry_repository import (
    claim_video_for_fetching,
    get_phone_info_for_video,
    get_video_row,
    set_video_failed,
    set_video_fetched_raw,
    set_video_not_fetched,
)
from app.services.srt_processor import process_srt_to_txt
from app.services.storage_service import delete_file, upload_file
from app.services.translation_service import translate_to_english
from app.services.youtube_transcript_client import fetch_transcript_srt, get_language_priority
from app.utils.path_builder import build_youtube_storage_paths

logger = logging.getLogger(__name__)

# Content failures are terminal — the video has no usable transcript.
# Infrastructure failures are retryable — network, quota, storage, etc.
CONTENT_FAILURE_TYPES = (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable)


# ---------------------------------------------------------------------------
# Task 16.1 — run_transcript_pipeline()
# ---------------------------------------------------------------------------

async def run_transcript_pipeline(video_registry_id: int) -> dict:
    """
    Full atomic transcript pipeline for one video.
    Fetch SRT → Translate (if non-English) → Process to TXT → Insert DB row → Mark fetched_raw.
    Returns dict with: success, exec_id, message.

    Invariants:
    1. exec_id = None init — failure path never crashes calling finish_transcript_execution
       on a non-existent row.
    2. claimed flag — set_video_failed/set_video_not_fetched only called if we claimed the row.
    3. video_reg_id = None init — status reset skipped if video row fetch failed.
    4. srt_uploaded flag — derivative cleanup only runs if SRT was successfully uploaded.
    5. started_at = None init — execution log close only called if both exec_id and started_at set.
    6. HTTPException (409) re-raises directly — we never owned the row, no status reset needed.
    7. Content failures → set_video_failed (terminal).
       All other exceptions → set_video_not_fetched (retryable).
    8. SRT is never deleted after srt_uploaded = True. Only processed_path and
       translated_path are cleaned up on failure.
    """
    # Initialize to None — failure path must never assume these were set
    exec_id = None
    video_reg_id = None
    claimed = False
    srt_uploaded = False
    paths = None
    started_at = None
    language_code = None

    try:
        # 1. Fetch video row with channel info (language, channel_name, yt_channel_id)
        video_row = await get_video_row(video_registry_id)
        video_reg_id = video_row["video_registry_id"]
        channel_language = video_row["language"]

        # 2. Resolve phone identity for storage path construction
        phone_info = await get_phone_info_for_video(video_registry_id)

        # 3. Build all storage paths upfront — before any claim or upload
        paths = build_youtube_storage_paths(
            brand=phone_info["brand"],
            model=phone_info["model_name"],
            channel_name=video_row["channel_name"],
            yt_channel_id=video_row["yt_channel_id"],
            yt_video_id=video_row["yt_video_id"],
        )

        # 4. Atomic claim — returns False if already currently_fetching
        claimed = await claim_video_for_fetching(video_reg_id)
        if not claimed:
            raise HTTPException(
                status_code=409,
                detail="Pipeline already running for this video."
            )

        # 5. Pre-emptively delete derivative files from any previous failed run.
        #    This ensures re-fetch always starts clean.
        #    The SRT is intentionally NOT deleted — it is overwritten at Step 7.
        await delete_file(paths["processed_path"])
        await delete_file(paths["translated_path"])

        # 6. Open execution log with orchestrator clock timestamp
        started_at = datetime.utcnow()
        exec_id = await create_transcript_execution(video_reg_id, started_at)

        # 7. Determine language fetch priority from channel language
        lang_priority = get_language_priority(channel_language)

        # === STEP 1: FETCH AND STORE SRT ===
        srt_content, language_code, is_auto_generated = await fetch_transcript_srt(
            video_row["yt_video_id"], lang_priority
        )

        srt_bytes = srt_content.encode("utf-8")
        await upload_file(paths["srt_path"], srt_bytes, "text/plain")
        srt_uploaded = True
        # SRT is now the immutable artifact. It is never deleted from this point,
        # even if all downstream steps fail.

        # === STEP 2: TRANSLATE (non-English transcripts only) ===
        needs_translation = language_code != "en"
        translated_path = None

        if needs_translation:
            translated_text = await translate_to_english(srt_content)
            translated_bytes = translated_text.encode("utf-8")
            await upload_file(paths["translated_path"], translated_bytes, "text/plain")
            translated_path = paths["translated_path"]
            text_for_processing = translated_text
        else:
            text_for_processing = srt_content

        # === STEP 3: PROCESS TO TXT ===
        # For English: strips SRT structure from srt_content
        # For Hindi: strips any residual structure from translated_text (plain text passes through)
        processed_text = process_srt_to_txt(text_for_processing)
        if not processed_text.strip():
            raise ValueError("Processed transcript is empty after cleaning.")

        processed_bytes = processed_text.encode("utf-8")
        await upload_file(paths["processed_path"], processed_bytes, "text/plain")

        # === STEP 4: COMPLETE ===
        await insert_raw_transcript_data(
            video_registry_id=video_reg_id,
            transcript_exec_id=exec_id,
            srt_path=paths["srt_path"],
            processed_transcript_path=paths["processed_path"],
            translated_transcript_path=translated_path,
            file_size_bytes=len(srt_bytes),
            language_code=language_code,
            is_auto_generated=is_auto_generated,
            translation_status="translation_complete" if needs_translation else "not_required",
            translation_language_code="en" if needs_translation else None,
        )

        finished_at = datetime.utcnow()
        await finish_transcript_execution(
            exec_id, True, started_at, finished_at, language_code=language_code
        )
        await set_video_fetched_raw(video_reg_id)

        return {
            "success": True,
            "exec_id": exec_id,
            "message": "Pipeline completed successfully",
        }

    except HTTPException:
        raise  # 409 passes through unchanged — do not reset status

    except Exception as e:
        finished_at = datetime.utcnow()
        is_content_failure = isinstance(e, CONTENT_FAILURE_TYPES)

        # Clean up derivative files — but NEVER the SRT
        if srt_uploaded and paths is not None:
            await delete_file(paths["processed_path"])
            await delete_file(paths["translated_path"])

        # Log execution failure if the log row was created
        if exec_id is not None and started_at is not None:
            await finish_transcript_execution(
                exec_id,
                False,
                started_at,
                finished_at,
                language_code=language_code,
                error_message=str(e)
            )

        # Reset video status based on failure type
        if claimed and video_reg_id is not None:
            if is_content_failure:
                await set_video_failed(video_reg_id)       # terminal — not retried
            else:
                await set_video_not_fetched(video_reg_id)  # retryable

        return {
            "success": False,
            "exec_id": exec_id,
            "message": str(e),
        }
