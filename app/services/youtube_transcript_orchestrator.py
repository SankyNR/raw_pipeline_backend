"""Full atomic transcript pipeline per video — fetch SRT, translate if needed, process to TXT."""

import logging
from datetime import datetime

from fastapi import HTTPException
from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled, VideoUnavailable

from app.repositories.youtube_execution_repository import (
    create_transcript_execution,
    finish_transcript_execution,
)
from app.repositories.youtube_raw_transcript_repository import (
    get_raw_transcript_data,
    insert_raw_transcript_data,
    delete_raw_transcript_data_by_execution,
)
from app.repositories.youtube_video_registry_repository import (
    claim_video_for_fetching,
    get_phone_info_for_video,
    get_video_row,
    set_video_failed,
    set_video_fetched_raw,
    set_video_not_fetched,
)
from app.services.srt_processor import process_srt_to_txt
from app.services.storage_service import delete_file, upload_file, StorageDuplicateError
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

async def run_transcript_pipeline(video_registry_id: int, proxy_session_id: str | None = None) -> dict:
    """
    Full atomic transcript pipeline for one video.
    Fetch SRT → Store raw SRT → Strip SRT structure → Translate (if non-English)
    → Store processed TXT → Insert DB row → Mark fetched_raw.
    Returns dict with: success, exec_id, message.

    Language policy: English always wins. fetch_transcript_srt tries
    manual EN → auto EN → manual HI → auto HI. Hindi is only fetched and
    translated when no English transcript exists in any form.

    Files produced: raw .srt + processed .txt only. There is no separate
    translated file — for Hindi videos the translation output IS the
    processed file.

    Args:
        video_registry_id: The video to process.
        proxy_session_id:  Sticky session ID for the residential proxy. Passed directly to
                           fetch_transcript_srt() so all videos in one batch share the same
                           residential IP. Generate once per batch at the API layer (uuid4).
                           None = no sticky session (proxy still used if PROXY_ENABLED=true).

    Invariants:
    1. exec_id = None init — failure path never crashes calling finish_transcript_execution
       on a non-existent row.
    2. claimed flag — set_video_failed/set_video_not_fetched only called if we claimed the row.
    3. video_reg_id = None init — status reset skipped if video row fetch failed.
    4. srt_uploaded flag — processed-file cleanup only runs if SRT was successfully uploaded.
    5. started_at = None init — execution log close only called if both exec_id and started_at set.
    6. HTTPException (409) re-raises directly — we never owned the row, no status reset needed.
    7. Content failures → set_video_failed (terminal).
       All other exceptions → set_video_not_fetched (retryable).
    8. SRT is never deleted after srt_uploaded = True. Only the processed file
       is cleaned up on failure.
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

        # 3.5. Early-exit guard — handles the crash-after-insert scenario.
        #      If a complete youtube_raw_transcript_data row already exists for
        #      this video (translation_status = complete or not_required), the
        #      transcript is fully stored. All that's missing is the
        #      set_video_fetched_raw call (the process was killed right after
        #      the DB insert). Skip the entire pipeline, just fix the status.
        #      This also prevents Step 5's delete_file() from orphaning a valid
        #      processed file that was written by a prior complete run.
        existing_transcript = await get_raw_transcript_data(video_reg_id)
        if existing_transcript and existing_transcript.get("translation_status") in (
            "translation_complete", "not_required"
        ):
            logger.info(
                "run_transcript_pipeline: transcript already complete for "
                "video_registry_id=%d (raw_transcript_id=%s) — "
                "skipping pipeline, marking fetched_raw.",
                video_reg_id,
                existing_transcript.get("raw_transcript_id"),
            )
            await set_video_fetched_raw(video_reg_id)
            return {
                "success": True,
                "exec_id": existing_transcript.get("transcript_exec_id"),
                "message": "Pipeline skipped — transcript data already complete.",
            }

        # 4. Atomic claim — returns False if already currently_fetching
        claimed = await claim_video_for_fetching(video_reg_id)
        if not claimed:
            raise HTTPException(
                status_code=409,
                detail="Pipeline already running for this video."
            )

        # 5. Pre-emptively delete the processed file from any previous failed run.
        #    This ensures re-fetch always starts clean.
        #    The SRT is intentionally NOT deleted — it is overwritten at Step 1.
        #    No translated file exists in the current pipeline (translation output
        #    IS the processed output) so there is nothing else to clean.
        await delete_file(paths["processed_path"])

        # 6. Open execution log with orchestrator clock timestamp
        started_at = datetime.utcnow()
        exec_id = await create_transcript_execution(video_reg_id, started_at)

        # === STEP 1: FETCH AND STORE RAW SRT ===
        # fetch_transcript_srt enforces English-first: manual EN -> auto EN ->
        # manual HI -> auto HI. language_priority is accepted for signature
        # compatibility but no longer drives ordering.
        srt_content, language_code, is_auto_generated = await fetch_transcript_srt(
            video_row["yt_video_id"], get_language_priority(channel_language),
            proxy_session_id=proxy_session_id,
        )

        srt_bytes = srt_content.encode("utf-8")
        try:
            await upload_file(paths["srt_path"], srt_bytes, "text/plain")
        except StorageDuplicateError:
            # 409 Duplicate — raw SRT already exists from a prior partial run.
            # The file is intact; reuse it and continue the pipeline.
            logger.warning(
                "run_transcript_pipeline: raw SRT already exists at %r — "
                "reusing existing file (prior partial run). Continuing pipeline.",
                paths["srt_path"],
            )
        srt_uploaded = True
        # SRT is now the immutable artifact. It is never deleted from this point,
        # even if all downstream steps fail.

        # === STEP 2: STRIP SRT STRUCTURE ===
        # Deterministic, code-based removal of sequence numbers, timestamps and
        # blank lines. Runs BEFORE translation so the LLM only ever sees clean
        # text — no SRT noise to confuse it.
        cleaned_text = process_srt_to_txt(srt_content)
        if not cleaned_text.strip():
            raise ValueError("Transcript is empty after stripping SRT structure.")

        # === STEP 3: TRANSLATE (Hindi / non-English transcripts only) ===
        needs_translation = language_code != "en"
        if needs_translation:
            phone_name = f"{phone_info['brand']} {phone_info['model_name']}"
            final_text = await translate_to_english(cleaned_text, phone_name=phone_name)
            if not final_text.strip():
                raise ValueError("Translation returned empty output.")
        else:
            final_text = cleaned_text

        # === STEP 4: STORE PROCESSED FILE ===
        # For English: cleaned text. For Hindi: translated English text.
        # There is no separate translated file — translation output is the
        # processed output.
        processed_bytes = final_text.encode("utf-8")
        await upload_file(paths["processed_path"], processed_bytes, "text/plain")

        # === STEP 5: COMPLETE ===
        await insert_raw_transcript_data(
            video_registry_id=video_reg_id,
            transcript_exec_id=exec_id,
            srt_path=paths["srt_path"],
            processed_transcript_path=paths["processed_path"],
            translated_transcript_path=None,  # no separate translated file
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

        # Each cleanup step is independently guarded so that a network failure
        # in any one step (e.g. delete_file during an internet outage) does NOT
        # cascade and prevent the final status reset from running.
        # Previously, an unguarded delete_file() failure would exit the entire
        # block before set_video_not_fetched() was reached, permanently stranding
        # the row at status='currently_fetching' with no active worker.

        # Clean up the processed file — but NEVER the SRT.
        # There is no separate translated file in the current pipeline.
        if srt_uploaded and paths is not None:
            try:
                await delete_file(paths["processed_path"])
            except Exception as _del_exc:
                logger.warning(
                    "run_transcript_pipeline: delete processed_path failed "
                    "(video_registry_id=%s): %s — continuing cleanup",
                    video_reg_id, _del_exc,
                )

        # Log execution failure if the log row was created
        if exec_id is not None and started_at is not None:
            try:
                await finish_transcript_execution(
                    exec_id,
                    False,
                    started_at,
                    finished_at,
                    language_code=language_code,
                    error_message=str(e)
                )
            except Exception as _log_exc:
                logger.warning(
                    "run_transcript_pipeline: finish_transcript_execution failed "
                    "(exec_id=%s): %s — continuing cleanup",
                    exec_id, _log_exc,
                )

        # Orphan row cleanup
        if not is_content_failure and exec_id is not None:
            try:
                await delete_raw_transcript_data_by_execution(exec_id)
            except Exception as cleanup_exc:
                logger.warning(
                    "run_transcript_pipeline: failed to clean orphan "
                    "youtube_raw_transcript_data row for exec_id=%s: %s",
                    exec_id, cleanup_exc,
                )

        # Status reset — MUST always run. Independently guarded with ERROR-level
        # logging because a stuck 'currently_fetching' row requires manual intervention.
        if claimed and video_reg_id is not None:
            try:
                if is_content_failure:
                    await set_video_failed(video_reg_id)       # terminal — not retried
                else:
                    await set_video_not_fetched(video_reg_id)  # retryable
            except Exception as _reset_exc:
                logger.error(
                    "CRITICAL: run_transcript_pipeline: failed to reset status for "
                    "video_registry_id=%s: %s — row may be permanently stuck at "
                    "currently_fetching. Manual DB intervention required.",
                    video_reg_id, _reset_exc,
                )

        return {
            "success": False,
            "exec_id": exec_id,
            "message": str(e),
        }
