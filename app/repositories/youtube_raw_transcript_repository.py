"""YouTube raw transcript data repository — insert and fetch rows from pipeline.youtube_raw_transcript_data."""

from app.core.supabase_client import get_client


# ---------------------------------------------------------------------------
# Task 15.1 — insert_raw_transcript_data()
# ---------------------------------------------------------------------------

async def insert_raw_transcript_data(
    video_registry_id: int,
    transcript_exec_id: int,
    srt_path: str,
    processed_transcript_path: str,
    translated_transcript_path: str | None,
    file_size_bytes: int,
    language_code: str | None,
    is_auto_generated: bool,
    translation_status: str,
    translation_language_code: str | None,
) -> int:
    """
    Inserts into pipeline.youtube_raw_transcript_data.

    translation_status must be one of: 'not_required', 'translation_complete'
    These are the only valid values at insert time. In-progress states
    ('pending_translation', 'currently_translating', 'translation_failed') are
    never inserted here — a row only exists when the full atomic pipeline completed.

    processed_transcript_path is NOT NULL — the atomic pipeline guarantees
    the TXT was created before this insert is called.

    transcript_exec_id has a UNIQUE constraint — inserting twice for the same
    execution will raise a DB error. This is intentional.

    Returns raw_transcript_id.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("youtube_raw_transcript_data")
        .insert({
            "video_registry_id": video_registry_id,
            "transcript_exec_id": transcript_exec_id,
            "srt_path": srt_path,
            "processed_transcript_path": processed_transcript_path,
            "translated_transcript_path": translated_transcript_path,
            "file_size_bytes": file_size_bytes,
            "language_code": language_code,
            "is_auto_generated": is_auto_generated,
            "translation_status": translation_status,
            "translation_language_code": translation_language_code,
        })
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            f"insert_raw_transcript_data() failed — no row returned for "
            f"video_registry_id={video_registry_id}, transcript_exec_id={transcript_exec_id}."
        )

    return result.data[0]["raw_transcript_id"]


# ---------------------------------------------------------------------------
# Task 15.2 — get_raw_transcript_data()
# ---------------------------------------------------------------------------

async def get_raw_transcript_data(video_registry_id: int) -> dict | None:
    """
    Fetches the raw transcript data row for the given video.
    Returns None if no row exists (pipeline not yet completed for this video).
    Returns the row dict if found.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("youtube_raw_transcript_data")
        .select("*")
        .eq("video_registry_id", video_registry_id)
        .order("fetched_at", desc=True)
        .limit(1)
        .execute()
    )

    if not result.data:
        return None

    return result.data[0]
