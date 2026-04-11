"""youtube-transcript-api wrapper — fetches SRT transcripts with language priority and SRTFormatter."""

import asyncio
import http.cookiejar
import os

from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)
import requests

from youtube_transcript_api.formatters import SRTFormatter


# ---------------------------------------------------------------------------
# Task 12.1 — get_language_priority()
# ---------------------------------------------------------------------------

def get_language_priority(channel_language: str) -> list[str]:
    """
    Returns transcript fetch language priority based on channel's language field.
    Hindi channels: try Hindi first, then English fallback.
    English and Hindi/English channels: try English first, then Hindi fallback.
    """
    if channel_language == "Hindi":
        return ["hi", "en"]
    else:
        return ["en", "hi"]


# ---------------------------------------------------------------------------
# Task 12.1 — fetch_transcript_srt()
# ---------------------------------------------------------------------------

async def fetch_transcript_srt(
    yt_video_id: str,
    language_priority: list[str],
) -> tuple[str, str, bool]:
    """
    Fetches the transcript for the given video in the preferred language order.
    Tries manual transcripts first (in language_priority order),
    then falls back to auto-generated transcripts (same language order).

    Never uses YouTube's built-in translation API — only the original language is fetched.

    Returns: (srt_content, language_code, is_auto_generated)
        srt_content:       full SRT-formatted string
        language_code:     actual language fetched (e.g. 'en', 'hi')
        is_auto_generated: True if auto-generated captions were used

    Raises:
        NoTranscriptFound    — content failure, terminal, do not retry
        TranscriptsDisabled  — content failure, terminal, do not retry
        VideoUnavailable     — content failure, terminal, do not retry
        Exception            — infrastructure failure, retryable

    The orchestrator uses the exception type to decide between
    set_video_failed() (terminal) and set_video_not_fetched() (retryable).

    Note: YouTubeTranscriptApi calls are synchronous/blocking. They are run
    via asyncio.to_thread() so they don't block the FastAPI event loop.
    """
    formatter = SRTFormatter()

    def _fetch_sync() -> tuple[str, str, bool]:
        COOKIES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "cookies.txt")
        cookie_jar = http.cookiejar.MozillaCookieJar(COOKIES_PATH)
        cookie_jar.load(ignore_discard=True, ignore_expires=True)
        session = requests.Session()
        session.cookies = cookie_jar  # type: ignore[assignment]
        api = YouTubeTranscriptApi(http_client=session)
        transcript_list = api.list(yt_video_id)

        # Try manual transcripts first
        try:
            transcript = transcript_list.find_manually_created_transcript(language_priority)
            fetched_data = transcript.fetch()
            srt_content = formatter.format_transcript(fetched_data)
            return srt_content, transcript.language_code, False
        except NoTranscriptFound:
            pass

        # Fall back to auto-generated transcripts
        transcript = transcript_list.find_generated_transcript(language_priority)
        fetched_data = transcript.fetch()
        srt_content = formatter.format_transcript(fetched_data)
        return srt_content, transcript.language_code, True

    return await asyncio.to_thread(_fetch_sync)
