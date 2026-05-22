"""youtube-transcript-api wrapper — fetches SRT transcripts with language priority and SRTFormatter.

Proxy Strategy:
  - Uses a residential proxy (configured via .env PROXY_* vars) to avoid datacenter IP bans.
  - Cookies are NOT used. Residential IP + browser headers replaces the old cookie approach.
  - Caller passes a proxy_session_id (str) for sticky session support. The session_id is
    embedded in the proxy username using the provider's standard format:
      {PROXY_USERNAME}-session-{session_id}
    This keeps the same residential IP for the duration of one batch (5-10 videos).
    After the batch, the caller generates a new session_id, rotating the IP.
  - If PROXY_ENABLED is false or any PROXY_* var is missing, falls back to a plain session
    (no proxy) so the pipeline degrades gracefully in local dev.
  - Transient network failures (ProxyError, ConnectionError, ChunkedEncodingError,
    Timeout) are retried up to _FETCH_MAX_RETRIES times with jittered delays.
    Each retry builds a fresh requests.Session and opens a new TCP connection —
    the failed connection is never reused. OSError excluded from retry scope
    intentionally — too broad, would mask genuine bugs.
"""

import asyncio
import logging

from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)
import random
import requests
from requests.exceptions import (
    ChunkedEncodingError,
    ConnectionError as RequestsConnectionError,
    ProxyError as RequestsProxyError,
    Timeout as RequestsTimeout,
)

from youtube_transcript_api.formatters import SRTFormatter
from app.core.config import (
    PROXY_ENABLED,
    PROXY_SCHEME,
    PROXY_HOST,
    PROXY_PORT,
    PROXY_USERNAME,
    PROXY_PASSWORD,
)

logger = logging.getLogger(__name__)

# Browser-like headers — masks python-requests User-Agent which Google actively flags.
# Chosen to match a common, stable Chrome version on Windows.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


# ---------------------------------------------------------------------------
# Retry config for transient proxy/network failures.
#
# _TRANSIENT_ERRORS: narrow set of known network failure types only.
#   OSError intentionally excluded — too broad, would mask genuine bugs.
#   Content failures (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable)
#   are never retried — they propagate immediately as terminal failures.
#
# Delays use jitter (base + random 0–3s) to prevent synchronised retries
# when multiple videos in a batch hit the same exit node simultaneously.
# ---------------------------------------------------------------------------
_FETCH_MAX_RETRIES = 2                # retry attempts after first — 3 total
_FETCH_RETRY_BASE_DELAYS = [10, 20]   # base seconds before attempt 2 and 3

_TRANSIENT_ERRORS = (
    RequestsProxyError,        # 407, tunnel failures
    RequestsConnectionError,   # connection reset, remote host closed
    ChunkedEncodingError,      # IncompleteRead — response truncated mid-stream
    RequestsTimeout,           # requests-level timeout (not Python builtin TimeoutError)
)


# ---------------------------------------------------------------------------
# Task 12.1 — get_language_priority()
# ---------------------------------------------------------------------------

def get_language_priority(channel_language: str) -> list[str]:
    """
    Retained for backward compatibility. The fetch order is now fixed inside
    fetch_transcript_srt() (manual EN -> auto EN -> manual HI -> auto HI) and
    does NOT depend on this return value. Always returns ['en', 'hi'].
    """
    return ["en", "hi"]


# ---------------------------------------------------------------------------
# Task 12.1 — fetch_transcript_srt()
# ---------------------------------------------------------------------------

async def fetch_transcript_srt(
    yt_video_id: str,
    language_priority: list[str],
    proxy_session_id: str | None = None,
) -> tuple[str, str, bool]:
    """
    Fetches the transcript for the given video in the preferred language order.
    Tries manual transcripts first (in language_priority order),
    then falls back to auto-generated transcripts (same language order).

    Never uses YouTube's built-in translation API — only the original language is fetched.

    Args:
        yt_video_id:       YouTube video ID (e.g. 'dQw4w9WgXcQ').
        language_priority: Ordered list of language codes to try (e.g. ['en', 'hi']).
        proxy_session_id:  Optional sticky session ID for the residential proxy.
                           Embedded into the proxy username as:
                             {PROXY_USERNAME}-session-{proxy_session_id}
                           If None, proxy is still used but without a sticky session
                           (IP may rotate per connection, depending on provider config).

    Returns: (srt_content, language_code, is_auto_generated)
        srt_content:       full SRT-formatted string
        language_code:     actual language fetched (e.g. 'en', 'hi')
        is_auto_generated: True if auto-generated captions were used

    Raises:

    Retry behaviour:
        Transient network failures (_TRANSIENT_ERRORS: ProxyError, ConnectionError,
        ChunkedEncodingError, Timeout) are retried up to _FETCH_MAX_RETRIES times.
        Delays: base delay + random jitter (0–3s) to prevent synchronised retries.
        Each retry calls asyncio.to_thread(_fetch_sync) which builds a completely
        fresh requests.Session and opens a new TCP connection to the proxy.
        The failed connection is never reused.
        Content failures (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable)
        are never retried — they propagate immediately as terminal failures.
        If all attempts exhaust, the final exception propagates to the orchestrator
        which correctly calls set_video_not_fetched() for future retry.
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

    def _build_session() -> requests.Session:
        """
        Builds a requests.Session with browser-like headers and optional proxy injection.
        Session is local to this thread — safe for asyncio.to_thread() usage.
        """
        session = requests.Session()
        session.headers.update(_BROWSER_HEADERS)

        if PROXY_ENABLED and PROXY_HOST and PROXY_USERNAME and PROXY_PASSWORD:
            # Build sticky username if a session ID was provided by the caller.
            # Decodo/Smartproxy sticky session format: {username}-session-{id}
            # Fix 11: proxy_session_id was being ignored — always used plain PROXY_USERNAME.
            # Now the sticky suffix is correctly appended when a session_id is given.
            if proxy_session_id is not None:
                username = f"{PROXY_USERNAME}-session-{proxy_session_id}"
            else:
                username = PROXY_USERNAME
            proxy_url = f"{PROXY_SCHEME}://{username}:{PROXY_PASSWORD}@{PROXY_HOST}:{PROXY_PORT}"
            session.proxies = {
                "http":  proxy_url,
                "https": proxy_url,
            }
            logger.debug(
                "fetch_transcript_srt: using residential proxy (host=%s, session_id=%r)",
                PROXY_HOST, proxy_session_id,
            )
        else:
            logger.debug(
                "fetch_transcript_srt: PROXY_ENABLED=False or missing config — "
                "proceeding without proxy (dev/local mode)"
            )

        return session

    def _fetch_sync() -> tuple[str, str, bool]:
        session = _build_session()
        api = YouTubeTranscriptApi(http_client=session)
        transcript_list = api.list(yt_video_id)

        # English-first policy: English (manual OR auto) ALWAYS beats Hindi.
        # Fetch order: manual EN -> auto EN -> manual HI -> auto HI.
        # language_priority is intentionally ignored for ordering here — the
        # English-first rule is fixed regardless of channel language.
        attempts = [
            ("en", "manual"),
            ("en", "auto"),
            ("hi", "manual"),
            ("hi", "auto"),
        ]
        for lang, kind in attempts:
            try:
                if kind == "manual":
                    transcript = transcript_list.find_manually_created_transcript([lang])
                    is_auto = False
                else:
                    transcript = transcript_list.find_generated_transcript([lang])
                    is_auto = True
            except NoTranscriptFound:
                continue
            fetched_data = transcript.fetch()
            srt_content = formatter.format_transcript(fetched_data)
            return srt_content, transcript.language_code, is_auto

        # No English or Hindi transcript in any form.
        raise NoTranscriptFound(yt_video_id, ["en", "hi"], transcript_list)

    last_exc: Exception | None = None

    for attempt in range(_FETCH_MAX_RETRIES + 1):  # attempts: 0, 1, 2
        try:
            return await asyncio.to_thread(_fetch_sync)

        except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
            raise  # content failure — terminal, never retry

        except _TRANSIENT_ERRORS as exc:
            last_exc = exc
            if attempt < _FETCH_MAX_RETRIES:
                base_delay = _FETCH_RETRY_BASE_DELAYS[attempt]
                jitter = random.uniform(0, 3)
                delay = base_delay + jitter
                logger.warning(
                    "fetch_transcript_srt: transient error on attempt %d/%d "
                    "for yt_video_id=%r (proxy_host=%s) — retrying in %.1fs. Error: %s",
                    attempt + 1, _FETCH_MAX_RETRIES + 1,
                    yt_video_id, PROXY_HOST or "none", delay, exc,
                )
                await asyncio.sleep(delay)
            else:
                logger.warning(
                    "fetch_transcript_srt: all %d attempts exhausted "
                    "for yt_video_id=%r (proxy_host=%s). Final error: %s",
                    _FETCH_MAX_RETRIES + 1, yt_video_id,
                    PROXY_HOST or "none", exc,
                )

    raise last_exc or RuntimeError(
        f"fetch_transcript_srt: retry loop exited unexpectedly "
        f"for yt_video_id={yt_video_id!r}"
    )
