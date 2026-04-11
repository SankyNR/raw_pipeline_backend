"""Gemini LLM translation service — translates Hindi/mixed SRT transcripts to English plain text."""

import asyncio
import logging

from google import genai
from google.genai import errors, types

from app.core.config import GEMINI_API_KEY
from app.services.gemini_client import (
    GeminiRateLimitError,
    GeminiTransientError,
    GeminiNonRetryableError,
    _classify_gemini_exception,
    _retry_with_backoff,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton client
# ---------------------------------------------------------------------------

_client: genai.Client = genai.Client(api_key=GEMINI_API_KEY)

_TRANSLATION_MODEL = "gemini-2.5-flash-lite"

_TRANSLATION_SYSTEM_PROMPT = (
    "You are a transcript translator. You will receive a YouTube video transcript "
    "in SRT format. The transcript may be in Hindi, or a mix of Hindi and English. "
    "Your task: translate all spoken content to clear, natural English. "
    "Return ONLY the translated spoken text as plain paragraphs. "
    "Do NOT include sequence numbers, timestamps, or any SRT formatting in your output. "
    "Do NOT add any preamble, explanation, or commentary."
)


# ---------------------------------------------------------------------------
# Task 13.1 — translate_to_english()
# ---------------------------------------------------------------------------

async def translate_to_english(srt_content: str) -> str:
    """
    Translates the speech content of a Hindi (or mixed) SRT transcript to English.
    Uses gemini-2.5-flash-lite — fast and cost-effective, available on Gemini free tier.

    Input:  Full SRT string (includes sequence numbers, timestamps, Hindi text)
    Output: Plain English text only — no SRT structure, no timestamps, no sequence numbers.

    The model reads the full SRT, translates only the spoken dialogue, and returns
    plain paragraphs. This output is passed directly into process_srt_to_txt()
    which passes plain text through unchanged.

    Retry logic (exponential backoff, max 3 attempts — 2s → 4s → 8s):
        429 / rate-limit  → GeminiRateLimitError  → retried
        5xx / timeout     → GeminiTransientError   → retried
        4xx non-rate      → GeminiNonRetryableError → raises immediately
        empty response    → RuntimeError            → raises immediately (non-retryable)

    If all retries are exhausted the final exception propagates as-is so the
    orchestrator's except-Exception block can mark the video 'not_fetched' for retry.

    Raises:
        RuntimeError          — if model returns empty output (non-retryable)
        GeminiRateLimitError  — after exhausting retries on 429
        GeminiTransientError  — after exhausting retries on 5xx / network errors
        GeminiNonRetryableError — immediately on bad prompt / 4xx non-rate errors
    """
    async def _call() -> str:
        try:
            response = await asyncio.wait_for(
                _client.aio.models.generate_content(
                    model=_TRANSLATION_MODEL,
                    contents=f"Translate this transcript to English:\n\n{srt_content}",
                    config=types.GenerateContentConfig(
                        system_instruction=_TRANSLATION_SYSTEM_PROMPT,
                        temperature=0.1,
                    ),
                ),
                timeout=120,
            )
        except asyncio.TimeoutError:
            # Treat timeout as transient — worth retrying
            raise GeminiTransientError("Gemini translation timed out after 120 seconds.")
        except (GeminiRateLimitError, GeminiTransientError, GeminiNonRetryableError):
            raise  # pass through our own exceptions unchanged
        except (errors.ClientError, errors.ServerError) as exc:
            error_class = _classify_gemini_exception(exc)
            raise error_class(f"translate_to_english error: {exc}") from exc
        except Exception as exc:
            error_class = _classify_gemini_exception(exc)
            raise error_class(f"translate_to_english error: {exc}") from exc

        if not response or not response.text:
            raise RuntimeError("Translation returned no text output.")

        translated_text = response.text.strip()
        if not translated_text:
            raise RuntimeError(
                "Translation returned empty output — check input transcript content."
            )

        return translated_text

    return await _retry_with_backoff(_call)
