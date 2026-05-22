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
# S4-P1-1: converted to lazy singleton to prevent import-time crash when
# GEMINI_API_KEY is missing. Client is created on first call, not at import.
# ---------------------------------------------------------------------------

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client

_TRANSLATION_MODEL = "gemini-2.5-flash-lite"

_TRANSLATION_SYSTEM_PROMPT_TEMPLATE = (
    "You are a transcript translator specialising in tech reviews. "
    "You will receive a YouTube video transcript as clean plain text "
    "(one caption segment per line, no timestamps or sequence numbers). "
    "The transcript may be in Hindi, or a mix of Hindi and English. "
    "\n\n"
    "CRITICAL RULES — follow these exactly:\n"
    "1. The phone being reviewed in this video is: {phone_name}. "
    "Preserve this exact name every time it appears. "
    "Do NOT alter, shorten, or substitute it (e.g. never replace it with a different model name).\n"
    "2. Preserve ALL product names, brand names, model numbers, and tech specs verbatim — "
    "never paraphrase, shorten, or substitute them.\n"
    "3. Translate the spoken content line by line in the same order it appears. "
    "Do not reorder, merge, or skip any lines.\n"
    "4. Return ONLY the translated spoken text as plain paragraphs. "
    "Do NOT add sequence numbers, timestamps, or any other formatting.\n"
    "5. Do NOT add any preamble, explanation, or commentary."
)


# ---------------------------------------------------------------------------
# Task 13.1 — translate_to_english()
# ---------------------------------------------------------------------------

async def translate_to_english(transcript_text: str, phone_name: str = "Unknown phone") -> str:
    """
    Translates the speech content of a Hindi (or mixed) transcript to English.
    Uses gemini-2.5-flash-lite — fast and cost-effective, available on Gemini free tier.

    Args:
        transcript_text: Clean plain-text transcript (timestamps and sequence numbers
                          already stripped by process_srt_to_txt). One caption
                          segment per line.
        phone_name:       The exact phone model being reviewed (e.g. "Motorola Edge 50 Fusion").
                          Injected into the system prompt so the LLM never alters product names.

    Output: Plain English text only — no timestamps, no sequence numbers.
    """
    async def _call() -> str:
        system_prompt = _TRANSLATION_SYSTEM_PROMPT_TEMPLATE.format(phone_name=phone_name)
        try:
            response = await asyncio.wait_for(
                _get_client().aio.models.generate_content(
                    model=_TRANSLATION_MODEL,
                    contents=f"Translate this transcript to English:\n\n{transcript_text}",
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
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
