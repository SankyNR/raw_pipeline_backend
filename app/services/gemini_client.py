"""
Task 0.2 — Gemini Client (new google-genai SDK)

Unified wrapper for all Gemini API calls in the extraction pipeline.
Uses the new google-genai SDK (google-generativeai reached EOL Nov 2025).

All retry logic is centralised HERE so it applies uniformly to:
  - Enrichment                     → call_gemini_grounded()

Retry schedule (exponential backoff): 2s → 4s → 8s, max 3 attempts.
  - Rate limit (429) and transient server errors are retried.
  - Non-retryable errors (bad prompt, 4xx) raise immediately.

Module-level singleton (M6 fix):
  _client = genai.Client(api_key=...) — initialised once at import time.

Grounding + JSON mode (C2 fix):
  The Gemini 2.x API does NOT support response_mime_type="application/json"
  combined with google_search grounding in the same call.
  call_gemini_grounded() therefore omits response_mime_type and parses
  the plain-text JSON response manually after stripping any markdown fences.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from pydantic import BaseModel

from google import genai
from google.genai import errors, types

from app.core.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model name
# ---------------------------------------------------------------------------

EXTRACTION_MODEL = "gemini-2.5-flash-lite"

# ---------------------------------------------------------------------------
# Module-level singleton client (M6 fix — instantiated ONCE, not per call)
# ---------------------------------------------------------------------------

_client: genai.Client | None = None

def get_gemini_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class GeminiRateLimitError(Exception):
    """Raised when Gemini returns HTTP 429 (Resource Exhausted / rate limit)."""


class GeminiTransientError(Exception):
    """Raised on transient failures: 5xx, network timeouts, connection errors."""


class GeminiNonRetryableError(Exception):
    """
    Raised on permanent failures: bad prompt, invalid schema, 4xx errors.
    Retrying will not help — fail immediately.
    """


# ---------------------------------------------------------------------------
# Internal — exception classifier (M5 fix: uses typed SDK exceptions)
# ---------------------------------------------------------------------------

def _classify_gemini_exception(exc: Exception) -> type[Exception]:
    """
    Maps a google-genai SDK exception to one of our three error classes.

    Uses typed exceptions from google.genai.errors:
      - errors.ClientError: 4xx errors (bad request, quota, not-found)
      - errors.ServerError: 5xx errors (service unavailable, internal)
    Falls back to message-string heuristics for non-SDK network errors.
    """
    if isinstance(exc, errors.ClientError):
        # 429 = Resource Exhausted / quota
        if "429" in str(exc) or "resource_exhausted" in str(exc).lower():
            return GeminiRateLimitError
        # Other 4xx: bad request, invalid schema, etc.
        return GeminiNonRetryableError

    if isinstance(exc, errors.ServerError):
        return GeminiTransientError

    # Network/timeout errors (not from SDK — e.g. httpx, asyncio)
    msg = str(exc).lower()
    if any(k in msg for k in ("timeout", "connection", "network", "unavailable")):
        return GeminiTransientError

    return GeminiNonRetryableError


# ---------------------------------------------------------------------------
# Internal — retry engine
# ---------------------------------------------------------------------------

async def _retry_with_backoff(coro_factory, max_retries: int = 3, label: str = "gemini"):
    """
    Runs `coro_factory()` (a zero-argument callable returning a coroutine)
    with exponential backoff retries for rate-limit and transient errors.

    Backoff schedule: 2s → 4s → 8s (2 ** (attempt + 1)).
    Non-retryable errors propagate immediately without sleeping.

    Args:
        label: Caller name for structured error logs (e.g. "call_gemini_json").
    """
    for attempt in range(max_retries):
        try:
            return await coro_factory()
        except (GeminiRateLimitError, GeminiTransientError) as exc:
            if attempt == max_retries - 1:
                logger.error(
                    "Gemini call failed after %d attempts [%s]: %s",
                    max_retries, label, exc,
                )
                raise
            wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
            logger.warning(
                "Gemini transient/rate-limit error (attempt %d/%d) [%s], "
                "retrying in %ds: %s",
                attempt + 1, max_retries, label, wait, exc,
            )
            await asyncio.sleep(wait)
        except GeminiNonRetryableError:
            raise  # No point retrying — bad prompt, invalid schema, etc.


# ---------------------------------------------------------------------------
# Safe float coercion helper (for grounded enrichment confidence values)
# ---------------------------------------------------------------------------

def _safe_float(val, default: float = 0.5) -> float:
    """
    Coerces an enrichment confidence value to float.
    Handles model responses like "0.90 (Confirmed)" by stripping non-numeric
    trailing text before conversion. Falls back to `default` on any error.
    """
    try:
        return float(str(val).split()[0].rstrip('.,)'))
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# JSON fence stripper helper
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json_object(text: str) -> str:
    """
    Robustly extracts a JSON object from model output.

    Strategy (applied in order):
      1. Strip markdown fences (```json ... ```) — most common case.
      2. If result doesn't start with { or [, find first { or [ and last } or ]
         to strip any preamble or trailing text the model added despite the prompt.
      3. Return stripped text as-is if neither heuristic applies.

    This is more defensive than a pure fence-strip for call_gemini_grounded()
    """
    # Step 1: strip markdown fences
    m = _JSON_FENCE_RE.search(text)
    cleaned = m.group(1) if m else text.strip()

    # Step 2: if still not starting with a JSON delimiter, extract first { → last }
    stripped = cleaned.strip()
    if stripped and stripped[0] not in ("{", "["):
        # Find outermost object
        obj_start = stripped.find("{")
        arr_start = stripped.find("[")
        if obj_start == -1 and arr_start == -1:
            return stripped  # nothing to extract

        # Pick whichever delimiter comes first
        if obj_start == -1:
            start, close = arr_start, "]"
        elif arr_start == -1:
            start, close = obj_start, "}"
        else:
            start = min(obj_start, arr_start)
            close = "}" if start == obj_start else "]"

        # Brace-balanced scan: find the end of the FIRST complete JSON object.
        # rfind() was wrong — it found the last closing delimiter, which when
        # Gemini returns NDJSON (same object repeated on multiple lines) spanned
        # all repetitions and produced invalid JSON for json.loads().
        # This scanner counts open/close delimiters and stops at the first
        # complete balanced object, correctly handling nested braces.
        open_delim  = "{" if close == "}" else "["
        close_delim = close
        depth = 0
        end = -1
        for i in range(start, len(stripped)):
            if stripped[i] == open_delim:
                depth += 1
            elif stripped[i] == close_delim:
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end > start:
            return stripped[start:end + 1]

    return stripped



# ---------------------------------------------------------------------------
# Task 0.2 — call_gemini_grounded
# ---------------------------------------------------------------------------

async def _resolve_grounding_redirect(redirect_url: str, timeout: float = 3.0) -> str | None:
    """
    Follows a Vertex AI grounding redirect URL to extract the real source URL.
    Issues a HEAD request with redirect-following DISABLED.
    Returns the Location header value (real URL), or None on any error / timeout.
    A 3-second timeout prevents this from slowing enrichment on network issues.
    """
    import httpx
    try:
        async with httpx.AsyncClient(follow_redirects=False, timeout=timeout) as client:
            resp = await client.head(redirect_url)
            return resp.headers.get("location") or None
    except Exception:
        return None


async def call_gemini_grounded(
    prompt: str,
    output_schema: dict[str, Any],
    site_hint: str | None = None,
    max_retries: int = 3,
) -> dict[str, Any]:
    """
    Calls Gemini with Google Search grounding enabled.
    Used exclusively for enrichment queries (gap-fill for missing fields).

    One API call = search + fetch + extract. No separate HTTP layer needed.

    IMPORTANT — C2 fix:
    Gemini 2.x does NOT support response_mime_type="application/json" combined
    with google_search grounding in the same call (raises 400 API error).
    This function therefore uses plain text mode and parses the JSON manually.
    The prompt instructs the model to return clean JSON only.

    Args:
        prompt:       The enrichment question, e.g.
                      "What is the SAR head value for the Samsung Galaxy S25 Ultra
                      sold in India? Return JSON: {value, confidence, evidence}"
        output_schema: JSON schema for the expected structured response.
                       Used only for prompt context — NOT sent to API as schema
                       (incompatible with grounding in 2.x).
        site_hint:    Optional domain hint. If set, prepended to prompt as:
                      "If available, prefer results from {site_hint}."
        max_retries:  Maximum retry attempts.

    Returns:
        Dict with shape:
        {
            "value":         <extracted value or None>,
            "confidence":    <float 0.0–1.0>,
            "evidence":      <supporting text>,
            "source_url":    <grounding citation URL or None>,
            "source_domain": <domain extracted from source_url or None>,
        }

    Raises:
        GeminiRateLimitError, GeminiTransientError, GeminiNonRetryableError.
    """
    schema_hint = json.dumps(output_schema, indent=2)
    base_prompt = (
        f"{prompt}\n\n"
        f"Return ONLY valid JSON (no markdown, no explanation) matching this schema:\n"
        f"{schema_hint}"
    )
    if site_hint:
        full_prompt = f"If available, prefer results from {site_hint}.\n\n{base_prompt}"
    else:
        full_prompt = base_prompt

    async def _call():
        try:
            response = await get_gemini_client().aio.models.generate_content(
                model=EXTRACTION_MODEL,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    tools=[
                        types.Tool(google_search=types.GoogleSearch())
                    ],
                    # NOTE: response_mime_type intentionally OMITTED here —
                    # JSON mode + grounding is unsupported on Gemini 2.x (C2 fix).
                ),
            )

            raw_text = response.text
            if not raw_text or not raw_text.strip():
                raise GeminiNonRetryableError(
                    "call_gemini_grounded: empty or whitespace response."
                )

            # Parse JSON — strip fences and extract first { → last } for robustness
            cleaned = _extract_json_object(raw_text)
            try:
                # Strip any backslash that isn't part of a valid JSON escape sequence (\| → |, etc.)
                cleaned = re.sub(r'\\([^"\\/bfnrtu\n\r])', r'\1', cleaned)
                parsed = json.loads(cleaned)
            except json.JSONDecodeError as exc:
                raise GeminiNonRetryableError(
                    f"Gemini grounded JSON decode failed: {exc}. "
                    f"Raw response (first 500 chars): {raw_text[:500]}"
                ) from exc

            # Extract grounding metadata (citation URL) from the response
            source_url: str | None = None
            source_domain: str | None = None
            try:
                grounding_chunks = (
                    response.candidates[0]
                    .grounding_metadata
                    .grounding_chunks
                )
                if grounding_chunks:
                    raw_uri = grounding_chunks[0].web.uri
                    if raw_uri:
                        # Gemini returns a Vertex AI redirect URL, not the real source.
                        # Resolve it with a single HEAD request to get the actual domain.
                        if "vertexaisearch.cloud.google.com" in raw_uri:
                            resolved = await _resolve_grounding_redirect(raw_uri)
                            source_url = resolved or raw_uri  # fall back to redirect if resolve fails
                        else:
                            source_url = raw_uri
                        if source_url:
                            from urllib.parse import urlparse
                            source_domain = urlparse(source_url).netloc
            except (AttributeError, IndexError, TypeError):
                pass  # Grounding metadata may be absent — treat as parametric

            return {
                "value":         parsed.get("value"),
                "confidence":    _safe_float(parsed.get("confidence", 0.5)),
                "evidence":      parsed.get("evidence", ""),
                "source_url":    source_url,
                "source_domain": source_domain,
            }

        except (GeminiRateLimitError, GeminiTransientError, GeminiNonRetryableError):
            raise
        except Exception as exc:
            error_class = _classify_gemini_exception(exc)
            raise error_class(f"call_gemini_grounded error: {exc}") from exc

    return await _retry_with_backoff(_call, max_retries=max_retries, label="call_gemini_grounded")


# ---------------------------------------------------------------------------
# Phase 0 — call_gemini_json (v5 extraction engine)
# ---------------------------------------------------------------------------

async def call_gemini_json(
    system_prompt: str,
    user_content: str,
    output_schema: type | None,
    temperature: float = 0.1,
    max_retries: int = 3,
    return_raw_response: bool = False,
) -> dict[str, Any]:
    """
    Calls Gemini with JSON mode enforced (response_mime_type="application/json").
    Used for all extraction calls: Run A (spec) and Run B (experience).

    One API call per extraction. No chaining, no AFC, no tool calling.

    Args:
        system_prompt:        ECD for Run A; experience system prompt for Run B.
        user_content:         Assembled sources string (Run A) or transcript text (Run B).
        output_schema:        Must be a Pydantic BaseModel subclass (NOT a dict or raw type).
                              The Gemini SDK requires a structured schema object.
                              Passing a raw dict will fail or disable schema enforcement.
        temperature:          0.1 for Run A (deterministic); 0.3 for Run B (expressive).
        max_retries:          Maximum retry attempts (backoff: 2s → 4s → 8s).
        return_raw_response:  If True, returns (parsed_dict, raw_response) instead of just
                              parsed_dict. Used by Stage 1 to read usage_metadata for token
                              counting without an extra API call.

    Returns:
        Parsed dict matching output_schema structure.
        If return_raw_response=True: (parsed_dict, raw_response) tuple.

    Raises:
        GeminiNonRetryableError: If output_schema is not a Pydantic BaseModel subclass,
                                 or if the response is empty/unparseable.
        GeminiRateLimitError, GeminiTransientError: On transient API failures.
    """
    if output_schema is not None and not (isinstance(output_schema, type) and issubclass(output_schema, BaseModel)):
        raise GeminiNonRetryableError(
            "output_schema must be a Pydantic BaseModel subclass or None. "
            f"Got: {type(output_schema).__name__}"
        )

    async def _call():
        try:
            response = await get_gemini_client().aio.models.generate_content(
                model=EXTRACTION_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=user_content)],
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    response_mime_type="application/json",
                    **({
                        "response_schema": output_schema
                    } if output_schema is not None else {}),
                ),
            )

            raw_text = response.text
            if not raw_text or not raw_text.strip():
                raise GeminiNonRetryableError(
                    "call_gemini_json: empty or whitespace response."
                )

            try:
                # Strip any backslash that isn't part of a valid JSON escape sequence (\| → |, etc.)
                raw_text = re.sub(r'\\([^"\\\/bfnrtu\n\r])', r'\1', raw_text)
                parsed = json.loads(raw_text)
                logger.debug(
                    "call_gemini_json success | model=%s | response_chars=%d",
                    EXTRACTION_MODEL, len(raw_text),
                )
                return (parsed, response) if return_raw_response else parsed
            except json.JSONDecodeError as exc:
                cleaned = _extract_json_object(raw_text)
                try:
                    parsed = json.loads(cleaned)
                    logger.debug(
                        "call_gemini_json success (fallback parse) | model=%s | response_chars=%d",
                        EXTRACTION_MODEL, len(raw_text),
                    )
                    return (parsed, response) if return_raw_response else parsed
                except json.JSONDecodeError:
                    raise GeminiNonRetryableError(
                        f"call_gemini_json: JSON decode failed: {exc}. "
                        f"Raw (first 300): {raw_text[:300]}"
                    ) from exc

        except (GeminiRateLimitError, GeminiTransientError, GeminiNonRetryableError):
            raise
        except Exception as exc:
            error_class = _classify_gemini_exception(exc)
            raise error_class(f"call_gemini_json error: {exc}") from exc

    return await _retry_with_backoff(_call, max_retries=max_retries, label="call_gemini_json")
