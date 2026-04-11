"""
YouTube video relevance filter — two-stage pipeline.

Stage 1 (_stage1_prefilter):
    Pure presence check — no scoring, no penalties.
    Passes a video only if text contains BOTH a brand alias AND the model number
    (or a series/model identifier for number-free models).
    Eliminates videos with zero brand/number relationship to the target.
    Fast, free, no API calls.

Stage 2 (llm_filter_videos):
    Gemini LLM semantic verification of Stage 1 survivors.
    Handles: description pollution, comparison videos, variant suffix disambiguation,
    successor-framing ("after the G96, here is the G100...").
    All survivors classified in parallel via asyncio.gather.

Usage (called from youtube_search_orchestrator.py):
    from app.services.youtube_llm_filter import stage1_prefilter, llm_filter_videos

    stage1 = stage1_prefilter(videos, descriptions, brand, model_name)
    final  = await llm_filter_videos(stage1, descriptions, brand, model_name)
"""

import asyncio
import json
import logging
import re

from google import genai
from google.genai import types

from app.core.config import GEMINI_API_KEY
from app.services.video_relevance_filter import (
    extract_search_tokens,
    _normalize_text,
    _matches,
    _match_series_token,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton client (new google-genai SDK)
# ---------------------------------------------------------------------------

_client: genai.Client = genai.Client(api_key=GEMINI_API_KEY)

_GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Brand aliases — mirrors video_relevance_filter.py BRAND_ALIASES exactly.
# Duplicated here so Stage 1 can resolve aliases independently without
# importing the full scoring system into the call path.
#
# "iphone" is a SERIES token for Apple (not a brand alias) — same as in
#   video_relevance_filter.py. It is matched via primary_series_tokens in the
#   model check, not via the brand alias check.
# "pixel" is a SERIES token for Google — same reason.
# ---------------------------------------------------------------------------
_BRAND_ALIASES: dict[str, list[str]] = {
    "apple":    ["apple", "iphone"],
    "cmf":      ["cmf"],
    "google":   ["google", "pixel"],
    "iqoo":     ["iqoo"],
    "motorola": ["motorola", "moto"],
    "nothing":  ["nothing"],
    "oneplus":  ["oneplus", "one plus"],
    "oppo":     ["oppo"],
    "poco":     ["poco"],
    "realme":   ["realme"],
    "redmi":    ["redmi"],
    "samsung":  ["samsung"],
    "vivo":     ["vivo"],
    "xiaomi":   ["xiaomi", "mi"],
}

# ---------------------------------------------------------------------------
# Number-free model identifiers — used as Stage 1 fallback when the model
# has no numeric tokens.
#
# These are models that are identified by a name/word rather than a number.
# The list covers all known name-only phones across all supported brands.
# Checked against the combined title+description using _matches() (word boundary).
#
# When a new number-free model is released, add its key identifier here.
#
# Examples:
#   Apple iPhone Air            → "air"
#   Apple iPhone X, XR, XS     → these have numeric-equivalent tokens handled
#                                  by extract_search_tokens normally
#   Motorola Signature 5G       → "signature"
#   Motorola Razr (base)        → "razr" is a series token, covered by series check
#   Motorola Edge (base)        → "edge" is a series token, covered by series check
#   Motorola One Fusion+        → "one", "fusion" — "fusion" is in STRONG_SECONDARY,
#                                  treated as noncritical here; "one" caught by extract
#   OnePlus Open                → "open" is a series token in SERIES_TOKENS
#   OnePlus Nord (base)         → "nord" is a series token in SERIES_TOKENS
#   Google Pixel Fold           → "fold" not in series; added here
#   Google Pixel XL             → "xl" is a weak secondary, works via series check on "pixel"
#   Realme GT Master            → "master" added here; "gt" is the series token
#   Realme X, XT               → "xt" is a weak secondary; "x" is series token for Realme
#   Xiaomi Mi Max 2             → has "2", handled numerically
#   Xiaomi Mi A1/A2/A3          → "a" + number; handled numerically with "a" weak secondary
#   CMF Phone 1, CMF Phone 2 Pro → has number, handled numerically
#   Nothing Phone (1)(2)(3)     → has number, handled numerically
#   OPPO Find X (base)          → "find" + "x"; "x" may be in series tokens, covered
#   OPPO Find N2 Flip           → has number, handled numerically
#   Vivo NEX                    → "nex" added here
#   Vivo Z1 Pro, Z1X            → has number or "x", handled
# ---------------------------------------------------------------------------
_NUMBER_FREE_IDENTIFIERS: set[str] = {
    "air",          # Apple iPhone Air
    "signature",    # Motorola Signature 5G
    "fold",         # Google Pixel Fold, Google Pixel 9 Pro Fold (also has number)
    "master",       # Realme GT Master
    "nex",          # Vivo NEX
    "one",          # Motorola One Fusion+ (also "fusion")
}


def stage1_prefilter(
    videos: list[dict],
    descriptions: dict[str, str],
    brand: str,
    model_name: str,
) -> list[dict]:
    """
    Stage 1: Pure presence check. No scoring. No penalties whatsoever.

    A video passes if the combined title+description contains BOTH:
      1. At least one brand alias for this brand
      2. At least one of:
           (a) a primary numeric token (e.g. "96" for G96, "55" for A55)
           (b) a primary series token for number-free models
               (e.g. "edge" for Motorola Edge base, "nord" for OnePlus Nord base,
               "razr" for Motorola Razr base, "pixel" for Google Pixel base)
           (c) a number-free model identifier from _NUMBER_FREE_IDENTIFIERS
               (e.g. "air" for iPhone Air, "signature" for Motorola Signature)
           (d) any noncritical token for number-free models
               (e.g. "master" in "Realme GT Master" if not caught above)

    Condition (b)/(c)/(d) only applies when the model has NO numeric tokens.
    If the model HAS a number (most phones), only condition (a) is checked for
    the model side — the number is always the definitive identifier.

    WHY NO PENALTIES:
    Strong intruder penalties fire on common English adjectives. A Motorola G96 review
    description saying "ultra smooth display, power packed processor, pro-grade camera"
    triggers 3 intruder tokens (ultra, power, pro), all in STRONG_SECONDARY_TOKENS,
    producing a score of +10 −5 = 5 even though the video is perfectly correct.
    The intruder system is designed for variant disambiguation — that is Stage 2's job.
    Stage 1's only job is: does this video have any relationship to this brand + model?

    Never raises. Returns empty list if no videos pass.
    """
    brand_lower = brand.lower().strip()
    aliases = _BRAND_ALIASES.get(brand_lower, [brand_lower])

    # Use extract_search_tokens to get numeric and series tokens — avoids duplicating
    # the normalization and split logic already in video_relevance_filter.py
    tokens = extract_search_tokens(brand, model_name)
    numeric_tokens = tokens["primary_numeric_tokens"]   # e.g. ["96"] for G96
    series_tokens  = tokens["primary_series_tokens"]    # e.g. ["g"] for G-series
    noncrit_tokens = tokens["noncritical_tokens"]       # e.g. ["master"] for GT Master

    has_number = bool(numeric_tokens)

    passed = []
    for video in videos:
        vid_id = video.get("yt_video_id", "")
        title  = video.get("video_title") or ""
        desc   = descriptions.get(vid_id, "")
        text   = _normalize_text(f"{title} {desc}")

        # --- Condition 1: brand alias present ---
        brand_found = any(_matches(text, alias) for alias in aliases)

        # --- Condition 2: model identifier present ---
        if has_number:
            # Most phones: just check the number is in the text
            model_found = any(_matches(text, n) for n in numeric_tokens)
        else:
            # Number-free models: check series token, _NUMBER_FREE_IDENTIFIERS, or noncritical
            # _match_series_token used for single-char series like "g", "a" to avoid
            # matching the letter fragment from "5g" → "5 g" normalization
            series_found = any(_match_series_token(text, s) for s in series_tokens)
            nonfree_found = any(_matches(text, t) for t in _NUMBER_FREE_IDENTIFIERS
                                if _matches(text, t))
            noncrit_found = any(_matches(text, t) for t in noncrit_tokens)
            model_found = series_found or nonfree_found or noncrit_found

        if brand_found and model_found:
            passed.append(video)
        else:
            logger.debug(
                "Stage1 REJECT yt_video_id=%r brand_found=%s model_found=%s title=%r",
                vid_id, brand_found, model_found, title,
            )

    return passed


# ---------------------------------------------------------------------------
# Stage 2 — LLM semantic verification
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Gemini model configuration.
# Uses gemini-2.5-flash-lite — fast, cheap, sufficient for binary classification.
# Same SDK already used by translation_service.py.
# temperature=0.0 → deterministic. This is classification, not generation.
# max_output_tokens=256 → JSON response is small; cap prevents runaway cost.
# ---------------------------------------------------------------------------
_GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"

# ---------------------------------------------------------------------------
# Rate-limit guard — free tier is 30 RPM for gemini-2.0-flash-lite.
# Semaphore caps concurrent in-flight Gemini calls so we never blast all
# ~45 Stage 1 survivors at once and instantly saturate the quota.
# 10 concurrent calls → well under 30 RPM even accounting for fast responses.
#
# Retry config — on HTTP 429 TooManyRequests the call is retried with
# exponential backoff. Without retries every 429 silently fails open
# (is_relevant=True) and Stage 2 effectively becomes a no-op.
# ---------------------------------------------------------------------------
_LLM_CONCURRENCY = 10          # max concurrent in-flight Gemini calls
_LLM_MAX_RETRIES = 3           # number of retry attempts after a 429
_LLM_RETRY_DELAYS = [5, 15, 30]  # seconds to wait before each retry attempt

# ---------------------------------------------------------------------------
# LLM system prompt — binary classification, not generation.
#
# Five rules in priority order:
#
# 1. PRIMARY SUBJECT RULE: video must be *primarily* about the target phone,
#    not just mention it.
#
# 2. DESCRIPTION POLLUTION RULE (most important):
#    "After the Galaxy A55, here is the A56..." → video is about A56, not A55.
#    "After the G96, here comes the G100..." → video is about G100, not G96.
#    Key phrases: "after", "unlike", "compared to", "successor to", "previous",
#    "last year's", "new from", "next after".
#
# 3. VARIANT PRECISION RULE:
#    Realme 8 4G ≠ Realme 8 5G ≠ Realme 8i ≠ Realme 8s 5G
#    Nothing Phone (3a) ≠ Nothing Phone (3a) Pro
#    Motorola Edge 50 Fusion ≠ Motorola Edge 60 Fusion ≠ Motorola Edge 50
#    Samsung Galaxy S25 ≠ Galaxy S25 FE ≠ Galaxy S25 Ultra
#    Motorola G96 ≠ Motorola G85 ≠ Motorola G86
#
# 4. COMPARISON RULE:
#    "G96 vs G85 — Which Should You Buy?" → relevant for BOTH G96 and G85.
#    If target is one of exactly two phones compared head-to-head: is_relevant=true.
#
# 5. UNCERTAINTY RULE:
#    Fail open on ambiguity. is_relevant=true, confidence=low.
#    Never silently drop a video for lack of information.
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a video relevance classifier for a smartphone database.

Given a YouTube video title and description excerpt, and a TARGET phone model,
determine whether the video is PRIMARILY about that exact phone.

RULES (apply in order):

1. PRIMARY SUBJECT RULE
   The video must be primarily about the target phone — a full review, hands-on,
   unboxing, camera test, long-term review, or detailed feature coverage OF that phone.
   Passing mentions, brief references, or predecessor comparisons do NOT qualify.

2. DESCRIPTION POLLUTION RULE
   YouTube descriptions frequently mention other phones as context or backdrop:
     "After the Galaxy A55, Samsung has released the A56 5G..."
     "Compared to the Motorola G96, the G100 offers..."
     "Unlike the Edge 50 Fusion, the new Edge 60 Fusion has..."
     "After the Nothing Phone 2a, here comes the Phone 3a..."
   In these cases the TARGET phone is mentioned as backdrop, NOT as the subject.
   Key signal phrases: "after", "unlike", "compared to", "successor to",
   "previous", "last year's", "following", "replacing", "new from".
   If the description introduces the target phone with these phrases, mark is_relevant=false.

3. VARIANT PRECISION RULE
   Phone variants are completely different products. Do NOT treat them as interchangeable:
     Realme 8 4G  ≠  Realme 8 5G  ≠  Realme 8i  ≠  Realme 8s 5G
     Nothing Phone (3a)  ≠  Nothing Phone (3a) Pro  ≠  Nothing Phone (3a) Lite
     Motorola Edge 50 Fusion  ≠  Motorola Edge 60 Fusion  ≠  Motorola Edge 50
     Samsung Galaxy S25  ≠  Galaxy S25 FE  ≠  Galaxy S25 Ultra  ≠  Galaxy S25 Edge
     Motorola G96 5G  ≠  Motorola G85 5G  ≠  Motorola G86 Power 5G
     OnePlus Nord CE 4  ≠  OnePlus Nord CE 4 Lite  ≠  OnePlus Nord CE5
     iQOO Z9  ≠  iQOO Z9s  ≠  iQOO Z9s Pro  ≠  iQOO Z9x
   If the video is about a different variant, mark is_relevant=false.
   If ambiguous about which variant, mark confidence=low.

4. COMPARISON RULE
   A direct head-to-head comparison video counts as relevant for BOTH phones:
     "Motorola G96 vs G85 — Which to Buy?" → relevant for G96 AND G85.
     "Galaxy A55 5G vs A56 5G Camera Comparison" → relevant for BOTH.
   If the target is one of exactly two phones compared, mark is_relevant=true.
   Do NOT mark relevant if the target is mentioned among 3+ phones in a group test.

5. UNCERTAINTY RULE
   If the title and description are too brief or ambiguous to determine the primary
   subject with confidence, mark is_relevant=true and confidence=low.
   Never reject on ambiguity alone.

Respond ONLY with a JSON object. No preamble. No markdown fences. No explanation outside JSON:
{
  "is_relevant": true or false,
  "primary_subject": "the phone this video is primarily about, as you read it from the text",
  "confidence": "high" or "medium" or "low",
  "reason": "one sentence, plain English"
}"""


# ---------------------------------------------------------------------------
# Gemini client — lazy-initialized on first call.
# Same pattern as translation_service.get_gemini_client().
# ---------------------------------------------------------------------------
# Semaphore — created once at module level, shared across all concurrent calls.
_llm_semaphore = asyncio.Semaphore(_LLM_CONCURRENCY)


def _build_prompt(
    brand: str,
    model_name: str,
    title: str,
    description: str,
) -> str:
    """
    Builds the per-video classification prompt.

    Description truncated to 500 chars — sufficient to catch description pollution,
    which always appears in the opening sentence. Tail content (hashtags, channel
    links, equipment lists) is boilerplate and irrelevant to classification.
    """
    desc_excerpt = (description or "").strip()[:500]
    if not desc_excerpt:
        desc_excerpt = "(no description available)"

    return (
        f"TARGET PHONE: {brand} {model_name}\n\n"
        f"VIDEO TITLE: {title}\n\n"
        f"VIDEO DESCRIPTION (first 500 chars):\n{desc_excerpt}\n\n"
        f"Is this video primarily about the TARGET PHONE?"
    )


async def _classify_one(
    brand: str,
    model_name: str,
    title: str,
    description: str,
) -> dict:
    """
    Classifies one video with Gemini. Returns classification dict.

    Acquires the shared semaphore before calling Gemini — this caps concurrent
    in-flight calls to _LLM_CONCURRENCY (10) so the free-tier 30 RPM limit
    is never instantly saturated when processing a large Stage 1 batch.

    On HTTP 429 (TooManyRequests): retries up to _LLM_MAX_RETRIES times with
    exponential backoff (_LLM_RETRY_DELAYS). Without retries, 429s silently
    fail open (is_relevant=True) and Stage 2 becomes a no-op.

    On all other errors (network, JSON parse, timeout): fails OPEN.
    Returns is_relevant=True with confidence=low so infrastructure failures
    never silently drop valid videos.

    Uses asyncio.to_thread() because the Gemini SDK is synchronous.
    """
    prompt = _build_prompt(brand, model_name, title, description)

    async with _llm_semaphore:
        for attempt in range(_LLM_MAX_RETRIES + 1):
            try:
                response = await _client.aio.models.generate_content(
                    model=_GEMINI_MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        temperature=0.0,
                        max_output_tokens=256,
                    ),
                )
                raw = (response.text or "").strip()

                # Strip markdown fences if model ignores response_mime_type
                raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

                result = json.loads(raw)

                return {
                    "is_relevant":     bool(result.get("is_relevant", True)),
                    "primary_subject": str(result.get("primary_subject", "unknown")),
                    "confidence":      str(result.get("confidence", "low")),
                    "reason":          str(result.get("reason", "")),
                }

            except json.JSONDecodeError as e:
                logger.warning(
                    "LLM returned non-JSON for brand=%r model=%r title=%r: %s — failing open",
                    brand, model_name, title, e,
                )
                break  # JSON error is not retryable — bad response, not quota

            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower()

                if is_rate_limit and attempt < _LLM_MAX_RETRIES:
                    delay = _LLM_RETRY_DELAYS[attempt]
                    logger.warning(
                        "LLM 429 rate limit for brand=%r model=%r title=%r — "
                        "retry %d/%d in %ds",
                        brand, model_name, title, attempt + 1, _LLM_MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                    continue  # retry

                logger.warning(
                    "LLM call failed for brand=%r model=%r title=%r: %s — failing open",
                    brand, model_name, title, e,
                )
                break  # non-retryable error

    return {
        "is_relevant":     True,
        "primary_subject": "unknown (llm error)",
        "confidence":      "low",
        "reason":          "LLM call failed — passed through for manual review",
    }


async def llm_filter_videos(
    videos: list[dict],
    descriptions: dict[str, str],
    brand: str,
    model_name: str,
) -> list[dict]:
    """
    Stage 2: LLM semantic verification of Stage 1 survivors.

    Classifies all videos in parallel via asyncio.gather — one Gemini call per
    video, all concurrent. For ~30 Stage 1 survivors this is significantly
    faster than sequential calls.

    Args:
        videos:       Stage 1 survivors. Each dict must have yt_video_id, video_title.
        descriptions: dict mapping yt_video_id → description string.
        brand:        phone brand (e.g. "Motorola")
        model_name:   phone model (e.g. "G96 5G")

    Returns:
        Filtered list of video dicts confirmed relevant by the LLM.
        Order preserved. Empty list if nothing passes.

    Never raises — all errors are caught per-video and fail open.
    """
    if not videos:
        return []

    async def _verify_one(video: dict) -> tuple[dict, dict]:
        vid_id = video.get("yt_video_id", "")
        title  = video.get("video_title") or ""
        desc   = descriptions.get(vid_id, "")
        result = await _classify_one(brand, model_name, title, desc)
        return video, result

    results = await asyncio.gather(*[_verify_one(v) for v in videos])

    final = []
    for video, llm_result in results:
        vid_id = video.get("yt_video_id", "")
        title  = video.get("video_title") or ""

        if llm_result["is_relevant"]:
            final.append(video)
            logger.info(
                "LLM PASS  yt_video_id=%r confidence=%s subject=%r title=%r",
                vid_id, llm_result["confidence"],
                llm_result["primary_subject"], title,
            )
        else:
            logger.info(
                "LLM REJECT yt_video_id=%r confidence=%s subject=%r reason=%r title=%r",
                vid_id, llm_result["confidence"],
                llm_result["primary_subject"], llm_result["reason"], title,
            )

    return final
