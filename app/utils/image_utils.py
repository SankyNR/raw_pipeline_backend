"""
Image Assembly Utilities

Task 2.1  — assemble_screenshot(): converts a Firecrawl screenshot (either a
            signed HTTP URL or a base64 string) into image bytes, choosing the
            format automatically:
              • WebP (quality=85) when both dimensions are ≤ WEBP_MAX_PX
              • PNG                when either dimension exceeds WEBP_MAX_PX
Task 8.2  — extract_screenshots(): extracts and routes screenshots from a full
            Firecrawl response by template name.

Firecrawl may return screenshots in one of two formats:
  1. A signed HTTPS URL  → downloaded with httpx.AsyncClient
  2. A base64 string     → decoded directly (data URI prefix stripped if present)

Both functions are async because URL-based screenshots require an async HTTP fetch.
"""

import base64
import io
import logging

import httpx
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WebP hard limit — any image exceeding this on either axis is saved as PNG.
# ---------------------------------------------------------------------------

WEBP_MAX_PX = 16383


# ---------------------------------------------------------------------------
# Task 2.1 — assemble_screenshot
# ---------------------------------------------------------------------------

async def assemble_screenshot(screenshot_string: str) -> tuple[bytes, str]:
    """
    Converts a Firecrawl screenshot into image bytes ready for upload to
    Supabase Storage, and returns the MIME type alongside.

    Format selection:
      • WebP (quality=85)  — used when both width and height are ≤ 16383 px
      • PNG                — used when either dimension exceeds 16383 px
                             (WebP's hard encoder limit)

    Firecrawl may return the screenshot in two input formats:

    1. Signed HTTPS URL (e.g. "https://storage.googleapis.com/...")
       → Downloads the image with httpx and uses response.content as raw bytes.

    2. Base64 string, with or without a data URI prefix
       (e.g. "data:image/png;base64,iVBORw0KGgo..." or just "iVBORw0KGgo...")
       → Strips the prefix if present, then base64-decodes.

    Args:
        screenshot_string: Signed URL or base64 string returned by Firecrawl.

    Returns:
        (image_bytes, mime_type) where mime_type is "image/webp" or "image/png".

    Raises:
        ValueError:  If screenshot_string is empty or None.
        httpx.HTTPStatusError: If URL download returns a non-2xx status.
        binascii.Error: If the string is not valid base64.
        PIL.UnidentifiedImageError: If the bytes are not a recognisable image.
    """
    if not screenshot_string:
        raise ValueError("assemble_screenshot() received an empty or None screenshot string.")

    if screenshot_string.startswith("http"):
        # --- Format 1: signed URL ---
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(screenshot_string)
            resp.raise_for_status()
            image_bytes = resp.content
    else:
        # --- Format 2: base64 string (strip data URI prefix if present) ---
        if "," in screenshot_string:
            screenshot_string = screenshot_string.split(",", 1)[1]
        image_bytes = base64.b64decode(screenshot_string)

    image = Image.open(io.BytesIO(image_bytes))
    image.load()  # Force decode now to surface corrupt image errors early

    output_buffer = io.BytesIO()

    if image.width > WEBP_MAX_PX or image.height > WEBP_MAX_PX:
        # Image exceeds the WebP encoder hard limit — fall back to PNG.
        image.save(output_buffer, format="PNG")
        mime_type = "image/png"
    else:
        image.save(output_buffer, format="WEBP", quality=85)
        mime_type = "image/webp"

    output_buffer.seek(0)
    return output_buffer.read(), mime_type


# ---------------------------------------------------------------------------
# Task 8.2 — extract_screenshots
# ---------------------------------------------------------------------------

async def extract_screenshots(response: dict, template_name: str) -> dict:
    """
    Extracts and converts screenshots from the Firecrawl response.

    Returns:
        {
            "before": {"data": <bytes>, "mime_type": <str>} | None,
            "after":  {"data": <bytes>, "mime_type": <str>} | None,
        }

    Each screenshot is encoded as WebP when both dimensions are ≤ 16383 px,
    or as PNG when either dimension exceeds the WebP encoder hard limit.
    The "mime_type" value ("image/webp" or "image/png") reflects the actual
    format chosen, so callers can set the correct Content-Type and file extension.

    Routing table (driven by screenshot count in each config template):
    ┌──────────────────────────────────────────────────────────┬──────────────────┬──────────────────┐
    │ template_name                                            │ before           │ after            │
    ├──────────────────────────────────────────────────────────┼──────────────────┼──────────────────┤
    │ apple_config_template                (1 screenshot)      │ screenshots[0]   │ None             │
    │ devicespecification_config_template  (1 screenshot)      │ screenshots[0]   │ None             │
    │ iqoo_config_template                 (1 screenshot)      │ screenshots[0]   │ None             │
    │ oneplus_config_template              (1 screenshot)      │ screenshots[0]   │ None             │
    │ oppo_config_template                 (1 screenshot)      │ screenshots[0]   │ None             │
    │ poco_config_template                 (1 screenshot)      │ screenshots[0]   │ None             │
    │ realme_config_template               (1 screenshot)      │ screenshots[0]   │ None             │
    │ vivo_config_template                 (1 screenshot)      │ screenshots[0]   │ None             │
    │ xiaomi_and_redmi_config_template     (1 screenshot)      │ screenshots[0]   │ None             │
    ├──────────────────────────────────────────────────────────┼──────────────────┼──────────────────┤
    │ motorola_config_template             (3 screenshots)     │ screenshots[1]   │ screenshots[2]   │
    ├──────────────────────────────────────────────────────────┼──────────────────┼──────────────────┤
    │ all 2-screenshot templates (gsmarena, pixel, samsung,    │ screenshots[0]   │ screenshots[1]   │
    │   smartprix, nothing, samsung_direct_scrape, etc.)       │                  │                  │
    └──────────────────────────────────────────────────────────┴──────────────────┴──────────────────┘

    Calls assemble_screenshot() on each screenshot string to produce image bytes.
    Both this function and assemble_screenshot() are async because URL-based
    screenshots require an async HTTP download.
    """
    data = response["data"]
    screenshots = data["actions"]["screenshots"]

    def _wrap(result):
        """Convert (bytes, mime_type) tuple to {"data": ..., "mime_type": ...} dict."""
        img_bytes, mime_type = result
        return {"data": img_bytes, "mime_type": mime_type}

    # --- 1-screenshot templates: before only, no after ---
    ONE_SCREENSHOT_TEMPLATES = {
        "apple_config_template",
        "devicespecification_config_template",
        "iqoo_config_template",
        "oneplus_config_template",
        "oppo_config_template",
        "poco_config_template",
        "realme_config_template",
        "vivo_config_template",
        "xiaomi_and_redmi_config_template",
    }

    if template_name in ONE_SCREENSHOT_TEMPLATES:
        before = _wrap(await assemble_screenshot(screenshots[0]))
        after = None

    # --- motorola: 3 screenshots — skip first, use second (before) and third (after) ---
    elif template_name == "motorola_config_template":
        # motorola expects 3 screenshots — guard against Firecrawl returning fewer
        if len(screenshots) < 3:
            logger.warning(
                "extract_screenshots: motorola_config_template expected 3 screenshots, "
                "got %d. Falling back to first available.",
                len(screenshots),
            )
            before = _wrap(await assemble_screenshot(screenshots[0])) if screenshots else None
            after  = _wrap(await assemble_screenshot(screenshots[1])) if len(screenshots) > 1 else None
        else:
            before = _wrap(await assemble_screenshot(screenshots[1]))
            after  = _wrap(await assemble_screenshot(screenshots[2]))

    # --- All 2-screenshot templates (gsmarena, pixel_newer, pixel_older, nothing,
    #     samsung_direct_scrape, samsung_one-click, samsung_specs-expand_*,
    #     smartprix, google_pixel_newer/older, etc.) ---
    else:
        # Default 2-screenshot templates — guard against Firecrawl returning only 1
        before = _wrap(await assemble_screenshot(screenshots[0])) if screenshots else None
        after  = _wrap(await assemble_screenshot(screenshots[1])) if len(screenshots) > 1 else None

    return {"before": before, "after": after}

