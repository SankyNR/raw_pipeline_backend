"""
Phase 4 — Task 4.2: Supabase Storage File Path Generator

Builds all storage paths for a single scrape run from brand, model, and site_name.
The timestamp is generated once per call so all files in the same scrape share
the same timestamp — do not call build_storage_paths() more than once per scrape.
"""

import re
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Folder and short-code mapping
# ---------------------------------------------------------------------------

_SITE_FOLDER_MAP = {
    "gsmarena":             "GSMarena",
    "smartprix":            "Smartprix",
    "devicespecifications": "DeviceSpecifications",
}
_SITE_CODE_MAP = {
    "gsmarena":             "gsm",
    "smartprix":            "smartprix",
    "devicespecifications": "devspecs",
}


def _get_site_meta(site_name: str) -> tuple[str, str]:
    """
    Returns (folder, short_code) for a given site_name.

    Resolution order:
      1. Explicit map lookup (gsmarena, smartprix, devicespecifications)
      2. site_name ends with '_official' — OEM pages (samsung_official, etc.)
         These go into 'official/' folder with code 'official'.
      3. Unknown site — derive folder/code from site_name rather than silently
         using 'official', which would mix unknown sites with OEM content.
    """
    if site_name in _SITE_FOLDER_MAP:
        return _SITE_FOLDER_MAP[site_name], _SITE_CODE_MAP[site_name]
    if site_name.endswith("_official"):
        return "official", "official"
    # Unknown site — derive from name to avoid polluting the official/ folder
    derived_folder = site_name.replace("_", " ").title().replace(" ", "")
    derived_code   = site_name[:10].replace("_", "")
    return derived_folder, derived_code


# ---------------------------------------------------------------------------
# Run A source concatenation order (Phase 2)
# ---------------------------------------------------------------------------

# Concatenation priority for assemble_run_a_input().
# Lower = assembled earlier in the combined content string.
# OEM official always anchors first (clearest field layout for schema comprehension).
# Transcript always last (contextual, not structural).
_SITE_CONCAT_ORDER: dict[str, int] = {
    "gsmarena":             2,
    "smartprix":            3,
    "devicespecifications": 4,
}


def get_concat_order(site_name: str) -> int:
    """
    Returns the concatenation priority for a source file in Run A.
    Lower number = assembled earlier in the combined content string.

    Order:
      1  — OEM official site (structural anchor, clearest field layout)
      2  — GSMArena (comprehensive secondary coverage)
      3  — Smartprix (supplementary coverage)
      4  — DeviceSpecifications (additional aggregator)
      5  — Any other/unknown aggregator
      99 — Transcript (contextual, always last)

    The transcript caller must pass site_name='transcript' explicitly.
    """
    if site_name == "transcript":
        return 99
    if site_name.endswith("_official"):
        return 1
    return _SITE_CONCAT_ORDER.get(site_name, 5)


# ---------------------------------------------------------------------------
# Task 4.2 — slugify
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """
    Normalises a brand or model name into a URL- and filesystem-safe slug.

    Steps:
        1. Lowercase
        2. Replace any character that is NOT a letter or digit with a hyphen
        3. Collapse consecutive hyphens into one
        4. Strip leading and trailing hyphens

    Examples:
        "Galaxy S25 Ultra"  → "galaxy-s25-ultra"
        "iPhone 16 Pro+"    → "iphone-16-pro"
        "Pixel 9 Pro Fold"  → "pixel-9-pro-fold"
        "CMF Phone (2) Pro" → "cmf-phone-2-pro"
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


# ---------------------------------------------------------------------------
# Task 4.2 — build_storage_paths
# ---------------------------------------------------------------------------

def build_storage_paths(brand: str, model: str, site_name: str) -> dict:
    """
    Builds all storage paths for a single scrape run.

    The timestamp is generated once here (DD-MM-YY-HH-MM-SS-MICROSECONDS, UTC)
    so all three files in the same scrape share the same timestamp value.
    Microseconds are included to prevent filename collisions when the same phone
    is scraped twice within the same second.
    Do NOT call this function twice for one scrape.

    Args:
        brand:     Raw brand name from url_registry (e.g. "Samsung").
        model:     Raw model name from url_registry (e.g. "Galaxy S25 Ultra").
        site_name: site_name slug from lookup_source_registry (e.g. "gsmarena").

    Returns:
        {
            "markdown_path":           str,   # e.g. "samsung/galaxy-s25-ultra/GSMarena/samsung-galaxy-s25-ultra-gsm-21-02-26-22-10-00-123456.md"
            "screenshot_before_path":  str,
            "screenshot_after_path":   str,
            "timestamp":               str,   # "DD-MM-YY-HH-MM-SS-MICROSECONDS"
        }
    """
    brand_slug = slugify(brand)
    model_slug = slugify(model)

    folder, short_code = _get_site_meta(site_name)

    now = datetime.now(tz=timezone.utc)
    timestamp = now.strftime("%d-%m-%y-%H-%M-%S-%f")

    base = f"{brand_slug}/{model_slug}/{folder}"
    stem = f"{brand_slug}-{model_slug}-{short_code}-{timestamp}"

    return {
        "markdown_path":          f"{base}/{stem}.md",
        "screenshot_before_path": f"{base}/screenshot_before-{stem}.webp",
        "screenshot_after_path":  f"{base}/screenshot_after-{stem}.webp",
        "timestamp":              timestamp,
    }


# ---------------------------------------------------------------------------
# Phase 9 — Task 9.1: build_youtube_storage_paths
# ---------------------------------------------------------------------------

def build_youtube_storage_paths(
    brand: str,
    model: str,
    channel_name: str,
    yt_channel_id: str,
    yt_video_id: str,
) -> dict:
    """
    Returns dict with keys: srt_path, processed_path, translated_path

    Path format:
        {brand_slug}/{model_slug}/raw_transcripts/{brand_slug}-{model_slug}-{channel_slug}-{yt_video_id}-raw.srt
        {brand_slug}/{model_slug}/processed_transcripts/{brand_slug}-{model_slug}-{channel_slug}-{yt_video_id}-processed.txt
        {brand_slug}/{model_slug}/processed_transcripts/{brand_slug}-{model_slug}-{channel_slug}-{yt_video_id}-translated.txt

    Channel slug: slugify(channel_name).
    If the slug is empty after slugification (e.g. pure Devanagari/Hindi name with no ASCII
    characters), fall back to yt_channel_id as the channel slug.
    This prevents paths like:
        samsung/galaxy-s25/raw_transcripts/samsung-galaxy-s25--raw.srt  ← broken double hyphen

    Uses the existing slugify() function already in this file.

    All three paths are always returned. translated_path is included even for English
    videos — the orchestrator decides whether to write to it based on language_code.
    """
    brand_slug = slugify(brand)
    model_slug = slugify(model)
    channel_slug = slugify(channel_name)
    if not channel_slug:
        channel_slug = yt_channel_id  # fallback for pure non-ASCII channel names

    base = f"{brand_slug}/{model_slug}"
    prefix = f"{brand_slug}-{model_slug}-{channel_slug}-{yt_video_id}"

    return {
        "srt_path":        f"{base}/raw_transcripts/{prefix}-raw.srt",
        "processed_path":  f"{base}/processed_transcripts/{prefix}-processed.txt",
        "translated_path": f"{base}/processed_transcripts/{prefix}-translated.txt",
    }
