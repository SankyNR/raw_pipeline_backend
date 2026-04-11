"""
Phase 7 — URL Registry Status Updates

Task 7.1:
    get_url_registry_row()   — Fetch the full url_registry row for a (brand, model, site).
    claim_for_scraping()     — Atomic single-UPDATE lock: sets 'currently_scraping' only if not
                               already claimed. Avoids race conditions from SELECT+UPDATE patterns.
    set_status_scraped_raw() — Mark the URL as successfully scraped.
    set_status_not_scraped() — Release the lock on failure, allowing future retries.
"""

import re
from app.core.supabase_client import get_client
from app.services.video_relevance_filter import _normalize_text

# ---------------------------------------------------------------------------
# Task 7.1a — Fetch row
# ---------------------------------------------------------------------------

async def get_url_registry_row(brand: str, model_name: str, site_name: str) -> dict:
    """
    Fetches the full url_registry row matching (brand, model_name, site_name).

    Returns the row as a dict.

    Raises:
        ValueError: If no matching row is found.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("url_registry")
        .select("*")
        .eq("brand", brand)
        .eq("model_name", model_name)
        .eq("site_name", site_name)
        .execute()
    )

    if not result.data:
        raise ValueError(
            f"No url_registry row found for brand={brand!r}, "
            f"model_name={model_name!r}, site_name={site_name!r}."
        )

    return result.data[0]


# ---------------------------------------------------------------------------
# Task 7.1b — Atomic claim
# ---------------------------------------------------------------------------

async def claim_for_scraping(url_id: int) -> bool:
    """
    Atomically sets status = 'currently_scraping' ONLY IF the current status
    is NOT already 'currently_scraping'.

    Uses a single UPDATE with a WHERE clause — NOT a separate SELECT + UPDATE.
    The two-step approach has a race condition where two simultaneous requests
    can both read 'not_scraped' and both proceed. This implementation prevents that.

    SQL equivalent:
        UPDATE pipeline.url_registry
        SET    status = 'currently_scraping', updated_at = NOW()
        WHERE  url_id = {url_id}
          AND  status != 'currently_scraping'
        RETURNING url_id

    Note: statuses 'scraped_raw' and 'stored_mainDB' ARE claimable — the Admin UI
    shows a warning popup before re-scraping, but the backend does not block it.

    Returns:
        True  — row was claimed (status was updated).
        False — row was already 'currently_scraping'; no update was made.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("url_registry")
        .update({"status": "currently_scraping"})
        .eq("url_id", url_id)
        .neq("status", "currently_scraping")
        .execute()
    )

    # If RETURNING returned a row, the update succeeded (claim granted).
    return bool(result.data)


# ---------------------------------------------------------------------------
# Task 7.1c — Status setters
# ---------------------------------------------------------------------------

async def set_status_scraped_raw(url_id: int) -> None:
    """
    Sets status = 'scraped_raw'.
    Called by the orchestrator after a successful upload (raw file is in Storage).
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("url_registry")
        .update({"status": "scraped_raw"})
        .eq("url_id", url_id)
        .execute()
    )
    if not result.data:
        raise RuntimeError(
            f"set_status_scraped_raw() failed — url_id={url_id} not found or not updated."
        )


async def set_status_not_scraped(url_id: int) -> None:
    """
    Sets status = 'not_scraped'.
    Called by the orchestrator on failure to release the lock and allow retry.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("url_registry")
        .update({"status": "not_scraped"})
        .eq("url_id", url_id)
        .execute()
    )
    if not result.data:
        raise RuntimeError(
            f"set_status_not_scraped() failed — url_id={url_id} not found or not updated."
        )


# ---------------------------------------------------------------------------
# YouTube pipeline — Canonical anchor resolver
# ---------------------------------------------------------------------------

async def get_gsmarena_anchor(brand: str, model_name: str) -> dict:
    """
    Fetches the single url_registry row for this phone where site_name = 'gsmarena'.
    This row is the canonical anchor for all YouTube enrichment for this phone.

    The GSMArena row is used as the anchor because:
      - It is always present per phone (required source).
      - It is always unique per phone (no per-site duplicates).
      - It is never deleted in normal operations.

    Raises ValueError with a clear message if:
      - Zero rows found: phone not in pipeline, or gsmarena row missing.
      - More than one row found: data integrity violation — ambiguous anchor
        would produce fragmented or duplicate video registrations.

    Returns the full row dict on success.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("url_registry")
        .select("*")
        .eq("brand", brand)
        .eq("model_name", model_name)
        .eq("site_name", "gsmarena")
        .execute()
    )

    rows = result.data or []

    if len(rows) == 0:
        raise ValueError(
            f"No GSMArena anchor found for {brand!r} {model_name!r}. "
            f"Ensure the phone has a url_registry row with site_name='gsmarena' "
            f"before running a YouTube search."
        )

    if len(rows) > 1:
        raise ValueError(
            f"Data integrity violation: {len(rows)} GSMArena rows found for "
            f"{brand!r} {model_name!r}. Expected exactly 1. "
            f"url_ids: {[r['url_id'] for r in rows]}"
        )

    return rows[0]


async def check_connectivity_variants(brand: str, model_name: str) -> dict:
    """
    Checks if this phone exists in both 4G and 5G variants in the registry.
    Shared "base model" is identified by stripping 4G/5G suffixes.

    Returns:
        {"has_4g": bool, "has_5g": bool}
    """
    # 1. Identify base model name by stripping connectivity tokens and normalizing
    base_model = _normalize_text(
        re.sub(r'\b(4g|5g)\b', '', model_name, flags=re.IGNORECASE)
    )

    # 2. Fetch all models for this brand to find variants sharing the same base
    result = (
        get_client()
        .schema("pipeline")
        .table("url_registry")
        .select("model_name")
        .eq("brand", brand)
        .execute()
    )

    models = [r["model_name"] for r in result.data or []]

    has_4g = False
    has_5g = False

    # 3. Detect if any model variants share the same base name and have 4G/5G tokens
    for m in models:
        # Strip connectivity from each model in DB and normalize for robust comparison
        m_base = _normalize_text(
            re.sub(r'\b(4g|5g)\b', '', m, flags=re.IGNORECASE)
        )
        if m_base.lower() == base_model.lower():
            if re.search(r'\b4g\b', m, re.IGNORECASE):
                has_4g = True
            if re.search(r'\b5g\b', m, re.IGNORECASE):
                has_5g = True

    return {
        "has_4g": has_4g,
        "has_5g": has_5g
    }
