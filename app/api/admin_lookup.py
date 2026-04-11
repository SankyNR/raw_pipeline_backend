"""
Admin Lookup API Router.

Phase 0  (Task 0.3):  GET /admin/health/db
Phase 1  (Tasks 1.1–1.3): GET /admin/brands, /admin/models, /admin/sites
"""

import logging
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Query

from app.core.supabase_client import get_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# ---------------------------------------------------------------------------
# Task 0.3 — DB Health Check
# ---------------------------------------------------------------------------

@router.get("/health/db")
async def health_db():
    """
    Confirms Supabase Postgres connectivity by running a minimal query
    against the pipeline schema.

    Returns: { "status": "ok" } on success.
    Raises HTTP 500 on any DB error.
    """
    try:
        get_client() \
            .schema("pipeline") \
            .table("url_registry") \
            .select("url_id") \
            .limit(1) \
            .execute()
        return {"status": "ok"}
    except Exception as e:
        logger.error("DB health check failed: %s", e)
        raise HTTPException(status_code=500, detail=f"DB connection failed: {e}")


# ---------------------------------------------------------------------------
# Task 1.1 — GET /admin/brands
# ---------------------------------------------------------------------------

@router.get("/brands")
async def get_brands():
    """
    Returns all distinct brand values from pipeline.url_registry,
    sorted alphabetically.

    Response: { "brands": ["Apple", "CMF", "Google", ...] }
    """
    try:
        result = (
            get_client()
            .schema("pipeline")
            .table("url_registry")
            .select("brand")
            .execute()
        )
    except Exception as e:
        logger.error("get_brands DB error: %s", e)
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    brands = sorted({row["brand"] for row in result.data})
    return {"brands": brands}


# ---------------------------------------------------------------------------
# Task 1.2 — GET /admin/models?brand=...
# ---------------------------------------------------------------------------

# Status sets used for colour logic
_DONE_STATUSES = {"scraped_raw", "stored_mainDB"}
_NOT_DONE_STATUSES = {"not_scraped", "currently_scraping"}


def _compute_model_color(statuses: list[str]) -> str:
    """
    Determines the aggregate colour for a model based on the statuses of
    all its source sites.

    | Condition                                                  | colour   |
    |------------------------------------------------------------|----------|
    | All sites are `not_scraped`                                | "red"    |
    | Mix — at least one not-done, at least one done             | "orange" |
    | All sites are `scraped_raw`                                | "yellow" |
    | All sites are `stored_mainDB`                              | "green"  |

    `currently_scraping` counts as "not done" for the orange condition.
    """
    status_set = set(statuses)

    has_done = bool(status_set & _DONE_STATUSES)
    has_not_done = bool(status_set & _NOT_DONE_STATUSES)

    if has_done and has_not_done:
        return "orange"

    if not has_done:
        # Everything is in the not-done bucket
        return "red"

    # All are done — now distinguish yellow vs green
    if all(s == "stored_mainDB" for s in statuses):
        return "green"
    return "yellow"


@router.get("/models")
async def get_models(brand: str = Query(..., description="Brand name to filter by")):
    """
    Returns all models for the given brand with an aggregate colour
    based on the scrape statuses of their source sites.

    Response:
        {
          "models": [
            { "model_name": "Galaxy S25", "color": "red" },
            ...
          ]
        }
    """
    try:
        result = (
            get_client()
            .schema("pipeline")
            .table("url_registry")
            .select("model_name, status")
            .eq("brand", brand)
            .execute()
        )
    except Exception as e:
        logger.error("get_models DB error (brand=%s): %s", brand, e)
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    if not result.data:
        return {"models": []}

    # Group statuses by model_name, preserving insertion order for sorted output
    model_statuses: dict[str, list[str]] = defaultdict(list)
    for row in result.data:
        model_statuses[row["model_name"]].append(row["status"])

    models = [
        {"model_name": model_name, "color": _compute_model_color(statuses)}
        for model_name, statuses in sorted(model_statuses.items())
    ]
    return {"models": models}


# ---------------------------------------------------------------------------
# Task 1.3 — GET /admin/sites?brand=...&model=...
# ---------------------------------------------------------------------------

def _site_badge_color(status: str) -> str:
    """
    Maps a url_registry status value to a badge colour for the site level.

    | status             | badge_color |
    |--------------------|-------------|
    | not_scraped        | "red"       |
    | currently_scraping | "blue"      |
    | scraped_raw        | "green"     |
    | stored_mainDB      | "green"     |
    """
    if status == "currently_scraping":
        return "blue"
    if status in _DONE_STATUSES:
        return "green"
    return "red"  # not_scraped (and any unknown value)


@router.get("/sites")
async def get_sites(
    brand: str = Query(..., description="Brand name"),
    model: str = Query(..., description="Model name"),
):
    """
    Returns all source sites for the given brand + model, each with its
    current scrape status and badge colour.

    Joins url_registry → lookup_source_registry to get display_name.

    Response:
        {
          "sites": [
            {
              "site_name": "gsmarena",
              "display_name": "GSMArena",
              "status": "scraped_raw",
              "badge_color": "green"
            },
            ...
          ]
        }
    """
    try:
        result = (
            get_client()
            .schema("pipeline")
            .table("url_registry")
            .select("site_name, status, lookup_source_registry(display_name)")
            .eq("brand", brand)
            .eq("model_name", model)
            .execute()
        )
    except Exception as e:
        logger.error(
            "get_sites DB error (brand=%s, model=%s): %s", brand, model, e
        )
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    if not result.data:
        return {"sites": []}

    sites = [
        {
            "site_name": row["site_name"],
            "display_name": (
                row["lookup_source_registry"]["display_name"]
                if row.get("lookup_source_registry")
                else row["site_name"]   # fallback if join returns nothing
            ),
            "status": row["status"],
            "badge_color": _site_badge_color(row["status"]),
        }
        for row in result.data
    ]
    return {"sites": sites}
