"""
Scraper API Router.

Phase 9 (Task 9.1): POST /admin/scrape
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.scrape_orchestrator import run_scrape

router = APIRouter(prefix="/admin", tags=["scraper"])


# ---------------------------------------------------------------------------
# Phase 9 — Task 9.1: POST /admin/scrape
# ---------------------------------------------------------------------------

class ScrapeRequest(BaseModel):
    model_config = {"protected_namespaces": ()}  # suppress false-positive on model_name field

    brand:      str
    model_name: str
    site_name:  str


@router.post("/scrape")
async def trigger_scrape(body: ScrapeRequest):
    """
    Triggers a full scrape for a single (brand, model_name, site_name) combination.

    Success (HTTP 200):
        { "success": true, "execution_id": 42, "message": "Scrape completed successfully" }

    Already in progress (HTTP 409):
        { "detail": "Scrape already in progress for this source." }

    Any other failure (HTTP 500):
        { "detail": "<error message>" }
    """
    try:
        result = await run_scrape(body.brand, body.model_name, body.site_name)

        # run_scrape() catches all non-409 errors and returns a dict with success=False.
        # It never re-raises to the caller — so the only way to return HTTP 500 on failure
        # is to check result["success"] here explicitly.
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])

        return result

    except HTTPException:
        raise  # 409 (and now 500 from above) pass through unchanged
