"""
Scraper API Router.

Phase 9 (Task 9.1): POST /admin/scrape
Phase 9 (Task 9.2): GET  /admin/scrape/progress
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.scrape_orchestrator import get_current_step, run_scrape

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


# ---------------------------------------------------------------------------
# Phase 9 — Task 9.2: GET /admin/scrape/progress
# ---------------------------------------------------------------------------

@router.get("/scrape/progress")
async def get_scrape_progress(
    brand:      str = Query(...),
    model_name: str = Query(...),
    site_name:  str = Query(...),
):
    """
    Returns the live scraping progress for a given phone+site combination.
    Reads from an in-memory dict in scrape_orchestrator.py — zero DB cost.
    Polled by the frontend every 1.5s while scrapeState.kind === 'loading'.

    Returns:
        { "current_step": "calling_firecrawl" | "processing_response" |
                          "uploading_files" | "writing_to_database" | null }
    """
    return {"current_step": get_current_step(brand, model_name, site_name)}
