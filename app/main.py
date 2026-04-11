"""
Raw Data Pipeline — FastAPI Application Entry Point

Startup:
  - Validates all required environment variables (see app/core/config.py)
  - Mounts the admin router

Run with:
  uvicorn app.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import validate_config
from app.api import scraper, youtube, extraction, inference, approval, admin_lookup
from app.services.ecd_generator import pre_warm_ecd
from app.services.normalizer import build_lookup_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once on server startup before accepting requests."""
    validate_config()
    pre_warm_ecd()          # Task 0.4 — load and validate ECD YAML files immediately
    await build_lookup_cache()  # Task 4.1 — populate LOOKUP_CACHE for normaliser

    # MIN-3: Verify lx.io.create_html_from_jsonl exists in the installed LangExtract version.
    # If the API changed (e.g. to lx.visualize), this fails at startup rather than first use.
    try:
        import langextract as lx
        if not callable(getattr(lx.io, "create_html_from_jsonl", None)):
            raise AttributeError(
                "lx.io.create_html_from_jsonl is not callable in the installed LangExtract "
                "version. Check the installed version and update the visualization endpoint "
                "in app/api/approval.py accordingly."
            )
        logger.info("Startup: lx.io.create_html_from_jsonl verified OK.")
    except AttributeError as exc:
        raise RuntimeError(f"LangExtract API mismatch: {exc}") from exc

    logger.info("Raw pipeline backend started.")
    yield
    logger.info("Raw pipeline backend shut down.")


app = FastAPI(
    title="Raw Data Pipeline — Admin API",
    description="Backend for the phone specs raw data scraping pipeline.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(admin_lookup.router)
app.include_router(scraper.router)
app.include_router(youtube.router)
app.include_router(inference.router)
app.include_router(extraction.router)
app.include_router(approval.router)
