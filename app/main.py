"""
Raw Data Pipeline — FastAPI Application Entry Point

Startup:
  - Validates all required environment variables (see app/core/config.py)
  - Pre-warms ECD YAML cache
  - Populates normaliser lookup cache
  - Mounts all routers

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


class _SuppressProgressPolls(logging.Filter):
    """
    Drops uvicorn access-log lines for GET /admin/scrape/progress.
    These are high-frequency polling requests (every 3s) that add noise
    without useful information. All other endpoints remain fully logged.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "scrape/progress" not in msg and "youtube/progress" not in msg


logging.getLogger("uvicorn.access").addFilter(_SuppressProgressPolls())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once on server startup before accepting requests."""
    validate_config()                 # includes v2 coverage check
    pre_warm_ecd()                    # Task 0.4 — load and validate ECD YAML files
    await build_lookup_cache()        # Task 4.1 — populate LOOKUP_CACHE for normaliser
    logger.info("Extraction pipeline backend (v5) started.")
    yield
    logger.info("Extraction pipeline backend shut down.")


app = FastAPI(
    title="Raw Data Pipeline — Admin API",
    description="Backend for the phone specs raw data scraping pipeline.",
    version="0.1.0",
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FIX-11: Attach rate limiter — extraction endpoints use @limiter.limit() decorators
from app.api.extraction import limiter  # noqa: E402
from slowapi import _rate_limit_exceeded_handler  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(admin_lookup.router)
app.include_router(scraper.router)
app.include_router(youtube.router)
app.include_router(inference.router)
app.include_router(extraction.router)
app.include_router(approval.router)
