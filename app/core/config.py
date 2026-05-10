"""
Environment Config Loader

Loads environment variables from .env using python-dotenv.
Exposes all required API keys as module-level settings.
Does NOT print secret values on startup.
"""

import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env from the project root (one level up from app/)
load_dotenv()

SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
FIRECRAWL_API_KEY: str = os.environ.get("FIRECRAWL_API_KEY", "")
YOUTUBE_DATA_API_KEY: str = os.environ.get("YOUTUBE_DATA_API_KEY", "")
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
# Phase 9: LANGEXTRACT_API_KEY removed — pipeline uses Gemini directly.

# CRIT-3: Shared secret for admin/approval endpoint authentication.
# Set ADMIN_API_KEY in .env to a random strong token.
ADMIN_API_KEY: str = os.environ.get("ADMIN_API_KEY", "")

# Residential Proxy Configuration (YouTube transcript fetching).
# Set PROXY_ENABLED=true in .env to activate. If false or unset, the transcript
# client falls back to a plain session (suitable for local dev/testing).
PROXY_ENABLED: bool = os.environ.get("PROXY_ENABLED", "false").lower() == "true"
PROXY_SCHEME:   str = os.environ.get("PROXY_SCHEME", "http")
PROXY_HOST:     str = os.environ.get("PROXY_HOST", "")
PROXY_PORT:     str = os.environ.get("PROXY_PORT", "")
PROXY_USERNAME: str = os.environ.get("PROXY_USERNAME", "")
PROXY_PASSWORD: str = os.environ.get("PROXY_PASSWORD", "")


def validate_config() -> None:
    """
    Called at startup to confirm all required env vars are present.
    Raises RuntimeError if any are missing, clearly naming which key is absent.
    Does NOT log or print any secret values.

    Phase 9: LANGEXTRACT_API_KEY removed from required vars.
    Runs extraction_examples_run_a_v2 coverage check to catch schema drift
    between the few-shot examples and the current extraction schema.
    """
    missing = []
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_SERVICE_ROLE_KEY:
        missing.append("SUPABASE_SERVICE_ROLE_KEY")
    if not FIRECRAWL_API_KEY:
        missing.append("FIRECRAWL_API_KEY")
    if not YOUTUBE_DATA_API_KEY:
        missing.append("YOUTUBE_DATA_API_KEY")
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if not ADMIN_API_KEY:
        missing.append("ADMIN_API_KEY")

    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Check your .env file."
        )

    logger.info("Config: SUPABASE_URL loaded.")
    logger.info("Config: SUPABASE_SERVICE_ROLE_KEY loaded.")
    logger.info("Config: FIRECRAWL_API_KEY loaded.")
    logger.info("Config: YOUTUBE_DATA_API_KEY loaded.")
    logger.info("Config: GEMINI_API_KEY loaded.")
    logger.info("Config: ADMIN_API_KEY loaded.")

    # Phase 9 Task 9.1: v5 example coverage check.
    # Ensures extraction_examples_run_a_v2 has not drifted from the current
    # extraction schema. If run_coverage_check() raises, the server will not start.
    try:
        from app.config.extraction_examples_run_a_v2 import run_coverage_check
        run_coverage_check()
        logger.info("Config: Extraction Run A (v2) example coverage check PASSED.")
    except ImportError:
        # run_coverage_check not yet defined in v2 — skip gracefully.
        logger.info(
            "Config: extraction_examples_run_a_v2 has no run_coverage_check — skipping."
        )
    except AssertionError as exc:
        raise RuntimeError(
            f"Extraction Run A (v2) example coverage check FAILED — server will not start. "
            f"Fix app/config/extraction_examples_run_a_v2.py. Details: {exc}"
        ) from exc

    if PROXY_ENABLED:
        logger.info(
            "Config: Residential proxy ENABLED (host=%s, port=%s). Cookies disabled.",
            PROXY_HOST, PROXY_PORT,
        )
        proxy_missing = [
            name for name, val in [
                ("PROXY_HOST", PROXY_HOST),
                ("PROXY_PORT", PROXY_PORT),
                ("PROXY_USERNAME", PROXY_USERNAME),
                ("PROXY_PASSWORD", PROXY_PASSWORD),
            ]
            if not val
        ]
        if proxy_missing:
            raise RuntimeError(
                f"PROXY_ENABLED=true but the following proxy environment variables are "
                f"missing or empty: {', '.join(proxy_missing)}. Check your .env file."
            )
    else:
        logger.info("Config: PROXY_ENABLED=false — transcript client will use direct connection.")

    logger.info("Config loaded successfully. All required environment variables are present.")
