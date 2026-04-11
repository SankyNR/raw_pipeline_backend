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
# Phase L1 (LangExtract Migration) — LangExtract uses its own key.
# MED-2: If not explicitly set, auto-assigned from GEMINI_API_KEY in validate_config().
LANGEXTRACT_API_KEY: str = os.environ.get("LANGEXTRACT_API_KEY", "")

# CRIT-3: Shared secret for admin/approval endpoint authentication.
# Set ADMIN_API_KEY in .env to a random strong token.
ADMIN_API_KEY: str = os.environ.get("ADMIN_API_KEY", "")


def validate_config() -> None:
    """
    Called at startup to confirm all required env vars are present.
    Raises RuntimeError if any are missing, clearly naming which key is absent.
    Does NOT log or print any secret values.

    MED-2: Auto-assigns LANGEXTRACT_API_KEY from GEMINI_API_KEY if not set separately.
    CRIT-4: Runs LangExtract example coverage check — server fails to start if examples
            are missing required extraction classes or attributes (catches schema drift).
    """
    global LANGEXTRACT_API_KEY

    # MED-2: Auto-assign LANGEXTRACT_API_KEY from GEMINI_API_KEY if absent.
    # Both keys share the same value — this avoids requiring operators to set both.
    if not LANGEXTRACT_API_KEY and GEMINI_API_KEY:
        os.environ["LANGEXTRACT_API_KEY"] = GEMINI_API_KEY
        LANGEXTRACT_API_KEY = GEMINI_API_KEY
        logger.info("Config: LANGEXTRACT_API_KEY auto-assigned from GEMINI_API_KEY.")

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
    if not LANGEXTRACT_API_KEY:
        missing.append("LANGEXTRACT_API_KEY")
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
    logger.info("Config: LANGEXTRACT_API_KEY loaded.")
    logger.info("Config: ADMIN_API_KEY loaded.")

    # CRIT-4: LangExtract example coverage check.
    # Ensures example files have not drifted from the current extraction schema.
    # If run_coverage_check() raises AssertionError, the server must not start.
    try:
        from app.config.langextract_examples_run_a_v1 import run_coverage_check
        run_coverage_check()
        logger.info("Config: LangExtract Run A example coverage check PASSED.")
    except AssertionError as exc:
        raise RuntimeError(
            f"LangExtract Run A example coverage check FAILED — server will not start. "
            f"Fix app/config/langextract_examples_run_a_v1.py. Details: {exc}"
        ) from exc
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import run_coverage_check from langextract_examples_run_a_v1: {exc}. "
            f"Ensure the file exists and has no syntax errors."
        ) from exc

    logger.info("Config loaded successfully. All 7 required environment variables are present.")
