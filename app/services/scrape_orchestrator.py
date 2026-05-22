"""
Phase 8 — Task 8.3: Scrape Orchestrator

run_scrape() is the single entry point for all scraping.
It coordinates every phase: registry lookup, atomic claim, execution logging,
config resolution, Firecrawl call, response validation, screenshot assembly,
Storage upload, DB insert, and status transition.

Failure handling invariants:
  - execution_id / url_id / claimed initialised to None/False before the outer try
    so the failure path never crashes on unset variables.
  - Inner try/except cleans up orphan Storage files before re-raising.
  - HTTPException(409) passes straight through — status is NOT reset because
    claim_for_scraping() returned False (we never owned the row).
  - Outer except resets status to 'not_scraped' only if we own the row (claimed=True).
"""

import logging
import re
from datetime import datetime

from fastapi import HTTPException

from app.repositories.config_repository import get_config_template, get_lookup_source
from app.repositories.execution_repository import create_execution, finish_execution
from app.repositories.raw_data_repository import (
    insert_raw_scraped_data,
    delete_raw_scraped_data_by_execution,   # add this
)
from app.repositories.url_registry_repository import (
    claim_for_scraping,
    get_url_registry_row,
    set_status_not_scraped,
    set_status_scraped_raw,
)
from app.services.config_resolver import resolve_config
from app.services.firecrawl_client import (
    scrape_with_firecrawl,
    validate_firecrawl_response,
)
from app.services.storage_service import delete_file, upload_file
from app.utils.image_utils import extract_screenshots
from app.utils.path_builder import build_storage_paths

logger = logging.getLogger(__name__)

# ── In-memory progress store ─────────────────────────────────────────────────────────────
# Keyed by "brand:model_name:site_name". Safe because --workers 1 is required.
# The frontend polls GET /admin/scrape/progress which reads directly from here.
# No DB writes, no migrations — pure in-process state.

_progress: dict[str, str] = {}


def _set_step(brand: str, model_name: str, site_name: str, step: str) -> None:
    _progress[f"{brand}:{model_name}:{site_name}"] = step


def _clear_step(brand: str, model_name: str, site_name: str) -> None:
    _progress.pop(f"{brand}:{model_name}:{site_name}", None)


def get_current_step(brand: str, model_name: str, site_name: str) -> str | None:
    """Called by the progress endpoint in scraper.py."""
    return _progress.get(f"{brand}:{model_name}:{site_name}")

async def run_scrape(brand: str, model_name: str, site_name: str) -> dict:
    """
    Full scrape flow. Returns dict with success, execution_id, message.

    Args:
        brand:      Brand name from url_registry (e.g. "Samsung").
        model_name: Model name from url_registry (e.g. "Galaxy S25 Ultra").
        site_name:  Site slug from lookup_source_registry (e.g. "gsmarena").

    Returns:
        {"success": True,  "execution_id": int, "message": "Scrape completed successfully"}
        {"success": False, "execution_id": int_or_None, "message": str(exception)}

    Raises:
        HTTPException(409): If the row is already being scraped (passes through unchanged).
    """
    # Initialize to None — failure path must never assume these were set
    execution_id = None
    url_id       = None
    claimed      = False

    try:
        # 1. Fetch registry row
        row    = await get_url_registry_row(brand, model_name, site_name)
        url_id = row["url_id"]

        # 2. Atomically claim the row — single UPDATE+WHERE, no race condition
        claimed = await claim_for_scraping(url_id)
        if not claimed:
            raise HTTPException(
                status_code=409,
                detail="Scrape already in progress for this source.",
            )

        # 3. Capture started_at on the orchestrator clock — NEVER use DB NOW()
        #    Both started_at and finished_at must come from the same clock so
        #    duration_ms calculation in finish_execution() is always correct.
        started_at = datetime.utcnow()

        # 4. Fetch template config
        source   = await get_lookup_source(site_name)          # → template_id, etc.
        template = await get_config_template(source["template_id"])  # → config_template, template_name

        # 5. Insert execution row with explicit started_at
        execution_id = await create_execution(url_id, source["template_id"], started_at)

        # 6. Resolve config — inject {{URL}}, {{PHONE_ID}}, or {{PHONE_NAME}}
        resolved = resolve_config(
            template["config_template"],
            site_name,
            row["url"],
            row.get("scrape_identifier"),
        )

        # 7. Call Firecrawl
        _set_step(brand, model_name, site_name, "calling_firecrawl")
        response = await scrape_with_firecrawl(resolved)

        # 8. Validate response structure before touching any data
        _set_step(brand, model_name, site_name, "processing_response")
        validate_firecrawl_response(response, template["template_name"])

        # 9. Success block — track uploads for cleanup on partial failure
        uploaded_paths = []
        raw_scraped_data_inserted = False   # Track if DB row was created — for orphan cleanup
        try:
            markdown_content = response["data"]["markdown"]
            # Strip invalid JSON escape sequences (e.g. \| \™) introduced by Firecrawl's
            # markdown renderer. These cause json.loads() to fail if Gemini copies them
            # verbatim into evidence_text fields in the extraction output.
            # 1. Strip invalid JSON escape sequences (e.g. \| \™) from Firecrawl markdown.
            #    Gemini copies evidence_text verbatim — invalid escapes crash json.loads().
            markdown_content = re.sub(r'\\([^"\\/bfnrtu\n\r])', r'\1', markdown_content)
            # 2. Strip markdown hyperlinks, preserving only the visible label.
            #    Firecrawl wraps many spec values and headers in links e.g.:
            #    [Technology](https://www.gsmarena.com/network-bands.php3) → Technology
            #    [eSIM](https://www.gsmarena.com/glossary.php3?term=esim) → eSIM
            #    These URLs waste tokens, pollute evidence_text, and increase hallucination risk.
            #    Table structure, bullets, and separators are NOT affected by this regex.
            markdown_content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', markdown_content)
            # 3. Remove ALL blank lines entirely (including whitespace-only lines).
            #    Firecrawl produces excessive vertical whitespace between sections.
            #    Removes both single and multiple consecutive blank lines.
            #    Before: "Line A\n\nLine B\n\n\n\nLine C"
            #    After:  "Line A\nLine B\nLine C"
            markdown_content = re.sub(r'\n\s*\n+', '\n', markdown_content)
            
            markdown_bytes   = markdown_content.encode("utf-8")  # explicit UTF-8
            screenshots      = await extract_screenshots(response, template["template_name"])
            paths            = build_storage_paths(brand, model_name, site_name)

            _set_step(brand, model_name, site_name, "uploading_files")
            await upload_file(paths["markdown_path"], markdown_bytes, "text/markdown")
            uploaded_paths.append(paths["markdown_path"])

            before_path = None
            after_path  = None

            if screenshots["before"]:
                ss_before = screenshots["before"]
                before_path = paths["screenshot_before_path"].replace(
                    ".webp", ".png" if ss_before["mime_type"] == "image/png" else ".webp"
                )
                await upload_file(before_path, ss_before["data"], ss_before["mime_type"])
                uploaded_paths.append(before_path)

            if screenshots["after"]:
                ss_after = screenshots["after"]
                after_path = paths["screenshot_after_path"].replace(
                    ".webp", ".png" if ss_after["mime_type"] == "image/png" else ".webp"
                )
                await upload_file(after_path, ss_after["data"], ss_after["mime_type"])
                uploaded_paths.append(after_path)

            # Insert raw_scraped_data BEFORE updating url_registry status.
            # If the DB insert fails, the outer except will delete the uploaded files
            # and reset status to 'not_scraped'. Order matters: DB first, status second.
            finished_at = datetime.utcnow()
            _set_step(brand, model_name, site_name, "writing_to_database")
            await insert_raw_scraped_data(
                url_registry_id=url_id,
                execution_id=execution_id,
                template_id=source["template_id"],
                phone_brand=brand,
                phone_model=model_name,
                markdown_path=paths["markdown_path"],
                screenshot_before_path=before_path,
                screenshot_after_path=after_path,
                file_size_bytes=len(markdown_bytes),
            )
            raw_scraped_data_inserted = True    # Row exists in DB from this point on

            await finish_execution(execution_id, True, started_at, finished_at)
            await set_status_scraped_raw(url_id)
            _clear_step(brand, model_name, site_name)

            return {
                "success":      True,
                "execution_id": execution_id,
                "message":      "Scrape completed successfully",
            }

        except Exception as upload_or_db_error:
            # Clean up any files already uploaded to avoid orphans in Storage.
            # delete_file() is best-effort and never raises, so cleanup cannot
            # shadow the original error.
            for path in uploaded_paths:
                await delete_file(path)

            # Orphan row cleanup: if the DB insert succeeded before this exception,
            # delete the row to prevent extraction from loading a row with deleted storage paths.
            if raw_scraped_data_inserted:
                try:
                    await delete_raw_scraped_data_by_execution(execution_id)
                except Exception as cleanup_exc:
                    logger.warning(
                        "run_scrape: failed to clean orphan raw_scraped_data row "
                        "for execution_id=%s: %s",
                        execution_id, cleanup_exc,
                    )
            raise  # re-raise so the outer except handles logging and status reset

    except HTTPException:
        raise  # 409 passes through unchanged — do not reset status for this case

    except Exception as e:
        finished_at = datetime.utcnow()

        # Log the execution if it was created.
        # Individually guarded: a raise here must NOT prevent the status reset below.
        if execution_id is not None:
            try:
                await finish_execution(
                    execution_id,
                    False,
                    started_at,
                    finished_at,
                    error_message=str(e),
                )
            except Exception as log_error:
                logger.error(
                    "Failed to finalize execution log for execution_id=%s: %s",
                    execution_id,
                    log_error,
                )

        # Always reset status if we own the row — prevents permanent stuck state.
        # Individually guarded: a stuck 'currently_scraping' row requires manual
        # DB intervention, so we log at ERROR with a CRITICAL prefix.
        if claimed and url_id is not None:
            try:
                await set_status_not_scraped(url_id)
            except Exception as reset_error:
                logger.error(
                    "CRITICAL: Failed to reset status to not_scraped for url_id=%s: %s — "
                    "row may be permanently stuck at currently_scraping.",
                    url_id,
                    reset_error,
                )
        _clear_step(brand, model_name, site_name)

        return {
            "success":      False,
            "execution_id": execution_id,
            "message":      str(e),
        }
