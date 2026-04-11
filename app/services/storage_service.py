"""
Storage Service — Supabase Storage Helpers

Task 4.1 (original): Upload and delete helpers.
Task 0.1 (extraction pipeline): fetch_file_content() — generates a signed URL,
fetches content via HTTP, and discards the URL. Never stores or logs signed URLs.

Operates on the private 'raw_files' bucket using the service role client,
which bypasses Row Level Security (RLS).
"""

import asyncio
import logging

import httpx

from app.core.supabase_client import get_client

logger = logging.getLogger(__name__)

BUCKET = "raw_files"

_FETCH_TIMEOUT_SECONDS = 30
_FETCH_MAX_RETRIES = 3   # m2: retry on transient 5xx from Supabase Storage


class StorageFetchError(Exception):
    """
    Raised when fetch_file_content() fails to retrieve a file from Supabase Storage.
    Wraps the underlying cause for upstream error handling and logging.
    """


async def upload_file(path: str, file_bytes: bytes, content_type: str) -> str:
    """
    Uploads file_bytes to the raw_files bucket at the given path.

    Args:
        path:         Storage path relative to the bucket root,
                      e.g. "samsung/galaxy-s25/GSMarena/samsung-galaxy-s25-gsm-21-02-26-22-10-00.md"
        file_bytes:   Raw bytes to upload.
        content_type: MIME type, e.g. "text/markdown" or "image/webp".

    Returns:
        The storage path string (same as the input path).

    Raises:
        Exception: On any upload failure — propagated to the orchestrator.
    """
    try:
        await asyncio.to_thread(
            lambda: get_client().storage.from_(BUCKET).upload(
                path=path,
                file=file_bytes,
                file_options={"content-type": content_type},
            )
        )
    except Exception as e:
        raise RuntimeError(f"Storage upload failed for path '{path}': {e}")
    logger.info("Uploaded to raw_files: %s (%d bytes)", path, len(file_bytes))
    return path


async def delete_file(path: str) -> None:
    """
    Deletes a file from the raw_files bucket at the given path.

    Used by the orchestrator to clean up orphaned files when a partial
    upload succeeds but a later step (DB insert, etc.) fails.

    Does NOT raise if the file does not exist — silent no-op is correct here,
    because cleanup must be best-effort and must never shadow the original error.

    Args:
        path: Storage path relative to the bucket root.
    """
    try:
        get_client().storage.from_(BUCKET).remove([path])
        logger.info("Deleted from raw_files: %s", path)
    except Exception as e:
        # Log but swallow — a missing file during cleanup must not crash
        # the failure path that is already handling a real error.
        logger.warning("delete_file silent error (path=%s): %s", path, e)


# ---------------------------------------------------------------------------
# Task 0.1 (Extraction Pipeline) — fetch_file_content
# ---------------------------------------------------------------------------

async def fetch_file_content(storage_path: str, ttl_seconds: int = 60) -> str:
    """
    Fetches the UTF-8 text content of a file from the raw_files private bucket.

    Generates a signed URL with the given TTL (default 60 seconds), performs a
    single HTTP GET to retrieve the file bytes, then discards the URL immediately.

    The signed URL is NEVER stored, cached, logged, or passed to the frontend.
    Every call generates a fresh URL — do not call this more than once per need.

    Args:
        storage_path: Path within the raw_files bucket, e.g.
                      "samsung/galaxy-s25-ultra/GSMarena/samsung-...-gsm-....md"
        ttl_seconds:  Signed URL TTL in seconds. Default 60.
                      Keep short — unused URLs still count against rate limits.

    Returns:
        The file content decoded as UTF-8 string.

    Raises:
        StorageFetchError: On any failure (signed URL generation, HTTP error,
                           decode error). Wraps the underlying exception.
    """
    # Step 1: Generate signed URL (never stored or logged)
    try:
        response = await asyncio.to_thread(
            lambda: (
                get_client()
                .storage
                .from_(BUCKET)
                .create_signed_url(storage_path, ttl_seconds)
            )
        )
        # supabase-py v2 returns an object with .data (dict), v1 returns a plain dict
        if isinstance(response, dict):
            signed_url = response.get("signedURL") or response.get("signedUrl")
        elif hasattr(response, "data") and isinstance(response.data, dict):
            signed_url = (
                response.data.get("signedURL") or response.data.get("signedUrl")
            )
        else:
            signed_url = None

        if not signed_url:
            raise StorageFetchError(
                f"Failed to extract signed URL from storage response "
                f"(path prefix={storage_path[:40]!r}). Response: {response}"
            )
    except StorageFetchError:
        raise
    except Exception as exc:
        raise StorageFetchError(
            f"Signed URL generation failed for path prefix={storage_path[:40]!r}: {exc}"
        ) from exc

    # Step 2: HTTP GET content with retry on 5xx (m2 fix)
    # signed URL is used once per attempt and immediately discarded after use.
    last_exc: Exception | None = None
    for attempt in range(_FETCH_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_SECONDS) as http_client:
                http_resp = await http_client.get(signed_url)
                http_resp.raise_for_status()
                content = http_resp.text
            break  # success
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status >= 500 and attempt < _FETCH_MAX_RETRIES - 1:
                # m2: transient server error — retry
                import asyncio
                wait = 2 ** (attempt + 1)  # 2s, 4s
                logger.warning(
                    "fetch_file_content: HTTP %d on attempt %d/%d, retrying in %ds "
                    "(path prefix=%r)",
                    status, attempt + 1, _FETCH_MAX_RETRIES, wait,
                    storage_path[:40],
                )
                await asyncio.sleep(wait)
                last_exc = exc
                continue
            raise StorageFetchError(
                f"HTTP {status} fetching file "
                f"(path prefix={storage_path[:40]!r}): {exc}"
            ) from exc
        except Exception as exc:
            raise StorageFetchError(
                f"HTTP fetch failed for path prefix={storage_path[:40]!r}: {exc}"
            ) from exc
    else:
        raise StorageFetchError(
            f"fetch_file_content: exhausted {_FETCH_MAX_RETRIES} retries "
            f"(path prefix={storage_path[:40]!r})"
        ) from last_exc

    # m1 fix: log only char count and redacted path prefix — never the full path
    logger.info(
        "fetch_file_content: retrieved %d chars (path prefix=%r)",
        len(content),
        storage_path[:40],
    )
    return content


# ---------------------------------------------------------------------------
# Phase L0 (LangExtract Migration) — store_jsonl_content
# ---------------------------------------------------------------------------

async def store_jsonl_content(storage_path: str, jsonl_text: str) -> str:
    """
    Uploads a LangExtract JSONL string to the raw_files bucket.

    Called by langextract_run_a.py and langextract_run_b.py after each successful
    lx.extract() call to persist the annotated document for on-demand visualization.

    Storage path convention:
      Run A: {brand_slug}/{model_slug}/extractions/spec_{extraction_run_id}.jsonl
      Run B: {brand_slug}/{model_slug}/extractions/exp_{exp_run_id}.jsonl

    The HTML visualization is NEVER stored — generated on demand at serve time by
    GET /approval/visualization/{run_type}/{run_id} via lx.visualize(jsonl_path).
    See langextract_migration_v4.md Section 10.

    Args:
        storage_path: Target path within the raw_files bucket.
        jsonl_text:   JSONL content as a UTF-8 string (output of lx.io.save_annotated_documents).

    Returns:
        The storage path string (same as the input path).

    Raises:
        RuntimeError: On any upload failure — propagated to the orchestrator.
    """
    file_bytes = jsonl_text.encode("utf-8")
    return await upload_file(
        path=storage_path,
        file_bytes=file_bytes,
        content_type="application/jsonl",
    )
