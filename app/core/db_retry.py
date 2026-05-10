"""
app/core/db_retry.py
Thin exponential-backoff wrapper for synchronous Supabase calls executed via
asyncio.to_thread().

Mirrors the retry logic already applied to call_gemini_grounded / call_gemini_json.
Without this, a transient Supabase connection-pool exhaustion or cold-start error
produces an immediate hard failure for every asyncio.to_thread(db_call, ...) site
in the codebase.

Usage
-----
    from app.core.db_retry import db_call_with_retry

    result = await db_call_with_retry(fetch_spec_extraction_output, output_id)

    # With keyword arguments:
    result = await db_call_with_retry(insert_normalisation_run, payload, retries=5)

Notes
-----
- Default: 3 attempts, initial sleep 0.5 s, doubling each attempt (0.5 → 1.0 → 2.0).
- Only retries on Exception subclasses — does NOT catch BaseException (KeyboardInterrupt, etc.).
- The wrapped function must be a regular (sync) callable — it is dispatched to a thread
  pool internally via asyncio.to_thread().
- NOT a replacement for idempotency: callers are responsible for ensuring that the
  underlying DB call is safe to retry (most SELECTs and ON CONFLICT DO NOTHING INSERTs are).
  Use with caution on plain INSERT calls that lack ON CONFLICT guards.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Defaults
_DEFAULT_RETRIES: int   = 3
_DEFAULT_BASE_SLEEP: float = 0.5   # seconds; doubles each attempt


async def db_call_with_retry(
    fn: Callable[..., T],
    /,
    *args: Any,
    retries: int = _DEFAULT_RETRIES,
    base_sleep: float = _DEFAULT_BASE_SLEEP,
    **kwargs: Any,
) -> T:
    """
    Runs ``fn(*args, **kwargs)`` in a thread pool with exponential-backoff retry.

    Parameters
    ----------
    fn:         Synchronous callable to execute.
    *args:      Positional arguments forwarded to fn.
    retries:    Total number of attempts (default 3).
    base_sleep: Initial sleep in seconds before the second attempt (default 0.5 s).
    **kwargs:   Keyword arguments forwarded to fn.

    Returns
    -------
    Return value of fn on success.

    Raises
    ------
    The last exception if all attempts are exhausted.
    """
    last_exc: Exception | None = None
    sleep = base_sleep

    for attempt in range(1, retries + 1):
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                logger.warning(
                    "db_call_with_retry: attempt %d/%d failed for %s — retrying in %.1fs. Error: %s",
                    attempt, retries, getattr(fn, "__name__", repr(fn)), sleep, exc,
                )
                await asyncio.sleep(sleep)
                sleep *= 2
            else:
                logger.error(
                    "db_call_with_retry: all %d attempts exhausted for %s. Last error: %s",
                    retries, getattr(fn, "__name__", repr(fn)), exc,
                )

    raise last_exc  # type: ignore[misc]
