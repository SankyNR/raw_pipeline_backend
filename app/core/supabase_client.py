"""
Task 0.3 — Supabase Client Initialization

Provides a lazily-initialized supabase-py client using the service role key.
The client is created on first call to get_client() — not at import time.
This keeps module imports clean even when .env is not yet present.

All DB operations must use .schema("pipeline") before .table(...).
Use get_client() everywhere instead of a bare module-level instance.
"""

from __future__ import annotations

from supabase import create_client, Client
from app.core.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

_client: Client | None = None


def get_client() -> Client:
    """
    Returns the singleton Supabase client, creating it on first call.
    Raises RuntimeError if SUPABASE_URL or key are not set.
    """
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            raise RuntimeError(
                "Supabase client requested but SUPABASE_URL / "
                "SUPABASE_SERVICE_ROLE_KEY are not set. Check your .env file."
            )
        _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _client