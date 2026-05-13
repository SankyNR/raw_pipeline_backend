"""
staging_resolver.py — Section 3.2 of the Camera Features Ontology roadmap

auto_resolve_staging_at_commit() is called by commit_orchestrator.py at Step 2.5,
immediately after the commit_run_id audit row is created and before Step 3 (brand
resolution).

Responsibilities
────────────────
1. Fetch all rows from pipeline.lookup_value_staging WHERE
   url_registry_id = $1 AND status = 'pending_review'.
2. For each row branch on resolution_target_table:
     - NULL (default path)   → INSERT into target_lookup_table as a new canonical row.
     - 'mobile_specs.lookup_feature_aliases' → INSERT into the alias table.
     - Anything else         → log error, skip (defensive guard).
3. Hot-patch the in-memory LOOKUP_CACHE / ALIAS_CACHE so the rest of the commit
   picks up the new PK without a cache reload.
4. Back-patch final_json at the staging row's field_path:
     - If the current value at field_path is a list → APPEND the resolved PK.
     - Otherwise                                    → REPLACE with the resolved PK.
5. UPDATE the staging row: status='inserted_new', resolution='inserted_new',
   resolved_lookup_id=<new_pk or feature_id>.
6. Race condition (C7 scenario): if INSERT fails with UNIQUE violation (code 23505)
   fetch the existing PK and reuse it — two admins committing the same new value.

Returns
───────
{
  "inserted_count": int,    # how many staging rows were successfully resolved
  "patches": [...],         # back-patch descriptions for logging
  "errors": [...],          # per-row errors (do NOT abort commit — just log)
}

Design rules
────────────
- All DB calls are synchronous (supabase-py). The caller wraps this function
  in asyncio.to_thread() — no async inside this module.
- Errors in individual rows are caught and appended to errors[]; they do NOT
  abort the whole commit. Phase 9 validation will catch any remaining
  unresolved strings that were not patched.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.supabase_client import get_client
from app.services.normalizer import ALIAS_CACHE, LOOKUP_CACHE, clean_for_lookup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal path helpers (intentionally self-contained — no import from normalizer
# to keep the dependency graph clean; these are tiny and copy-safe).
# ---------------------------------------------------------------------------

def _get_at_path(data: Any, field_path: str) -> Any:
    """
    Navigate a dotted + bracketed path string and return the value.
    Returns None if any segment is missing.
    Examples:
        "camera_features"       → data["camera_features"]
        "charging.cable_type"   → data["charging"]["cable_type"]
    """
    if not field_path:
        return data
    current: Any = data
    for part in _split_path(field_path):
        if current is None:
            return None
        if isinstance(part, int):
            current = current[part] if isinstance(current, list) and 0 <= part < len(current) else None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _set_at_path(data: Any, field_path: str, value: Any) -> None:
    """
    Set the value at a dotted + bracketed field_path, creating intermediate
    dicts where needed. No-op if an intermediate list index is out of range.
    """
    parts = _split_path(field_path)
    current: Any = data
    for part in parts[:-1]:
        if isinstance(part, int):
            if isinstance(current, list) and 0 <= part < len(current):
                current = current[part]
            else:
                return
        elif isinstance(current, dict):
            if part not in current:
                current[part] = {}
            current = current[part]
        else:
            return
    last = parts[-1]
    if isinstance(last, int):
        if isinstance(current, list) and 0 <= last < len(current):
            current[last] = value
    elif isinstance(current, dict):
        current[last] = value


def _split_path(path: str) -> list[str | int]:
    """
    Tokenise a dotted + bracketed path into a list of keys / indices.
    "camera_features"       → ["camera_features"]
    "charging.cable_type"   → ["charging", "cable_type"]
    "camera_lenses[0].fov"  → ["camera_lenses", 0, "fov"]
    """
    import re
    tokens: list[str | int] = []
    for part in re.split(r"\.|(\[\d+\])", path):
        if not part:
            continue
        if part.startswith("[") and part.endswith("]"):
            tokens.append(int(part[1:-1]))
        else:
            tokens.append(part)
    return tokens


# ---------------------------------------------------------------------------
# Supabase error code helper
# ---------------------------------------------------------------------------

def _is_unique_violation(exc: Exception) -> bool:
    """
    Returns True if exc is a Supabase / PostgREST UNIQUE violation (code 23505).
    Supabase-py wraps errors in a PostgrestAPIError with a 'code' attribute.
    """
    code = getattr(exc, "code", None) or ""
    message = str(exc).lower()
    return code == "23505" or "unique" in message or "duplicate" in message


# ---------------------------------------------------------------------------
# Staging table helpers
# ---------------------------------------------------------------------------

def _fetch_pending_staging_rows(url_registry_id: int) -> list[dict]:
    """
    Fetch all pending staging rows for a given url_registry_id.
    """
    client = get_client()
    result = (
        client
        .schema("pipeline")
        .table("lookup_value_staging")
        .select("*")
        .eq("url_registry_id", url_registry_id)
        .eq("status", "pending_review")
        .execute()
    )
    return result.data or []


def _mark_staging_resolved(staging_id: int, resolved_pk: int) -> None:
    """
    Mark a staging row as resolved after a successful INSERT.
    """
    client = get_client()
    (
        client
        .schema("pipeline")
        .table("lookup_value_staging")
        .update({
            "status":              "inserted_new",
            "resolution":          "inserted_new",
            "resolved_lookup_id":  resolved_pk,
        })
        .eq("staging_id", staging_id)
        .execute()
    )


# ---------------------------------------------------------------------------
# Path 1 — Insert into target_lookup_table (default / canonical path)
# ---------------------------------------------------------------------------

def _insert_canonical(
    target_lookup_table: str,
    extracted_value: str,
    new_row_data: dict | None,
) -> int:
    """
    INSERT a new row into the canonical lookup table identified by target_lookup_table
    (format: "schema.table.column").

    If new_row_data is provided (camera features: full metadata payload), it is used
    as the insertion payload. Otherwise, the extracted_value is inserted as the sole
    value column (column name = third dot-segment of target_lookup_table).

    Returns the PK of the newly created row.

    Raises on error (including UNIQUE violation — caller handles C7 scenario).
    """
    parts = target_lookup_table.rsplit(".", 2)
    if len(parts) != 3:
        raise ValueError(
            f"_insert_canonical: unexpected target_lookup_table format={target_lookup_table!r}. "
            "Expected 'schema.table.column'."
        )
    schema, table, column = parts

    client = get_client()

    if new_row_data:
        payload = dict(new_row_data)
    else:
        payload = {column: extracted_value}

    result = (
        client
        .schema(schema)
        .table(table)
        .insert(payload)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise RuntimeError(
            f"_insert_canonical: INSERT into {schema}.{table} returned no row. payload={payload!r}"
        )

    # Detect PK column: the _id-suffixed column that is not the value column
    pk_col = next(
        (k for k in rows[0] if k.endswith("_id") and k != column),
        None,
    )
    if pk_col is None:
        raise RuntimeError(
            f"_insert_canonical: could not detect PK column in result row={rows[0]!r}"
        )
    return int(rows[0][pk_col])


def _fetch_existing_canonical_pk(target_lookup_table: str, extracted_value: str) -> int | None:
    """
    C7 race handler — SELECT the existing PK from target_lookup_table using extracted_value.
    Returns None if no row found (should not happen after a UNIQUE violation).
    """
    parts = target_lookup_table.rsplit(".", 2)
    if len(parts) != 3:
        return None
    schema, table, column = parts

    client = get_client()
    result = (
        client
        .schema(schema)
        .table(table)
        .select("*")
        .eq(column, extracted_value)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    if not rows:
        return None
    pk_col = next((k for k in rows[0] if k.endswith("_id") and k != column), None)
    return int(rows[0][pk_col]) if pk_col else None


# ---------------------------------------------------------------------------
# Path 2 — Insert into lookup_feature_aliases
# ---------------------------------------------------------------------------

def _insert_alias(new_row_data: dict) -> tuple[int, int]:
    """
    INSERT a row into mobile_specs.lookup_feature_aliases.

    new_row_data must contain:
        feature_id  (int)  — the canonical camera feature this alias maps to
        brand_alias (str)  — the brand-specific marketing name
        brand_id    (int)  — the brand PK

    Returns (alias_id, feature_id). The caller uses feature_id (not alias_id)
    for back-patching final_json, because phone_camera_features stores feature_id.

    Raises on error (including UNIQUE violation — caller handles C7 scenario).
    """
    client = get_client()
    payload = {
        "feature_id":  new_row_data["feature_id"],
        "brand_alias": new_row_data["brand_alias"],
        "brand_id":    new_row_data["brand_id"],
    }
    result = (
        client
        .schema("mobile_specs")
        .table("lookup_feature_aliases")
        .insert(payload)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise RuntimeError(
            f"_insert_alias: INSERT into lookup_feature_aliases returned no row. payload={payload!r}"
        )
    alias_id = int(rows[0]["alias_id"])
    feature_id = int(new_row_data["feature_id"])
    return alias_id, feature_id


def _fetch_existing_alias_pk(brand_alias: str, brand_id: int) -> tuple[int | None, int | None]:
    """
    C7 race handler — SELECT the existing (alias_id, feature_id) from
    lookup_feature_aliases using the (brand_alias, brand_id) unique key.
    """
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("lookup_feature_aliases")
        .select("alias_id, feature_id")
        .eq("brand_alias", brand_alias)
        .eq("brand_id", brand_id)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    if not rows:
        return None, None
    return int(rows[0]["alias_id"]), int(rows[0]["feature_id"])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _auto_resolve_staging_at_commit_sync(
    url_registry_id: int,
    final_json: dict,
) -> dict:
    """
    Synchronous core of auto_resolve_staging_at_commit. Called inside
    asyncio.to_thread() by the orchestrator.
    """
    rows = _fetch_pending_staging_rows(url_registry_id)
    if not rows:
        return {"inserted_count": 0, "patches": [], "errors": []}

    inserted_count = 0
    patches: list[dict] = []
    errors: list[dict] = []

    for row in rows:
        staging_id:              int       = row["staging_id"]
        extracted_value:         str       = str(row.get("extracted_value") or "")
        target_lookup_table:     str | None = row.get("target_lookup_table")
        resolution_target_table: str | None = row.get("resolution_target_table")
        field_path:              str | None = row.get("field_path") or ""
        new_row_data:            dict | None = row.get("new_row_data")

        # The effective insertion target
        effective_target = resolution_target_table or target_lookup_table

        try:
            # ── Path 2: alias insert ─────────────────────────────────────────
            if resolution_target_table == "mobile_specs.lookup_feature_aliases":
                if not new_row_data or "feature_id" not in new_row_data or "brand_alias" not in new_row_data:
                    raise ValueError(
                        f"staging_id={staging_id}: alias path requires new_row_data with "
                        "feature_id, brand_alias, brand_id."
                    )

                try:
                    alias_id, feature_id = _insert_alias(new_row_data)
                    logger.info(
                        "staging_resolver: inserted alias_id=%d brand_alias=%r → feature_id=%d",
                        alias_id, new_row_data["brand_alias"], feature_id,
                    )
                except Exception as exc:
                    if _is_unique_violation(exc):
                        alias_id, feature_id = _fetch_existing_alias_pk(
                            new_row_data["brand_alias"],
                            new_row_data["brand_id"],
                        )
                        if feature_id is None:
                            raise RuntimeError(
                                f"C7 race: could not find existing alias for "
                                f"brand_alias={new_row_data['brand_alias']!r} brand_id={new_row_data['brand_id']}"
                            ) from exc
                        logger.info(
                            "staging_resolver: race detected for alias=%r brand_id=%d "
                            "— reusing existing feature_id=%d",
                            new_row_data["brand_alias"], new_row_data["brand_id"], feature_id,
                        )
                    else:
                        raise

                # Hot-patch ALIAS_CACHE
                cleaned_alias = clean_for_lookup(str(new_row_data["brand_alias"]))
                ALIAS_CACHE.setdefault(cleaned_alias, {})[new_row_data.get("brand_id")] = feature_id

                # Back-patch final_json with the canonical feature_id (NOT alias_id)
                resolved_pk = feature_id

            # ── Path 1: canonical insert ─────────────────────────────────────
            elif effective_target:
                try:
                    new_pk = _insert_canonical(effective_target, extracted_value, new_row_data)
                    logger.info(
                        "staging_resolver: inserted into %s extracted_value=%r → pk=%d",
                        effective_target, extracted_value, new_pk,
                    )
                except Exception as exc:
                    if _is_unique_violation(exc):
                        new_pk = _fetch_existing_canonical_pk(effective_target, extracted_value)
                        if new_pk is None:
                            raise RuntimeError(
                                f"C7 race: UNIQUE violation on {effective_target} for "
                                f"value={extracted_value!r} but could not SELECT existing row."
                            ) from exc
                        logger.info(
                            "staging_resolver: race detected for value=%r in %s "
                            "— reusing existing pk=%d",
                            extracted_value, effective_target, new_pk,
                        )
                    else:
                        raise

                # Hot-patch LOOKUP_CACHE
                parts = effective_target.rsplit(".", 2)
                if len(parts) == 3:
                    LOOKUP_CACHE.setdefault(effective_target, {})[
                        clean_for_lookup(extracted_value)
                    ] = new_pk

                resolved_pk = new_pk

            else:
                # Neither path has a valid target — skip with warning
                logger.warning(
                    "staging_resolver: staging_id=%d has no target table — skipping.",
                    staging_id,
                )
                errors.append({
                    "staging_id": staging_id,
                    "error":      "no effective target table",
                })
                continue

            # ── Back-patch final_json ────────────────────────────────────────
            if field_path:
                current_val = _get_at_path(final_json, field_path)
                if isinstance(current_val, list):
                    # Array FK path — APPEND the resolved PK
                    if resolved_pk not in current_val:
                        current_val.append(resolved_pk)
                    # _get_at_path returns the live list object, so mutation is in-place
                else:
                    # Scalar FK path — REPLACE
                    _set_at_path(final_json, field_path, resolved_pk)

                patches.append({
                    "staging_id":  staging_id,
                    "field_path":  field_path,
                    "resolved_pk": resolved_pk,
                    "mode":        "append" if isinstance(current_val, list) else "replace",
                })

            # ── Mark staging row resolved ────────────────────────────────────
            _mark_staging_resolved(staging_id, resolved_pk)
            inserted_count += 1

        except Exception as exc:
            logger.error(
                "staging_resolver: FAILED to resolve staging_id=%d extracted_value=%r: %s",
                staging_id, extracted_value, exc,
            )
            errors.append({
                "staging_id":     staging_id,
                "extracted_value": extracted_value,
                "error":          str(exc),
            })
            # Do NOT re-raise — collect error and continue with remaining rows

    return {
        "inserted_count": inserted_count,
        "patches":        patches,
        "errors":         errors,
    }


async def auto_resolve_staging_at_commit(
    url_registry_id: int,
    final_json: dict,
) -> dict:
    """
    Async wrapper. Called from commit_orchestrator.py Step 2.5.

    Wraps _auto_resolve_staging_at_commit_sync in asyncio.to_thread() so the
    synchronous supabase-py calls do not block the event loop.

    Returns:
        {
          "inserted_count": int,
          "patches": [...],
          "errors": [...],
        }
    """
    import asyncio
    return await asyncio.to_thread(
        _auto_resolve_staging_at_commit_sync,
        url_registry_id,
        final_json,
    )
