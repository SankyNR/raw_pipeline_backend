"""
Admin Lookup API Router.

Phase 0  (Task 0.3):  GET /admin/health/db
Phase 1  (Tasks 1.1–1.3): GET /admin/brands, /admin/models, /admin/sites
Phase 11 (Lookup Wizard): GET /admin/lookup/schema, /admin/lookup/values
"""

import logging
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Query

from app.core.supabase_client import get_client
from app.config.field_mapping import SCALAR_FK_MAP, ARRAY_FK_MAP

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# ---------------------------------------------------------------------------
# Lookup table whitelist — only tables referenced in field_mapping are allowed.
# Built once at import time from SCALAR_FK_MAP + ARRAY_FK_MAP values.
# ---------------------------------------------------------------------------

def _build_allowed_tables() -> set[str]:
    """Extract the set of 'schema.table' strings from FK maps."""
    tables: set[str] = set()
    for table_path in list(SCALAR_FK_MAP.values()) + list(ARRAY_FK_MAP.values()):
        parts = table_path.rsplit(".", 1)
        if len(parts) == 2:
            tables.add(parts[0])          # "mobile_specs.lookup_panel_types"
    return tables

_ALLOWED_LOOKUP_TABLES: set[str] = _build_allowed_tables()

# Postgres type → wizard field type mapping
_PG_TYPE_MAP: dict[str, str] = {
    "integer":                   "integer",
    "bigint":                    "integer",
    "smallint":                  "integer",
    "boolean":                   "boolean",
    "text":                      "text",
    "character varying":         "text",
    "timestamp with time zone":  "timestamp",
    "timestamp without time zone": "timestamp",
    "USER-DEFINED":              "text",       # enums
    "ARRAY":                     "text",
}


def _wizard_type(pg_type: str, col_name: str) -> str:
    """
    Map a Postgres data_type string to the wizard field type.
    Serial PKs are detected by convention (*_id columns that are PKs).
    """
    return _PG_TYPE_MAP.get(pg_type, "text")


# ---------------------------------------------------------------------------
# Task 0.3 — DB Health Check
# ---------------------------------------------------------------------------

@router.get("/health/db")
async def health_db():
    """
    Confirms Supabase Postgres connectivity by running a minimal query
    against the pipeline schema.

    Returns: { "status": "ok" } on success.
    Raises HTTP 500 on any DB error.
    """
    try:
        get_client() \
            .schema("pipeline") \
            .table("url_registry") \
            .select("url_id") \
            .limit(1) \
            .execute()
        return {"status": "ok"}
    except Exception as e:
        logger.error("DB health check failed: %s", e)
        raise HTTPException(status_code=500, detail=f"DB connection failed: {e}")


# ---------------------------------------------------------------------------
# Task 1.1 — GET /admin/brands
# ---------------------------------------------------------------------------

@router.get("/brands")
async def get_brands():
    """
    Returns all distinct brand values from pipeline.url_registry,
    sorted alphabetically.

    Response: { "brands": ["Apple", "CMF", "Google", ...] }
    """
    try:
        result = (
            get_client()
            .schema("pipeline")
            .table("url_registry")
            .select("brand")
            .execute()
        )
    except Exception as e:
        logger.error("get_brands DB error: %s", e)
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    brands = sorted({row["brand"] for row in result.data})
    return {"brands": brands}


# ---------------------------------------------------------------------------
# Task 1.2 — GET /admin/models?brand=...
# ---------------------------------------------------------------------------

# Status sets used for colour logic
_DONE_STATUSES = {"scraped_raw", "stored_mainDB"}
_NOT_DONE_STATUSES = {"not_scraped", "currently_scraping"}


def _compute_model_color(statuses: list[str]) -> str:
    """
    Determines the aggregate colour for a model based on the statuses of
    all its source sites.

    | Condition                                                  | colour   |
    |------------------------------------------------------------|----------|
    | All sites are `not_scraped`                                | "red"    |
    | Mix — at least one not-done, at least one done             | "orange" |
    | All sites are `scraped_raw`                                | "yellow" |
    | All sites are `stored_mainDB`                              | "green"  |

    `currently_scraping` counts as "not done" for the orange condition.
    """
    status_set = set(statuses)

    has_done = bool(status_set & _DONE_STATUSES)
    has_not_done = bool(status_set & _NOT_DONE_STATUSES)

    if has_done and has_not_done:
        return "orange"

    if not has_done:
        # Everything is in the not-done bucket
        return "red"

    # All are done — now distinguish yellow vs green
    if all(s == "stored_mainDB" for s in statuses):
        return "green"
    return "yellow"


@router.get("/models")
async def get_models(brand: str = Query(..., description="Brand name to filter by")):
    """
    Returns all models for the given brand with an aggregate colour
    based on the scrape statuses of their source sites.

    Response:
        {
          "models": [
            { "model_name": "Galaxy S25", "color": "red" },
            ...
          ]
        }
    """
    try:
        result = (
            get_client()
            .schema("pipeline")
            .table("url_registry")
            .select("url_id, model_name, status")    # add url_id
            .eq("brand", brand)
            .execute()
        )
    except Exception as e:
        logger.error("get_models DB error (brand=%s): %s", brand, e)
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    if not result.data:
        return {"models": []}

    # Group statuses by model_name, preserving insertion order for sorted output
    model_statuses: dict[str, list[str]] = defaultdict(list)
    model_url_ids:  dict[str, int]       = {}
    for row in result.data:
        model_statuses[row["model_name"]].append(row["status"])
        if row["model_name"] not in model_url_ids:
            model_url_ids[row["model_name"]] = row["url_id"]

    models = [
        {
            "model_name":       model_name,
            "url_registry_id":  model_url_ids[model_name],
            "color":            _compute_model_color(statuses),
        }
        for model_name, statuses in sorted(model_statuses.items())
    ]
    return {"models": models}


# ---------------------------------------------------------------------------
# Task 1.3 — GET /admin/sites?brand=...&model=...
# ---------------------------------------------------------------------------

def _site_badge_color(status: str) -> str:
    """
    Maps a url_registry status value to a badge colour for the site level.

    | status             | badge_color |
    |--------------------|-------------|
    | not_scraped        | "red"       |
    | currently_scraping | "blue"      |
    | scraped_raw        | "green"     |
    | stored_mainDB      | "green"     |
    """
    if status == "currently_scraping":
        return "blue"
    if status in _DONE_STATUSES:
        return "green"
    return "red"  # not_scraped (and any unknown value)


@router.get("/sites")
async def get_sites(
    brand: str = Query(..., description="Brand name"),
    model: str = Query(..., description="Model name"),
):
    """
    Returns all source sites for the given brand + model, each with its
    current scrape status and badge colour.

    Joins url_registry → lookup_source_registry to get display_name.

    Response:
        {
          "sites": [
            {
              "site_name": "gsmarena",
              "display_name": "GSMArena",
              "status": "scraped_raw",
              "badge_color": "green"
            },
            ...
          ]
        }
    """
    try:
        result = (
            get_client()
            .schema("pipeline")
            .table("url_registry")
            .select("url_id, site_name, status, lookup_source_registry(display_name)")
            .eq("brand", brand)
            .eq("model_name", model)
            .execute()
        )
    except Exception as e:
        logger.error(
            "get_sites DB error (brand=%s, model=%s): %s", brand, model, e
        )
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    if not result.data:
        return {"sites": []}

    sites = [
        {
            "url_id":       row["url_id"],
            "site_name":    row["site_name"],
            "display_name": (
                row["lookup_source_registry"]["display_name"]
                if row.get("lookup_source_registry")
                else row["site_name"]
            ),
            "status":       row["status"],
            "badge_color":  _site_badge_color(row["status"]),
        }
        for row in result.data
    ]
    return {"sites": sites}


# ---------------------------------------------------------------------------
# Phase 11 — GET /admin/lookup/schema
# ---------------------------------------------------------------------------

def _validate_lookup_table(table: str) -> tuple[str, str]:
    """
    Validates the table param (e.g. 'mobile_specs.lookup_camera_features').
    Returns (schema, table_name). Raises 400 if not in the whitelist.
    """
    if table not in _ALLOWED_LOOKUP_TABLES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Table '{table}' is not a recognised lookup table. "
                f"Allowed tables: {sorted(_ALLOWED_LOOKUP_TABLES)}"
            ),
        )
    parts = table.split(".", 1)
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail=f"Invalid table format: '{table}'. Expected 'schema.table'.")
    return parts[0], parts[1]


@router.get("/lookup/schema")
async def get_lookup_schema(
    table: str = Query(..., description="Fully qualified table name, e.g. mobile_specs.lookup_camera_features"),
):
    """
    Returns column definitions for a lookup table so the wizard can render
    the right input type per column.

    Requires Migration 54 (RPC function `get_lookup_table_schema`).

    Response shape:
        {
          "table": "mobile_specs.lookup_camera_features",
          "columns": [
            { "name": "feature_name", "type": "text", "nullable": false,
              "is_pk": false, "is_fk": false },
            { "name": "subcategory_id", "type": "integer", "nullable": false,
              "is_pk": false, "is_fk": true,
              "fk_table": "mobile_specs.lookup_camera_subcategories",
              "fk_display_column": null },
            ...
          ]
        }
    """
    schema_name, table_name = _validate_lookup_table(table)

    try:
        result = (
            get_client()
            .rpc(
                "get_lookup_table_schema",
                {"p_schema": schema_name, "p_table": table_name},
            )
            .execute()
        )
    except Exception as e:
        logger.error("get_lookup_schema RPC error (table=%s): %s", table, e)
        raise HTTPException(
            status_code=500,
            detail=(
                f"Schema introspection failed: {e}. "
                f"Has Migration 54 (get_lookup_table_schema RPC) been applied?"
            ),
        )

    rows = result.data or []
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No columns found for table '{table}'. Does it exist in the database?",
        )

    columns = []
    for row in rows:
        pg_type = row.get("data_type", "text")
        col_name = row["column_name"]
        is_pk = row.get("is_pk", False)

        # Serial PKs get a special wizard type.
        # information_schema reports 'integer' for serial and 'bigint' for bigserial.
        wizard_type = "serial" if is_pk and pg_type in ("integer", "bigint") else _wizard_type(pg_type, col_name)

        col_def: dict = {
            "name":     col_name,
            "type":     wizard_type,
            "nullable": row.get("is_nullable", "YES") == "YES",
            "is_pk":    is_pk,
            "is_fk":    row.get("is_fk", False),
        }
        if row.get("is_fk"):
            fk_schema = row.get("fk_schema", "")
            fk_tbl    = row.get("fk_table", "")
            col_def["fk_table"] = f"{fk_schema}.{fk_tbl}" if fk_schema else fk_tbl
            col_def["fk_display_column"] = row.get("fk_column")
        columns.append(col_def)

    return {"table": table, "columns": columns}


# ---------------------------------------------------------------------------
# Phase 11 — GET /admin/lookup/values
# ---------------------------------------------------------------------------

@router.get("/lookup/values")
async def get_lookup_values(
    table:  str = Query(..., description="Fully qualified table name, e.g. mobile_specs.lookup_camera_features"),
    column: str = Query(..., description="Column to return values for, e.g. feature_name"),
):
    """
    Returns all existing canonical values for one column in a lookup table,
    plus any pending `inserted_new` staging rows for the same table so that
    values staged earlier in the same review session appear immediately.

    Response shape:
        {
          "table":   "mobile_specs.lookup_camera_features",
          "column":  "feature_name",
          "values": [
            { "id": 1, "value": "OIS",           "source": "committed" },
            { "id": null, "value": "Macro mode", "source": "pending_staging" }
          ],
          "includes_pending_staging": true
        }
    """
    schema_name, table_name = _validate_lookup_table(table)

    # --- 1. Query the committed lookup table rows ---
    try:
        committed_result = (
            get_client()
            .schema(schema_name)
            .table(table_name)
            .select("*")
            .execute()
        )
    except Exception as e:
        logger.error("get_lookup_values DB error (table=%s): %s", table, e)
        raise HTTPException(status_code=500, detail=f"DB error querying {table}: {e}")

    committed_rows = committed_result.data or []

    # Resolve the true PK column via the schema RPC — do NOT guess by name convention.
    # Guessing (first key ending in _id) silently picks the wrong column for any
    # multi-FK table where a FK column appears before the PK in definition order.
    pk_col: str | None = None
    try:
        schema_result = (
            get_client()
            .rpc(
                "get_lookup_table_schema",
                {"p_schema": schema_name, "p_table": table_name},
            )
            .execute()
        )
        pk_col = next(
            (r["column_name"] for r in (schema_result.data or []) if r.get("is_pk")),
            None,
        )
    except Exception as e:
        logger.warning(
            "get_lookup_values: could not resolve PK column for %s via RPC: %s "
            "— row IDs will be null.", table, e,
        )

    values = []
    for row in committed_rows:
        val = row.get(column)
        if val is None:
            continue
        values.append({
            "id":     row.get(pk_col) if pk_col else None,
            "value":  val,
            "source": "committed",
        })

    # --- 2. Union with pending staging rows for this table ---
    includes_pending = False
    try:
        staging_result = (
            get_client()
            .schema("pipeline")
            .table("lookup_value_staging")
            .select("staging_id, extracted_value")
            .eq("target_lookup_table", table)
            .eq("status", "inserted_new")
            .execute()
        )
        staging_rows = staging_result.data or []
        includes_pending = True  # query succeeded — we can report coverage

        # Deduplicate: skip staging values already present in committed set
        committed_vals = {v["value"] for v in values}
        for srow in staging_rows:
            extracted = srow.get("extracted_value")
            if extracted and extracted not in committed_vals:
                values.append({
                    "id":     None,
                    "value":  extracted,
                    "source": "pending_staging",
                })
                committed_vals.add(extracted)

    except Exception as e:
        logger.warning(
            "get_lookup_values: staging union failed (table=%s): %s — "
            "returning committed values only.", table, e,
        )
        includes_pending = False

    return {
        "table":                    table,
        "column":                   column,
        "values":                   values,
        "includes_pending_staging": includes_pending,
    }
