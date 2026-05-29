"""
app/repositories/embedding_inputs_repository.py
================================================
Phase EM2.2 — Embedding Pipeline Input Reads

Thin read-only functions that supply the document assembler (EM3) with its
three input data classes:
  1. Run B experiences  (pipeline.phone_experiences + db_commit_runs join)
  2. Run C inferences   (pipeline.inference_entries WHERE emitted_to_embedding)
  3. Spec summary       (mobile_specs.* for the [SPECS] tail)

CRITICAL JOIN NOTE — phone_experiences has NO model_id column.
The link from model_id → experiences goes through db_commit_runs:
    phone_experiences.url_registry_id = db_commit_runs.url_registry_id
    WHERE db_commit_runs.model_id = $model_id
      AND db_commit_runs.status   = 'completed'

inference_entries has model_id directly — no join needed.

All functions are SYNCHRONOUS. Async callers wrap with asyncio.to_thread().
"""

import logging

from app.core.supabase_client import get_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EM2.2 — Run B experiences for embedding
# ---------------------------------------------------------------------------

def fetch_run_b_for_embedding(model_id: int) -> list[dict]:
    """
    Fetch Run B phone_experiences for a phone, filtered for embedding.

    Join path (phone_experiences has no model_id column):
        pipeline.phone_experiences pe
        JOIN pipeline.db_commit_runs dcr
             ON  dcr.url_registry_id = pe.url_registry_id
             AND dcr.model_id        = model_id
             AND dcr.status          = 'completed'
        JOIN pipeline.lookup_experience_categories lec
             ON  lec.category_id = pe.category_id
        WHERE pe.is_suppressed  = FALSE
          AND pe.is_superseded  = FALSE
          AND pe.confidence    >= 0.75
        ORDER BY lec.category_name, pe.confidence DESC

    The Supabase PostgREST client cannot express this three-table join via
    the fluent API (no direct foreign-key from phone_experiences → model_id).
    We resolve it in two steps:
      Step 1: resolve model_id → url_registry_ids via db_commit_runs.
      Step 2: fetch experiences for those url_registry_ids with the filters.

    Returns:
        List of dicts with keys:
          experience_id, experience_text, sentiment, confidence,
          category_name, source_transcript_count, url_registry_id
        Ordered by category_name ASC, confidence DESC.
        Empty list if no qualifying experiences exist.
    """
    client = get_client()

    # ── Step 1: resolve url_registry_ids for this model_id ────────────────
    commit_result = (
        client
        .schema("pipeline")
        .table("db_commit_runs")
        .select("url_registry_id")
        .eq("model_id", model_id)
        .eq("status", "completed")
        .execute()
    )
    commit_rows = commit_result.data or []
    if not commit_rows:
        logger.debug(
            "fetch_run_b_for_embedding: no completed db_commit_runs for model_id=%d",
            model_id,
        )
        return []

    url_registry_ids = list({r["url_registry_id"] for r in commit_rows})

    # ── Step 2: fetch experiences with filters ─────────────────────────────
    exp_result = (
        client
        .schema("pipeline")
        .table("phone_experiences")
        .select(
            "experience_id, experience_text, sentiment, confidence, "
            "source_transcript_count, url_registry_id, "
            "lookup_experience_categories(category_name)"
        )
        .in_("url_registry_id", url_registry_ids)
        .eq("is_suppressed", False)
        .eq("is_superseded", False)
        .gte("confidence", 0.75)
        .order("confidence", desc=True)
        .execute()
    )
    rows = exp_result.data or []

    # Flatten the nested category join
    for row in rows:
        cat = row.pop("lookup_experience_categories", None) or {}
        row["category_name"] = cat.get("category_name", "Overall")

    # Sort: category_name ASC, then confidence DESC (Python sort — stable)
    rows.sort(key=lambda r: (r.get("category_name", ""), -float(r.get("confidence", 0))))

    logger.debug(
        "fetch_run_b_for_embedding: model_id=%d -> %d experiences across %d url_registry_ids",
        model_id, len(rows), len(url_registry_ids),
    )
    return rows


# ---------------------------------------------------------------------------
# EM2.2 — Run C inference entries for embedding
# ---------------------------------------------------------------------------

def fetch_run_c_for_embedding(model_id: int) -> list[dict]:
    """
    Fetch Run C inference_entries that are cleared for embedding.

    Query:
        SELECT id, inference_text, sentiment, rule_key,
               defers_to_runb_category, conflict_flag
        FROM   pipeline.inference_entries
        WHERE  model_id             = model_id
          AND  emitted_to_embedding = TRUE
        ORDER BY rule_key

    The gating (emitted_to_embedding=TRUE / FALSE) was applied by run_c_engine.
    The embedding pipeline reads only the result — never re-applies gating.

    Returns:
        List of dicts with keys:
          id, inference_text, sentiment, rule_key,
          defers_to_runb_category, conflict_flag
        Ordered by rule_key ASC.
        Empty list if no qualifying inference entries exist.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("inference_entries")
        .select(
            "id, inference_text, sentiment, rule_key, "
            "defers_to_runb_category, conflict_flag"
        )
        .eq("model_id", model_id)
        .eq("emitted_to_embedding", True)
        .order("rule_key")
        .execute()
    )
    rows = result.data or []
    logger.debug(
        "fetch_run_c_for_embedding: model_id=%d -> %d emitted inference entries",
        model_id, len(rows),
    )
    return rows


# ---------------------------------------------------------------------------
# EM2.2 — Spec summary inputs for [SPECS] tail
# ---------------------------------------------------------------------------

def fetch_spec_summary_inputs(model_id: int) -> dict:
    """
    Fetch the columns needed to render the [SPECS] tail of the embedding document.

    Reads from mobile_specs.* tables. All joins are via model_id.
    Returns a flat dict with well-known keys (missing values are None).

    Keys returned:
        brand_name              str | None
        model_name              str | None
        chipset_name            str | None
        ram_gb_options          list[int] — sorted unique RAM capacities
        storage_gb_options      list[int] — sorted unique storage capacities
        battery_capacity        int | None  (mAh)
        charging_power          int | None  (W)
        primary_camera_mp       float | None
        primary_camera_aperture float | None
        panel_class             str | None  — e.g. "AMOLED", "LCD"
        display_size_inch       float | None
        refresh_rate            int | None  (Hz)
        has_5g                  bool
    """
    client = get_client()

    # ── Phones (brand_id + chipset_id — no cross-schema embed via PostgREST) ─
    # public.brands and mobile_specs.phones are in different schemas;
    # PostgREST cannot resolve the FK join across schemas. Resolve separately.
    phone_result = (
        client
        .schema("mobile_specs")
        .table("phones")
        .select("model_name, brand_id, chipset_id")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    )
    phone_rows = phone_result.data or []
    phone_row  = phone_rows[0] if phone_rows else {}
    model_name = phone_row.get("model_name")
    brand_id   = phone_row.get("brand_id")
    chipset_id = phone_row.get("chipset_id")

    # ── Brand name (public schema) ─────────────────────────────────────────
    brand_name: str | None = None
    if brand_id is not None:
        brand_result = (
            client
            .schema("public")
            .table("brands")
            .select("brand_name")
            .eq("brand_id", brand_id)
            .limit(1)
            .execute()
        )
        brand_rows = brand_result.data or []
        brand_name = brand_rows[0].get("brand_name") if brand_rows else None

    # ── Chipset name (same schema — but avoid embed, use plain FK lookup) ──
    chipset_name: str | None = None
    if chipset_id is not None:
        chipset_result = (
            client
            .schema("mobile_specs")
            .table("chipsets")
            .select("chipset_name")
            .eq("chipset_id", chipset_id)
            .limit(1)
            .execute()
        )
        chipset_rows = chipset_result.data or []
        chipset_name = chipset_rows[0].get("chipset_name") if chipset_rows else None



    # ── Variants (RAM + storage options) ──────────────────────────────────
    variant_result = (
        client
        .schema("mobile_specs")
        .table("variant")
        .select("ram_capacity, storage_capacity")
        .eq("model_id", model_id)
        .execute()
    )
    variant_rows = variant_result.data or []
    ram_gb_options     = sorted({r["ram_capacity"] for r in variant_rows if r.get("ram_capacity")})
    storage_gb_options = sorted({r["storage_capacity"] for r in variant_rows if r.get("storage_capacity")})

    # ── Charging specs ─────────────────────────────────────────────────────
    charging_result = (
        client
        .schema("mobile_specs")
        .table("charging_specs")
        .select("battery_capacity, charging_power")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    )
    charging_rows    = charging_result.data or []
    charging_row     = charging_rows[0] if charging_rows else {}
    battery_capacity = charging_row.get("battery_capacity")
    charging_power   = charging_row.get("charging_power")

    # ── Primary (main rear) camera lens ────────────────────────────────────
    # lens_type_id for main/primary is typically the smallest (first inserted).
    # We pick the lens with the highest megapixels for the main camera summary.
    lens_result = (
        client
        .schema("mobile_specs")
        .table("camera_lens_specs")
        .select("megapixels, aperture")
        .eq("model_id", model_id)
        .order("megapixels", desc=True)
        .limit(1)
        .execute()
    )
    lens_rows = lens_result.data or []
    lens_row  = lens_rows[0] if lens_rows else {}
    primary_camera_mp       = lens_row.get("megapixels")
    primary_camera_aperture = lens_row.get("aperture")

    # ── Main display ───────────────────────────────────────────────────────
    display_result = (
        client
        .schema("mobile_specs")
        .table("phone_displays")
        .select("size_inch, refresh_rate, panel_type_id, display_type")
        .eq("model_id", model_id)
        .eq("display_type", "main")
        .limit(1)
        .execute()
    )
    display_rows     = display_result.data or []
    display_row      = display_rows[0] if display_rows else {}
    display_size_inch = display_row.get("size_inch")
    refresh_rate      = display_row.get("refresh_rate")
    panel_type_id     = display_row.get("panel_type_id")

    # Resolve panel class from panel_type_id (lookup_panel_types not yet cached;
    # keep it simple — a None panel_type_id produces None panel_class)
    panel_class: str | None = None
    if panel_type_id is not None:
        panel_result = (
            client
            .schema("mobile_specs")
            .table("lookup_panel_types")
            .select("panel_class")
            .eq("panel_type_id", panel_type_id)
            .limit(1)
            .execute()
        )
        panel_rows  = panel_result.data or []
        panel_class = panel_rows[0].get("panel_class") if panel_rows else None

    # ── 5G presence (network table) ────────────────────────────────────────
    network_result = (
        client
        .schema("mobile_specs")
        .table("network")
        .select("network_id")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    )
    # We check for 5G bands via phone_network_bands JOIN lookup_network_bands
    has_5g = False
    if network_result.data:
        bands_result = (
            client
            .schema("mobile_specs")
            .table("phone_network_bands")
            .select("lookup_network_bands(band_name)")
            .eq("model_id", model_id)
            .execute()
        )
        for band_row in (bands_result.data or []):
            band_name = (band_row.get("lookup_network_bands") or {}).get("band_name", "")
            if band_name.startswith("n") and band_name[1:].isdigit():
                has_5g = True
                break

    spec_inputs = {
        "brand_name":              brand_name,
        "model_name":              model_name,
        "chipset_name":            chipset_name,
        "ram_gb_options":          ram_gb_options,
        "storage_gb_options":      storage_gb_options,
        "battery_capacity":        battery_capacity,
        "charging_power":          charging_power,
        "primary_camera_mp":       float(primary_camera_mp) if primary_camera_mp is not None else None,
        "primary_camera_aperture": float(primary_camera_aperture) if primary_camera_aperture is not None else None,
        "panel_class":             panel_class,
        "display_size_inch":       float(display_size_inch) if display_size_inch is not None else None,
        "refresh_rate":            refresh_rate,
        "has_5g":                  has_5g,
    }

    logger.debug(
        "fetch_spec_summary_inputs: model_id=%d brand=%s model=%s chipset=%s",
        model_id, brand_name, model_name, chipset_name,
    )
    return spec_inputs
