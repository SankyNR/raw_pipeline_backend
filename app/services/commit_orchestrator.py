"""
Phase 10 — DB Commit Orchestrator

Task 10.1 — run_db_commit

Atomic pipeline: final_merged_json → mobile_specs schema + phone_experiences committed.

TRANSACTION MODEL
─────────────────
Supabase-py does NOT expose a native asynchronous BEGIN/COMMIT API in the Python
client. We simulate atomicity by:
  1. Running a pre-flight check (validation gate) before any writes.
  2. Executing every write as an upsert (idempotent). A partial-failure restart
     simply re-runs from the same final_json and produces identical rows.
  3. Tracking progress in pipeline.db_commit_runs (status='running' → 'completed'/'failed').
  4. Using the UNIQUE / ON CONFLICT DO UPDATE semantics in mobile_specs_repository
     so repeated calls never create duplicate rows and never delete existing data.

This is the industry-standard "idempotent saga" pattern for pipelines that cannot
hold a lock across an async boundary.

STRICT WRITE ORDER (dependencies must be inserted before their children):
  1.  public.brands           — must already exist (pre-seeded); we resolve brand_id
  2.  mobile_specs.chipsets   — UPSERT on chipset_name
  3.  mobile_specs.phones     — UPSERT on (brand_id, model_name); gets model_id
  4.  mobile_specs.variant[]  — UPSERT per variant
  5.  mobile_specs.phone_displays[] → phone_display_features[]
  6.  mobile_specs.body_features
  7.  mobile_specs.charging_specs → phone_charger_technologies[]
  8.  mobile_specs.audio → phone_audio_codecs[]
  9.  mobile_specs.sensors → phone_sensors[]
  10. mobile_specs.connectivity → phone_wifi_technologies[]
                                → phone_location_services[]
                                → phone_usb_features[]
  11. mobile_specs.network → phone_network_bands[]
                           → phone_cellular_features[]
  12. mobile_specs.camera_overview
  13. mobile_specs.camera_lens_specs[] → lens_stabilization[]
                                       → phone_camera_features[]
  14. mobile_specs.os_and_security → phone_biometrics[]
                                   → phone_unlock_methods[]
                                   → phone_security_features[]
  15. mobile_specs.certifications → phone_ip_ratings[]
                                  → phone_video_certifications[]
                                  → phone_audio_certifications[]
  16. mobile_specs.ai_capabilities → phone_ai_features[]
  17. mobile_specs.phone_extra_features[]
  18. mobile_specs.phone_box_contents[]
  20a. mobile_specs.video_capabilities
  20b. mobile_specs.phone_camera_features[]  (root-level camera_features)
  19. pipeline.url_registry.status = 'stored_mainDB'   ← LAST WRITE (commit marker)

FK RESOLUTION AT COMMIT TIME
─────────────────────────────
For every field in SCALAR_FK_MAP:
  - If the value is an int → already resolved (normaliser did this).
  - If the value is a str  → resolve via LOOKUP_CACHE.
  - If not found in cache  → write NULL, append to unresolved_fields.

AFTER SUCCESSFUL COMMIT
────────────────────────
  asyncio.create_task(run_inference_engine(model_id, url_registry_id))
  # Run C runs asynchronously. Does NOT block the commit response.

FAILURE HANDLING
─────────────────
Any exception mid-sequence:
  - update db_commit_runs status='failed', error_message=str(exc)
  - Re-raise as HTTP 500 from the API layer.
  - Because all writes are idempotent, retrying the commit endpoint will
    safely complete the missing steps without creating duplicates.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any

from app.config.field_mapping import ARRAY_FK_MAP, SCALAR_FK_MAP
from app.repositories.extraction_repository import (
    fetch_final_merged_json,
    fetch_non_suppressed_experiences_for_commit as fetch_phone_experiences_for_commit,
)
from app.repositories.mobile_specs_repository import (
    fetch_brand_id,
    insert_ai_feature,
    insert_audio_certification,
    insert_audio_codec,
    insert_biometric,
    insert_box_content,
    insert_camera_feature,
    insert_cellular_feature,
    insert_charger_technology,
    insert_display_feature,
    insert_extra_feature,
    insert_ip_rating,
    insert_lens_stabilization,
    insert_location_service,
    insert_network_band,
    insert_phone_sensor,
    insert_security_feature,
    insert_unlock_method,
    insert_usb_feature,
    insert_video_certification,
    insert_wifi_technology,
    insert_commit_run,
    mark_url_registry_stored,
    update_commit_run,
    upsert_ai_capabilities,
    upsert_audio,
    upsert_body_features,
    upsert_camera_lens,
    upsert_camera_overview,
    upsert_certifications,
    upsert_charging_specs,
    upsert_chipset,
    upsert_connectivity,
    upsert_display,
    upsert_network,
    upsert_os_and_security,
    upsert_phone,
    upsert_sensors,
    upsert_variant,
    upsert_video_capabilities,
)
from app.services.normalizer import LOOKUP_CACHE, clean_for_lookup
from app.services.staging_resolver import auto_resolve_staging_at_commit
from app.services.validation_service import run_pre_commit_validation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# S2-P1-8: Display type → position semantic mapping
# Replaces brittle index-based defaulting (idx==0 → "Primary").
# Foldable phones may have Cover display at index 0 in final_json.
# ---------------------------------------------------------------------------
_DISPLAY_TYPE_TO_POSITION: dict[str, str] = {
    "main":  "Primary",
    "inner": "Primary",    # foldable main/inner screen → Primary
    "cover": "Secondary",  # foldable cover/external screen → Secondary
}


# ---------------------------------------------------------------------------
# FK resolution helper
# ---------------------------------------------------------------------------

def _resolve_fk(value: Any, table_path: str) -> int | None:
    """
    Resolve a value to a lookup table PK at commit time.

    Args:
        value:      The field value. May already be an int (already-resolved PK),
                    a string (needs lookup), or None.
        table_path: \"schema.table.column\" key in LOOKUP_CACHE.

    Returns:
        Integer PK, or None if not resolvable.
    """
    if value is None:
        return None
    if isinstance(value, int):
        # Already resolved by normaliser.
        return value
    if not isinstance(value, str):
        logger.warning("_resolve_fk: unexpected type %s for table_path=%r value=%r", type(value).__name__, table_path, value)
        return None

    cache = LOOKUP_CACHE.get(table_path, {})
    cleaned = clean_for_lookup(value)
    pk = cache.get(cleaned)
    if pk is None:
        logger.warning("_resolve_fk: value=%r not found in cache for table_path=%r", value, table_path)
    return pk


def _resolve_array_fk(values: list[Any] | None, table_path: str) -> tuple[list[int], list[str]]:
    """
    Resolve a list of values to lookup PKs.

    Returns:
        (resolved_ids, unresolved_values)
    """
    if not values:
        return [], []
    resolved: list[int] = []
    unresolved: list[str] = []
    for v in values:
        pk = _resolve_fk(v, table_path)
        if pk is not None:
            resolved.append(pk)
        else:
            unresolved.append(str(v))
    return resolved, unresolved


# ---------------------------------------------------------------------------
# Thin wrappers — run each sync repo call in a thread
# ---------------------------------------------------------------------------

async def _t(fn, *args, **kwargs):
    """Run a synchronous repository function in a thread pool."""
    return await asyncio.to_thread(fn, *args, **kwargs)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

async def run_db_commit(final_id: int, session_id: int | None) -> dict:
    """
    Phase 10 atomic commit: final_merged_json → mobile_specs schema.

    PRE-COMMIT (raises HTTP 400 if failed):
      Runs run_pre_commit_validation (Layer 3). If it fails, returns immediately
      without opening any writes. The db_commit_runs row is NOT created — nothing
      is written to the DB.

    SEQUENCE:
      Creates db_commit_runs row (status='running').
      Executes upserts in dependency order (see module docstring).
      On success: updates status='completed', marks url_registry='stored_mainDB'.
      On failure: updates status='failed', re-raises.

    Returns:
      {
        \"success\":           bool,
        \"model_id\":          int,
        \"commit_run_id\":     int,
        \"tables_written\":    list[str],
        \"rows_inserted\":     int,
        \"unresolved_fields\": list[str],
      }
    """
    # ── Step 0: Fetch final_merged_json package ──────────────────────────────
    package = await _t(fetch_final_merged_json, final_id)
    if not package:
        raise ValueError(f"run_db_commit: final_id={final_id} not found in final_merged_json")

    url_registry_id: int = package["url_registry_id"]
    final_json: dict = dict(package["final_json"])  # shallow copy — safe to mutate
    normalized_id: int = package.get("normalized_id")

    # video_capabilities is committed in Step 20a below.
    # The field is left in final_json — no pop needed.

    # ── Step 1: Pre-commit validation (Layer 3) ──────────────────────────────
    validation = await run_pre_commit_validation(
        final_id=final_id,
        url_registry_id=url_registry_id,
    )
    if not validation["passed"]:
        logger.warning(
            "run_db_commit: pre-commit validation FAILED for final_id=%d errors=%d",
            final_id, len(validation["errors"]),
        )
        raise ValueError(
            f"Pre-commit validation failed ({len(validation['errors'])} errors). "
            f"Commit aborted. commit_val_id={validation.get('commit_val_id')}"
        )

    # ── Step 2: Create audit row ─────────────────────────────────────────────
    commit_run_id = await _t(insert_commit_run, final_id, url_registry_id, session_id)

    # ── Step 2.5: Auto-resolve pending staging entries ────────────────────────
    # Reads pipeline.lookup_value_staging WHERE url_registry_id=... AND status='pending_review'.
    # For each entry: INSERT into target table, hot-patch LOOKUP_CACHE / ALIAS_CACHE,
    # back-patch final_json (APPEND for arrays, REPLACE for scalars), mark resolved.
    # Skipped silently if no pending entries (steady-state behaviour).
    # Individual row errors are collected but do NOT abort the commit.
    auto_resolve_result = await auto_resolve_staging_at_commit(
        url_registry_id=url_registry_id,
        final_json=final_json,
    )
    if auto_resolve_result["inserted_count"] > 0:
        logger.info(
            "run_db_commit: Step 2.5 — auto-resolved %d staging entries for url_registry_id=%d",
            auto_resolve_result["inserted_count"], url_registry_id,
        )
    if auto_resolve_result["errors"]:
        logger.warning(
            "run_db_commit: Step 2.5 — %d staging entries failed to auto-resolve for "
            "url_registry_id=%d (commit continues): %s",
            len(auto_resolve_result["errors"]), url_registry_id,
            auto_resolve_result["errors"],
        )

    # ── Commit state tracking ────────────────────────────────────────────────
    tables_written: list[str] = []
    rows_inserted: int = 0
    unresolved_fields: list[str] = []
    model_id: int = -1

    try:
        # ── Step 3: Resolve brand ─────────────────────────────────────────────
        brand_name: str = final_json.get("brand", {}).get("brand_name", "")  # v5: was basic.brand
        brand_id: int | None = await _t(fetch_brand_id, brand_name)
        if brand_id is None:
            raise ValueError(
                f"run_db_commit: brand_name={brand_name!r} not found in public.brands. "
                "The brand must be pre-seeded before committing."
            )

        # ── Step 4: Upsert chipset ────────────────────────────────────────────
        chipset_data = final_json.get("chipset", {})
        chipset_id: int | None = None

        chipset_name = chipset_data.get("chipset_name")
        if chipset_name:
            # Change 6b: If the normalizer dedup step already resolved the chipset
            # (chipset_id sentinel present), skip the INSERT and reuse the existing row.
            dedup_chipset_id: int | None = chipset_data.get("chipset_id")
            if dedup_chipset_id is not None:
                chipset_id = dedup_chipset_id
                logger.info(
                    "run_db_commit: chipset %r already in DB (chipset_id=%d) — "
                    "skipping upsert_chipset, reusing existing row.",
                    chipset_name, chipset_id,
                )
            else:
                cs_row: dict[str, Any] = {"chipset_name": chipset_name}
                _copy_if_present(chipset_data, cs_row, {
                    "cpu_architecture":           "cpu_architecture",
                    "fabrication_node":           "fabrication_node",
                    "number_of_cores":            "number_of_cores",
                    "cpu_ultra_high_performance_cores": "cpu_ultra_high_performance_cores",  # Change 1c
                    "cpu_high_performance_cores":      "cpu_high_performance_cores",
                    "cpu_efficiency_cores":       "cpu_efficiency_cores",
                    # cpu_clock_speed removed — Migration 49: phone-level, written to phones row
                    "gpu_name":                   "gpu_name",
                    # gpu_unit_count + gpu_unit_type removed — Migration 50: replaced by gpu_cores
                    "gpu_cores":                  "gpu_cores",
                    # gpu_clock_speed removed — Migration 49: phone-level, written to phones row
                    "npu_details":                "npu_details",
                    "npu_tops":                   "npu_tops",
                })
                chipset_id = await _t(upsert_chipset, cs_row)
                tables_written.append("mobile_specs.chipsets")
                rows_inserted += 1

        # ── Step 5: Upsert phone ──────────────────────────────────────────────
        # v5: model_name and launch_date moved from "basic" → "phone_identity".
        # We strip whitespace only; casing is preserved for the UNIQUE CONSTRAINT.
        phone_identity = final_json.get("phone_identity", {})
        raw_model_name: str = (phone_identity.get("model_name") or "").strip()
        if not raw_model_name:
            raise ValueError(
                f"run_db_commit: phone_identity.model_name is empty for final_id={final_id}. "
                "Cannot upsert phones without a model_name."
            )
        phone_row: dict[str, Any] = {
            "brand_id":   brand_id,
            "model_name": raw_model_name,
        }
        if chipset_id is not None:
            phone_row["chipset_id"] = chipset_id
        launch_date = phone_identity.get("launch_date")
        if launch_date:
            phone_row["launch_date"] = launch_date

        # Migration 49 — cpu_clock_speed and gpu_clock_speed are now phone-level columns.
        # Read from chipset block (where LLM placed them) and write to the phones row.
        _copy_if_present(chipset_data, phone_row, {
            "cpu_clock_speed": "cpu_clock_speed",
            "gpu_clock_speed": "gpu_clock_speed",
        })

        model_id = await _t(upsert_phone, phone_row)
        tables_written.append("mobile_specs.phones")
        rows_inserted += 1

        # ── Step 6: Upsert variants ───────────────────────────────────────────
        raw_variants: list[dict] = final_json.get("variants", [])
        
        # Deduplicate by (ram_capacity, storage_capacity), keeping the lowest launch_price
        dedup_map: dict[tuple[Any, Any], dict] = {}
        for variant in raw_variants:
            ram = variant.get("ram_capacity")
            storage = variant.get("storage_capacity")
            key = (ram, storage)
            price = variant.get("launch_price")
            
            if key not in dedup_map:
                dedup_map[key] = variant
            else:
                existing_price = dedup_map[key].get("launch_price")
                if price is not None:
                    if existing_price is None or price < existing_price:
                        dedup_map[key] = variant
        
        variants: list[dict] = list(dedup_map.values())

        for idx, variant in enumerate(variants):
            ram_type_pk = _resolve_fk(
                variant.get("ram_type"),
                "mobile_specs.lookup_ram_types.ram_type",
            )
            storage_type_pk = _resolve_fk(
                variant.get("storage_type"),
                "mobile_specs.lookup_storage_types.storage_type",
            )
            if ram_type_pk is None and variant.get("ram_type"):
                unresolved_fields.append(f"variants[{idx}].ram_type")
            if storage_type_pk is None and variant.get("storage_type"):
                unresolved_fields.append(f"variants[{idx}].storage_type")

            vrow: dict[str, Any] = {"model_id": model_id}
            _copy_if_present(variant, vrow, {
                "ram_capacity":             "ram_capacity",
                "ram_frequency":            "ram_frequency",
                "virtual_ram_availability": "virtual_ram_availability",
                "virtual_ram_size":         "virtual_ram_size",
                "storage_capacity":         "storage_capacity",
                "expandable_storage":       "expandable_storage",
                "launch_price":             "launch_price",
                "is_base_variant":          "is_base_variant",
            })
            vrow["ram_type_id"] = ram_type_pk       # None → NULL; 0 is not a valid FK
            vrow["storage_type_id"] = storage_type_pk  # None → NULL; 0 is not a valid FK

            await _t(upsert_variant, vrow)
            rows_inserted += 1

        if variants:
            tables_written.append("mobile_specs.variant")

        # ── Step 7: Upsert displays ───────────────────────────────────────────
        displays: list[dict] = final_json.get("displays", [])
        display_features_written = False

        for idx, display in enumerate(displays):
            # S2-P1-8: derive display_position from display_type semantics,
            # not from array index. Foldable phones may have Cover at index 0.
            display_type = display.get("display_type") or "main"
            display_type_lower = str(display_type).lower()
            fallback_position = _DISPLAY_TYPE_TO_POSITION.get(display_type_lower, "Primary")
            display_position = display.get("display_position") or fallback_position

            panel_type_pk = _resolve_fk(
                display.get("panel_type"),
                "mobile_specs.lookup_panel_types.panel_type",
            )
            shape_pk = _resolve_fk(
                display.get("screen_shape"),
                "mobile_specs.lookup_screen_shapes.shape_name",
            )
            protection_pk = _resolve_fk(
                display.get("glass_protection"),
                "mobile_specs.lookup_glass_protection.protection_name",
            )
            for fpath, pvar in [
                (f"displays[{idx}].panel_type",    panel_type_pk),
                (f"displays[{idx}].screen_shape",  shape_pk),
                (f"displays[{idx}].glass_protection", protection_pk),
            ]:
                val_raw = display.get(fpath.rsplit(".", 1)[-1])
                if val_raw and pvar is None:
                    unresolved_fields.append(fpath)

            drow: dict[str, Any] = {
                "model_id":        model_id,
                "display_type":    display_type,
                "display_position": display_position,
            }
            if panel_type_pk is not None:
                drow["panel_type_id"] = panel_type_pk
            if shape_pk is not None:
                drow["shape_id"] = shape_pk
            if protection_pk is not None:
                drow["protection_id"] = protection_pk

            _copy_if_present(display, drow, {
                "size_inch":             "size_inch",
                "resolution_height_px":  "resolution_height_px",
                "resolution_width_px":   "resolution_width_px",
                "aspect_ratio":          "aspect_ratio",
                "ppi":                   "ppi",
                "colour_depth":          "colour_depth",
                "refresh_rate":          "refresh_rate",
                "brightness_hbm":        "brightness_hbm",
                "brightness_peak":       "brightness_peak",
                "pwm_frequency":         "pwm_frequency",
                "screen_to_body_ratio":  "screen_to_body_ratio",
            })

            display_id = await _t(upsert_display, drow)
            rows_inserted += 1

            # Display features (junction)
            feat_table = "mobile_specs.lookup_display_features.feature_name"
            raw_feats: list[Any] = display.get("display_features", []) or []
            feat_ids, feat_unresolved = _resolve_array_fk(raw_feats, feat_table)
            for uid in feat_unresolved:
                unresolved_fields.append(f"displays[{idx}].display_features.{uid}")
            for fid in feat_ids:
                inserted = await _t(insert_display_feature, display_id, fid)
                if inserted:
                    rows_inserted += 1
                    display_features_written = True

        if displays:
            tables_written.append("mobile_specs.phone_displays")
        if display_features_written:
            tables_written.append("mobile_specs.phone_display_features")

        # ── Step 8: Body features ─────────────────────────────────────────────
        body = final_json.get("body", {})
        if body:
            brow: dict[str, Any] = {"model_id": model_id}
            _copy_if_present(body, brow, {
                "height":           "height",
                "width":            "width",
                "thickness":        "thickness",
                "height_folded":    "height_folded",
                "width_folded":     "width_folded",
                "thickness_folded": "thickness_folded",
                "weight":           "weight",
                "build":            "build",
                "buttons":          "buttons",
                "colors":           "colors",
                "has_stylus":       "has_stylus",
                "stylus_features":  "stylus_features",
                "other_features":   "other_features",
            })
            await _t(upsert_body_features, brow)
            tables_written.append("mobile_specs.body_features")
            rows_inserted += 1

        # ── Step 9: Charging specs ────────────────────────────────────────────
        charging = final_json.get("charging", {})
        charger_tech_written = False

        if charging:
            battery_type_pk = _resolve_fk(
                charging.get("battery_type"),
                "mobile_specs.lookup_battery_types.battery_type",
            )
            cable_type_pk = _resolve_fk(
                charging.get("cable_type"),
                "mobile_specs.lookup_cable_types.cable_type",
            )
            prop_charging_pk = _resolve_fk(
                charging.get("proprietary_charging"),
                "mobile_specs.lookup_proprietary_charging.technology_name",
            )
            wireless_std_pk = _resolve_fk(
                charging.get("wireless_charging_standard"),
                "mobile_specs.lookup_wireless_charging_standards.standard_name",
            )
            for fpath, pvar, raw_key in [
                ("charging.battery_type",             battery_type_pk,  "battery_type"),
                ("charging.cable_type",               cable_type_pk,    "cable_type"),
                ("charging.proprietary_charging",     prop_charging_pk, "proprietary_charging"),
                ("charging.wireless_charging_standard", wireless_std_pk, "wireless_charging_standard"),
            ]:
                if charging.get(raw_key) and pvar is None:
                    unresolved_fields.append(fpath)

            crow: dict[str, Any] = {"model_id": model_id}
            _copy_if_present(charging, crow, {
                "battery_capacity":                "battery_capacity",
                "charging_voltage":                "charging_voltage",
                "charging_ampere":                 "charging_ampere",
                "charging_power":                  "charging_power",
                "charger_in_box":                  "charger_in_box",
                "wireless_charging":               "wireless_charging",
                "wireless_charging_power":         "wireless_charging_power",
                "reverse_wireless_charging":       "reverse_wireless_charging",
                "reverse_wireless_charging_power": "reverse_wireless_charging_power",
                "battery_and_charging_features":   "battery_and_charging_features",
            })
            if battery_type_pk is not None:
                crow["battery_type_id"] = battery_type_pk
            if cable_type_pk is not None:
                crow["cable_type_id"] = cable_type_pk
            if prop_charging_pk is not None:
                crow["proprietary_charging_id"] = prop_charging_pk
            if wireless_std_pk is not None:
                crow["wireless_charging_standard_id"] = wireless_std_pk

            await _t(upsert_charging_specs, crow)
            tables_written.append("mobile_specs.charging_specs")
            rows_inserted += 1

            # Charger technologies (junction)
            ct_table = "mobile_specs.lookup_charger_technologies.technology_name"
            ct_ids, ct_unresolved = _resolve_array_fk(
                charging.get("charger_technologies", []), ct_table
            )
            for uid in ct_unresolved:
                unresolved_fields.append(f"charging.charger_technologies.{uid}")
            for tid in ct_ids:
                inserted = await _t(insert_charger_technology, model_id, tid)
                if inserted:
                    rows_inserted += 1
                    charger_tech_written = True

        if charger_tech_written:
            tables_written.append("mobile_specs.phone_charger_technologies")

        # ── Step 10: Audio ────────────────────────────────────────────────────
        audio = final_json.get("audio", {})
        audio_codecs_written = False

        if audio:
            arow: dict[str, Any] = {"model_id": model_id}
            _copy_if_present(audio, arow, {
                "speaker_count":       "speaker_count",
                "speaker_positions":   "speaker_positions",
                "microphone_count":    "microphone_count",
                "microphone_positions": "microphone_positions",
                "has_3_5mm_jack":      "has_3_5mm_jack",
                "audio_features":      "audio_features",
            })
            await _t(upsert_audio, arow)
            tables_written.append("mobile_specs.audio")
            rows_inserted += 1

            # Audio codecs (junction)
            ac_table = "mobile_specs.lookup_audio_codecs.codec_name"
            ac_ids, ac_unresolved = _resolve_array_fk(audio.get("audio_codecs", []), ac_table)
            for uid in ac_unresolved:
                unresolved_fields.append(f"audio.audio_codecs.{uid}")
            for cid in ac_ids:
                inserted = await _t(insert_audio_codec, model_id, cid)
                if inserted:
                    rows_inserted += 1
                    audio_codecs_written = True

        if audio_codecs_written:
            tables_written.append("mobile_specs.phone_audio_codecs")

        # ── Step 11: Sensors ──────────────────────────────────────────────────
        sensors = final_json.get("sensors", {})
        sensors_written = False

        if sensors:
            serow: dict[str, Any] = {"model_id": model_id}
            _copy_if_present(sensors, serow, {
                "fingerprint_sensor": "fingerprint_sensor",
            })
            await _t(upsert_sensors, serow)
            tables_written.append("mobile_specs.sensors")
            rows_inserted += 1

            # phone_sensors junction
            ps_table = "mobile_specs.lookup_sensors.sensor_name"
            ps_ids, ps_unresolved = _resolve_array_fk(sensors.get("other_sensors", []), ps_table)
            for uid in ps_unresolved:
                unresolved_fields.append(f"sensors.other_sensors.{uid}")
            for sid in ps_ids:
                inserted = await _t(insert_phone_sensor, model_id, sid)
                if inserted:
                    rows_inserted += 1
                    sensors_written = True

        if sensors_written:
            tables_written.append("mobile_specs.phone_sensors")

        # ── Step 12: Connectivity ─────────────────────────────────────────────
        connectivity = final_json.get("connectivity", {})
        conn_junctions_written: set[str] = set()

        if connectivity:
            wifi_pk = _resolve_fk(
                connectivity.get("wifi_standard"),
                "mobile_specs.lookup_wifi_standards.wifi_standard",
            )
            bt_pk = _resolve_fk(
                connectivity.get("bluetooth_version"),
                "mobile_specs.lookup_bluetooth_versions.bluetooth_version",
            )
            usb_pk = _resolve_fk(
                connectivity.get("usb_standard"),
                "mobile_specs.lookup_usb_standards.usb_standard",
            )
            for fpath, pvar, raw_key in [
                ("connectivity.wifi_standard",     wifi_pk, "wifi_standard"),
                ("connectivity.bluetooth_version", bt_pk,   "bluetooth_version"),
                ("connectivity.usb_standard",      usb_pk,  "usb_standard"),
            ]:
                if connectivity.get(raw_key) and pvar is None:
                    unresolved_fields.append(fpath)

            cconn: dict[str, Any] = {"model_id": model_id}
            _copy_if_present(connectivity, cconn, {
                "nfc":           "nfc",
                "uwb":           "uwb",
                "ir_blaster":    "ir_blaster",
                "wifi_hotspot":  "wifi_hotspot",
            })
            if wifi_pk is not None:
                cconn["wifi_id"] = wifi_pk
            if bt_pk is not None:
                cconn["bluetooth_id"] = bt_pk
            if usb_pk is not None:
                cconn["usb_id"] = usb_pk

            await _t(upsert_connectivity, cconn)
            tables_written.append("mobile_specs.connectivity")
            rows_inserted += 1

            # Wifi technologies
            wt_table = "mobile_specs.lookup_wifi_technologies.technology_name"
            wt_ids, wt_un = _resolve_array_fk(connectivity.get("wifi_technologies", []), wt_table)
            for uid in wt_un:
                unresolved_fields.append(f"connectivity.wifi_technologies.{uid}")
            for wid in wt_ids:
                inserted = await _t(insert_wifi_technology, model_id, wid)
                if inserted:
                    rows_inserted += 1
                    conn_junctions_written.add("mobile_specs.phone_wifi_technologies")

            # Location services
            ls_table = "mobile_specs.lookup_location_services.location_system"
            ls_ids, ls_un = _resolve_array_fk(connectivity.get("location_services", []), ls_table)
            for uid in ls_un:
                unresolved_fields.append(f"connectivity.location_services.{uid}")
            for lid in ls_ids:
                inserted = await _t(insert_location_service, model_id, lid)
                if inserted:
                    rows_inserted += 1
                    conn_junctions_written.add("mobile_specs.phone_location_services")

            # USB features
            uf_table = "mobile_specs.lookup_usb_features.feature_name"
            uf_ids, uf_un = _resolve_array_fk(connectivity.get("usb_features", []), uf_table)
            for uid in uf_un:
                unresolved_fields.append(f"connectivity.usb_features.{uid}")
            for fid in uf_ids:
                inserted = await _t(insert_usb_feature, model_id, fid)
                if inserted:
                    rows_inserted += 1
                    conn_junctions_written.add("mobile_specs.phone_usb_features")

        tables_written.extend(sorted(conn_junctions_written))

        # ── Step 13: Network ──────────────────────────────────────────────────
        network = final_json.get("network", {})
        net_junctions_written: set[str] = set()

        if network:
            sim_config_pk = _resolve_fk(
                network.get("sim_configuration"),
                "mobile_specs.lookup_sim_configurations.configuration_name",
            )
            if network.get("sim_configuration") and sim_config_pk is None:
                unresolved_fields.append("network.sim_configuration")

            nrow: dict[str, Any] = {"model_id": model_id}
            _copy_if_present(network, nrow, {
                "number_of_sims":    "number_of_sims",
                "esim_support":      "esim_support",
                "sim_tray_position": "sim_tray_position",
                "bands_2g":          "bands_2g",
                "bands_3g":          "bands_3g",
                "volte":             "volte",
                "vo5g":              "vo5g",
                "vowifi":            "vowifi",
            })
            if sim_config_pk is not None:
                nrow["sim_config_id"] = sim_config_pk

            await _t(upsert_network, nrow)
            tables_written.append("mobile_specs.network")
            rows_inserted += 1

            # 5G + 4G bands → phone_network_bands (same junction table)
            band_table = "mobile_specs.lookup_network_bands.band_name"
            all_bands: list = list(network.get("bands_5g", []) or []) + list(network.get("bands_4g", []) or [])
            band_ids, band_un = _resolve_array_fk(all_bands, band_table)
            seen_band_ids: set[int] = set()
            for uid in band_un:
                unresolved_fields.append(f"network.bands.{uid}")
            for bid in band_ids:
                if bid in seen_band_ids:
                    continue
                seen_band_ids.add(bid)
                inserted = await _t(insert_network_band, model_id, bid)
                if inserted:
                    rows_inserted += 1
                    net_junctions_written.add("mobile_specs.phone_network_bands")

            # Satellite bands (Migration 51) → same phone_network_bands junction table
            sat_bands: list = list(network.get("bands_satellite", []) or [])
            sat_band_ids, sat_band_un = _resolve_array_fk(sat_bands, band_table)
            for uid in sat_band_un:
                unresolved_fields.append(f"network.bands_satellite.{uid}")
            for bid in sat_band_ids:
                if bid in seen_band_ids:
                    continue
                seen_band_ids.add(bid)
                inserted = await _t(insert_network_band, model_id, bid)
                if inserted:
                    rows_inserted += 1
                    net_junctions_written.add("mobile_specs.phone_network_bands")

            # Cellular features
            cf_table = "mobile_specs.lookup_cellular_features.feature_name"
            cf_ids, cf_un = _resolve_array_fk(network.get("cellular_features", []), cf_table)
            for uid in cf_un:
                unresolved_fields.append(f"network.cellular_features.{uid}")
            for fid in cf_ids:
                inserted = await _t(insert_cellular_feature, model_id, fid)
                if inserted:
                    rows_inserted += 1
                    net_junctions_written.add("mobile_specs.phone_cellular_features")

        tables_written.extend(sorted(net_junctions_written))

        # ── Step 14: Camera overview ──────────────────────────────────────────
        camera_overview_data = final_json.get("camera_overview", {})

        if camera_overview_data:
            rear_setup_pk     = _resolve_fk(camera_overview_data.get("rear_camera_setup"),    "mobile_specs.lookup_rear_camera_setup.setup_name")
            front_setup_pk    = _resolve_fk(camera_overview_data.get("front_camera_setup"),   "mobile_specs.lookup_front_camera_setup.setup_name")
            front_shape_pk    = _resolve_fk(camera_overview_data.get("front_camera_shape"),   "mobile_specs.lookup_front_camera_shape.shape_name")
            island_shape_pk   = _resolve_fk(camera_overview_data.get("rear_camera_island_shape"), "mobile_specs.lookup_rear_camera_island_shape.shape_name")
            island_pos_pk     = _resolve_fk(camera_overview_data.get("rear_camera_island_position"), "mobile_specs.lookup_rear_camera_island_position.position_name")
            front_pos_pk      = _resolve_fk(camera_overview_data.get("front_camera_position"), "mobile_specs.lookup_front_camera_position.position_name")
            flash_pk          = _resolve_fk(camera_overview_data.get("flash"),                 "mobile_specs.lookup_flash_types.flash_type")

            corow: dict[str, Any] = {"model_id": model_id}
            if rear_setup_pk:
                corow["rear_camera_setup_id"] = rear_setup_pk
            else:
                if camera_overview_data.get("rear_camera_setup"):
                    unresolved_fields.append("camera_overview.rear_camera_setup")
            if front_setup_pk:
                corow["front_camera_setup_id"] = front_setup_pk
            else:
                if camera_overview_data.get("front_camera_setup"):
                    unresolved_fields.append("camera_overview.front_camera_setup")
            if front_shape_pk:
                corow["front_camera_shape_id"] = front_shape_pk
            if island_shape_pk:
                corow["rear_camera_island_shape_id"] = island_shape_pk
            if island_pos_pk:
                corow["rear_camera_island_position_id"] = island_pos_pk
            if front_pos_pk:
                corow["front_camera_position_id"] = front_pos_pk
            if flash_pk:
                corow["flash_id"] = flash_pk
            else:
                if camera_overview_data.get("flash"):
                    unresolved_fields.append("camera_overview.flash")

            await _t(upsert_camera_overview, corow)
            tables_written.append("mobile_specs.camera_overview")
            rows_inserted += 1

        # ── Step 15: Camera lenses ────────────────────────────────────────────
        camera_lenses: list[dict] = final_json.get("camera_lenses", [])
        lens_stab_written = False
        cam_feat_written = False

        for idx, lens in enumerate(camera_lenses):
            lens_type_pk = _resolve_fk(
                lens.get("lens_type"),
                "mobile_specs.lookup_lens_types.lens_type",
            )
            af_pk = _resolve_fk(
                lens.get("autofocus_type"),
                "mobile_specs.lookup_autofocus_types.autofocus_type",
            )
            if lens.get("lens_type") and lens_type_pk is None:
                unresolved_fields.append(f"camera_lenses[{idx}].lens_type")
            if lens.get("autofocus_type") and af_pk is None:
                unresolved_fields.append(f"camera_lenses[{idx}].autofocus_type")

            lrow: dict[str, Any] = {"model_id": model_id}
            if lens_type_pk is not None:
                lrow["lens_type_id"] = lens_type_pk
            if af_pk is not None:
                lrow["autofocus_type_id"] = af_pk

            _copy_if_present(lens, lrow, {
                "sensor_model":           "sensor_model",
                "sensor_type":            "sensor_type",
                "megapixels":             "megapixels",
                "sensor_size_denominator": "sensor_size_denominator",
                "sensor_size_decimal":    "sensor_size_decimal",
                "pixel_size":             "pixel_size",
                "aperture":               "aperture",
                "focal_length":           "focal_length",
                "fov":                    "fov",
                "digital_zoom_capacity":  "digital_zoom_capacity",
                "optical_zoom_capacity":  "optical_zoom_capacity",
                "is_macro_capable":       "is_macro_capable",
                "lens_features":          "lens_features",
            })

            lens_id = await _t(upsert_camera_lens, lrow)
            rows_inserted += 1

            # Stabilization (junction on lens_id)
            stab_table = "mobile_specs.lookup_stabilization_types.stabilization_type"
            stab_ids, stab_un = _resolve_array_fk(lens.get("stabilization", []), stab_table)
            for uid in stab_un:
                unresolved_fields.append(f"camera_lenses[{idx}].stabilization.{uid}")
            for sid in stab_ids:
                inserted = await _t(insert_lens_stabilization, lens_id, sid)
                if inserted:
                    rows_inserted += 1
                    lens_stab_written = True

            # camera_features is now a root-level field (committed in Step 20b below).
            # Per-lens camera_features is always [] after the schema change — do not read it here.

        if camera_lenses:
            tables_written.append("mobile_specs.camera_lens_specs")
        if lens_stab_written:
            tables_written.append("mobile_specs.lens_stabilization")
        if cam_feat_written:
            tables_written.append("mobile_specs.phone_camera_features")

        # ── Step 16: OS & Security ────────────────────────────────────────────
        os_sec = final_json.get("os_and_security", {})
        os_junctions_written: set[str] = set()

        if os_sec:
            os_pk = _resolve_fk(
                os_sec.get("os_name"),
                "mobile_specs.lookup_os_versions.os_name",
            )
            ui_pk = _resolve_fk(
                os_sec.get("ui_skin"),
                "mobile_specs.lookup_ui_skins.ui_skin_name",
            )
            if os_sec.get("os_name") and os_pk is None:
                unresolved_fields.append("os_and_security.os_name")
            if os_sec.get("ui_skin") and ui_pk is None:
                unresolved_fields.append("os_and_security.ui_skin")

            osrow: dict[str, Any] = {"model_id": model_id}
            _copy_if_present(os_sec, osrow, {
                "os_update_years":      "os_update_years",
                "security_update_years": "security_update_years",
            })
            if os_pk is not None:
                osrow["os_id"] = os_pk
            if ui_pk is not None:
                osrow["ui_skin_id"] = ui_pk

            await _t(upsert_os_and_security, osrow)
            tables_written.append("mobile_specs.os_and_security")
            rows_inserted += 1

            # Biometrics
            bm_ids, bm_un = _resolve_array_fk(
                os_sec.get("biometrics", []),
                "mobile_specs.lookup_biometric_types.biometric_name",
            )
            for uid in bm_un:
                unresolved_fields.append(f"os_and_security.biometrics.{uid}")
            for bid in bm_ids:
                inserted = await _t(insert_biometric, model_id, bid)
                if inserted:
                    rows_inserted += 1
                    os_junctions_written.add("mobile_specs.phone_biometrics")

            # Unlock methods
            um_ids, um_un = _resolve_array_fk(
                os_sec.get("unlock_methods", []),
                "mobile_specs.lookup_unlock_methods.method_name",
            )
            for uid in um_un:
                unresolved_fields.append(f"os_and_security.unlock_methods.{uid}")
            for uid in um_ids:
                inserted = await _t(insert_unlock_method, model_id, uid)
                if inserted:
                    rows_inserted += 1
                    os_junctions_written.add("mobile_specs.phone_unlock_methods")

            # Security features
            sf_ids, sf_un = _resolve_array_fk(
                os_sec.get("security_features", []),
                "mobile_specs.lookup_security_features.feature_name",
            )
            for uid in sf_un:
                unresolved_fields.append(f"os_and_security.security_features.{uid}")
            for fid in sf_ids:
                inserted = await _t(insert_security_feature, model_id, fid)
                if inserted:
                    rows_inserted += 1
                    os_junctions_written.add("mobile_specs.phone_security_features")

        tables_written.extend(sorted(os_junctions_written))

        # ── Step 17: Certifications ───────────────────────────────────────────
        certs = final_json.get("certifications", {})
        cert_junctions_written: set[str] = set()

        if certs:
            widevine_pk = _resolve_fk(
                certs.get("widevine_level"),
                "mobile_specs.lookup_widevine_levels.level_name",
            )
            if certs.get("widevine_level") and widevine_pk is None:
                unresolved_fields.append("certifications.widevine_level")

            certrow: dict[str, Any] = {"model_id": model_id}
            _copy_if_present(certs, certrow, {
                "sar_head":              "sar_head",
                "sar_body":              "sar_body",
                "widevine_support":      "widevine_support",
                "bis_certification":     "bis_certification",
                "other_certifications":  "other_certifications",
            })
            if widevine_pk is not None:
                certrow["widevine_level_id"] = widevine_pk
                certrow.setdefault("widevine_support", True)

            await _t(upsert_certifications, certrow)
            tables_written.append("mobile_specs.certifications")
            rows_inserted += 1

            # IP ratings
            ip_ids, ip_un = _resolve_array_fk(
                certs.get("ip_ratings", []),
                "mobile_specs.lookup_ip_ratings.rating_name",
            )
            for uid in ip_un:
                unresolved_fields.append(f"certifications.ip_ratings.{uid}")
            for iid in ip_ids:
                inserted = await _t(insert_ip_rating, model_id, iid)
                if inserted:
                    rows_inserted += 1
                    cert_junctions_written.add("mobile_specs.phone_ip_ratings")

            # Video certifications
            vc_ids, vc_un = _resolve_array_fk(
                certs.get("video_certifications", []),
                "mobile_specs.lookup_video_certifications.certification_name",
            )
            for uid in vc_un:
                unresolved_fields.append(f"certifications.video_certifications.{uid}")
            for vid in vc_ids:
                inserted = await _t(insert_video_certification, model_id, vid)
                if inserted:
                    rows_inserted += 1
                    cert_junctions_written.add("mobile_specs.phone_video_certifications")

            # Audio certifications
            ac_ids, ac_un = _resolve_array_fk(
                certs.get("audio_certifications", []),
                "mobile_specs.lookup_audio_certifications.certification_name",
            )
            for uid in ac_un:
                unresolved_fields.append(f"certifications.audio_certifications.{uid}")
            for acid in ac_ids:
                inserted = await _t(insert_audio_certification, model_id, acid)
                if inserted:
                    rows_inserted += 1
                    cert_junctions_written.add("mobile_specs.phone_audio_certifications")

        tables_written.extend(sorted(cert_junctions_written))

        # ── Step 18: AI Capabilities ──────────────────────────────────────────
        ai_data = final_json.get("ai_capabilities", {})
        ai_feat_junc_written = False

        if ai_data:
            ai_sys_pk = _resolve_fk(
                ai_data.get("ai_system"),
                "mobile_specs.lookup_ai_systems.system_name",
            )
            proc_type_pk = _resolve_fk(
                ai_data.get("processing_type"),
                "mobile_specs.lookup_ai_processing_types.processing_type",
            )
            if ai_data.get("ai_system") and ai_sys_pk is None:
                unresolved_fields.append("ai_capabilities.ai_system")
            if ai_data.get("processing_type") and proc_type_pk is None:
                unresolved_fields.append("ai_capabilities.processing_type")

            airow: dict[str, Any] = {"model_id": model_id}
            if ai_sys_pk is not None:
                airow["ai_system_id"] = ai_sys_pk
            if proc_type_pk is not None:
                airow["processing_type_id"] = proc_type_pk

            await _t(upsert_ai_capabilities, airow)
            tables_written.append("mobile_specs.ai_capabilities")
            rows_inserted += 1

            # AI features (junction)
            af_table = "mobile_specs.lookup_ai_features.feature_name"
            af_ids, af_un = _resolve_array_fk(ai_data.get("ai_features", []), af_table)
            for uid in af_un:
                unresolved_fields.append(f"ai_capabilities.ai_features.{uid}")
            for fid in af_ids:
                inserted = await _t(insert_ai_feature, model_id, fid)
                if inserted:
                    rows_inserted += 1
                    ai_feat_junc_written = True

        if ai_feat_junc_written:
            tables_written.append("mobile_specs.phone_ai_features")

        # ── Step 19: Extra features ───────────────────────────────────────────
        extra_feat_written = False
        ef_table = "mobile_specs.lookup_extra_features.feature_name"
        ef_ids, ef_un = _resolve_array_fk(final_json.get("extra_features", []), ef_table)
        for uid in ef_un:
            unresolved_fields.append(f"extra_features.{uid}")
        for fid in ef_ids:
            inserted = await _t(insert_extra_feature, model_id, fid)
            if inserted:
                rows_inserted += 1
                extra_feat_written = True
        if extra_feat_written:
            tables_written.append("mobile_specs.phone_extra_features")

        # ── Step 20: In-the-box ───────────────────────────────────────────────
        box_written = False
        box_table = "mobile_specs.lookup_box_items.item_name"
        in_the_box: list = final_json.get("in_the_box", []) or []
        for item in in_the_box:
            if isinstance(item, str):
                item_name = item
                item_spec = None
                item_qty = 1
            elif isinstance(item, dict):
                item_name = item.get("item_name") or item.get("name", "")
                item_spec = item.get("item_specification") or item.get("specification")
                item_qty = item.get("quantity", 1)
            else:
                continue

            box_item_pk = _resolve_fk(item_name, box_table)
            if box_item_pk is None:
                unresolved_fields.append(f"in_the_box.{item_name}")
            else:
                inserted = await _t(insert_box_content, model_id, box_item_pk, item_spec, item_qty)
                if inserted:
                    rows_inserted += 1
                    box_written = True

        if box_written:
            tables_written.append("mobile_specs.phone_box_contents")

        # ── Step 20a: Video capabilities ─────────────────────────────────────
        vc_data = final_json.get("video_capabilities", {})
        if vc_data and isinstance(vc_data, dict):
            vcrow: dict[str, Any] = {"model_id": model_id}
            _copy_if_present(vc_data, vcrow, {
                "rear_video_resolutions":  "rear_video_resolutions",
                "front_video_resolutions": "front_video_resolutions",
                "slow_motion_resolutions": "slow_motion_resolutions",
            })
            if len(vcrow) > 1:  # at least one field beyond model_id
                await _t(upsert_video_capabilities, vcrow)
                tables_written.append("mobile_specs.video_capabilities")
                rows_inserted += 1

        # ── Step 20b: Root-level camera_features (junction on model_id) ───────
        root_cf_table = "mobile_specs.lookup_camera_features.feature_name"
        root_cf_ids, root_cf_un = _resolve_array_fk(
            final_json.get("camera_features", []), root_cf_table
        )
        for uid in root_cf_un:
            unresolved_fields.append(f"camera_features.{uid}")
        for fid in root_cf_ids:
            inserted = await _t(insert_camera_feature, model_id, fid)
            if inserted:
                rows_inserted += 1
                cam_feat_written = True
        if cam_feat_written:
            tables_written.append("mobile_specs.phone_camera_features")

        # ── Step 21: Commit phone_experiences (mark as committed) ─────────────
        # phone_experiences live in the pipeline schema. For Phase 10 the design
        # is to mark committed = TRUE on the pipeline rows (they are already the
        # final source of truth for Run B). The mobile_specs schema does not have
        # a separate experiences table — they are read via the pipeline schema.
        # We mark them via is_verified = TRUE on all non-suppressed rows.
        exp_count = await _t(_mark_experiences_committed, url_registry_id)
        if exp_count > 0:
            tables_written.append("pipeline.phone_experiences (marked committed)")
            rows_inserted += exp_count

        # ── Step 22: Mark url_registry.status = 'stored_mainDB' (LAST WRITE) ─
        await _t(mark_url_registry_stored, url_registry_id)
        tables_written.append("pipeline.url_registry")

        # ── Step 23: Update audit row ─────────────────────────────────────────
        await _t(
            update_commit_run,
            commit_run_id,
            "completed",
            model_id,
            list(dict.fromkeys(tables_written)),   # dedup, preserve order
            rows_inserted,
            0,
            unresolved_fields,
            None,
        )

        logger.info(
            "run_db_commit: SUCCESS final_id=%d model_id=%d commit_run_id=%d "
            "tables=%d rows_inserted=%d unresolved=%d",
            final_id, model_id, commit_run_id,
            len(tables_written), rows_inserted, len(unresolved_fields),
        )

        # ── Step 24: Fire Run C asynchronously ────────────────────────────────
        # M6 fix: wrap the task coroutine so exceptions are logged at ERROR level
        # and never silently swallowed by the event loop.
        async def _run_inference_safely(
            _model_id: int = model_id,
            _url_registry_id: int = url_registry_id,
            _commit_run_id: int = commit_run_id,
        ) -> None:
            try:
                from app.services.inference_engine import run_inference_engine  # deferred
                await run_inference_engine(
                    model_id=_model_id,
                    url_registry_id=_url_registry_id,
                )
                logger.info(
                    "Run C: inference_engine completed for model_id=%d", _model_id
                )
            except ImportError:
                logger.info(
                    "Run C: inference_engine not yet implemented — skipping (model_id=%d).",
                    _model_id,
                )
            except Exception as _exc:
                logger.error(
                    "Run C: inference_engine FAILED for model_id=%d commit_run_id=%d: %s",
                    _model_id, _commit_run_id, _exc,
                    exc_info=True,
                )

        asyncio.create_task(_run_inference_safely())
        logger.info("run_db_commit: Run C task scheduled for model_id=%d", model_id)

        return {
            "success":           True,
            "model_id":          model_id,
            "commit_run_id":     commit_run_id,
            "tables_written":    list(dict.fromkeys(tables_written)),
            "rows_inserted":     rows_inserted,
            "unresolved_fields": unresolved_fields,
        }

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(
            "run_db_commit: FAILED final_id=%d commit_run_id=%d error=%s\n%s",
            final_id, commit_run_id, exc, tb,
        )
        # Best-effort: update audit row to failed
        try:
            await _t(
                update_commit_run,
                commit_run_id,
                "failed",
                model_id if model_id != -1 else None,
                list(dict.fromkeys(tables_written)),
                rows_inserted,
                0,
                unresolved_fields,
                str(exc),
            )
        except Exception as audit_exc:
            logger.error(
                "run_db_commit: also failed to update db_commit_runs: %s", audit_exc
            )
        raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy_if_present(
    src: dict[str, Any],
    dst: dict[str, Any],
    field_map: dict[str, str],  # {src_key: dst_key}
) -> None:
    """
    Copy fields from src → dst only when the value is not None.
    field_map maps source field name → destination column name.

    NOTE — intentional "never clear" behaviour (N2):
      Fields whose value is None in src are silently skipped. This means:
      - A previously committed non-NULL value in the DB is never overwritten
        with NULL by a re-commit, even if the admin explicitly nulled a field
        via a Phase 8 override.
      - This is a deliberate v1 policy to prevent accidental data loss.
      - To explicitly clear a committed field, an admin must issue a direct SQL
        update or a future "force_null" override path (not yet implemented).
    """
    for src_key, dst_key in field_map.items():
        val = src.get(src_key)
        if val is not None:
            dst[dst_key] = val


def _mark_experiences_committed(url_registry_id: int) -> int:
    """
    Mark all non-suppressed phone_experiences for this phone as is_verified=TRUE.
    Returns the number of rows updated.
    """
    from app.core.supabase_client import get_client as _get_client
    client = _get_client()
    result = (
        client
        .schema("pipeline")
        .table("phone_experiences")
        .update({"is_verified": True})
        .eq("url_registry_id", url_registry_id)
        .eq("is_suppressed", False)
        .eq("is_superseded", False)   # C3 fix: never touch superseded rows at commit
        .execute()
    )
    count = len(result.data or [])
    logger.debug("_mark_experiences_committed: url_registry_id=%d rows=%d", url_registry_id, count)
    return count
