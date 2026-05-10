"""
Phase 10 — mobile_specs Repository

All write operations to the mobile_specs schema performed during the DB commit.
Every function in this module is called from within a Supabase RPC/SQL transaction
managed by the commit orchestrator.

Design rules:
  - Each function is a pure Supabase-py wrapper (synchronous). The orchestrator
    calls them via asyncio.to_thread() inside the transaction block.
  - ON CONFLICT … DO UPDATE (upsert) semantics are used everywhere to guarantee
    idempotency. Re-running a commit for the same phone must produce the same state.
  - "safe_upsert" means: if the row already exists, UPDATE non-PK columns.
    Junction tables: ON CONFLICT DO NOTHING — the row either exists or is inserted.
  - Never DELETE existing junction rows before INSERT. Only add what is new.
    This prevents unintended data loss if only some fields are re-committed.
  - All functions return the PK of the affected row (or a count for junction tables).
  - The commit orchestrator keeps "tables_written" + "rows_inserted" / "rows_updated"
    counters updated after each function call.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.supabase_client import get_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SECTION 1 — phones + chipsets (central hub)
# ---------------------------------------------------------------------------

def upsert_chipset(data: dict[str, Any]) -> int:
    """
    Upsert a row in mobile_specs.chipsets.

    Conflict key: chipset_name (UNIQUE).
    If the chipset already exists, update all technical columns so a re-commit
    keeps the best data. Returns chipset_id.

    Args:
        data: Dict with keys matching chipsets column names.
              Must include 'chipset_name'. All other columns are optional.
    Returns:
        chipset_id (int)
    """
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("chipsets")
        .upsert(data, on_conflict="chipset_name", ignore_duplicates=False)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise RuntimeError(f"upsert_chipset: no row returned for chipset_name={data.get('chipset_name')!r}")
    chipset_id = rows[0]["chipset_id"]
    logger.debug("upsert_chipset: chipset_id=%d chipset_name=%r", chipset_id, data.get("chipset_name"))
    return chipset_id


def upsert_phone(data: dict[str, Any]) -> int:
    """
    Upsert a row in mobile_specs.phones.

    Idempotency: UNIQUE CONSTRAINT uq_phone_brand_model on (brand_id, model_name).
    Added by migration P10_schema_fixes.sql (replaces the old expression index
    on LOWER(model_name) which PostgREST could not use for ON CONFLICT).
    Uses ON CONFLICT (brand_id, model_name) DO UPDATE to refresh chipset_id and
    launch_date when a re-commit supplies better data.

    Args:
        data: Must include 'brand_id' and 'model_name'. 'chipset_id' and
              'launch_date' are optional.
    Returns:
        model_id (int)
    """
    client = get_client()

    result = (
        client
        .schema("mobile_specs")
        .table("phones")
        .upsert(data, on_conflict="brand_id,model_name", ignore_duplicates=False)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise RuntimeError(
            f"upsert_phone: no row returned for brand_id={data.get('brand_id')} "
            f"model_name={data.get('model_name')!r}"
        )
    model_id = rows[0]["model_id"]
    logger.debug("upsert_phone: model_id=%d model_name=%r", model_id, data.get("model_name"))
    return model_id


def fetch_chipset_by_name(chipset_name: str) -> dict | None:
    """
    Looks up a chipset row in mobile_specs.chipsets by exact chipset_name match.

    Called by the normalizer's chipset deduplication step (Change 6b).
    If found, returns the full row dict so the normalizer can overwrite the
    extracted chipset block with DB-canonical values before gap analysis runs.

    Returns None if the chipset is not yet in the database.
    """
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("chipsets")
        .select("*")
        .eq("chipset_name", chipset_name.strip())
        .limit(1)
        .execute()
    )
    rows = result.data or []
    if not rows:
        logger.debug("fetch_chipset_by_name: chipset_name=%r not found in DB", chipset_name)
        return None
    logger.debug("fetch_chipset_by_name: found chipset_id=%d for chipset_name=%r", rows[0].get("chipset_id"), chipset_name)
    return rows[0]


def fetch_brand_id(brand_name: str) -> int | None:
    """
    Resolve brand_name → brand_id from public.brands.
    Case-insensitive match (ILIKE). Returns None if not found.
    """
    client = get_client()
    result = (
        client
        .schema("public")
        .table("brands")
        .select("brand_id")
        .ilike("brand_name", brand_name.strip())
        .limit(1)
        .execute()
    )
    rows = result.data or []
    if not rows:
        logger.warning("fetch_brand_id: brand_name=%r not found in public.brands", brand_name)
        return None
    return rows[0]["brand_id"]


# ---------------------------------------------------------------------------
# SECTION 2 — Variants
# ---------------------------------------------------------------------------

def upsert_variant(data: dict[str, Any]) -> int:
    """
    Upsert a row in mobile_specs.variant.

    Idempotency: UNIQUE CONSTRAINT uq_variant_identity on (model_id, ram_capacity,
    storage_capacity). Added by migration P10_schema_fixes.sql (replaces the old
    expression index on COALESCE columns, which PostgREST could not use for ON CONFLICT).

    Accepted trade-off: ram_type_id is NOT part of the conflict key. Two variants
    differing only in ram_type map to the same variant slot. The ram_type_id stored
    on the row reflects the last-committed value.

    ON CONFLICT DO UPDATE: refreshes launch_price, is_base_variant, virtual_ram_*.

    Args:
        data: Must include 'model_id', 'ram_capacity', 'storage_capacity'.
    Returns:
        variant_id (int)
    """
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("variant")
        .upsert(
            data,
            on_conflict="model_id,ram_capacity,storage_capacity",
            ignore_duplicates=False,
        )
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise RuntimeError(f"upsert_variant: no row returned. data={data!r}")
    variant_id = rows[0]["variant_id"]
    logger.debug("upsert_variant: variant_id=%d model_id=%d", variant_id, data.get("model_id", -1))
    return variant_id


# ---------------------------------------------------------------------------
# SECTION 3 — Displays
# ---------------------------------------------------------------------------

def upsert_display(data: dict[str, Any]) -> int:
    """
    Upsert a row in mobile_specs.phone_displays.

    Idempotency: (model_id, display_type, display_position).
    ON CONFLICT DO UPDATE: refresh all spec columns.

    Returns:
        display_id (int)
    """
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("phone_displays")
        .upsert(
            data,
            on_conflict="model_id,display_type,display_position",
            ignore_duplicates=False,
        )
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise RuntimeError(f"upsert_display: no row returned. data={data!r}")
    display_id = rows[0]["display_id"]
    logger.debug("upsert_display: display_id=%d model_id=%d type=%r", display_id, data.get("model_id", -1), data.get("display_type"))
    return display_id


def insert_display_feature(display_id: int, feature_id: int) -> bool:
    """
    Insert a junction row in phone_display_features.
    ON CONFLICT DO NOTHING — safe to call on re-commit.
    Returns True if inserted, False if already existed.
    """
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("phone_display_features")
        .upsert(
            {"display_id": display_id, "feature_id": feature_id},
            on_conflict="display_id,feature_id",
            ignore_duplicates=True,
        )
        .execute()
    )
    inserted = bool(result.data)
    logger.debug("insert_display_feature: display_id=%d feature_id=%d inserted=%s", display_id, feature_id, inserted)
    return inserted


# ---------------------------------------------------------------------------
# SECTION 4 — Body
# ---------------------------------------------------------------------------

def upsert_body_features(data: dict[str, Any]) -> int:
    """
    Upsert mobile_specs.body_features.
    Idempotency: model_id UNIQUE.
    Returns body_id.
    """
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("body_features")
        .upsert(data, on_conflict="model_id", ignore_duplicates=False)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise RuntimeError(f"upsert_body_features: no row returned. model_id={data.get('model_id')}")
    body_id = rows[0]["body_id"]
    logger.debug("upsert_body_features: body_id=%d model_id=%d", body_id, data.get("model_id", -1))
    return body_id


# ---------------------------------------------------------------------------
# SECTION 5 — Charging
# ---------------------------------------------------------------------------

def upsert_charging_specs(data: dict[str, Any]) -> int:
    """
    Upsert mobile_specs.charging_specs.
    Idempotency: model_id UNIQUE.
    Returns charging_id.
    """
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("charging_specs")
        .upsert(data, on_conflict="model_id", ignore_duplicates=False)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise RuntimeError(f"upsert_charging_specs: no row returned. model_id={data.get('model_id')}")
    charging_id = rows[0]["charging_id"]
    logger.debug("upsert_charging_specs: charging_id=%d model_id=%d", charging_id, data.get("model_id", -1))
    return charging_id


def insert_charger_technology(model_id: int, technology_id: int) -> bool:
    """Junction: phone_charger_technologies. ON CONFLICT DO NOTHING."""
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("phone_charger_technologies")
        .upsert(
            {"model_id": model_id, "technology_id": technology_id},
            on_conflict="model_id,technology_id",
            ignore_duplicates=True,
        )
        .execute()
    )
    return bool(result.data)


# ---------------------------------------------------------------------------
# SECTION 6 — Audio
# ---------------------------------------------------------------------------

def upsert_audio(data: dict[str, Any]) -> int:
    """
    Upsert mobile_specs.audio.
    Idempotency: model_id UNIQUE.
    Returns audio_id.
    """
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("audio")
        .upsert(data, on_conflict="model_id", ignore_duplicates=False)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise RuntimeError(f"upsert_audio: no row returned. model_id={data.get('model_id')}")
    audio_id = rows[0]["audio_id"]
    logger.debug("upsert_audio: audio_id=%d model_id=%d", audio_id, data.get("model_id", -1))
    return audio_id


def insert_audio_codec(model_id: int, codec_id: int) -> bool:
    """Junction: phone_audio_codecs. ON CONFLICT DO NOTHING."""
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("phone_audio_codecs")
        .upsert(
            {"model_id": model_id, "codec_id": codec_id},
            on_conflict="model_id,codec_id",
            ignore_duplicates=True,
        )
        .execute()
    )
    return bool(result.data)


# ---------------------------------------------------------------------------
# SECTION 7 — Sensors
# ---------------------------------------------------------------------------

def upsert_sensors(data: dict[str, Any]) -> None:
    """
    Upsert mobile_specs.sensors.
    Idempotency: model_id is PRIMARY KEY.
    """
    client = get_client()
    (
        client
        .schema("mobile_specs")
        .table("sensors")
        .upsert(data, on_conflict="model_id", ignore_duplicates=False)
        .execute()
    )
    logger.debug("upsert_sensors: model_id=%d", data.get("model_id", -1))


def insert_phone_sensor(model_id: int, sensor_id: int) -> bool:
    """Junction: phone_sensors. ON CONFLICT DO NOTHING."""
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("phone_sensors")
        .upsert(
            {"model_id": model_id, "sensor_id": sensor_id},
            on_conflict="model_id,sensor_id",
            ignore_duplicates=True,
        )
        .execute()
    )
    return bool(result.data)


# ---------------------------------------------------------------------------
# SECTION 8 — Connectivity
# ---------------------------------------------------------------------------

def upsert_connectivity(data: dict[str, Any]) -> int:
    """
    Upsert mobile_specs.connectivity.
    Idempotency: model_id UNIQUE.
    Returns connectivity_id.
    """
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("connectivity")
        .upsert(data, on_conflict="model_id", ignore_duplicates=False)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise RuntimeError(f"upsert_connectivity: no row returned. model_id={data.get('model_id')}")
    connectivity_id = rows[0]["connectivity_id"]
    logger.debug("upsert_connectivity: connectivity_id=%d model_id=%d", connectivity_id, data.get("model_id", -1))
    return connectivity_id


def insert_wifi_technology(model_id: int, wifi_tech_id: int) -> bool:
    """Junction: phone_wifi_technologies. ON CONFLICT DO NOTHING."""
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("phone_wifi_technologies")
        .upsert(
            {"model_id": model_id, "wifi_tech_id": wifi_tech_id},
            on_conflict="model_id,wifi_tech_id",
            ignore_duplicates=True,
        )
        .execute()
    )
    return bool(result.data)


def insert_location_service(model_id: int, location_id: int) -> bool:
    """Junction: phone_location_services. ON CONFLICT DO NOTHING."""
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("phone_location_services")
        .upsert(
            {"model_id": model_id, "location_id": location_id},
            on_conflict="model_id,location_id",
            ignore_duplicates=True,
        )
        .execute()
    )
    return bool(result.data)


def insert_usb_feature(model_id: int, feature_id: int) -> bool:
    """Junction: phone_usb_features. ON CONFLICT DO NOTHING."""
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("phone_usb_features")
        .upsert(
            {"model_id": model_id, "feature_id": feature_id},
            on_conflict="model_id,feature_id",
            ignore_duplicates=True,
        )
        .execute()
    )
    return bool(result.data)


# ---------------------------------------------------------------------------
# SECTION 9 — Network
# ---------------------------------------------------------------------------

def upsert_network(data: dict[str, Any]) -> int:
    """
    Upsert mobile_specs.network.
    Idempotency: model_id UNIQUE.
    Returns network_id.
    """
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("network")
        .upsert(data, on_conflict="model_id", ignore_duplicates=False)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise RuntimeError(f"upsert_network: no row returned. model_id={data.get('model_id')}")
    network_id = rows[0]["network_id"]
    logger.debug("upsert_network: network_id=%d model_id=%d", network_id, data.get("model_id", -1))
    return network_id


def insert_network_band(model_id: int, band_id: int) -> bool:
    """Junction: phone_network_bands. ON CONFLICT DO NOTHING."""
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("phone_network_bands")
        .upsert(
            {"model_id": model_id, "band_id": band_id},
            on_conflict="model_id,band_id",
            ignore_duplicates=True,
        )
        .execute()
    )
    return bool(result.data)


def insert_cellular_feature(model_id: int, feature_id: int) -> bool:
    """Junction: phone_cellular_features. ON CONFLICT DO NOTHING."""
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("phone_cellular_features")
        .upsert(
            {"model_id": model_id, "feature_id": feature_id},
            on_conflict="model_id,feature_id",
            ignore_duplicates=True,
        )
        .execute()
    )
    return bool(result.data)


# ---------------------------------------------------------------------------
# SECTION 10 — Camera overview
# ---------------------------------------------------------------------------

def upsert_camera_overview(data: dict[str, Any]) -> int:
    """
    Upsert mobile_specs.camera_overview.
    Idempotency: model_id UNIQUE.
    Returns camera_overview_id.
    """
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("camera_overview")
        .upsert(data, on_conflict="model_id", ignore_duplicates=False)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise RuntimeError(f"upsert_camera_overview: no row returned. model_id={data.get('model_id')}")
    camera_overview_id = rows[0]["camera_overview_id"]
    logger.debug("upsert_camera_overview: camera_overview_id=%d model_id=%d", camera_overview_id, data.get("model_id", -1))
    return camera_overview_id


# ---------------------------------------------------------------------------
# SECTION 11 — Camera lens specs + stabilization junction
# ---------------------------------------------------------------------------

def upsert_camera_lens(data: dict[str, Any]) -> int:
    """
    Upsert mobile_specs.camera_lens_specs.

    Three conflict resolution paths:

    1. lens_type_id IS NOT NULL:
       ON CONFLICT (model_id, lens_type_id) DO UPDATE via CONSTRAINT uq_lens_typed.
       PostgREST can resolve this named constraint directly.

    2. lens_type_id IS NULL, fov IS NOT NULL:
       PostgreSQL has a partial unique index uq_lens_null_type on
       (model_id, fov) WHERE lens_type_id IS NULL — but PostgREST cannot
       pass the WHERE predicate in on_conflict. We use Option C instead:
       an explicit SELECT to find an existing row, then UPDATE or INSERT.
       This is safe and idempotent without requiring any extra constraint.

    3. lens_type_id IS NULL, fov IS NULL:
       Cannot safely deduplicate. INSERT with DO NOTHING + a logged WARNING.
       Phase 9 should prevent this case from reaching commit.

    Returns lens_id.
    """
    client = get_client()

    if data.get("lens_type_id") is not None:
        # ── Path 1: typed lens ── ON CONFLICT (model_id, lens_type_id) DO UPDATE
        result = (
            client
            .schema("mobile_specs")
            .table("camera_lens_specs")
            .upsert(
                data,
                on_conflict="model_id,lens_type_id",
                ignore_duplicates=False,
            )
            .execute()
        )

    elif data.get("fov") is not None:
        # ── Path 2: NULL-typed lens with FOV ── SELECT then UPDATE or INSERT
        # PostgREST cannot resolve a partial index via on_conflict, so we
        # implement the upsert semantics manually (C3-RESIDUAL fix, Option C).
        model_id = data["model_id"]
        fov_val = data["fov"]

        existing = (
            client
            .schema("mobile_specs")
            .table("camera_lens_specs")
            .select("lens_id")
            .eq("model_id", model_id)
            .eq("fov", fov_val)
            .is_("lens_type_id", "null")
            .limit(1)
            .execute()
        )

        if existing.data:
            # Row exists — UPDATE non-key columns only
            lens_id_existing = existing.data[0]["lens_id"]
            update_payload = {k: v for k, v in data.items()
                              if k not in ("model_id", "fov", "lens_type_id") and v is not None}
            if update_payload:
                (
                    client
                    .schema("mobile_specs")
                    .table("camera_lens_specs")
                    .update(update_payload)
                    .eq("lens_id", lens_id_existing)
                    .execute()
                )
            # Re-fetch to return the full row
            result = (
                client
                .schema("mobile_specs")
                .table("camera_lens_specs")
                .select("*")
                .eq("lens_id", lens_id_existing)
                .execute()
            )
        else:
            # Row does not exist — INSERT
            result = (
                client
                .schema("mobile_specs")
                .table("camera_lens_specs")
                .insert(data)
                .execute()
            )

    else:
        # ── Path 3: NULL-typed + no FOV — cannot safely deduplicate ──────────
        # Phase 9 Layer 1 should have blocked this. Log and INSERT with DO NOTHING.
        logger.warning(
            "upsert_camera_lens: lens_type_id=NULL and fov=NULL for model_id=%d. "
            "Cannot deduplicate — duplicate rows possible on re-commit. "
            "Improve source data to include lens_type or fov.",
            data.get("model_id", -1),
        )
        result = (
            client
            .schema("mobile_specs")
            .table("camera_lens_specs")
            .upsert(
                data,
                on_conflict="model_id,lens_type_id",
                ignore_duplicates=True,
            )
            .execute()
        )

    rows = result.data or []
    if not rows:
        raise RuntimeError(f"upsert_camera_lens: no row returned. data={data!r}")
    lens_id = rows[0]["lens_id"]
    logger.debug("upsert_camera_lens: lens_id=%d model_id=%d", lens_id, data.get("model_id", -1))
    return lens_id


def insert_lens_stabilization(lens_id: int, stabilization_id: int) -> bool:
    """Junction: lens_stabilization. ON CONFLICT DO NOTHING."""
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("lens_stabilization")
        .upsert(
            {"lens_id": lens_id, "stabilization_id": stabilization_id},
            on_conflict="lens_id,stabilization_id",
            ignore_duplicates=True,
        )
        .execute()
    )
    return bool(result.data)


def insert_camera_feature(model_id: int, feature_id: int) -> bool:
    """Junction: phone_camera_features. ON CONFLICT DO NOTHING."""
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("phone_camera_features")
        .upsert(
            {"model_id": model_id, "feature_id": feature_id},
            on_conflict="model_id,feature_id",
            ignore_duplicates=True,
        )
        .execute()
    )
    return bool(result.data)


# ---------------------------------------------------------------------------
# SECTION 12 — OS & Security
# ---------------------------------------------------------------------------

def upsert_os_and_security(data: dict[str, Any]) -> int:
    """
    Upsert mobile_specs.os_and_security.
    Idempotency: model_id UNIQUE.
    Returns os_security_id.
    """
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("os_and_security")
        .upsert(data, on_conflict="model_id", ignore_duplicates=False)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise RuntimeError(f"upsert_os_and_security: no row returned. model_id={data.get('model_id')}")
    os_security_id = rows[0]["os_security_id"]
    logger.debug("upsert_os_and_security: os_security_id=%d model_id=%d", os_security_id, data.get("model_id", -1))
    return os_security_id


def insert_biometric(model_id: int, biometric_id: int) -> bool:
    """Junction: phone_biometrics. ON CONFLICT DO NOTHING."""
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("phone_biometrics")
        .upsert(
            {"model_id": model_id, "biometric_id": biometric_id},
            on_conflict="model_id,biometric_id",
            ignore_duplicates=True,
        )
        .execute()
    )
    return bool(result.data)


def insert_unlock_method(model_id: int, unlock_method_id: int) -> bool:
    """Junction: phone_unlock_methods. ON CONFLICT DO NOTHING."""
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("phone_unlock_methods")
        .upsert(
            {"model_id": model_id, "unlock_method_id": unlock_method_id},
            on_conflict="model_id,unlock_method_id",
            ignore_duplicates=True,
        )
        .execute()
    )
    return bool(result.data)


def insert_security_feature(model_id: int, security_feature_id: int) -> bool:
    """Junction: phone_security_features. ON CONFLICT DO NOTHING."""
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("phone_security_features")
        .upsert(
            {"model_id": model_id, "security_feature_id": security_feature_id},
            on_conflict="model_id,security_feature_id",
            ignore_duplicates=True,
        )
        .execute()
    )
    return bool(result.data)


# ---------------------------------------------------------------------------
# SECTION 13 — Certifications
# ---------------------------------------------------------------------------

def upsert_certifications(data: dict[str, Any]) -> int:
    """
    Upsert mobile_specs.certifications.
    Idempotency: model_id UNIQUE.
    Returns certification_id.
    """
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("certifications")
        .upsert(data, on_conflict="model_id", ignore_duplicates=False)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise RuntimeError(f"upsert_certifications: no row returned. model_id={data.get('model_id')}")
    certification_id = rows[0]["certification_id"]
    logger.debug("upsert_certifications: certification_id=%d model_id=%d", certification_id, data.get("model_id", -1))
    return certification_id


def insert_ip_rating(model_id: int, ip_rating_id: int) -> bool:
    """Junction: phone_ip_ratings. ON CONFLICT DO NOTHING."""
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("phone_ip_ratings")
        .upsert(
            {"model_id": model_id, "ip_rating_id": ip_rating_id},
            on_conflict="model_id,ip_rating_id",
            ignore_duplicates=True,
        )
        .execute()
    )
    return bool(result.data)


def insert_video_certification(model_id: int, video_cert_id: int) -> bool:
    """Junction: phone_video_certifications. ON CONFLICT DO NOTHING."""
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("phone_video_certifications")
        .upsert(
            {"model_id": model_id, "video_cert_id": video_cert_id},
            on_conflict="model_id,video_cert_id",
            ignore_duplicates=True,
        )
        .execute()
    )
    return bool(result.data)


def insert_audio_certification(model_id: int, audio_cert_id: int) -> bool:
    """Junction: phone_audio_certifications. ON CONFLICT DO NOTHING."""
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("phone_audio_certifications")
        .upsert(
            {"model_id": model_id, "audio_cert_id": audio_cert_id},
            on_conflict="model_id,audio_cert_id",
            ignore_duplicates=True,
        )
        .execute()
    )
    return bool(result.data)


# ---------------------------------------------------------------------------
# SECTION 14 — AI Capabilities
# ---------------------------------------------------------------------------

def upsert_ai_capabilities(data: dict[str, Any]) -> int:
    """
    Upsert mobile_specs.ai_capabilities.
    Idempotency: model_id UNIQUE.
    Returns ai_capability_id.
    """
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("ai_capabilities")
        .upsert(data, on_conflict="model_id", ignore_duplicates=False)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise RuntimeError(f"upsert_ai_capabilities: no row returned. model_id={data.get('model_id')}")
    ai_capability_id = rows[0]["ai_capability_id"]
    logger.debug("upsert_ai_capabilities: ai_capability_id=%d model_id=%d", ai_capability_id, data.get("model_id", -1))
    return ai_capability_id


def insert_ai_feature(model_id: int, ai_feature_id: int) -> bool:
    """Junction: phone_ai_features. ON CONFLICT DO NOTHING."""
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("phone_ai_features")
        .upsert(
            {"model_id": model_id, "ai_feature_id": ai_feature_id},
            on_conflict="model_id,ai_feature_id",
            ignore_duplicates=True,
        )
        .execute()
    )
    return bool(result.data)


# ---------------------------------------------------------------------------
# SECTION 15 — Extra features + In-the-box
# ---------------------------------------------------------------------------

def insert_extra_feature(model_id: int, extra_feature_id: int) -> bool:
    """Junction: phone_extra_features. ON CONFLICT DO NOTHING."""
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("phone_extra_features")
        .upsert(
            {"model_id": model_id, "extra_feature_id": extra_feature_id},
            on_conflict="model_id,extra_feature_id",
            ignore_duplicates=True,
        )
        .execute()
    )
    return bool(result.data)


def insert_box_content(model_id: int, box_item_id: int, item_specification: str | None, quantity: int) -> bool:
    """Junction: phone_box_contents. ON CONFLICT (model_id, box_item_id) DO NOTHING."""
    client = get_client()
    data: dict[str, Any] = {
        "model_id": model_id,
        "box_item_id": box_item_id,
        "quantity": quantity,
    }
    if item_specification:
        data["item_specification"] = item_specification

    result = (
        client
        .schema("mobile_specs")
        .table("phone_box_contents")
        .upsert(
            data,
            on_conflict="model_id,box_item_id",
            ignore_duplicates=True,
        )
        .execute()
    )
    return bool(result.data)


# ---------------------------------------------------------------------------
# SECTION 16a — Video capabilities
# ---------------------------------------------------------------------------

def upsert_video_capabilities(data: dict[str, Any]) -> int:
    """
    Upsert a row in mobile_specs.video_capabilities.

    Idempotency: model_id UNIQUE.
    ON CONFLICT DO UPDATE: refresh all spec columns so a re-commit keeps the
    best extracted values.

    Args:
        data: Dict with keys matching video_capabilities column names.
              Must include 'model_id'.
    Returns:
        video_capabilities_id (int)
    """
    client = get_client()
    result = (
        client
        .schema("mobile_specs")
        .table("video_capabilities")
        .upsert(data, on_conflict="model_id", ignore_duplicates=False)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise RuntimeError(
            f"upsert_video_capabilities: no row returned. model_id={data.get('model_id')}"
        )
    vc_id = rows[0]["video_capabilities_id"]
    logger.debug(
        "upsert_video_capabilities: video_capabilities_id=%d model_id=%d",
        vc_id, data.get("model_id", -1),
    )
    return vc_id


# ---------------------------------------------------------------------------
# SECTION 16 — pipeline.url_registry status update
# ---------------------------------------------------------------------------

def mark_url_registry_stored(url_registry_id: int) -> None:
    """
    SET url_registry.status = 'stored_mainDB' after a successful commit.
    This is the final write in the commit sequence.
    """
    client = get_client()
    (
        client
        .schema("pipeline")
        .table("url_registry")
        .update({"status": "stored_mainDB"})
        .eq("url_id", url_registry_id)
        .execute()
    )
    logger.info("mark_url_registry_stored: url_registry_id=%d → stored_mainDB", url_registry_id)


# ---------------------------------------------------------------------------
# SECTION 17 — pipeline.db_commit_runs audit log
# ---------------------------------------------------------------------------

def insert_commit_run(
    final_id: int,
    url_registry_id: int,
    session_id: int | None,
) -> int:
    """
    Insert a new pipeline.db_commit_runs row with status='running'.
    Returns commit_run_id.
    """
    client = get_client()
    result = (
        client
        .schema("pipeline")
        .table("db_commit_runs")
        .insert({
            "final_id": final_id,
            "url_registry_id": url_registry_id,
            "session_id": session_id,
            "status": "running",
            "tables_written": [],
            "rows_inserted": 0,
            "rows_updated": 0,
            "unresolved_fields": [],
        })
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise RuntimeError("insert_commit_run: failed to create db_commit_runs row")
    commit_run_id = rows[0]["commit_run_id"]
    logger.info("insert_commit_run: commit_run_id=%d final_id=%d", commit_run_id, final_id)
    return commit_run_id


def update_commit_run(
    commit_run_id: int,
    status: str,
    model_id: int | None = None,
    tables_written: list[str] | None = None,
    rows_inserted: int = 0,
    rows_updated: int = 0,
    unresolved_fields: list[str] | None = None,
    error_message: str | None = None,
) -> None:
    """
    Update a pipeline.db_commit_runs row on completion or failure.
    """
    from datetime import datetime, timezone

    client = get_client()
    patch: dict[str, Any] = {
        "status": status,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "rows_inserted": rows_inserted,
        "rows_updated": rows_updated,
    }
    if model_id is not None:
        patch["model_id"] = model_id
    if tables_written is not None:
        patch["tables_written"] = tables_written
    if unresolved_fields is not None:
        patch["unresolved_fields"] = unresolved_fields
    if error_message is not None:
        patch["error_message"] = error_message[:2000]

    (
        client
        .schema("pipeline")
        .table("db_commit_runs")
        .update(patch)
        .eq("commit_run_id", commit_run_id)
        .execute()
    )
    logger.info(
        "update_commit_run: commit_run_id=%d status=%s rows_inserted=%d",
        commit_run_id, status, rows_inserted,
    )
