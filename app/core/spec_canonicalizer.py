"""
spec_canonicalizer.py
Reorders a fully-assembled spec dict into the canonical display order
derived from My_data_schema.xlsx.

Called once in extraction_run_a.py after build_spec_json() returns partial_json
and before insert_extraction_output(). Every downstream table
(normalized_spec_json, enrichment, staging, admin UI) inherits the order.

Unknown keys are always appended — never silently dropped.
"""

from typing import Any


# ── Top-level section order ──────────────────────────────────────────────────

_SECTION_ORDER: list[str] = [
    "brand",
    "phone_identity",
    "variants",
    "body",
    "displays",
    "chipset",
    "performance_benchmarks",
    "camera_overview",
    "camera_lenses",
    "video_capabilities",
    "camera_features",
    "charging",
    "audio",
    "connectivity",
    "network",
    "sensors",
    "os_and_security",
    "certifications",
    "ai_capabilities",
    "extra_features",
    "in_the_box",
]


# ── Per-section field order ───────────────────────────────────────────────────

_FIELD_ORDER: dict[str, list[str]] = {
    "brand": [
        "brand_name",
    ],
    "phone_identity": [
        "model_name",
        "launch_date",
    ],
    "variants": [
        "is_base_variant",
        "launch_price",
        "ram_capacity",
        "ram_type",
        "ram_frequency",
        "virtual_ram_availability",
        "virtual_ram_size",
        "storage_capacity",
        "storage_type",
        "expandable_storage",
    ],
    "body": [
        "height",
        "width",
        "thickness",
        "height_folded",
        "width_folded",
        "thickness_folded",
        "weight",
        "build",
        "buttons",
        "colors",
        "has_stylus",
        "stylus_features",
        "other_features",
    ],
    "displays": [
        "display_type",
        "display_position",
        "panel_type",
        "colour_depth",
        "size_inch",
        "aspect_ratio",
        "resolution_width_px",
        "resolution_height_px",
        "refresh_rate",
        "brightness_hbm",
        "brightness_peak",
        "pwm_frequency",
        "screen_to_body_ratio",
        "screen_shape",
        "glass_protection",
        "display_features",
    ],
    "chipset": [
        "chipset_name",
        "cpu_architecture",
        "fabrication_node",
        "number_of_cores",
        "cpu_ultra_high_performance_cores",
        "cpu_high_performance_cores",
        "cpu_efficiency_cores",
        "cpu_clock_speed",
        "gpu_name",
        "gpu_cores",
        "gpu_clock_speed",
        "npu_details",
        "npu_tops",
    ],
    "performance_benchmarks": [
        "antutu_score",
        "antutu_version",
        "geekbench_single_core",
        "geekbench_multi_core",
        "geekbench_version",
        "three_d_mark_test",
        "three_d_mark_score",
        "cooling_system",
    ],
    "camera_overview": [
        "rear_camera_setup",
        "rear_camera_island_shape",
        "rear_camera_island_position",
        "front_camera_setup",
        "front_camera_shape",
        "front_camera_position",
        "flash",
    ],
    "camera_lenses": [
        "lens_type",
        "sensor_model",
        "sensor_type",
        "megapixels",
        "sensor_size_denominator",
        "pixel_size",
        "aperture",
        "focal_length",
        "fov",
        "optical_zoom_capacity",
        "digital_zoom_capacity",
        "autofocus_type",
        "stabilization",
        "is_macro_capable",
        "lens_features",
    ],
    "video_capabilities": [
        "rear_video_resolutions",
        "slow_motion_resolutions",
        "front_video_resolutions",
    ],
    # camera_features is a flat list — no inner field reordering
    "charging": [
        "battery_capacity",
        "battery_type",
        "charging_power",
        "charging_voltage",
        "charging_ampere",
        "cable_type",
        "wireless_charging",
        "wireless_charging_power",
        "wireless_charging_standard",
        "reverse_wired_charging",
        "reverse_wireless_charging",
        "reverse_wireless_charging_power",
        "proprietary_charging",
        "charger_technologies",
        "charger_in_box",
        "battery_and_charging_features",
    ],
    "audio": [
        "speaker_count",
        "speaker_positions",
        "microphone_count",
        "microphone_positions",
        "has_3_5mm_jack",
        "audio_codecs",
        "audio_features",
    ],
    "connectivity": [
        "wifi_standard",
        "wifi_hotspot",
        "wifi_technologies",
        "bluetooth_version",
        "location_services",
        "nfc",
        "uwb",
        "ir_blaster",
        "usb_standard",
        "usb_features",
    ],
    "network": [
        "number_of_sims",
        "sim_configuration",
        "esim_support",
        "sim_tray_position",
        "bands_2g",
        "bands_3g",
        "bands_4g",
        "bands_5g",
        "satellite_bands",
        "volte",
        "vo5g",
        "vowifi",
        "cellular_features",
    ],
    "sensors": [
        "fingerprint_sensor",
        "other_sensors",
    ],
    "os_and_security": [
        "os_name",
        "ui_skin",
        "os_update_years",
        "security_update_years",
        "biometrics",
        "unlock_methods",
        "security_features",
    ],
    "certifications": [
        "ip_ratings",
        "sar_head",
        "sar_body",
        "widevine_support",
        "widevine_level",
        "video_certifications",
        "audio_certifications",
        "bis_certification",
        "other_certifications",
    ],
    "ai_capabilities": [
        "ai_system",
        "processing_type",
        "ai_features",
    ],
    # extra_features is a flat list — no inner field reordering
    "in_the_box": [
        "item_name",
        "quantity",
        "item_specification",
    ],
}


# ── Core helpers ──────────────────────────────────────────────────────────────

def _reorder_dict(d: dict, field_order: list[str]) -> dict:
    """Return a new dict with keys in field_order first, then any extras."""
    ordered = {k: d[k] for k in field_order if k in d}
    extras = {k: v for k, v in d.items() if k not in ordered}
    return {**ordered, **extras}


def _reorder_section(section_key: str, value: Any) -> Any:
    """Recursively reorder a section value (dict, list-of-dicts, or passthrough)."""
    field_order = _FIELD_ORDER.get(section_key)
    if field_order is None or value is None:
        return value
    if isinstance(value, dict):
        return _reorder_dict(value, field_order)
    if isinstance(value, list):
        return [
            _reorder_dict(item, field_order) if isinstance(item, dict) else item
            for item in value
        ]
    return value


# ── Public API ────────────────────────────────────────────────────────────────

def canonicalize_spec(spec: dict) -> dict:
    """
    Return a new spec dict with sections and fields in canonical display order.

    Unknown top-level keys are appended at the end (future-proof — new sections
    added to the schema won't be silently dropped).

    Usage in extraction_run_a.py (after Step 7, before Step 8):
        from app.core.spec_canonicalizer import canonicalize_spec
        partial_json = canonicalize_spec(partial_json)
    """
    if not isinstance(spec, dict):
        return spec

    result: dict = {}

    # 1. Known sections in canonical order
    for section_key in _SECTION_ORDER:
        if section_key in spec:
            result[section_key] = _reorder_section(section_key, spec[section_key])

    # 2. Unknown sections appended (never silently dropped)
    for key, val in spec.items():
        if key not in result:
            result[key] = val

    return result
