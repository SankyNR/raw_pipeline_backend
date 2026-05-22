"""
Task 0.3 — Field Mapping Config  (C4 fix: all field paths now match spec_template.yaml)

Central deterministic mapping from YAML field paths to lookup tables.

Imported by:
  - normalizer.py     — resolves string values to lookup PKs
  - commit_orchestrator.py — FK resolution at commit time
  - gap_analyzer.py   — identifies Type B junction-table gaps

Conventions:
  [*] wildcard: expands to specific indices [0], [1], etc. at runtime.
  SCALAR_FK_MAP:  one lookup table lookup per field value.
  ARRAY_FK_MAP:   each array element → one junction table row.
  JUNCTION_TABLE_FIELDS: alias of ARRAY_FK_MAP for gap_analyzer.py.
  RUN_C_CALCULATED_FIELDS: fields NEVER extracted by Run A — computed post-commit.
  NUMERIC_PRECISION_FIELDS: fields where aggregator beats transcript on conflicts.

SOURCE OF TRUTH:  app/config/spec_template.yaml  (output section).
All field paths below MUST exactly match keys in the spec_template.yaml output section.
If spec_template.yaml is updated, update this file in lockstep.

C4 fix history (2026-04-03):
  - cameras.*               → split into camera_overview.* and camera_lenses[*].*
  - cameras.lenses[*].*     → camera_lenses[*].*
  - cameras.camera_features → camera_lenses[*].camera_features
  - os_software.*           → os_and_security.*
  - security.*              → os_and_security.*
  - ai.*                    → ai_capabilities.*
  - displays[*].resolution_px_horizontal → displays[*].resolution_height_px
  - displays[*].resolution_px_vertical   → displays[*].resolution_width_px
  - charging.battery_capacity_mah        → charging.battery_capacity
  - cameras.lenses[*].autofocus         → camera_lenses[*].autofocus_type
  - body.*_mm / body.*_grams            → body.* (no units in field name)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Scalar FK fields
# One lookup table lookup per extracted string value.
# Field path → "schema.table.column"
# All paths match spec_template.yaml output section exactly.
# ---------------------------------------------------------------------------

SCALAR_FK_MAP: dict[str, str] = {
    # Variants
    "variants[*].ram_type":                    "mobile_specs.lookup_ram_types.ram_type",
    "variants[*].storage_type":                "mobile_specs.lookup_storage_types.storage_type",

    # Displays
    "displays[*].panel_type":                  "mobile_specs.lookup_panel_types.panel_type",
    "displays[*].glass_protection":            "mobile_specs.lookup_glass_protection.protection_name",
    "displays[*].screen_shape":                "mobile_specs.lookup_screen_shapes.shape_name",

    # Charging
    "charging.battery_type":                   "mobile_specs.lookup_battery_types.battery_type",
    "charging.cable_type":                     "mobile_specs.lookup_cable_types.cable_type",
    "charging.proprietary_charging":           "mobile_specs.lookup_proprietary_charging.technology_name",
    "charging.wireless_charging_standard":     "mobile_specs.lookup_wireless_charging_standards.standard_name",

    # Network
    "network.sim_configuration":               "mobile_specs.lookup_sim_configurations.configuration_name",

    # Camera overview  (C4 fix: was "cameras.*")
    "camera_overview.rear_camera_setup":       "mobile_specs.lookup_rear_camera_setup.setup_name",
    "camera_overview.front_camera_setup":      "mobile_specs.lookup_front_camera_setup.setup_name",
    "camera_overview.front_camera_shape":      "mobile_specs.lookup_front_camera_shape.shape_name",
    "camera_overview.flash":                   "mobile_specs.lookup_flash_types.flash_type",

    # Camera lenses  (C4 fix: was "cameras.lenses[*].*")
    "camera_lenses[*].lens_type":              "mobile_specs.lookup_lens_types.lens_type",
    "camera_lenses[*].autofocus_type":         "mobile_specs.lookup_autofocus_types.autofocus_type",

    # OS & Security  (C4 fix: was "os_software.*" and "security.*")
    "os_and_security.os_name":                 "mobile_specs.lookup_os_versions.os_name",
    "os_and_security.ui_skin":                 "mobile_specs.lookup_ui_skins.ui_skin_name",

    # Certifications
    "certifications.widevine_level":           "mobile_specs.lookup_widevine_levels.level_name",

    # Connectivity
    "connectivity.wifi_standard":              "mobile_specs.lookup_wifi_standards.wifi_standard",
    "connectivity.bluetooth_version":          "mobile_specs.lookup_bluetooth_versions.bluetooth_version",
    "connectivity.usb_standard":              "mobile_specs.lookup_usb_standards.usb_standard",

    # AI capabilities  (C4 fix: was "ai.*")
    "ai_capabilities.ai_system":              "mobile_specs.lookup_ai_systems.system_name",
    "ai_capabilities.processing_type":        "mobile_specs.lookup_ai_processing_types.processing_type",
}


# ---------------------------------------------------------------------------
# Array FK fields
# Each element in the extracted array → one junction table row.
# Field path → "schema.table.column"
# ---------------------------------------------------------------------------

ARRAY_FK_MAP: dict[str, str] = {
    # Displays
    "displays[*].display_features":            "mobile_specs.lookup_display_features.feature_name",

    # Audio
    "audio.audio_codecs":                      "mobile_specs.lookup_audio_codecs.codec_name",

    # Camera lenses  (C4 fix: was "cameras.lenses[*].*" / "cameras.camera_features")
    "camera_lenses[*].stabilization":          "mobile_specs.lookup_stabilization_types.stabilization_type",
    # camera_lenses[*].camera_features removed — Change 2e: moved to top-level camera_features

    # Phone-level camera features  (Change 2e: now a single shared junction, not per-lens)
    "camera_features":                         "mobile_specs.lookup_camera_features.canonical",

    # Connectivity
    "connectivity.wifi_technologies":          "mobile_specs.lookup_wifi_technologies.technology_name",
    "connectivity.location_services":          "mobile_specs.lookup_location_services.location_system",
    "connectivity.usb_features":               "mobile_specs.lookup_usb_features.feature_name",

    # Charging
    "charging.charger_technologies":           "mobile_specs.lookup_charger_technologies.technology_name",

    # Network
    "network.bands_5g":                        "mobile_specs.lookup_network_bands.band_name",
    "network.bands_4g":                        "mobile_specs.lookup_network_bands.band_name",
    "network.bands_satellite":                 "mobile_specs.lookup_network_bands.band_name",  # Migration 51
    "network.cellular_features":               "mobile_specs.lookup_cellular_features.feature_name",

    # OS & Security  (C4 fix: was "security.*")
    "os_and_security.biometrics":              "mobile_specs.lookup_biometric_types.biometric_name",
    "os_and_security.security_features":       "mobile_specs.lookup_security_features.feature_name",
    "os_and_security.unlock_methods":          "mobile_specs.lookup_unlock_methods.method_name",

    # Sensors
    "sensors.other_sensors":                   "mobile_specs.lookup_sensors.sensor_name",

    # AI capabilities  (C4 fix: was "ai.ai_features")
    "ai_capabilities.ai_features":             "mobile_specs.lookup_ai_features.feature_name",

    # Top-level arrays
    "extra_features":                          "mobile_specs.lookup_extra_features.feature_name",
    "in_the_box":                              "mobile_specs.lookup_box_items.item_name",
    "certifications.ip_ratings":               "mobile_specs.lookup_ip_ratings.rating_name",
    "certifications.video_certifications":     "mobile_specs.lookup_video_certifications.certification_name",
    "certifications.audio_certifications":     "mobile_specs.lookup_audio_certifications.certification_name",
}


# ---------------------------------------------------------------------------
# Junction table fields alias
# Used by gap_analyzer.py to identify Type B missing fields.
# ---------------------------------------------------------------------------

JUNCTION_TABLE_FIELDS: dict[str, str] = {k: v for k, v in ARRAY_FK_MAP.items()}


# ---------------------------------------------------------------------------
# Run C calculated fields
# NEVER extracted by Run A. Computed post-commit by the deterministic Python
# inference engine (Run C). If the LLM includes these, discard them.
#
# Rule: Discard any field in this set if it appears in the LLM output.
# ---------------------------------------------------------------------------

RUN_C_CALCULATED_FIELDS: set[str] = {
    "displays[*].ppi",                           # sqrt(h² + w²) / size_inch
    "camera_lenses[*].sensor_size_decimal",      # 1.0 / sensor_size_denominator
}


# ---------------------------------------------------------------------------
# Numeric precision fields  (C4 fix: paths now match spec_template.yaml)
# These fields involve exact measurements that transcripts routinely approximate
# in verbal reviews. Aggregator values take priority over transcript values for
# these paths, even though YouTube transcripts have higher general priority.
#
# This set is referred to in:
#   - extraction_orchestrator.py  (system prompt NUMERIC_PRECISION_FIELDS exception)
#   - conflict_resolver.py        (auto-resolution: aggregator beats transcript here)
# ---------------------------------------------------------------------------

NUMERIC_PRECISION_FIELDS: set[str] = {
    "displays[*].resolution_height_px",    # C4 fix: was resolution_px_horizontal
    "displays[*].resolution_width_px",     # C4 fix: was resolution_px_vertical
    "charging.battery_capacity",           # C4 fix: was battery_capacity_mah
    "camera_lenses[*].megapixels",         # C4 fix: was cameras.lenses[*].megapixels
    "body.height",                         # C4 fix: was body.length_mm
    "body.width",                          # C4 fix: was body.breadth_mm
    "body.thickness",                      # C4 fix: was body.height_mm
    "body.weight",                         # C4 fix: was body.weight_grams
    "network.bands_5g",
    "network.bands_4g",
}


# ---------------------------------------------------------------------------
# Site hint map
# Used by gap_analyzer.py to pick the best Gemini grounded search target.
# Field path → domain hint string (passed to call_gemini_grounded as site_hint).
# Chipset fields are resolved dynamically at runtime in gap_analyzer.py.
# ---------------------------------------------------------------------------

FIELD_SITE_HINTS: dict[str, str] = {
    "certifications.sar_head":       "tec.fptc.gov.in",
    "certifications.sar_body":       "tec.fptc.gov.in",
    "certifications.widevine_level": "gsmarena.com",
    "charging.charger_in_box":       "gsmarena.com",
    # chipset.npu_tops: resolved dynamically based on chipset vendor
}


# ---------------------------------------------------------------------------
# Field priority map
# Used by gap_analyzer.py to assign enrichment priority per missing field.
# "high" → sent to enrichment first; "medium" → standard queue; "skip" → omit
# ---------------------------------------------------------------------------

FIELD_PRIORITY_MAP: dict[str, str] = {
    "certifications.sar_head":       "high",
    "certifications.sar_body":       "high",
    "certifications.widevine_level": "high",
    "network.vo5g":                  "high",
    "charging.charger_in_box":       "high",
    "chipset.npu_details":           "medium",
    "chipset.npu_tops":              "medium",

    # ------------------------------------------------------------------
    # Camera — sensor name is the only P0 camera field.
    # All other camera sub-fields are medium or skip.
    # [*] wildcard is required — gap_analyzer uses _generic_path() which
    # converts concrete indices back to [*] before looking up here.
    # ------------------------------------------------------------------
    "camera_lenses[*].sensor_model":          "high",
    "camera_lenses[*].aperture":              "medium",
    "camera_lenses[*].focal_length":          "medium",
    "camera_lenses[*].autofocus_type":        "medium",

    # Camera micro-fields — unreliably sourced, low comparison value
    "camera_lenses[*].fov":                   "skip",
    "camera_lenses[*].sensor_type":           "skip",
    "camera_lenses[*].is_macro_capable":      "skip",
    "camera_lenses[*].digital_zoom_capacity": "skip",
    "camera_lenses[*].optical_zoom_capacity": "skip",
    "camera_lenses[*].lens_features":         "skip",
    # camera_lenses[*].camera_features removed — Change 2e: path moved to top-level
    "camera_features":                        "skip",  # Change 2e: phone-level, not per-lens

    # Camera overview — cosmetic visual metadata, not sourced by text enrichment
    "camera_overview.flash":                          "skip",
    "camera_overview.front_camera_position":          "skip",
    "camera_overview.front_camera_shape":             "skip",
    "camera_overview.rear_camera_island_position":    "skip",
    "camera_overview.rear_camera_island_shape":       "skip",

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    "displays[*].brightness_hbm":   "high",
    "displays[*].brightness_peak":  "high",

    # ------------------------------------------------------------------
    # Certifications
    # ------------------------------------------------------------------
    "certifications.video_certifications": "high",
    "certifications.bis_certification":    "skip",
    "certifications.other_certifications": "skip",

    # ------------------------------------------------------------------
    # Charging
    # ------------------------------------------------------------------
    "charging.wireless_charging":               "high",
    "charging.wireless_charging_power":         "high",
    "charging.charger_technologies":            "high",
    "charging.proprietary_charging":            "high",
    "charging.reverse_wireless_charging":       "medium",
    "charging.reverse_wireless_charging_power": "medium",
    "charging.reverse_charging":                "medium",

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------
    "network.volte":             "high",
    "network.sim_configuration": "high",
    "network.vowifi":            "medium",

    # Bands are always in spec sheets and extracted by Run A.
    # If they appear as gaps (extraction failure) re-run extraction — not enrichment.
    "network.bands_4g":       "skip",
    "network.bands_5g":       "skip",
    "network.bands_satellite": "skip",  # NTN/satellite bands — extraction-only

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------
    "connectivity.usb_standard":      "high",
    "connectivity.wifi_technologies": "medium",

    # ------------------------------------------------------------------
    # Body — form-factor-specific fields: skip at priority level.
    # (N/A prompt labels in enrichment_orchestrator.py are the secondary guard)
    # ------------------------------------------------------------------
    "body.height_folded":    "skip",
    "body.width_folded":     "skip",
    "body.thickness_folded": "skip",
    "body.stylus_features": "skip",  # managed by admin for known stylus phones
    "body.has_stylus":      "high",

    # ------------------------------------------------------------------
    # AI capabilities — low comparison value for current UI scope
    # ------------------------------------------------------------------
    "ai_capabilities.ai_system":       "skip",
    "ai_capabilities.processing_type": "skip",

    # ------------------------------------------------------------------
    # Extra features — too broad for reliable enrichment
    # ------------------------------------------------------------------
    "extra_features": "skip",

    # ------------------------------------------------------------------
    # Chipset GPU/clock fields — Migration 49/50: skip enrichment;
    # gpu_cores is admin-entered canonical data;
    # clock speeds are now phone-level facts written at commit time.
    # ------------------------------------------------------------------
    "chipset.gpu_cores":       "skip",
    "chipset.cpu_clock_speed": "skip",  # removed from chipset; written to phones table
    "chipset.gpu_clock_speed": "skip",  # removed from chipset; written to phones table

    # ------------------------------------------------------------------
    # Video capabilities — scalar free-text fields, skip enrichment
    # (Change 2c: new group)
    # ------------------------------------------------------------------
    "video_capabilities.rear_video_resolutions":  "skip",
    "video_capabilities.front_video_resolutions": "skip",
    "video_capabilities.slow_motion_resolutions":             "skip",

    # ------------------------------------------------------------------
    # Performance benchmarks — skip gap enrichment; sourced during extraction only
    # (Change 2d: new group)
    # ------------------------------------------------------------------
    "performance_benchmarks.antutu_version":       "skip",
    "performance_benchmarks.antutu_score":         "skip",
    "performance_benchmarks.geekbench_version":    "skip",
    "performance_benchmarks.geekbench_single_core": "skip",
    "performance_benchmarks.geekbench_multi_core": "skip",
    "performance_benchmarks.three_d_mark_test":    "skip",
    "performance_benchmarks.three_d_mark_score":   "skip",
    "performance_benchmarks.cooling_system":       "skip",
}

DEFAULT_FIELD_PRIORITY = "medium"
