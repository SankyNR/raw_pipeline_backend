"""
Pydantic Output Schema — Run A (Spec Extraction) — v5 Custom Gemini Architecture

SCHEMA RULES (must match extraction_templates_v4.md Part 1):
  - Non-null scalar fields use ExtractedString / ExtractedInt / ExtractedFloat / ExtractedBool
    with an optional _source tag containing EITHER raw_id OR raw_transcript_id (not both)
    and a verbatim evidence_text substring.
  - Junction table arrays (display_features, bands_5g, audio_codecs, etc.) are plain
    List[str] with NO _source wrapper.
  - Structural fields (display_type, display_position, lens_type, is_base_variant) are
    plain Python types with NO _source wrapper.
  - String fields that are single free-text values (buttons, colors, etc.) use
    ExtractedString — NOT ExtractedStringList.
  - quantity in InTheBoxItemData is a plain int (no _source).
"""

from typing import Optional, List
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Source attribution
# ---------------------------------------------------------------------------

class SourceInfo(BaseModel):
    """
    Exactly one of raw_id / raw_transcript_id will be populated per field.
    raw_id           → scraped source (Firecrawl / OEM / aggregator)
    raw_transcript_id → YouTube transcript source
    """
    raw_id: Optional[int] = None
    raw_transcript_id: Optional[int] = None
    evidence_text: str


# ---------------------------------------------------------------------------
# Extracted scalar wrappers  (value + optional _source)
# ---------------------------------------------------------------------------

class ExtractedString(BaseModel):
    value: Optional[str] = None
    source: Optional[SourceInfo] = Field(default=None, alias="_source")

    class Config:
        populate_by_name = True


class ExtractedInt(BaseModel):
    value: Optional[int] = None
    source: Optional[SourceInfo] = Field(default=None, alias="_source")

    class Config:
        populate_by_name = True


class ExtractedFloat(BaseModel):
    value: Optional[float] = None
    source: Optional[SourceInfo] = Field(default=None, alias="_source")

    class Config:
        populate_by_name = True


class ExtractedBool(BaseModel):
    value: Optional[bool] = None
    source: Optional[SourceInfo] = Field(default=None, alias="_source")

    class Config:
        populate_by_name = True


# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------

class BrandData(BaseModel):
    brand_name: Optional[ExtractedString] = None


class PhoneIdentityData(BaseModel):
    model_name: Optional[ExtractedString] = None
    launch_date: Optional[ExtractedString] = None


class ChipsetData(BaseModel):
    chipset_name: Optional[ExtractedString] = None
    cpu_architecture: Optional[ExtractedString] = None
    fabrication_node: Optional[ExtractedInt] = None
    number_of_cores: Optional[ExtractedInt] = None
    cpu_high_performance_cores: Optional[ExtractedString] = None  # Change 1c: renamed from cpu_high_performance
    cpu_performance_cores: Optional[ExtractedString] = None
    cpu_efficiency_cores: Optional[ExtractedString] = None
    cpu_clock_speed: Optional[ExtractedFloat] = None
    gpu_name: Optional[ExtractedString] = None
    # gpu_architecture removed — Change 1d: redundant, already embedded in gpu_name
    gpu_unit_count: Optional[ExtractedInt] = None
    gpu_unit_type: Optional[ExtractedString] = None
    gpu_clock_speed: Optional[ExtractedFloat] = None
    npu_details: Optional[ExtractedString] = None
    npu_tops: Optional[ExtractedFloat] = None


class VariantData(BaseModel):
    # Scalar extracted fields
    ram_capacity: Optional[ExtractedInt] = None
    ram_type: Optional[ExtractedString] = None
    ram_frequency: Optional[ExtractedFloat] = None
    virtual_ram_availability: Optional[ExtractedBool] = None
    virtual_ram_size: Optional[ExtractedInt] = None
    storage_capacity: Optional[ExtractedInt] = None
    storage_type: Optional[ExtractedString] = None
    expandable_storage: Optional[ExtractedBool] = None
    launch_price: Optional[ExtractedFloat] = None
    # Structural field — no _source
    is_base_variant: Optional[bool] = None


class DisplayData(BaseModel):
    # Structural fields — plain types, no _source
    display_type: Optional[str] = None
    display_position: Optional[str] = None
    # Extracted scalar fields
    panel_type: Optional[ExtractedString] = None
    size_inch: Optional[ExtractedFloat] = None
    resolution_height_px: Optional[ExtractedInt] = None
    resolution_width_px: Optional[ExtractedInt] = None
    aspect_ratio: Optional[ExtractedString] = None
    colour_depth: Optional[ExtractedInt] = None
    refresh_rate: Optional[ExtractedInt] = None
    brightness_hbm: Optional[ExtractedInt] = None
    brightness_peak: Optional[ExtractedInt] = None
    pwm_frequency: Optional[ExtractedInt] = None
    screen_to_body_ratio: Optional[ExtractedFloat] = None
    screen_shape: Optional[ExtractedString] = None
    glass_protection: Optional[ExtractedString] = None
    # Junction array — plain list, no _source
    display_features: Optional[List[str]] = None


class BodyData(BaseModel):
    height: Optional[ExtractedFloat] = None
    width: Optional[ExtractedFloat] = None
    thickness: Optional[ExtractedFloat] = None
    height_folded: Optional[ExtractedFloat] = None
    width_folded: Optional[ExtractedFloat] = None
    thickness_folded: Optional[ExtractedFloat] = None
    weight: Optional[ExtractedFloat] = None
    build: Optional[ExtractedString] = None
    # Single free-text strings with evidence
    buttons: Optional[ExtractedString] = None
    colors: Optional[ExtractedString] = None
    has_stylus: Optional[ExtractedBool] = None
    stylus_features: Optional[ExtractedString] = None
    other_features: Optional[ExtractedString] = None


class ChargingData(BaseModel):
    battery_capacity: Optional[ExtractedInt] = None
    battery_type: Optional[ExtractedString] = None
    charging_voltage: Optional[ExtractedFloat] = None
    charging_ampere: Optional[ExtractedFloat] = None
    charging_power: Optional[ExtractedInt] = None
    cable_type: Optional[ExtractedString] = None
    proprietary_charging: Optional[ExtractedString] = None
    charger_in_box: Optional[ExtractedBool] = None
    wireless_charging: Optional[ExtractedBool] = None
    wireless_charging_power: Optional[ExtractedInt] = None
    wireless_charging_standard: Optional[ExtractedString] = None
    reverse_wireless_charging: Optional[ExtractedBool] = None
    reverse_wireless_charging_power: Optional[ExtractedInt] = None
    # Junction array — plain list, no _source
    charger_technologies: Optional[List[str]] = None
    # Single free-text string with evidence
    battery_and_charging_features: Optional[ExtractedString] = None


class AudioData(BaseModel):
    speaker_count: Optional[ExtractedInt] = None
    # Single free-text strings with evidence
    speaker_positions: Optional[ExtractedString] = None
    microphone_count: Optional[ExtractedInt] = None
    microphone_positions: Optional[ExtractedString] = None
    has_3_5mm_jack: Optional[ExtractedBool] = None
    audio_features: Optional[ExtractedString] = None
    # Junction array — plain list, no _source
    audio_codecs: Optional[List[str]] = None


class SensorsData(BaseModel):
    fingerprint_sensor: Optional[ExtractedString] = None
    # Junction array — plain list, no _source
    other_sensors: Optional[List[str]] = None


class ConnectivityData(BaseModel):
    wifi_standard: Optional[ExtractedString] = None
    # Junction arrays — plain lists, no _source
    wifi_technologies: Optional[List[str]] = None
    bluetooth_version: Optional[ExtractedString] = None
    usb_standard: Optional[ExtractedString] = None
    usb_features: Optional[List[str]] = None
    nfc: Optional[ExtractedBool] = None
    uwb: Optional[ExtractedBool] = None
    ir_blaster: Optional[ExtractedBool] = None
    wifi_hotspot: Optional[ExtractedBool] = None
    # Junction array — plain list, no _source
    location_services: Optional[List[str]] = None


class NetworkData(BaseModel):
    number_of_sims: Optional[ExtractedInt] = None
    esim_support: Optional[ExtractedBool] = None
    sim_configuration: Optional[ExtractedString] = None
    sim_tray_position: Optional[ExtractedString] = None
    # Single strings with evidence
    bands_2g: Optional[ExtractedString] = None
    bands_3g: Optional[ExtractedString] = None
    # Junction arrays — plain lists, no _source
    bands_4g: Optional[List[str]] = None
    bands_5g: Optional[List[str]] = None
    cellular_features: Optional[List[str]] = None
    volte: Optional[ExtractedBool] = None
    vo5g: Optional[ExtractedBool] = None
    vowifi: Optional[ExtractedBool] = None


class CameraOverviewData(BaseModel):
    rear_camera_setup: Optional[ExtractedString] = None
    rear_camera_island_shape: Optional[ExtractedString] = None
    rear_camera_island_position: Optional[ExtractedString] = None
    front_camera_setup: Optional[ExtractedString] = None
    front_camera_shape: Optional[ExtractedString] = None
    front_camera_position: Optional[ExtractedString] = None
    flash: Optional[ExtractedString] = None


class VideoCapabilitiesData(BaseModel):  # Change 1e: new class
    rear_video_resolutions: Optional[ExtractedString] = None
    front_video_resolutions: Optional[ExtractedString] = None
    slow_motion_resolutions: Optional[ExtractedString] = None


class PerformanceBenchmarksData(BaseModel):  # Change 1f: new class
    antutu_version: Optional[ExtractedString] = None
    antutu_score: Optional[ExtractedInt] = None
    geekbench_version: Optional[ExtractedString] = None
    geekbench_single_core: Optional[ExtractedInt] = None
    geekbench_multi_core: Optional[ExtractedInt] = None
    three_d_mark_test: Optional[ExtractedString] = None
    three_d_mark_score: Optional[ExtractedInt] = None
    cooling_system: Optional[ExtractedString] = None


class CameraLensData(BaseModel):
    # Structural field — plain string, no _source
    lens_type: Optional[str] = None
    # Extracted scalar fields
    sensor_model: Optional[ExtractedString] = None
    sensor_type: Optional[ExtractedString] = None
    megapixels: Optional[ExtractedFloat] = None
    sensor_size_denominator: Optional[ExtractedFloat] = None
    pixel_size: Optional[ExtractedFloat] = None
    aperture: Optional[ExtractedFloat] = None
    focal_length: Optional[ExtractedInt] = None
    fov: Optional[ExtractedInt] = None
    optical_zoom_capacity: Optional[ExtractedFloat] = None
    digital_zoom_capacity: Optional[ExtractedFloat] = None
    autofocus_type: Optional[ExtractedString] = None
    is_macro_capable: Optional[ExtractedBool] = None
    lens_features: Optional[ExtractedString] = None
    # Junction array — plain list, no _source
    stabilization: Optional[List[str]] = None
    # camera_features removed — Change 1a: moved to phone-level root field



class OsAndSecurityData(BaseModel):
    os_name: Optional[ExtractedString] = None
    ui_skin: Optional[ExtractedString] = None
    os_update_years: Optional[ExtractedInt] = None
    security_update_years: Optional[ExtractedInt] = None
    # Junction arrays — plain lists, no _source
    biometrics: Optional[List[str]] = None
    unlock_methods: Optional[List[str]] = None
    security_features: Optional[List[str]] = None


class CertificationsData(BaseModel):
    # Junction array — plain list, no _source
    ip_ratings: Optional[List[str]] = None
    sar_head: Optional[ExtractedFloat] = None
    sar_body: Optional[ExtractedFloat] = None
    widevine_support: Optional[ExtractedBool] = None
    widevine_level: Optional[ExtractedString] = None
    bis_certification: Optional[ExtractedBool] = None
    # Single free-text string with evidence
    other_certifications: Optional[ExtractedString] = None
    # Junction arrays — plain lists, no _source
    video_certifications: Optional[List[str]] = None
    audio_certifications: Optional[List[str]] = None


class AiCapabilitiesData(BaseModel):
    ai_system: Optional[ExtractedString] = None
    processing_type: Optional[ExtractedString] = None
    # Junction array — plain list, no _source
    ai_features: Optional[List[str]] = None


class InTheBoxItemData(BaseModel):
    item_name: Optional[ExtractedString] = None
    item_specification: Optional[ExtractedString] = None
    # Plain int — no _source
    quantity: int = 1


# ---------------------------------------------------------------------------
# Root schema
# ---------------------------------------------------------------------------

class RunAExtractionSchema(BaseModel):
    brand: Optional[BrandData] = None
    phone_identity: Optional[PhoneIdentityData] = None
    chipset: Optional[ChipsetData] = None
    variants: Optional[List[VariantData]] = None
    displays: Optional[List[DisplayData]] = None
    body: Optional[BodyData] = None
    charging: Optional[ChargingData] = None
    audio: Optional[AudioData] = None
    sensors: Optional[SensorsData] = None
    connectivity: Optional[ConnectivityData] = None
    network: Optional[NetworkData] = None
    camera_overview: Optional[CameraOverviewData] = None
    camera_lenses: Optional[List[CameraLensData]] = None
    video_capabilities: Optional[VideoCapabilitiesData] = None  # Change 1e: re-enabled (DB table exists)
    os_and_security: Optional[OsAndSecurityData] = None
    certifications: Optional[CertificationsData] = None
    ai_capabilities: Optional[AiCapabilitiesData] = None
    performance_benchmarks: Optional[PerformanceBenchmarksData] = None  # Change 1f: new
    # Junction array — plain list, no _source
    extra_features: Optional[List[str]] = None
    camera_features: Optional[List[str]] = None  # Change 1b: phone-level, not per-lens
    in_the_box: Optional[List[InTheBoxItemData]] = None

# ---------------------------------------------------------------------------
# Simplified schema — used for Gemini constrained generation.
# Strips _source wrappers from all scalar fields to stay within Gemini's
# FSM state limit. spec_json_builder.py handles plain values gracefully:
# _extract_value_and_source() returns (value, None) → no evidence entries,
# but partial_json is fully populated. Evidence quality can be improved later
# by re-introducing source attribution at the prompt level (not schema level).
# ---------------------------------------------------------------------------

class SimpleBrandData(BaseModel):
    brand_name: Optional[str] = None

class SimplePhoneIdentityData(BaseModel):
    model_name: Optional[str] = None
    launch_date: Optional[str] = None

class SimpleChipsetData(BaseModel):
    chipset_name: Optional[str] = None
    cpu_architecture: Optional[str] = None
    fabrication_node: Optional[int] = None
    number_of_cores: Optional[int] = None
    cpu_high_performance_cores: Optional[str] = None  # Change 1c: renamed from cpu_high_performance
    cpu_performance_cores: Optional[str] = None
    cpu_efficiency_cores: Optional[str] = None
    cpu_clock_speed: Optional[float] = None
    gpu_name: Optional[str] = None
    # gpu_architecture removed — Change 1d: redundant, already embedded in gpu_name
    gpu_unit_count: Optional[int] = None
    gpu_unit_type: Optional[str] = None
    gpu_clock_speed: Optional[float] = None
    npu_details: Optional[str] = None
    npu_tops: Optional[float] = None

class SimpleVariantData(BaseModel):
    ram_capacity: Optional[int] = None
    ram_type: Optional[str] = None
    ram_frequency: Optional[float] = None
    virtual_ram_availability: Optional[bool] = None
    virtual_ram_size: Optional[int] = None
    storage_capacity: Optional[int] = None
    storage_type: Optional[str] = None
    expandable_storage: Optional[bool] = None
    launch_price: Optional[float] = None
    is_base_variant: Optional[bool] = None

class SimpleDisplayData(BaseModel):
    display_type: Optional[str] = None
    display_position: Optional[str] = None
    panel_type: Optional[str] = None
    size_inch: Optional[float] = None
    resolution_height_px: Optional[int] = None
    resolution_width_px: Optional[int] = None
    aspect_ratio: Optional[str] = None
    colour_depth: Optional[int] = None
    refresh_rate: Optional[int] = None
    brightness_hbm: Optional[int] = None
    brightness_peak: Optional[int] = None
    pwm_frequency: Optional[int] = None
    screen_to_body_ratio: Optional[float] = None
    screen_shape: Optional[str] = None
    glass_protection: Optional[str] = None
    display_features: Optional[List[str]] = None

class SimpleBodyData(BaseModel):
    height: Optional[float] = None
    width: Optional[float] = None
    thickness: Optional[float] = None
    height_folded: Optional[float] = None
    width_folded: Optional[float] = None
    thickness_folded: Optional[float] = None
    weight: Optional[float] = None
    build: Optional[str] = None
    buttons: Optional[str] = None
    colors: Optional[str] = None
    has_stylus: Optional[bool] = None
    stylus_features: Optional[str] = None
    other_features: Optional[str] = None

class SimpleChargingData(BaseModel):
    battery_capacity: Optional[int] = None
    battery_type: Optional[str] = None
    charging_voltage: Optional[float] = None
    charging_ampere: Optional[float] = None
    charging_power: Optional[int] = None
    cable_type: Optional[str] = None
    proprietary_charging: Optional[str] = None
    charger_in_box: Optional[bool] = None
    wireless_charging: Optional[bool] = None
    wireless_charging_power: Optional[int] = None
    wireless_charging_standard: Optional[str] = None
    reverse_wireless_charging: Optional[bool] = None
    reverse_wireless_charging_power: Optional[int] = None
    charger_technologies: Optional[List[str]] = None
    battery_and_charging_features: Optional[str] = None

class SimpleAudioData(BaseModel):
    speaker_count: Optional[int] = None
    speaker_positions: Optional[str] = None
    microphone_count: Optional[int] = None
    microphone_positions: Optional[str] = None
    has_3_5mm_jack: Optional[bool] = None
    audio_features: Optional[str] = None
    audio_codecs: Optional[List[str]] = None

class SimpleSensorsData(BaseModel):
    fingerprint_sensor: Optional[str] = None
    other_sensors: Optional[List[str]] = None

class SimpleConnectivityData(BaseModel):
    wifi_standard: Optional[str] = None
    wifi_technologies: Optional[List[str]] = None
    bluetooth_version: Optional[str] = None
    usb_standard: Optional[str] = None
    usb_features: Optional[List[str]] = None
    nfc: Optional[bool] = None
    uwb: Optional[bool] = None
    ir_blaster: Optional[bool] = None
    wifi_hotspot: Optional[bool] = None
    location_services: Optional[List[str]] = None

class SimpleNetworkData(BaseModel):
    number_of_sims: Optional[int] = None
    esim_support: Optional[bool] = None
    sim_configuration: Optional[str] = None
    sim_tray_position: Optional[str] = None
    bands_2g: Optional[str] = None
    bands_3g: Optional[str] = None
    bands_4g: Optional[List[str]] = None
    bands_5g: Optional[List[str]] = None
    cellular_features: Optional[List[str]] = None
    volte: Optional[bool] = None
    vo5g: Optional[bool] = None
    vowifi: Optional[bool] = None

class SimpleCameraOverviewData(BaseModel):
    rear_camera_setup: Optional[str] = None
    rear_camera_island_shape: Optional[str] = None
    rear_camera_island_position: Optional[str] = None
    front_camera_setup: Optional[str] = None
    front_camera_shape: Optional[str] = None
    front_camera_position: Optional[str] = None
    flash: Optional[str] = None

class SimpleVideoCapabilitiesData(BaseModel):  # Change 1e: new class
    rear_video_resolutions: Optional[str] = None
    front_video_resolutions: Optional[str] = None
    slow_motion_resolutions: Optional[str] = None


class SimplePerformanceBenchmarksData(BaseModel):  # Change 1f: new class
    antutu_version: Optional[str] = None
    antutu_score: Optional[int] = None
    geekbench_version: Optional[str] = None
    geekbench_single_core: Optional[int] = None
    geekbench_multi_core: Optional[int] = None
    three_d_mark_test: Optional[str] = None
    three_d_mark_score: Optional[int] = None
    cooling_system: Optional[str] = None


class SimpleCameraLensData(BaseModel):
    lens_type: Optional[str] = None
    sensor_model: Optional[str] = None
    sensor_type: Optional[str] = None
    megapixels: Optional[float] = None
    sensor_size_denominator: Optional[float] = None
    pixel_size: Optional[float] = None
    aperture: Optional[float] = None
    focal_length: Optional[int] = None
    fov: Optional[int] = None
    optical_zoom_capacity: Optional[float] = None
    digital_zoom_capacity: Optional[float] = None
    autofocus_type: Optional[str] = None
    is_macro_capable: Optional[bool] = None
    lens_features: Optional[str] = None
    stabilization: Optional[List[str]] = None
    # camera_features removed — Change 1a: moved to phone-level root field

class SimpleOsAndSecurityData(BaseModel):
    os_name: Optional[str] = None
    ui_skin: Optional[str] = None
    os_update_years: Optional[int] = None
    security_update_years: Optional[int] = None
    biometrics: Optional[List[str]] = None
    unlock_methods: Optional[List[str]] = None
    security_features: Optional[List[str]] = None

class SimpleCertificationsData(BaseModel):
    ip_ratings: Optional[List[str]] = None
    sar_head: Optional[float] = None
    sar_body: Optional[float] = None
    widevine_support: Optional[bool] = None
    widevine_level: Optional[str] = None
    bis_certification: Optional[bool] = None
    other_certifications: Optional[str] = None
    video_certifications: Optional[List[str]] = None
    audio_certifications: Optional[List[str]] = None

class SimpleAiCapabilitiesData(BaseModel):
    ai_system: Optional[str] = None
    processing_type: Optional[str] = None
    ai_features: Optional[List[str]] = None

class SimpleInTheBoxItemData(BaseModel):
    item_name: Optional[str] = None
    item_specification: Optional[str] = None
    quantity: int = 1

class RunAExtractionSchemaSimple(BaseModel):
    brand: Optional[SimpleBrandData] = None
    phone_identity: Optional[SimplePhoneIdentityData] = None
    chipset: Optional[SimpleChipsetData] = None
    variants: Optional[List[SimpleVariantData]] = None
    displays: Optional[List[SimpleDisplayData]] = None
    body: Optional[SimpleBodyData] = None
    charging: Optional[SimpleChargingData] = None
    audio: Optional[SimpleAudioData] = None
    sensors: Optional[SimpleSensorsData] = None
    connectivity: Optional[SimpleConnectivityData] = None
    network: Optional[SimpleNetworkData] = None
    camera_overview: Optional[SimpleCameraOverviewData] = None
    camera_lenses: Optional[List[SimpleCameraLensData]] = None
    video_capabilities: Optional[SimpleVideoCapabilitiesData] = None  # Change 1e: re-enabled (DB table exists)
    os_and_security: Optional[SimpleOsAndSecurityData] = None
    certifications: Optional[SimpleCertificationsData] = None
    ai_capabilities: Optional[SimpleAiCapabilitiesData] = None
    performance_benchmarks: Optional[SimplePerformanceBenchmarksData] = None  # Change 1f: new
    extra_features: Optional[List[str]] = None
    camera_features: Optional[List[str]] = None  # Change 1b: phone-level, not per-lens
    in_the_box: Optional[List[SimpleInTheBoxItemData]] = None
