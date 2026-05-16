"""
Gemini Extraction Few-Shot Examples — Run A (Spec Extraction) — Version 2

These examples teach the Gemini JSON-mode model:
  - The expected JSON structure (matching extraction_schema_run_a.py)
  - The `_source` attribution format (raw_id OR raw_transcript_id + evidence_text)
  - How to handle multiple displays (single-invariant position rule)
  - Junction arrays and structural fields (NO _source wrapper)
"""

EXAMPLE_FLAGSHIP = {
    "description": "Standard flagship phone with full spec coverage, showing structural fields and junction arrays without _source tags.",
    "input_excerpt": "<source raw_id=\"12\">\nSamsung Galaxy S25 Ultra\nLaunched in India on January 22, 2025\nPowered by the Qualcomm Snapdragon 8 Elite for Galaxy\nQualcomm Hexagon NPU for on-device AI\n45 TOPS on-device AI performance — updated: 72.5 TOPS confirmed\nRAM Plus virtual memory expansion supported\nUp to 8GB virtual RAM via RAM Plus\nStarting at ₹1,32,999 for 12GB/256GB\n₹1,54,999 for 12GB/512GB variant\n₹1,65,999 for 12GB/1TB variant\n6.9-inch Dynamic AMOLED 2X display\n1–120Hz adaptive refresh rate\n2000 nits HBM brightness\n2600 nits peak brightness\nProtected by Corning Gorilla Glass Armor\nGlass front (Gorilla Glass Armor 2), Titanium frame (Grade 5), Glass back (Gorilla Glass Victus 2)\nAvailable in Titanium Silverblue, Titanium Black, Titanium Gray, Titanium Whitesilver, Titanium Jetblack, Titanium Jadegreen, Titanium Pinkgold\nIncludes S Pen in the box\nS Pen included (non-Bluetooth), Air Commands support\n45W wired fast charging\nSuper Fast Charging 2.0 (45W)\n15W wireless charging supported\nQi2 wireless charging certified\nWireless PowerShare (reverse wireless charging) supported\nWireless PowerShare at up to 5W\nOptimised Charging, Overcharge Protection, Adaptive Battery\neSIM supported\nAndroid 15, One UI 7\n7 years of OS updates guaranteed\n7 years of security updates guaranteed\nOfficially sold in India with BIS certification\nPowered by Galaxy AI\nGalaxy AI features run on-device and via cloud\nSIM ejector tool included in package\nQuick Start Guide included\nSafety information booklet included\nUp to 100x Space Zoom (digital)\n</source>\n\n<source raw_id=\"15\">\nCPU Architecture: ARMv9.2 (Qualcomm Oryon V2)\nBuilt on TSMC 3nm process\nOcta-core (2x4.47 GHz + 6x3.53 GHz)\n2x4.47 GHz Oryon V2 Phoenix L + 6x3.53 GHz Oryon V2 Phoenix M\nMax clock speed 4.47 GHz\nGPU: Qualcomm Adreno 830\nAdreno 830 with 3 shader slices\nGPU clock: 1200 MHz\n12GB RAM / 256GB storage\n12GB RAM / 512GB storage\n12GB RAM / 1TB storage\nLPDDR5X RAM\nRAM frequency: 4.8 GHz\nUFS 4.0 internal storage\nNo microSD card slot\n3120 x 1440 pixels resolution\nAspect ratio: 19.5:9\n8-bit color depth (16M colors)\nPWM dimming frequency: 480Hz\nScreen-to-body ratio: ~92.3%\nFlat display, no curve\n162.8 x 77.6 x 8.2 mm\nWeight: 218 g\nPower button (Right), Volume controls (Right)\n5000 mAh battery\nNon-removable Li-Po 5000 mAh battery\nCharging: 9V/5A\nUSB Type-C 3.2 Gen 1\nStereo speakers (loudspeaker + earpiece)\nBottom speaker + earpiece stereo configuration\n2 microphones\nMicrophones at top and bottom\nNo 3.5mm headphone jack\nFingerprint (under display, ultrasonic)\nWi-Fi 802.11 a/b/g/n/ac/6e/7\nBluetooth 5.4\nUSB 3.2 Gen 1 Type-C\nNFC\nUltra-wideband (UWB) chip\nNo IR blaster\nWi-Fi hotspot supported\nDual SIM (Nano-SIM, dual stand-by)\nSIM tray located at the bottom\nGSM 850 / 900 / 1800 / 1900 - SIM 1 & SIM 2\nHSDPA 850 / 900 / 1700(AWS) / 1900 / 2100\nVoLTE supported\nVoNR (Voice over NR) supported\nWi-Fi calling supported\nQuad camera setup on rear\nIndividual lens layout, no camera island\nCamera modules in top-left corner\nSingle 12MP front camera\nCenter punch-hole front camera\nDual-tone LED flash\n200 MP, f/1.7, 24mm (wide), 1/1.3\", 0.6µm, multi-directional PDAF, OIS\nSamsung ISOCELL HP2 sensor\n85° field of view\nNo macro mode on main lens\n50 MP, f/1.9, 13mm (ultrawide), 1/2.52\", 0.7µm\nSamsung ISOCELL JN3 sensor\n50 MP ultra-wide\n120° field of view\nDual Pixel PDAF autofocus\nUltra-wide supports macro photography\n10 MP, f/2.4, 67mm (telephoto), 1/3.52\", 1.12µm\nSony IMX754 CMOS sensor\n10 MP telephoto\n3x optical zoom\nPDAF autofocus\nTelephoto not macro capable\n50 MP, f/3.4, 111mm (periscope telephoto), 1/2.52\", 0.7µm\nSony IMX854 CMOS sensor\n50 MP periscope telephoto\n5x optical zoom periscope\nPeriscope not macro capable\n12 MP, f/2.2, 26mm (wide), 1/2.82\", 0.7µm\nSamsung ISOCELL 3LU sensor\n12 MP front camera\nf/2.2 aperture front camera\n80° field of view front camera\nDual Pixel PDAF on front camera\nFront camera not macro capable\n8K@24/30fps, 4K@30/60/120fps, 1080p@30/60/120/240fps\nFront camera: 4K@30/60fps, 1080p@30/60fps\nSlow-mo: 4K@120fps, 1080p@240fps\nSAR head: 1.26 W/kg\nSAR body: 0.64 W/kg\nWidevine DRM: L1\n</source>\n\n<source raw_transcript_id=\"7\">\nSamsung does not include a charger in the box with the Indian unit this time\nyou get the Galaxy S25 Ultra handset in the box\nthe S Pen is included in the box — non-Bluetooth version\nyou do get the USB-C cable in the box\n</source>",
    "expected_output": {
        "brand": {
            "brand_name": {"value": "Samsung", "_source": {"raw_id": 12, "evidence_text": "Samsung Galaxy S25 Ultra"}}
        },
        "phone_identity": {
            "model_name": {"value": "Galaxy S25 Ultra", "_source": {"raw_id": 12, "evidence_text": "Samsung Galaxy S25 Ultra"}},
            "launch_date": {"value": "2025-01-22", "_source": {"raw_id": 12, "evidence_text": "Launched in India on January 22, 2025"}}
        },
        "chipset": {
            "chipset_name": {"value": "Qualcomm Snapdragon 8 Elite for Galaxy", "_source": {"raw_id": 12, "evidence_text": "Powered by the Qualcomm Snapdragon 8 Elite for Galaxy"}},
            "cpu_architecture": {"value": "ARMv9.2 (Qualcomm Oryon V2)", "_source": {"raw_id": 15, "evidence_text": "CPU Architecture: ARMv9.2 (Qualcomm Oryon V2)"}},
            "fabrication_node": {"value": 3, "_source": {"raw_id": 15, "evidence_text": "Built on TSMC 3nm process"}},
            "number_of_cores": {"value": 8, "_source": {"raw_id": 15, "evidence_text": "Octa-core (2x4.47 GHz + 6x3.53 GHz)"}},
            "cpu_ultra_high_performance_cores": {"value": "2x4.47 GHz Oryon V2 Phoenix L", "_source": {"raw_id": 15, "evidence_text": "2x4.47 GHz Oryon V2 Phoenix L + 6x3.53 GHz Oryon V2 Phoenix M"}},
            "cpu_high_performance_cores": {"value": "6x3.53 GHz Oryon V2 Phoenix M", "_source": {"raw_id": 15, "evidence_text": "2x4.47 GHz Oryon V2 Phoenix L + 6x3.53 GHz Oryon V2 Phoenix M"}},
            "cpu_efficiency_cores": None,
            "cpu_clock_speed": {"value": 4.47, "_source": {"raw_id": 15, "evidence_text": "Max clock speed 4.47 GHz"}},
            "gpu_name": {"value": "Qualcomm Adreno 830", "_source": {"raw_id": 15, "evidence_text": "GPU: Qualcomm Adreno 830"}},
            "gpu_unit_count": {"value": 3, "_source": {"raw_id": 15, "evidence_text": "Adreno 830 with 3 shader slices"}},
            "gpu_unit_type": {"value": "slices", "_source": {"raw_id": 15, "evidence_text": "Adreno 830 with 3 shader slices"}},
            "gpu_clock_speed": None,
            "npu_details": None,
            "npu_tops": None
        },
        "variants": [
            {
                "ram_capacity": {"value": 12, "_source": {"raw_id": 12, "evidence_text": "Starting at ₹1,32,999 for 12GB/256GB"}},
                "ram_type": {"value": "LPDDR5X", "_source": {"raw_id": 15, "evidence_text": "LPDDR5X RAM"}},
                "ram_frequency": {"value": 4.8, "_source": {"raw_id": 15, "evidence_text": "RAM frequency: 4.8 GHz"}},
                "virtual_ram_availability": {"value": True, "_source": {"raw_id": 12, "evidence_text": "RAM Plus virtual memory expansion supported"}},
                "virtual_ram_size": {"value": 8, "_source": {"raw_id": 12, "evidence_text": "Up to 8GB virtual RAM via RAM Plus"}},
                "storage_capacity": {"value": 256, "_source": {"raw_id": 12, "evidence_text": "Starting at ₹1,32,999 for 12GB/256GB"}},
                "storage_type": {"value": "UFS 4.0", "_source": {"raw_id": 15, "evidence_text": "UFS 4.0 internal storage"}},
                "expandable_storage": {"value": False, "_source": {"raw_id": 12, "evidence_text": "Starting at ₹1,32,999 for 12GB/256GB"}},
                "launch_price": {"value": 132999.00, "_source": {"raw_id": 12, "evidence_text": "Starting at ₹1,32,999 for 12GB/256GB"}},
                "is_base_variant": True
            },
            {
                "ram_capacity": {"value": 12, "_source": {"raw_id": 12, "evidence_text": "₹1,54,999 for 12GB/512GB variant"}},
                "ram_type": {"value": "LPDDR5X", "_source": {"raw_id": 15, "evidence_text": "LPDDR5X RAM"}},
                "ram_frequency": {"value": 4.8, "_source": {"raw_id": 15, "evidence_text": "RAM frequency: 4.8 GHz"}},
                "virtual_ram_availability": {"value": True, "_source": {"raw_id": 12, "evidence_text": "RAM Plus virtual memory expansion supported"}},
                "virtual_ram_size": {"value": 8, "_source": {"raw_id": 12, "evidence_text": "Up to 8GB virtual RAM via RAM Plus"}},
                "storage_capacity": {"value": 512, "_source": {"raw_id": 12, "evidence_text": "₹1,54,999 for 12GB/512GB variant"}},
                "storage_type": {"value": "UFS 4.0", "_source": {"raw_id": 15, "evidence_text": "UFS 4.0 internal storage"}},
                "expandable_storage": {"value": False, "_source": {"raw_id": 12, "evidence_text": "₹1,54,999 for 12GB/512GB variant"}},
                "launch_price": {"value": 154999.00, "_source": {"raw_id": 12, "evidence_text": "₹1,54,999 for 12GB/512GB variant"}},
                "is_base_variant": False
            },
            {
                "ram_capacity": {"value": 12, "_source": {"raw_id": 12, "evidence_text": "₹1,65,999 for 12GB/1TB variant"}},
                "ram_type": {"value": "LPDDR5X", "_source": {"raw_id": 15, "evidence_text": "LPDDR5X RAM"}},
                "ram_frequency": {"value": 4.8, "_source": {"raw_id": 15, "evidence_text": "RAM frequency: 4.8 GHz"}},
                "virtual_ram_availability": {"value": True, "_source": {"raw_id": 12, "evidence_text": "RAM Plus virtual memory expansion supported"}},
                "virtual_ram_size": {"value": 8, "_source": {"raw_id": 12, "evidence_text": "Up to 8GB virtual RAM via RAM Plus"}},
                "storage_capacity": {"value": 1024, "_source": {"raw_id": 12, "evidence_text": "₹1,65,999 for 12GB/1TB variant"}},
                "storage_type": {"value": "UFS 4.0", "_source": {"raw_id": 15, "evidence_text": "UFS 4.0 internal storage"}},
                "expandable_storage": {"value": False, "_source": {"raw_id": 12, "evidence_text": "₹1,65,999 for 12GB/1TB variant"}},
                "launch_price": {"value": 165999.00, "_source": {"raw_id": 12, "evidence_text": "₹1,65,999 for 12GB/1TB variant"}},
                "is_base_variant": False
            }
        ],
        "displays": [
            {
                "display_type": "Main",
                "display_position": "Primary",
                "panel_type": {"value": "Dynamic AMOLED 2X", "_source": {"raw_id": 12, "evidence_text": "6.9-inch Dynamic AMOLED 2X display"}},
                "size_inch": {"value": 6.9, "_source": {"raw_id": 12, "evidence_text": "6.9-inch Dynamic AMOLED 2X display"}},
                "resolution_height_px": {"value": 3120, "_source": {"raw_id": 15, "evidence_text": "3120 x 1440 pixels resolution"}},
                "resolution_width_px": {"value": 1440, "_source": {"raw_id": 15, "evidence_text": "3120 x 1440 pixels resolution"}},
                "aspect_ratio": {"value": "19.5:9", "_source": {"raw_id": 15, "evidence_text": "Aspect ratio: 19.5:9"}},
                "colour_depth": {"value": 8, "_source": {"raw_id": 15, "evidence_text": "8-bit color depth (16M colors)"}},
                "refresh_rate": {"value": 120, "_source": {"raw_id": 12, "evidence_text": "1–120Hz adaptive refresh rate"}},
                "brightness_hbm": {"value": 2000, "_source": {"raw_id": 12, "evidence_text": "2000 nits HBM brightness"}},
                "brightness_peak": {"value": 2600, "_source": {"raw_id": 12, "evidence_text": "2600 nits peak brightness"}},
                "pwm_frequency": {"value": 480, "_source": {"raw_id": 15, "evidence_text": "PWM dimming frequency: 480Hz"}},
                "screen_to_body_ratio": {"value": 92.3, "_source": {"raw_id": 15, "evidence_text": "Screen-to-body ratio: ~92.3%"}},
                "screen_shape": {"value": "Flat", "_source": {"raw_id": 15, "evidence_text": "Flat display, no curve"}},
                "glass_protection": {"value": "Gorilla Glass Armor", "_source": {"raw_id": 12, "evidence_text": "Protected by Corning Gorilla Glass Armor"}},
                "display_features": [
                    "HDR10+", "Dolby Vision", "Always-on Display", "Adaptive Refresh Rate",
                    "LTPO", "10-bit Color", "DCI-P3", "Anti-glare Coating",
                    "Eye Comfort Mode", "Vision Booster"
                ]
            }
        ],
        "body": {
            "height": {"value": 162.8, "_source": {"raw_id": 15, "evidence_text": "162.8 x 77.6 x 8.2 mm"}},
            "width": {"value": 77.6, "_source": {"raw_id": 15, "evidence_text": "162.8 x 77.6 x 8.2 mm"}},
            "thickness": {"value": 8.2, "_source": {"raw_id": 15, "evidence_text": "162.8 x 77.6 x 8.2 mm"}},
            "height_folded": None,
            "width_folded": None,
            "thickness_folded": None,
            "weight": {"value": 218.0, "_source": {"raw_id": 15, "evidence_text": "Weight: 218 g"}},
            "build": {"value": "Glass front (Gorilla Glass Armor 2), Titanium frame (Grade 5), Glass back (Gorilla Glass Victus 2)", "_source": {"raw_id": 12, "evidence_text": "Glass front (Gorilla Glass Armor 2), Titanium frame (Grade 5), Glass back (Gorilla Glass Victus 2)"}},
            "buttons": {"value": "Power button (Right), Volume controls (Right)", "_source": {"raw_id": 15, "evidence_text": "Power button (Right), Volume controls (Right)"}},
            "colors": {"value": "Titanium Silverblue, Titanium Black, Titanium Gray, Titanium Whitesilver, Titanium Jetblack, Titanium Jadegreen, Titanium Pinkgold", "_source": {"raw_id": 12, "evidence_text": "Available in Titanium Silverblue, Titanium Black, Titanium Gray, Titanium Whitesilver, Titanium Jetblack, Titanium Jadegreen, Titanium Pinkgold"}},
            "has_stylus": {"value": True, "_source": {"raw_id": 12, "evidence_text": "Includes S Pen in the box"}},
            "stylus_features": {"value": "S Pen included (non-Bluetooth), Air Commands support", "_source": {"raw_id": 12, "evidence_text": "S Pen included (non-Bluetooth), Air Commands support"}},
            "other_features": None
        },
        "charging": {
            "battery_capacity": {"value": 5000, "_source": {"raw_id": 15, "evidence_text": "5000 mAh battery"}},
            "battery_type": {"value": "Li-Po", "_source": {"raw_id": 15, "evidence_text": "Non-removable Li-Po 5000 mAh battery"}},
            "charging_voltage": {"value": 9.0, "_source": {"raw_id": 15, "evidence_text": "Charging: 9V/5A"}},
            "charging_ampere": {"value": 5.0, "_source": {"raw_id": 15, "evidence_text": "Charging: 9V/5A"}},
            "charging_power": {"value": 45, "_source": {"raw_id": 12, "evidence_text": "45W wired fast charging"}},
            "cable_type": {"value": "Type-C to Type-C", "_source": {"raw_id": 15, "evidence_text": "USB Type-C 3.2 Gen 1"}},
            "proprietary_charging": {"value": "Super Fast Charging 2.0", "_source": {"raw_id": 12, "evidence_text": "Super Fast Charging 2.0 (45W)"}},
            "charger_in_box": {"value": False, "_source": {"raw_transcript_id": 7, "evidence_text": "Samsung does not include a charger in the box with the Indian unit this time"}},
            "wireless_charging": {"value": True, "_source": {"raw_id": 12, "evidence_text": "15W wireless charging supported"}},
            "wireless_charging_power": {"value": 15, "_source": {"raw_id": 12, "evidence_text": "15W wireless charging supported"}},
            "wireless_charging_standard": {"value": "Qi2", "_source": {"raw_id": 12, "evidence_text": "Qi2 wireless charging certified"}},
            "reverse_wireless_charging": {"value": True, "_source": {"raw_id": 12, "evidence_text": "Wireless PowerShare (reverse wireless charging) supported"}},
            "reverse_wireless_charging_power": {"value": 5, "_source": {"raw_id": 12, "evidence_text": "Wireless PowerShare at up to 5W"}},
            "charger_technologies": ["GaN", "PD 3.0", "PPS"],
            "battery_and_charging_features": {"value": "Optimised Charging, Overcharge Protection, Adaptive Battery", "_source": {"raw_id": 12, "evidence_text": "Optimised Charging, Overcharge Protection, Adaptive Battery"}}
        },
        "audio": {
            "speaker_count": {"value": 2, "_source": {"raw_id": 15, "evidence_text": "Stereo speakers (loudspeaker + earpiece)"}},
            "speaker_positions": {"value": "Bottom + Earpiece (Stereo)", "_source": {"raw_id": 15, "evidence_text": "Bottom speaker + earpiece stereo configuration"}},
            "microphone_count": {"value": 2, "_source": {"raw_id": 15, "evidence_text": "2 microphones"}},
            "microphone_positions": {"value": "Top, Bottom", "_source": {"raw_id": 15, "evidence_text": "Microphones at top and bottom"}},
            "has_3_5mm_jack": {"value": False, "_source": {"raw_id": 15, "evidence_text": "No 3.5mm headphone jack"}},
            "audio_features": None,
            "audio_codecs": []
        },
        "sensors": {
            "fingerprint_sensor": {"value": "Under-display (Ultrasonic)", "_source": {"raw_id": 15, "evidence_text": "Fingerprint (under display, ultrasonic)"}},
            "other_sensors": [
                "Accelerometer", "Gyroscope", "Magnetometer", "Proximity Sensor",
                "Ambient Light Sensor", "Barometer", "Hall Sensor", "E-Compass",
                "Gravity Sensor", "Linear Acceleration", "Rotation Vector"
            ]
        },
        "connectivity": {
            "wifi_standard": {"value": "Wi-Fi 7 (802.11be)", "_source": {"raw_id": 15, "evidence_text": "Wi-Fi 802.11 a/b/g/n/ac/6e/7"}},
            "wifi_technologies": ["ETH320", "4096-QAM", "MLO", "MU-MIMO", "OFDMA", "WPA3"],
            "bluetooth_version": {"value": "5.4", "_source": {"raw_id": 15, "evidence_text": "Bluetooth 5.4"}},
            "usb_standard": {"value": "USB Type-C 3.2 Gen 1", "_source": {"raw_id": 15, "evidence_text": "USB 3.2 Gen 1 Type-C"}},
            "usb_features": ["OTG", "USB Tethering", "DisplayPort", "USB Power Delivery"],
            "nfc": {"value": True, "_source": {"raw_id": 15, "evidence_text": "NFC"}},
            "uwb": {"value": True, "_source": {"raw_id": 15, "evidence_text": "Ultra-wideband (UWB) chip"}},
            "ir_blaster": {"value": False, "_source": {"raw_id": 15, "evidence_text": "No IR blaster"}},
            "wifi_hotspot": {"value": True, "_source": {"raw_id": 15, "evidence_text": "Wi-Fi hotspot supported"}},
            "location_services": ["GPS", "GLONASS", "BDS", "Galileo", "QZSS"]
        },
        "network": {
            "number_of_sims": {"value": 2, "_source": {"raw_id": 15, "evidence_text": "Dual SIM (Nano-SIM, dual stand-by)"}},
            "esim_support": {"value": True, "_source": {"raw_id": 12, "evidence_text": "eSIM supported"}},
            "sim_configuration": {"value": "Dual SIM (Nano-SIM, dual stand-by)", "_source": {"raw_id": 15, "evidence_text": "Dual SIM (Nano-SIM, dual stand-by)"}},
            "sim_tray_position": {"value": "Bottom", "_source": {"raw_id": 15, "evidence_text": "SIM tray located at the bottom"}},
            "bands_2g": {"value": "GSM: B2/B3/B5/B8", "_source": {"raw_id": 15, "evidence_text": "GSM 850 / 900 / 1800 / 1900 - SIM 1 & SIM 2"}},
            "bands_3g": {"value": "UMTS: B1/B2/B4/B5/B8", "_source": {"raw_id": 15, "evidence_text": "HSDPA 850 / 900 / 1700(AWS) / 1900 / 2100"}},
            "bands_4g": [
                "B1", "B2", "B3", "B4", "B5", "B7", "B8", "B12", "B13", "B17",
                "B18", "B19", "B20", "B25", "B26", "B28", "B38", "B39", "B40", "B41", "B66"
            ],
            "bands_5g": [
                "n1", "n2", "n3", "n5", "n7", "n8", "n12", "n20", "n25", "n26",
                "n28", "n38", "n40", "n41", "n66", "n77", "n78"
            ],
            "cellular_features": [
                "5G SA", "5G NSA",
                "5G Dual Connectivity (EN-DC) (LTE + NR Dual Connectivity)",
                "Carrier Aggregation"
            ],
            "volte": {"value": True, "_source": {"raw_id": 15, "evidence_text": "VoLTE supported"}},
            "vo5g": {"value": True, "_source": {"raw_id": 15, "evidence_text": "VoNR (Voice over NR) supported"}},
            "vowifi": {"value": True, "_source": {"raw_id": 15, "evidence_text": "Wi-Fi calling supported"}}
        },
        "camera_overview": {
            "rear_camera_setup": {"value": "Quad", "_source": {"raw_id": 15, "evidence_text": "Quad camera setup on rear"}},
            "rear_camera_island_shape": {"value": "Individual Lenses", "_source": {"raw_id": 15, "evidence_text": "Individual lens layout, no camera island"}},
            "rear_camera_island_position": {"value": "Top Left", "_source": {"raw_id": 15, "evidence_text": "Camera modules in top-left corner"}},
            "front_camera_setup": {"value": "Single", "_source": {"raw_id": 15, "evidence_text": "Single 12MP front camera"}},
            "front_camera_shape": {"value": "Punch-hole", "_source": {"raw_id": 15, "evidence_text": "Center punch-hole front camera"}},
            "front_camera_position": {"value": "Center", "_source": {"raw_id": 15, "evidence_text": "Center punch-hole front camera"}},
            "flash": {"value": "Dual-tone LED", "_source": {"raw_id": 15, "evidence_text": "Dual-tone LED flash"}}
        },
        "camera_lenses": [
            {
                "lens_type": "Main",
                "sensor_model": {"value": "Samsung ISOCELL HP2", "_source": {"raw_id": 15, "evidence_text": "Samsung ISOCELL HP2 sensor"}},
                "sensor_type": {"value": "BSI CMOS", "_source": {"raw_id": 15, "evidence_text": "Samsung ISOCELL HP2 sensor"}},
                "megapixels": {"value": 200.0, "_source": {"raw_id": 15, "evidence_text": "200 MP, f/1.7, 24mm (wide)"}},
                "sensor_size_denominator": {"value": 1.3, "_source": {"raw_id": 15, "evidence_text": "1/1.3\""}},
                "pixel_size": {"value": 0.6, "_source": {"raw_id": 15, "evidence_text": "0.6µm"}},
                "aperture": {"value": 1.7, "_source": {"raw_id": 15, "evidence_text": "f/1.7"}},
                "focal_length": {"value": 24, "_source": {"raw_id": 15, "evidence_text": "24mm (wide)"}},
                "fov": {"value": 85, "_source": {"raw_id": 15, "evidence_text": "85° field of view"}},
                "optical_zoom_capacity": None,
                "digital_zoom_capacity": {"value": 100.0, "_source": {"raw_id": 12, "evidence_text": "Up to 100x Space Zoom (digital)"}},
                "autofocus_type": {"value": "Multi-directional PDAF", "_source": {"raw_id": 15, "evidence_text": "multi-directional PDAF"}},
                "is_macro_capable": {"value": False, "_source": {"raw_id": 15, "evidence_text": "No macro mode on main lens"}},
                "lens_features": None,
                "stabilization": ["OIS"]
            },
            {
                "lens_type": "Ultra-wide",
                "sensor_model": {"value": "Samsung ISOCELL JN3", "_source": {"raw_id": 15, "evidence_text": "Samsung ISOCELL JN3 sensor"}},
                "sensor_type": {"value": "BSI CMOS", "_source": {"raw_id": 15, "evidence_text": "Samsung ISOCELL JN3 sensor"}},
                "megapixels": {"value": 50.0, "_source": {"raw_id": 15, "evidence_text": "50 MP ultra-wide"}},
                "sensor_size_denominator": {"value": 2.52, "_source": {"raw_id": 15, "evidence_text": "1/2.52\""}},
                "pixel_size": {"value": 0.7, "_source": {"raw_id": 15, "evidence_text": "0.7µm"}},
                "aperture": {"value": 1.9, "_source": {"raw_id": 15, "evidence_text": "f/1.9"}},
                "focal_length": {"value": 13, "_source": {"raw_id": 15, "evidence_text": "13mm (ultrawide)"}},
                "fov": {"value": 120, "_source": {"raw_id": 15, "evidence_text": "120° field of view"}},
                "optical_zoom_capacity": None,
                "digital_zoom_capacity": None,
                "autofocus_type": {"value": "Dual Pixel PDAF", "_source": {"raw_id": 15, "evidence_text": "Dual Pixel PDAF autofocus"}},
                "is_macro_capable": {"value": True, "_source": {"raw_id": 15, "evidence_text": "Ultra-wide supports macro photography"}},
                "lens_features": None,
                "stabilization": ["EIS"]
            },
            {
                "lens_type": "Telephoto",
                "sensor_model": {"value": "Sony IMX754", "_source": {"raw_id": 15, "evidence_text": "Sony IMX754 CMOS sensor"}},
                "sensor_type": {"value": "CMOS", "_source": {"raw_id": 15, "evidence_text": "Sony IMX754 CMOS sensor"}},
                "megapixels": {"value": 10.0, "_source": {"raw_id": 15, "evidence_text": "10 MP telephoto"}},
                "sensor_size_denominator": {"value": 3.52, "_source": {"raw_id": 15, "evidence_text": "1/3.52\""}},
                "pixel_size": {"value": 1.12, "_source": {"raw_id": 15, "evidence_text": "1.12µm"}},
                "aperture": {"value": 2.4, "_source": {"raw_id": 15, "evidence_text": "f/2.4"}},
                "focal_length": {"value": 67, "_source": {"raw_id": 15, "evidence_text": "67mm (telephoto)"}},
                "fov": None,
                "optical_zoom_capacity": {"value": 3.0, "_source": {"raw_id": 15, "evidence_text": "3x optical zoom"}},
                "digital_zoom_capacity": None,
                "autofocus_type": {"value": "PDAF", "_source": {"raw_id": 15, "evidence_text": "PDAF autofocus"}},
                "is_macro_capable": {"value": False, "_source": {"raw_id": 15, "evidence_text": "Telephoto not macro capable"}},
                "lens_features": None,
                "stabilization": ["OIS"]
            },
            {
                "lens_type": "Periscope",
                "sensor_model": {"value": "Sony IMX854", "_source": {"raw_id": 15, "evidence_text": "Sony IMX854 CMOS sensor"}},
                "sensor_type": {"value": "CMOS", "_source": {"raw_id": 15, "evidence_text": "Sony IMX854 CMOS sensor"}},
                "megapixels": {"value": 50.0, "_source": {"raw_id": 15, "evidence_text": "50 MP periscope telephoto"}},
                "sensor_size_denominator": {"value": 2.52, "_source": {"raw_id": 15, "evidence_text": "1/2.52\""}},
                "pixel_size": {"value": 0.7, "_source": {"raw_id": 15, "evidence_text": "0.7µm"}},
                "aperture": {"value": 3.4, "_source": {"raw_id": 15, "evidence_text": "f/3.4"}},
                "focal_length": {"value": 111, "_source": {"raw_id": 15, "evidence_text": "111mm (periscope telephoto)"}},
                "fov": None,
                "optical_zoom_capacity": {"value": 5.0, "_source": {"raw_id": 15, "evidence_text": "5x optical zoom periscope"}},
                "digital_zoom_capacity": None,
                "autofocus_type": {"value": "PDAF", "_source": {"raw_id": 15, "evidence_text": "PDAF autofocus"}},
                "is_macro_capable": {"value": False, "_source": {"raw_id": 15, "evidence_text": "Periscope not macro capable"}},
                "lens_features": None,
                "stabilization": ["OIS"]
            },
            {
                "lens_type": "Front",
                "sensor_model": {"value": "Samsung ISOCELL 3LU", "_source": {"raw_id": 15, "evidence_text": "Samsung ISOCELL 3LU sensor"}},
                "sensor_type": {"value": "BSI CMOS", "_source": {"raw_id": 15, "evidence_text": "Samsung ISOCELL 3LU sensor"}},
                "megapixels": {"value": 12.0, "_source": {"raw_id": 15, "evidence_text": "12 MP front camera"}},
                "sensor_size_denominator": {"value": 2.82, "_source": {"raw_id": 15, "evidence_text": "1/2.82\""}},
                "pixel_size": {"value": 0.7, "_source": {"raw_id": 15, "evidence_text": "0.7µm"}},
                "aperture": {"value": 2.2, "_source": {"raw_id": 15, "evidence_text": "f/2.2 aperture front camera"}},
                "focal_length": {"value": 26, "_source": {"raw_id": 15, "evidence_text": "26mm (wide)"}},
                "fov": {"value": 80, "_source": {"raw_id": 15, "evidence_text": "80° field of view front camera"}},
                "optical_zoom_capacity": None,
                "digital_zoom_capacity": None,
                "autofocus_type": {"value": "Dual Pixel PDAF", "_source": {"raw_id": 15, "evidence_text": "Dual Pixel PDAF on front camera"}},
                "is_macro_capable": {"value": False, "_source": {"raw_id": 15, "evidence_text": "Front camera not macro capable"}},
                "lens_features": None,
                "stabilization": ["EIS"]
            }
        ],
        "video_capabilities": {
            "rear_video_resolutions": {"value": "8K@24/30fps, 4K@30/60fps, 1080p@30/60fps", "_source": {"raw_id": 15, "evidence_text": "8K@24/30fps, 4K@30/60/120fps, 1080p@30/60/120/240fps"}},
            "front_video_resolutions": {"value": "4K@30/60fps, 1080p@30/60fps", "_source": {"raw_id": 15, "evidence_text": "Front camera: 4K@30/60fps, 1080p@30/60fps"}},
            "slow_motion_resolutions": {"value": "4K@120fps, 1080p@240fps", "_source": {"raw_id": 15, "evidence_text": "Slow-mo: 4K@120fps, 1080p@240fps"}}
        },
        "performance_benchmarks": {
            "antutu_version": None,
            "antutu_score": None,
            "geekbench_version": None,
            "geekbench_single_core": None,
            "geekbench_multi_core": None,
            "three_d_mark_test": None,
            "three_d_mark_score": None,
            "cooling_system": None
        },
        "os_and_security": {
            "os_name": {"value": "Android 15", "_source": {"raw_id": 12, "evidence_text": "Android 15, One UI 7"}},
            "ui_skin": {"value": "One UI 7", "_source": {"raw_id": 12, "evidence_text": "Android 15, One UI 7"}},
            "os_update_years": {"value": 7, "_source": {"raw_id": 12, "evidence_text": "7 years of OS updates guaranteed"}},
            "security_update_years": {"value": 7, "_source": {"raw_id": 12, "evidence_text": "7 years of security updates guaranteed"}},
            "biometrics": ["In-display Fingerprint (Ultrasonic)", "Face Unlock"],
            "unlock_methods": ["Swipe", "Pattern", "PIN", "Password", "Biometric"],
            "security_features": ["Samsung Knox", "Knox Vault", "Secure Folder"]
        },
        "certifications": {
            "ip_ratings": ["IP68"],
            "sar_head": None,
            "sar_body": None,
            "widevine_support": None,
            "widevine_level": None,
            "bis_certification": None,
            "other_certifications": None,
            "video_certifications": ["HDR10+ Certified", "Netflix HDR", "Netflix HD"],
            "audio_certifications": ["Dolby Atmos Certified"]
        },
        "ai_capabilities": {
            "ai_system": {"value": "Galaxy AI", "_source": {"raw_id": 12, "evidence_text": "Powered by Galaxy AI"}},
            "processing_type": {"value": "Hybrid", "_source": {"raw_id": 12, "evidence_text": "Galaxy AI features run on-device and via cloud"}},
            "ai_features": [
                "Visual Search", "Live Translation", "Note Taking AI", "AI Writing Assistant",
                "Email Summarization", "Call Summarization", "Smart Reply", "AI Night Mode",
                "AI Zoom Enhancement", "AI Scene Detection", "AI Stabilization",
                "Object Removal", "Background Removal", "Subject Repositioning",
                "Photo Remaster", "Photo Unblur", "AI Photo Enhancement", "Photo Search",
                "AI Color Adaptation", "AI Brightness Optimization", "AI Eye Protection",
                "Screen Content Recognition", "Smart Charging", "Adaptive Battery",
                "App Power Management", "Adaptive Performance", "Thermal Management AI",
                "Resource Management AI", "AI Noise Cancellation", "Audio Eraser",
                "App Prediction", "Smart Search", "Notification Summary",
                "Priority Notifications", "Smart Album Organization"
            ]
        },
        "extra_features": [
            "Desktop Mode", "Wireless Desktop Mode", "Wireless File Sharing",
            "Advanced UI Customization", "Ecosystem Device Integration",
            "Screen Recorder", "One-Handed Mode", "Always-On Now Bar",
            "Modes and Routines", "Bypass Charging"
        ],
        "camera_features": [
            "Night Mode", "Portrait Mode", "RAW Capture", "Pro Mode", "HDR Photography",
            "Phase Detection Autofocus", "Object Tracking Autofocus",
            "High Resolution Photo", "Scene Detection", "Astrophotography",
            "Long Exposure", "Burst Mode", "Ultra-Wide Macro", "Panorama Mode",
            "Optical Zoom", "Super Resolution Zoom"
        ],
        "in_the_box": [
            {
                "item_name": {"value": "Handset", "_source": {"raw_transcript_id": 7, "evidence_text": "you get the Galaxy S25 Ultra handset in the box"}},
                "item_specification": None,
                "quantity": 1
            },
            {
                "item_name": {"value": "Stylus Pen", "_source": {"raw_transcript_id": 7, "evidence_text": "the S Pen is included in the box"}},
                "item_specification": {"value": "S Pen (Non-Bluetooth)", "_source": {"raw_transcript_id": 7, "evidence_text": "the S Pen is included in the box — non-Bluetooth version"}},
                "quantity": 1
            },
            {
                "item_name": {"value": "USB-C to C Cable", "_source": {"raw_transcript_id": 7, "evidence_text": "you do get the USB-C cable in the box"}},
                "item_specification": {"value": "Type-C to Type-C", "_source": {"raw_transcript_id": 7, "evidence_text": "you do get the USB-C cable in the box"}},
                "quantity": 1
            },
            {
                "item_name": {"value": "SIM Ejector Tool", "_source": {"raw_id": 12, "evidence_text": "SIM ejector tool included in package"}},
                "item_specification": None,
                "quantity": 1
            },
            {
                "item_name": {"value": "Quick Start Guide", "_source": {"raw_id": 12, "evidence_text": "Quick Start Guide included"}},
                "item_specification": None,
                "quantity": 1
            },
            {
                "item_name": {"value": "Safety Information", "_source": {"raw_id": 12, "evidence_text": "Safety information booklet included"}},
                "item_specification": None,
                "quantity": 1
            }
        ]
    }
}

EXAMPLE_FOLDABLE = {
    "description": "Foldable phone demonstrating multiple displays with correct structural identity (Inner/Cover vs Primary/Secondary) without _source tags on structural fields.",
    "input_excerpt": "<source raw_id=\"24\">\nDisplay Size\n- Main display: 6.96\" 1224p Super HD display\n120Hz refresh rate\nResolution: 2790 x 1224 pixels\nPeak HBM brightness reaches 1400 nits\nExternal display: 4.0\" pOLED display\nLTPO\nHDR10+\n</source>",
    "expected_output": {
        "displays": [
            {
                "display_type": "Inner",
                "display_position": "Primary",
                "size_inch": {"value": 6.96, "_source": {"raw_id": 24, "evidence_text": "Main display: 6.96\" 1224p Super HD display"}},
                "resolution_height_px": {"value": 2790, "_source": {"raw_id": 24, "evidence_text": "Resolution: 2790 x 1224 pixels"}},
                "resolution_width_px": {"value": 1224, "_source": {"raw_id": 24, "evidence_text": "Resolution: 2790 x 1224 pixels"}},
                "refresh_rate": {"value": 120, "_source": {"raw_id": 24, "evidence_text": "120Hz refresh rate"}},
                "brightness_hbm": {"value": 1400, "_source": {"raw_id": 24, "evidence_text": "Peak HBM brightness reaches 1400 nits"}},
                "display_features": ["LTPO", "HDR10+"]
            },
            {
                "display_type": "Cover",
                "display_position": "Secondary",
                "size_inch": {"value": 4.0, "_source": {"raw_id": 24, "evidence_text": "External display: 4.0\" pOLED display"}},
                "panel_type": {"value": "pOLED", "_source": {"raw_id": 24, "evidence_text": "External display: 4.0\" pOLED display"}},
                "display_features": []
            }
        ],
        "camera_lenses": [
            {
                "lens_type": "Main",
                "megapixels": {"value": 50.0, "_source": {"raw_id": 24, "evidence_text": "50MP main camera"}},
                "aperture": {"value": 1.8, "_source": {"raw_id": 24, "evidence_text": "f/1.8 aperture"}},
                "autofocus_type": {"value": "PDAF", "_source": {"raw_id": 24, "evidence_text": "PDAF autofocus"}},
                "sensor_model": None,
                "sensor_type": None,
                "sensor_size_denominator": None,
                "pixel_size": None,
                "focal_length": None,
                "fov": None,
                "optical_zoom_capacity": None,
                "digital_zoom_capacity": None,
                "is_macro_capable": None,
                "lens_features": None,
                "stabilization": ["OIS"],
                "front_camera_position_id": None
            },
            {
                "lens_type": "Front (Cover Display)",
                "megapixels": {"value": 10.0, "_source": {"raw_id": 24, "evidence_text": "10MP cover display selfie camera"}},
                "aperture": {"value": 2.2, "_source": {"raw_id": 24, "evidence_text": "f/2.2 aperture"}},
                "autofocus_type": None,
                "sensor_model": None,
                "sensor_type": None,
                "sensor_size_denominator": None,
                "pixel_size": None,
                "focal_length": None,
                "fov": None,
                "optical_zoom_capacity": None,
                "digital_zoom_capacity": None,
                "is_macro_capable": None,
                "lens_features": None,
                "stabilization": [],
                "front_camera_position_id": {"value": "Center", "_source": {"raw_id": 24, "evidence_text": "center punch-hole on cover display"}}
            },
            {
                "lens_type": "Front (Inner Display)",
                "megapixels": {"value": 10.0, "_source": {"raw_id": 24, "evidence_text": "10MP inner display selfie camera"}},
                "aperture": {"value": 2.2, "_source": {"raw_id": 24, "evidence_text": "f/2.2 aperture"}},
                "autofocus_type": None,
                "sensor_model": None,
                "sensor_type": None,
                "sensor_size_denominator": None,
                "pixel_size": None,
                "focal_length": None,
                "fov": None,
                "optical_zoom_capacity": None,
                "digital_zoom_capacity": None,
                "is_macro_capable": None,
                "lens_features": None,
                "stabilization": [],
                "front_camera_position_id": {"value": "Center", "_source": {"raw_id": 24, "evidence_text": "center punch-hole on inner display"}}
            }
        ]
    }
}

EXAMPLE_BUDGET = {
    "description": "Budget phone with basic variants, showing boolean structural field `is_base_variant` without _source tags.",
    "input_excerpt": "<source raw_id=\"18\">\nMemory (RAM)\n- available in 8GB RAM option | comes with 12GB RAM configuration\nBattery Size\n- 5000mAh\nIncludes 33W charger in the box\n6.5-inch IPS LCD display with 90Hz refresh rate.\nNo NFC support available.\n</source>",
    "expected_output": {
        "variants": [
            {
                "ram_capacity": {"value": 8, "_source": {"raw_id": 18, "evidence_text": "available in 8GB RAM option"}},
                "is_base_variant": True
            },
            {
                "ram_capacity": {"value": 12, "_source": {"raw_id": 18, "evidence_text": "comes with 12GB RAM configuration"}},
                "is_base_variant": False
            }
        ],
        "charging": {
            "battery_capacity": {"value": 5000, "_source": {"raw_id": 18, "evidence_text": "5000mAh"}},
            "charger_in_box": {"value": True, "_source": {"raw_id": 18, "evidence_text": "Includes 33W charger in the box"}},
            "charging_power": {"value": 33, "_source": {"raw_id": 18, "evidence_text": "Includes 33W charger in the box"}},
            "charger_technologies": []
        },
        "displays": [
            {
                "display_type": "Main",
                "display_position": "Primary",
                "panel_type": {"value": "IPS LCD", "_source": {"raw_id": 18, "evidence_text": "6.5-inch IPS LCD display with 90Hz refresh rate."}},
                "size_inch": {"value": 6.5, "_source": {"raw_id": 18, "evidence_text": "6.5-inch IPS LCD display with 90Hz refresh rate."}},
                "refresh_rate": {"value": 90, "_source": {"raw_id": 18, "evidence_text": "6.5-inch IPS LCD display with 90Hz refresh rate."}},
                "display_features": []
            }
        ],
        "connectivity": {
            "nfc": {"value": False, "_source": {"raw_id": 18, "evidence_text": "No NFC support available."}}
        }
    }
}

RUN_A_EXAMPLES = {
    "standard": [EXAMPLE_FLAGSHIP, EXAMPLE_BUDGET],
    "foldable": [EXAMPLE_FOLDABLE],
    "flippable": [EXAMPLE_FOLDABLE]
}


def run_coverage_check() -> None:
    """
    Startup guard: verifies that all top-level sections in RunAExtractionSchema
    are represented across the three example outputs.

    Does NOT block startup on failure — logs CRITICAL to allow investigation
    without taking the server offline. An AssertionError would be appropriate
    in a CI environment but is too aggressive for a live admin backend.
    """
    import logging
    from app.config.extraction_schema_run_a import RunAExtractionSchema

    logger = logging.getLogger(__name__)

    # All top-level field names defined in the schema
    schema_fields = set(RunAExtractionSchema.model_fields.keys())

    # Collect all top-level keys present across all examples
    covered: set[str] = set()
    for phone_type, examples in RUN_A_EXAMPLES.items():
        for ex in examples:
            covered.update(ex.get("expected_output", {}).keys())

    missing = schema_fields - covered
    if missing:
        logger.critical(
            "run_coverage_check: %d schema field(s) have NO example coverage: %s. "
            "Update RUN_A_EXAMPLES to include these sections.",
            len(missing), sorted(missing),
        )
    else:
        logger.info(
            "run_coverage_check: all %d schema sections have example coverage.",
            len(schema_fields),
        )

