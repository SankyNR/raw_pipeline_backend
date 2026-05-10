"""
Run B Stage 1 Examples — Version 2

Teaches Stage 1 v2 key behaviors:
  - No spec facts extracted
  - No competitor comparisons extracted
  - Atomic observations only
  - Hard maximum of ~12 per transcript
"""


EXAMPLE_B2_STRICT_EXTRACTION = {
    "description": "Demonstrates v2 Stage 1: no specs, no comparisons, atomic, max ~10.",
    "input_excerpt": (
        "<source raw_transcript_id=\"24\">\n"
        "Comparing Motorola with OnePlus, Moto is definitely better for camera. "
        "I got 7 to 8 hours of screen on time with 5000 mAh battery. "
        "The Snapdragon 7s Gen 2 handles daily tasks very smoothly, no stutters. "
        "It has WiFi 6 and NFC which OnePlus does not have. "
        "The display is 144Hz which is higher than OnePlus 120Hz. "
        "In direct sunlight the display is clearly readable. "
        "The suede finish gets dirty really fast. "
        "Speaker sounds very loud and rich with excellent spaciousness. "
        "OnePlus has a better vibration motor for haptics. "
        "</source>"
    ),
    "expected_output": [
        {
            "experience_text": "Battery provides 7-8 hours of screen-on time under typical usage.",
            "sentiment": "Positive",
            "evidence_quote": "I got 7 to 8 hours of screen on time with 5000 mAh battery.",
            "category": "Battery Life",
            "confidence": 0.95,
        },
        {
            "experience_text": "Daily performance is smooth with no stutters or lag.",
            "sentiment": "Positive",
            "evidence_quote": "The Snapdragon 7s Gen 2 handles daily tasks very smoothly, no stutters.",
            "category": "Performance",
            "confidence": 0.90,
        },
        {
            "experience_text": "Display is clearly readable in direct sunlight.",
            "sentiment": "Positive",
            "evidence_quote": "In direct sunlight the display is clearly readable.",
            "category": "Display",
            "confidence": 0.90,
        },
        {
            "experience_text": "Suede back finish picks up dirt and grime with regular use.",
            "sentiment": "Negative",
            "evidence_quote": "The suede finish gets dirty really fast.",
            "category": "Build Quality",
            "confidence": 0.92,
        },
        {
            "experience_text": "Speakers produce loud, rich sound with excellent spatial quality.",
            "sentiment": "Positive",
            "evidence_quote": "Speaker sounds very loud and rich with excellent spaciousness.",
            "category": "Audio",
            "confidence": 0.90,
        },
        {
            "experience_text": "Haptic feedback is weak and lacks precision.",
            "sentiment": "Negative",
            "evidence_quote": "OnePlus has a better vibration motor for haptics.",
            "category": "Haptics",
            "confidence": 0.72,
        },
    ],
    # WiFi 6 / NFC / 5000mAh / 144Hz — spec facts, NOT extracted.
    # "better camera than OnePlus", "144Hz vs 120Hz" — comparisons, NOT extracted.
}


EXAMPLE_B2_SPEC_REJECTION = {
    "description": "Demonstrates rejection of spec facts while keeping the subjective experience.",
    "input_excerpt": (
        "<source raw_transcript_id=\"26\">\n"
        "The IP68 certification means it can survive in water. "
        "I used it confidently in heavy rain with no worries at all. "
        "It supports Wi-Fi 6, Bluetooth 5.2, and NFC. "
        "The in-display fingerprint sensor is fast and accurate. "
        "Battery is 5000mAh and it easily lasts all day with moderate use. "
        "The packaging is eco-friendly and pleasant on opening. "
        "</source>"
    ),
    "expected_output": [
        {
            "experience_text": "Using the phone confidently in heavy rain without any water damage concerns.",
            "sentiment": "Positive",
            "evidence_quote": "I used it confidently in heavy rain with no worries at all.",
            "category": "Durability",
            "confidence": 0.92,
        },
        {
            "experience_text": "In-display fingerprint scanner is fast and accurate.",
            "sentiment": "Positive",
            "evidence_quote": "The in-display fingerprint sensor is fast and accurate.",
            "category": "Performance",
            "confidence": 0.90,
        },
        {
            "experience_text": "Battery lasts a full day with moderate usage without issue.",
            "sentiment": "Positive",
            "evidence_quote": "it easily lasts all day with moderate use.",
            "category": "Battery Life",
            "confidence": 0.90,
        },
    ],
    # "IP68 certification means it can survive in water" — spec fact, NOT extracted.
    # "Wi-Fi 6, Bluetooth 5.2, NFC" — spec facts, NOT extracted.
    # "5000mAh" — spec fact inside battery sentence, NOT extracted standalone.
    # "eco-friendly packaging" — too trivial (under 10 meaningful words of observation).
}


RUN_B_STAGE1_EXAMPLES = [
    EXAMPLE_B2_STRICT_EXTRACTION,
    EXAMPLE_B2_SPEC_REJECTION,
]
