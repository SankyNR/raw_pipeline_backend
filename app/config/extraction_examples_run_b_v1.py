"""
Gemini Extraction Few-Shot Examples — Run B (Experience Extraction) — Version 1

These examples teach the Gemini JSON-mode model:
  - How to extract subjective experiences (pros/cons/notes)
  - The correct output schema without _source wrappers
  - Handling colloquial transcript text correctly without paraphrasing evidence_quote
"""

EXAMPLE_B_DEDUP = {
    "description": "Demonstrates deduplicating multiple overlapping reviewer observations into a single clean experience.",
    "input_excerpt": "<source raw_transcript_id=\"10\">\nthe battery life is incredible. i was getting 8 hours of screen time. honestly the battery lasts forever.\n</source>",
    "expected_output": [
        {
            "experience_text": "Battery life is excellent, providing up to 8 hours of screen-on time.",
            "sentiment": "Positive",
            "evidence_quote": "i was getting 8 hours of screen time.",
            "confidence": 0.95,
            "category": "Battery Life"
        }
    ]
}

EXAMPLE_B_MIXED = {
    "description": "Demonstrates extracting an experience with a mixed sentiment where a feature has both clear pros and cons.",
    "input_excerpt": "<source raw_transcript_id=\"11\">\nthe main camera is super sharp during the day but really struggles with noise when the sun goes down.\n</source>",
    "expected_output": [
        {
            "experience_text": "Main camera performs very well in daylight but produces noisy images in low light.",
            "sentiment": "Mixed",
            "evidence_quote": "the main camera is super sharp during the day but really struggles with noise when the sun goes down.",
            "confidence": 0.9,
            "category": "Camera"
        }
    ]
}

EXAMPLE_B_MULTI = {
    "description": "Realistic transcript with multiple categories and experiences, written in casual, messy language.",
    "input_excerpt": "<source raw_transcript_id=\"12\">\nso the phone feels super snappy when opening apps and scrolling through twitter. however the battery drained completely by 4 pm which was a huge letdown. photos from the main lens look punchy and vibrant, really good for instagram. but man, this new ui update is so clunky and full of random pre-installed games.\n</source>",
    "expected_output": [
        {
            "experience_text": "Performance is fast and responsive during daily app usage and scrolling.",
            "sentiment": "Positive",
            "evidence_quote": "so the phone feels super snappy when opening apps and scrolling through twitter.",
            "confidence": 0.95,
            "category": "Performance"
        },
        {
            "experience_text": "Battery life is poor, draining quickly by late afternoon.",
            "sentiment": "Negative",
            "evidence_quote": "however the battery drained completely by 4 pm which was a huge letdown.",
            "confidence": 0.9,
            "category": "Battery Life"
        },
        {
            "experience_text": "Main camera takes vibrant and punchy photos suitable for social media.",
            "sentiment": "Positive",
            "evidence_quote": "photos from the main lens look punchy and vibrant, really good for instagram.",
            "confidence": 0.9,
            "category": "Camera"
        },
        {
            "experience_text": "Software interface is clunky and comes with excessive pre-installed bloatware.",
            "sentiment": "Negative",
            "evidence_quote": "but man, this new ui update is so clunky and full of random pre-installed games.",
            "confidence": 0.95,
            "category": "Software"
        }
    ]
}

RUN_B_EXAMPLES = [
    EXAMPLE_B_DEDUP,
    EXAMPLE_B_MIXED,
    EXAMPLE_B_MULTI
]
