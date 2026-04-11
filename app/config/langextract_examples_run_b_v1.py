"""
LangExtract Few-Shot Examples — Run B (Experience Extraction) — Version 1

PURPOSE
-------
These ExampleData objects teach the model how to extract subjective reviewer
observations from YouTube transcript text — NOT technical specs.

The extraction_class values here are experience categories (matching
pipeline.lookup_experience_categories.category_name), not spec schema sections.

VERSION PROTOCOL
----------------
Versioned in lockstep with the Run B prompt:
  langextract_examples_run_b_v1.py → schema_version="v1"
  langextract_examples_run_b_v2.py → schema_version="v2"  (when prompt changes)

Run B schema_version is recorded in experience_extraction_runs.extraction_schema_version.

CRITICAL RULES (same as Run A, adapted for transcript text)
-----------------------------------------------------------
1. ALL source texts MUST be real trimmed processed transcript content
   from raw_scraped_data or youtube_raw_transcript_data.processed_transcript_path.
   NEVER write synthetic transcript text. The model must learn the real register
   (colloquial speech, filler words, approximate numbers, reviewer voice).

2. Every extraction_text MUST be a verbatim substring of its ExampleData.text.
   Run the verbatim check at the bottom of this file.

3. The examples must collectively demonstrate:
   - Multiple experience categories (Thermal, Camera, Battery Life, etc.)
   - All four sentiments: Positive, Negative, Neutral, Mixed
   - Confidence levels (high/medium/low)
   - evidence_quote usage (verbatim reviewer sentence)

RUN B SCHEMA
------------
Each extraction produces one phone_experiences row. Attributes:
  experience_text  str              — clean 1–3 sentence summary (NOT verbatim)
  sentiment        str              — "Positive" | "Negative" | "Neutral" | "Mixed"
  evidence_quote   str | None       — verbatim reviewer sentence (immutable after insert)
  confidence       float            — 0.0–1.0
  category_name    str              — must match lookup_experience_categories.category_name

NOTE: category_name is resolved to category_id by the orchestrator before DB insert.
The extraction_class IS the category_name for Run B.

PLACEHOLDER STATUS
------------------
These stubs must be replaced with real transcript examples before Phase L5.
DO NOT use for production while text="PLACEHOLDER" appears in this file.
"""

import langextract as lx

# ---------------------------------------------------------------------------
# VERBATIM CHECK (run after filling examples, before Phase L5)
# ---------------------------------------------------------------------------
# for ex in RUN_B_EXAMPLES:
#     for e in ex.extractions:
#         assert e.extraction_text in ex.text, (
#             f"NOT VERBATIM: extraction_text={e.extraction_text[:60]!r}"
#         )
# print("Run B verbatim check PASSED.")

# ---------------------------------------------------------------------------
# Example 1 — Mixed sentiment English transcript (flagship phone)
# Source: [FILL IN — real trimmed processed transcript from a TrakinTech or TechWiser review]
# Phone:  [FILL IN — e.g. Samsung Galaxy S25 Ultra]
# Purpose: Demonstrate Positive + Negative + Mixed extractions from the same transcript.
#          Show evidence_quote = verbatim reviewer sentence.
# ---------------------------------------------------------------------------

EXAMPLE_B_FLAGSHIP = lx.data.ExampleData(
    # REPLACE: fetch real trimmed processed transcript content (processed_transcript_path).
    # Trim to 800–1,500 chars. Keep actual reviewer commentary. Remove SRT timestamps.
    text="PLACEHOLDER — replace with real trimmed processed transcript text (800–1,500 chars)",
    extractions=[
        # --- REPLACE ALL BELOW with real Run B extractions ---
        # extraction_class = experience category name (must match lookup table exactly)
        # extraction_text = verbatim substring from transcript text above
        # attributes = the structured experience data

        lx.data.Extraction(
            extraction_class="Thermal",             # Must match lookup_experience_categories.category_name
            extraction_text="PLACEHOLDER",          # Verbatim substring of text above
            attributes={
                "experience_text": "PLACEHOLDER",   # str — clean 1–3 sentence summary
                "sentiment":       "Negative",      # "Positive"|"Negative"|"Neutral"|"Mixed"
                "evidence_quote":  "PLACEHOLDER",   # str | None — exact reviewer words
                "confidence":      0.85,            # float 0.0–1.0
                "category_name":   "Thermal",       # Must match extraction_class
            },
        ),
        lx.data.Extraction(
            extraction_class="Camera",
            extraction_text="PLACEHOLDER",
            attributes={
                "experience_text": "PLACEHOLDER",
                "sentiment":       "Positive",
                "evidence_quote":  "PLACEHOLDER",
                "confidence":      0.90,
                "category_name":   "Camera",
            },
        ),
        lx.data.Extraction(
            extraction_class="Battery Life",
            extraction_text="PLACEHOLDER",
            attributes={
                "experience_text": "PLACEHOLDER",
                "sentiment":       "Positive",
                "evidence_quote":  "PLACEHOLDER",
                "confidence":      0.80,
                "category_name":   "Battery Life",
            },
        ),
        lx.data.Extraction(
            extraction_class="Display",
            extraction_text="PLACEHOLDER",
            attributes={
                "experience_text": "PLACEHOLDER",
                "sentiment":       "Positive",
                "evidence_quote":  "PLACEHOLDER",
                "confidence":      0.85,
                "category_name":   "Display",
            },
        ),
    ],
)


# ---------------------------------------------------------------------------
# Example 2 — Hindi/translated transcript (mid-range phone)
# Source: [FILL IN — real trimmed translated transcript from a Trakin Tech or Technical Guruji review]
# Phone:  [FILL IN — e.g. OnePlus 13R or similar mid-range]
# Purpose: Show extractions from translated Hindi transcripts.
#          These are more colloquial. Demonstrate Neutral and Mixed sentiments.
#          Show that evidence_quote can be None when synthesised from scattered references.
# ---------------------------------------------------------------------------

EXAMPLE_B_MIDRANGE = lx.data.ExampleData(
    text="PLACEHOLDER — replace with real trimmed translated transcript (800–1,500 chars)",
    extractions=[
        lx.data.Extraction(
            extraction_class="Performance",
            extraction_text="PLACEHOLDER",
            attributes={
                "experience_text": "PLACEHOLDER",
                "sentiment":       "Positive",
                "evidence_quote":  "PLACEHOLDER",
                "confidence":      0.80,
                "category_name":   "Performance",
            },
        ),
        lx.data.Extraction(
            extraction_class="Build Quality",
            extraction_text="PLACEHOLDER",
            attributes={
                "experience_text": "PLACEHOLDER",
                "sentiment":       "Mixed",           # KEY: demonstrate Mixed sentiment
                "evidence_quote":  None,              # KEY: None when synthesised — not verbatim
                "confidence":      0.70,
                "category_name":   "Build Quality",
            },
        ),
        lx.data.Extraction(
            extraction_class="In the Box",
            extraction_text="PLACEHOLDER",
            attributes={
                "experience_text": "PLACEHOLDER",
                "sentiment":       "Positive",        # India phones often include charger
                "evidence_quote":  "PLACEHOLDER",
                "confidence":      0.90,
                "category_name":   "In the Box",
            },
        ),
    ],
)


# ---------------------------------------------------------------------------
# Example 3 — Budget phone transcript (multiple negatives, India-specific)
# Source: [FILL IN — real trimmed transcript from a budget phone review]
# Phone:  [FILL IN — e.g. Redmi 13C or similar entry-level]
# Purpose: Show Negative extractions. Show low-confidence extractions.
#          Demonstrate Gaming and Audio categories.
#          Show India-specific observations (network, pricing).
# ---------------------------------------------------------------------------

EXAMPLE_B_BUDGET = lx.data.ExampleData(
    text="PLACEHOLDER — replace with real trimmed transcript for a budget phone (800–1,500 chars)",
    extractions=[
        lx.data.Extraction(
            extraction_class="Gaming",
            extraction_text="PLACEHOLDER",
            attributes={
                "experience_text": "PLACEHOLDER",
                "sentiment":       "Negative",
                "evidence_quote":  "PLACEHOLDER",
                "confidence":      0.75,
                "category_name":   "Gaming",
            },
        ),
        lx.data.Extraction(
            extraction_class="Audio",
            extraction_text="PLACEHOLDER",
            attributes={
                "experience_text": "PLACEHOLDER",
                "sentiment":       "Neutral",          # KEY: demonstrate Neutral
                "evidence_quote":  "PLACEHOLDER",
                "confidence":      0.65,               # KEY: lower confidence example
                "category_name":   "Audio",
            },
        ),
        lx.data.Extraction(
            extraction_class="Overall",
            extraction_text="PLACEHOLDER",
            attributes={
                "experience_text": "PLACEHOLDER",
                "sentiment":       "Mixed",
                "evidence_quote":  "PLACEHOLDER",
                "confidence":      0.80,
                "category_name":   "Overall",
            },
        ),
    ],
)


# ---------------------------------------------------------------------------
# Run B prompt — loaded by langextract_run_b.py
# ---------------------------------------------------------------------------
# TODO (Phase L5): Build the Run B prompt description following the four-layer
# structure from langextract_migration_v4.md Section 7.
# Run B prompt is experience-focused (subjective observations from reviewers),
# NOT spec extraction. Keep it separate from Run A prompt.

RUN_B_PROMPT = """
PLACEHOLDER — replace with the Run B prompt description (Phase L5).

Extract subjective reviewer observations from the provided YouTube transcript.
Output one Extraction object per observation, with the experience category as the
extraction_class. Use verbatim transcript text as extraction_text.

Each extraction must include: experience_text (clean summary), sentiment, evidence_quote
(exact reviewer words — null if synthesised), confidence (0.0–1.0), category_name.
"""


# ---------------------------------------------------------------------------
# Public export — used by langextract_run_b.py
# ---------------------------------------------------------------------------

RUN_B_EXAMPLES = [EXAMPLE_B_FLAGSHIP, EXAMPLE_B_MIDRANGE, EXAMPLE_B_BUDGET]


# ---------------------------------------------------------------------------
# Verbatim check — run after filling examples (Phase L5 gate)
# ---------------------------------------------------------------------------

def run_verbatim_check() -> None:
    """
    Verifies that all extraction_text values are verbatim substrings of their
    ExampleData.text. Must pass before Phase L5 production use.

    Run: python -c "from app.config.langextract_examples_run_b_v1 import run_verbatim_check; run_verbatim_check()"
    """
    for i, ex in enumerate(RUN_B_EXAMPLES):
        if ex.text == "PLACEHOLDER — replace with real trimmed processed transcript text (800–1,500 chars)":
            print(f"WARNING: Example {i} still has placeholder text. Fill in real content before Phase L5.")
            continue
        for e in ex.extractions:
            if e.extraction_text == "PLACEHOLDER":
                continue  # Skip stubs
            assert e.extraction_text in ex.text, (
                f"NOT VERBATIM in example {i} (class={e.extraction_class!r}): "
                f"extraction_text={e.extraction_text[:60]!r}"
            )
    print("Run B verbatim check PASSED (or skipped stubs).")


if __name__ == "__main__":
    run_verbatim_check()
