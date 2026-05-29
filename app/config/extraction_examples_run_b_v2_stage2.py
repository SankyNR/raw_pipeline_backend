"""
Run B Stage 2 Examples — Version 2

Teaches Stage 2 v2 key behaviors:
  - Same-aspect, different wording → MERGE
  - Different aspects within one category → KEEP SEPARATE
  - Decoration of one insight (multiple qualifying details) → MERGE into one entry
  - Hybrid spec-fact with subjective decoration → DROP
  - Bundled multi-aspect candidates → DROP or split (not "decorate and keep")
  - Per-aspect cap, not per-category cap — Camera and Display naturally produce
    many entries when reviewers covered many independent aspects.
  - Within a Camera (or any) aspect, multiple reviewers' candidates still MERGE.
    "Many entries in a category" only happens when aspects differ, never when
    multiple reviewers describe the same aspect.

Each example pairs INPUT candidate lists (the format Stage 2 actually receives)
with the EXPECTED aggregated output, and trailing comments explain what was
merged or dropped and WHY.

The first two examples are a deliberate CONTRAST PAIR (Battery): same category,
same number of candidates, but one merges to ONE entry and the other stays as
TWO. This is the single most important pedagogical pair — it teaches the LLM
the difference between "same observation, different wording" (merge) and
"same category, genuinely different scenario" (keep both).

Example 5 (Camera) demonstrates BOTH behaviors simultaneously: 10 candidates
spanning 8 different aspects, with two aspects (Daylight color science,
Portrait edge detection) having TWO reviewer candidates each that must merge.
Output is 8 entries — proving that within-aspect merging applies in Camera
just like everywhere else.
"""


# ---------------------------------------------------------------------------
# EXAMPLE 1 — Same-aspect, different wording → MERGE
# ---------------------------------------------------------------------------
EXAMPLE_B2_S2_SAME_INSIGHT_MERGE = {
    "description": (
        "Three reviewers describe the same screen-on-time insight in different "
        "words and with slightly different numbers. Must collapse to ONE entry."
    ),
    "input_excerpt": (
        "[CANDIDATE] transcript=101 category=Battery Life conf=0.92\n"
        "experience_text: Battery comfortably lasts a full day of moderate use.\n"
        "evidence_quote: I got through a full day with about 40% left in the evening.\n"
        "\n"
        "[CANDIDATE] transcript=102 category=Battery Life conf=0.90\n"
        "experience_text: Battery provides approximately 7 hours of screen-on time.\n"
        "evidence_quote: My screen-on time was around 7 hours which is solid.\n"
        "\n"
        "[CANDIDATE] transcript=103 category=Battery Life conf=0.88\n"
        "experience_text: Battery lasts all day even on heavier days.\n"
        "evidence_quote: Even on heavy days the battery saw me through to bedtime.\n"
    ),
    "expected_output": [
        {
            "experience_text": (
                "Battery comfortably lasts a full day of mixed use, delivering "
                "around 7 hours of screen-on time and surviving heavier days "
                "without needing a midday top-up."
            ),
            "sentiment": "Positive",
            "evidence_quote": "My screen-on time was around 7 hours which is solid.",
            "category": "Battery Life",
            "source_transcript_count": 3,
            "representative_raw_transcript_id": 102,
        }
    ],
    # All three candidates describe the SAME aspect: "Screen-on time / day-long endurance".
    # Different reviewers, different exact numbers, but the buyer-facing insight is one.
    # The merged experience_text folds in the supporting details (7 hours, holds up
    # on heavy days) without splitting them into separate entries.
    # Representative quote chosen from transcript 102 because it carries the most
    # specific number — best teaches the next stage of the pipeline.
}


# ---------------------------------------------------------------------------
# EXAMPLE 2 — Same category, different aspects → KEEP SEPARATE (contrast with 1)
# ---------------------------------------------------------------------------
EXAMPLE_B2_S2_DIFFERENT_ASPECTS_KEEP = {
    "description": (
        "Three Battery Life candidates that look mergeable at first glance but "
        "describe three GENUINELY different aspects: screen-on time, standby drain, "
        "and heavy-use drain rate. All three must be kept as separate entries."
    ),
    "input_excerpt": (
        "[CANDIDATE] transcript=201 category=Battery Life conf=0.93\n"
        "experience_text: Battery delivers around 7 hours of screen-on time per charge.\n"
        "evidence_quote: I was getting consistent 7-hour SOT across multiple charge cycles.\n"
        "\n"
        "[CANDIDATE] transcript=202 category=Battery Life conf=0.88\n"
        "experience_text: Phone loses only about 2-3% battery overnight on standby.\n"
        "evidence_quote: Standby drain was minimal, just 2 to 3 percent overnight.\n"
        "\n"
        "[CANDIDATE] transcript=203 category=Battery Life conf=0.90\n"
        "experience_text: Battery drains roughly 15% per hour during intensive gaming.\n"
        "evidence_quote: One hour of BGMI cost me about 15 percent of the battery.\n"
    ),
    "expected_output": [
        {
            "experience_text": "Battery delivers around 7 hours of screen-on time per charge under typical mixed use.",
            "sentiment": "Positive",
            "evidence_quote": "I was getting consistent 7-hour SOT across multiple charge cycles.",
            "category": "Battery Life",
            "source_transcript_count": 1,
            "representative_raw_transcript_id": 201,
        },
        {
            "experience_text": "Standby battery drain is minimal, losing only 2-3% overnight when idle.",
            "sentiment": "Positive",
            "evidence_quote": "Standby drain was minimal, just 2 to 3 percent overnight.",
            "category": "Battery Life",
            "source_transcript_count": 1,
            "representative_raw_transcript_id": 202,
        },
        {
            "experience_text": "Heavy gaming sessions drain the battery quickly, at roughly 15% per hour during demanding titles.",
            "sentiment": "Neutral",
            "evidence_quote": "One hour of BGMI cost me about 15 percent of the battery.",
            "category": "Battery Life",
            "source_transcript_count": 1,
            "representative_raw_transcript_id": 203,
        },
    ],
    # These three candidates all live in the Battery Life category but describe
    # three DIFFERENT aspects from the per-aspect list:
    #   - Screen-on time
    #   - Background / standby drain
    #   - Heavy-use endurance (gaming drain rate)
    # A buyer reading all three learns three distinct things. A buyer reading
    # one learns only one of them. They MUST stay separate.
    #
    # Compare directly with Example 1: there, three candidates described the SAME
    # aspect (general day-long endurance) so they merged. Same category, opposite
    # outcome — driven entirely by whether the aspects are the same or different.
}


# ---------------------------------------------------------------------------
# EXAMPLE 3 — Decoration cluster (one insight + qualifying details) → MERGE
# ---------------------------------------------------------------------------
EXAMPLE_B2_S2_DECORATION_MERGE = {
    "description": (
        "Three Audio candidates that each add a qualifying detail to the same "
        "core insight ('stereo speakers are very good'). Must merge into ONE "
        "entry that folds all qualifying details inline, not three entries."
    ),
    "input_excerpt": (
        "[CANDIDATE] transcript=301 category=Audio conf=0.92\n"
        "experience_text: Stereo speakers produce loud, rich, crisp audio.\n"
        "evidence_quote: The stereo speakers are loud and rich with crisp clarity.\n"
        "\n"
        "[CANDIDATE] transcript=302 category=Audio conf=0.89\n"
        "experience_text: Speakers stay clean and undistorted even at maximum volume.\n"
        "evidence_quote: Even cranked to max, there is no crackling or distortion.\n"
        "\n"
        "[CANDIDATE] transcript=303 category=Audio conf=0.87\n"
        "experience_text: Dolby Atmos creates an immersive atmosphere for movies.\n"
        "evidence_quote: With Dolby Atmos turned on, watching a movie feels really immersive.\n"
    ),
    "expected_output": [
        {
            "experience_text": (
                "Stereo speakers deliver loud, rich, crisp audio that stays "
                "clean without distortion at maximum volume, and Dolby Atmos "
                "support adds an immersive feel to movies."
            ),
            "sentiment": "Positive",
            "evidence_quote": "Even cranked to max, there is no crackling or distortion.",
            "category": "Audio",
            "source_transcript_count": 3,
            "representative_raw_transcript_id": 302,
        }
    ],
    # All three candidates orbit the same underlying insight: "the speakers are
    # very good." They each add a qualifying detail (loudness/richness,
    # distortion-free at max, Dolby Atmos immersion). A buyer learns the full
    # picture from one well-composed entry. Three separate entries would make
    # the embedding document repeat the same praise three times.
    #
    # The representative quote was chosen from transcript 302 because the
    # "no distortion at max volume" claim is the most concrete and verifiable.
    # Note that the merged experience_text deliberately weaves all three details
    # in — it does not just pick one and drop the others.
}


# ---------------------------------------------------------------------------
# EXAMPLE 4 — Hybrid spec-fact with subjective decoration → DROP
# ---------------------------------------------------------------------------
EXAMPLE_B2_S2_HYBRID_SPEC_DROP = {
    "description": (
        "Two candidates dress up spec facts with subjective decoration "
        "('standout', 'rare at this price'). Both must be DROPPED. The "
        "decoration does not turn a spec into an experience. A third "
        "candidate genuinely describes a subjective experience derived "
        "from the same spec and is KEPT."
    ),
    "input_excerpt": (
        "[CANDIDATE] transcript=401 category=Durability conf=0.85\n"
        "experience_text: IP68 rating at this price is a standout feature in the segment.\n"
        "evidence_quote: Getting an IP68 rating in this price range is genuinely impressive.\n"
        "\n"
        "[CANDIDATE] transcript=402 category=Software conf=0.84\n"
        "experience_text: 3 years of OS updates and 4 years of security patches is a strong commitment.\n"
        "evidence_quote: Motorola is promising 3 OS updates and 4 years of security patches.\n"
        "\n"
        "[CANDIDATE] transcript=403 category=Durability conf=0.91\n"
        "experience_text: Used the phone in heavy monsoon rain for an hour with no water damage.\n"
        "evidence_quote: I used it for an hour in heavy rain and the phone was completely fine afterwards.\n"
    ),
    "expected_output": [
        {
            "experience_text": "Phone handled heavy monsoon rain over an hour of continuous exposure with no water damage or malfunction.",
            "sentiment": "Positive",
            "evidence_quote": "I used it for an hour in heavy rain and the phone was completely fine afterwards.",
            "category": "Durability",
            "source_transcript_count": 1,
            "representative_raw_transcript_id": 403,
        }
    ],
    # Candidate 401 is dropped: "IP68 rating at this price is standout" is a spec
    # fact ("phone has IP68") decorated with subjective language ("standout",
    # "this price range"). Test: would this statement be true if every reviewer
    # hated the phone? Yes — IP68 is still IP68. DROP.
    #
    # Candidate 402 is dropped for the same reason: "3 years OS updates" is the
    # commitment Motorola made for every unit. The "strong commitment" decoration
    # does not change that. DROP.
    #
    # Candidate 403 is kept because it describes a real subjective experience
    # derived from the spec — what the reviewer ACTUALLY did with the phone in
    # the rain and what the outcome was. This kind of entry is what makes
    # experience data more valuable than spec data.
}


# ---------------------------------------------------------------------------
# EXAMPLE 5 — Camera category: many different aspects KEPT, but multiple
#             reviewers on the SAME aspect still MERGE
# ---------------------------------------------------------------------------
EXAMPLE_B2_S2_CAMERA_NO_OVERMERGE = {
    "description": (
        "Ten Camera candidates covering eight genuinely different camera aspects. "
        "Two aspects (Daylight color science, Portrait edge detection) have TWO "
        "reviewers each who described the same aspect — those merge to one entry "
        "per aspect. The other six aspects have one reviewer each. Final output: "
        "8 entries. This example teaches BOTH behaviours simultaneously: "
        "Camera-category entries stay separate WHEN aspects differ, but reviewers "
        "describing the SAME Camera aspect must still merge. 'Many entries' in a "
        "category never means 'duplicate reviewers on the same aspect'."
    ),
    "input_excerpt": (
        "[CANDIDATE] transcript=501 category=Camera conf=0.93\n"
        "experience_text: Main camera produces natural daylight colors without oversaturation.\n"
        "evidence_quote: Daylight photos from the main camera have very natural color tones.\n"
        "\n"
        "[CANDIDATE] transcript=509 category=Camera conf=0.91\n"
        "experience_text: Daylight shots have realistic, true-to-life color reproduction.\n"
        "evidence_quote: In daylight the colors look very accurate and lifelike, not punched up.\n"
        "\n"
        "[CANDIDATE] transcript=502 category=Camera conf=0.90\n"
        "experience_text: Portrait mode edge detection is precise even on hair and glasses.\n"
        "evidence_quote: The edge cutout in portrait mode handles tricky details like hair really well.\n"
        "\n"
        "[CANDIDATE] transcript=510 category=Camera conf=0.88\n"
        "experience_text: Portrait subject separation is accurate around fine details like hair strands.\n"
        "evidence_quote: It cuts around individual hair strands cleanly in portrait shots.\n"
        "\n"
        "[CANDIDATE] transcript=503 category=Camera conf=0.86\n"
        "experience_text: Portrait bokeh blur is intense and sometimes looks artificial.\n"
        "evidence_quote: The background blur in portraits is too strong and looks fake at times.\n"
        "\n"
        "[CANDIDATE] transcript=504 category=Camera conf=0.88\n"
        "experience_text: Night mode photos show oversharpening artifacts on text and edges.\n"
        "evidence_quote: At night you can see the oversharpening on signs and edges.\n"
        "\n"
        "[CANDIDATE] transcript=505 category=Camera conf=0.84\n"
        "experience_text: Ultrawide lens suffers heavy color shift and softer detail than the main camera.\n"
        "evidence_quote: The ultrawide does not match the main sensor in color or sharpness.\n"
        "\n"
        "[CANDIDATE] transcript=506 category=Camera conf=0.91\n"
        "experience_text: Telephoto delivers consistent sharpness up to 3x optical zoom.\n"
        "evidence_quote: Up to 3x the telephoto stays sharp without quality drop.\n"
        "\n"
        "[CANDIDATE] transcript=507 category=Camera conf=0.89\n"
        "experience_text: Video stabilization holds steady while walking and running.\n"
        "evidence_quote: While walking and even jogging the video stays remarkably stable.\n"
        "\n"
        "[CANDIDATE] transcript=508 category=Camera conf=0.85\n"
        "experience_text: Pro mode exposes full manual controls including shutter speed, ISO, and white balance.\n"
        "evidence_quote: Pro mode gives you complete control over shutter, ISO, and white balance.\n"
    ),
    "expected_output": [
        {
            "experience_text": "Main camera renders daylight scenes with natural, true-to-life colors that are accurate and lifelike rather than oversaturated.",
            "sentiment": "Positive",
            "evidence_quote": "In daylight the colors look very accurate and lifelike, not punched up.",
            "category": "Camera",
            "source_transcript_count": 2,
            "representative_raw_transcript_id": 509,
        },
        {
            "experience_text": "Portrait mode edge detection is precise, accurately cutting around difficult subjects like individual hair strands and glasses.",
            "sentiment": "Positive",
            "evidence_quote": "The edge cutout in portrait mode handles tricky details like hair really well.",
            "category": "Camera",
            "source_transcript_count": 2,
            "representative_raw_transcript_id": 502,
        },
        {
            "experience_text": "Portrait background blur is too intense and occasionally looks artificial rather than natural.",
            "sentiment": "Negative",
            "evidence_quote": "The background blur in portraits is too strong and looks fake at times.",
            "category": "Camera",
            "source_transcript_count": 1,
            "representative_raw_transcript_id": 503,
        },
        {
            "experience_text": "Night mode introduces visible oversharpening artifacts on text, signs, and high-contrast edges.",
            "sentiment": "Negative",
            "evidence_quote": "At night you can see the oversharpening on signs and edges.",
            "category": "Camera",
            "source_transcript_count": 1,
            "representative_raw_transcript_id": 504,
        },
        {
            "experience_text": "Ultrawide lens shows noticeable color shift and softer detail compared to the main sensor.",
            "sentiment": "Negative",
            "evidence_quote": "The ultrawide does not match the main sensor in color or sharpness.",
            "category": "Camera",
            "source_transcript_count": 1,
            "representative_raw_transcript_id": 505,
        },
        {
            "experience_text": "Telephoto lens maintains consistent sharpness throughout the optical zoom range up to 3x.",
            "sentiment": "Positive",
            "evidence_quote": "Up to 3x the telephoto stays sharp without quality drop.",
            "category": "Camera",
            "source_transcript_count": 1,
            "representative_raw_transcript_id": 506,
        },
        {
            "experience_text": "Video stabilization remains steady during walking and light running, producing smooth handheld footage.",
            "sentiment": "Positive",
            "evidence_quote": "While walking and even jogging the video stays remarkably stable.",
            "category": "Camera",
            "source_transcript_count": 1,
            "representative_raw_transcript_id": 507,
        },
        {
            "experience_text": "Pro mode provides full manual control over shutter speed, ISO, and white balance for advanced photography.",
            "sentiment": "Positive",
            "evidence_quote": "Pro mode gives you complete control over shutter, ISO, and white balance.",
            "category": "Camera",
            "source_transcript_count": 1,
            "representative_raw_transcript_id": 508,
        },
    ],
    # CRITICAL — this example teaches two things at once:
    #
    # (A) Camera-category entries stay separate when they describe different aspects.
    #     Eight distinct aspects from the Camera aspect list are covered (daylight
    #     color, portrait edge detection, portrait bokeh, night mode artifacts,
    #     ultrawide consistency, telephoto zoom, video stabilization, Pro mode),
    #     and the output keeps all 8 as separate entries. There is no category cap.
    #
    # (B) Within a SINGLE Camera aspect, multiple reviewers' candidates still merge.
    #     Inputs 501 + 509 both describe "Daylight color science" — they collapse
    #     to ONE entry with source_transcript_count=2 and representative_raw_transcript_id
    #     pointing to the candidate with the most specific evidence quote.
    #     Inputs 502 + 510 both describe "Portrait edge detection" — same treatment.
    #     The other six aspects have only one candidate each, so they don't need merging.
    #
    # The output is 8 entries from 10 inputs. Camera-category entry count is
    # NOT driven by reviewer count — it is driven by ASPECT count. If 10
    # reviewers all said "daylight colors are great" and nothing else, the
    # output for Camera would be 1 entry with source_transcript_count=10.
    # If 1 reviewer covered 15 distinct aspects, the output would be 15 entries
    # with source_transcript_count=1 each.
    #
    # This is the rule that catches the failure pattern from the original Edge
    # 50 Fusion output where lines 43+44+45 (three reviewers, same speaker
    # observation) were kept as three entries: same-aspect-different-reviewer
    # is ALWAYS a merge, regardless of category.
}


# ---------------------------------------------------------------------------
# Export — matches the v2 Stage 1 convention (list constant at module bottom).
# ---------------------------------------------------------------------------

RUN_B_STAGE2_EXAMPLES = [
    EXAMPLE_B2_S2_SAME_INSIGHT_MERGE,
    EXAMPLE_B2_S2_DIFFERENT_ASPECTS_KEEP,
    EXAMPLE_B2_S2_DECORATION_MERGE,
    EXAMPLE_B2_S2_HYBRID_SPEC_DROP,
    EXAMPLE_B2_S2_CAMERA_NO_OVERMERGE,
]
