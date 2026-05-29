# app/core/constants.py
"""
Shared pipeline constants used across multiple modules.
Import from here — never redeclare locally.
"""

# Prefix used to distinguish gate check failures from other ValueErrors.
# API layer uses this to return HTTP 422 (Unprocessable Entity) instead of HTTP 500.
GATE_ERROR_PREFIX = "Pre-extraction validation not passed"


# ---------------------------------------------------------------------------
# Pipeline run stage names — used in pipeline_runs.current_stage updates.
# Always import from here; never redeclare locally.
# ---------------------------------------------------------------------------

class PipelineStage:
    # Run A stages
    ASSEMBLING_SOURCES    = "assembling_sources"
    GEMINI_EXTRACTION     = "gemini_extraction"
    BUILDING_SPEC_JSON    = "building_spec_json"
    SAVING_OUTPUT         = "saving_output"

    # Run B stages
    EXPERIENCE_EXTRACTION = "experience_extraction"
    TRANSCRIPT_FETCH      = "transcript_fetch"

    # YouTube stages
    LLM_CLASSIFICATION    = "llm_classification"
    TRANSCRIPT_PIPELINE   = "transcript_pipeline"

    # Other stages
    NORMALISING           = "normalising"
    ENRICHING             = "enriching"
    RESOLVING_CONFLICTS   = "resolving_conflicts"


# Jarvis commit gate — the five conditions that must all be satisfied before a commit.
# Used by validate_pre_commit, fetch_approval_package, and the commit orchestrator.
GATE_CONDITION_KEYS: tuple[str, ...] = (
    "spec_human_approved",
    "experience_human_approved",
    "experience_entries_reviewed",
    "has_unresolved_conflicts",
    "pending_staging_values",
)


# ---------------------------------------------------------------------------
# Run C — Deterministic Inference Engine
# Section 3: Price Tiers
# ---------------------------------------------------------------------------

# Tuple format: (min_price_inr, max_price_inr)
# max_price_inr is None for ULTRA_FLAGSHIP (unbounded upper limit).
# Use the base variant's current price (launch price until Dynamic Pricing Layer).
PRICE_TIERS: dict[str, tuple[int, int | None]] = {
    "ULTRA_FLAGSHIP":   (120001, None),
    "FLAGSHIP":         (80000,  120000),
    "PREMIUM_MIDRANGE": (50000,  80000),   # "Flagship Killer"
    "UPPER_MIDRANGE":   (35000,  50000),
    "LOWER_MIDRANGE":   (25000,  35000),
    "BUDGET":           (15000,  25000),
    "ENTRY":            (5000,   15000),
    "ULTRA_BUDGET":     (0,      5000),    # optional, sparse data
}

# A phone within ±15% of a price-tier edge is scored against BOTH adjacent tiers.
# e.g. ₹34,500 scores against both LOWER_MIDRANGE and UPPER_MIDRANGE.
SOFT_BOUNDARY_PCT: float = 0.15


# ---------------------------------------------------------------------------
# Run C — Section 4: India Telecom Band Reference
# All operator band sets reflect actively deployed consumer service (2025–2026).
# mmWave (n258) excluded — allocated, not consumer-live.
# ---------------------------------------------------------------------------

# ── 4.2 India-relevant 5G NR band map ──────────────────────────────────────
INDIA_5G_BAND_MAP: dict[str, dict] = {
    # ── Low-band (coverage, indoor penetration, rural) ────────────────────
    "n28": {
        "freq_mhz": 700,
        "type": "low",
        "operators": ["Jio", "BSNL"],
        "note": "Premium coverage band. Jio & BSNL only — Airtel/Vi did not buy 700 MHz.",
    },

    # ── Mid-band (the workhorse — ~84% of India 5G traffic) ──────────────
    "n78": {
        "freq_mhz": 3500,
        "type": "mid",
        "operators": ["Jio", "Airtel", "Vi", "BSNL"],
        "note": "THE make-or-break India 5G band. All four operators. Gate every rule on this.",
    },
    "n77": {
        "freq_mhz": 3700,
        "type": "mid",
        "operators": ["BSNL"],
        "note": "BSNL uses n77 (3300–3800 overlap with n78). Treat as n78-equivalent for BSNL only.",
    },
    "n41": {
        "freq_mhz": 2500,
        "type": "mid",
        "operators": ["Jio"],
        "note": "Jio carrier aggregation in dense urban. Boosts CA tier for Jio scoring.",
    },
    "n40": {
        "freq_mhz": 2300,
        "type": "mid",
        "operators": ["Airtel", "Vi", "BSNL"],
        "note": "TDD mid-band. Used for CA by Airtel/Vi in metros.",
    },
    "n38": {
        "freq_mhz": 2600,
        "type": "mid",
        "operators": ["Airtel"],
        "note": "TDD mid-band. Airtel supplementary CA.",
    },

    # ── NSA anchor bands (5G NR needs a 4G anchor for NSA architecture) ──
    "n1": {
        "freq_mhz": 2100,
        "type": "mid",
        "operators": ["Airtel", "Vi", "BSNL"],
        "note": "Primary NSA anchor for Airtel and Vi 5G. Also carries 4G traffic.",
    },
    "n3": {
        "freq_mhz": 1800,
        "type": "mid",
        "operators": ["Airtel", "Vi"],
        "note": "Secondary NSA anchor. Airtel uses in select circles.",
    },
    "n8": {
        "freq_mhz": 900,
        "type": "low",
        "operators": ["Airtel", "Vi", "BSNL"],
        "note": "Low-band. NSA anchor in some circles, also 4G fallback.",
    },
    "n5": {
        "freq_mhz": 850,
        "type": "low",
        "operators": ["Jio"],
        "note": "Jio low-band. Coverage/rural.",
    },
}

# Bands NOT in INDIA_5G_BAND_MAP are irrelevant to India scoring.
# Examples of globally common but India-irrelevant 5G bands:
# n2 (1900 MHz, USA), n7 (2600 MHz, Europe), n12 (700 MHz, USA),
# n20 (800 MHz, Europe), n25 (1900 MHz, USA), n66 (AWS-3, USA),
# n71 (600 MHz, USA), n75/n76 (Europe), n79 (4700 MHz, Japan/China),
# n48 (CBRS, USA), n53 (USA), n30 (WCS, USA).
# These are IGNORED for India scoring even if the phone lists them.

# Derived sets for rule logic
INDIA_5G_ALL_RELEVANT: set[str]   = set(INDIA_5G_BAND_MAP.keys())
INDIA_5G_CRITICAL_BAND: str       = "n78"                          # Gate every operator rule on this
INDIA_5G_COVERAGE_BANDS: set[str] = {"n28", "n8", "n5"}           # Low-band = indoor/rural
INDIA_5G_CA_BANDS: set[str]       = {"n41", "n40", "n38"}         # Carrier aggregation boosters

# Per-operator 5G band sets (India deployed, 2025–2026)
JIO_5G_BANDS:    set[str] = {"n28", "n78", "n41", "n5"}
AIRTEL_5G_BANDS: set[str] = {"n78", "n1", "n3", "n40", "n38", "n8"}
VI_5G_BANDS:     set[str] = {"n78", "n1", "n3", "n40"}
BSNL_5G_BANDS:   set[str] = {"n28", "n78", "n77", "n40", "n1"}    # n77 counts for BSNL


# ── 4.3 India-relevant 4G LTE band map ────────────────────────────────────
# 4G matters: (a) NSA 5G runs on a 4G anchor, (b) 4G is the fallback
# everywhere 5G hasn't reached — which is most of India.
INDIA_4G_BAND_MAP: dict[str, dict] = {
    # ── Low-band (coverage, rural, indoor) ────────────────────────────────
    "B5": {
        "freq_mhz": 850,
        "type": "low",
        "operators": ["Jio"],
        "note": "Jio low-band. Best rural/indoor penetration.",
    },
    "B8": {
        "freq_mhz": 900,
        "type": "low",
        "operators": ["Airtel", "Vi", "BSNL"],
        "note": "Low-band coverage for Airtel, Vi, BSNL.",
    },
    "B28": {
        "freq_mhz": 700,
        "type": "low",
        "operators": ["Jio", "BSNL"],
        "note": "700 MHz 4G. Jio & BSNL only. Same freq as n28 5G.",
    },

    # ── Mid-band (primary data carrying, nationwide) ───────────────────────
    "B1": {
        "freq_mhz": 2100,
        "type": "mid",
        "operators": ["Jio", "Airtel", "Vi", "BSNL"],
        "note": "All operators. Primary NSA 5G anchor for Airtel/Vi.",
    },
    "B3": {
        "freq_mhz": 1800,
        "type": "mid",
        "operators": ["Jio", "Airtel", "Vi", "BSNL"],
        "note": "All operators. The single most widely deployed 4G band in India.",
    },

    # ── TDD mid-band (urban capacity) ─────────────────────────────────────
    "B40": {
        "freq_mhz": 2300,
        "type": "tdd_mid",
        "operators": ["Airtel", "Vi", "BSNL"],
        "note": "TDD. Urban capacity. Airtel/Vi primary capacity band.",
    },
    "B41": {
        "freq_mhz": 2500,
        "type": "tdd_mid",
        "operators": ["Jio", "BSNL"],
        "note": "TDD. Jio primary urban capacity band.",
    },
    "B39": {
        "freq_mhz": 1900,
        "type": "tdd_mid",
        "operators": ["Jio"],
        "note": "TDD. Jio supplementary.",
    },
    "B42": {
        "freq_mhz": 3500,
        "type": "tdd_mid",
        "operators": ["Airtel"],
        "note": "TDD 3500 MHz. Airtel capacity in some circles.",
    },
}

# Bands NOT in INDIA_4G_BAND_MAP are globally common but India-irrelevant.
# Examples of ignored bands:
# B2 (1900 MHz, USA), B4 (AWS-1, USA), B7 (2600 MHz, Europe),
# B12/B13/B17 (USA 700 MHz variants), B18/B19 (Japan),
# B20 (800 MHz, Europe), B25 (1900 MHz, USA), B26 (850 MHz extended, USA),
# B29/B30/B32/B34 (USA/niche), B38 (2600 TDD, limited India deployment),
# B43/B48 (CBRS/niche, no India deployment), B66 (AWS-3, USA).

# Derived sets
INDIA_4G_ALL_RELEVANT: set[str]   = set(INDIA_4G_BAND_MAP.keys())
INDIA_4G_PRIORITY_BANDS: set[str] = {"B1", "B3", "B5", "B8", "B28", "B40", "B41"}
INDIA_4G_COVERAGE_BANDS: set[str] = {"B5", "B8", "B28"}           # Low-band = rural/indoor

# Per-operator 4G band sets (India deployed, 2025–2026)
JIO_4G_BANDS:    set[str] = {"B1", "B3", "B5", "B28", "B39", "B41"}
AIRTEL_4G_BANDS: set[str] = {"B1", "B3", "B8", "B40", "B42"}
VI_4G_BANDS:     set[str] = {"B1", "B3", "B8", "B40"}
BSNL_4G_BANDS:   set[str] = {"B1", "B3", "B28", "B40", "B41"}


# ---------------------------------------------------------------------------
# Embedding Pipeline (EM0.3)
# Constants for the semantic embedding pipeline.
# Import from here — never redeclare locally.
# ---------------------------------------------------------------------------

EMBEDDING_MODEL               = "gemini-embedding-001"
EMBEDDING_DIM                 = 768
EMBEDDING_TOKEN_BUDGET_TARGET = 1900   # assembler trims to stay below this
EMBEDDING_TOKEN_HARD_CEILING  = 2048   # model's hard input window
RUN_B_CONFIDENCE_FLOOR        = 0.75   # phone_experiences read filter
QUEUE_CLAIM_TIMEOUT_SECONDS   = 600    # stale-claim reaper threshold (10 min)
QUEUE_WORKER_POLL_INTERVAL_SECONDS = 30  # seconds between queue poll ticks

