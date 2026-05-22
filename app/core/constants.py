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
