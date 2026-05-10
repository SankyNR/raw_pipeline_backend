"""
Phase 11 — Run C: Inference Engine

Evaluates deterministic Python rules from pipeline.lookup_inference_rules
against committed mobile_specs data. Writes results to pipeline.phone_spec_inferences.

Design rules:
  - Zero API cost. No LLM. Pure Python.
  - Fires asynchronously after db_commit_runs.status = 'completed'.
  - Each rule is a DB row. Rule dispatch is by rule_name string.
  - Unknown rule_names are skipped with a WARNING — no crash.
  - All writes are upserts: ON CONFLICT (model_id, rule_id) DO UPDATE.
    Re-commit with updated specs → inference row is updated, not duplicated.
  - Every call writes an audit row to pipeline.inference_runs (M2).
  - Per-rule asyncio.timeout (N4) isolates hung DB queries.

INVERSE METRIC DIRECTIONS (critical — read before editing any camera/sensor rule):
  aperture:          f/1.4 < f/2.8 numerically; f/1.4 is BETTER.
                     Use MIN(aperture) for the best low-light lens — NEVER MAX.
  sar_head/sar_body: MIN W/kg = less radiation = better for user health.
  fabrication_node:  MIN nm = more efficient chip = better performance/watt.
  battery_capacity:  MAX mAh = more endurance = better.
  pwm_frequency:     MAX Hz = less flicker = better eye comfort.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import sys
from typing import Any

from app.core.supabase_client import get_client
from app.repositories.extraction_repository import (
    fetch_active_inference_rules,
    insert_inference_run,
    update_inference_run,
    upsert_phone_spec_inference,
)

logger = logging.getLogger(__name__)

# Per-rule evaluation timeout (N4). Uses asyncio.timeout (Python 3.11+).
# On older runtimes the guard is silently skipped (graceful degradation).
_HAS_ASYNCIO_TIMEOUT = sys.version_info >= (3, 11)
_RULE_TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------------
# Entry point — called by commit_orchestrator after successful commit
# ---------------------------------------------------------------------------

async def run_inference_engine(
    model_id: int,
    url_registry_id: int,
    triggered_by: str = "post_commit",
) -> dict:
    """
    Evaluates all active rules in pipeline.lookup_inference_rules for one phone.
    Upserts results into pipeline.phone_spec_inferences.
    Writes an audit row to pipeline.inference_runs.

    triggered_by: 'post_commit' | 'admin_batch'

    Returns:
        {
            "success":            bool,
            "model_id":           int,
            "inference_run_id":   int,
            "rules_evaluated":    int,
            "rules_written":      int,
            "rules_skipped":      int,
            "errors":             list[str]
        }
    """
    logger.info(
        "Run C: starting model_id=%d url_registry_id=%d triggered_by=%r",
        model_id, url_registry_id, triggered_by,
    )

    # M2 — open audit row
    try:
        inference_run_id = await asyncio.to_thread(
            insert_inference_run, model_id, url_registry_id, triggered_by
        )
    except Exception as exc:
        logger.error("Run C: failed to create audit row for model_id=%d: %s", model_id, exc)
        inference_run_id = -1  # proceed without audit — do not abort the run

    rules = await asyncio.to_thread(fetch_active_inference_rules)

    # N1 — Startup handler validation: warn on unknown active rules
    _validate_active_rules_vs_handlers(rules)

    if not rules:
        logger.warning("Run C: no active inference rules found — nothing to evaluate.")
        await _finalize_audit(inference_run_id, 0, 0, 0, [])
        return {
            "success":          True,
            "model_id":         model_id,
            "inference_run_id": inference_run_id,
            "rules_evaluated":  0,
            "rules_written":    0,
            "rules_skipped":    0,
            "errors":           [],
        }

    rules_written = 0
    rules_skipped = 0
    errors: list[str] = []

    for rule in rules:
        rule_name = rule.get("rule_name", "")
        rule_id   = rule["rule_id"]
        try:
            result = await _evaluate_rule_with_timeout(rule, model_id, url_registry_id)
            if result is None:
                rules_skipped += 1
                logger.debug("Run C: rule_name=%r skipped (no data) model_id=%d", rule_name, model_id)
                continue

            inference_id = await asyncio.to_thread(upsert_phone_spec_inference, result)
            rules_written += 1
            logger.debug(
                "Run C: rule_name=%r inference_id=%d sentiment=%r model_id=%d",
                rule_name, inference_id, result["sentiment"], model_id,
            )

        except Exception as exc:
            msg = f"rule_name={rule_name!r} rule_id={rule_id}: {exc}"
            logger.error("Run C: FAILED %s (model_id=%d)", msg, model_id, exc_info=True)
            errors.append(msg)

    success = len(errors) == 0
    logger.info(
        "Run C: DONE model_id=%d evaluated=%d written=%d skipped=%d errors=%d",
        model_id, len(rules), rules_written, rules_skipped, len(errors),
    )

    # M2 — close audit row
    await _finalize_audit(inference_run_id, len(rules), rules_written, rules_skipped, errors)

    return {
        "success":          success,
        "model_id":         model_id,
        "inference_run_id": inference_run_id,
        "rules_evaluated":  len(rules),
        "rules_written":    rules_written,
        "rules_skipped":    rules_skipped,
        "errors":           errors,
    }


async def _finalize_audit(
    inference_run_id: int,
    rules_evaluated: int,
    rules_written: int,
    rules_skipped: int,
    errors: list[str],
) -> None:
    """Updates the inference_runs audit row. Non-fatal if it fails."""
    if inference_run_id == -1:
        return
    try:
        # S4-P1-9: wrap sync DB call in asyncio.to_thread to avoid blocking the
        # FastAPI event loop for the duration of the DB round-trip (~50–200ms).
        await asyncio.to_thread(
            update_inference_run,
            inference_run_id=inference_run_id,
            rules_evaluated=rules_evaluated,
            rules_written=rules_written,
            rules_skipped=rules_skipped,
            errors=errors,
        )
    except Exception as exc:
        logger.warning("Run C: could not update audit row inference_run_id=%d: %s", inference_run_id, exc)


def _validate_active_rules_vs_handlers(rules: list[dict]) -> None:
    """
    N1 — Logs a WARNING for any active DB rule that has no registered Python handler.
    Called once per run_inference_engine invocation, not per rule (avoids log spam).
    At startup this surfaces misconfigured rules immediately via log aggregation.
    """
    unhandled = [
        r["rule_name"] for r in rules
        if r.get("rule_name") not in _RULE_HANDLERS
    ]
    if unhandled:
        logger.warning(
            "Run C: %d active rule(s) have no registered handler and will be SKIPPED: %s. "
            "Register handlers in _RULE_HANDLERS in inference_engine.py to activate them.",
            len(unhandled),
            unhandled,
        )


# ---------------------------------------------------------------------------
# Rule dispatcher
# ---------------------------------------------------------------------------

_RULE_HANDLERS: dict[str, Any] = {}  # populated below after handler definitions


async def _evaluate_rule_with_timeout(
    rule: dict,
    model_id: int,
    url_registry_id: int,
) -> dict | None:
    """
    N4 — Wraps _evaluate_rule with a per-rule asyncio.timeout guard (Python 3.11+).
    On older runtimes the guard is silently skipped to preserve compatibility.
    """
    if _HAS_ASYNCIO_TIMEOUT:
        try:
            async with asyncio.timeout(_RULE_TIMEOUT_SECONDS):
                return await _evaluate_rule(rule, model_id, url_registry_id)
        except TimeoutError:
            rule_name = rule.get("rule_name", "?")
            raise RuntimeError(
                f"rule timed out after {_RULE_TIMEOUT_SECONDS}s — "
                f"possible hung DB query in handler for rule_name={rule_name!r}"
            )
    else:
        return await _evaluate_rule(rule, model_id, url_registry_id)


async def _evaluate_rule(
    rule: dict,
    model_id: int,
    url_registry_id: int,
) -> dict | None:
    """
    Dispatches to the correct rule handler by rule_name.
    Returns a payload dict ready for upsert_phone_spec_inference, or None if
    the handler could not produce a result (e.g. missing required data).
    """
    rule_name = rule.get("rule_name", "")
    handler   = _RULE_HANDLERS.get(rule_name)

    if handler is None:
        # N1 warning is emitted once in bulk above; per-invocation is DEBUG only
        logger.debug(
            "Run C: no handler for rule_name=%r (rule_id=%d) — skipping.",
            rule_name, rule["rule_id"],
        )
        return None

    try:
        result = await handler(rule, model_id, url_registry_id)
    except Exception:
        raise  # propagate — caller logs at ERROR level

    if result is None:
        return None

    # Stamp common envelope fields
    result.update({
        "url_registry_id": url_registry_id,
        "model_id":        model_id,
        "rule_id":         rule["rule_id"],
        "category_id":     rule["category_id"],
        "confidence":      float(rule.get("confidence_score", 0.90)),
        "is_suppressed":   False,
        "generated_at":    datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_ms_client():
    """Returns a Supabase client scoped to mobile_specs schema."""
    return get_client().schema("mobile_specs")


def _render_template(template: str, **kwargs: Any) -> str:
    """
    Simple named-placeholder template rendering.
    Template uses {key} notation, e.g. "Supports {band_count} Jio 5G bands."
    """
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        logger.warning("Run C: template rendering missing key %s in %r", exc, template)
        return template


# ---------------------------------------------------------------------------
# Rule Handlers
# ---------------------------------------------------------------------------

# ── jio_5g_compatibility ────────────────────────────────────────────────────

async def _rule_jio_5g(rule: dict, model_id: int, url_registry_id: int) -> dict | None:
    """
    Counts how many Jio India 5G bands (n28, n78, n5, n8) the phone supports.

    thresholds.bands_excellent: 4  → Positive ("Supports all four Jio 5G bands")
    thresholds.bands_partial:   2  → Neutral  ("Supports {n} of 4 Jio 5G bands")
    <partial                        → Negative ("Missing critical Jio 5G bands: {list}")

    Band names are case-/spacing-normalised for matching (e.g. "n28" == "N28" == "N 28").
    """
    JIO_BANDS       = {"n28", "n78", "n5", "n8"}
    thresholds      = rule.get("thresholds") or {}
    bands_excellent = int(thresholds.get("bands_excellent", 4))
    bands_partial   = int(thresholds.get("bands_partial",   2))

    client = _get_ms_client()
    result = await asyncio.to_thread(lambda: (
        client
        .table("phone_network_bands")
        .select("band_id, lookup_network_bands(band_name)")
        .eq("model_id", model_id)
        .execute()
    ))
    rows = result.data or []

    found_bands = set()
    for row in rows:
        lookup    = row.get("lookup_network_bands") or {}
        band_name = (lookup.get("band_name") or "").lower().replace(" ", "").replace("-", "")
        if band_name in JIO_BANDS:
            found_bands.add(band_name)

    band_count    = len(found_bands)
    missing_bands = JIO_BANDS - found_bands
    snapshot      = {
        "found_bands":   sorted(found_bands),
        "missing_bands": sorted(missing_bands),
        "band_count":    band_count,
    }

    if band_count >= bands_excellent:
        sentiment = rule.get("sentiment_positive", "Positive")
        text      = _render_template(rule["positive_template"], band_count=band_count, missing_bands="")
    elif band_count >= bands_partial:
        sentiment = rule.get("sentiment_neutral", "Neutral")
        text      = _render_template(
            rule.get("neutral_template") or rule["positive_template"],
            band_count=band_count,
            missing_bands=", ".join(sorted(missing_bands)),
        )
    else:
        sentiment = rule.get("sentiment_negative", "Negative")
        text      = _render_template(
            rule["negative_template"],
            band_count=band_count,
            missing_bands=", ".join(sorted(missing_bands)),
        )

    return {"inference_text": text, "sentiment": sentiment, "input_field_snapshot": snapshot}


# ── airtel_5g_compatibility ─────────────────────────────────────────────────

async def _rule_airtel_5g(rule: dict, model_id: int, url_registry_id: int) -> dict | None:
    """
    Counts how many Airtel India 5G bands (n8, n78) the phone supports.

    thresholds.bands_full: 2    → Positive
    thresholds.bands_partial: 1 → Neutral
    0                           → Negative
    """
    AIRTEL_BANDS  = {"n8", "n78"}
    thresholds    = rule.get("thresholds") or {}
    bands_full    = int(thresholds.get("bands_full",    2))
    bands_partial = int(thresholds.get("bands_partial", 1))

    client = _get_ms_client()
    result = await asyncio.to_thread(lambda: (
        client
        .table("phone_network_bands")
        .select("band_id, lookup_network_bands(band_name)")
        .eq("model_id", model_id)
        .execute()
    ))
    rows = result.data or []

    found_bands = set()
    for row in rows:
        lookup    = row.get("lookup_network_bands") or {}
        band_name = (lookup.get("band_name") or "").lower().replace(" ", "").replace("-", "")
        if band_name in AIRTEL_BANDS:
            found_bands.add(band_name)

    band_count    = len(found_bands)
    missing_bands = AIRTEL_BANDS - found_bands
    snapshot      = {
        "found_bands":   sorted(found_bands),
        "missing_bands": sorted(missing_bands),
        "band_count":    band_count,
    }

    if band_count >= bands_full:
        sentiment = rule.get("sentiment_positive", "Positive")
        text      = _render_template(rule["positive_template"], band_count=band_count, missing_bands="")
    elif band_count >= bands_partial:
        sentiment = rule.get("sentiment_neutral", "Neutral")
        text      = _render_template(
            rule.get("neutral_template") or rule["positive_template"],
            band_count=band_count,
            missing_bands=", ".join(sorted(missing_bands)),
        )
    else:
        sentiment = rule.get("sentiment_negative", "Negative")
        text      = _render_template(
            rule["negative_template"],
            band_count=band_count,
            missing_bands=", ".join(sorted(missing_bands)),
        )

    return {"inference_text": text, "sentiment": sentiment, "input_field_snapshot": snapshot}


# ── battery_endurance_tier ──────────────────────────────────────────────────

async def _rule_battery_endurance(rule: dict, model_id: int, url_registry_id: int) -> dict | None:
    """
    Scores battery endurance based on mAh and display refresh rate.

    thresholds:
      mah_excellent: 5000   → base Positive
      mah_good:      4500   → base Neutral
      mah_weak:      4000   → base Negative
      hi_refresh_hz: 90     → threshold for considering high refresh

    High refresh rate (>= hi_refresh_hz) on a sub-excellent battery downgrades
    by one tier — UNLESS the display uses adaptive refresh (LTPO) in which case
    the downgrade is skipped (N2 fix: LTPO panels spend most time at ≤1Hz).

    Battery capacity: MAX mAh = better (not an inverse metric).

    C3 FIX: Schema column is refresh_rate (SMALLINT), not refresh_rate_max_hz.
    N2  FIX: Check phone_display_features for 'ltpo' or 'adaptive' before downgrading.
    """
    thresholds    = rule.get("thresholds") or {}
    mah_excellent = int(thresholds.get("mah_excellent", 5000))
    mah_good      = int(thresholds.get("mah_good",      4500))
    hi_refresh_hz = int(thresholds.get("hi_refresh_hz",   90))

    client = _get_ms_client()

    # Battery capacity
    charge_res = await asyncio.to_thread(lambda: (
        client
        .table("charging_specs")
        .select("battery_capacity")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    ))
    charge_rows = charge_res.data or []
    if not charge_rows or charge_rows[0].get("battery_capacity") is None:
        logger.debug("battery_endurance: no battery_capacity for model_id=%d — skip", model_id)
        return None

    mah = float(charge_rows[0]["battery_capacity"])

    # C3 FIX: correct column name is refresh_rate, not refresh_rate_max_hz
    disp_res = await asyncio.to_thread(lambda: (
        client
        .table("phone_displays")
        .select("display_id, refresh_rate, display_position")  # Fix 12: added display_id
        .eq("model_id", model_id)
        .execute()
    ))
    disp_rows = disp_res.data or []
    primary_refresh = None
    primary_display_id = None
    for d in disp_rows:
        if (d.get("display_position") or "Primary").lower() == "primary":
            primary_refresh    = d.get("refresh_rate")        # C3 FIX
            primary_display_id = d.get("display_id")
            break
    if primary_refresh is None and disp_rows:
        primary_refresh    = disp_rows[0].get("refresh_rate") # C3 FIX
        primary_display_id = disp_rows[0].get("display_id")

    primary_refresh = float(primary_refresh) if primary_refresh is not None else 60.0
    high_refresh    = primary_refresh >= hi_refresh_hz

    # N2 FIX: exempt LTPO/adaptive displays from battery downgrade
    is_adaptive = False
    if high_refresh and primary_display_id is not None:
        feat_res = await asyncio.to_thread(lambda: (
            client
            .table("phone_display_features")
            .select("lookup_display_features(feature_name)")
            .eq("display_id", primary_display_id)
            .execute()
        ))
        for feat_row in (feat_res.data or []):
            feat_lookup = feat_row.get("lookup_display_features") or {}
            feat_name   = (feat_lookup.get("feature_name") or "").lower()
            if "ltpo" in feat_name or "adaptive" in feat_name:
                is_adaptive = True
                break

    snapshot = {
        "battery_capacity_mah": mah,
        "primary_refresh_hz":   primary_refresh,
        "high_refresh":         high_refresh,
        "is_adaptive_refresh":  is_adaptive,
    }

    # Base tier by mAh
    if mah >= mah_excellent:
        tier = "Positive"
    elif mah >= mah_good:
        tier = "Neutral"
    else:
        tier = "Negative"

    # Downgrade one tier for high static refresh rate (skip for adaptive LTPO panels)
    if high_refresh and not is_adaptive:
        if tier == "Positive":
            tier = "Neutral"
        elif tier == "Neutral":
            tier = "Negative"

    sentiment = {
        "Positive": rule.get("sentiment_positive", "Positive"),
        "Neutral":  rule.get("sentiment_neutral",  "Neutral"),
        "Negative": rule.get("sentiment_negative", "Negative"),
    }[tier]

    template = {
        "Positive": rule["positive_template"],
        "Neutral":  rule.get("neutral_template") or rule["positive_template"],
        "Negative": rule["negative_template"],
    }[tier]

    text = _render_template(
        template,
        mah=int(mah),
        refresh_hz=int(primary_refresh),
        high_refresh="Yes" if high_refresh else "No",
        adaptive="Yes" if is_adaptive else "No",
    )
    return {"inference_text": text, "sentiment": sentiment, "input_field_snapshot": snapshot}


# ── charger_in_box_value ────────────────────────────────────────────────────

async def _rule_charger_in_box(rule: dict, model_id: int, url_registry_id: int) -> dict | None:
    """
    Evaluates charger inclusion and wattage.

    thresholds:
      watt_premium:  65  → Positive (fast charger included)
      watt_standard: 25  → Neutral  (charger included, average speed)
      charger_in_box=FALSE → Negative (no charger in box)

    C2 FIX: Schema column is charging_power (SMALLINT), not max_charging_speed_watt.
    """
    thresholds   = rule.get("thresholds") or {}
    watt_premium = float(thresholds.get("watt_premium", 65))

    client = _get_ms_client()
    res = await asyncio.to_thread(lambda: (
        client
        .table("charging_specs")
        .select("charger_in_box, charging_power")             # C2 FIX
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    ))
    rows = res.data or []
    if not rows:
        return None

    row            = rows[0]
    charger_in_box = row.get("charger_in_box")
    charging_power = row.get("charging_power")                # C2 FIX
    watt_f         = float(charging_power) if charging_power is not None else 0.0
    snapshot       = {"charger_in_box": charger_in_box, "charging_power": charging_power}

    if charger_in_box is False:
        sentiment = rule.get("sentiment_negative", "Negative")
        text      = _render_template(
            rule["negative_template"],
            watt=int(watt_f) if charging_power else "N/A",
        )
    elif watt_f >= watt_premium:
        sentiment = rule.get("sentiment_positive", "Positive")
        text      = _render_template(rule["positive_template"], watt=int(watt_f))
    else:
        sentiment = rule.get("sentiment_neutral", "Neutral")
        text      = _render_template(
            rule.get("neutral_template") or rule["positive_template"],
            watt=int(watt_f) if watt_f else "unknown",
        )

    return {"inference_text": text, "sentiment": sentiment, "input_field_snapshot": snapshot}


# ── hybrid_sim_limitation ───────────────────────────────────────────────────

async def _rule_hybrid_sim(rule: dict, model_id: int, url_registry_id: int) -> dict | None:
    """
    Flags hybrid SIM tray configurations (SIM2 slot shared with microSD).
    Hybrid = Negative warning for users who want Dual SIM + expandable storage.

    M1 FIX: Schema column is sim_config_id (not sim_configuration_id).
    """
    client = _get_ms_client()
    res = await asyncio.to_thread(lambda: (
        client
        .table("network")
        .select("sim_config_id, lookup_sim_configurations(configuration_name)")  # M1 FIX
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    ))
    rows = res.data or []
    if not rows:
        return None

    row      = rows[0]
    lookup   = row.get("lookup_sim_configurations") or {}
    config   = (lookup.get("configuration_name") or "").lower()
    snapshot = {"sim_configuration": config}

    is_hybrid = "hybrid" in config

    if is_hybrid:
        sentiment = rule.get("sentiment_negative", "Negative")
        text      = _render_template(rule["negative_template"], sim_config=config.title())
    else:
        sentiment = rule.get("sentiment_positive", "Positive")
        text      = _render_template(rule["positive_template"], sim_config=config.title())

    return {"inference_text": text, "sentiment": sentiment, "input_field_snapshot": snapshot}


# ── multimedia_quality ──────────────────────────────────────────────────────

async def _rule_multimedia_quality(rule: dict, model_id: int, url_registry_id: int) -> dict | None:
    """
    Evaluates speaker count and stereo configuration for multimedia quality.

    thresholds:
      speakers_stereo: 2 → Positive (stereo or more)
      (== 1)           → Neutral   (mono)
      (== 0 or null)   → Negative  (no speaker data or no speaker)
    """
    thresholds      = rule.get("thresholds") or {}
    speakers_stereo = int(thresholds.get("speakers_stereo", 2))

    client = _get_ms_client()
    res = await asyncio.to_thread(lambda: (
        client
        .table("audio")
        .select("speaker_count, speaker_positions")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    ))
    rows = res.data or []
    if not rows:
        return None

    row               = rows[0]
    speaker_count     = row.get("speaker_count") or 0
    speaker_positions = row.get("speaker_positions") or ""
    snapshot          = {"speaker_count": speaker_count, "speaker_positions": speaker_positions}

    if speaker_count >= speakers_stereo:
        sentiment = rule.get("sentiment_positive", "Positive")
        text      = _render_template(
            rule["positive_template"],
            speaker_count=speaker_count,
            speaker_positions=speaker_positions or "stereo",
        )
    elif speaker_count == 1:
        sentiment = rule.get("sentiment_neutral", "Neutral")
        text      = _render_template(
            rule.get("neutral_template") or rule["positive_template"],
            speaker_count=speaker_count,
            speaker_positions="mono",
        )
    else:
        sentiment = rule.get("sentiment_negative", "Negative")
        text      = _render_template(
            rule["negative_template"],
            speaker_count=speaker_count,
            speaker_positions="",
        )

    return {"inference_text": text, "sentiment": sentiment, "input_field_snapshot": snapshot}


# ── aperture_low_light ──────────────────────────────────────────────────────

async def _rule_aperture_low_light(rule: dict, model_id: int, url_registry_id: int) -> dict | None:
    """
    Evaluates best low-light camera quality using the minimum aperture value.

    CRITICAL — INVERSE METRIC:
      aperture is an inverse metric. f/1.4 < f/2.8 numerically.
      f/1.4 is BETTER (wider aperture = more light admitted).
      We use MIN(aperture) to find the best low-light lens.
      NEVER MAX(aperture) — that returns the worst lens.

    thresholds:
      aperture_excellent: 1.8  → Positive (f/1.8 or better)
      aperture_good:      2.2  → Neutral  (f/2.2 or better)
      aperture_min_sane:  0.8  → skip if best_aperture < this (N5: corrupted data guard)
      (worse than aperture_good) → Negative
    """
    thresholds         = rule.get("thresholds") or {}
    aperture_excellent = float(thresholds.get("aperture_excellent", 1.8))
    aperture_good      = float(thresholds.get("aperture_good",      2.2))
    aperture_min_sane  = float(thresholds.get("aperture_min_sane",  0.8))  # N5

    client = _get_ms_client()
    res = await asyncio.to_thread(lambda: (
        client
        .table("camera_lens_specs")
        .select("aperture, lens_type_id")
        .eq("model_id", model_id)
        .execute()
    ))
    rows = res.data or []

    apertures = [
        float(r["aperture"])
        for r in rows
        if r.get("aperture") is not None
    ]

    if not apertures:
        logger.debug("aperture_low_light: no aperture data for model_id=%d — skip", model_id)
        return None

    # MIN = best low-light lens (widest aperture = lowest f-number)
    best_aperture = min(apertures)

    # N5 — sanity bound: f < 0.8 is physically impossible; skip as corrupted data
    if best_aperture < aperture_min_sane:
        logger.warning(
            "aperture_low_light: best_aperture=%.2f < sane minimum %.2f for model_id=%d — "
            "skipping (likely corrupted extraction data).",
            best_aperture, aperture_min_sane, model_id,
        )
        return None

    snapshot = {
        "all_apertures": apertures,
        "best_aperture": best_aperture,
        "note":          "MIN aperture used — lower f-number = wider = more light (inverse metric)",
    }

    if best_aperture <= aperture_excellent:
        sentiment = rule.get("sentiment_positive", "Positive")
        text      = _render_template(rule["positive_template"], aperture=best_aperture)
    elif best_aperture <= aperture_good:
        sentiment = rule.get("sentiment_neutral", "Neutral")
        text      = _render_template(
            rule.get("neutral_template") or rule["positive_template"],
            aperture=best_aperture,
        )
    else:
        sentiment = rule.get("sentiment_negative", "Negative")
        text      = _render_template(rule["negative_template"], aperture=best_aperture)

    return {"inference_text": text, "sentiment": sentiment, "input_field_snapshot": snapshot}


# ---------------------------------------------------------------------------
# Handler registry — populated after all definitions
# ---------------------------------------------------------------------------

_RULE_HANDLERS = {
    "jio_5g_compatibility":    _rule_jio_5g,
    "airtel_5g_compatibility": _rule_airtel_5g,
    "battery_endurance_tier":  _rule_battery_endurance,
    "charger_in_box_value":    _rule_charger_in_box,
    "hybrid_sim_limitation":   _rule_hybrid_sim,
    "multimedia_quality":      _rule_multimedia_quality,
    "aperture_low_light":      _rule_aperture_low_light,
}
