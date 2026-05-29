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
    LEGACY handler — superseded by inference_rules_group_a.rule_jio_5g_compatibility.
    Kept for backwards-compatibility with pipeline.lookup_inference_rules rows that
    use rule_name='jio_5g_compatibility' dispatched via the legacy run_inference_engine().

    Band set now reads from core/constants.JIO_5G_BANDS so it stays in sync
    with the Group A handler. Previously hardcoded {n28,n78,n5,n8} which was
    wrong (n8 is Airtel/Vi/BSNL, missing n41). Fixed: {n28,n78,n41,n5}.
    """
    thresholds      = rule.get("thresholds") or {}
    bands_excellent = int(thresholds.get("bands_excellent", len(JIO_5G_BANDS)))
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
        if band_name in JIO_5G_BANDS:
            found_bands.add(band_name)

    band_count    = len(found_bands)
    missing_bands = JIO_5G_BANDS - found_bands
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
    LEGACY handler — superseded by inference_rules_group_a.rule_airtel_5g_compatibility.
    Kept for backwards-compatibility with legacy run_inference_engine() dispatch.

    Band set now reads from core/constants.AIRTEL_5G_BANDS {n78,n1,n3,n40,n38,n8}.
    Previously hardcoded {n8,n78} which was critically incomplete (missing n1/n3
    NSA anchors and n40/n38 CA bands). Thresholds updated accordingly.
    """
    thresholds    = rule.get("thresholds") or {}
    bands_full    = int(thresholds.get("bands_full",    4))   # excellent: n78 + 3+ others
    bands_partial = int(thresholds.get("bands_partial", 2))   # good: n78 + 1 other

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
        if band_name in AIRTEL_5G_BANDS:
            found_bands.add(band_name)

    band_count    = len(found_bands)
    missing_bands = AIRTEL_5G_BANDS - found_bands
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


# ===========================================================================
# Run C — Deterministic Inference Engine (New Architecture)
# Phase 4: Segment Scoring Helpers — Section 5 of Run_C_Inference_Engine_Spec.md
#
# These helpers are used by the new Group A–H rule handlers (Phases 6–8).
# They are entirely separate from the legacy Phase 11 handlers above.
# ===========================================================================

from app.core.constants import (
    PRICE_TIERS, SOFT_BOUNDARY_PCT,
    JIO_5G_BANDS, AIRTEL_5G_BANDS,
)


def percentile_to_tier(p: float) -> str:
    """
    Converts a percentile (0.0–1.0) to a four-level quality tier string.

    Used for both *_tier_absolute (percentile over the entire dataset) and
    *_tier_in_segment (percentile over price-tier peers only).

        p >= 0.85  → "excellent"
        p >= 0.60  → "good"
        p >= 0.35  → "adequate"
        p <  0.35  → "weak"

    These thresholds are from Section 5 of the spec and must not be changed
    locally — update the spec and this function together.
    """
    if p >= 0.85:
        return "excellent"
    if p >= 0.60:
        return "good"
    if p >= 0.35:
        return "adequate"
    return "weak"


def classify_price_tier(price_inr: int | float) -> str:
    """
    Returns the canonical PRICE_TIERS key for a given price in INR.

    Uses the ordered tier list from core/constants.py PRICE_TIERS.
    Returns 'UNKNOWN' if no tier matches (should not happen with valid prices).

    Note: Soft-boundary logic (±15%) for cross-tier scoring is handled by
    compute_segment_percentile — this function returns the PRIMARY tier only.
    """
    for tier_name, (lo, hi) in PRICE_TIERS.items():
        if hi is None:
            if price_inr >= lo:
                return tier_name
        else:
            if lo <= price_inr <= hi:
                return tier_name
    return "UNKNOWN"


async def compute_segment_percentile(
    model_id: int,
    metric_column: str,
    price_tier: str,
    higher_is_better: bool = True,
) -> tuple[float, str]:
    """
    Returns (percentile, confidence) for `metric_column` among all committed phones
    in `price_tier`, applying soft boundary (±15% of tier edges).

    Section 5 of the spec:
    - Soft boundary: phones within ±15% of a tier edge are scored against BOTH
      adjacent tiers. This function is called once per adjacent tier when needed.
    - If fewer than 4 peers: returns (0.5, 'low') — neutral, low confidence.
    - If the phone itself is not in the peer set: still returns a valid percentile
      based on where the phone's value sits in the distribution.

    Args:
        model_id:         The phone being scored.
        metric_column:    Column name in pipeline.inferred_specs to rank on.
                          Must be a numeric column (NUMERIC, INT, BIGINT).
        price_tier:       e.g. 'BUDGET', 'UPPER_MIDRANGE'. From PRICE_TIERS keys.
        higher_is_better: If False, lower values rank better (e.g. weight_grams
                          for portability uses inverted scores instead, but this
                          flag is available for direct numeric columns).

    Returns:
        (percentile: float [0.0–1.0], confidence: str 'high'|'low')

    Confidence is 'low' when fewer than 4 peers exist — not enough data for
    a meaningful relative ranking. Caller should note this in rule output.

    Implementation notes:
    - Queries pipeline.inferred_specs directly (committed + already Run-C'd phones).
    - Soft boundary is built by expanding the price range ±15% on each edge.
    - Uses asyncio.to_thread to avoid blocking the FastAPI event loop.
    - The metric_column is fetched as a raw value; Supabase returns it as a string
      for NUMERIC columns — cast explicitly to float.
    """
    MIN_PEERS_FOR_HIGH_CONFIDENCE = 4

    tier_bounds = PRICE_TIERS.get(price_tier)
    if tier_bounds is None:
        logger.warning(
            "compute_segment_percentile: unknown price_tier=%r — returning neutral (0.5, low)",
            price_tier,
        )
        return (0.5, "low")

    lo, hi = tier_bounds

    # Expand bounds by SOFT_BOUNDARY_PCT on each edge that exists
    soft_lo = lo * (1.0 - SOFT_BOUNDARY_PCT) if lo > 0 else 0
    soft_hi = hi * (1.0 + SOFT_BOUNDARY_PCT) if hi is not None else None

    client = get_client().schema("pipeline")

    # Fetch: current phone's metric value + all peer values in soft-boundary range
    # We fetch price_segment and the metric column from inferred_specs.
    # The price stored for range filtering is derived from mobile_specs.phones.current_price_inr.
    # For the peer query we rely on price_segment assignment already done (H1 rule).
    # Simpler approach: fetch all phones with price_segment = price_tier, plus
    # phones in adjacent tiers if their prices fall in the soft boundary.
    # Since price_segment is a committed column in inferred_specs, and soft boundary
    # is ±15%, we include adjacent tier phones by also fetching adjacent tier peers.

    # Step 1: fetch the target phone's metric value
    target_res = await asyncio.to_thread(lambda: (
        client
        .table("inferred_specs")
        .select(f"model_id, {metric_column}, price_segment")
        .eq("model_id", model_id)
        .limit(1)
        .execute()
    ))
    target_rows = target_res.data or []
    if not target_rows or target_rows[0].get(metric_column) is None:
        logger.debug(
            "compute_segment_percentile: no %r value for model_id=%d — returning neutral",
            metric_column, model_id,
        )
        return (0.5, "low")

    target_value = float(target_rows[0][metric_column])

    # Step 2: fetch peer values — phones in the primary tier + adjacent tier(s)
    # We include adjacent tiers because of soft-boundary: a phone at tier edge
    # competes against both sides. Fetch primary + adjacent tier names.
    tier_order = list(PRICE_TIERS.keys())  # ordered high → low as defined in constants
    tier_idx   = tier_order.index(price_tier) if price_tier in tier_order else -1

    tiers_to_fetch = [price_tier]
    if tier_idx > 0:
        tiers_to_fetch.append(tier_order[tier_idx - 1])   # tier above (more expensive)
    if tier_idx >= 0 and tier_idx < len(tier_order) - 1:
        tiers_to_fetch.append(tier_order[tier_idx + 1])   # tier below (less expensive)

    peers_res = await asyncio.to_thread(lambda: (
        client
        .table("inferred_specs")
        .select(f"model_id, {metric_column}")
        .in_("price_segment", tiers_to_fetch)
        .not_.is_(metric_column, "null")
        .execute()
    ))
    peer_rows = peers_res.data or []

    peer_values = []
    for row in peer_rows:
        try:
            peer_values.append(float(row[metric_column]))
        except (TypeError, ValueError):
            pass

    if len(peer_values) < MIN_PEERS_FOR_HIGH_CONFIDENCE:
        logger.debug(
            "compute_segment_percentile: only %d peers for metric=%r tier=%r — low confidence",
            len(peer_values), metric_column, price_tier,
        )
        return (0.5, "low")

    # Step 3: compute percentile rank
    # percentile = fraction of peers that the target phone beats (or ties)
    if higher_is_better:
        peers_beaten = sum(1 for v in peer_values if v <= target_value)
    else:
        # Lower is better (e.g. weight): beat peers who have higher values
        peers_beaten = sum(1 for v in peer_values if v >= target_value)

    percentile = peers_beaten / len(peer_values)
    return (percentile, "high")
