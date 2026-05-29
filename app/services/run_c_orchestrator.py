 # app/services/run_c_orchestrator.py
"""
Run C — Deterministic Inference Engine — Orchestrator

Entry point: run_c_engine(model_id, url_registry_id, triggered_by)

Execution order (spec Section 12):
  1. Resolve model_id / url_registry_id
  2. Load Run B experience categories present for this phone
  3. Run Groups A–G (rules within each group run concurrently; groups can run in parallel)
  4. Upsert inferred_specs after each group completes (so Group H sees A–G data)
  5. Run Group H sequentially H1→H6 (each reads previous inferred_specs output)
  6. Upsert final inferred_specs row (all columns written atomically)
  7. Upsert inference_entries (one row per rule)
  8. Apply embedding narrative gate → set emitted_to_embedding
  9. conflict_flag=TRUE rows → log warning for HITL review
  10. Write audit row to pipeline.inference_runs
  11. Update url_registry.status → 'run_c_complete'

Fire-and-forget contract: wrap in try/except, never raise into the commit response.
All writes are idempotent ON CONFLICT DO UPDATE — re-running is safe.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from app.core.supabase_client import get_client

# Group handler registries
from app.services.inference_rules_group_a import GROUP_A_HANDLERS
from app.services.inference_rules_group_b import GROUP_B_HANDLERS
from app.services.inference_rules_group_c import GROUP_C_HANDLERS
from app.services.inference_rules_group_d import GROUP_D_HANDLERS
from app.services.inference_rules_group_e import GROUP_E_HANDLERS
from app.services.inference_rules_group_f import GROUP_F_HANDLERS
from app.services.inference_rules_group_g import GROUP_G_HANDLERS
from app.services.inference_rules_group_h import (
    GROUP_H_HANDLERS,
    GROUP_H_EXECUTION_ORDER,
)

logger = logging.getLogger(__name__)

_PL = lambda: get_client().schema("pipeline")

# Map rule_key → inferred_specs column name.
# Most rule_keys are identical to their column; exceptions listed here.
_RULE_KEY_TO_COLUMN: dict[str, str | list[str]] = {
    # Group A
    "jio_5g_compatibility":    "jio_5g_tier",
    "airtel_5g_compatibility":  "airtel_5g_tier",
    "vi_5g_compatibility":      "vi_5g_tier",
    "bsnl_5g_compatibility":    "bsnl_5g_tier",
    "india_4g_band_coverage":   ["india_4g_coverage", "has_low_band_4g"],
    "sim_connectivity_profile": "sim_profile",
    # Group B
    "chipset_tier":             "chipset_tier",
    "gaming_capability":        ["gaming_tier_absolute", "gaming_tier_in_segment"],
    "memory_performance":       ["ram_tier_in_segment", "storage_speed_class"],
    "multitasking_longevity":   "multitasking_longevity",
    # Group C
    "display_panel_quality":    ["panel_class", "hdr_effective"],
    "display_sharpness":        "display_sharpness_score",
    "display_smoothness":       ["display_smoothness_score", "refresh_rate_class"],
    "outdoor_visibility":       "outdoor_visibility",
    "multimedia_experience":    "multimedia_tier",
    # Group D
    "battery_endurance":        ["endurance_tier_absolute", "endurance_tier_in_segment"],
    "charging_speed":           "charging_speed_tier",
    "charger_in_box":           "charger_in_box",
    # Group E
    "main_camera_hardware":     "main_camera_hw",
    "camera_versatility":       "camera_versatility",
    "zoom_capability":          "zoom_capability",
    "video_capability":         "video_capability",
    # Group F
    "water_dust_resistance":    "ip_resistance",
    "build_material_quality":   "build_material_tier",
    "portability":              "portability",
    "display_durability":       "display_durability",
    # Group G
    "os_update_longevity":      "os_update_years",
    "security_update_longevity":"security_update_years",
    "software_cleanliness":     "software_cleanliness",
    "ecosystem_depth":          "ecosystem_depth",
    "ecosystem_profile":        "ecosystem_profile",
    # Group H
    "price_segment":            "price_segment",
    "price_performance_verdict":"price_performance",
    "use_case_fitness":         "use_case_fitness",
    "target_audience":          "target_audience",
    "strengths_and_compromises":["strengths", "compromises"],
    "value_verdict":            "value_verdict",
}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

async def _resolve_model_id(url_registry_id: int) -> int | None:
    """Resolve model_id from the most recent completed db_commit_run for url_registry_id."""
    res = await asyncio.to_thread(lambda: (
        _PL()
        .table("db_commit_runs")
        .select("model_id")
        .eq("url_registry_id", url_registry_id)
        .eq("status", "completed")
        .order("commit_run_id", desc=True)
        .limit(1)
        .execute()
    ))
    rows = res.data or []
    return rows[0]["model_id"] if rows else None


async def _load_runb_categories(url_registry_id: int) -> set[str]:
    """
    Returns the set of Run B category names present for this phone.
    Source: pipeline.phone_experiences JOIN lookup_experience_categories.
    Used by the embedding narrative gate.
    """
    res = await asyncio.to_thread(lambda: (
        _PL()
        .table("phone_experiences")
        .select("lookup_experience_categories(category_name)")
        .eq("url_registry_id", url_registry_id)
        .eq("is_suppressed", False)
        .eq("is_superseded", False)
        .execute()
    ))
    cats: set[str] = set()
    for row in (res.data or []):
        lk = row.get("lookup_experience_categories") or {}
        name = (lk.get("category_name") or "").strip()
        if name:
            cats.add(name)
    return cats


async def _upsert_inferred_specs(model_id: int, url_registry_id: int, updates: dict) -> None:
    """Idempotent upsert of inferred_specs. updates dict contains only non-None columns."""
    if not updates:
        return
    payload = {"model_id": model_id, "url_registry_id": url_registry_id, **updates,
                "updated_at": datetime.now(timezone.utc).isoformat()}
    await asyncio.to_thread(lambda: (
        _PL()
        .table("inferred_specs")
        .upsert(payload, on_conflict="model_id")
        .execute()
    ))


async def _upsert_inference_entry(
    model_id: int,
    url_registry_id: int,
    result: dict,
    emitted: bool,
) -> None:
    """Idempotent upsert of one inference_entries row."""
    payload = {
        "model_id":               model_id,
        "url_registry_id":        url_registry_id,
        "rule_key":               result["rule_key"],
        "inference_text":         result["inference_text"],
        "sentiment":              result.get("sentiment", "Neutral"),
        "confidence":             result.get("confidence", "medium"),
        "defers_to_runb_category": result.get("defers_to_runb_category"),
        "emitted_to_embedding":   emitted,
        "conflict_flag":          result.get("conflict_flag", False),
        "input_field_snapshot":   result.get("input_field_snapshot"),
        "rule_engine_version":    "v1",
    }
    await asyncio.to_thread(lambda: (
        _PL()
        .table("inference_entries")
        .upsert(payload, on_conflict="model_id,rule_key")
        .execute()
    ))


async def _write_inference_run(
    model_id: int,
    url_registry_id: int,
    triggered_by: str,
    rules_evaluated: int,
    rules_written: int,
    rules_skipped: int,
    errors: list[str],
    status: str,
) -> None:
    """Write audit row to pipeline.inference_runs."""
    payload = {
        "model_id":        model_id,
        "url_registry_id": url_registry_id,
        "triggered_by":    triggered_by,
        "rules_evaluated": rules_evaluated,
        "rules_written":   rules_written,
        "rules_skipped":   rules_skipped,
        "errors":          errors or None,
        "status":          status,
        "finished_at":     datetime.now(timezone.utc).isoformat(),
    }
    await asyncio.to_thread(lambda: (
        _PL()
        .table("inference_runs")
        .insert(payload)
        .execute()
    ))


async def _mark_run_c_complete(url_registry_id: int) -> None:
    """Update url_registry.status = 'run_c_complete'."""
    await asyncio.to_thread(lambda: (
        get_client()
        .schema("pipeline")
        .table("url_registry")
        .update({"status": "run_c_complete"})
        .eq("url_id", url_registry_id)
        .execute()
    ))


# ---------------------------------------------------------------------------
# Structured value → inferred_specs column mapper
# ---------------------------------------------------------------------------

def _extract_column_updates(result: dict) -> dict:
    """
    Given a rule result dict, return {column_name: value} for inferred_specs.
    Handles both single-column and multi-column rules.
    Unpacks JSONB structured_values for rules that split across two columns.
    """
    rule_key = result["rule_key"]
    sv       = result.get("structured_value")
    if sv is None:
        return {}

    mapping = _RULE_KEY_TO_COLUMN.get(rule_key)
    if mapping is None:
        # Fallback: use rule_key as column name
        return {rule_key: sv}

    if isinstance(mapping, str):
        return {mapping: sv}

    # Multi-column rules: structured_value must be a dict
    updates: dict = {}
    if isinstance(sv, dict):
        for col in mapping:
            if col in sv:
                updates[col] = sv[col]
    return updates


# ---------------------------------------------------------------------------
# Run group helpers
# ---------------------------------------------------------------------------

async def _run_group(
    group_handlers: dict[str, Any],
    model_id: int,
    url_registry_id: int,
    group_name: str,
) -> list[dict]:
    """Run all handlers in a group concurrently. Returns list of result dicts."""
    tasks = {
        rule_key: asyncio.create_task(handler(model_id, url_registry_id))
        for rule_key, handler in group_handlers.items()
    }
    results: list[dict] = []
    for rule_key, task in tasks.items():
        try:
            result = await task
            if result is not None:
                results.append(result)
        except Exception as exc:
            logger.error(
                "run_c: %s rule_key=%r FAILED model_id=%d: %s",
                group_name, rule_key, model_id, exc, exc_info=True,
            )
    return results


async def _run_group_h_sequential(
    model_id: int,
    url_registry_id: int,
) -> list[dict]:
    """
    Run Group H rules strictly in order H1→H6.
    After each rule, immediately upsert inferred_specs so the next rule sees it.
    """
    results: list[dict] = []
    for rule_key in GROUP_H_EXECUTION_ORDER:
        handler = GROUP_H_HANDLERS.get(rule_key)
        if handler is None:
            continue
        try:
            result = await handler(model_id, url_registry_id)
            if result is None:
                continue
            results.append(result)
            # Immediately persist so next H rule reads it
            updates = _extract_column_updates(result)
            if updates:
                await _upsert_inferred_specs(model_id, url_registry_id, updates)
        except Exception as exc:
            logger.error(
                "run_c: Group H rule_key=%r FAILED model_id=%d: %s",
                rule_key, model_id, exc, exc_info=True,
            )
    return results


# ---------------------------------------------------------------------------
# Main engine entry point
# ---------------------------------------------------------------------------

async def run_c_engine(
    model_id: int,
    url_registry_id: int,
    triggered_by: str = "post_commit",
) -> dict:
    """
    Full Run C pipeline for one phone.

    Returns:
        {
            "success":         bool,
            "model_id":        int,
            "url_registry_id": int,
            "rules_evaluated": int,
            "rules_written":   int,
            "rules_skipped":   int,
            "errors":          list[str],
        }
    """
    logger.info(
        "run_c: START model_id=%d url_registry_id=%d triggered_by=%r",
        model_id, url_registry_id, triggered_by,
    )

    rules_evaluated = 0
    rules_written   = 0
    rules_skipped   = 0
    errors: list[str] = []
    all_results: list[dict] = []

    try:
        # ── Step 2: Load Run B categories ─────────────────────────────────────
        runb_categories = await _load_runb_categories(url_registry_id)
        logger.debug("run_c: Run B categories for url_registry_id=%d: %r", url_registry_id, runb_categories)

        # ── Step 3: Run Groups A–G concurrently ───────────────────────────────
        group_ag = [
            ("A", GROUP_A_HANDLERS),
            ("B", GROUP_B_HANDLERS),
            ("C", GROUP_C_HANDLERS),
            ("D", GROUP_D_HANDLERS),
            ("E", GROUP_E_HANDLERS),
            ("F", GROUP_F_HANDLERS),
            ("G", GROUP_G_HANDLERS),
        ]

        # Groups run in declared order — B4 reads B1/B3 outputs from inferred_specs
        # so we flush each group before running the next.
        for group_name, handlers in group_ag:
            group_results = await _run_group(handlers, model_id, url_registry_id, f"Group {group_name}")

            # Flush this group's structured values to inferred_specs immediately
            group_updates: dict = {}
            for result in group_results:
                group_updates.update(_extract_column_updates(result))
            if group_updates:
                await _upsert_inferred_specs(model_id, url_registry_id, group_updates)

            all_results.extend(group_results)

        # ── Step 4: Run Group H sequentially (H1→H6) ──────────────────────────
        h_results = await _run_group_h_sequential(model_id, url_registry_id)
        all_results.extend(h_results)

        # ── Step 6: Determine emitted_to_embedding per rule ───────────────────
        #   Rule is emitted if:
        #     - defers_to_runb_category is None, OR
        #     - that category is NOT in runb_categories
        #   A conflict_flag=True rule is NOT emitted (Run B wins).

        # ── Step 7: Upsert inference_entries for all rules ────────────────────
        for result in all_results:
            rules_evaluated += 1
            defer_cat  = result.get("defers_to_runb_category")
            conflict   = result.get("conflict_flag", False)

            # Embedding gate
            if conflict:
                emitted = False
            elif defer_cat is None:
                emitted = True
            else:
                emitted = defer_cat not in runb_categories

            try:
                await _upsert_inference_entry(model_id, url_registry_id, result, emitted)
                rules_written += 1
            except Exception as exc:
                msg = f"inference_entries upsert failed for rule_key={result['rule_key']!r}: {exc}"
                logger.error("run_c: %s model_id=%d", msg, model_id)
                errors.append(msg)
                rules_skipped += 1

            # ── Step 8: HITL conflict flag logging ────────────────────────────
            if conflict:
                logger.warning(
                    "run_c: CONFLICT_FLAG rule_key=%r model_id=%d url_registry_id=%d "
                    "— Run B wins in embedding. Queued for HITL review.",
                    result["rule_key"], model_id, url_registry_id,
                )

        # ── Step 9: Mark url_registry run_c_complete ─────────────────────────
        try:
            await _mark_run_c_complete(url_registry_id)
        except Exception as exc:
            logger.warning("run_c: failed to mark run_c_complete url_registry_id=%d: %s", url_registry_id, exc)

        success = len(errors) == 0
        logger.info(
            "run_c: DONE model_id=%d evaluated=%d written=%d skipped=%d errors=%d success=%s",
            model_id, rules_evaluated, rules_written, rules_skipped, len(errors), success,
        )

    except Exception as fatal:
        success = False
        msg = f"fatal: {fatal}"
        errors.append(msg)
        logger.error("run_c: FATAL model_id=%d: %s", model_id, fatal, exc_info=True)

    # ── Step 10: Write audit row ──────────────────────────────────────────────
    try:
        await _write_inference_run(
            model_id=model_id,
            url_registry_id=url_registry_id,
            triggered_by=triggered_by,
            rules_evaluated=rules_evaluated,
            rules_written=rules_written,
            rules_skipped=rules_skipped,
            errors=errors,
            status="completed" if success else "failed",
        )
    except Exception as audit_exc:
        logger.error("run_c: failed to write audit row model_id=%d: %s", model_id, audit_exc)

    return {
        "success":         success,
        "model_id":        model_id,
        "url_registry_id": url_registry_id,
        "rules_evaluated": rules_evaluated,
        "rules_written":   rules_written,
        "rules_skipped":   rules_skipped,
        "errors":          errors,
    }


# ---------------------------------------------------------------------------
# EM5 — Post-Run-C embedding trigger
# ---------------------------------------------------------------------------

async def _trigger_embedding_pipeline(model_id: int, url_registry_id: int) -> None:
    """
    EM5 — Post-Run-C embedding trigger.

    Called by run_c_safely after run_c_engine completes (success or partial).
    Uses a LOCAL import of embedding_pipeline to avoid the circular import
    that would occur if embedding_pipeline.py imported run_c_orchestrator.

    Never raises — any failure is logged and swallowed so that the Run C
    audit row and url_registry status are not affected.
    """
    try:
        # Local import — avoids circular: embedding_pipeline → run_c_orchestrator
        from app.services.embedding_pipeline import enqueue_or_first_embed
        await enqueue_or_first_embed(
            model_id=model_id,
            url_registry_id=url_registry_id,
            reason="run_c_updated",
        )
    except Exception as exc:
        logger.error(
            "EM5 trigger: FAILED model_id=%d url_registry_id=%d: %s",
            model_id, url_registry_id, exc, exc_info=True,
        )


# ---------------------------------------------------------------------------
# Fire-and-forget wrapper (for post-commit trigger)
# ---------------------------------------------------------------------------

async def run_c_safely(model_id: int, url_registry_id: int, commit_run_id: int = -1) -> None:
    """
    Wraps run_c_engine for use with asyncio.create_task().
    Exceptions are caught and logged — never propagated to caller.

    After run_c_engine finishes (regardless of success/partial errors),
    fires the EM5 embedding trigger so the embedding pipeline can pick up
    the newly emitted inference_entries.
    """
    try:
        await run_c_engine(model_id=model_id, url_registry_id=url_registry_id, triggered_by="post_commit")
    except Exception as exc:
        logger.error(
            "run_c_safely: UNHANDLED model_id=%d commit_run_id=%d: %s",
            model_id, commit_run_id, exc, exc_info=True,
        )

    # ── EM5: trigger embedding pipeline regardless of Run C success/failure ──
    # Even a partial Run C run may have written emitted_to_embedding=TRUE rows
    # that should be picked up. The embedding pipeline is idempotent.
    await _trigger_embedding_pipeline(model_id, url_registry_id)
