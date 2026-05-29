"""
Inference API Router.

Phase 11 (Task 11.1): POST /admin/inference/run-batch
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.supabase_client import get_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["inference"])


# ---------------------------------------------------------------------------
# Phase 11 -- Run C Inference Engine Admin Endpoints
# ---------------------------------------------------------------------------

class RunBatchRequest(BaseModel):
    model_ids:  list[int] | None = None
    """
    List of model_ids to re-run inference for.
    Pass null (omit) to process all committed phones from mobile_specs.phones.
    """
    batch_size: int = 50
    """
    M3: Max phones to process per request. For full-catalogue runs, use offset
    to paginate (call repeatedly until total_remaining == 0).
    Range: 1-100. Default: 50.
    """
    offset: int = 0
    """
    M3: Skip this many phones from the resolved target list before processing.
    Used for paginated full-catalogue runs.
    """


@router.post("/inference/run-batch")
async def run_inference_batch(body: RunBatchRequest):
    """
    POST /admin/inference/run-batch

    Triggers Run C (Groups A–H, new orchestrator) for one or more committed phones.
    Previously called the legacy run_inference_engine(); now calls run_c_engine()
    from run_c_orchestrator.py which uses the correct Group A–H handlers with
    band sets from constants.py.

    body.model_ids = [1, 2, 3]  -> run for those model_ids only
    body.model_ids = null        -> run for ALL committed phones (paginated by batch_size)

    Runs sequentially per phone.
    Returns per-phone results including rules_written and any errors.

    M3 -- Pagination for full-catalogue runs:
      Default batch_size=50. Max 100. Pass offset to continue from where you left off.
      Repeat until response.total_remaining == 0.
    """
    import asyncio
    from app.services.run_c_orchestrator import run_c_engine

    # Clamp batch_size
    batch_size = max(1, min(body.batch_size, 100))
    offset     = max(0, body.offset)

    # Resolve full target list
    if body.model_ids is not None:
        all_target_ids: list[int] = body.model_ids
    else:
        client = get_client()
        res = (
            client
            .schema("mobile_specs")
            .table("phones")
            .select("model_id")
            .execute()
        )
        all_target_ids = [r["model_id"] for r in (res.data or [])]

    total_phones    = len(all_target_ids)
    target_ids      = all_target_ids[offset: offset + batch_size]
    total_remaining = max(0, total_phones - offset - len(target_ids))

    if not target_ids:
        return {
            "total":           total_phones,
            "batch_size":      batch_size,
            "offset":          offset,
            "processed":       0,
            "total_remaining": total_remaining,
            "succeeded":       0,
            "failed":          0,
            "skipped":         0,
            "results":         [],
        }

    # Fast path: resolve url_registry_id from existing inferred_specs rows
    # (written by run_c_engine / Groups A–H). Falls back to db_commit_runs.
    client = get_client()
    inf_res = (
        client
        .schema("pipeline")
        .table("inferred_specs")
        .select("model_id, url_registry_id")
        .in_("model_id", target_ids)
        .execute()
    )
    model_to_url: dict[int, int] = {}
    for row in (inf_res.data or []):
        model_to_url[row["model_id"]] = row["url_registry_id"]

    # C4 FIX: fallback to db_commit_runs (not url_registry -- no model_id column there)
    missing_url = [mid for mid in target_ids if mid not in model_to_url]
    if missing_url:
        commit_res = (
            get_client()
            .schema("pipeline")
            .table("db_commit_runs")
            .select("model_id, url_registry_id")
            .in_("model_id", missing_url)
            .eq("status", "completed")
            .order("commit_run_id", desc=True)
            .execute()
        )
        seen_fallback: set[int] = set()
        for row in (commit_res.data or []):
            mid = row.get("model_id")
            uid = row.get("url_registry_id")
            if mid and uid and mid not in seen_fallback:
                model_to_url[mid] = uid
                seen_fallback.add(mid)

    succeeded = 0
    failed    = 0
    skipped   = 0
    results: list[dict] = []

    for mid in target_ids:
        url_registry_id = model_to_url.get(mid)
        if url_registry_id is None:
            logger.warning(
                "run_inference_batch: could not resolve url_registry_id "
                "for model_id=%d -- skipping.", mid,
            )
            skipped += 1
            results.append({"model_id": mid, "status": "skipped", "reason": "no url_registry_id"})
            continue

        try:
            result = await run_c_engine(
                model_id=mid,
                url_registry_id=url_registry_id,
                triggered_by="admin_batch",
            )
            if result["success"]:
                succeeded += 1
                results.append({"model_id": mid, "status": "ok", **result})
            else:
                failed += 1
                results.append({"model_id": mid, "status": "partial_failure", **result})
        except Exception as exc:
            failed += 1
            logger.error(
                "run_inference_batch: FAILED for model_id=%d: %s", mid, exc, exc_info=True,
            )
            results.append({"model_id": mid, "status": "error", "detail": str(exc)})

    return {
        "total":           total_phones,
        "batch_size":      batch_size,
        "offset":          offset,
        "processed":       len(target_ids),
        "total_remaining": total_remaining,
        "succeeded":       succeeded,
        "failed":          failed,
        "skipped":         skipped,
        "results":         results,
    }


# ---------------------------------------------------------------------------
# POST /admin/inference/run-c/{url_registry_id}
# Run C deterministic engine — Groups A–H — for a single phone.
# ---------------------------------------------------------------------------

@router.post("/inference/run-c/{url_registry_id}")
async def run_c_for_phone(url_registry_id: int):
    """
    POST /admin/inference/run-c/{url_registry_id}

    Triggers the full Run C deterministic inference engine (Groups A–H)
    for the committed phone identified by url_registry_id.

    Resolves model_id from pipeline.db_commit_runs (most recent completed row).
    Runs synchronously — waits for all groups to complete before returning.

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
    from app.services.run_c_orchestrator import run_c_engine, _resolve_model_id

    # Resolve model_id
    model_id = await _resolve_model_id(url_registry_id)
    if model_id is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No completed db_commit_run found for url_registry_id={url_registry_id}. "
                "Run the DB commit first."
            ),
        )

    try:
        result = await run_c_engine(
            model_id=model_id,
            url_registry_id=url_registry_id,
            triggered_by="admin_manual",
        )
    except Exception as exc:
        logger.error(
            "run_c_for_phone: FAILED url_registry_id=%d model_id=%d: %s",
            url_registry_id, model_id, exc, exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(exc))

    if not result["success"]:
        # Partial failure — return 207 Multi-Status with details
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=207, content=result)

    return result


# ---------------------------------------------------------------------------
# GET /admin/inference/specs/{url_registry_id}
# Read inferred_specs + inference_entries for one phone.
# ---------------------------------------------------------------------------

@router.get("/inference/specs/{url_registry_id}")
async def get_inferred_specs(url_registry_id: int):
    """
    GET /admin/inference/specs/{url_registry_id}

    Returns the full inferred_specs row and all inference_entries for a phone.
    Useful for admin review / debugging Run C outputs.

    Returns 404 if Run C has not yet been run for this phone.
    """
    client = get_client()

    # Fetch inferred_specs
    specs_res = await __import__("asyncio").to_thread(lambda: (
        client
        .schema("pipeline")
        .table("inferred_specs")
        .select("*")
        .eq("url_registry_id", url_registry_id)
        .limit(1)
        .execute()
    ))
    specs = (specs_res.data or [])
    if not specs:
        raise HTTPException(
            status_code=404,
            detail=f"No inferred_specs found for url_registry_id={url_registry_id}. Run C may not have run yet.",
        )

    # Fetch inference_entries
    entries_res = await __import__("asyncio").to_thread(lambda: (
        client
        .schema("pipeline")
        .table("inference_entries")
        .select("rule_key, inference_text, sentiment, confidence, defers_to_runb_category, emitted_to_embedding, conflict_flag, confidence")
        .eq("url_registry_id", url_registry_id)
        .order("rule_key")
        .execute()
    ))
    entries = entries_res.data or []

    return {
        "url_registry_id": url_registry_id,
        "inferred_specs":  specs[0],
        "inference_entries": entries,
        "total_entries":   len(entries),
        "emitted_count":   sum(1 for e in entries if e.get("emitted_to_embedding")),
        "conflict_count":  sum(1 for e in entries if e.get("conflict_flag")),
    }


# ---------------------------------------------------------------------------
# GET /admin/inference/conflicts
# HITL queue — all conflict_flag=True inference_entries across all phones.
# ---------------------------------------------------------------------------

@router.get("/inference/conflicts")
async def get_conflict_flags(limit: int = 50, offset: int = 0):
    """
    GET /admin/inference/conflicts?limit=50&offset=0

    Returns all inference_entries where conflict_flag=True.
    These are cases where Run C fired but Run B has authoritative data —
    an admin should verify the Run B entry is correct before clearing.

    Ordered by url_registry_id, rule_key for deterministic pagination.
    """
    client = get_client()

    res = await __import__("asyncio").to_thread(lambda: (
        client
        .schema("pipeline")
        .table("inference_entries")
        .select(
            "url_registry_id, model_id, rule_key, inference_text, sentiment, "
            "confidence, defers_to_runb_category, input_field_snapshot"
        )
        .eq("conflict_flag", True)
        .order("url_registry_id")
        .order("rule_key")
        .range(offset, offset + limit - 1)
        .execute()
    ))
    rows = res.data or []

    return {
        "total_returned": len(rows),
        "limit":          limit,
        "offset":         offset,
        "conflicts":      rows,
    }
