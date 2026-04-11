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

    Triggers Run C (inference engine) for one or more committed phones.

    body.model_ids = [1, 2, 3]  -> run for those model_ids only
    body.model_ids = null        -> run for ALL committed phones (paginated by batch_size)

    Runs sequentially per phone (zero API cost -- no parallelism needed).
    Returns per-phone results including rules_written and any errors.

    M3 -- Pagination for full-catalogue runs:
      Default batch_size=50. Max 100. Pass offset to continue from where you left off.
      Repeat until response.total_remaining == 0.

    C4 -- url_registry_id resolution:
      Uses pipeline.phone_spec_inferences first (fast path for re-runs), then falls
      back to pipeline.db_commit_runs (has both model_id and url_registry_id).
      Does NOT query url_registry.model_id (column does not exist on that table).
    """
    import asyncio
    from app.services.inference_engine import run_inference_engine

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

    # Fast path: resolve url_registry_id from existing phone_spec_inferences rows
    client = get_client()
    inf_res = (
        client
        .schema("pipeline")
        .table("phone_spec_inferences")
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
            result = await run_inference_engine(
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
