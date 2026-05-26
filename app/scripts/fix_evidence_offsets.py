"""
One-time migration: re-compute char_start / char_end in spec_extraction_output.evidence_json
by running str.find() against individual raw source files instead of the assembled blob.

Run once after deploying the spec_json_builder fix:
    python -m app.scripts.fix_evidence_offsets

Reads:
    pipeline.spec_extraction_output  — evidence_json JSONB column
    pipeline.raw_scraped_data        — markdown_path per raw_id
    youtube_raw_transcript_data      — processed/translated path per raw_transcript_id

Writes:
    pipeline.spec_extraction_output  — updated evidence_json in-place per row

Safe to re-run: uses ON CONFLICT DO UPDATE / full row replace.
No LLM calls. Cost: one Storage download per unique source file.
"""

import asyncio
import logging

from app.core.supabase_client import get_client
from app.services.storage_service import fetch_file_content
from app.services.spec_json_builder import _normalize_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _recompute_offsets(evidence_text: str, file_content: str) -> tuple[int | None, int | None]:
    if not evidence_text or not file_content:
        return None, None
    idx = file_content.find(evidence_text)
    if idx != -1:
        return idx, idx + len(evidence_text)
    # Normalized fallback — cannot map back to original positions
    norm_s = _normalize_text(file_content)
    norm_e = _normalize_text(evidence_text)
    if norm_s.find(norm_e) != -1:
        logger.debug("Normalized match only for %r — offsets set to None.", evidence_text[:60])
    else:
        logger.warning("No match for evidence_text=%r — offsets set to None.", evidence_text[:60])
    return None, None


async def fix_all_rows() -> None:
    client = get_client()

    # Fetch all output rows that have evidence_json
    rows = (
        client.schema("pipeline")
        .table("spec_extraction_output")
        .select("output_id, evidence_json")
        .execute()
    ).data or []

    logger.info("Found %d spec_extraction_output rows to process.", len(rows))

    # Cache: raw_id → file content; raw_transcript_id → file content
    scraped_cache: dict[int, str]    = {}
    transcript_cache: dict[int, str] = {}

    async def _get_scraped(raw_id: int) -> str | None:
        if raw_id in scraped_cache:
            return scraped_cache[raw_id]
        row = (
            client.schema("pipeline")
            .table("raw_scraped_data")
            .select("markdown_path")
            .eq("raw_id", raw_id)
            .execute()
        ).data
        if not row:
            return None
        try:
            content = await fetch_file_content(row[0]["markdown_path"])
            scraped_cache[raw_id] = content
            return content
        except Exception as e:
            logger.error("Failed to fetch raw_id=%d: %s", raw_id, e)
            return None

    async def _get_transcript(rtid: int) -> str | None:
        if rtid in transcript_cache:
            return transcript_cache[rtid]
        row = (
            client.schema("pipeline")
            .table("youtube_raw_transcript_data")
            .select("processed_transcript_path, translated_transcript_path, translation_status")
            .eq("raw_transcript_id", rtid)
            .execute()
        ).data
        if not row:
            return None
        r = row[0]
        path = (
            r.get("translated_transcript_path")
            if r.get("translation_status") == "translation_complete"
            else r.get("processed_transcript_path")
        )
        if not path:
            return None
        try:
            content = await fetch_file_content(path)
            transcript_cache[rtid] = content
            return content
        except Exception as e:
            logger.error("Failed to fetch raw_transcript_id=%d: %s", rtid, e)
            return None

    fixed_rows = 0
    for row in rows:
        output_id    = row["output_id"]
        evidence_json: dict = row.get("evidence_json") or {}
        if not evidence_json:
            continue

        updated = False
        for field_path, entry in evidence_json.items():
            evidence_text = entry.get("evidence_text")
            if not evidence_text:
                continue

            raw_id = entry.get("raw_id")
            rtid   = entry.get("raw_transcript_id")

            if raw_id is not None:
                file_content = await _get_scraped(raw_id)
            elif rtid is not None:
                file_content = await _get_transcript(rtid)
            else:
                continue

            if file_content is None:
                continue

            new_start, new_end = _recompute_offsets(evidence_text, file_content)
            old_start = entry.get("char_start")
            old_end   = entry.get("char_end")

            if new_start != old_start or new_end != old_end:
                entry["char_start"] = new_start
                entry["char_end"]   = new_end
                updated = True

        if updated:
            (
                client.schema("pipeline")
                .table("spec_extraction_output")
                .update({"evidence_json": evidence_json})
                .eq("output_id", output_id)
                .execute()
            )
            fixed_rows += 1
            logger.info("Fixed output_id=%d", output_id)

    logger.info("Done. Fixed %d / %d rows.", fixed_rows, len(rows))


if __name__ == "__main__":
    asyncio.run(fix_all_rows())
