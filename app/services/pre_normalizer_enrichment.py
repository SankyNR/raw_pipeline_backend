"""
Section 2 — Pre-normalizer Enrichment Pass (v2 Roadmap)

Runs immediately after spec_json_builder writes partial_json to
pipeline.spec_extraction_output, and before normalizer.run_normalisation()
processes it.

Three sub-steps (always in this order):
  2.1  Launch price enrichment — one Gemini grounded call per phone.
       Verifies LLM-extracted prices against current web data; corrects
       discounted / hallucinated values. Mandatory for every phone
       (first ingest or admin force-rerun); skipped on routine re-runs
       when price_verified_at IS NOT NULL and force_price_check is False.

  2.2  Chipset name enrichment — conditional (only when chipset.chipset_name
       is null after Run A). Most common for Samsung India phones where the
       OEM product page never lists the SoC name. Without a chipset name,
       the normalizer's DB-injection branch (Section 3.1) cannot fire, and
       all chipset detail fields remain null.

  2.3  Mini price normalizer — strips raw currency strings returned by Gemini
       ("₹20,999", "20,999 INR") to clean integers. Runs synchronously after
       enrichment so the normalizer receives clean data from the start.

Design:
  - All DB writes (flag updates) are fire-and-forget: a write failure never
    blocks the pipeline. The enrichment result is still applied to the in-memory
    extracted_json even if the DB flag update fails.
  - On any Gemini failure, the pipeline continues with the original LLM value
    and logs an issue entry (never raises, never blocks).
  - normalized_id is optional at call time — it is None when called from within
    run_normalisation() before the normalized_spec_json row is created. Flag
    updates that require a normalized_id use asyncio.to_thread() but silently
    no-op when normalized_id is None.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any

from app.services.gemini_client import call_gemini_grounded, GeminiNonRetryableError

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

_PRICE_TOLERANCE_PCT = 0.02   # ±2% — within this the LLM value is "verified"

# Regex: strip ₹, commas, whitespace, and trailing/leading "INR"
_PRICE_CLEAN_RE = re.compile(r"[₹,\s]|INR", re.IGNORECASE)

# Output schema for launch price Gemini call
_PRICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "variants": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ram_gb":         {"type": "integer"},
                    "storage_gb":     {"type": "integer"},
                    "launch_price":   {"type": "integer"},
                },
                "required": ["ram_gb", "storage_gb", "launch_price"],
            },
        },
        "confidence": {"type": "number"},
        "evidence":   {"type": "string"},
    },
    "required": ["variants", "confidence"],
}

# Output schema for chipset name Gemini call
_CHIPSET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "chipset_name": {"type": "string"},
        "confidence":   {"type": "number"},
        "evidence":     {"type": "string"},
    },
    "required": ["chipset_name", "confidence"],
}


# ─── Public entry point ────────────────────────────────────────────────────────

async def run_pre_normalizer_enrichment(
    extracted_json: dict[str, Any],
    *,
    brand: str,
    model_name: str,
    normalized_id: int | None = None,
    price_verified_at: datetime | None = None,
    force_price_check: bool = False,
) -> tuple[dict[str, Any], list[dict]]:
    """
    Runs immediately after spec_json_builder, before normalizer.run_normalisation().

    Mutates extracted_json in place (launch_price integers, chipset_name string)
    and returns the same dict alongside an issues list for audit logging.

    Args:
        extracted_json:    The partial_json dict fetched from spec_extraction_output.
                           Modified in place; also returned for chaining convenience.
        brand:             Phone brand (e.g. "Samsung").
        model_name:        Phone model (e.g. "Galaxy S25 Ultra").
        normalized_id:     PK of pipeline.normalized_spec_json, if the row already
                           exists (re-run scenario). None on first normalisation —
                           DB flag writes are skipped in that case and the normalizer
                           will set them after writing the row.
        price_verified_at: Value of normalized_spec_json.price_verified_at. If non-null
                           and force_price_check is False, step 2.1 is skipped.
        force_price_check: Admin override — forces price enrichment even if previously
                           verified. Use for admin re-trigger flows.

    Returns:
        (updated_extracted_json, issues_list)
        issues_list entries have keys: field, issue, raw_value, corrected_value.
    """
    issues: list[dict] = []

    # ─── Step 2.1 — Launch price enrichment ──────────────────────────────────
    if _needs_price_enrichment(extracted_json, force_price_check, price_verified_at):
        price_issues = await _enrich_launch_prices(
            extracted_json, brand, model_name, normalized_id
        )
        issues.extend(price_issues)
    else:
        logger.debug(
            "pre_normalizer: launch price enrichment SKIPPED — "
            "price_verified_at=%s, force=%s", price_verified_at, force_price_check
        )

    # ─── Step 2.2 — Chipset name enrichment (conditional) ────────────────────
    chipset = extracted_json.get("chipset") or {}
    if not chipset.get("chipset_name"):
        chip_issues = await _enrich_chipset_name(
            extracted_json, brand, model_name, normalized_id
        )
        issues.extend(chip_issues)

    # ─── Step 2.3 — Mini price normalizer ────────────────────────────────────
    parse_issues = _normalize_price_strings(extracted_json)
    issues.extend(parse_issues)

    return extracted_json, issues


# ─── Step 2.1 helpers ─────────────────────────────────────────────────────────

def _needs_price_enrichment(
    extracted_json: dict,
    force: bool,
    price_verified_at: datetime | None,
) -> bool:
    """
    Returns True if the launch price enrichment call should run.

    Decision logic (checked in order):
      1. force=True  → always run (admin re-trigger).
      2. price_verified_at IS NOT NULL → skip (already verified, save ₹2.70).
      3. Any variant has a null launch_price → run.
      4. Otherwise → skip (all prices already present from extraction).
    """
    if force:
        return True
    if price_verified_at is not None:
        return False
    variants = extracted_json.get("variants") or []
    return any(v.get("launch_price") is None for v in variants)


async def _enrich_launch_prices(
    extracted_json: dict,
    brand: str,
    model_name: str,
    normalized_id: int | None,
) -> list[dict]:
    """
    Single Gemini grounded call to verify / correct all variant launch prices.

    Matching strategy: RAM + storage capacities from extracted_json variants
    matched against Gemini response rows. Prices within ±2% are kept as-is
    (verified); prices outside ±2% are overridden (corrected).

    On success: sets normalized_spec_json.price_verified_at = NOW() (if
    normalized_id is provided).
    On failure: logs issue, does NOT set price_verified_at, continues pipeline.
    """
    issues: list[dict] = []
    variants: list[dict] = extracted_json.get("variants") or []

    prompt = (
        f"What is the official launch MRP (maximum retail price) in India for all "
        f"RAM and storage variants of the {brand} {model_name}? "
        f"Return the RAM in GB, storage in GB, and launch_price in INR as integers. "
        f"Example: 8GB/128GB = 20999. List every distinct configuration."
    )

    try:
        result = await call_gemini_grounded(
            prompt=prompt,
            output_schema=_PRICE_SCHEMA,
            site_hint="flipkart.com|amazon.in",
        )
        gemini_variants: list[dict] = (result.get("value") or {}).get("variants") or []

        # If value is directly a list (Gemini sometimes returns the array directly)
        if isinstance(result.get("value"), list):
            gemini_variants = result["value"]

        if not gemini_variants:
            logger.warning(
                "pre_normalizer: price enrichment — no variants in Gemini response "
                "for %s %s. Keeping LLM values.", brand, model_name
            )
            issues.append({
                "field": "variants[*].launch_price",
                "issue": "launch_price_enrichment_no_data",
                "raw_value": None,
                "corrected_value": None,
            })
            return issues

        # Build lookup: (ram_gb, storage_gb) → gemini_price
        gemini_price_map: dict[tuple, int] = {}
        for gv in gemini_variants:
            try:
                key = (int(gv["ram_gb"]), int(gv["storage_gb"]))
                gemini_price_map[key] = int(gv["launch_price"])
            except (KeyError, TypeError, ValueError):
                continue

        any_corrected = False
        for i, variant in enumerate(variants):
            try:
                ram = int(variant.get("ram_capacity") or 0)
                storage = int(variant.get("storage_capacity") or 0)
            except (TypeError, ValueError):
                continue

            key = (ram, storage)
            if key not in gemini_price_map:
                # No matching Gemini entry for this variant — leave as-is
                continue

            gemini_price = gemini_price_map[key]
            llm_price = variant.get("launch_price")

            if llm_price is None:
                # LLM missed it entirely — fill from Gemini
                variant["launch_price"] = gemini_price
                issues.append({
                    "field": f"variants[{i}].launch_price",
                    "issue": "launch_price_filled",
                    "raw_value": None,
                    "corrected_value": gemini_price,
                })
                any_corrected = True
            else:
                # Compare — coerce to int for tolerance check
                try:
                    llm_int = int(str(llm_price).replace(",", "").strip())
                except (ValueError, TypeError):
                    llm_int = None

                if llm_int is not None:
                    tolerance = llm_int * _PRICE_TOLERANCE_PCT
                    if abs(gemini_price - llm_int) > tolerance:
                        # Outside ±2% — override
                        variant["launch_price"] = gemini_price
                        issues.append({
                            "field": f"variants[{i}].launch_price",
                            "issue": "launch_price_corrected",
                            "raw_value": llm_price,
                            "corrected_value": gemini_price,
                        })
                        any_corrected = True
                    # else: within tolerance — verified, no override

        # Mark price as verified in DB (fire-and-forget)
        if normalized_id is not None:
            try:
                from app.repositories.pipeline_run_repository import update_normalized_spec_flag
                await asyncio.to_thread(
                    update_normalized_spec_flag,
                    normalized_id,
                    price_verified_at="now()",
                )
            except Exception as flag_exc:
                logger.warning(
                    "pre_normalizer: failed to set price_verified_at "
                    "for normalized_id=%d: %s", normalized_id, flag_exc
                )

        logger.info(
            "pre_normalizer: price enrichment complete for %s %s — "
            "%d variants matched, any_corrected=%s",
            brand, model_name, len(gemini_price_map), any_corrected,
        )

    except GeminiNonRetryableError as exc:
        logger.warning(
            "pre_normalizer: price enrichment non-retryable failure for %s %s: %s "
            "— keeping LLM values, price_verified_at NOT set.",
            brand, model_name, exc,
        )
        issues.append({
            "field": "variants[*].launch_price",
            "issue": "launch_price_enrichment_failed",
            "raw_value": str(exc)[:200],
            "corrected_value": None,
        })
    except Exception as exc:
        logger.warning(
            "pre_normalizer: price enrichment unexpected failure for %s %s: %s "
            "— keeping LLM values.",
            brand, model_name, exc,
        )
        issues.append({
            "field": "variants[*].launch_price",
            "issue": "launch_price_enrichment_failed",
            "raw_value": str(exc)[:200],
            "corrected_value": None,
        })

    return issues


# ─── Step 2.2 helpers ─────────────────────────────────────────────────────────

async def _enrich_chipset_name(
    extracted_json: dict,
    brand: str,
    model_name: str,
    normalized_id: int | None,
) -> list[dict]:
    """
    Conditional Gemini grounded call to fill chipset.chipset_name.

    Only called when chipset.chipset_name is null or empty after Run A.
    Most common for Samsung India phones (OEM site never lists SoC name).

    On success: writes name to extracted_json["chipset"]["chipset_name"],
                sets normalized_spec_json.chipset_name_enriched = TRUE.
    On failure: logs issue, chipset_name stays null, pipeline continues.
    """
    issues: list[dict] = []

    prompt = (
        f"What is the exact SoC / chipset name in the {brand} {model_name} smartphone? "
        f"Return only the canonical name in format 'Brand Marketing-Name' "
        f"(e.g. 'Qualcomm Snapdragon 7s Gen 2', 'MediaTek Dimensity 9200', "
        f"'Apple A18 Pro'). Strip any internal part numbers like SM7435-AB or MT6985."
    )

    try:
        result = await call_gemini_grounded(
            prompt=prompt,
            output_schema=_CHIPSET_SCHEMA,
            site_hint="gsmarena.com",
        )

        # call_gemini_grounded returns {"value": ..., "confidence": ..., ...}
        chipset_name = result.get("value")
        if isinstance(chipset_name, dict):
            chipset_name = chipset_name.get("chipset_name")
        confidence = result.get("confidence", 0.0)

        if not chipset_name or not isinstance(chipset_name, str) or not chipset_name.strip():
            logger.warning(
                "pre_normalizer: chipset name enrichment returned empty for %s %s",
                brand, model_name,
            )
            issues.append({
                "field": "chipset.chipset_name",
                "issue": "chipset_name_enrichment_no_data",
                "raw_value": None,
                "corrected_value": None,
            })
            return issues

        chipset_name = chipset_name.strip()

        # Write into extracted_json
        if not isinstance(extracted_json.get("chipset"), dict):
            extracted_json["chipset"] = {}
        extracted_json["chipset"]["chipset_name"] = chipset_name

        issues.append({
            "field": "chipset.chipset_name",
            "issue": "chipset_name_enriched",
            "raw_value": None,
            "corrected_value": chipset_name,
        })

        # Set DB flag (fire-and-forget)
        if normalized_id is not None:
            try:
                from app.repositories.pipeline_run_repository import update_normalized_spec_flag
                await asyncio.to_thread(
                    update_normalized_spec_flag,
                    normalized_id,
                    chipset_name_enriched=True,
                )
            except Exception as flag_exc:
                logger.warning(
                    "pre_normalizer: failed to set chipset_name_enriched "
                    "for normalized_id=%d: %s", normalized_id, flag_exc
                )

        logger.info(
            "pre_normalizer: chipset name enriched for %s %s → %r (confidence=%.2f)",
            brand, model_name, chipset_name, confidence,
        )

    except GeminiNonRetryableError as exc:
        logger.warning(
            "pre_normalizer: chipset name enrichment non-retryable failure for %s %s: %s",
            brand, model_name, exc,
        )
        issues.append({
            "field": "chipset.chipset_name",
            "issue": "chipset_name_enrichment_failed",
            "raw_value": str(exc)[:200],
            "corrected_value": None,
        })
    except Exception as exc:
        logger.warning(
            "pre_normalizer: chipset name enrichment unexpected failure for %s %s: %s",
            brand, model_name, exc,
        )
        issues.append({
            "field": "chipset.chipset_name",
            "issue": "chipset_name_enrichment_failed",
            "raw_value": str(exc)[:200],
            "corrected_value": None,
        })

    return issues


# ─── Step 2.3 — Mini price normalizer ─────────────────────────────────────────

def _normalize_price_strings(extracted_json: dict) -> list[dict]:
    """
    Strips currency symbols / commas / whitespace from launch_price values
    and casts them to int. Runs synchronously — no DB or API calls.

    Handles: "₹20,999", "20,999 INR", "INR 20999", "20999", 20999 (already int).
    On parse failure: sets launch_price = None and logs a price_parse_failed issue.
    """
    issues: list[dict] = []
    variants: list[dict] = extracted_json.get("variants") or []
    for i, variant in enumerate(variants):
        raw = variant.get("launch_price")
        if raw is None or isinstance(raw, int):
            continue  # already clean
        cleaned = _PRICE_CLEAN_RE.sub("", str(raw)).strip()
        try:
            variant["launch_price"] = int(cleaned)
        except (ValueError, TypeError):
            variant["launch_price"] = None
            issues.append({
                "field": f"variants[{i}].launch_price",
                "issue": "price_parse_failed",
                "raw_value": raw,
                "corrected_value": None,
            })
            logger.warning(
                "pre_normalizer: price_parse_failed variants[%d].launch_price=%r",
                i, raw,
            )
    return issues
