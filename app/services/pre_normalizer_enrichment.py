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
from pydantic import BaseModel

from app.services.gemini_client import (
    call_gemini_grounded,
    call_gemini_json,
    GeminiNonRetryableError,
)

class _VariantCombo(BaseModel):
    ram_gb: int
    storage_gb: int

class _VariantListResponse(BaseModel):
    variants: list[_VariantCombo]
    confidence: float

class _SinglePriceResponse(BaseModel):
    launch_price: int
    confidence: float
    evidence: str = ""

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

_PRICE_TOLERANCE_PCT = 0.02   # ±2% — within this the LLM value is "verified"

# Regex: strip ₹, commas, whitespace, and trailing/leading "INR"
_PRICE_CLEAN_RE = re.compile(r"[₹,\s]|INR", re.IGNORECASE)



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

# Output schema for SIM dual standby Gemini call
_SIM_STANDBY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "value": {"type": "string", "enum": ["YES", "NO"]},
        "confidence": {"type": "number"},
        "evidence": {"type": "string"},
    },
    "required": ["value", "confidence"],
}




# ─── Public entry point ────────────────────────────────────────────────────────

async def run_pre_normalizer_enrichment(
    extracted_json: dict[str, Any],
    *,
    brand: str,
    model_name: str,
    normalized_id: int | None = None,
) -> tuple[dict[str, Any], list[dict]]:
    """
    Runs immediately after spec_json_builder, before normalizer.run_normalisation().

    Mutates extracted_json in place (launch_price integers, chipset_name string)
    and returns the same dict alongside an issues list for audit logging.

    Price enrichment always runs — it verifies which variants actually exist in
    India (removing hallucinated ones) and corrects every launch price via
    dedicated per-variant Gemini grounded calls.

    Args:
        extracted_json:  The partial_json dict fetched from spec_extraction_output.
                         Modified in place; also returned for chaining convenience.
        brand:           Phone brand (e.g. "Samsung").
        model_name:      Phone model (e.g. "Galaxy S25 Ultra").
        normalized_id:   PK of pipeline.normalized_spec_json, if the row already
                         exists (re-run scenario). None on first normalisation —
                         DB flag writes are skipped in that case and the normalizer
                         will set them after writing the row.

    Returns:
        (updated_extracted_json, issues_list)
        issues_list entries have keys: field, issue, raw_value, corrected_value.
    """
    issues: list[dict] = []

    # ─── Step 2.1 — Launch price enrichment (always runs) ────────────────────
    # Price enrichment always runs — verifies variant count and corrects prices
    price_issues = await _enrich_launch_prices(
        extracted_json, brand, model_name, normalized_id
    )
    issues.extend(price_issues)

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

    # ─── Step 2.4 — SIM dual standby verification ────────────────────────────
    # Runs for all phones with a Dual SIM config to confirm DSDS support.
    # Must run BEFORE normalizer Step 4 so the corrected string is FK-resolved.
    try:
        sim_issues = await _verify_sim_dual_standby(
            extracted_json, brand, model_name
        )
        issues.extend(sim_issues)
    except Exception as exc:
        logger.warning(
            "run_pre_normalizer_enrichment: SIM standby verification failed "
            "for %s %s: %s — continuing.", brand, model_name, exc,
        )

    return extracted_json, issues


# ─── Step 2.1 helpers ─────────────────────────────────────────────────────────


async def _enrich_launch_prices(
    extracted_json: dict,
    brand: str,
    model_name: str,
    normalized_id: int | None,
) -> list[dict]:
    """
    Three-step launch price enrichment. Always runs — never skipped.

    Step 1 — Variant verification:
        Asks Gemini which RAM+storage combinations officially exist in India.
        Any extracted variant whose (ram_capacity, storage_capacity) combination
        is NOT in Gemini's response is removed from extracted_json entirely.
        This eliminates hallucinated variants (e.g. a 8+256 that doesn't exist).

    Step 2 — Per-variant price lookup:
        For each surviving variant, fires one dedicated Gemini grounded call:
        "What is the launch price of the {ram}GB RAM + {storage}GB storage
        variant of {brand} {model} in India?"
        Extracts the integer price from the response.

    Step 3 — Exact comparison and replace:
        Compares Gemini's price against the LLM-extracted launch_price.
        Exact match → keep as-is.
        Any difference (even 1 rupee) → replace with Gemini's value.
        Null LLM price → fill with Gemini's value.

    On any Gemini failure at any step: logs issue, continues with
    whatever data is available. Never raises, never blocks the pipeline.
    """
    issues: list[dict] = []
    variants: list[dict] = extracted_json.get("variants") or []

    if not variants:
        return issues

    # ── Step 1: Variant verification ─────────────────────────────────────
    variant_prompt = (
        f"Which RAM and storage variants of the {brand} {model_name} "
        f"were officially launched for sale in India? "
        f"List every distinct RAM GB + storage GB combination that was "
        f"officially announced and sold in India. "
        f"Return only combinations that were actually released — "
        f"do not include rumoured, cancelled, or region-exclusive variants."
    )

    verified_combinations: set[tuple[int, int]] | None = None

    try:
        parsed = await call_gemini_json(
            system_prompt=(
                "You are a product database assistant. Return only valid JSON "
                "matching the provided schema. Be precise and factual."
            ),
            user_content=variant_prompt,
            output_schema=_VariantListResponse,
            temperature=0.1,
        )
        raw_variants = parsed.get("variants") or []
        if raw_variants:
            verified_combinations = set()
            for gv in raw_variants:
                try:
                    verified_combinations.add(
                        (int(gv["ram_gb"]), int(gv["storage_gb"]))
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            logger.info(
                "_enrich_launch_prices: %s %s — Gemini verified %d combinations: %s",
                brand, model_name,
                len(verified_combinations), verified_combinations,
            )
        else:
            logger.warning(
                "_enrich_launch_prices: variant verification returned no data "
                "for %s %s — skipping variant removal step.",
                brand, model_name,
            )
    except Exception as exc:
        logger.warning(
            "_enrich_launch_prices: variant verification failed for %s %s: %s "
            "— skipping variant removal step.",
            brand, model_name, exc,
        )

    # ── Remove hallucinated variants ──────────────────────────────────────
    if verified_combinations is not None:
        original_count = len(variants)
        surviving = []
        for variant in variants:
            try:
                ram     = int(variant.get("ram_capacity") or 0)
                storage = int(variant.get("storage_capacity") or 0)
            except (TypeError, ValueError):
                surviving.append(variant)  # can't check — keep it
                continue
            if (ram, storage) in verified_combinations:
                surviving.append(variant)
            else:
                issues.append({
                    "field":           "variants",
                    "issue":           "hallucinated_variant_removed",
                    "raw_value":       f"{ram}GB+{storage}GB",
                    "corrected_value": None,
                })
                logger.info(
                    "_enrich_launch_prices: removed hallucinated variant "
                    "%dGB+%dGB for %s %s",
                    ram, storage, brand, model_name,
                )

        extracted_json["variants"] = surviving
        variants = surviving  # update local reference

        if len(surviving) < original_count:
            logger.info(
                "_enrich_launch_prices: %s %s — %d variant(s) removed, "
                "%d surviving.",
                brand, model_name,
                original_count - len(surviving), len(surviving),
            )

    # ── Step 2 + 3: Per-variant price lookup and exact comparison ─────────
    for i, variant in enumerate(variants):
        try:
            ram     = int(variant.get("ram_capacity") or 0)
            storage = int(variant.get("storage_capacity") or 0)
        except (TypeError, ValueError):
            continue

        if ram == 0 or storage == 0:
            continue

        price_prompt = (
            f"What is the official launch price of the "
            f"{ram}GB RAM + {storage}GB storage variant of the "
            f"{brand} {model_name} in India? "
            f"Return the ORIGINAL launch MRP in INR as an integer "
            f"at the time it was first announced — "
            f"not the current selling price or any discounted price."
        )

        try:
            price_result = await call_gemini_json(
                system_prompt=(
                    "You are a product pricing assistant for the Indian smartphone market. "
                    "Return only valid JSON matching the provided schema. "
                    "Provide the original launch MRP at announcement, not current price."
                ),
                user_content=price_prompt,
                output_schema=_SinglePriceResponse,
                temperature=0.1,
            )
            gemini_price_raw = price_result.get("launch_price")

            if gemini_price_raw is None:
                logger.warning(
                    "_enrich_launch_prices: no price returned for "
                    "%s %s %dGB+%dGB — keeping LLM value.",
                    brand, model_name, ram, storage,
                )
                issues.append({
                    "field":           f"variants[{i}].launch_price",
                    "issue":           "price_lookup_no_data",
                    "raw_value":       variant.get("launch_price"),
                    "corrected_value": None,
                })
                continue

            try:
                gemini_price = int(gemini_price_raw)
            except (ValueError, TypeError):
                logger.warning(
                    "_enrich_launch_prices: could not parse Gemini price %r "
                    "for %s %s %dGB+%dGB.",
                    gemini_price_raw, brand, model_name, ram, storage,
                )
                continue

            llm_price = variant.get("launch_price")

            # Coerce LLM price to int for exact comparison
            try:
                llm_int = int(llm_price) if llm_price is not None else None
            except (ValueError, TypeError):
                llm_int = None

            if llm_int is None:
                # LLM missed it — fill
                variant["launch_price"] = gemini_price
                issues.append({
                    "field":           f"variants[{i}].launch_price",
                    "issue":           "launch_price_filled",
                    "raw_value":       None,
                    "corrected_value": gemini_price,
                })
                logger.info(
                    "_enrich_launch_prices: filled %s %s %dGB+%dGB → ₹%d",
                    brand, model_name, ram, storage, gemini_price,
                )
            elif llm_int == gemini_price:
                # Exact match — verified, no change
                logger.info(
                    "_enrich_launch_prices: verified %s %s %dGB+%dGB "
                    "₹%d exact match.",
                    brand, model_name, ram, storage, gemini_price,
                )
            else:
                # Any difference → replace
                variant["launch_price"] = gemini_price
                issues.append({
                    "field":           f"variants[{i}].launch_price",
                    "issue":           "launch_price_corrected",
                    "raw_value":       llm_price,
                    "corrected_value": gemini_price,
                })
                logger.info(
                    "_enrich_launch_prices: corrected %s %s %dGB+%dGB "
                    "₹%d → ₹%d",
                    brand, model_name, ram, storage, llm_int, gemini_price,
                )

        except GeminiNonRetryableError as exc:
            logger.warning(
                "_enrich_launch_prices: non-retryable failure for "
                "%s %s %dGB+%dGB: %s — keeping LLM value.",
                brand, model_name, ram, storage, exc,
            )
            issues.append({
                "field":           f"variants[{i}].launch_price",
                "issue":           "launch_price_enrichment_failed",
                "raw_value":       str(exc)[:200],
                "corrected_value": None,
            })
        except Exception as exc:
            logger.warning(
                "_enrich_launch_prices: unexpected failure for "
                "%s %s %dGB+%dGB: %s — keeping LLM value.",
                brand, model_name, ram, storage, exc,
            )
            issues.append({
                "field":           f"variants[{i}].launch_price",
                "issue":           "launch_price_enrichment_failed",
                "raw_value":       str(exc)[:200],
                "corrected_value": None,
            })

    # ── Mark price as verified in DB (fire-and-forget) ────────────────────
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
                "_enrich_launch_prices: failed to set price_verified_at "
                "for normalized_id=%d: %s", normalized_id, flag_exc
            )

    logger.info(
        "_enrich_launch_prices: COMPLETE %s %s — "
        "%d surviving variants, %d issues logged.",
        brand, model_name, len(variants), len(issues),
    )
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

    Handles: "₹20,999", "20,999 INR", "INR 20999", "20999", 20999 (int),
             20999.0 (float from Gemini grounded response).
    On parse failure: sets launch_price = None and logs a price_parse_failed issue.
    """
    issues: list[dict] = []
    variants: list[dict] = extracted_json.get("variants") or []
    for i, variant in enumerate(variants):
        raw = variant.get("launch_price")
        # Already a clean int — nothing to do
        if raw is None or isinstance(raw, int):
            continue
        # Whole-number float from Gemini (e.g. 20999.0) — cast directly
        if isinstance(raw, float):
            if raw.is_integer():
                variant["launch_price"] = int(raw)
            else:
                # Non-integer float is nonsensical for a price — null it
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
            continue
        # String with currency symbols / commas — strip and parse
        cleaned = _PRICE_CLEAN_RE.sub("", str(raw)).strip()
        # Handle float strings like "20999.0"
        try:
            as_float = float(cleaned)
            if as_float.is_integer():
                variant["launch_price"] = int(as_float)
                continue
        except (ValueError, TypeError):
            pass
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


# ─── Step 2.4 helpers ─────────────────────────────────────────────────────────

async def _verify_sim_dual_standby(
    data: dict,
    brand: str,
    model_name: str,
) -> list[dict]:
    """
    Step 2.4 — SIM dual standby verification via Gemini grounded search.

    Fired for ALL phones that have a Dual SIM configuration, regardless of
    tier. Verifies whether the phone supports Dual SIM Dual Standby (DSDS),
    then writes the canonical sim_configuration string.

    Canonical values after correction:
      DSDS confirmed → "Dual SIM (Nano-SIM, dual stand-by)"
      Not DSDS       → "Dual SIM (Nano-SIM)"  (rare; most modern phones are DSDS)
      eSIM variant   → leave unchanged (not a DSDS question)

    Only fires when sim_configuration is a string starting with "Dual SIM"
    and does NOT already contain "dual stand-by" (already canonical) or
    "eSIM" (separate eSIM handling, not a standby question).
    """
    issues: list[dict] = []
    network = data.get("network")
    if not isinstance(network, dict):
        return issues

    sim_config = network.get("sim_configuration")
    if not isinstance(sim_config, str):
        return issues

    # Skip if already canonical or eSIM variant
    sim_lower = sim_config.lower()
    if "dual stand-by" in sim_lower or "esim" in sim_lower.replace("-", ""):
        return issues
    if not sim_lower.startswith("dual sim"):
        return issues  # single SIM — not relevant

    prompt = (
        f"Does the {brand} {model_name} support Dual SIM Dual Standby (DSDS)? "
        f"DSDS means both SIM cards can simultaneously receive calls and data "
        f"without manually switching. Answer only YES or NO."
    )

    try:
        result = await call_gemini_grounded(
            prompt=prompt,
            output_schema=_SIM_STANDBY_SCHEMA,
        )
        answer = result.get("value")
        answer_clean = (answer or "").strip().upper()

        if answer_clean.startswith("YES"):
            canonical = "Dual SIM (Nano-SIM, dual stand-by)"
        else:
            canonical = "Dual SIM (Nano-SIM)"

        old_value = network["sim_configuration"]
        network["sim_configuration"] = canonical
        issues.append({
            "field":           "network.sim_configuration",
            "issue":           "sim_standby_verified",
            "raw_value":       old_value,
            "corrected_value": canonical,
        })
        logger.info(
            "_verify_sim_dual_standby: %s %s → %r (was %r)",
            brand, model_name, canonical, old_value,
        )
    except Exception as exc:
        logger.warning(
            "_verify_sim_dual_standby: Gemini call failed for %s %s: %s "
            "— sim_configuration left unchanged.",
            brand, model_name, exc,
        )

    return issues

