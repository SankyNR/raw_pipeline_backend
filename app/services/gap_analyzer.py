"""
Phase 5 — Gap Analysis  (Section 4 rewrite)

### Section 4 new flow (roadmap 4.1)

1.  Fetch normalized_spec_json
2.  Compute base variant launch_price → _classify_tier() → tier_id
3.  UPDATE pipeline.normalized_spec_json SET tier_id = X (fire-and-forget)
4.  _walk_nulls() — collect all null + empty-array paths
5.  Dependency filters — has_stylus, wireless_charging cascade (unchanged)
6.  For each gap path:
        Look up POLICY_CACHE[generic_path]
        If no policy row → fall back to legacy FIELD_PRIORITY_MAP (transitional)
        Switch on policy.policy:
          skip          → drop, no log row
          enrich_always → drop (handled by pre/post enrichment), no gap row
          flag_only     → log with is_flag_only=TRUE
          enrich        → check tier eligibility (phone_tier.sort_order ≤ field_min_tier.sort_order)
                          If threshold specified: measure value length (comma or array)
                          If threshold not met → drop
                          else → log normally (is_flag_only=FALSE)
7.  Type B candidate logic — unchanged

### Caches

_TIER_CACHE     — {tier_id: row_dict} — built from pipeline.lookup_price_tiers
_TIER_BY_ORDER  — sorted list of (sort_order, min_inr, max_inr, tier_id) for price→tier lookup
POLICY_CACHE    — {field_path: row_dict} — built from pipeline.gap_enrichment_policy

All three loaded once at app startup via warm_gap_caches() (called from build_lookup_cache).
Re-loadable via POST /extraction/cache/refresh.

### Design notes

- Supabase-py is synchronous; all DB calls wrapped in asyncio.to_thread().
- Wildcards in field paths resolved to concrete indices before logging.
- Idempotent: ON CONFLICT (normalized_id, field_path) DO UPDATE.
- Transitional fallback: if POLICY_CACHE is empty (e.g. startup failure),
  the legacy FIELD_PRIORITY_MAP path is used so the pipeline never silently drops gaps.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from app.config.field_mapping import (
    DEFAULT_FIELD_PRIORITY,
    FIELD_PRIORITY_MAP,
    FIELD_SITE_HINTS,
    JUNCTION_TABLE_FIELDS,
)
from app.repositories.extraction_repository import (
    fetch_gap_enrichment_policies,
    fetch_normalized_spec,
    fetch_price_tiers,
    fetch_url_registry_status,
    insert_missing_field_log,
    insert_type_b_gap_candidates,
    fetch_junction_table_existing_pks,
)
from app.repositories.pipeline_run_repository import update_normalized_spec_flag

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section 4 — In-memory caches
# ---------------------------------------------------------------------------

# Tier rows: {tier_id: full_row_dict}
_TIER_CACHE: dict[int, dict] = {}

# Sorted ascending by sort_order (lowest = most premium).
# Used by _classify_tier() to find the right tier for a price.
_TIER_BY_ORDER: list[dict] = []

# Policy rows: {field_path (wildcard) → full_row_dict}
POLICY_CACHE: dict[str, dict] = {}


def warm_gap_caches() -> None:
    """
    Section 4 — Loads price tier and gap enrichment policy rows into module caches.

    Called once at app startup from build_lookup_cache() (normalizer.py) after
    all lookup tables are loaded.  Re-callable on POST /extraction/cache/refresh.

    Synchronous — wrapped in asyncio.to_thread() by callers.
    """
    # Price tiers
    tiers = fetch_price_tiers()
    _TIER_CACHE.clear()
    _TIER_BY_ORDER.clear()
    for row in tiers:
        _TIER_CACHE[row["tier_id"]] = row
    # Sort ascending by sort_order (1 = Ultra Flagship … 6 = Entry Level)
    _TIER_BY_ORDER.extend(sorted(tiers, key=lambda r: r["sort_order"]))
    logger.info(
        "warm_gap_caches: loaded %d price tiers into _TIER_CACHE.", len(_TIER_CACHE)
    )

    # Gap enrichment policies
    policies = fetch_gap_enrichment_policies()
    POLICY_CACHE.clear()
    POLICY_CACHE.update(policies)
    logger.info(
        "warm_gap_caches: loaded %d policy rows into POLICY_CACHE.", len(POLICY_CACHE)
    )


# ---------------------------------------------------------------------------
# Section 4.2 — Tier classification
# ---------------------------------------------------------------------------

def _classify_tier(base_variant_price: int | None) -> int | None:
    """
    Returns tier_id from _TIER_BY_ORDER for the given launch price.
    Returns None if price is null, below 5 000, or _TIER_BY_ORDER is empty.

    Tiers are matched in sort_order order (most premium first):
      price >= min_inr AND (max_inr is null OR price <= max_inr)
    """
    if base_variant_price is None or base_variant_price < 5000:
        return None
    if not _TIER_BY_ORDER:
        return None
    for tier in _TIER_BY_ORDER:
        min_inr = tier["min_inr"]
        max_inr = tier.get("max_inr")  # None for Ultra Flagship (open upper bound)
        if base_variant_price >= min_inr and (max_inr is None or base_variant_price <= max_inr):
            return tier["tier_id"]
    return None


# ---------------------------------------------------------------------------
# Chipset vendor → site hint (unchanged from legacy)
# ---------------------------------------------------------------------------

_CHIPSET_VENDOR_HINTS: dict[str, str] = {
    "snapdragon": "qualcomm.com",
    "dimensity":  "mediatek.com",
    "apple":      "apple.com",
    "exynos":     "samsung.com",
    "tensor":     "store.google.com",
}


def _resolve_chipset_site_hint(chipset_name: str | None) -> str | None:
    if not chipset_name:
        return None
    lower = chipset_name.lower()
    for vendor, hint in _CHIPSET_VENDOR_HINTS.items():
        if vendor in lower:
            return hint
    return None


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def _generic_path(concrete_path: str) -> str:
    """
    Converts a concrete array-index path to a wildcard path for cache lookups.
    "displays[0].panel_type" → "displays[*].panel_type"
    """
    return re.sub(r"\[\d+\]", "[*]", concrete_path)


def _extract_variant_index(field_path: str) -> int | None:
    """
    Parses the numeric index from a 'variants[N].field' path string.

    Returns the integer index if the path starts with 'variants[N]',
    or None if the path doesn't match the expected pattern.

    Examples:
        "variants[0].virtual_ram_size"  → 0
        "variants[2].ram_frequency"     → 2
        "charging.wireless_charging"    → None
    """
    m = re.match(r"^variants\[(\d+)\]", field_path)
    return int(m.group(1)) if m else None


def _walk_nulls(data: Any, prefix: str = "") -> list[str]:
    """
    Recursively walks a nested dict/list. Returns a list of concrete dot-bracket
    paths for every null or empty-array leaf.
    """
    gaps: list[str] = []
    if isinstance(data, dict):
        for k, v in data.items():
            full_path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                gaps.extend(_walk_nulls(v, full_path))
            elif v is None:
                gaps.append(full_path)
    elif isinstance(data, list):
        if not data:
            gaps.append(prefix)
        else:
            for i, item in enumerate(data):
                gaps.extend(_walk_nulls(item, f"{prefix}[{i}]"))
    return gaps


# ---------------------------------------------------------------------------
# Section 4 — Threshold check helper
# ---------------------------------------------------------------------------

def _threshold_met(
    value: Any,
    threshold_count: int,
    threshold_op: str,
    comma_split: bool,
) -> bool:
    """
    Evaluates whether the threshold condition is satisfied (i.e. gap should be logged).
    Returns True if the field IS a gap (threshold says "enrich"), False if sufficient data.

    threshold_op='lt': log if length < threshold_count   (too few items extracted)

    comma_split=True: length = number of non-empty comma-split parts.
    comma_split=False: length = len(value) for lists, or 0 if value is null/non-iterable.
    """
    if comma_split:
        if isinstance(value, str):
            length = len([p for p in value.split(",") if p.strip()])
        else:
            length = 0
    else:
        if isinstance(value, list):
            length = len(value)
        else:
            length = 0  # null or wrong type → treat as empty

    if threshold_op == "lt":
        return length < threshold_count

    # Future-proof: other ops (gt, lte, gte) can be added here
    logger.warning("_threshold_met: unknown threshold_op=%r — treating as True.", threshold_op)
    return True


# ---------------------------------------------------------------------------
# Section 4.1 — detect_missing_fields (main public API)
# ---------------------------------------------------------------------------

async def detect_missing_fields(normalized_id: int) -> list[int]:
    """
    Section 4.1 — Detects null / empty-array fields in normalized_spec_json.
    Creates pipeline.missing_fields_log rows for each actionable gap.

    New (Section 4) vs legacy:
    - Classifies phone into a price tier and writes tier_id to normalized_spec_json.
    - Routes each null field through gap_enrichment_policy (POLICY_CACHE):
        skip          → silently dropped
        enrich_always → dropped (pre/post enrichment handles these)
        flag_only     → logged with is_flag_only=TRUE
        enrich        → logged only if phone_tier ≥ min_tier AND threshold satisfied
    - Falls back to FIELD_PRIORITY_MAP for any field_path not in POLICY_CACHE
      (transitional — will be empty once all fields are in the policy table).

    Args:
        normalized_id: pipeline.normalized_spec_json.normalized_id

    Returns:
        List of missing_field_id values created.

    Raises:
        ValueError: If normalized_id not found.
    """
    logger.info("detect_missing_fields: START normalized_id=%d", normalized_id)

    # --- 1. Fetch source data ---
    norm_row = await asyncio.to_thread(fetch_normalized_spec, normalized_id)
    normalized_json: dict = norm_row["normalized_json"] or {}
    url_registry_id: int  = norm_row["url_registry_id"]

    # --- 2. Classify tier from base variant price ---
    variants = normalized_json.get("variants") or []
    base_variant_price: int | None = next(
        (v.get("launch_price") for v in variants if v.get("is_base_variant") is True),
        None,
    )
    phone_tier_id: int | None = _classify_tier(base_variant_price)
    phone_tier_row: dict | None = _TIER_CACHE.get(phone_tier_id) if phone_tier_id else None

    logger.info(
        "detect_missing_fields: normalized_id=%d base_price=%s tier_id=%s (%s)",
        normalized_id,
        base_variant_price,
        phone_tier_id,
        (phone_tier_row or {}).get("tier_name", "unknown"),
    )

    # --- 3. Persist tier_id (fire-and-forget) ---
    if phone_tier_id is not None:
        try:
            await asyncio.to_thread(
                update_normalized_spec_flag, normalized_id, tier_id=phone_tier_id
            )
        except Exception as _tier_exc:
            logger.warning(
                "detect_missing_fields: failed to persist tier_id=%d for "
                "normalized_id=%d: %s", phone_tier_id, normalized_id, _tier_exc
            )

    # --- 4. Walk nulls ---
    gap_paths: list[str] = _walk_nulls(normalized_json)
    logger.info(
        "detect_missing_fields: normalized_id=%d found %d gaps before filtering",
        normalized_id, len(gap_paths),
    )

    # --- Supplementary data for dependency filters and site hints ---
    registry_row    = await asyncio.to_thread(fetch_url_registry_status, url_registry_id)
    registry_status = registry_row.get("status", "")
    chipset_name: str | None = (normalized_json.get("chipset") or {}).get("chipset_name")
    phone_is_stored = (registry_status == "stored_mainDB")

    created_ids: list[int] = []

    # --- 5–6. Filter and log gaps ---
    for concrete_path in gap_paths:
        generic = _generic_path(concrete_path)

        # ── Dependency filter: stylus_features requires has_stylus=True ──
        if generic == "body.stylus_features":
            if (normalized_json.get("body") or {}).get("has_stylus") is False:
                continue

        # ── Dependency filter: wireless charging sub-fields ──
        _WIRELESS_DEPS = {
            "charging.wireless_charging_power",
            "charging.wireless_charging_standard",
            "charging.reverse_wireless_charging",
            "charging.reverse_wireless_charging_power",
        }
        if generic in _WIRELESS_DEPS:
            if (normalized_json.get("charging") or {}).get("wireless_charging") is False:
                continue

        # ── Dependency filter: virtual_ram_size requires virtual_ram_availability=True ──
        if generic == "variants[*].virtual_ram_size":
            variant_idx = _extract_variant_index(concrete_path)
            if variant_idx is not None:
                all_variants = normalized_json.get("variants") or []
                parent_variant = all_variants[variant_idx] if variant_idx < len(all_variants) else {}
                if not parent_variant.get("virtual_ram_availability"):
                    continue  # skip — virtual RAM not available on this variant

        # ── Policy routing ──
        policy_row = POLICY_CACHE.get(generic)

        if policy_row:
            policy = policy_row["policy"]

            if policy in ("skip", "enrich_always"):
                # skip: not enriched, not logged.
                # enrich_always: handled by pre/post enrichment pass — no gap row needed.
                continue

            is_flag_only: bool = (policy == "flag_only")

            if policy == "enrich":
                # Tier eligibility: phone tier sort_order must be >= field's min_tier sort_order
                # (higher sort_order = cheaper tier; a Budget phone at sort=5 is eligible
                # for a min_tier=Entry Level (sort=6) but NOT for Premium (sort=3)).
                min_tier_id = policy_row.get("min_tier_id")
                if min_tier_id and phone_tier_row:
                    field_min_tier = _TIER_CACHE.get(min_tier_id)
                    if field_min_tier:
                        # phone sort_order must be <= field's min sort_order
                        # (lower sort_order = more premium → also eligible)
                        if phone_tier_row["sort_order"] > field_min_tier["sort_order"]:
                            continue  # phone too cheap for this field — skip
                elif min_tier_id and not phone_tier_row:
                    # Price unknown → can't confirm tier eligibility — skip to be safe
                    continue

                # Threshold check (only when threshold_count is set)
                threshold_count = policy_row.get("threshold_count")
                if threshold_count is not None:
                    threshold_op  = policy_row.get("threshold_op", "lt")
                    comma_split   = policy_row.get("comma_split", False)
                    # Need to retrieve the current (possibly partial) value for threshold eval
                    _value = _get_nested(normalized_json, concrete_path)
                    if not _threshold_met(_value, threshold_count, threshold_op, comma_split):
                        continue  # value already has enough data — gap not needed

            # Site hint: prefer policy table's preferred_site_hint, then legacy FIELD_SITE_HINTS
            site_hint = policy_row.get("preferred_site_hint") or FIELD_SITE_HINTS.get(generic)

        else:
            # ── Transitional fallback: field not yet in POLICY_CACHE ──
            priority = FIELD_PRIORITY_MAP.get(generic, DEFAULT_FIELD_PRIORITY)
            if priority == "skip":
                continue
            is_flag_only = False
            site_hint    = FIELD_SITE_HINTS.get(generic)
            if generic == "chipset.npu_tops":
                site_hint = _resolve_chipset_site_hint(chipset_name)

        # ── Insert missing_fields_log row (idempotent) ──
        log_payload = {
            "normalized_id":          normalized_id,
            "url_registry_id":        url_registry_id,
            "field_path":             concrete_path,
            "missing_type":           "type_b" if generic in JUNCTION_TABLE_FIELDS else "type_a",
            "priority":               FIELD_PRIORITY_MAP.get(generic, DEFAULT_FIELD_PRIORITY),
            "preferred_site_hint":    site_hint,
            "query_template_override": None,
            "is_flag_only":           is_flag_only,
        }

        try:
            missing_field_id = await asyncio.to_thread(
                insert_missing_field_log, log_payload
            )
            if missing_field_id is not None:
                created_ids.append(missing_field_id)
        except Exception as exc:
            logger.error(
                "detect_missing_fields: failed to log gap %r for normalized_id=%d: %s",
                concrete_path, normalized_id, exc,
            )
            continue

        # ── Type B gap candidates (only if phone already committed to mainDB) ──
        if log_payload["missing_type"] == "type_b" and phone_is_stored and missing_field_id is not None:
            await _create_type_b_candidates(
                missing_field_id=missing_field_id,
                url_registry_id=url_registry_id,
                generic_path=generic,
            )

    logger.info(
        "detect_missing_fields: COMPLETE normalized_id=%d created %d log rows",
        normalized_id, len(created_ids),
    )
    return created_ids


# ---------------------------------------------------------------------------
# Helper: resolve a concrete path to its current value in normalized_json
# ---------------------------------------------------------------------------

def _get_nested(data: Any, concrete_path: str) -> Any:
    """
    Traverses data using a concrete dot-bracket path and returns the value at that path.
    Returns None if the path doesn't exist or any intermediate node is None.

    Examples:
        "charging.wireless_charging" → data["charging"]["wireless_charging"]
        "displays[0].display_features" → data["displays"][0]["display_features"]
    """
    parts = re.split(r"\.(?![^[]*\])", concrete_path)  # split on '.' not inside brackets
    current: Any = data
    for part in parts:
        if current is None:
            return None
        m = re.match(r"^(\w+)\[(\d+)\]$", part)
        if m:
            key, idx = m.group(1), int(m.group(2))
            current = current.get(key) if isinstance(current, dict) else None
            if isinstance(current, list) and idx < len(current):
                current = current[idx]
            else:
                return None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


# ---------------------------------------------------------------------------
# Internal: Type B gap candidate creation (unchanged)
# ---------------------------------------------------------------------------

async def _create_type_b_candidates(
    missing_field_id: int,
    url_registry_id: int,
    generic_path: str,
) -> None:
    """
    For Type B (junction table) gaps on phones already committed to mainDB:
    Queries what PKs already exist for this phone in the junction table,
    then creates type_b_gap_candidates rows so the enrichment layer can
    verify each candidate against the specific phone.

    Best-effort — errors are logged and swallowed.
    """
    table_path = JUNCTION_TABLE_FIELDS.get(generic_path)
    if not table_path:
        return

    try:
        existing_rows: list[dict] = await asyncio.to_thread(
            fetch_junction_table_existing_pks,
            url_registry_id,
            generic_path,
            table_path,
        )
        if not existing_rows:
            return

        candidates = [
            {
                "missing_field_id": missing_field_id,
                "url_registry_id":  url_registry_id,
                "candidate_value":  row["canonical_value"],
                "lookup_table":     table_path,
                "lookup_row_id":    row["pk"],
                "inclusion_reason": "existing_maindb",
            }
            for row in existing_rows
        ]
        await asyncio.to_thread(insert_type_b_gap_candidates, candidates)
        logger.debug(
            "_create_type_b_candidates: field=%r %d candidates inserted",
            generic_path, len(candidates),
        )

    except Exception as exc:
        logger.warning(
            "_create_type_b_candidates: failed for field=%r missing_field_id=%d: %s",
            generic_path, missing_field_id, exc,
        )
