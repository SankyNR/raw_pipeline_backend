"""
Phase 3 — spec_json_builder.py (v5 Rewrite)

Converts the v5 Gemini JSON output dict into the nested partial_json and
evidence_json structures expected by the extraction pipeline.

REWRITE NOTES (v4 → v5)
------------------------
v4 operated on list[lx.data.Extraction] objects produced by LangExtract. Each
object carried .char_interval, .extraction_class, .attributes, and .extraction_text.
Source attribution was computed by mapping char_interval.start_pos against
pre-built source_section_offsets ranges (_get_source_for_char_position).

v5 operates on the raw dict output from call_gemini_json(), which already matches
RunAExtractionSchema. Each non-null scalar field is wrapped:
    {"value": X, "_source": {"raw_id": N, "evidence_text": "..."}}
    {"value": X, "_source": {"raw_transcript_id": N, "evidence_text": "..."}}

Junction table arrays (display_features, bands_5g, etc.) and structural fields
(display_type, display_position, lens_type, is_base_variant) are plain values
with no _source wrapper.

Two-layer evidence construction:
  Layer 1 — parse raw_id/raw_transcript_id + evidence_text from inline _source tags
  Layer 2 — locate evidence_text in assembled_source_string via str.find() to
             compute char_start/char_end for admin UI click-to-highlight

Removed from v4:
  - All lx.data.Extraction references
  - .char_interval, .extraction_class, .attributes, .extraction_text
  - _get_source_for_char_position()
  - source_section_offsets parameter
  - raw_source_ids_by_site parameter
  - _build_source_section_offsets() (belonged to langextract_run_a.py)

Retained from v4 (logic only, reimplemented):
  - Direct assignment for single-entity sections (Gemini guarantees a single JSON object per run)
  - Index-grouped assembly for variants[], displays[], camera_lenses[]
  - Append-all + dedup for extra_features[] and in_the_box[]
  - _strip_run_c_fields_from_partial()
  - _check_array_completeness()
  - _get_at_path()
  - _ARRAY_MINIMUMS
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.config.field_mapping import RUN_C_CALCULATED_FIELDS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structural fields — plain values, no _source tag, no evidence entry
# ---------------------------------------------------------------------------

_STRUCTURAL_FIELDS: frozenset[str] = frozenset({
    "display_type",
    "display_position",
    "lens_type",
    "is_base_variant",
})

# ---------------------------------------------------------------------------
# Section 9.5 — Array completeness monitor thresholds (unchanged from v4)
# ---------------------------------------------------------------------------

_ARRAY_MINIMUMS: dict[str, int] = {
    "network.bands_4g":             6,   # typical flagship: 18–22
    "network.bands_5g":             4,   # typical flagship: 10–17
    "displays[0].display_features": 2,   # flagship: usually >= 6
    "camera_lenses":                1,   # flag total absence only
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """
    Collapses all whitespace sequences to a single space and strips ends.
    Applied to both the assembled_source_string search target and the
    evidence_text before str.find() to handle multi-line content.
    """
    return re.sub(r'\s+', ' ', text).strip()


def _get_at_path(obj: dict, path: str) -> Any:
    """
    Reads a simple dotted or bracket path from a nested dict.
    Supports: "network.bands_4g", "displays[0].display_features", "camera_lenses".
    Returns None if the path does not exist.
    (Unchanged from v4.)
    """
    try:
        if "[" in path:
            bracket_start = path.index("[")
            bracket_end = path.index("]")
            parent_key = path[:bracket_start]
            idx = int(path[bracket_start + 1:bracket_end])
            rest = path[bracket_end + 2:] if bracket_end + 2 < len(path) else None
            collection = obj.get(parent_key)
            if not isinstance(collection, list) or idx >= len(collection):
                return None
            item = collection[idx]
            if rest:
                return item.get(rest) if isinstance(item, dict) else None
            return item
        elif "." in path:
            parent, child = path.split(".", 1)
            return _get_at_path(obj.get(parent, {}), child)
        else:
            return obj.get(path)
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _check_array_completeness(partial_json: dict, url_registry_id: int | None) -> None:
    """
    Logs warnings if extracted arrays fall below heuristic minimums.
    Does NOT modify partial_json. Never blocks the pipeline.
    (Unchanged from v4.)
    """
    for path, min_len in _ARRAY_MINIMUMS.items():
        value = _get_at_path(partial_json, path)
        if isinstance(value, list) and len(value) < min_len:
            logger.warning(
                "Array completeness: url_registry_id=%s path=%r has %d item(s) "
                "(expected >= %d). Possible extraction truncation or missing source. "
                "Review example coverage if this appears systematically.",
                url_registry_id, path, len(value), min_len,
            )


# Junction arrays that are high-risk for hallucination from training data.
# For these fields, at least one value should appear in the source text.
_HIGH_RISK_JUNCTION_ARRAYS: dict[str, str] = {
    "audio.audio_codecs":               "audio",
    "connectivity.wifi_technologies":   "connectivity",
    "sensors.other_sensors":            "sensors",
}


def _check_junction_array_grounding(
    partial_json: dict,
    assembled_source_string: str,
    url_registry_id: int | None,
) -> None:
    """
    For high-risk junction arrays, warns if none of the extracted values
    appear anywhere in the assembled source string.

    A junction array where zero values appear in the source is a strong
    signal that the model hallucinated the list from training knowledge.
    This does not block the pipeline — it surfaces the problem for review.
    """
    if not assembled_source_string:
        return

    source_lower = assembled_source_string.lower()

    for field_path, _section_key in _HIGH_RISK_JUNCTION_ARRAYS.items():
        # Navigate to the value
        parts = field_path.split(".")
        obj = partial_json
        for part in parts:
            if not isinstance(obj, dict):
                obj = None
                break
            obj = obj.get(part)

        if not isinstance(obj, list) or not obj:
            continue  # empty or missing — nothing to check

        # Check if at least one value appears in the source
        found_any = any(
            isinstance(v, str) and v.lower() in source_lower
            for v in obj
        )

        if not found_any:
            logger.warning(
                "Junction array grounding check: NONE of the %d values in "
                "'%s' were found in the assembled source string. "
                "This strongly suggests hallucination from training data. "
                "url_registry_id=%s. Values: %s",
                len(obj), field_path, url_registry_id,
                obj[:5],  # show first 5 items only
            )


def _strip_run_c_fields_from_partial(partial_json: dict) -> dict:
    """
    Removes any RUN_C_CALCULATED_FIELDS the model may have included.
    Paths:
        "displays[*].ppi"                      → remove from each displays item
        "camera_lenses[*].sensor_size_decimal" → remove from each camera_lenses item
    (Unchanged from v4.)
    """
    for path in RUN_C_CALCULATED_FIELDS:
        if "[*]" in path:
            collection_key, rest = path.split("[*].", 1)
            leaf_key = rest.rsplit(".", 1)[-1]
            collection = partial_json.get(collection_key)
            if isinstance(collection, list):
                for item in collection:
                    if isinstance(item, dict):
                        item.pop(leaf_key, None)
        else:
            leaf_key = path.rsplit(".", 1)[-1]
            partial_json.pop(leaf_key, None)
    return partial_json


# ---------------------------------------------------------------------------
# v5 core helpers — two-layer evidence extraction
# ---------------------------------------------------------------------------

def _extract_value_and_source(field_value: Any) -> tuple[Any, dict | None]:
    """
    If field_value is {"value": X, "_source": {...}}, extract both.
    Otherwise treat the whole value as the plain value (list, scalar, None).

    Returns:
        (actual_value, source_dict | None)

    source_dict examples:
        {"raw_id": 15, "evidence_text": "..."}
        {"raw_transcript_id": 7, "evidence_text": "..."}
    """
    if isinstance(field_value, dict) and "value" in field_value:
        return field_value["value"], field_value.get("_source")
    return field_value, None


def _build_evidence_entry(
    source_dict: dict | None,
    assembled_source_string: str,
) -> dict | None:
    """
    Builds the final evidence_json entry including char offsets.

    Layer 1: parse raw_id/raw_transcript_id + evidence_text from source_dict.
    Layer 2: locate evidence_text in assembled_source_string via str.find().

    Normalization: before matching, both strings are normalized via
    _normalize_text() to handle multi-line source content. If normalized match 
    succeeds, offsets are NOT set — they cannot be reliably mapped back to 
    original string positions. Tooltip works; click-to-highlight disabled gracefully.
    If str.find() returns -1 (model paraphrased), char_start and char_end are
    None. Hover tooltip still works; click-to-highlight disabled gracefully.

    Returns None if source_dict is None (ungrounded field).
    """
    if source_dict is None:
        return None

    evidence_text: str | None = source_dict.get("evidence_text")
    char_start: int | None = None
    char_end: int | None = None

    if evidence_text and assembled_source_string:
        # Attempt 1: exact match
        idx = assembled_source_string.find(evidence_text)
        if idx != -1:
            char_start = idx
            char_end = idx + len(evidence_text)
        else:
            # Attempt 2: normalized match
            norm_source = _normalize_text(assembled_source_string)
            norm_evidence = _normalize_text(evidence_text)
            idx_norm = norm_source.find(norm_evidence)

            if idx_norm != -1:
                # Normalized match confirms extraction is correct but offsets
                # cannot be reliably mapped back to original string positions.
                # Do NOT assign char_start / char_end.
                logger.debug(
                    "_build_evidence_entry: normalized match succeeded for "
                    "evidence_text=%r. Offsets not set to avoid incorrect highlighting.",
                    evidence_text[:80],
                )
            else:
                logger.warning(
                    "_build_evidence_entry: str.find() returned -1 for "
                    "evidence_text=%r (both exact and normalized). "
                    "Model may have paraphrased. char_start/char_end set to None.",
                    evidence_text[:80] if evidence_text else "",
                )

    raw_id = source_dict.get("raw_id")
    raw_transcript_id = source_dict.get("raw_transcript_id")

    # Guard: both null means the model emitted _source without a valid source reference
    if raw_id is None and raw_transcript_id is None:
        logger.warning(
            "_build_evidence_entry: _source has both raw_id=null and "
            "raw_transcript_id=null for evidence_text=%r. Skipping evidence entry.",
            (source_dict.get("evidence_text") or "")[:80],
        )
        return None

    if raw_id is not None:
        return {
            "evidence_text":     evidence_text,
            "char_start":        char_start,
            "char_end":          char_end,
            "source_type":       "scraped",
            "raw_id":            raw_id,
            "raw_transcript_id": None,
            "grounded":          True,
        }
    if raw_transcript_id is not None:
        return {
            "evidence_text":     evidence_text,
            "char_start":        char_start,
            "char_end":          char_end,
            "source_type":       "transcript",
            "raw_id":            None,
            "raw_transcript_id": raw_transcript_id,
            "grounded":          True,
        }

    # source_dict exists but has neither key — treat as ungrounded
    logger.warning(
        "_build_evidence_entry: source_dict has neither 'raw_id' nor "
        "'raw_transcript_id': %r. Treating as ungrounded.",
        source_dict,
    )
    return None


# ---------------------------------------------------------------------------
# Section walkers
# ---------------------------------------------------------------------------

def _walk_section(
    section_dict: dict | None,
    section_key: str,
    assembled_source_string: str,
    partial_json: dict,
    evidence_json: dict,
) -> None:
    """
    Walks a single-entity section dict (e.g. chipset, body, charging).

    For each leaf field:
      - Calls _extract_value_and_source() to split value from _source.
      - Plain list values (junction arrays) are written directly; no evidence entry.
      - Attributed scalar values produce an evidence entry at "section_key.field_key".
      - Structural fields (display_type etc.) are written as plain values; no evidence.

    Writes results into partial_json[section_key] and evidence_json.
    """
    if section_dict is None or not isinstance(section_dict, dict):
        partial_json[section_key] = {}
        return

    section_out: dict = {}

    for field_key, field_value in section_dict.items():
        # Structural fields — plain value, no evidence
        if field_key in _STRUCTURAL_FIELDS:
            section_out[field_key] = field_value
            continue

        value, source_dict = _extract_value_and_source(field_value)

        # Junction arrays are plain lists — write directly, no evidence
        if isinstance(value, list):
            section_out[field_key] = value
            continue

        section_out[field_key] = value

        if value is not None and source_dict is not None:
            ev = _build_evidence_entry(source_dict, assembled_source_string)
            if ev is not None:
                field_path = f"{section_key}.{field_key}"
                evidence_json[field_path] = ev

    partial_json[section_key] = section_out


def _walk_array_section(
    array_list: list | None,
    schema_key: str,
    assembled_source_string: str,
    partial_json: dict,
    evidence_json: dict,
) -> None:
    """
    Walks an indexed array section (variants[], displays[], camera_lenses[]).

    For each item at index idx:
      - Structural fields → plain value, no evidence.
      - Junction array fields → plain list, no evidence.
      - Attributed scalar fields → evidence entry at "schema_key[idx].field_key".

    Writes results into partial_json[schema_key] and evidence_json.
    """
    if not array_list or not isinstance(array_list, list):
        partial_json[schema_key] = []
        return

    items_out: list[dict] = []

    for idx, item in enumerate(array_list):
        if not isinstance(item, dict):
            continue

        item_out: dict = {}

        for field_key, field_value in item.items():
            # Structural fields — plain value, no evidence
            if field_key in _STRUCTURAL_FIELDS:
                item_out[field_key] = field_value
                continue

            value, source_dict = _extract_value_and_source(field_value)

            # Junction arrays are plain lists — write directly, no evidence
            if isinstance(value, list):
                item_out[field_key] = value
                continue

            item_out[field_key] = value

            if value is not None and source_dict is not None:
                ev = _build_evidence_entry(source_dict, assembled_source_string)
                if ev is not None:
                    field_path = f"{schema_key}[{idx}].{field_key}"
                    evidence_json[field_path] = ev

        items_out.append(item_out)

    partial_json[schema_key] = items_out


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

#: Single-entity top-level section keys in the Gemini output dict.
_SINGLE_ENTITY_SECTIONS: tuple[str, ...] = (
    "brand",
    "phone_identity",
    "chipset",
    "body",
    "charging",
    "audio",
    "sensors",
    "connectivity",
    "network",
    "camera_overview",
    "os_and_security",
    "certifications",
    "ai_capabilities",
    "video_capabilities",
    "performance_benchmarks",
)

#: Indexed array section keys in the Gemini output dict.
_INDEXED_ARRAY_SECTIONS: tuple[str, ...] = (
    "variants",
    "displays",
    "camera_lenses",
)


def build_spec_json(
    raw_output: dict,
    raw_source_ids: list[int],
    raw_transcript_ids: list[int],
    assembled_source_string: str,
    url_registry_id: int | None = None,
) -> tuple[dict, dict]:
    """
    Converts the v5 Gemini JSON output dict → nested partial_json + evidence_json.

    Args:
        raw_output:              Raw dict from call_gemini_json() matching
                                 RunAExtractionSchema. Already Pydantic-validated
                                 before this function is called.
        raw_source_ids:          All raw_scraped_data.raw_id values included in
                                 the assembled source. Used for audit/logging only;
                                 attribution comes from inline _source tags.
        raw_transcript_ids:      All raw_transcript_id values included in the run
                                 (up to 3). Used for audit/logging only.
        assembled_source_string: The full assembled source string passed to
                                 call_gemini_json(). Required for str.find()
                                 char offset computation.
        url_registry_id:         Optional — used only for warning log messages.

    Returns:
        partial_json:   Nested spec JSON dict matching spec_template.yaml shape.
                        All _source tags are stripped. Run C calculated fields
                        (ppi, sensor_size_decimal) are stripped.
        evidence_json:  {field_path: evidence_entry} where evidence_entry contains:
                        {evidence_text, char_start, char_end, source_type,
                         raw_id, raw_transcript_id, grounded}.
                        Ungrounded fields (no _source) are absent from this dict.

    Raises:
        Nothing — all errors are logged as warnings and the builder returns what
        it can. The pipeline never crashes on extraction quality issues.
    """
    # -------------------------------------------------------------------------
    # Guard: empty output
    # -------------------------------------------------------------------------
    if not raw_output:
        logger.warning(
            "build_spec_json: url_registry_id=%s — raw_output is empty. "
            "This is likely a Gemini call failure or empty JSON response. "
            "Returning empty dicts.",
            url_registry_id,
        )
        return {}, {}

    partial_json: dict = {}
    evidence_json: dict = {}

    # -------------------------------------------------------------------------
    # Step 1 — Single-entity sections
    # -------------------------------------------------------------------------
    for section_key in _SINGLE_ENTITY_SECTIONS:
        _walk_section(
            section_dict=raw_output.get(section_key),
            section_key=section_key,
            assembled_source_string=assembled_source_string,
            partial_json=partial_json,
            evidence_json=evidence_json,
        )

    # -------------------------------------------------------------------------
    # Step 2 — Indexed array sections
    # -------------------------------------------------------------------------
    for schema_key in _INDEXED_ARRAY_SECTIONS:
        _walk_array_section(
            array_list=raw_output.get(schema_key),
            schema_key=schema_key,
            assembled_source_string=assembled_source_string,
            partial_json=partial_json,
            evidence_json=evidence_json,
        )

    # -------------------------------------------------------------------------
    # Step 3 — extra_features[] — plain list[str], no evidence entries
    # Junction array — written directly with deduplication.
    # -------------------------------------------------------------------------
    raw_extra = raw_output.get("extra_features")
    if isinstance(raw_extra, list):
        seen_features: set[str] = set()
        deduped_features: list[str] = []
        for item in raw_extra:
            if isinstance(item, str) and item not in seen_features:
                seen_features.add(item)
                deduped_features.append(item)
        partial_json["extra_features"] = deduped_features
    else:
        partial_json["extra_features"] = []

    # -------------------------------------------------------------------------
    # Step 3b — camera_features[] — top-level plain list[str], no evidence entries.
    # Change 7: camera_features has moved from camera_lenses[*].camera_features
    # to a single root-level junction array (same pattern as extra_features).
    # -------------------------------------------------------------------------
    raw_cam_feats = raw_output.get("camera_features")
    if isinstance(raw_cam_feats, list):
        seen_cam: set[str] = set()
        deduped_cam: list[str] = []
        for item in raw_cam_feats:
            if isinstance(item, str) and item not in seen_cam:
                seen_cam.add(item)
                deduped_cam.append(item)
        partial_json["camera_features"] = deduped_cam
    else:
        partial_json["camera_features"] = []

    # -------------------------------------------------------------------------
    # Step 4 — in_the_box[] — structured objects with attribution
    # Dedup by item_name. item_name and item_specification may be attributed;
    # quantity is always a plain int.
    # -------------------------------------------------------------------------
    raw_itb = raw_output.get("in_the_box")
    in_the_box: list[dict] = []
    seen_item_names: set[str] = set()

    if isinstance(raw_itb, list):
        for item in raw_itb:
            if not isinstance(item, dict):
                continue

            # item_name
            item_name_field = item.get("item_name")
            item_name_value, item_name_source = _extract_value_and_source(item_name_field)

            if not item_name_value or item_name_value in seen_item_names:
                continue
            seen_item_names.add(item_name_value)

            # item_specification
            item_spec_field = item.get("item_specification")
            item_spec_value, item_spec_source = _extract_value_and_source(item_spec_field)

            # quantity — plain int, no _source
            quantity = item.get("quantity", 1)
            if not isinstance(quantity, int):
                quantity = 1

            idx = len(in_the_box)
            in_the_box.append({
                "item_name":          item_name_value,
                "item_specification": item_spec_value,
                "quantity":           quantity,
            })

            # Evidence for item_name
            if item_name_source is not None:
                ev = _build_evidence_entry(item_name_source, assembled_source_string)
                if ev is not None:
                    evidence_json[f"in_the_box[{idx}].item_name"] = ev

            # Evidence for item_specification
            if item_spec_value is not None and item_spec_source is not None:
                ev = _build_evidence_entry(item_spec_source, assembled_source_string)
                if ev is not None:
                    evidence_json[f"in_the_box[{idx}].item_specification"] = ev

    partial_json["in_the_box"] = in_the_box

    # -------------------------------------------------------------------------
    # Step 5 — Strip Run C calculated fields
    # -------------------------------------------------------------------------
    partial_json = _strip_run_c_fields_from_partial(partial_json)

    # -------------------------------------------------------------------------
    # Step 6 — Ungrounded rate check
    # Count non-null scalar values across partial_json that have no evidence entry.
    # If > 20%, log a warning recommending prompt/example quality review.
    # -------------------------------------------------------------------------
    _check_ungrounded_rate(partial_json, evidence_json, url_registry_id)

    # -------------------------------------------------------------------------
    # Step 7 — Array completeness monitor (log-only, never blocking)
    # -------------------------------------------------------------------------
    _check_array_completeness(partial_json, url_registry_id)

    # -------------------------------------------------------------------------
    # Step 8 — Junction array source sanity check (log-only, never blocking)
    # For high-risk junction arrays that are prone to hallucination from training
    # data, check whether ANY of the extracted values appear in the
    # assembled_source_string. If none match, log a critical warning.
    # This does not modify partial_json — it only surfaces problems for review.
    # -------------------------------------------------------------------------
    _check_junction_array_grounding(partial_json, assembled_source_string, url_registry_id)

    logger.info(
        "build_spec_json: built partial_json with %d top-level keys, "
        "%d evidence entries. url_registry_id=%s raw_source_ids=%s raw_transcript_ids=%s",
        len(partial_json), len(evidence_json), url_registry_id,
        raw_source_ids, raw_transcript_ids,
    )

    return partial_json, evidence_json


# ---------------------------------------------------------------------------
# Ungrounded rate check
# ---------------------------------------------------------------------------

def _collect_non_null_scalar_paths(obj: Any, prefix: str, paths: set[str]) -> None:
    """
    Recursively collects dotted field paths for all non-null, non-list, non-dict
    leaf values in a nested structure. Used to count total attributed fields.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            # Structural fields are intentionally ungrounded
            if k in _STRUCTURAL_FIELDS:
                continue
            _collect_non_null_scalar_paths(v, f"{prefix}.{k}" if prefix else k, paths)

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            # Skip plain string items (junction arrays)
            if isinstance(v, str):
                continue
            _collect_non_null_scalar_paths(v, f"{prefix}[{i}]", paths)

    elif obj is not None:
        paths.add(prefix)


def _check_ungrounded_rate(
    partial_json: dict,
    evidence_json: dict,
    url_registry_id: int | None,
) -> None:
    """
    If more than 20% of non-null scalar values lack a evidence_json entry,
    logs a warning recommending prompt and example quality review.
    """
    all_paths: set[str] = set()
    _collect_non_null_scalar_paths(partial_json, "", all_paths)

    total = len(all_paths)
    if total == 0:
        return

    grounded = len(evidence_json)
    ungrounded = total - grounded
    if ungrounded < 0:
        ungrounded = 0

    pct = (ungrounded / total) * 100
    if pct > 20.0:
        logger.warning(
            "build_spec_json: ungrounded rate %.0f%% (%d/%d fields lack _source). "
            "url_registry_id=%s. "
            "Recommend reviewing prompt instructions and few-shot example coverage.",
            pct, ungrounded, total, url_registry_id,
        )
