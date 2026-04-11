"""
Phase L3 â€” spec_json_builder.py

Converts LangExtract's flat Extraction list into the nested partial_json
and evidence_json structures expected by the extraction pipeline.

This module replaces the role of evidence_utils.split_extraction_response()
for LangExtract-sourced data. It does NOT replace evidence_utils.py itself â€”
that file remains until Phase L7 for the legacy orchestrator path.

DESIGN (Section 9 of langextract_migration_v4.md)
--------------------------------------------------

Input:
  extractions          : list[lx.data.Extraction]  â€” flat list from lx.extract()
  source_text          : str                       â€” the full merged input text
  raw_source_ids       : list[int]                 â€” all raw_scraped_data.raw_id values
  raw_transcript_id    : int | None
  source_section_offsets: dict[str, tuple[int,int]] â€” {"site_name": (start, end), ...}
                           Maps each source's char range in the merged string.
                           Used to attribute char_interval positions to a raw_id or transcript.

Output:
  partial_json   : dict  â€” nested spec JSON matching spec_template.yaml structure
  evidence_json  : dict  â€” {field_path: {evidence_text, source_type, raw_id, grounded}, ...}

Merge algorithm (Section 9.4):
  1. Sort extractions by char_interval.start_pos ascending (None â†’ last).
     This ensures OEM content (chars 0â€“12k) always wins over GSMArena on conflicts.
  2. Single-entity sections: first-seen-wins per attribute key.
  3. Indexed array sections (variant, display, camera_lens): group by index attribute,
     first-seen-wins per (index, attribute key).
  4. extra_feature: append-all, deduplicate by feature_name value.
  5. in_the_box_item: append-all structured objects, deduplicate by item_name.

char_interval safety (Section 9.3):
  If char_interval is None (extraction ungrounded), the field is still included in
  partial_json but receives no evidence_json entry. A warning is logged.
  Caller sees "grounded": false for these fields in the admin UI (greyed-out hover).

Array completeness monitor (Section 9.5):
  Heuristic minimums logged as warnings after build, never blocking.
"""

import logging
from collections import defaultdict

from app.config.field_mapping import RUN_C_CALCULATED_FIELDS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section 9.1 â€” extraction_class â†’ schema section mapping
# ---------------------------------------------------------------------------

# Single-entity sections: one dict merged from all extractions of this class.
SINGLE_ENTITY_CLASSES: dict[str, str] = {
    "brand":             "brand",
    "phone_identity":    "phone_identity",
    "chipset":           "chipset",
    "body":              "body",
    "charging":          "charging",
    "audio":             "audio",
    "sensors":           "sensors",
    "connectivity":      "connectivity",
    "network":           "network",
    "camera_overview":   "camera_overview",
    "os_and_security":   "os_and_security",
    "certifications":    "certifications",
    "ai_capabilities":   "ai_capabilities",
    "video_capabilities": "video_capabilities",
}

# Indexed array sections: group by numeric index attribute, produce a sorted list.
INDEXED_ARRAY_CLASSES: dict[str, tuple[str, str]] = {
    # extraction_class -> (schema_key, index_attribute_name)
    "variant":     ("variants",      "variant_index"),
    "display":     ("displays",      "display_index"),
    "camera_lens": ("camera_lenses", "lens_index"),
}


# ---------------------------------------------------------------------------
# Section 9.5 â€” Array completeness monitor thresholds
# ---------------------------------------------------------------------------

_ARRAY_MINIMUMS: dict[str, int] = {
    "network.bands_4g":             6,   # typical flagship: 18â€“22
    "network.bands_5g":             4,   # typical flagship: 10â€“17
    "displays[0].display_features": 2,   # flagship: usually â‰¥ 6
    "camera_lenses":                1,   # flag total absence only (single-lens phones are valid)
    "audio.audio_codecs":           3,   # basic flag for total absence
}


def _get_at_path(obj: dict, path: str):
    """
    Reads a simple dotted or bracket path from a nested dict.
    Supports: "network.bands_4g", "displays[0].display_features", "camera_lenses".
    Returns None if the path does not exist.
    """
    try:
        if "[" in path:
            # e.g. "displays[0].display_features"
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


# ---------------------------------------------------------------------------
# Section 9.3 â€” char_interval â†’ evidence attribution
# ---------------------------------------------------------------------------

def _get_source_for_char_position(
    start_pos: int,
    source_section_offsets: dict[str, tuple[int, int]],
    raw_source_ids_by_site: dict[str, int],
    raw_transcript_id: int | None,
) -> tuple[str, int | None]:
    """
    Resolves a char_interval.start_pos to (source_type, source_id).

    source_type: "scraped" | "transcript"
    source_id:   raw_id (int) for scraped, raw_transcript_id for transcript, None if unknown.

    Falls back to the first raw_source_id if no offset range matches.
    """
    for site_name, (start, end) in source_section_offsets.items():
        if start <= start_pos <= end:
            if site_name == "transcript":
                return "transcript", raw_transcript_id
            raw_id = raw_source_ids_by_site.get(site_name)
            return "scraped", raw_id

    # Fallback â€” position not in any known range (shouldn't happen in normal operation)
    logger.warning(
        "_get_source_for_char_position: start_pos=%d not in any offset range %s. "
        "Falling back to first available raw source.",
        start_pos, list(source_section_offsets.keys()),
    )
    if raw_source_ids_by_site:
        first_id = next(iter(raw_source_ids_by_site.values()))
        return "scraped", first_id
    return "scraped", None


def _build_evidence_entry(
    extraction,
    source_section_offsets: dict[str, tuple[int, int]],
    raw_source_ids_by_site: dict[str, int],
    raw_transcript_id: int | None,
) -> dict | None:
    """
    Builds one evidence dict for a single extraction, or returns None if ungrounded.

    Ungrounded (char_interval=None) â†’ field is in partial_json but NOT in evidence_json.
    The admin UI renders these as greyed-out hover entries (existing behaviour, no crash).
    """
    if extraction.char_interval is None:
        logger.warning(
            "Ungrounded extraction: class=%r text=%r â€” char_interval is None. "
            "Field included in partial_json but excluded from evidence_json.",
            extraction.extraction_class,
            extraction.extraction_text[:60],
        )
        return None

    start_pos = extraction.char_interval.start_pos
    source_type, source_id = _get_source_for_char_position(
        start_pos,
        source_section_offsets,
        raw_source_ids_by_site,
        raw_transcript_id,
    )

    return {
        "evidence_text":       extraction.extraction_text,
        "source_type":         source_type,
        "raw_id":              source_id if source_type == "scraped" else None,
        "raw_transcript_id":   source_id if source_type == "transcript" else None,
        "grounded":            True,
    }


# ---------------------------------------------------------------------------
# RUN_C field path stripper (preserves path-awareness from extraction_orchestrator.py)
# ---------------------------------------------------------------------------

def _strip_run_c_fields_from_partial(partial_json: dict) -> dict:
    """
    Removes any RUN_C_CALCULATED_FIELDS that the model may have sneaked into
    the output. Uses path-aware traversal matching extraction_orchestrator.py A9 fix.

    RUN_C_CALCULATED_FIELDS paths:
        "displays[*].ppi"                      â†’ remove "ppi" from each item in displays[]
        "camera_lenses[*].sensor_size_decimal" â†’ remove from each item in camera_lenses[]
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
# Main builder
# ---------------------------------------------------------------------------

def build_spec_json(
    extractions: list,
    source_text: str,
    raw_source_ids: list[int],
    raw_transcript_id: int | None,
    source_section_offsets: dict[str, tuple[int, int]],
    raw_source_ids_by_site: dict[str, int],
    url_registry_id: int | None = None,
) -> tuple[dict, dict]:
    """
    Converts LangExtract's flat Extraction list â†’ nested partial_json + evidence_json.

    Args:
        extractions:              List of lx.data.Extraction objects from lx.extract().
        source_text:              The full merged input string passed to lx.extract().
                                  Used for reference only (not re-parsed here).
        raw_source_ids:           All raw_scraped_data.raw_id values that were included.
        raw_transcript_id:        youtube_raw_transcript_data.raw_transcript_id, or None.
        source_section_offsets:   Maps each site_name â†’ (start_char, end_char) range
                                  in the merged source_text. Built by
                                  _build_source_section_offsets() in langextract_run_a.
                                  Example:
                                    {
                                      "vivo_official":  (0, 6500),
                                      "gsmarena":       (6501, 13000),
                                      "transcript":     (13001, 25000),
                                    }
        raw_source_ids_by_site:   Maps each site_name â†’ raw_id (from file_map returned
                                  by assemble_run_a_input). Used for evidence attribution.
                                  Example: {"vivo_official": 42, "gsmarena": 43,
                                            "transcript": 7}
        url_registry_id:          Optional â€” used only for array completeness warning logs.

    Returns:
        partial_json : Nested spec JSON dict matching spec_template.yaml shape.
                       Array fields (variants, displays, camera_lenses) are sorted
                       by their index attribute. Run C calculated fields are stripped.
        evidence_json: {field_path: evidence_entry} where evidence_entry contains
                       evidence_text, source_type, raw_id, grounded.
                       Ungrounded extractions (char_interval=None) are excluded.

    Notes:
        - Merge is first-seen-wins (by char_interval.start_pos, ascending).
          OEM chars 0â€“N always beat GSMArena chars N+1â€“M for the same field.
        - Extractions with char_interval=None sort LAST (lowest priority).
        - extra_feature and in_the_box_item use append-all with deduplication.
    """
    # -------------------------------------------------------------------------
    # Step 1 â€” Sort extractions by char_interval.start_pos (None â†’ last)
    # Section 9.2: OEM chars 0â€“N win over GSMArena N+1â€“M for the same field.
    # -------------------------------------------------------------------------
    sorted_extractions = sorted(
        extractions,
        key=lambda e: (
            e.char_interval.start_pos if e.char_interval is not None else float("inf")
        ),
    )

    # Monitor ungrounded count
    ungrounded_count = sum(1 for e in sorted_extractions if e.char_interval is None)
    if ungrounded_count > 0:
        ungrounded_pct = (ungrounded_count / max(len(sorted_extractions), 1)) * 100
        logger.warning(
            "build_spec_json: %d/%d extractions ungrounded (char_interval=None, %.0f%%). "
            "Review example quality if > 20%%.",
            ungrounded_count, len(sorted_extractions), ungrounded_pct,
        )

    # -------------------------------------------------------------------------
    # Step 2 â€” Group by extraction_class
    # -------------------------------------------------------------------------
    grouped: dict[str, list] = defaultdict(list)
    for e in sorted_extractions:
        grouped[e.extraction_class].append(e)

    # -------------------------------------------------------------------------
    # Step 3a â€” Single-entity sections (first-seen-wins per attribute key)
    # Section 9.4 Step 2
    # -------------------------------------------------------------------------
    partial_json: dict = {}
    evidence_json: dict = {}

    for cls, schema_key in SINGLE_ENTITY_CLASSES.items():
        merged_attrs: dict = {}
        first_evidence: dict[str, dict | None] = {}  # attr_key â†’ evidence entry

        for e in grouped.get(cls, []):
            ev = _build_evidence_entry(
                e, source_section_offsets, raw_source_ids_by_site, raw_transcript_id
            )
            for attr_key, attr_val in e.attributes.items():
                if attr_key not in merged_attrs:
                    merged_attrs[attr_key] = attr_val
                    first_evidence[attr_key] = ev

        partial_json[schema_key] = merged_attrs if merged_attrs else {}

        # Build evidence_json entries for this section
        for attr_key, ev in first_evidence.items():
            if ev is not None:
                field_path = f"{schema_key}.{attr_key}"
                evidence_json[field_path] = ev

    # -------------------------------------------------------------------------
    # Step 3b â€” Indexed array sections (group by index attribute, first-seen-wins)
    # Section 9.4 Step 3
    # -------------------------------------------------------------------------
    for cls, (schema_key, index_attr) in INDEXED_ARRAY_CLASSES.items():
        by_index: dict[int, dict] = defaultdict(dict)
        by_index_evidence: dict[int, dict[str, dict | None]] = defaultdict(dict)

        for e in grouped.get(cls, []):
            idx = e.attributes.get(index_attr, 0)
            ev = _build_evidence_entry(
                e, source_section_offsets, raw_source_ids_by_site, raw_transcript_id
            )
            for attr_key, attr_val in e.attributes.items():
                if attr_key == index_attr:
                    continue  # Don't store the index key itself in the object
                if attr_key not in by_index[idx]:
                    by_index[idx][attr_key] = attr_val
                    by_index_evidence[idx][attr_key] = ev

        # Sort by index to produce ordered list
        sorted_indices = sorted(by_index.keys())
        partial_json[schema_key] = [by_index[i] for i in sorted_indices]

        # Build evidence_json
        for idx in sorted_indices:
            for attr_key, ev in by_index_evidence[idx].items():
                if ev is not None:
                    field_path = f"{schema_key}[{idx}].{attr_key}"
                    evidence_json[field_path] = ev

    # -------------------------------------------------------------------------
    # Step 3c â€” extra_feature: append-all, deduplicate by feature_name value
    # Section 9.4 Step 3b (NEW v4)
    # -------------------------------------------------------------------------
    extra_features: list[str] = []
    seen_features: set[str] = set()

    for e in grouped.get("extra_feature", []):
        val = e.attributes.get("feature_name") or e.attributes.get("value")
        if val and val not in seen_features:
            seen_features.add(val)
            extra_features.append(val)
            ev = _build_evidence_entry(
                e, source_section_offsets, raw_source_ids_by_site, raw_transcript_id
            )
            if ev is not None:
                field_path = f"extra_features[{len(extra_features) - 1}]"
                evidence_json[field_path] = ev

    partial_json["extra_features"] = extra_features

    # -------------------------------------------------------------------------
    # Step 3d â€” in_the_box_item: append-all structured objects, dedupe by item_name
    # Section 9.4 Step 3c (NEW v4)
    # CRITICAL: structured objects {item_name, item_specification, quantity} NOT plain strings
    # -------------------------------------------------------------------------
    in_the_box: list[dict] = []
    seen_item_names: set[str] = set()

    for e in grouped.get("in_the_box_item", []):
        item_name = e.attributes.get("item_name")
        if item_name and item_name not in seen_item_names:
            seen_item_names.add(item_name)
            in_the_box.append({
                "item_name":          item_name,
                "item_specification": e.attributes.get("item_specification"),
                "quantity":           e.attributes.get("quantity", 1),
            })
            ev = _build_evidence_entry(
                e, source_section_offsets, raw_source_ids_by_site, raw_transcript_id
            )
            if ev is not None:
                field_path = f"in_the_box[{len(in_the_box) - 1}].item_name"
                evidence_json[field_path] = ev

    partial_json["in_the_box"] = in_the_box

    # -------------------------------------------------------------------------
    # Step 4 â€” Strip any Run C calculated fields the model may have included
    # -------------------------------------------------------------------------
    partial_json = _strip_run_c_fields_from_partial(partial_json)

    # -------------------------------------------------------------------------
    # Step 5 â€” Array completeness monitor (Section 9.5) â€” warnings only
    # -------------------------------------------------------------------------
    _check_array_completeness(partial_json, url_registry_id)

    logger.info(
        "build_spec_json: built partial_json with %d top-level keys, "
        "%d evidence entries. url_registry_id=%s",
        len(partial_json), len(evidence_json), url_registry_id,
    )

    return partial_json, evidence_json

