"""Video relevance filter — scores YouTube search results against a target phone using title + description."""

import re
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Brand alias map — covers known cases where the brand name in DB differs
# from how it appears in video titles. Add entries as new brands are added.
#
# NOTE: "iphone" is a series token (Apple), NOT a brand alias.
#       "pixel"  is a series token (Google), NOT a brand alias.
#       These were moved to SERIES_TOKENS in the updated spec to prevent
#       false competing-brand filtering against review titles that contain
#       "pixel" or "iphone" for a different brand's comparison video.
# ---------------------------------------------------------------------------

BRAND_ALIASES: dict[str, list[str]] = {
    "motorola": ["motorola", "moto"],
    "apple":    ["apple"],            # "iphone" is a series token, not brand alias
    "samsung":  ["samsung"],
    "xiaomi":   ["xiaomi", "mi"],
    "redmi":    ["redmi"],            # independent brand — never associated with xiaomi
    "oneplus":  ["oneplus", "one plus"],
    "realme":   ["realme"],
    "vivo":     ["vivo"],
    "oppo":     ["oppo"],
    "nothing":  ["nothing", "cmf"],
    "cmf":      ["cmf", "nothing"],
    "iqoo":     ["iqoo"],
    "poco":     ["poco"],
    "google":   ["google"],           # "pixel" is a series token, not brand alias
}

# All known brand aliases flattened — used to build competing brand list per search
_ALL_BRAND_ALIASES: list[str] = [
    alias
    for aliases in BRAND_ALIASES.values()
    for alias in aliases
]

# ---------------------------------------------------------------------------
# Primary series tokens per brand — identifies the product line within a brand.
# Critical for separating phones that share a numeric token but are different series.
# Example: "Motorola Edge 50 Fusion" vs "Motorola G50" — both have "50", only series differs.
# Add new series tokens here as new product lines are introduced.
#
# "iphone" → Apple series identifier (not brand alias)
# "pixel"  → Google series identifier (not brand alias)
# "neo"    → iQOO series (secondary critical for all other brands)
# "zfold", "zflip" → Samsung compound series, pre-normalized in _normalize_text step 3
# ---------------------------------------------------------------------------

SERIES_TOKENS: dict[str, list[str]] = {
    "motorola": ["edge", "g", "razr"],
    "samsung":  ["galaxy", "s", "a", "m", "zfold", "zflip"],
    # "z fold" and "z flip" are pre-normalized to "zfold"/"zflip" in _normalize_text step 3
    "apple":    ["iphone"],           # iphone is the series identifier for Apple
    "xiaomi":   [],                   # number-based lineup only — no series classification
    "redmi":    ["note", "k", "a", "y"],
    "oneplus":  ["nord", "open"],
    "realme":   ["c", "gt", "narzo"],
    "vivo":     ["y", "v", "x", "t"],
    "oppo":     ["reno", "find"],
    "google":   ["pixel"],            # pixel is the series identifier for Google
    "nothing":  ["phone"],
    "cmf":      ["phone"],
    "iqoo":     ["neo", "z"],         # "neo" is series for iQOO but secondary for other brands
    "poco":     ["x", "f", "m", "c"],
}

# ---------------------------------------------------------------------------
# Secondary critical tokens — variant-distinguishing suffixes.
# Split into STRONG (always valid) and WEAK (only valid if in model name).
#
# STRONG: unambiguous variant names — safe to match anywhere in video text.
# WEAK: single-char or ambiguous suffixes — must be gated by presence in model_name.
#   "a" searching Galaxy S25 must NOT match random text.
#   "a" searching Pixel 9a IS valid because it's in the model name.
# ---------------------------------------------------------------------------

STRONG_SECONDARY_TOKENS: set[str] = {
    "pro", "plus", "ultra", "max",
    "mini", "lite", "fe",
    "fusion", "power", "prime",
    "turbo", "speed", "hypercharge",
    "stylus", "zoom",
    "reloaded", "civi", "sport",
}

WEAK_SECONDARY_TOKENS: set[str] = {
    "se", "t", "r", "x", "s", "e", "a",
    "xl", "ce", "i", "y", "ne",
}

# Full set for reference — actual per-search secondary tokens are built in extract_search_tokens
SECONDARY_CRITICAL_TOKENS: set[str] = STRONG_SECONDARY_TOKENS | WEAK_SECONDARY_TOKENS

RELEVANCE_THRESHOLD = 7


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """
    Normalizes a text string for token matching.

    Steps (must run in this order):
      1. Lowercase
      2. Replace "+" with " plus " — handles "Reno14 Pro+" → "Reno14 Pro Plus",
         "Redmi Note 13 Pro+" → "Redmi Note 13 Pro Plus". Must happen before
         digit-expansion so "pro+" becomes "pro plus" not "pro +".
      3. Normalize Samsung compound series to single tokens:
         "z fold" → "zfold", "z flip" → "zflip".
         Must happen before digit-expansion so they survive as single identifiers.
      4. Expand run-together alphanumeric tokens by inserting a space between
         letter→digit and digit→letter transitions — "reno15" → "reno 15",
         "edge50fusion" → "edge 50 fusion", "y28e" → "y 28 e".
      5. Replace all remaining non-alphanumeric characters with spaces.
      6. Collapse multiple spaces.

    Applied to: combined title+description text, AND model_name before token extraction.
    """
    text = text.lower()
    # Step 2: normalize "+" to "plus"
    text = text.replace("+", " plus ")
    # Step 3: normalize Samsung compound series to single tokens
    text = text.replace("z fold", "zfold")
    text = text.replace("z flip", "zflip")
    # Step 4: insert space between letter→digit and digit→letter transitions
    text = re.sub(r'([a-z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-z])', r'\1 \2', text)
    # Step 5: replace non-alphanumeric with space
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Step 6: collapse spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _split_alphanumeric(token: str) -> list[str]:
    """
    Splits an alphanumeric token into its letter and digit components.
    "y28e"     → ["y", "28", "e"]
    "s25"      → ["s", "25"]
    "g96"      → ["g", "96"]
    "reno15c"  → ["reno", "15", "c"]
    "zfold"    → ["zfold"]  (all letters, returned as-is)
    Pure letter or pure digit tokens are returned as-is in a single-element list.
    """
    parts = re.findall(r'[a-z]+|\d+', token)
    return [p for p in parts if p]


def _matches(text: str, token: str) -> bool:
    """
    Word-boundary aware token match.
    Prevents short aliases like "mi" matching inside unrelated words like "time" or "premium",
    and prevents model numbers like "21" matching inside "210".
    Uses regex word boundaries (\\b) around the escaped token.
    Always call on already-normalized text.
    """
    return bool(re.search(rf"\b{re.escape(token)}\b", text))


def _match_series_token(text: str, token: str) -> bool:
    """
    Series-aware token match for single-character series identifiers.

    Single-char series tokens like "g", "s", "y", "a" need special handling
    because normalization expands "5G" → "5 g", which would produce a false
    series match — "g" from "5g" would incorrectly score as a Motorola G-series hit.

    For single-char tokens: requires the token to appear followed by a space and
    digit, matching the actual series pattern: "g 50", "y 28", "s 21".
    This confirms the letter is acting as a series name, not a letter fragment.

    For multi-char tokens: falls back to standard word-boundary _matches().

    Examples:
        "motorola g 96 review" with token "g" → matches (pattern: g followed by digit)
        "motorola 5 g review"  with token "g" → does NOT match (g follows digit, not series pattern)
        "samsung galaxy s 21"  with token "s" → matches
        "oppo reno 15"         with token "reno" → _matches() used, matches normally
    """
    if len(token) == 1:
        # Single char must appear as a series prefix: letter followed by whitespace and digit
        return bool(re.search(rf"\b{re.escape(token)}\s+\d", text))
    return _matches(text, token)


def _match_weak_token(text: str, token: str) -> bool:
    """
    Weak-secondary-aware token match for single-character variant suffixes.
    Used for SCORING and MISSING-PENALTY on normalized text.

    Single-char weak tokens like 'a', 'i', 's' are written as numeric suffixes
    in the model name: "Pixel 9a" -> normalized "9 a", "Realme 8i" -> "8 i".
    Requires the token to appear PRECEDED by a digit and whitespace on normalized text.
    This prevents the English article "a" in "A Budget Beast" from matching,
    since it has no preceding digit.

    For multi-char tokens ("se", "xl", "ce", "ne"): falls back to _matches().

    NOTE: For intruder detection use _match_weak_intruder() on raw text instead.
    """
    if len(token) == 1:
        return bool(re.search(rf"\d\s+{re.escape(token)}\b", text))
    return _matches(text, token)


def _match_weak_intruder(raw_text: str, token: str) -> bool:
    """
    Weak intruder check on RAW lowercased (pre-normalization) text.

    Genuine variant suffixes ('8i', '3a', '8s') are glued directly to the digit
    in the original title with no separator. Checking raw text for the pattern
    \d{token} (digit immediately followed by the letter) correctly identifies
    these, while a title like "OnePlus 15 - A Flagship Killer" has a space or
    punctuation between '15' and 'A' and will NOT match.

    Examples:
        raw 'realme 8i review'       token 'i' -> matches  (8i glued)
        raw 'nothing phone 3a review' token 'a' -> matches  (3a glued)
        raw 'oneplus 15 - a flagship' token 'a' -> NO match (15 space a)
        raw 'a complete review'       token 'a' -> NO match (no digit before)

    For multi-char tokens: falls back to _matches() on the raw text.
    """
    if len(token) == 1:
        return bool(re.search(rf"\d{re.escape(token)}\b", raw_text))
    return _matches(raw_text, token)


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------

def extract_search_tokens(brand: str, model_name: str) -> dict:
    """
    Extracts structured scoring tokens from brand + model_name.

    Returns:
        {
            "brand_aliases":            list[str],  # all aliases for this brand
            "primary_numeric_tokens":   list[str],  # digit components — most distinctive
            "primary_series_tokens":    list[str],  # series identifiers (edge, g, iphone, pixel, etc.)
            "strong_secondary_tokens":  list[str],  # strong variant suffixes in model (pro, ultra, fusion…)
            "weak_secondary_tokens":    list[str],  # weak variant suffixes in model (se, xl, a, x…)
            "noncritical_tokens":       list[str],  # remaining tokens
            "connectivity":             str | None, # "5g" or "4g" if present, else None
            "competing_aliases":        list[str],  # all brand aliases NOT belonging to this phone
            "dictionary_variants":      list[str],  # normalized model name variants for exact match
        }

    Classification priority (brand-aware, in order):
        1. Connectivity tokens (4g, 5g) — handled separately, excluded from all other categories
        2. Brand aliases — skip, not scored as model tokens
        3. Primary numeric (digit parts from split)
        4. Primary series (brand-specific — checked BEFORE secondary to avoid misclassification)
        5. Secondary critical (strong always valid; weak only if present in model name)
        6. Non-critical (everything else with len > 1)

    The series-first rule ensures context-aware classification:
        "neo" → series for iQOO (iQOO Neo 10), secondary for Motorola (Edge 50 Neo)
        "flip" → part of "zflip" series for Samsung, secondary suffix for Oppo Find N2 Flip

    Weak secondary tokens (single-char ambiguous suffixes) are only added to the
    valid secondary set if they appear in the model name itself. This prevents "a"
    from matching random text when searching Galaxy S25, while correctly matching
    when searching Pixel 9a (where "a" is explicitly in the model).

    Never raises — returns empty token lists on unexpected input.
    """
    brand_lower = brand.lower().strip()
    # Connectivity variant — detected on raw input before normalization expands "5G" -> "5 g"
    model_lower_raw = model_name.lower()
    connectivity = None
    if re.search(r'\b5g\b', model_lower_raw):
        connectivity = "5g"
    elif re.search(r'\b4g\b', model_lower_raw):
        connectivity = "4g"

    # Strip connectivity before normalization
    model_stripped = re.sub(r'\b(4g|5g)\b', '', model_lower_raw).strip()
    model_normalized = _normalize_text(model_stripped)

    # Brand aliases for this phone
    brand_alias_list = BRAND_ALIASES.get(brand_lower, [brand_lower])

    # Series tokens for this brand
    brand_series = SERIES_TOKENS.get(brand_lower, [])

    # Split model into tokens
    raw_tokens = model_normalized.split()

    # Build valid secondary set for this model — strong tokens always included,
    # weak tokens only if they appear in the model's own token set.
    # This is computed from raw_tokens (the model's token set) before splitting alphanumerics.
    model_token_set = set(raw_tokens)
    valid_secondary = STRONG_SECONDARY_TOKENS.copy()
    for weak_token in WEAK_SECONDARY_TOKENS:
        if weak_token in model_token_set:
            valid_secondary.add(weak_token)

    primary_numeric = []
    primary_series = []
    strong_secondary = []
    weak_secondary = []
    noncritical = []

    for token in raw_tokens:
        # Skip connectivity tokens — handled separately
        if token in ("4g", "5g"):
            continue
        # Skip brand aliases — not scored as model tokens
        if token in brand_alias_list:
            continue
        # Skip very short tokens unless they are in the brand's series list, valid secondary tokens, or are digits
        if len(token) == 1 and not token.isdigit() and token not in brand_series and token not in valid_secondary:
            continue

        # Split alphanumeric token into parts
        # "s25" → ["s", "25"], "reno15c" → ["reno", "15", "c"], "zfold" → ["zfold"] (all letters)
        parts = _split_alphanumeric(token)
        for part in parts:
            if part in ("4g", "5g"):
                continue
            if part.isdigit():
                primary_numeric.append(part)
            elif part in brand_series:
                # Series checked BEFORE secondary — "neo" is series for iQOO,
                # so it won't fall through to secondary even if in valid_secondary
                primary_series.append(part)
            elif part in valid_secondary:
                # Split into strong vs weak for differentiated scoring
                if part in STRONG_SECONDARY_TOKENS:
                    strong_secondary.append(part)
                else:
                    weak_secondary.append(part)
            elif len(part) > 1:
                noncritical.append(part)

    # Competing brand aliases: every known alias not belonging to this brand
    competing = [
        alias
        for alias in _ALL_BRAND_ALIASES
        if alias not in brand_alias_list
    ]

    # Dictionary variants — normalized model name strings for exact-match boost.
    # Generated at scoring time from brand + model, no DB call needed.
    # Covers common ways the phone name appears in video titles and descriptions.
    model_no_conn = re.sub(r'\b(4g|5g)\b', '', model_normalized).strip()
    model_no_spaces = model_no_conn.replace(' ', '')
    variants = set()
    for alias in brand_alias_list:
        variants.add(f"{alias} {model_no_conn}".strip())
        variants.add(f"{alias}{model_no_spaces}")
    variants.add(model_no_conn)
    variants.add(model_no_spaces)
    # Filter empty strings
    dictionary_variants = [v for v in variants if len(v) > 2]

    return {
        "brand_aliases":            brand_alias_list,
        "primary_numeric_tokens":   list(dict.fromkeys(primary_numeric)),    # deduped, order preserved
        "primary_series_tokens":    list(dict.fromkeys(primary_series)),
        "strong_secondary_tokens":  list(dict.fromkeys(strong_secondary)),
        "weak_secondary_tokens":    list(dict.fromkeys(weak_secondary)),
        "noncritical_tokens":       list(dict.fromkeys(noncritical)),
        "connectivity":             connectivity,
        "competing_aliases":        competing,
        "dictionary_variants":      dictionary_variants,
        "model_token_set":          model_token_set,   # raw model tokens — used for weak-intruder check
    }


# ---------------------------------------------------------------------------
# Task 3.2 — score_video() and filter_videos()
# ---------------------------------------------------------------------------

def score_video(
    title: str,
    description: str,
    brand: str,
    model_name: str,
) -> int:
    """
    Scores a video's relevance to a specific phone.

    Combines title and description into one normalized text block.
    Series tokens use _match_series_token() to prevent single-char false matches.
    Dictionary boost uses space-stripped substring matching (not word-boundary regex).

    WHY two matching strategies:
    - Token matching (_matches): word-boundary regex. Prevents "mi" matching "premium",
      "21" matching "210". Correct for individual tokens that must appear as whole words.
    - Dictionary matching: space-stripped substring. After normalization, text has spaces
      between tokens ("edge 50 fusion"). A no-spaces variant ("edge50fusion") would never
      match word-boundary regex against a spaced string. Stripping spaces from both sides
      and doing a plain substring check is the correct approach.
    - Series matching (_match_series_token): word-boundary plus digit lookahead for
      single-char tokens. Prevents "g" from "5 g" (normalized "5G") falsely scoring
      as Motorola G-series. Single-char series must appear as "g 50" pattern.
    - Secondary token matching: uses valid_secondary built per-model in extract_search_tokens.
      Strong secondary tokens match anywhere; weak secondary tokens only appear in valid_secondary
      when they are present in the model name itself — prevents "a" matching Galaxy S25 searches.

    Scoring:
        +5  dictionary variant match — SUPPRESSED if any intruder fires (see below)
        +4  each primary numeric token found
        +3  each primary series token found
        +3  brand alias found
        +3  each strong secondary token found
        +1  each weak secondary token found
        +2  connectivity token found (bonus only — never a gate, see inline comment)
        +1  each non-critical token found
        −4  primary numeric tokens present in model but none found in video text (applied once)
        −5  strong secondary tokens present in model but none found in video text (applied once)
        −4  weak secondary tokens present in model but none found in video text (applied once)
        −5  any competing brand alias found (applied once)
        −3  primary series tokens present in model but none found in video text (applied once)
        −5  strong intruder fires (applied once — see intruder penalty section below)
        −4  weak intruder fires (applied once — see intruder penalty section below)

    Dict bonus suppression + intruder penalties:
        Intruder = a secondary token in the video text that does NOT belong to this model.
        Strong intruder: any STRONG_SECONDARY_TOKENS word found in text that is not in
          this model's own strong_secondary_tokens list.
        Weak intruder:   any WEAK_SECONDARY_TOKENS word found in text that is not in
          this model's raw token set (model_token_set).
        When either fires:
          1. The +5 dict bonus is suppressed (substring dict hits like "realme8" ⊂
             "realme8ireview" cannot overcome the variant mismatch).
          2. A direct penalty is applied: −5 for a strong intruder, −4 for a weak
             intruder. This is necessary because brand+numeric alone (3+4=7) already
             reaches the threshold, so dict suppression alone is insufficient to reject
             cross-variant false positives like "Nothing Phone 4a Pro" or "Realme 8i".

    Threshold: score >= RELEVANCE_THRESHOLD (7) to pass.

    Safe fallback: returns 999 on any exception.
    """
    try:
        tokens = extract_search_tokens(brand, model_name)

        # Normalize combined text — same normalization as applied to model_name during extraction
        text = _normalize_text(f"{title} {description}")

        # Raw lowercased text — used only for weak intruder detection (see below)
        raw_text = f"{title} {description}".lower()

        # Space-stripped version for dictionary matching — see dictionary boost section below
        text_no_spaces = text.replace(" ", "")

        score = 0

        # ----------------------------------------
        # Pre-compute intruder flags — must run before dict bonus
        # ----------------------------------------
        # Strong intruder: a STRONG_SEC token appears in the video text but is NOT in
        # this model's own strong_secondary_tokens (it belongs to a different variant).
        # E.g. searching "Nothing Phone (4a)" and the video contains "pro".
        strong_intruder_fires = any(
            _matches(text, w)
            for w in STRONG_SECONDARY_TOKENS
            if w not in tokens["strong_secondary_tokens"]
        )

        # Weak intruder: a WEAK_SEC token appears glued to a digit in the raw title
        # but is NOT in this model's raw token set (the model doesn't have that suffix).
        # E.g. searching "Realme 8 4G" and the video title contains "8i" (Realme 8i).
        # Uses _match_weak_intruder() on RAW text: checks pattern \d{token} (glued)
        # so "8i" fires but "15 - A Flagship Killer" does NOT (15 and A are separated).
        weak_intruder_fires = any(
            _match_weak_intruder(raw_text, w)
            for w in WEAK_SECONDARY_TOKENS
            if w not in tokens["model_token_set"]
        )

        any_intruder = strong_intruder_fires or weak_intruder_fires

        # ----------------------------------------
        # Intruder penalties — applied before dict and scoring
        # ----------------------------------------
        # A video containing a variant suffix that does NOT belong to this model is
        # penalised directly, in addition to losing the dict bonus. This is required
        # because brand+numeric alone can reach the threshold (3+4=7) without any dict
        # contribution, so suppression alone is insufficient.
        if strong_intruder_fires:
            score -= 5
        if weak_intruder_fires:
            score -= 4

        # Dictionary boost — checks if any normalized model name variant appears as a
        # contiguous substring in the space-stripped text.
        #
        # WHY NOT _matches() here: after normalization, text has spaces between all tokens
        # ("edge 50 fusion"), but dictionary variants include space-free forms ("edge50fusion").
        # Word-boundary regex on "edge50fusion" against "edge 50 fusion" will never match
        # because the variant is a different string after normalization adds spaces.
        # The correct approach is to strip spaces from both text and variant and do a plain
        # substring check — this catches "edge50fusion" in "motorolaedge50fusion" and also
        # "edge 50 fusion" in "motorola edge 50 fusion" (after stripping spaces from both).
        #
        # SUPPRESSED when any_intruder is True: a substring dict hit on "realme8" inside
        # "realme8ireview" would otherwise inflate the score past the threshold despite
        # the weak intruder "i" signalling a different variant.
        if not any_intruder:
            if any(v.replace(" ", "") in text_no_spaces for v in tokens["dictionary_variants"] if v):
                score += 5

        # Primary numeric tokens — most distinctive identifiers
        for token in tokens["primary_numeric_tokens"]:
            if _matches(text, token):
                score += 4

        # ----------------------------------------
        # Numeric mismatch penalty (CRITICAL)
        # ----------------------------------------
        if tokens["primary_numeric_tokens"]:
            numeric_found = any(
                _matches(text, t) for t in tokens["primary_numeric_tokens"]
            )
            if not numeric_found:
                score -= 4

        # Primary series tokens — use _match_series_token to prevent single-char false matches.
        # "5g" normalizes to "5 g" — without special handling, "g" would falsely match as
        # Motorola G-series. _match_series_token requires single-char series to appear
        # followed by a digit (pattern: "g 50", "s 21", "y 28").
        for token in tokens["primary_series_tokens"]:
            if _match_series_token(text, token):
                score += 3

        # Brand match
        if any(_matches(text, alias) for alias in tokens["brand_aliases"]):
            score += 3

        # Secondary critical tokens — strong variants score +3, weak score +1
        # Uses _match_weak_token() for single-char weak tokens to avoid matching the
        # English article 'a' or possessive 's' in place of a phone variant suffix.
        for token in tokens["strong_secondary_tokens"]:
            if _matches(text, token):
                score += 3
        for token in tokens["weak_secondary_tokens"]:
            if _match_weak_token(text, token):
                score += 1

        # ----------------------------------------
        # Weak secondary missing penalty
        # ----------------------------------------
        # Fires once if the model has weak secondary tokens (e.g. 'a' in Phone(3a),
        # 's' in Realme 8s) but NONE of them appear in the video text.
        # A "Phone 3" review is a different product from "Phone (3a)" — the missing
        # weak suffix should cost points, not just lose the +1 bonus.
        # Uses _match_weak_token() for the same digit-adjacency requirement.
        if tokens["weak_secondary_tokens"]:
            weak_secondary_found = any(
                _match_weak_token(text, t) for t in tokens["weak_secondary_tokens"]
            )
            if not weak_secondary_found:
                score -= 4

        # ----------------------------------------
        # Strong secondary missing penalty (CRITICAL)
        # ----------------------------------------
        # Fires if model has strong secondary tokens but none appear in video text.
        # Distinguishes e.g. "Galaxy S25 Ultra" from a plain "Galaxy S25" video.
        if tokens["strong_secondary_tokens"]:
            strong_found = any(
                _matches(text, t) for t in tokens["strong_secondary_tokens"]
            )
            if not strong_found:
                score -= 5

        # Connectivity variant match — +2 bonus only, never a gate.
        #
        # Design decision: connectivity is NOT gated on whether multiple variants exist
        # in the registry. This is intentional. The filter is stateless and must not
        # make DB calls. Under the new scoring (+4 numeric, +3 series, +3 brand = 10
        # baseline), a correct video scores well above the threshold of 7 without the
        # connectivity bonus. Not finding "5g" in a video about a 5G-only phone drops
        # the score from 12 to 10 — still passes. Connectivity is a tiebreaker for
        # phones that genuinely exist in both 4G and 5G variants; it is never the
        # reason a video passes or fails.
        if tokens["connectivity"] and _matches(text, tokens["connectivity"]):
            score += 2

        # Non-critical tokens
        for token in tokens["noncritical_tokens"]:
            if _matches(text, token):
                score += 1

        # Missing series penalty — fires if model has series tokens but none appear in video text.
        # Uses _match_series_token for the same single-char false-match protection.
        if tokens["primary_series_tokens"]:
            series_found = any(
                _match_series_token(text, t) for t in tokens["primary_series_tokens"]
            )
            if not series_found:
                score -= 3

        # Competing brand penalty — applied once regardless of how many aliases found
        if any(_matches(text, alias) for alias in tokens["competing_aliases"]):
            score -= 5

        return score

    except Exception as e:
        logger.warning(
            "score_video() raised unexpectedly for brand=%r model=%r title=%r: %s — passing video through",
            brand, model_name, title, e
        )
        return 999  # safe fallback — pass through rather than block


def filter_videos(
    videos: list[dict],
    descriptions: dict[str, str],
    brand: str,
    model_name: str,
) -> list[dict]:
    """
    Filters a list of video dicts returned by search_videos_for_channel()
    using relevance scoring.

    Args:
        videos:       list of dicts, each with keys: yt_video_id, video_title, ...
        descriptions: dict mapping yt_video_id → description string
                      (from fetch_video_descriptions())
        brand:        phone brand
        model_name:   phone model name

    Returns:
        Filtered list containing only videos that score >= RELEVANCE_THRESHOLD.
        Order is preserved. Empty list if no videos pass.

    Never raises.
    """
    passed = []
    for video in videos:
        title = video.get("video_title") or ""
        description = descriptions.get(video.get("yt_video_id", ""), "")
        score = score_video(title, description, brand, model_name)
        if score >= RELEVANCE_THRESHOLD:
            passed.append(video)
        else:
            logger.debug(
                "Filtered out video yt_video_id=%r title=%r score=%d (threshold=%d)",
                video.get("yt_video_id"), title, score, RELEVANCE_THRESHOLD
            )
    return passed
