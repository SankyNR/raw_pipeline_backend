"""SRT to plain TXT stripping utility — removes sequence numbers, timestamps, and empty lines."""

import re


# ---------------------------------------------------------------------------
# Task 14.1 — process_srt_to_txt()
# ---------------------------------------------------------------------------

def process_srt_to_txt(content: str) -> str:
    """
    Converts SRT content (or plain text) to clean plain text for LangExtract.

    For SRT input, removes:
      1. Sequence number lines  — purely numeric lines (e.g. '1', '42')
      2. Timestamp lines        — matching HH:MM:SS,mmm --> HH:MM:SS,mmm
      3. Empty lines            — including whitespace-only lines

    For plain text input (translated Hindi content from translate_to_english()),
    the same function passes it through correctly — there are no sequence numbers
    or timestamps to strip, so only empty lines are removed.

    Joins remaining lines with single newlines.
    Returns empty string if all lines stripped (valid edge case, no crash).
    """
    TIMESTAMP_PATTERN = re.compile(
        r'^\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}$'
    )
    SEQUENCE_PATTERN = re.compile(r'^\d+$')

    output_lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if SEQUENCE_PATTERN.match(stripped):
            continue
        if TIMESTAMP_PATTERN.match(stripped):
            continue
        output_lines.append(stripped)

    return "\n".join(output_lines)
