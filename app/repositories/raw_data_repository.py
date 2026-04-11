"""
Phase 6 — Raw Scraped Data Insert Layer

Task 6.1: insert_raw_scraped_data() inserts one row into pipeline.raw_scraped_data.
"""

from app.core.supabase_client import get_client


async def insert_raw_scraped_data(
    url_registry_id: int,
    execution_id: int,
    template_id: int,
    phone_brand: str,
    phone_model: str,
    markdown_path: str,
    screenshot_before_path: str | None,
    screenshot_after_path: str | None,
    file_size_bytes: int | None,
) -> int:
    """
    Inserts one row into pipeline.raw_scraped_data.

    'status' is left to the DB default ('review_pending') — do NOT pass it here.

    file_size_bytes:
        Size of the markdown file only, in bytes — len(markdown_content.encode("utf-8")).
        Screenshots are already represented via the URL columns.
        Pass None if file size is unavailable.

    Constraints to be aware of:
        - execution_id has a UNIQUE constraint: calling this twice for the same
          execution will raise a DB error. This is intentional — one scrape
          execution must map to exactly one raw_scraped_data row.
        - screenshot_before_path / screenshot_after_path may be None depending
          on the template config. Always pass None explicitly, not an empty string.

    Returns:
        raw_id (int) of the newly inserted row.

    Raises:
        RuntimeError: If the insert returns no data.
    """
    payload = {
        "url_registry_id":       url_registry_id,
        "execution_id":          execution_id,
        "template_id":           template_id,
        "phone_brand":           phone_brand,
        "phone_model":           phone_model,
        "markdown_path":         markdown_path,
        "screenshot_before_path": screenshot_before_path,
        "screenshot_after_path":  screenshot_after_path,
        "file_size_bytes":       file_size_bytes,
    }

    result = (
        get_client()
        .schema("pipeline")
        .table("raw_scraped_data")
        .insert(payload)
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            f"insert_raw_scraped_data() returned no data for "
            f"url_registry_id={url_registry_id}, execution_id={execution_id}."
        )

    return result.data[0]["raw_id"]
