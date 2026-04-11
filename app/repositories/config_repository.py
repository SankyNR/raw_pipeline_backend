"""
Phase 8 — Config Repository

DB lookup helpers used by the scrape orchestrator.
Each function fetches exactly one row and raises ValueError if not found.
"""

from app.core.supabase_client import get_client


async def get_lookup_source(site_name: str) -> dict:
    """
    Fetches the full lookup_source_registry row for the given site_name.

    Returns the row as a dict (includes template_id, display_name, etc.).

    Raises:
        ValueError: If no matching row is found.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("lookup_source_registry")
        .select("*")
        .eq("site_name", site_name)
        .execute()
    )

    if not result.data:
        raise ValueError(
            f"get_lookup_source(): no lookup_source_registry row found for "
            f"site_name={site_name!r}."
        )

    return result.data[0]


async def get_config_template(template_id: int) -> dict:
    """
    Fetches the full scraper_config_templates row for the given template_id.

    Returns the row as a dict (includes config_template JSONB and template_name).

    Raises:
        ValueError: If no matching row is found.
    """
    result = (
        get_client()
        .schema("pipeline")
        .table("scraper_config_templates")
        .select("*")
        .eq("template_id", template_id)
        .execute()
    )

    if not result.data:
        raise ValueError(
            f"get_config_template(): no scraper_config_templates row found for "
            f"template_id={template_id}."
        )

    return result.data[0]
