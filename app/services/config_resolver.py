"""
Phase 3 — Config Resolution Engine

Task 3.1: resolve_config() substitutes {{PHONE_ID}}, {{PHONE_NAME}}, or {{URL}}
placeholders in the Firecrawl config template JSONB before calling Firecrawl.
"""

import json


def resolve_config(
    template_jsonb: dict,
    site_name: str,
    url: str,
    scrape_identifier: str | None,
) -> dict:
    """
    Substitutes placeholders in the Firecrawl config template.

    Placeholder rules:
    - google_pixel_newer → replaces {{PHONE_ID}} with scrape_identifier
                           (model name with spaces stripped, e.g. "Pixel9proXL")
    - google_pixel_older → replaces {{PHONE_NAME}} with scrape_identifier
                           (exact model name, e.g. "Pixel 5a")
    - all other sites    → replaces {{URL}} with url

    Args:
        template_jsonb:    The raw JSONB config dict from scraper_config_templates.
        site_name:         The site_name slug from lookup_source_registry.
        url:               The URL from url_registry (used by all non-Pixel sites).
        scrape_identifier: Pixel phones only — injected as {{PHONE_ID}} or
                           {{PHONE_NAME}}. NULL for all other brands.

    Returns:
        A new dict with all placeholders replaced by real values.
        The original template_jsonb dict is never mutated.

    Raises:
        ValueError: If site_name is a Pixel site and scrape_identifier is None
                    or empty. Without this guard Python's str.replace() would
                    silently inject the string "None" into the config, causing
                    Firecrawl to receive a broken identifier with no obvious error.
    """
    # Guard: Pixel sites require scrape_identifier — fail loudly with a clear message.
    if site_name in ("google_pixel_newer", "google_pixel_older"):
        if not scrape_identifier:
            raise ValueError(
                f"scrape_identifier is required for site_name='{site_name}' "
                f"but got {scrape_identifier!r}. Check the url_registry row for this phone."
            )

    config_str = json.dumps(template_jsonb)

    if site_name == "google_pixel_newer":
        config_str = config_str.replace("{{PHONE_ID}}", scrape_identifier)
    elif site_name == "google_pixel_older":
        config_str = config_str.replace("{{PHONE_NAME}}", scrape_identifier)
    else:
        config_str = config_str.replace("{{URL}}", url)

    return json.loads(config_str)
