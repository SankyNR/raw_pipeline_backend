"""
Task 0.4 — Firecrawl HTTP Client Wrapper

Async httpx wrapper for the Firecrawl /scrape API.
Uses a 120-second total timeout to prevent indefinitely hanging requests
which would leave url_registry.status stuck at 'currently_scraping'.
"""

import httpx
from app.core.config import FIRECRAWL_API_KEY

FIRECRAWL_TIMEOUT = httpx.Timeout(120.0)  # 120 seconds total
FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v1/scrape"


async def scrape_with_firecrawl(resolved_config: dict) -> dict:
    """
    Posts resolved_config to the Firecrawl /scrape endpoint.
    Returns the raw JSON response as a dict.

    Raises a descriptive exception if:
      - HTTP request fails or times out
      - Response JSON has success=False (includes Firecrawl's own error message)
      - Response is missing expected top-level structure
    """
    async with httpx.AsyncClient(timeout=FIRECRAWL_TIMEOUT) as client:
        response = await client.post(
            FIRECRAWL_SCRAPE_URL,
            headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}"},
            json=resolved_config,
        )
        response.raise_for_status()
        data = response.json()
        if not data.get("success"):
            raise ValueError(
                f"Firecrawl returned success=False: {data.get('error', 'no error message')}"
            )
        return data


def validate_firecrawl_response(response: dict, template_name: str) -> None:
    """
    Validates the Firecrawl response structure before the orchestrator
    attempts to access any fields. Raises ValueError with a descriptive
    message on any structural problem.

    Checks:
    1. response["success"] is True (belt-and-suspenders — also checked in scrape_with_firecrawl)
    2. response["data"] exists and is a dict
    3. response["data"]["markdown"] exists and is a non-empty string
    4. For simple_config_template: response["data"]["screenshot"] exists and is a non-empty string
    5. For all other templates: response["data"]["actions"]["screenshots"] exists,
       is a list, and has at least one item
    """
    if not response.get("success"):
        raise ValueError(
            f"validate_firecrawl_response: response['success'] is not True "
            f"(template={template_name!r})."
        )

    data = response.get("data")
    if not isinstance(data, dict):
        raise ValueError(
            f"validate_firecrawl_response: response['data'] missing or not a dict "
            f"(template={template_name!r})."
        )

    markdown = data.get("markdown")
    if not isinstance(markdown, str) or not markdown.strip():
        raise ValueError(
            f"validate_firecrawl_response: response['data']['markdown'] missing or empty "
            f"(template={template_name!r})."
        )

    if template_name == "simple_config_template":
        screenshot = data.get("screenshot")
        if not isinstance(screenshot, str) or not screenshot.strip():
            raise ValueError(
                "validate_firecrawl_response: response['data']['screenshot'] missing or "
                "empty for simple_config_template."
            )
    else:
        actions = data.get("actions")
        screenshots = actions.get("screenshots") if isinstance(actions, dict) else None
        if not isinstance(screenshots, list) or len(screenshots) == 0:
            raise ValueError(
                f"validate_firecrawl_response: response['data']['actions']['screenshots'] "
                f"missing, not a list, or empty (template={template_name!r})."
            )
