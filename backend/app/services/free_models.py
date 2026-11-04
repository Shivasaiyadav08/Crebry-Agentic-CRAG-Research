"""Discover Gemini text models eligible for Crebry's free-tier workflow.

Preserves the public helpers and catalog shape used by the API, model routing,
OCR, and frontend. Unknown model prices are never assumed to be free.
"""

import asyncio
import hashlib
import logging
import re
from datetime import date
from time import monotonic

import httpx
from bs4 import BeautifulSoup

from app.core.config import config

logger = logging.getLogger(__name__)
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta"
PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"

# Capability-reviewed endpoints, not an assertion of free pricing or key access.
# Both live Standard pricing AND authenticated Models.list must also agree.
TEXT_MODEL_IDS = frozenset({
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite", "gemini-3.5-flash-lite",
    "gemini-3-flash-preview", "gemini-3.1-pro-preview",
})
RETIRED_MODEL_IDS = frozenset({
    "gemini-3-pro-preview", "gemini-3.1-flash-lite-preview",
    "gemini-2.5-pro-preview-03-25", "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-pro-preview-06-05", "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash-preview-09-2025", "gemini-2.5-flash-lite-preview-09-2025",
})


class CatalogVerificationError(ValueError):
    """Safe, actionable catalog error suitable for an API response."""

_catalog: list[dict] = []
_expires = 0.0
_catalog_key = ""
_catalog_lock = asyncio.Lock()
_cooldowns: dict[str, float] = {}
_cooldown_key = ""


def model_allowed(model_id: str) -> bool:
    """Preserve operator exclusions across catalogs, plans, OCR, and direct calls."""
    model_id = model_id.removeprefix("models/")
    # 3.5/3.7 Flash aliases redirect to the operator-excluded 3.6/3.8 families.
    excluded = re.match(r"^gemini-3\.(?:5|6|7|8)-flash(?:$|-)", model_id) and model_id != "gemini-3.5-flash-lite"
    retired = model_id in RETIRED_MODEL_IDS or model_id.startswith(("gemini-1.", "gemini-2.0-"))
    redirected = model_id in {"gemini-2.5-pro", "gemini-2.5-flash"} and date.today() >= date(2026, 12, 1)
    unsupported = any(part in model_id for part in ("image", "tts", "audio", "live", "transcribe", "robotics", "embedding"))
    unsupported_preview = ("preview" in model_id or "-exp" in model_id) and model_id not in TEXT_MODEL_IDS
    return not (excluded or retired or redirected or unsupported or unsupported_preview)


def credential_fingerprint():
    """Identify a credential change without persisting the key in checkpoints."""
    return hashlib.sha256(config.gemini_api_key.encode()).hexdigest()


def reset_key_cooldowns():
    """Never carry one API key's cooldowns over to a replacement credential."""
    global _cooldown_key
    if _cooldown_key != config.gemini_api_key:
        _cooldowns.clear()
        _cooldown_key = config.gemini_api_key


def _html_pricing_ids(document: str) -> set[str]:
    """Parse older HTML pricing tables, scoped to each model heading."""
    soup = BeautifulSoup(document, "html.parser")
    eligible: set[str] = set()
    for heading in soup.find_all("h2"):
        codes = []
        table = None
        for element in heading.next_elements:
            if getattr(element, "name", None) == "h2":
                break
            if getattr(element, "name", None) == "code":
                codes.append(element.get_text(strip=True))
            if getattr(element, "name", None) == "table":
                table = element
                break
        if table is None:
            continue
        rows = [
            [cell.get_text(" ", strip=True).lower() for cell in row.find_all(["th", "td"])]
            for row in table.find_all("tr")
        ]
        if not rows:
            continue
        free_column = next((i for i, cell in enumerate(rows[0]) if cell == "free tier"), None)
        if free_column is None:
            continue
        prices = {}
        for row in rows[1:]:
            if len(row) <= free_column:
                continue
            if row[0].startswith("input price"):
                prices["input"] = row[free_column]
            elif row[0].startswith("output price"):
                prices["output"] = row[free_column]
        if prices.get("input") == "free of charge" and prices.get("output") == "free of charge":
            eligible.update(code for code in codes if re.fullmatch(r"gemini-[a-z0-9.-]+", code))
    return eligible


def _markdown_pricing_ids(document: str) -> set[str]:
    """Parse Google's current text/MDX documentation, Standard tier only.

    The redirected pricing page currently uses `{% tab title="Standard" %}` and
    `{% table %}` blocks. Read the Free Tier column, not the Paid Tier column,
    and never infer prices from Batch, Flex, or Priority tabs.
    """
    eligible: set[str] = set()
    sections = re.split(r"(?=^##\s+)", document, flags=re.MULTILINE)
    for section in sections:
        heading = section.split("\n", 1)[0]
        model_match = re.search(r'\{%\s*id="(gemini-[a-z0-9.-]+)"\s*%\}', heading)
        if not model_match:
            continue
        model_id = model_match.group(1)
        tab = re.search(
            r'\{%\s*tab\s+title="Standard"\s*%\}(.*?)\{%\s*/tab\s*%\}',
            section,
            flags=re.DOTALL,
        )
        if not tab:
            continue
        standard = tab.group(1)
        table = re.search(r"\{%\s*table\s*%\}(.*?)\{%\s*/table\s*%\}", standard, re.DOTALL)
        if not table:
            continue
        contents = table.group(1)
        headers = re.split(r"(?m)^\s*---\s*$", contents, maxsplit=1)[0]
        columns = re.findall(r"(?m)^\s*\*\s*(.*?)\s*$", headers)
        if len(columns) < 2 or columns[1] != "Free Tier":
            continue
        # Every row begins with its label, followed by the free-tier cell.
        # Exact cell values are required; $0 and unknown values are not inferred.
        input_free = re.search(
            r"(?mi)^\s*\*\s*Input price[^\n]*\n\s*\*\s*Free of charge\s*$",
            contents,
        )
        output_free = re.search(
            r"(?mi)^\s*\*\s*Output price[^\n]*\n\s*\*\s*Free of charge\s*$",
            contents,
        )
        if input_free and output_free:
            eligible.add(model_id)
    return eligible


def free_tier_ids(document: str) -> set[str]:
    """Verify free Standard input AND output prices from HTML or MDX markup."""
    eligible = _html_pricing_ids(document) if "<h2" in document.lower() else set()
    eligible |= _markdown_pricing_ids(document)
    if not eligible:
        raise ValueError("Gemini pricing format could not be verified.")
    return eligible


def _verified_pricing(document: str) -> set[str]:
    try:
        return free_tier_ids(document)
    except ValueError as error:
        raise CatalogVerificationError(
            "Gemini free-tier pricing could not be verified. Refresh the model list later; no unverified models were selected."
        ) from error


def _pricing_fallback(error: Exception) -> set[str]:
    # No error bodies, URLs containing credentials, or API keys in logs.
    logger.warning("Gemini pricing verification unavailable (%s)", type(error).__name__)
    raise CatalogVerificationError(
        "Google's pricing page is unavailable. Check backend network access and refresh the model list; free-tier eligibility must be verified."
    ) from None


def is_free_text_model(item: dict, eligible: set[str]) -> bool:
    model_id = str(item.get("name", "")).removeprefix("models/")
    return (
        model_id in eligible
        and model_id in TEXT_MODEL_IDS
        and model_allowed(model_id)
        and "generateContent" in item.get("supportedGenerationMethods", [])
        and not any(part in model_id for part in ("image", "tts", "audio", "live", "transcribe", "robotics"))
    )


def model_rank(item: dict):
    """Prefer newer families, then Pro/Flash/Lite; this is not a benchmark."""
    model_id = item["id"]
    match = re.search(r"gemini-(\d+)(?:\.(\d+))?", model_id)
    major, minor = (int(match[1]), int(match[2] or 0)) if match else (0, 0)
    family = 0 if "pro" in model_id else 2 if "lite" in model_id else 1
    return (-major, -minor, family, "preview" in model_id or "exp" in model_id, model_id)


def supported_parameters(model_id: str) -> list[str]:
    # Only families reviewed with the current streaming/structured/vision setup.
    if model_id in TEXT_MODEL_IDS:
        return ["structured_outputs", "streaming", "vision"]
    return []


def mark_model_unavailable(model: str, seconds: int = 60):
    reset_key_cooldowns()
    _cooldowns[model] = monotonic() + seconds


def mark_model_available(model: str):
    reset_key_cooldowns()
    _cooldowns.pop(model, None)


def model_availability(model: str) -> dict:
    reset_key_cooldowns()
    remaining = max(0, int(_cooldowns.get(model, 0) - monotonic() + 0.999))
    return {"available": remaining == 0, "retry_after_seconds": remaining}


def availability(models: list[dict]) -> list[dict]:
    return [{**item, **model_availability(item["id"])} for item in models if model_allowed(item["id"])]


def catalog_entry(item: dict) -> dict:
    model_id = item["name"].removeprefix("models/")
    return {
        "id": model_id,
        "name": item.get("displayName", item["name"]),
        "context_length": item.get("inputTokenLimit", 0),
        "output_token_limit": item.get("outputTokenLimit", 0),
        "supported_parameters": supported_parameters(model_id),
        "pricing": {"prompt": "0", "completion": "0"},
        "free_tier_eligible": True,
    }


def _save_catalog(items: list[dict], eligible: set[str], key: str) -> list[dict]:
    global _catalog, _expires, _catalog_key
    models = [catalog_entry(item) for item in items if is_free_text_model(item, eligible)]
    models = sorted({model["id"]: model for model in models}.values(), key=model_rank)
    if not models:
        raise CatalogVerificationError("No compatible, verified free-tier Gemini text models were listed for this API key. Check AI Studio model access and refresh.")
    if key != config.gemini_api_key:
        raise CatalogVerificationError("Gemini credentials changed during discovery. Refresh the model list.")
    _catalog, _expires, _catalog_key = models, monotonic() + 300, key
    return availability(models)


def get_free_models_sync() -> list[dict]:
    """Synchronous catalog access for OCR workers, no event-loop crossing."""
    if not config.gemini_api_key:
        raise ValueError("Gemini is not configured.")
    if _catalog and _catalog_key == config.gemini_api_key and monotonic() < _expires:
        return availability(_catalog)
    key = config.gemini_api_key
    deadline, token, seen, items = monotonic() + 30, None, set(), []
    with httpx.Client(timeout=12, follow_redirects=True) as client:
        try:
            pricing = client.get(PRICING_URL)
            pricing.raise_for_status()
            eligible = _verified_pricing(pricing.text)
        except (httpx.HTTPError, httpx.TimeoutException) as error:
            eligible = _pricing_fallback(error)
        for _ in range(10):
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("Gemini catalog deadline reached.")
            response = client.get(
                f"{GEMINI_API_URL}/models",
                headers={"x-goog-api-key": key},
                params={"pageSize": 1000, **({"pageToken": token} if token else {})},
                timeout=remaining,
            )
            response.raise_for_status()
            data = response.json()
            items.extend(data.get("models", []))
            token = data.get("nextPageToken")
            if not token:
                break
            if token in seen:
                raise ValueError("Repeated Gemini model page token.")
            seen.add(token)
        else:
            raise ValueError("Gemini model catalog exceeded the page limit.")
    return _save_catalog(items, eligible, key)


async def get_free_models(*, refresh: bool = False) -> list[dict]:
    """Discover authenticated models and cache verified eligibility for 5 minutes.

    Never guess pricing or key access after verification fails.
    """
    if not config.gemini_api_key:
        raise ValueError("Add GEMINI_API_KEY to backend/.env and restart the backend.")
    async with _catalog_lock:
        if not refresh and _catalog and _catalog_key == config.gemini_api_key and monotonic() < _expires:
            return availability(_catalog)
        key = config.gemini_api_key
        async with asyncio.timeout(30), httpx.AsyncClient(timeout=12, follow_redirects=True) as client:
            try:
                pricing = await client.get(PRICING_URL)
                pricing.raise_for_status()
                eligible = _verified_pricing(pricing.text)
            except (httpx.HTTPError, httpx.TimeoutException) as error:
                eligible = _pricing_fallback(error)
            items = []
            page_token = None
            seen_tokens = set()
            while True:
                response = await client.get(
                    f"{GEMINI_API_URL}/models",
                    headers={"x-goog-api-key": key},
                    params={"pageSize": 1000, **({"pageToken": page_token} if page_token else {})},
                )
                response.raise_for_status()
                data = response.json()
                items.extend(data.get("models", []))
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
                if page_token in seen_tokens:
                    raise ValueError("Gemini returned a repeated model page token.")
                seen_tokens.add(page_token)
                if len(seen_tokens) >= 10:
                    raise ValueError("Gemini model catalog exceeded the page limit.")
        return _save_catalog(items, eligible, key)
