"""Read-only pricing/catalog audit. Never prints credentials or sends user content.

Without local credentials, only public pricing is checked. Models.list proves
listing/capabilities, not that quota remains or generation succeeds.
"""

import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from app.core.config import config
from app.services.free_models import (
    PRICING_URL,
    RETIRED_MODEL_IDS,
    TEXT_MODEL_IDS,
    free_tier_ids,
    get_free_models,
    model_allowed,
)


async def main():
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        response = await client.get(PRICING_URL)
        response.raise_for_status()
    eligible = free_tier_ids(response.text)
    result = {
        "pricing_verified": True,
        "public_eligible_compatible_ids": sorted(m for m in eligible & TEXT_MODEL_IDS if model_allowed(m)),
        "known_retired_text_ids": sorted(RETIRED_MODEL_IDS),
        "key_configured": bool(config.gemini_api_key),
        "key_access": "not checked: no configured local API key",
        "generation_probed": False,
    }
    if config.gemini_api_key:
        models = await get_free_models(refresh=True)
        result["key_access"] = [{"id": m["id"], "available": m["available"]} for m in models]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
