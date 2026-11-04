import asyncio
from time import monotonic

import httpx
import pytest

from app.core.config import config
from app.services import free_models
from app.services.ai import safe_service_error
from test_free_models import pricing_section


@pytest.fixture(autouse=True)
def empty_catalog(monkeypatch):
    monkeypatch.setattr(config, "gemini_api_key", "synthetic-key-a")
    monkeypatch.setattr(free_models, "_catalog", [])
    monkeypatch.setattr(free_models, "_expires", 0)
    monkeypatch.setattr(free_models, "_catalog_key", "")
    monkeypatch.setattr(free_models, "_catalog_lock", asyncio.Lock())
    monkeypatch.setattr(free_models, "_cooldowns", {})


def catalog_transport(monkeypatch, responder):
    original = httpx.AsyncClient
    monkeypatch.setattr(
        free_models.httpx, "AsyncClient", lambda **_: original(transport=httpx.MockTransport(responder))
    )


@pytest.mark.parametrize("status,body", [(503, "provider-secret"), (200, "changed markup")])
async def test_expired_pricing_fails_closed_without_static_fallback(monkeypatch, status, body):
    def respond(request):
        if str(request.url) == free_models.PRICING_URL:
            return httpx.Response(status, text=body)
        pytest.fail("Do not build a catalog before verifying prices")

    catalog_transport(monkeypatch, respond)
    with pytest.raises(free_models.CatalogVerificationError) as caught:
        await free_models.get_free_models()
    assert free_models._catalog == []
    assert "pricing" in safe_service_error(caught.value).lower()
    assert "provider-secret" not in safe_service_error(caught.value)


async def test_catalog_single_flight_ttl_and_key_specific_discovery(monkeypatch):
    calls = []

    def respond(request):
        calls.append(request)
        if str(request.url) == free_models.PRICING_URL:
            return httpx.Response(
                200, text=pricing_section("gemini-3.1-flash-lite") + pricing_section("gemini-3.5-flash-lite")
            )
        model = (
            "gemini-3.1-flash-lite"
            if request.headers["x-goog-api-key"].endswith("a")
            else "gemini-3.5-flash-lite"
        )
        return httpx.Response(
            200,
            json={"models": [{"name": "models/" + model, "supportedGenerationMethods": ["generateContent"]}]},
        )

    catalog_transport(monkeypatch, respond)
    results = await asyncio.gather(*(free_models.get_free_models() for _ in range(8)))
    assert len(calls) == 2
    assert all(r[0]["id"] == "gemini-3.1-flash-lite" for r in results)
    free_models.mark_model_unavailable("gemini-3.1-flash-lite")
    assert not (await free_models.get_free_models())[0]["available"]
    assert len(calls) == 2  # Temporary limits never force catalog eviction.
    monkeypatch.setattr(free_models, "_expires", monotonic() - 1)
    await free_models.get_free_models()
    assert len(calls) == 4
    monkeypatch.setattr(config, "gemini_api_key", "synthetic-key-b")
    assert (await free_models.get_free_models())[0]["id"] == "gemini-3.5-flash-lite"
    assert len(calls) == 6


async def test_rotation_during_discovery_cannot_stamp_old_access_with_new_key(monkeypatch):
    def respond(request):
        if str(request.url) == free_models.PRICING_URL:
            return httpx.Response(200, text=pricing_section("gemini-3.1-flash-lite"))
        monkeypatch.setattr(config, "gemini_api_key", "replacement-key")
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "models/gemini-3.1-flash-lite",
                        "supportedGenerationMethods": ["generateContent"],
                    }
                ]
            },
        )

    catalog_transport(monkeypatch, respond)
    with pytest.raises(free_models.CatalogVerificationError, match="changed"):
        await free_models.get_free_models()
    assert free_models._catalog_key == ""


@pytest.mark.parametrize(
    "model",
    [
        "gemini-3.1-flash-lite-preview",
        "gemini-3-pro-preview",
        "gemini-2.0-flash",
        "gemini-3.1-flash-image",
        "gemini-3.1-flash-live-preview",
        "gemini-unreviewed-preview",
    ],
)
def test_retired_or_incompatible_models_cannot_enter_saved_plans(model):
    assert not free_models.model_allowed(model)
    assert not free_models.is_free_text_model(
        {"name": "models/" + model, "supportedGenerationMethods": ["generateContent"]}, {model}
    )


def test_stable_31_is_not_confused_with_retired_preview():
    model = "gemini-3.1-flash-lite"
    assert free_models.model_allowed(model)
    assert free_models.is_free_text_model(
        {"name": "models/" + model, "supportedGenerationMethods": ["generateContent"]}, {model}
    )
    assert "vision" in free_models.supported_parameters(model)


def test_mdx_paid_column_cannot_be_mistaken_for_free_column():
    document = """## Model {% id="gemini-3.1-flash-lite" %}
{% tab title="Standard" %}{% table %}
* Feature
* Paid Tier
* Free Tier
---
* Input price
* Free of charge
* Not available
---
* Output price
* Free of charge
* Not available
{% /table %}{% /tab %}"""
    with pytest.raises(ValueError):
        free_models.free_tier_ids(document)
