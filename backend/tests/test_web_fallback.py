import json

import httpx
import pytest
from langchain_core.messages import AIMessageChunk

from app.agents.graph import build_graph
from app.agents.state import AnswerValidation, EvidenceEvaluation, RewrittenQuery
from app.core.config import config
from app.services.ai import WebSearchError, create_llm, search_web, call_with_free_fallback
from app.services.free_models import is_free_text_model, model_allowed


@pytest.mark.parametrize(
    "model", ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.8-flash-preview", "gemini-3.6-flash-lite"]
)
def test_removed_models_are_never_eligible_or_called(model, monkeypatch):
    monkeypatch.setattr(config, "gemini_api_key", "test-only")
    assert not model_allowed(model)
    assert not is_free_text_model(
        {"name": "models/" + model, "supportedGenerationMethods": ["generateContent"]}, {model}
    )
    with pytest.raises(ValueError, match="disabled"):
        create_llm(model)


async def test_saved_fallback_plan_cannot_reintroduce_removed_models():
    calls = []

    async def operation(model):
        calls.append(model)
        return "answer"

    await call_with_free_fallback(
        {
            "models": ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"],
            "active": "gemini-3.8-flash",
            "unavailable": [],
        },
        operation,
        lambda *_: None,
    )
    assert calls == ["gemini-3.5-flash-lite"]


@pytest.mark.parametrize("status", [401, 403, 429, 432, 433, 503])
async def test_tavily_errors_are_specific_and_never_expose_provider_body(monkeypatch, status):
    monkeypatch.setattr(config, "tavily_api_key", "test-only")
    original = httpx.AsyncClient
    monkeypatch.setattr(
        "app.services.ai.httpx.AsyncClient",
        lambda **_: original(
            transport=httpx.MockTransport(lambda _: httpx.Response(status, text="secret diagnostics"))
        ),
    )
    with pytest.raises(WebSearchError) as error:
        await search_web("Brazil football captain")
    assert "Tavily" in str(error.value) and str(status) in str(error.value)
    assert "secret diagnostics" not in str(error.value)
    assert "Gemini" not in str(error.value)


async def test_youtube_documents_to_brazil_web_answer_preserves_intent_and_citations(vectors, monkeypatch):
    monkeypatch.setattr(config, "tavily_api_key", "test-only")
    vectors.index(
        "workspace",
        "doc",
        "YouTube.pdf",
        [{"text": "YouTube channel analytics and earning revenue.", "page": 1}],
        "v1",
    )
    question = "who is captain of brazil football team"
    rewritten = []

    async def structured(model, schema, instruction, data):
        if schema is EvidenceEvaluation:
            return schema(
                assessment={"sufficient": False, "reason": "Unrelated", "missing_information": "Football evidence"},
                supporting_ids=[],
                grades=[
                    {"id": item["id"], "score": 0, "reason": "YouTube is unrelated to football."}
                    for item in data["chunks"]
                ]
            )
        if schema is RewrittenQuery:
            rewritten.append(data)
            raise AssertionError(
                "A standalone football question must not be rewritten using YouTube history."
            )
        if schema is AnswerValidation:
            assert data["sources"][0]["id"] == "W1"
            return schema(grounded=True, reason="The answer cites the supplied web evidence.")
        raise AssertionError("Unrelated documents cannot establish sufficient evidence.")

    original = httpx.AsyncClient

    def search_response(request):
        payload = json.loads(request.content)
        assert payload["query"] == question
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.org/brazil-match-report",
                        "title": "Brazil match report",
                        "content": "The official report identifies the captain for that match.",
                    }
                ]
            },
        )

    monkeypatch.setattr(
        "app.services.ai.httpx.AsyncClient",
        lambda **_: original(transport=httpx.MockTransport(search_response)),
    )
    monkeypatch.setattr("app.agents.nodes.structured_call", structured)

    class Model:
        async def astream(self, messages):
            evidence = json.loads(messages[-1].content)["evidence"]
            assert all(item["kind"] == "web" for item in evidence)
            yield AIMessageChunk(content="The report identifies the captain for that match [W1].")

    monkeypatch.setattr("app.agents.nodes.create_llm", lambda _: Model())
    events = [
        part["data"]
        async for part in build_graph(vectors).astream(
            {
                "workspace_id": "workspace",
                "question": question,
                "history": [{"role": "user", "content": "Summarize my YouTube document"}],
                "settings": {
                    "model": "test",
                    "retrieval_count": 6,
                    "relevance_threshold": 0.5,
                    "validate_answer": True,
                    "max_retries": 1,
                },
                "sources": [],
                "warnings": [],
                "attempts": 0,
            },
            stream_mode="custom",
            version="v2",
        )
    ]
    assert next(event for event in events if event["type"] == "decision")["route"] == "web"
    assert not rewritten
    assert events[-1]["sources"][0]["id"] == "W1"
    assert events[-1]["valid"] and "[W1]" in events[-1]["answer"]


async def test_missing_tavily_reason_is_saved_in_answer(vectors, monkeypatch):
    monkeypatch.setattr(config, "tavily_api_key", "")
    events = [
        part["data"]
        async for part in build_graph(vectors).astream(
            {
                "workspace_id": "empty",
                "question": "Brazil football captain",
                "history": [],
                "sources": [],
                "warnings": [],
                "attempts": 0,
                "settings": {
                    "model": "test",
                    "retrieval_count": 6,
                    "relevance_threshold": 0.5,
                    "validate_answer": True,
                    "max_retries": 1,
                },
            },
            stream_mode="custom",
            version="v2",
        )
    ]
    assert "TAVILY_API_KEY" in events[-1]["answer"]
    assert events[-1]["sources"] == []
