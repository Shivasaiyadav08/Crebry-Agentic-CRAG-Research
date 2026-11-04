import json
import httpx
import pytest

from app.core.config import config
from app.agents.state import AnswerValidation, EvidenceEvaluation
from app.models.database import AppSettings
from test_graph import mock_ai


def upload(client, workspace, text="Astronomy studies stars"):
    return client.post(
        f"/api/workspaces/{workspace}/documents",
        files=[("files", ("space.txt", text.encode(), "text/plain"))],
    )


def test_health_and_validation(client):
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.post("/api/workspaces", json={"name": "   "}).status_code == 422
    assert client.get("/api/settings").status_code == 404
    assert client.post("/api/chat/stream", json={"workspace_id": "bad", "question": "  "}).status_code == 422


def test_model_list_does_not_erase_key_specific_cooldowns(client, monkeypatch):
    async def models(**_):
        return [{"id": "gemini-3.1-flash-lite", "available": False, "retry_after_seconds": 40}]

    monkeypatch.setattr("app.api.routes.get_free_models", models)
    result = client.get("/api/models").json()
    assert result[0]["available"] is False
    assert result[0]["retry_after_seconds"] == 40


def test_paid_model_is_rejected(client, workspace, monkeypatch):
    monkeypatch.setattr(config, "gemini_api_key", "test-only")
    response = client.post(
        "/api/chat/stream", json={"workspace_id": workspace, "question": "Stars?", "model": "provider/paid"}
    )
    assert response.status_code == 422
    assert "free" in response.json()["detail"]


@pytest.mark.parametrize("threshold", [-0.01, 1.01, "NaN", "Infinity", "invalid"])
def test_invalid_relevance_threshold_is_rejected(client, workspace, threshold):
    response = client.post(
        "/api/chat/stream",
        json={"workspace_id": workspace, "question": "Stars?", "relevance_threshold": threshold},
    )
    assert response.status_code == 422
    assert any(error["loc"][-1] == "relevance_threshold" for error in response.json()["detail"])
    assert client.get(f"/api/workspaces/{workspace}/conversations").json() == []


@pytest.mark.parametrize(
    "threshold,accepted", [(None, False), (0, True), (0.6, True), (0.61, False), (1, False)]
)
def test_chat_relevance_threshold_is_request_scoped(client, workspace, monkeypatch, threshold, accepted):
    monkeypatch.setattr(config, "gemini_api_key", "test-only")
    mock_ai(monkeypatch, score=0.6)
    upload(client, workspace)
    with client.app.state.db() as db:
        db.get(AppSettings, 1).relevance_threshold = 0.65
        db.commit()
    request = {"workspace_id": workspace, "question": "Stars?"}
    if threshold is not None:
        request["relevance_threshold"] = threshold
    response = client.post("/api/chat/stream", json=request)
    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    grade = next(event["source"] for event in events if event["type"] == "grade")
    assert grade["accepted"] is accepted
    assert any(event["type"] == "done" for event in events)
    if accepted:
        assert any(event["type"] == "node_start" and event["node"] == "validate" for event in events)
    with client.app.state.db() as db:
        assert db.get(AppSettings, 1).relevance_threshold == 0.65


def test_stale_gemini_choice_uses_verified_catalog_instead(client, workspace, monkeypatch):
    monkeypatch.setattr(config, "gemini_api_key", "test-only")
    mock_ai(monkeypatch)
    upload(client, workspace)
    response = client.post(
        "/api/chat/stream",
        json={"workspace_id": workspace, "question": "Stars?", "model": "gemini-retired-model"},
    )
    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    fallback = next(event for event in events if event["type"] == "model_fallback")
    assert fallback["model"] == "gemini-test-flash"
    assert "no longer" in fallback["message"]


def test_all_gemini_models_in_cooldown_returns_actionable_error(client, workspace, monkeypatch):
    monkeypatch.setattr(config, "gemini_api_key", "test-only")

    async def exhausted():
        return [{"id": "gemini-test-flash", "available": False}]

    monkeypatch.setattr("app.api.routes.get_free_models", exhausted)
    response = client.post("/api/chat/stream", json={"workspace_id": workspace, "question": "Stars?"})
    assert response.status_code == 503
    assert "quota" in response.json()["detail"]


@pytest.mark.parametrize("status", [401, 403, 404, 429, 503])
def test_catalog_errors_keep_status_without_exposing_credentials(client, workspace, monkeypatch, status):
    monkeypatch.setattr(config, "gemini_api_key", "test-only")

    async def failing(refresh=False):
        request = httpx.Request("GET", "https://example.org")
        response = httpx.Response(status, request=request)
        raise httpx.HTTPStatusError("private-api-key-and-provider-body", request=request, response=response)

    monkeypatch.setattr("app.api.routes.get_free_models", failing)
    for response in (
        client.get("/api/models?refresh=true"),
        client.post("/api/chat/stream", json={"workspace_id": workspace, "question": "Stars?"}),
    ):
        assert response.status_code == 503
        assert f"HTTP {status}" in response.json()["detail"]
        assert "private-api-key" not in response.text


@pytest.mark.parametrize("failure_schema", [EvidenceEvaluation, AnswerValidation])
@pytest.mark.parametrize("status", [401, 503])
def test_ai_service_failure_stops_without_web_search_or_answer_regeneration(
    client, workspace, monkeypatch, failure_schema, status
):
    import app.agents.nodes as nodes
    from google.genai.errors import ClientError

    monkeypatch.setattr(config, "gemini_api_key", "test-only")
    mock_ai(monkeypatch)
    upload(client, workspace)
    original = nodes.structured_call

    async def failing(model, schema, instruction, data):
        if schema is failure_schema:
            raise ClientError(status, {"error": {"code": status, "message": "private-provider-body"}})
        return await original(model, schema, instruction, data)

    monkeypatch.setattr(nodes, "structured_call", failing)
    response = client.post("/api/chat/stream", json={"workspace_id": workspace, "question": "Stars?"})
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    error = next(event for event in events if event["type"] == "error")
    assert f"HTTP {status}" in error["message"]
    assert "private-provider-body" not in response.text
    started_nodes = [event["node"] for event in events if event["type"] == "node_start"]
    assert "web_search" not in started_nodes
    assert "rewrite" not in started_nodes
    assert started_nodes.count("generate") <= 1
    assert not any(event["type"] == "final" for event in events)


def test_upload_duplicate_preview_reindex_delete(client, workspace):
    response = upload(client, workspace)
    assert response.status_code == 201
    document = response.json()[0]["document"]
    assert document["status"] == "ready" and document["chunk_count"] > 0
    assert upload(client, workspace).json()[0]["duplicate"]
    assert len(client.get(f"/api/workspaces/{workspace}/documents").json()) == 1
    path = f"/api/workspaces/{workspace}/documents/{document['id']}"
    assert "Astronomy" in client.get(path).json()["text"]
    assert client.get(path + "/file").content == b"Astronomy studies stars"
    assert client.post(path + "/reindex").status_code == 200
    assert client.app.state.vectors.collection(workspace).count() == document["chunk_count"]
    assert client.delete(path).status_code == 204
    assert client.get(path).status_code == 404
    assert client.app.state.vectors.retrieve(workspace, "stars") == []


def test_document_workspace_isolation(client, workspace):
    document = upload(client, workspace).json()[0]["document"]
    other = client.post("/api/workspaces", json={"name": "Private"}).json()["id"]
    path = f"/api/workspaces/{other}/documents/{document['id']}"
    assert client.get(path).status_code == 404
    assert client.delete(path).status_code == 404
    assert client.get(f"/api/workspaces/{other}/documents").json() == []


def test_bad_upload_and_size_limit(client, workspace, monkeypatch):
    monkeypatch.setattr(config, "max_upload_mb", 1)
    response = client.post(
        f"/api/workspaces/{workspace}/documents",
        files=[
            ("files", ("virus.exe", b"no")),
            ("files", ("large.txt", b"x" * 1_048_577)),
            ("files", ("empty.txt", b"")),
        ],
    )
    assert all("error" in result for result in response.json())


def test_missing_api_key_has_actionable_error(client, workspace):
    response = client.post("/api/chat/stream", json={"workspace_id": workspace, "question": "Stars?"})
    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_unrelated_browser_origins_cannot_write(client):
    response = client.post(
        "/api/workspaces", json={"name": "Unwanted"}, headers={"Origin": "https://unrelated.example"}
    )
    assert response.status_code == 403
    assert (
        client.post(
            "/api/workspaces", json={"name": "Allowed"}, headers={"Origin": config.frontend_origin}
        ).status_code
        == 201
    )


def test_sse_persistence_regeneration_and_isolation(client, workspace, monkeypatch):
    monkeypatch.setattr(config, "gemini_api_key", "test-secret-never-expose")
    mock_ai(monkeypatch)
    upload(client, workspace)
    request = {"workspace_id": workspace, "question": "What is astronomy?"}
    response = client.post("/api/chat/stream", json=request)
    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[-1]["type"] == "done"
    conversation_id = events[0]["conversation_id"]
    run_id = events[0]["run_id"]
    path = f"/api/workspaces/{workspace}/conversations/{conversation_id}"
    assert len(client.get(path).json()["messages"]) == 2
    run = client.get(f"/api/workspaces/{workspace}/workflows/{run_id}").json()
    assert run["status"] == "completed"
    assert any(event["type"] == "grade" for event in run["events"])
    assert "test-secret" not in response.text
    regen = client.post(
        "/api/chat/stream", json={**request, "conversation_id": conversation_id, "regenerate": True}
    )
    assert '"type": "done"' in regen.text
    assert len(client.get(path).json()["messages"]) == 2
    other = client.post("/api/workspaces", json={"name": "Other"}).json()["id"]
    assert client.get(f"/api/workspaces/{other}/conversations/{conversation_id}").status_code == 404
    assert client.get(f"/api/workspaces/{other}/workflows/{run_id}").status_code == 404
    assert (
        client.post(
            "/api/chat/stream", json={**request, "workspace_id": other, "conversation_id": conversation_id}
        ).status_code
        == 404
    )
    assert client.delete(f"/api/workspaces/{workspace}").status_code == 204
    assert client.get(path).status_code == 404


def test_failed_reindex_preserves_previous_vectors(client, workspace, monkeypatch):
    document = upload(client, workspace).json()[0]["document"]

    def fail(*args, **kwargs):
        raise RuntimeError("Embedding unavailable")

    monkeypatch.setattr(client.app.state.vectors, "index", fail)
    response = client.post(f"/api/workspaces/{workspace}/documents/{document['id']}/reindex")
    assert response.status_code == 503
    assert client.app.state.vectors.retrieve(workspace, "Astronomy")[0]["document_id"] == document["id"]
