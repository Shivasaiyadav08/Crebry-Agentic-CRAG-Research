import json

from sqlalchemy import func, select

from app.agents.state import EvidenceEvaluation
from app.core.config import config
from app.models.database import Message, ResearchJob, WorkflowRun
from test_api import mock_ai, upload


def frames(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_idempotency_replays_same_job_and_preserves_messages(client, workspace, monkeypatch):
    monkeypatch.setattr(config, "gemini_api_key", "test-only")
    mock_ai(monkeypatch)
    upload(client, workspace)
    body = {"workspace_id": workspace, "question": "What is astronomy?", "idempotency_key": "device-request-123"}
    first = frames(client.post("/api/chat/stream", json=body))
    second = frames(client.post("/api/chat/stream", json=body))
    assert first[0]["run_id"] == second[0]["run_id"]
    with client.app.state.db() as db:
        assert db.scalar(select(func.count(WorkflowRun.id))) == 1
        assert db.scalar(select(func.count(Message.id))) == 2
    assert client.post("/api/chat/stream", json={**body, "question": "Different input"}).status_code == 409


def test_retry_resumes_at_evaluation_without_repeating_retrieval(client, workspace, monkeypatch):
    import app.agents.nodes as nodes

    monkeypatch.setattr(config, "gemini_api_key", "test-only")
    mock_ai(monkeypatch)
    upload(client, workspace)
    retrieval_calls = []
    original_retrieve = client.app.state.vectors.retrieve

    def retrieve(*args):
        retrieval_calls.append(args)
        return original_retrieve(*args)

    monkeypatch.setattr(client.app.state.vectors, "retrieve", retrieve)
    structured = nodes.structured_call
    failing = True

    async def assessment(model, schema, instruction, data):
        if failing and schema is EvidenceEvaluation:
            error = RuntimeError("private-provider-body")
            error.code = 503
            raise error
        return await structured(model, schema, instruction, data)

    monkeypatch.setattr(nodes, "structured_call", assessment)
    events = frames(client.post("/api/chat/stream", json={"workspace_id": workspace, "question": "What is astronomy?"}))
    run_id, conversation_id = events[0]["run_id"], events[0]["conversation_id"]
    with client.app.state.db() as db:
        saved = db.get(ResearchJob, run_id)
        assert saved.next_node == "grade"
        assert saved.snapshot["retrieved"][0]["id"] == "D1"
        assert saved.status == "failed"
    failing = False
    # Advance task circuit recovery without waiting for wall-clock cooldown.
    with client.app.state.routing.transaction() as (_, state):
        for key in list(state):
            if key.startswith("EvidenceEvaluation:"):
                state[key] = {**state[key], "until": 0}
    root = f"/api/workspaces/{workspace}/workflows/{run_id}"
    cursor = events[-1]["event_id"]
    assert client.post(root + "/retry").status_code == 200
    recovered = frames(client.get(root + f"/events?after={cursor}"))
    assert recovered[-1]["status"] == "completed"
    assert len(retrieval_calls) == 1
    with client.app.state.db() as db:
        assert db.scalar(select(func.count(Message.id)).where(Message.conversation_id == conversation_id)) == 2
    assert client.post(root + "/retry").status_code == 409


def test_another_device_gets_saved_chat_and_other_account_cannot_resume(client, workspace, monkeypatch):
    monkeypatch.setattr(config, "gemini_api_key", "test-only")
    mock_ai(monkeypatch)
    upload(client, workspace)
    events = frames(client.post("/api/chat/stream", json={"workspace_id": workspace, "question": "What is astronomy?"}))
    run_id, conversation_id = events[0]["run_id"], events[0]["conversation_id"]
    root = f"/api/workspaces/{workspace}"
    device = {"X-Test-User": "user_alice", "User-Agent": "phone"}
    chat = client.get(root + f"/conversations/{conversation_id}", headers=device).json()
    assert len(chat["messages"]) == 2 and chat["latest_run"]["id"] == run_id
    stranger = {"X-Test-User": "user_bob"}
    assert client.get(root + f"/workflows/{run_id}/events", headers=stranger).status_code == 404
    for action in ("retry", "cancel"):
        assert client.post(root + f"/workflows/{run_id}/{action}", headers=stranger).status_code == 404
    replay = frames(client.get(root + f"/workflows/{run_id}/events", headers=device))
    assert replay[-1]["status"] == "completed"
