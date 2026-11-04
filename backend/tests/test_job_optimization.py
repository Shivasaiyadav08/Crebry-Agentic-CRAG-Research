import asyncio
from time import time

import pytest
from langchain_core.messages import AIMessageChunk
from sqlalchemy import func, select

from app.core.config import config
from app.models.database import Message, ResearchEvent, ResearchJob, WorkflowRun
from test_api import upload
from test_graph import mock_ai


def saved_job(client, workspace):
    from app.models.database import Conversation

    manager = client.app.state.jobs
    with client.app.state.db() as db:
        db.add(Conversation(id="batch-conversation", workspace_id=workspace, title="Synthetic"))
        db.flush()
        db.add(WorkflowRun(id="batch-run", conversation_id="batch-conversation"))
        db.flush()
        db.add(
            ResearchJob(
                id="batch-run",
                owner_id="user_alice",
                request_key="synthetic",
                fingerprint="synthetic",
                snapshot={},
                deadline_at=time() + 60,
                lease_owner=manager.owner,
                lease_until=time() + 600,
                status="running",
            )
        )
        db.commit()
    return manager


def test_batch_and_checkpoint_commit_ordered_events_and_detached_state(client, workspace):
    manager = saved_job(client, workspace)
    snapshot = {"sources": [{"id": "D1", "text": "Original evidence"}]}
    payloads = [{"type": "grade", "source": {"id": "D1"}}, {"type": "node_end", "node": "grade"}]
    manager.checkpoint("batch-run", snapshot, "assess", payloads)
    snapshot["sources"][0]["text"] = "Mutated later"
    events, _ = manager.events_after("batch-run", 0)
    assert [e["event_id"] for e in events] == [1, 2]
    with client.app.state.db() as db:
        job = db.get(ResearchJob, "batch-run")
        assert job.next_node == "assess"
        assert job.snapshot["sources"][0]["text"] == "Original evidence"
        assert len(db.get(WorkflowRun, "batch-run").events) == 2


def test_failed_event_batch_cannot_advance_checkpoint(client, workspace):
    manager = saved_job(client, workspace)
    with pytest.raises(KeyError):
        manager.checkpoint("batch-run", {"advanced": True}, "generate", [{"missing_type": True}])
    with client.app.state.db() as db:
        assert db.get(ResearchJob, "batch-run").snapshot == {}
        assert db.scalar(select(func.count(ResearchEvent.sequence))) == 0


def test_final_publication_is_idempotent_and_token_batches_skip_trace(client, workspace):
    manager = saved_job(client, workspace)
    final = {"type": "final", "answer": "Synthetic answer", "sources": [], "valid": False}
    manager.append_many("batch-run", [{"type": "token", "text": "Draft"}, final])
    manager.append("batch-run", final)
    events, _ = manager.events_after("batch-run", 0)
    assert [e["event_id"] for e in events] == [1, 2, 3]
    assert events[1]["message"]["id"] == events[2]["message"]["id"]
    with client.app.state.db() as db:
        assert db.scalar(select(func.count(Message.id))) == 1
        assert all(e["type"] != "token" for e in db.get(WorkflowRun, "batch-run").events)


def test_first_token_is_durable_before_the_second_token_arrives(client, workspace, monkeypatch):
    monkeypatch.setattr(config, "gemini_api_key", "synthetic")
    mock_ai(monkeypatch)
    upload(client, workspace)
    checked = []
    persisted = asyncio.Event()
    loop = None
    original_append = client.app.state.jobs.append_many

    def append(run_id, payloads):
        original_append(run_id, payloads)
        if any(p.get("text") == "Astronomy " for p in payloads):
            loop.call_soon_threadsafe(persisted.set)

    monkeypatch.setattr(client.app.state.jobs, "append_many", append)

    class Model:
        async def astream(self, _messages):
            nonlocal loop
            loop = asyncio.get_running_loop()
            yield AIMessageChunk(content="Astronomy ")
            # LangGraph runs producer and consumer concurrently. The next provider
            # chunk waits until persistence acknowledges the first one.
            await asyncio.wait_for(persisted.wait(), timeout=2)
            with client.app.state.db() as db:
                tokens = db.scalars(select(ResearchEvent).order_by(ResearchEvent.sequence)).all()
                assert any(e.payload.get("text") == "Astronomy " for e in tokens)
                checked.append(True)
            yield AIMessageChunk(content="studies stars [D1].")

    monkeypatch.setattr("app.agents.nodes.create_llm", lambda _: Model())
    response = client.post(
        "/api/chat/stream", json={"workspace_id": workspace, "question": "What is astronomy?"}
    )
    assert response.status_code == 200
    assert checked == [True]
    assert '"valid": true' in response.text
