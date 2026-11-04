import asyncio
from concurrent.futures import ThreadPoolExecutor
from time import time

import pytest
from sqlalchemy import select

from app.core.config import config
from app.models.database import Conversation, RoutingState, Workspace, create_database
from app.services.ai import ModelFallbackExhausted, call_with_free_fallback
from app.services.routing import RoutingStore


class ProviderFailure(Exception):
    def __init__(self, code):
        self.code = code


@pytest.fixture
def ledger(tmp_path):
    engine, sessions = create_database(tmp_path)
    with sessions() as db:
        db.add(Workspace(id="workspace", name="Research", owner_id="user_alice"))
        db.flush()
        db.add(Conversation(id="conversation", workspace_id="workspace", title="Question"))
        db.commit()
    yield sessions, RoutingStore(sessions, "google-project")
    engine.dispose()


def plan():
    return {"models": ["A", "B", "C", "D"], "conversation_id": "conversation", "unavailable": []}


async def test_distinct_fallback_sticky_across_instances_and_cascade_after_recovery(ledger):
    sessions, store = ledger
    calls = []

    async def first(model):
        calls.append(model)
        if model in {"A", "B"}:
            raise ProviderFailure(503)
        return "grounded [D1]"

    assert await call_with_free_fallback(plan(), first, lambda *_: None, store=store) == "grounded [D1]"
    assert calls == ["A", "B", "C"]
    # Construct another store: affinity must come from the database, not the object.
    other = RoutingStore(sessions, "google-project")
    calls.clear()
    assert await call_with_free_fallback(plan(), first, lambda *_: None, store=other) == "grounded [D1]"
    assert calls == ["C"]
    # Simulate recovery of A/B; C now fails and must cascade without repeating C.
    with other.transaction() as (_, state):
        for key in ("generate:A", "generate:B"):
            state[key] = {**state[key], "until": 0}
    calls.clear()

    async def later(model):
        calls.append(model)
        if model == "C":
            raise ProviderFailure(503)
        return "recovered [D1]"

    assert await call_with_free_fallback(plan(), later, lambda *_: None, store=other) == "recovered [D1]"
    assert calls[0] == "C" and len(calls) == 2 and calls[1] != "C"


async def test_exhaustive_distinct_pass_before_retry(monkeypatch):
    monkeypatch.setattr(config, "gemini_max_model_attempts", 3)
    calls = []

    async def operation(model):
        calls.append(model)
        if model != "D":
            raise ProviderFailure(503)
        return "answer"

    assert await call_with_free_fallback(plan(), operation, lambda *_: None) == "answer"
    assert calls == ["A", "B", "C", "D"]


def test_atomic_concurrency_and_project_quota_do_not_reset_between_instances(ledger, monkeypatch):
    sessions, store = ledger
    monkeypatch.setattr(config, "gemini_max_concurrency", 1)
    monkeypatch.setattr(config, "gemini_free_rpm", 1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        leases = list(pool.map(lambda _: RoutingStore(sessions, "google-project").reserve("A", "generate", 10, 30), range(4)))
    assert sum(lease is not None for lease in leases) == 1
    store.finish(next(lease for lease in leases if lease), "A", "generate", 5, success=True)
    assert RoutingStore(sessions, "google-project").reserve("A", "generate", 10, 30) is None
    with sessions() as db:
        assert db.scalar(select(RoutingState)).payload["quota:A"]["requests"] == 1


async def test_health_is_task_specific_and_quota_is_model_wide(ledger):
    _, store = ledger
    lease = store.reserve("A", "grade", 10, 30)
    store.finish(lease, "A", "grade", 10, status=503)
    assert store.ordered(["A"], "grade") == []
    assert store.ordered(["A"], "generate") == ["A"]
    lease = store.reserve("A", "generate", 10, 30)
    store.finish(lease, "A", "generate", 10, status=429)
    assert store.ordered(["A"], "different_task") == []


async def test_strict_selection_and_capabilities_fail_closed():
    calls = []

    async def operation(model):
        calls.append(model)
        raise ProviderFailure(503)

    p = {**plan(), "selected": "B", "automatic": False}
    with pytest.raises(ModelFallbackExhausted):
        await call_with_free_fallback(p, operation, lambda *_: None)
    assert calls == ["B"]
    p = {**plan(), "catalog": [{"id": "A", "free_tier_eligible": False}]}
    with pytest.raises(ModelFallbackExhausted):
        await call_with_free_fallback(p, operation, lambda *_: None)
    assert calls == ["B"]


async def test_shared_expired_deadline_makes_no_provider_calls():
    p = {**plan(), "deadline_at": time() - 1}

    async def operation(model):
        pytest.fail("Expired requests must never call the provider")

    with pytest.raises(ModelFallbackExhausted, match="time limit"):
        await call_with_free_fallback(p, operation, lambda *_: None)


async def test_cancellation_releases_shared_lease(ledger):
    sessions, store = ledger
    started = asyncio.Event()

    async def stalled(model):
        started.set()
        await asyncio.Event().wait()

    running = asyncio.create_task(call_with_free_fallback(plan(), stalled, lambda *_: None, store=store))
    await started.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    with sessions() as db:
        assert db.get(RoutingState, "google-project").payload["leases"] == {}


def test_lightweight_routing_prefers_measured_latency_and_keeps_generation_affinity(ledger):
    _, store = ledger
    models = ["gemini-2.5-pro", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]
    assert store.ordered(models, "EvidenceEvaluation")[0] == "gemini-3.1-flash-lite"
    for model, latency in zip(models, (30, 90, 60)):
        lease = store.reserve(model, "EvidenceEvaluation", 1, 30)
        store.finish(lease, model, "EvidenceEvaluation", latency, success=True)
    assert store.ordered(models, "EvidenceEvaluation")[0] == "gemini-2.5-pro"
    lease = store.reserve(models[2], "generate", 1, 30)
    store.finish(lease, models[2], "generate", 60, success=True, conversation_id="conversation")
    assert store.ordered(models, "generate", "conversation")[0] == models[2]


async def test_persisted_plan_revalidates_access_after_key_change(monkeypatch):
    from app.services import free_models

    monkeypatch.setattr(config, "gemini_api_key", "new-synthetic-key")
    checked, calls = [], []

    async def catalog():
        checked.append(True)
        return [{"id": "gemini-3.1-flash-lite", "free_tier_eligible": True,
                 "supported_parameters": ["streaming"], "context_length": 1000000, "output_token_limit": 8192}]

    monkeypatch.setattr(free_models, "get_free_models", catalog)

    async def operation(model):
        calls.append(model)
        return "Synthetic grounded answer"

    p = {"models": ["old-access"], "credential_fingerprint": "old-fingerprint", "unavailable": []}
    await call_with_free_fallback(p, operation, lambda *_: None)
    assert checked == [True]
    assert calls == ["gemini-3.1-flash-lite"]


async def test_retired_model_in_saved_plan_never_receives_a_provider_call():
    calls = []

    async def operation(model):
        calls.append(model)
        return "answer"

    p = {"models": ["gemini-3.1-flash-lite-preview", "gemini-3.1-flash-lite"], "unavailable": []}
    assert await call_with_free_fallback(p, operation, lambda *_: None) == "answer"
    assert calls == ["gemini-3.1-flash-lite"]
