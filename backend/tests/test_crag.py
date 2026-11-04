"""Focused CRAG routing, evidence preservation, and bounded agent recovery tests."""

import json

import pytest
from langchain_core.messages import AIMessageChunk
from pydantic import ValidationError

from app.agents.graph import next_node
from app.agents.state import CorrectionPlan, EvidenceSelection
from app.core.config import config
from test_graph import SETTINGS, mock_ai, run_graph


def test_invalid_tool_plan_schema_rejected():
    with pytest.raises(ValidationError):
        CorrectionPlan(action="delete_workspace", reason="not allowed")


def test_checkpoint_transitions_match_live_graph():
    assert next_node("assess", {"crag_verdict": "CORRECT"}) == "refine"
    assert next_node("assess", {"crag_verdict": "AMBIGUOUS"}) == "rewrite"
    assert next_node("web_search", {}) == "refine"
    assert (
        next_node("validate", {"valid": False, "attempts": 1, "settings": {"max_retries": 1}})
        == "plan_correction"
    )
    assert next_node("plan_correction", {"correction_action": "retrieve"}) == "retrieve"
    assert next_node("plan_correction", {"correction_action": "stop"}) == "finalize"


async def test_correct_documents_do_not_trigger_web_search(vectors, monkeypatch):
    vectors.index(
        "test_workspace",
        "doc",
        "notes.txt",
        [{"text": "Astronomy studies stars. Telescopes gather starlight.", "page": 1}],
        "v1",
    )
    mock_ai(monkeypatch)
    events = await run_graph(vectors)
    verdict = next(e for e in events if e["type"] == "decision")
    assert verdict["verdict"] == "CORRECT"
    assert not any(e.get("node") == "web_search" for e in events)
    assert next(e for e in events if e["type"] == "refinement")["mode"] == "selected"
    assert events[-1]["valid"]


async def test_ambiguous_evidence_combines_documents_and_web(vectors, monkeypatch):
    vectors.index(
        "test_workspace",
        "doc",
        "notes.txt",
        [{"text": "Astronomy studies stars. Observatories use telescopes.", "page": 1}],
        "v1",
    )
    mock_ai(monkeypatch, sufficient=False, answer="Astronomy studies stars [D1] and uses observatories [W1].")

    async def search(_):
        return [
            {
                "id": "W1",
                "kind": "web",
                "name": "Public paper",
                "url": "https://example.org",
                "text": "Observatories study astronomy.",
            }
        ]

    monkeypatch.setattr("app.agents.nodes.search_web", search)
    events = await run_graph(vectors)
    assert next(e for e in events if e["type"] == "decision")["verdict"] == "AMBIGUOUS"
    assert {s["id"] for s in events[-1]["sources"]} == {"D1", "W1"}


async def test_incorrect_document_replaces_context_with_web(vectors, monkeypatch):
    vectors.index(
        "test_workspace", "doc", "unrelated.txt", [{"text": "Car engines burn fuel.", "page": 1}], "v1"
    )
    mock_ai(monkeypatch, score=0.1, answer="Stars are studied in astronomy [W1].")

    async def search(_):
        return [
            {
                "id": "W1",
                "kind": "web",
                "name": "Public paper",
                "url": "https://example.org",
                "text": "Stars are studied in astronomy.",
            }
        ]

    monkeypatch.setattr("app.agents.nodes.search_web", search)
    events = await run_graph(vectors)
    assert next(e for e in events if e["type"] == "decision")["verdict"] == "INCORRECT"
    assert [s["id"] for s in events[-1]["sources"]] == ["W1"]


async def test_refinement_selects_verbatim_sentences_and_preserves_citations(vectors, monkeypatch):
    vectors.index(
        "test_workspace",
        "doc",
        "notes.txt",
        [{"text": "Astronomy studies stars. An irrelevant sentence describes racing cars.", "page": 1}],
        "v1",
    )
    mock_ai(monkeypatch)
    import app.agents.nodes as nodes

    previous = nodes.structured_call

    async def refined(model, schema, instruction, data):
        if schema is EvidenceSelection:
            return schema(keep_ids=[data["sentences"][0]["id"]])
        return await previous(model, schema, instruction, data)

    monkeypatch.setattr(nodes, "structured_call", refined)

    class Model:
        async def astream(self, messages):
            evidence = json.loads(messages[-1].content)["evidence"]
            assert evidence[0]["text"] == "Astronomy studies stars."
            assert evidence[0]["id"] == "D1"
            assert evidence[0]["name"] == "notes.txt"
            yield AIMessageChunk(content="Astronomy studies stars [D1].")

    monkeypatch.setattr(nodes, "create_llm", lambda _: Model())
    events = await run_graph(vectors)
    assert events[-1]["valid"]
    assert events[-1]["sources"][0]["id"] == "D1"


async def test_bad_refinement_ids_do_not_inject_text(vectors, monkeypatch):
    vectors.index(
        "test_workspace",
        "doc",
        "notes.txt",
        [{"text": "Astronomy studies stars. Other research studies galaxies.", "page": 1}],
        "v1",
    )
    mock_ai(monkeypatch)
    import app.agents.nodes as nodes

    previous = nodes.structured_call

    async def bad(model, schema, instruction, data):
        if schema is EvidenceSelection:
            return schema(keep_ids=["FAKE", "E2"])
        return await previous(model, schema, instruction, data)

    monkeypatch.setattr(nodes, "structured_call", bad)
    events = await run_graph(vectors)
    assert events[-1]["valid"]
    assert any("invalid IDs" in msg for msg in events[-1]["warnings"])


async def test_corrective_agent_can_retrieve_once_and_then_regenerate(vectors, monkeypatch):
    vectors.index(
        "test_workspace",
        "doc",
        "notes.txt",
        [{"text": "Astronomy studies stars. Telescopes study stellar light.", "page": 1}],
        "v1",
    )
    mock_ai(monkeypatch)
    import app.agents.nodes as nodes

    previous = nodes.structured_call
    retrievals = []
    original_retrieve = vectors.retrieve

    def retrieve(workspace, query, count):
        retrievals.append(query)
        return original_retrieve(workspace, query, count)

    monkeypatch.setattr(vectors, "retrieve", retrieve)
    answer_count = 0

    async def structured(model, schema, instruction, data):
        if schema is CorrectionPlan:
            return schema(action="retrieve", reason="Find precise source", query="astronomy stars")
        return await previous(model, schema, instruction, data)

    class Model:
        async def astream(self, messages):
            nonlocal answer_count
            answer_count += 1
            if answer_count == 1:
                yield AIMessageChunk(content="Unverifiable claim [D99].")
            else:
                yield AIMessageChunk(content="Astronomy studies stars [D1].")

    monkeypatch.setattr(nodes, "structured_call", structured)
    monkeypatch.setattr(nodes, "create_llm", lambda _: Model())
    events = await run_graph(vectors)
    assert events[-1]["valid"]
    assert len(retrievals) == 2
    assert retrievals[1] == "astronomy stars"
    assert any(e["type"] == "correction_plan" and e["action"] == "retrieve" for e in events)


async def test_planner_rejects_unavailable_web_tool(vectors, monkeypatch):
    vectors.index(
        "test_workspace", "doc", "notes.txt", [{"text": "Astronomy studies stars.", "page": 1}], "v1"
    )
    mock_ai(monkeypatch, answer="Invalid cite [D99].")
    monkeypatch.setattr(config, "tavily_api_key", "")
    import app.agents.nodes as nodes

    previous = nodes.structured_call

    async def invalid_action(model, schema, instruction, data):
        if schema is CorrectionPlan:
            return schema(action="web_search", reason="Try web", query="astronomy")
        return await previous(model, schema, instruction, data)

    monkeypatch.setattr(nodes, "structured_call", invalid_action)
    events = await run_graph(vectors)
    assert next(e for e in events if e["type"] == "correction_plan")["action"] == "regenerate"
    assert not any(e.get("node") == "web_search" for e in events)
    assert not events[-1]["valid"]


async def test_medium_confidence_is_ambiguous_even_if_coverage_looks_complete(vectors, monkeypatch):
    vectors.index(
        "test_workspace", "doc", "notes.txt", [{"text": "Astronomy studies stars.", "page": 1}], "v1"
    )
    mock_ai(monkeypatch, score=0.55, sufficient=True)

    async def search(_):
        return [
            {
                "id": "W1",
                "kind": "web",
                "name": "Additional research",
                "url": "https://example.org",
                "text": "Research on astronomy.",
            }
        ]

    monkeypatch.setattr("app.agents.nodes.search_web", search)
    events = await run_graph(vectors)
    assert next(event for event in events if event["type"] == "decision")["verdict"] == "AMBIGUOUS"
    assert any(event["type"] == "web_sources" for event in events)


async def test_lower_crag_band_rejects_untrustworthy_context_despite_zero_user_cutoff(vectors, monkeypatch):
    vectors.index(
        "test_workspace", "doc", "notes.txt", [{"text": "Random grocery product names.", "page": 1}], "v1"
    )
    mock_ai(monkeypatch, score=0.2)

    async def search(_):
        return [
            {
                "id": "W1",
                "kind": "web",
                "name": "Research report",
                "url": "https://example.org",
                "text": "Stars are studied in astronomy.",
            }
        ]

    monkeypatch.setattr("app.agents.nodes.search_web", search)
    events = await run_graph(vectors, {**SETTINGS, "relevance_threshold": 0.0})
    assert next(event for event in events if event["type"] == "decision")["verdict"] == "INCORRECT"
    assert [source["id"] for source in events[-1]["sources"]] == ["W1"]


async def test_no_sources_is_an_unverified_result_not_false_success(vectors, monkeypatch):
    mock_ai(monkeypatch)

    async def unavailable(_):
        return []

    monkeypatch.setattr("app.agents.nodes.search_web", unavailable)
    events = await run_graph(vectors)
    assert events[-1]["valid"] is False
    assert events[-1]["sources"] == []
    assert "do not provide evidence" in events[-1]["answer"]
    assert not any(event.get("node") == "plan_correction" for event in events)
