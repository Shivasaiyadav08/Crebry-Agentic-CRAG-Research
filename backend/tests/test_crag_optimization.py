"""Safety boundaries around call reduction and legacy checkpoint compatibility."""

from copy import deepcopy

import pytest

from app.agents.context import evidence_payload, recent_history, unique_sources
from app.agents.nodes import make_nodes
from app.agents.state import EvidenceAssessment, EvidenceEvaluation
from app.core.config import config
from test_graph import SETTINGS


def document(identifier="D1", score=None):
    source = {
        "id": identifier,
        "kind": "document",
        "name": "research.txt",
        "text": "Astronomy studies stars.",
        "document_id": identifier,
        "page": 2,
        "version": "v1",
        "chunk": 0,
    }
    if score is not None:
        source["relevance"] = score
    return source


@pytest.fixture
def state():
    return {
        "question": "What is astronomy?",
        "settings": SETTINGS,
        "sources": [],
        "retrieved": [document()],
        "warnings": [],
    }


@pytest.fixture
def nodes(monkeypatch):
    monkeypatch.setattr("app.agents.nodes.emit", lambda *_args, **_kwargs: None)
    return make_nodes(None)


@pytest.mark.parametrize(
    "score,sufficient,verdict",
    [(0.9, True, "CORRECT"), (0.9, False, "AMBIGUOUS"), (0.55, True, "AMBIGUOUS"), (0.1, True, "INCORRECT")],
)
async def test_one_call_preserves_three_way_coverage(nodes, state, monkeypatch, score, sufficient, verdict):
    calls = []

    async def evaluate(_model, schema, _instruction, data):
        calls.append(schema)
        return schema(
            grades=[{"id": "D1", "score": score, "reason": "Relevance"}],
            assessment={
                "sufficient": sufficient and score >= 0.5,
                "reason": "Coverage",
                "missing_information": "",
            },
            supporting_ids=["D1"] if score >= 0.5 else [],
        )

    monkeypatch.setattr("app.agents.nodes.structured_call", evaluate)
    state.update(await nodes["grade"](state))
    result = await nodes["assess"](state)
    assert result["crag_verdict"] == verdict
    assert calls == [EvidenceEvaluation]


async def test_rejected_chunk_cannot_establish_coverage(nodes, state, monkeypatch):
    state["retrieved"].append(document("D2"))
    calls = []

    async def evaluate(_model, schema, _instruction, data):
        calls.append(schema)
        if schema is EvidenceEvaluation:
            return schema(
                grades=[
                    {"id": "D1", "score": 0.9, "reason": "Partial"},
                    {"id": "D2", "score": 0.1, "reason": "Unrelated"},
                ],
                assessment={"sufficient": True, "reason": "False coverage", "missing_information": ""},
                supporting_ids=["D2"],
            )
        assert [s["id"] for s in data["evidence"]] == ["D1"]
        return schema(sufficient=False, reason="Only partial evidence", missing_information="More evidence")

    monkeypatch.setattr("app.agents.nodes.structured_call", evaluate)
    state.update(await nodes["grade"](state))
    result = await nodes["assess"](state)
    assert result["crag_verdict"] == "AMBIGUOUS"
    assert calls == [EvidenceEvaluation, EvidenceAssessment]


async def test_old_checkpoint_without_combined_evaluation_still_assesses(nodes, state, monkeypatch):
    state["sources"] = [document(score=0.9)]

    async def assess(_model, schema, _instruction, data):
        assert schema is EvidenceAssessment
        return schema(sufficient=True, reason="Legacy coverage checked", missing_information="")

    monkeypatch.setattr("app.agents.nodes.structured_call", assess)
    assert (await nodes["assess"](state))["crag_verdict"] == "CORRECT"


async def test_empty_evidence_skips_all_evaluation_calls(nodes, state, monkeypatch):
    state["retrieved"] = []

    async def unexpected(*_):
        pytest.fail("No provider call is needed without evidence")

    monkeypatch.setattr("app.agents.nodes.structured_call", unexpected)
    state.update(await nodes["grade"](state))
    assert (await nodes["assess"](state))["crag_verdict"] == "INCORRECT"


async def test_one_sentence_per_source_skips_refinement_without_dropping_sources(nodes, state, monkeypatch):
    state["sources"] = [document(), document("D2")]

    async def unexpected(*_):
        pytest.fail("Already atomic evidence needs no sentence selection call")

    monkeypatch.setattr("app.agents.nodes.structured_call", unexpected)
    assert (await nodes["refine"](state))["refined_sources"] == state["sources"]


async def test_second_search_does_not_reuse_web_ids(nodes, state, monkeypatch):
    state.update(web_attempts=1, rewritten_query="astronomy", sources=[document()])

    async def unexpected(*_):
        pytest.fail("Search budget must apply to the assess route as well as the planner")

    monkeypatch.setattr("app.agents.nodes.search_web", unexpected)
    result = await nodes["web_search"](state)
    assert "sources" not in result
    assert "budget exhausted" in result["web_search_error"]


def test_prompt_payload_keeps_text_and_provenance_without_bookkeeping():
    source = document(score=0.9)
    source["reason"] = "Large grading explanation"
    compact = evidence_payload([source])[0]
    assert compact == {key: source[key] for key in ("id", "kind", "name", "page", "text")}
    assert source["relevance"] == 0.9


def test_deduplication_preserves_distinct_original_attribution():
    first, second = document(), document("D2")
    assert unique_sources([first, deepcopy(first), second]) == [first, second]


def test_history_budget_keeps_recent_whole_messages():
    history = [{"role": "user", "content": "x" * 6000}, {"role": "user", "content": "Recent question"}]
    assert recent_history({"history": history}) == history[-1:]


def test_large_assistant_message_does_not_hide_the_previous_user_question():
    question = {"role": "user", "content": "Explain astronomy"}
    history = [question, {"role": "assistant", "content": "x" * 9000}]
    assert recent_history({"history": history}) == [question]


async def test_duplicate_corrective_retrieval_becomes_regeneration(nodes, state, monkeypatch):
    state.update(last_retrieval_query="Astronomy", attempts=1)

    async def plan(_model, schema, _instruction, _data):
        return schema(action="retrieve", reason="Try again", query="astronomy")

    monkeypatch.setattr("app.agents.nodes.structured_call", plan)
    monkeypatch.setattr(config, "tavily_api_key", "")
    assert (await nodes["plan_correction"](state))["correction_action"] == "regenerate"


async def test_combined_query_rewrite_keeps_the_rewrite_node_without_extra_call(nodes, state, monkeypatch):
    monkeypatch.setattr(config, "tavily_api_key", "synthetic")
    state.update(sources=[document()], evaluated_assessment={"rewritten_query": "focused astronomy gap"})

    async def unexpected(*_):
        pytest.fail("The combined evaluator already rewrote the query")

    monkeypatch.setattr("app.agents.nodes.structured_call", unexpected)
    assert (await nodes["rewrite"](state))["rewritten_query"] == "focused astronomy gap"
