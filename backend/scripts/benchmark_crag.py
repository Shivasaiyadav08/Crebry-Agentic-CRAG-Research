"""Offline, comparable graph benchmark. Never reads documents or calls providers.

Run from backend: python scripts/benchmark_crag.py --output ../docs/crag-before.json
Provider delays and token estimates are simulated; graph timings are measured locally.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
import sys
from time import perf_counter
from unittest.mock import patch

sys.path.insert(0, os.environ.get("CREBRY_BENCH_BACKEND", str(Path(__file__).resolve().parents[1])))

from langchain_core.messages import AIMessageChunk

from app.agents.graph import build_graph
from app.core.config import config
from app.services.ai import UNTRUSTED_RULE


async def sample(scenario, durable=False):
    calls, input_chars, output_chars = [], 0, 0
    drafts = 0
    fallback_failed = False
    score = 0.1 if scenario == "incorrect" else 0.9
    sufficient = scenario not in {"ambiguous", "incorrect"}

    class Vectors:
        def retrieve(self, *_):
            if scenario == "empty":
                return []
            return [
                {
                    "id": "D1",
                    "kind": "document",
                    "name": "synthetic.txt",
                    "page": 1,
                    "text": "Astronomy studies stars."
                    if scenario == "atomic"
                    else "Astronomy studies stars. Telescopes gather starlight.",
                    "document_id": "synthetic",
                    "version": "v1",
                    "chunk": 0,
                }
            ]

    async def structured(model, schema, instruction, data):
        nonlocal input_chars, output_chars
        calls.append(schema.__name__)
        import app.agents.nodes as nodes

        separators = (",", ":") if hasattr(nodes, "schema_json") else None
        input_chars += (
            len(UNTRUSTED_RULE)
            + len(instruction)
            + len(json.dumps(data, separators=separators))
            + len(json.dumps(schema.model_json_schema(), separators=separators))
        )
        await asyncio.sleep(0.25)
        grades = [
            {"id": c["id"], "score": score, "reason": "Synthetic relevance"} for c in data.get("chunks", [])
        ]
        assessment = {
            "sufficient": sufficient,
            "reason": "Synthetic coverage",
            "missing_information": "" if sufficient else "More evidence",
        }
        values = {
            "RelevanceBatch": {"grades": grades},
            "EvidenceEvaluation": {
                "grades": grades,
                "assessment": assessment,
                "supporting_ids": [g["id"] for g in grades if g["score"] >= 0.5],
                "rewritten_query": "astronomy evidence" if not sufficient else "",
            },
            "EvidenceAssessment": assessment,
            "EvidenceSelection": {"keep_ids": [s["id"] for s in data.get("sentences", [])]},
            "RewrittenQuery": {"query": "astronomy evidence"},
            "AnswerValidation": {"grounded": True, "reason": "Synthetic grounded answer"},
            "CorrectionPlan": {
                "action": "retrieve",
                "reason": "Find precise evidence",
                "query": "astronomy stars",
            },
        }
        result = schema(**values[schema.__name__])
        output_chars += len(result.model_dump_json())
        return result

    class Model:
        async def astream(self, messages):
            nonlocal input_chars, output_chars, drafts, fallback_failed
            calls.append("generate")
            drafts += 1
            input_chars += sum(len(str(m.content)) for m in messages)
            if scenario == "fallback" and not fallback_failed:
                fallback_failed = True
                await asyncio.sleep(0.01)
                error = RuntimeError("Synthetic quota exhaustion")
                error.code = 429
                raise error
            evidence = json.loads(messages[-1].content)["evidence"]
            answer = f"Astronomy studies stars [{evidence[0]['id']}]."
            if scenario == "recovery" and drafts == 1:
                answer = "Invalid citation [D99]."
            for part in (answer[:12], answer[12:]):
                await asyncio.sleep(0.05)
                output_chars += len(part)
                yield AIMessageChunk(content=part)

    async def search(_):
        await asyncio.sleep(0.01)
        return (
            []
            if scenario == "empty"
            else [
                {
                    "id": "W1",
                    "kind": "web",
                    "name": "Synthetic web",
                    "url": "https://example.org/astronomy",
                    "text": "Astronomy studies stars. Observatories use telescopes.",
                }
            ]
        )

    state = {
        "question": "What is astronomy?",
        "workspace_id": "synthetic",
        "settings": {
            "model": "test/model",
            "relevance_threshold": 0.5,
            "validate_answer": True,
            "max_retries": 1,
            "retrieval_count": 6,
        },
        "history": [],
        "sources": [],
        "warnings": [],
        "attempts": 0,
    }
    if scenario == "fallback":
        state["model_plan"] = {"models": ["synthetic-a", "synthetic-b"], "unavailable": []}
    elif durable:
        state["model_plan"] = {"models": ["synthetic-a"], "unavailable": []}
    events, first_token = [], None
    commits = 0
    start = perf_counter()
    with (
        patch("app.agents.nodes.structured_call", structured),
        patch("app.agents.nodes.create_llm", lambda _: Model()),
        patch("app.agents.nodes.search_web", search),
        patch.object(config, "tavily_api_key", "synthetic-only"),
    ):
        if not durable:
            async for part in build_graph(Vectors()).astream(state, stream_mode="custom", version="v2"):
                event = part["data"]
                events.append(event)
                if event["type"] == "token" and first_token is None:
                    first_token = (perf_counter() - start) * 1000
        else:
            from sqlalchemy import event
            from time import time
            from app.models.database import Conversation, ResearchJob, WorkflowRun, Workspace, create_database
            from app.services.jobs import JobManager

            with TemporaryDirectory(prefix="crebry-benchmark-") as directory:
                engine, sessions = create_database(Path(directory))
                manager = JobManager(sessions, build_graph(Vectors()))
                with sessions() as db:
                    db.add(Workspace(id="synthetic", name="Synthetic", owner_id="synthetic"))
                    db.flush()
                    db.add(Conversation(id="synthetic", workspace_id="synthetic", title="Synthetic"))
                    db.flush()
                    db.add(WorkflowRun(id="synthetic", conversation_id="synthetic"))
                    db.flush()
                    db.add(
                        ResearchJob(
                            id="synthetic",
                            owner_id="synthetic",
                            request_key="synthetic",
                            fingerprint="synthetic",
                            snapshot=state,
                            deadline_at=time() + 30,
                            lease_owner=manager.owner,
                            lease_until=time() + 30,
                        )
                    )
                    db.commit()

                def committed(_session):
                    nonlocal commits
                    commits += 1

                def observe(method, batch=False):
                    def append(run_id, payload):
                        nonlocal first_token
                        method(run_id, payload)
                        payloads = payload if batch else [payload]
                        if first_token is None and any(p["type"] == "token" for p in payloads):
                            first_token = (perf_counter() - start) * 1000

                    return append

                manager.append = observe(manager.append)
                if hasattr(manager, "append_many"):
                    manager.append_many = observe(manager.append_many, batch=True)
                event.listen(sessions, "after_commit", committed)
                start = perf_counter()
                await manager.execute("synthetic")
                events, _status = manager.events_after("synthetic", 0)
                total = (perf_counter() - start) * 1000
                event.remove(sessions, "after_commit", committed)
                engine.dispose()
    timings = {}
    for e in events:
        if e["type"] == "node_end":
            timings[e["node"]] = timings.get(e["node"], 0) + e["duration_ms"]
    final = next(e for e in events if e["type"] == "final")
    assert final["valid"] == (scenario != "empty")
    return {
        "total_ms": total if durable else (perf_counter() - start) * 1000,
        "first_token_ms": first_token,
        "llm_requests": len(calls),
        "input_token_estimate": (input_chars + 3) // 4,
        "output_token_estimate": (output_chars + 3) // 4,
        "node_ms": timings,
        "calls": calls,
        "valid": final["valid"],
        "sqlite_commits": commits,
        "fallbacks": sum(e["type"] == "model_fallback" for e in events),
        "retry_fallback_overhead_ms": 10 if scenario == "fallback" else 0,
    }


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--durable", action="store_true", help="Include actual local SQLite job persistence")
    args = parser.parse_args()
    result = {
        "mode": "mocked_providers_real_sqlite" if args.durable else "offline_mocked",
        "runs": args.runs,
        "assumptions": "250ms per structured request; two 50ms generated chunks; 10ms web; chars/4 token estimates, not provider usage; no external database/network latency",
        "scenarios": {},
    }
    for scenario in ("correct", "ambiguous", "incorrect", "empty", "recovery", "atomic", "fallback"):
        await sample(scenario, args.durable)  # Warm graph/import caches.
        samples = [await sample(scenario, args.durable) for _ in range(args.runs)]
        summary = {
            key: round(median(s[key] for s in samples), 2)
            for key in (
                "total_ms",
                "first_token_ms",
                "llm_requests",
                "input_token_estimate",
                "output_token_estimate",
                "sqlite_commits",
            )
        }
        summary["node_ms"] = {
            node: median(s["node_ms"].get(node, 0) for s in samples) for node in samples[0]["node_ms"]
        }
        summary.update(
            calls=samples[0]["calls"],
            valid=samples[0]["valid"],
            fallbacks=samples[0]["fallbacks"],
            simulated_retry_fallback_overhead_ms=samples[0]["retry_fallback_overhead_ms"],
        )
        result["scenarios"][scenario] = summary
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
