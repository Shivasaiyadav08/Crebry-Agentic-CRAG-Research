"""Durable research jobs. SSE clients observe jobs; disconnecting never cancels them.

Checkpoints are JSON snapshots at successful LangGraph node boundaries. A crash
can repeat the in-flight node; completed retrieval, evidence and citations survive.
"""
import asyncio
from contextlib import contextmanager
from copy import deepcopy
import json
import logging
from time import time
from uuid import uuid4

from sqlalchemy import delete, func, select

from app.agents.graph import next_node
from app.core.config import config
from app.models.database import Message, ResearchEvent, ResearchJob, WorkflowRun, as_dict
from app.services.ai import safe_service_error

logger = logging.getLogger(__name__)
TERMINAL = {"completed", "validation_failed", "failed", "cancelled", "interrupted"}


@contextmanager
def write_transaction(sessions):
    with sessions() as db:
        if db.bind.dialect.name == "sqlite":
            db.connection().exec_driver_sql("BEGIN IMMEDIATE")
        yield db
        db.commit()


class JobManager:
    def __init__(self, sessions, graph):
        self.sessions, self.graph = sessions, graph
        self.owner = uuid4().hex
        self.tasks = {}
        self.stopping = False
        self.poller = None

    async def start(self):
        self.poller = asyncio.create_task(self.poll(), name="research-job-poller")

    async def close(self):
        self.stopping = True
        if self.poller:
            self.poller.cancel()
        tasks = [*self.tasks.values(), *([self.poller] if self.poller else [])]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def claim(self):
        with write_transaction(self.sessions) as db:
            job = db.scalar(select(ResearchJob).where(
                ResearchJob.status.in_(["queued", "running"]), ResearchJob.lease_until < time(),
            ).order_by(ResearchJob.id).with_for_update(skip_locked=True).limit(1))
            if job is None:
                return None
            job.status, job.lease_owner, job.lease_until = "running", self.owner, time() + 30
            db.get(WorkflowRun, job.id).status = "running"
            return job.id

    async def poll(self):
        while True:
            self.tasks = {key: task for key, task in self.tasks.items() if not task.done()}
            while len(self.tasks) < config.max_research_jobs:
                run_id = await asyncio.to_thread(self.claim)
                if not run_id:
                    break
                self.tasks[run_id] = asyncio.create_task(self.execute(run_id), name=f"research-{run_id}")
            await asyncio.sleep(0.25)

    def renew(self, run_id):
        with write_transaction(self.sessions) as db:
            job = db.get(ResearchJob, run_id, with_for_update=True)
            if not job or job.lease_owner != self.owner:
                return False
            job.lease_until = time() + 30
            return not job.cancel_requested and job.deadline_at > time()

    async def heartbeat(self, run_id, task):
        while True:
            await asyncio.sleep(5)
            if not await asyncio.to_thread(self.renew, run_id):
                task.cancel()
                return

    def _append_events(self, db, run_id, payloads):
        if not payloads:
            return
        # Token batches do not need to load/rewrite the growing trace JSON.
        run = db.get(WorkflowRun, run_id) if any(p["type"] != "token" for p in payloads) else None
        sequence = db.scalar(select(func.max(ResearchEvent.sequence)).where(ResearchEvent.run_id == run_id)) or 0
        trace = []
        for payload in payloads:
            if payload["type"] == "final":
                existing = db.scalar(select(Message).where(Message.run_id == run_id, Message.role == "assistant"))
                if existing is None:
                    job = db.get(ResearchJob, run_id)
                    if job.replaced_ids:
                        db.execute(delete(Message).where(Message.id.in_(job.replaced_ids), Message.conversation_id == run.conversation_id))
                    existing = Message(conversation_id=run.conversation_id, role="assistant", content=payload["answer"], sources=payload["sources"], run_id=run_id)
                    db.add(existing)
                    db.flush()
                payload = {**payload, "message": as_dict(existing)}
            sequence += 1
            db.add(ResearchEvent(run_id=run_id, sequence=sequence, payload={**payload, "event_id": sequence}))
            if payload["type"] != "token":
                trace.append(payload)
        if trace:
            run.events = [*run.events, *trace]

    def append(self, run_id, payload):
        self.append_many(run_id, [payload])

    def append_many(self, run_id, payloads):
        with write_transaction(self.sessions) as db:
            job = db.get(ResearchJob, run_id, with_for_update=True)
            if job is None or job.lease_owner != self.owner:
                raise RuntimeError("Research job lease was lost.")
            self._append_events(db, run_id, payloads)

    def checkpoint(self, run_id, snapshot, following, payloads=()):
        # Events and the completed node snapshot commit together. JSON serialization
        # happens once in SQLAlchemy; deepcopy detaches mutable graph objects.
        with write_transaction(self.sessions) as db:
            job = db.get(ResearchJob, run_id, with_for_update=True)
            if job is None or job.lease_owner != self.owner:
                raise RuntimeError("Research job lease was lost.")
            self._append_events(db, run_id, payloads)
            job.snapshot = deepcopy(snapshot)
            job.next_node = following

    def finish(self, run_id, status, error, elapsed):
        with write_transaction(self.sessions) as db:
            job = db.get(ResearchJob, run_id, with_for_update=True)
            if job is None or job.lease_owner != self.owner:
                return
            job.status, job.lease_owner, job.lease_until = status, None, 0
            run = db.get(WorkflowRun, run_id)
            run.status, run.error, run.duration_ms = status, error, elapsed

    async def execute(self, run_id):
        started, status, error = time(), "failed", None
        heartbeat = asyncio.create_task(self.heartbeat(run_id, asyncio.current_task()))
        buffer, last_flush = "", time()
        pending = []
        first_token = True
        with self.sessions() as db:
            job = db.get(ResearchJob, run_id)
            snapshot = deepcopy(job.snapshot)
            snapshot["resume_node"] = job.next_node
            snapshot["model_plan"]["deadline_at"] = job.deadline_at
            run = db.get(WorkflowRun, run_id)
            conversation_id = run.conversation_id
            saved = db.scalar(select(Message).where(Message.run_id == run_id, Message.role == "assistant"))
        try:
            if saved is not None:
                # Final publication and its event were committed atomically before a crash.
                final = next((event for event in reversed(run.events) if event["type"] == "final"), {})
                status = "completed" if final.get("valid", True) else "validation_failed"
                return
            await asyncio.to_thread(self.append, run_id, {"type": "start", "conversation_id": conversation_id, "run_id": run_id})
            async with asyncio.timeout(max(0.001, job.deadline_at - time())):
                graph = self.graph() if callable(self.graph) else self.graph
                async for part in graph.astream(snapshot, stream_mode=["custom", "updates"], config={"recursion_limit": 30}, version="v2"):
                    if part.get("type", "custom") == "updates":
                        for node, update in part["data"].items():
                            if update is None:
                                continue
                            snapshot.update(update)
                            await asyncio.to_thread(self.checkpoint, run_id, snapshot, next_node(node, snapshot), pending)
                            pending = []
                        continue
                    payload = {**part["data"], "elapsed_ms": round((time() - started) * 1000)}
                    if payload["type"] == "token":
                        buffer += payload["text"]
                        if not first_token and len(buffer) < 100 and time() - last_flush < 0.2:
                            continue
                        payload = {**payload, "text": buffer}
                        buffer, last_flush = "", time()
                        first_token = False
                    elif buffer:
                        pending.append({"type": "token", "text": buffer})
                        buffer = ""
                    pending.append(payload)
                    if payload["type"] == "answer_reset":
                        first_token = True
                    # Show running steps and the first token promptly. Other events
                    # commit in order with the next node checkpoint.
                    if payload["type"] in {"node_start", "token", "answer_reset", "model_fallback", "final"}:
                        await asyncio.to_thread(self.append_many, run_id, pending)
                        pending = []
                    if payload["type"] == "final":
                        status = "completed" if payload["valid"] else "validation_failed"
            if status == "failed":
                raise RuntimeError("Research ended without a validated final result.")
        except asyncio.CancelledError:
            status = "queued" if self.stopping else "cancelled"
            error = "Backend stopped; research will resume from its last completed step." if self.stopping else "Research cancelled. Completed steps are saved."
        except Exception as caught:
            status = "failed"
            error = safe_service_error(caught)
            logger.error("Research failed run_id=%s error_type=%s reason=%s", run_id, type(caught).__name__, error)
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            elapsed = round((time() - started) * 1000)
            if pending or buffer:
                if buffer:
                    pending.append({"type": "token", "text": buffer})
                await asyncio.to_thread(self.append_many, run_id, pending)
            if error and not self.stopping:
                await asyncio.to_thread(self.append, run_id, {"type": "error", "message": error, "run_id": run_id})
            if status != "queued":
                await asyncio.to_thread(self.append, run_id, {"type": "done", "run_id": run_id, "status": status, "duration_ms": elapsed})
            await asyncio.to_thread(self.finish, run_id, status, error, elapsed)

    def events_after(self, run_id, cursor):
        with self.sessions() as db:
            rows = db.scalars(select(ResearchEvent).where(ResearchEvent.run_id == run_id, ResearchEvent.sequence > cursor).order_by(ResearchEvent.sequence).limit(100))
            events = [row.payload for row in rows]
            status = db.get(ResearchJob, run_id).status
            return events, status

    async def subscribe(self, run_id, cursor=0):
        last_heartbeat = time()
        while True:
            events, status = await asyncio.to_thread(self.events_after, run_id, cursor)
            for payload in events:
                cursor = payload["event_id"]
                yield f"id: {cursor}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if status in TERMINAL and len(events) < 100:
                if not events or events[-1].get("type") != "done":
                    yield f"data: {json.dumps({'type': 'done', 'run_id': run_id, 'status': status})}\n\n"
                return
            if time() - last_heartbeat >= 15:
                yield ": heartbeat\n\n"
                last_heartbeat = time()
            await asyncio.sleep(0.25)
