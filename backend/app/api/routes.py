import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
from time import time

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.orm import defer

from app.core.config import config
from app.core.auth import authorize_request, require_workspace_owner
from app.core.memory import check_and_collect_garbage, get_process_memory_mb
from app.models.database import (
    AppSettings,
    Conversation,
    Document,
    Message,
    ResearchJob,
    RoutingState,
    WorkflowRun,
    Workspace,
    as_dict,
    new_id,
)
from app.services import storage
from app.services.ai import safe_service_error
from app.services.free_models import credential_fingerprint, get_free_models
from app.services.documents import extract_text, index_document, upload_document
from app.services.jobs import write_transaction

router = APIRouter(prefix="/api", dependencies=[Depends(authorize_request)])


class WorkspaceInput(BaseModel):
    name: str = Field(min_length=1, max_length=100)

    @field_validator("name")
    @classmethod
    def trim_name(cls, value):
        if not value.strip():
            raise ValueError("Name must contain text.")
        return value.strip()


class ChatInput(BaseModel):
    workspace_id: str
    conversation_id: str | None = None
    question: str = Field(min_length=1, max_length=4000)
    regenerate: bool = False
    automatic_fallback: bool = True
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    model: str = Field(default="auto", min_length=1, max_length=200, pattern=r"^[a-zA-Z0-9_./:~\-]+$")
    relevance_threshold: float | None = Field(
        default=None, ge=0, le=1, allow_inf_nan=False,
        description="Minimum per-document-chunk relevance score. Omit to use the internal default.",
    )

    @field_validator("question")
    @classmethod
    def trim_question(cls, value):
        if not value.strip():
            raise ValueError("Question must contain text.")
        return value.strip()


def require_record(db, table, record_id):
    record = db.get(table, record_id)
    if record is None:
        raise HTTPException(404, "Record not found.")
    return record


def document_summary(document):
    """Serialize document metadata without loading or keeping the large text column in memory."""
    return {
        col.name: getattr(document, col.name)
        for col in document.__table__.columns
        if col.name not in ("text", "content_hash")
    }


def ensure_workspace_idle(request, workspace_id):
    if workspace_id in request.app.state.active_workspaces.values():
        raise HTTPException(409, "Wait for this workspace's active answer before changing its documents.")
    with request.app.state.db() as db:
        active = db.scalar(select(ResearchJob.id).join(WorkflowRun).join(Conversation).where(
            Conversation.workspace_id == workspace_id, ResearchJob.status.in_(["queued", "running"]),
        ).limit(1))
        if active:
            raise HTTPException(409, "Wait for this workspace's active research before changing its documents.")


@router.get("/health")
def health(request: Request):
    with request.app.state.db() as db:
        db.execute(select(1))
    payload = {
        "status": "ok",
        "gemini_configured": bool(config.gemini_api_key),
        "tavily_configured": bool(config.tavily_api_key),
        "embedding_model": config.embedding_model,
        "embedding_provider": "gemini" if getattr(request.app.state.vectors, "is_gemini_active", False) else "local",
        "embeddings_loaded": request.app.state.vectors.model is not None,
        "storage_provider": storage.provider(),
        "vector_provider": config.vector_provider,
        "ocr_enabled": config.gemini_ocr_enabled and bool(config.gemini_api_key),
        "max_upload_mb": storage.max_upload_mb(),
        "single_user": False,
        "authentication": "clerk",
        "clerk_configured": bool(config.clerk_secret_key or config.clerk_jwt_key),
    }
    if config.enable_memory_diagnostics:
        payload["memory_rss_mb"] = get_process_memory_mb()
        payload["embedding_batch_size"] = config.embedding_batch_size
    return payload


@router.get("/workspaces")
def list_workspaces(request: Request):
    with request.app.state.db() as db:
        rows = db.scalars(
            select(Workspace)
            .where(Workspace.owner_id == request.state.user_id)
            .order_by(Workspace.created_at)
        )
        result = []
        for row in rows:
            counts = db.execute(
                select(func.count(Document.id), func.coalesce(func.sum(Document.chunk_count), 0)).where(
                    Document.workspace_id == row.id
                )
            ).one()
            chats = db.scalar(select(func.count(Conversation.id)).where(Conversation.workspace_id == row.id))
            result.append(
                {
                    **as_dict(row),
                    "document_count": counts[0],
                    "chunk_count": counts[1],
                    "conversation_count": chats,
                }
            )
        return result


@router.post("/workspaces", status_code=201)
def create_workspace(data: WorkspaceInput, request: Request):
    with request.app.state.db() as db:
        workspace = Workspace(name=data.name, owner_id=request.state.user_id)
        db.add(workspace)
        db.commit()
        return as_dict(workspace)


@router.patch("/workspaces/{workspace_id}")
def rename_workspace(workspace_id: str, data: WorkspaceInput, request: Request):
    with request.app.state.db() as db:
        workspace = require_record(db, Workspace, workspace_id)
        workspace.name = data.name
        db.commit()
        return as_dict(workspace)


@router.delete("/workspaces/{workspace_id}", status_code=204)
def delete_workspace(workspace_id: str, request: Request):
    with request.app.state.mutation_lock, request.app.state.vectors.lock, request.app.state.db() as db:
        ensure_workspace_idle(request, workspace_id)
        workspace = require_record(db, Workspace, workspace_id)
        request.app.state.vectors.delete_workspace(workspace_id)
        storage.delete_workspace_files(workspace_id)
        db.delete(workspace)
        db.commit()


@router.get("/workspaces/{workspace_id}/documents")
def list_documents(workspace_id: str, request: Request, search: str = ""):
    with request.app.state.db() as db:
        require_record(db, Workspace, workspace_id)
        rows = db.scalars(
            select(Document)
            .options(defer(Document.text))
            .where(Document.workspace_id == workspace_id)
            .order_by(Document.created_at.desc())
        )
        return [document_summary(row) for row in rows if search.lower() in row.name.lower()]


@router.post("/workspaces/{workspace_id}/documents", status_code=201)
def upload_documents(workspace_id: str, request: Request, files: list[UploadFile]):
    if not 1 <= len(files) <= 10:
        raise HTTPException(422, "Upload between 1 and 10 files per request.")
    results = []
    sem = getattr(request.app.state, "embedding_semaphore", None)

    with request.app.state.mutation_lock, request.app.state.vectors.lock, request.app.state.db() as db:
        ensure_workspace_idle(request, workspace_id)
        require_record(db, Workspace, workspace_id)
        temp_dir = config.data_dir / ".tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        max_bytes = storage.max_upload_bytes()

        for file in files:
            temp_path = None
            try:
                temp_file = tempfile.NamedTemporaryFile(dir=temp_dir, delete=False)
                temp_path = Path(temp_file.name)
                hasher = hashlib.sha256()
                total_size = 0

                with temp_file:
                    while chunk := file.file.read(65536):
                        total_size += len(chunk)
                        if total_size > max_bytes:
                            raise ValueError(f"File exceeds the effective {storage.max_upload_mb():g} MB upload limit.")
                        hasher.update(chunk)
                        temp_file.write(chunk)

                if total_size == 0:
                    raise ValueError("Empty files cannot be indexed.")

                content_hash = hasher.hexdigest()

                if sem:
                    with sem:
                        document, duplicate = upload_document(
                            db,
                            request.app.state.vectors,
                            config.data_dir,
                            workspace_id,
                            file.filename or "file",
                            temp_path,
                            file_size=total_size,
                            content_hash=content_hash,
                        )
                else:
                    document, duplicate = upload_document(
                        db,
                        request.app.state.vectors,
                        config.data_dir,
                        workspace_id,
                        file.filename or "file",
                        temp_path,
                        file_size=total_size,
                        content_hash=content_hash,
                    )

                results.append({"document": document_summary(document), "duplicate": duplicate})
            except ValueError as error:
                results.append({"name": file.filename, "error": str(error)})
            finally:
                if temp_path and temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                file.file.close()

    check_and_collect_garbage(threshold_mb=config.gc_threshold_mb)
    return results


@router.get("/workspaces/{workspace_id}/documents/{document_id}")
def preview_document(workspace_id: str, document_id: str, request: Request):
    with request.app.state.db() as db:
        document = owned_document(db, workspace_id, document_id)
        return {
            **document_summary(document),
            "text": document.text[:100_000],
            "preview_truncated": len(document.text) > 100_000,
        }


def owned_document(db, workspace_id, document_id):
    document = require_record(db, Document, document_id)
    if document.workspace_id != workspace_id:
        raise HTTPException(404, "Document not found in this workspace.")
    return document


@router.get("/workspaces/{workspace_id}/documents/{document_id}/file")
def original_document(workspace_id: str, document_id: str, request: Request):
    with request.app.state.db() as db:
        document = owned_document(db, workspace_id, document_id)
        if storage.file_exists_locally(workspace_id, document_id):
            return FileResponse(storage.local_path(workspace_id, document_id), filename=document.name, media_type="application/octet-stream")
        try:
            content = storage.download_file(workspace_id, document_id)
            return Response(
                content=content,
                media_type="application/octet-stream",
                headers={"Content-Disposition": f'attachment; filename="{document.name}"'},
            )
        except Exception:
            raise HTTPException(404, "Original file is unavailable.")


@router.post("/workspaces/{workspace_id}/documents/{document_id}/reindex")
def reindex_document(workspace_id: str, document_id: str, request: Request):
    sem = getattr(request.app.state, "embedding_semaphore", None)
    with request.app.state.mutation_lock, request.app.state.vectors.lock, request.app.state.db() as db:
        ensure_workspace_idle(request, workspace_id)
        document = owned_document(db, workspace_id, document_id)
        source = None
        if storage.file_exists_locally(workspace_id, document_id):
            source = storage.local_path(workspace_id, document_id)
        else:
            try:
                source = storage.download_file(workspace_id, document_id)
            except Exception:
                pass
        if source is None:
            raise HTTPException(404, "Original file is unavailable; upload it again.")
        try:
            segments = extract_text(source, Path(document.name).suffix.lower())
            if sem:
                with sem:
                    index_document(db, request.app.state.vectors, document, segments)
            else:
                index_document(db, request.app.state.vectors, document, segments)
        except Exception as error:
            document.error = (
                "Reindexing failed. Any previous index is preserved. Check local model and disk access."
            )
            db.commit()
            raise HTTPException(503, document.error) from error
        finally:
            check_and_collect_garbage(threshold_mb=config.gc_threshold_mb)
        return document_summary(document)


@router.delete("/workspaces/{workspace_id}/documents/{document_id}", status_code=204)
def delete_document(workspace_id: str, document_id: str, request: Request):
    with request.app.state.mutation_lock, request.app.state.vectors.lock, request.app.state.db() as db:
        ensure_workspace_idle(request, workspace_id)
        document = owned_document(db, workspace_id, document_id)
        request.app.state.vectors.delete_document(workspace_id, document_id)
        storage.delete_file(workspace_id, document_id)
        db.delete(document)
        db.commit()


@router.get("/models")
async def list_models(request: Request, refresh: bool = False):
    try:
        models = await get_free_models(refresh=True) if refresh else await get_free_models()
        availability = await asyncio.to_thread(request.app.state.routing.availability_many,
                                               [item["id"] for item in models], "generate")
        return [{**item, "available": item.get("available", True) and availability[item["id"]]["available"],
                 "retry_after_seconds": max(item.get("retry_after_seconds", 0),
                                            availability[item["id"]]["retry_after_seconds"])} for item in models]
    except Exception as error:
        raise HTTPException(
            503,
            "Could not load free-tier Gemini models. " + safe_service_error(error),
        ) from error


@router.get("/workspaces/{workspace_id}/conversations")
def list_conversations(workspace_id: str, request: Request):
    with request.app.state.db() as db:
        require_record(db, Workspace, workspace_id)
        return [
            as_dict(row)
            for row in db.scalars(
                select(Conversation)
                .where(Conversation.workspace_id == workspace_id)
                .order_by(Conversation.created_at.desc())
            )
        ]


@router.get("/workspaces/{workspace_id}/conversations/{conversation_id}")
def get_conversation(workspace_id: str, conversation_id: str, request: Request):
    with request.app.state.db() as db:
        conversation = require_record(db, Conversation, conversation_id)
        if conversation.workspace_id != workspace_id:
            raise HTTPException(404, "Conversation not found in this workspace.")
        messages = db.scalars(
            select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at)
        )
        latest = db.scalar(select(WorkflowRun).where(WorkflowRun.conversation_id == conversation_id).order_by(WorkflowRun.created_at.desc()).limit(1))
        return {**as_dict(conversation), "messages": [as_dict(message) for message in messages],
                "latest_run": {"id": latest.id, "status": latest.status,
                               "resumable": db.get(ResearchJob, latest.id) is not None} if latest else None}


@router.delete("/workspaces/{workspace_id}/conversations/{conversation_id}", status_code=204)
def delete_conversation(workspace_id: str, conversation_id: str, request: Request):
    with request.app.state.mutation_lock, request.app.state.db() as db:
        if conversation_id in request.app.state.active_workspaces:
            raise HTTPException(409, "This conversation has an active answer.")
        if db.scalar(select(ResearchJob.id).join(WorkflowRun).where(
            WorkflowRun.conversation_id == conversation_id, ResearchJob.status.in_(["queued", "running"]),
        ).limit(1)):
            raise HTTPException(409, "This conversation has active research.")
        conversation = require_record(db, Conversation, conversation_id)
        if conversation.workspace_id != workspace_id:
            raise HTTPException(404, "Conversation not found in this workspace.")
        db.delete(conversation)
        db.commit()


@router.get("/workspaces/{workspace_id}/workflows")
def list_runs(workspace_id: str, request: Request):
    with request.app.state.db() as db:
        require_record(db, Workspace, workspace_id)
        rows = db.scalars(
            select(WorkflowRun)
            .join(Conversation)
            .where(Conversation.workspace_id == workspace_id)
            .order_by(WorkflowRun.created_at.desc())
            .limit(50)
        )
        return [as_dict(row) for row in rows]


@router.get("/workspaces/{workspace_id}/workflows/{run_id}")
def get_run(workspace_id: str, run_id: str, request: Request):
    with request.app.state.db() as db:
        row = require_record(db, WorkflowRun, run_id)
        conversation = require_record(db, Conversation, row.conversation_id)
        if conversation.workspace_id != workspace_id:
            raise HTTPException(404, "Workflow not found in this workspace.")
        return as_dict(row)


async def resolve_chat_models(data: ChatInput, request: Request):
    with request.app.state.db() as db:
        require_workspace_owner(db, data.workspace_id, request.state.user_id)
        if data.idempotency_key and db.scalar(select(ResearchJob.id).where(
            ResearchJob.owner_id == request.state.user_id, ResearchJob.request_key == data.idempotency_key,
        )):
            return {"models": ["auto"], "notice": None}
    if not config.gemini_api_key:
        raise HTTPException(503, "Add GEMINI_API_KEY to backend/.env and restart the backend.")
    try:
        catalog = await get_free_models()
    except Exception as error:
        raise HTTPException(503, "Could not verify free models. " + safe_service_error(error)) from error
    by_id = {item["id"] for item in catalog}
    available = [item for item in catalog if item.get("available", True)]
    if not available:
        retry_after = min(item.get("retry_after_seconds", 60) for item in catalog)
        raise HTTPException(
            503,
            "All eligible Gemini models are in cooldown after model-access or quota failures. "
            f"Retry in {max(1, retry_after)}s and refresh the model list.",
        )
    if not data.automatic_fallback and data.model != "auto":
        if data.model not in {item["id"] for item in available}:
            raise HTTPException(503, "The selected free-tier model is unavailable. Enable automatic fallback or select another model.")
        return {"models": [data.model], "notice": None, "catalog": catalog}
    notice = None
    if data.model not in by_id:
        if data.model != "auto" and not data.model.startswith("gemini-"):
            raise HTTPException(
                422,
                "Only Gemini models verified as free-tier eligible and accessible to this key can be selected. Refresh the model list.",
            )
        if data.model != "auto":
            notice = f"{data.model} is no longer in the verified free-tier Gemini catalog. Using {available[0]['id']}."
    available_ids = {item["id"] for item in available}
    if data.model in by_id and data.model not in available_ids:
        notice = f"{data.model} is in cooldown after a quota or service failure. Using {available[0]['id']}."
    models = ([data.model] if data.model in available_ids else []) + [
        item["id"] for item in available if item["id"] != data.model
    ]
    return {"models": models, "notice": notice, "catalog": catalog}


@router.post("/chat/stream")
async def chat_stream(data: ChatInput, request: Request, selection: dict = Depends(resolve_chat_models)):
    # This reservation prevents overlapping requests from corrupting a conversation.
    fingerprint = hashlib.sha256(json.dumps(data.model_dump(exclude={"idempotency_key"}), sort_keys=True).encode()).hexdigest()
    with request.app.state.mutation_lock, write_transaction(request.app.state.db) as db:
        db.execute(select(RoutingState.id).where(RoutingState.id == config.gemini_project_id).with_for_update())
        require_record(db, Workspace, data.workspace_id)
        if data.idempotency_key:
            previous = db.scalar(select(ResearchJob).where(ResearchJob.owner_id == request.state.user_id, ResearchJob.request_key == data.idempotency_key))
            if previous:
                if previous.fingerprint != fingerprint:
                    raise HTTPException(409, "This request key was already used for different input.")
                return StreamingResponse(request.app.state.jobs.subscribe(previous.id), media_type="text/event-stream", headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})
        if data.conversation_id:
            conversation = require_record(db, Conversation, data.conversation_id)
            if conversation.workspace_id != data.workspace_id:
                raise HTTPException(404, "Conversation not found in this workspace.")
            db.execute(select(Conversation.id).where(Conversation.id == conversation.id).with_for_update())
        else:
            if data.regenerate:
                raise HTTPException(422, "Regeneration requires an existing conversation.")
            conversation = Conversation(id=new_id(), workspace_id=data.workspace_id, title=data.question[:80])
            db.add(conversation)
            db.flush()
        if conversation.id in request.app.state.active_workspaces:
            raise HTTPException(409, "An answer is already running in this conversation.")
        active_job = db.scalar(select(ResearchJob.id).join(WorkflowRun).where(
            WorkflowRun.conversation_id == conversation.id, ResearchJob.status.in_(["queued", "running"]),
        ).limit(1))
        if active_job:
            raise HTTPException(409, "An answer is already running in this conversation.")
        rows = list(
            db.scalars(
                select(Message).where(Message.conversation_id == conversation.id).order_by(Message.created_at)
            )
        )
        replaced_ids = []
        if data.regenerate:
            last_user = next((message for message in reversed(rows) if message.role == "user"), None)
            if not last_user or last_user.content != data.question:
                raise HTTPException(422, "Only the most recent question can be regenerated.")
            history_rows = rows[: rows.index(last_user)]
            replaced_ids = [message.id for message in rows[rows.index(last_user) + 1 :]]
        else:
            history_rows = rows
            db.add(Message(conversation_id=conversation.id, role="user", content=data.question))
        # Bound history by both count and characters to avoid oversized prompts.
        history = []
        remaining = 12_000
        for row in reversed(history_rows[-10:]):
            if len(row.content) > remaining:
                break
            history.insert(0, {"role": row.role, "content": row.content})
            remaining -= len(row.content)
        settings = as_dict(db.get(AppSettings, 1))
        settings["model"] = selection["models"][0]
        if data.relevance_threshold is not None:
            settings["relevance_threshold"] = data.relevance_threshold
        settings["validate_answer"] = True
        run = WorkflowRun(id=new_id(), conversation_id=conversation.id)
        db.add(run)
        db.flush()
        conversation_id, run_id = conversation.id, run.id
        snapshot = {
            "question": data.question, "workspace_id": data.workspace_id,
            "history": history, "settings": settings, "sources": [], "warnings": [], "attempts": 0,
            "model_plan": {
                "models": selection["models"], "unavailable": [], "notice": selection["notice"],
                "catalog": selection.get("catalog"), "conversation_id": conversation_id,
                "credential_fingerprint": credential_fingerprint(), "catalog_verified_at": time(),
                "selected": data.model, "automatic": data.automatic_fallback,
                "deadline_at": time() + config.research_timeout_seconds,
            },
        }
        db.add(ResearchJob(
            id=run_id, owner_id=request.state.user_id, request_key=data.idempotency_key or new_id(),
            fingerprint=fingerprint, snapshot=snapshot, replaced_ids=replaced_ids,
            deadline_at=time() + config.research_timeout_seconds,
        ))

    return StreamingResponse(
        request.app.state.jobs.subscribe(run_id), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@router.get("/workspaces/{workspace_id}/workflows/{run_id}/events")
async def resume_events(workspace_id: str, run_id: str, request: Request, after: int = 0):
    get_run(workspace_id, run_id, request)
    with request.app.state.db() as db:
        if db.get(ResearchJob, run_id) is None:
            raise HTTPException(409, "This legacy run has no resumable checkpoint.")
    return StreamingResponse(request.app.state.jobs.subscribe(run_id, max(0, after)), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})


@router.post("/workspaces/{workspace_id}/workflows/{run_id}/cancel")
def cancel_job(workspace_id: str, run_id: str, request: Request):
    get_run(workspace_id, run_id, request)
    with write_transaction(request.app.state.db) as db:
        job = require_record(db, ResearchJob, run_id)
        job.cancel_requested = True
        if job.status == "queued":
            job.status = "cancelled"
            db.get(WorkflowRun, run_id).status = "cancelled"
    return {"status": "cancellation_requested"}


@router.post("/workspaces/{workspace_id}/workflows/{run_id}/retry")
def retry_job(workspace_id: str, run_id: str, request: Request):
    get_run(workspace_id, run_id, request)
    with write_transaction(request.app.state.db) as db:
        job = require_record(db, ResearchJob, run_id)
        if job.status not in {"failed", "cancelled", "interrupted"}:
            raise HTTPException(409, "Only interrupted research can be retried.")
        run = db.get(WorkflowRun, run_id)
        if db.scalar(select(ResearchJob.id).join(WorkflowRun).where(
            WorkflowRun.conversation_id == run.conversation_id,
            ResearchJob.id != run_id, ResearchJob.status.in_(["queued", "running"]),
        ).limit(1)):
            raise HTTPException(409, "Another answer is already running in this conversation.")
        job.status, job.cancel_requested, job.lease_owner, job.lease_until = "queued", False, None, 0
        job.deadline_at = time() + config.research_timeout_seconds
        run.status, run.error = "running", None
    return {"run_id": run_id, "status": "queued"}
