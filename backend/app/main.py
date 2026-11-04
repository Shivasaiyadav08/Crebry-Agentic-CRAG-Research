from contextlib import asynccontextmanager
import logging
from threading import BoundedSemaphore, RLock

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select

from app.agents.graph import build_graph
from app.api.routes import router
from app.core.config import config
from app.core.memory import configure_process_memory_environment
from app.models.database import AppSettings, ResearchJob, WorkflowRun, create_database
from app.services.vector import VectorStore
from app.services.routing import RoutingStore
from app.services.jobs import JobManager
from app.core.readiness import check_production_configuration, prepare_supabase_scratch_directory
from app.services.documents import configure_ocr_router

configure_process_memory_environment()


@asynccontextmanager
async def lifespan(app):
    check_production_configuration()
    prepare_supabase_scratch_directory()
    use_supabase = config.vector_provider == "supabase"
    engine, db_factory = create_database(config.data_dir, config.database_url or None, schema="crebry" if use_supabase else None)
    app.state.db = db_factory
    if use_supabase:
        from app.services.postgres_vector import PostgresVectorStore
        app.state.vectors = PostgresVectorStore(engine, config.data_dir, config.embedding_model)
    else:
        app.state.vectors = VectorStore(config.data_dir, config.embedding_model)
    if config.storage_provider == "supabase":
        from app.services.supabase_storage import check_bucket
        check_bucket()
    app.state.routing = RoutingStore(db_factory, config.gemini_project_id)
    configure_ocr_router(app.state.routing)
    app.state.graph = build_graph(app.state.vectors, app.state.routing)
    app.state.mutation_lock = RLock()
    app.state.embedding_semaphore = BoundedSemaphore(config.max_concurrent_embeddings)
    app.state.active_workspaces = {}
    with db_factory() as db:
        if db.get(AppSettings, 1) is None:
            db.add(AppSettings(id=1, model=config.gemini_model))
        # A process restart cannot resume a provider stream; mark interrupted runs honestly.
        for run in db.scalars(select(WorkflowRun).where(WorkflowRun.status == "running")):
            if db.get(ResearchJob, run.id) is not None:
                continue
            run.status = "interrupted"
            run.error = "Backend restarted before this answer completed. Regenerate the question."
        db.commit()
    app.state.jobs = JobManager(db_factory, lambda: app.state.graph)
    await app.state.jobs.start()
    try:
        yield
    finally:
        await app.state.jobs.close()
        configure_ocr_router(None)
        engine.dispose()


app = FastAPI(title="Aether Research", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list({config.frontend_origin, config.frontend_origin.replace("localhost", "127.0.0.1")}),
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)
app.include_router(router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.middleware("http")
async def protect_local_writes(request: Request, call_next):
    """Reject writes from unrelated browser origins; this is not user authentication."""
    origin = request.headers.get("origin")
    allowed = {config.frontend_origin, config.frontend_origin.replace("localhost", "127.0.0.1")}
    if request.method in {"POST", "PUT", "PATCH", "DELETE"} and origin and origin not in allowed:
        return JSONResponse(
            status_code=403, content={"detail": "This browser origin is not allowed to change local data."}
        )
    return await call_next(request)


@app.exception_handler(Exception)
async def unexpected_error(request: Request, error: Exception):
    # Return a safe message; no keys, provider bodies, or server paths are exposed.
    logging.getLogger("aether").error(
        "Unhandled %s during %s %s", type(error).__name__, request.method, request.url.path
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred. Check the backend terminal."},
    )
