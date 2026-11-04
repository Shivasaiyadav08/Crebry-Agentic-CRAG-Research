"""SQLAlchemy records and SQLite/PostgreSQL initialization."""

from datetime import datetime, timezone

from uuid import uuid4

from sqlalchemy import JSON, ForeignKey, String, UniqueConstraint, create_engine, event

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def new_id() -> str:
    return uuid4().hex


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Base(DeclarativeBase):
    pass


class RoutingState(Base):
    __tablename__ = "routing_state"
    id: Mapped[str] = mapped_column(String(200), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)


class RequestBudget(Base):
    __tablename__ = "request_budgets"
    id: Mapped[str] = mapped_column(String(200), primary_key=True)
    window: Mapped[int]
    count: Mapped[int]


class ConversationModelPreference(Base):
    __tablename__ = "conversation_model_preferences"
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), primary_key=True
    )
    task: Mapped[str] = mapped_column(String(100), primary_key=True)
    model: Mapped[str] = mapped_column(String(200))


class Workspace(Base):
    __tablename__ = "workspaces"  # Actual table name in the database.

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)

    name: Mapped[str] = mapped_column(String(100))

    owner_id: Mapped[str | None] = mapped_column(String(200), index=True)

    created_at: Mapped[str] = mapped_column(default=now)


class Document(Base):
    __tablename__ = "documents"

    __table_args__ = (UniqueConstraint("workspace_id", "content_hash"),)

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)

    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)

    name: Mapped[str]

    content_hash: Mapped[str]

    size: Mapped[int]

    status: Mapped[str] = mapped_column(default="indexing")

    chunk_count: Mapped[int] = mapped_column(default=0)

    text: Mapped[str] = mapped_column(default="")

    error: Mapped[str | None]

    index_version: Mapped[str | None]

    created_at: Mapped[str] = mapped_column(default=now)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)

    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)

    title: Mapped[str]

    created_at: Mapped[str] = mapped_column(default=now)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)

    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )

    role: Mapped[str]

    content: Mapped[str]

    sources: Mapped[list] = mapped_column(JSON, default=list)

    run_id: Mapped[str | None]

    created_at: Mapped[str] = mapped_column(default=now)


class WorkflowRun(Base):
    __tablename__ = "workflow_runs"

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)

    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )

    status: Mapped[str] = mapped_column(default="running")

    events: Mapped[list] = mapped_column(JSON, default=list)

    duration_ms: Mapped[int] = mapped_column(default=0)

    error: Mapped[str | None]

    created_at: Mapped[str] = mapped_column(default=now)


class ResearchJob(Base):
    __tablename__ = "research_jobs"
    __table_args__ = (UniqueConstraint("owner_id", "request_key"),)
    id: Mapped[str] = mapped_column(ForeignKey("workflow_runs.id", ondelete="CASCADE"), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(200), index=True)
    request_key: Mapped[str] = mapped_column(String(100))
    fingerprint: Mapped[str]
    snapshot: Mapped[dict] = mapped_column(JSON)
    next_node: Mapped[str] = mapped_column(default="retrieve")
    replaced_ids: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(default="queued", index=True)
    lease_owner: Mapped[str | None]
    lease_until: Mapped[float] = mapped_column(default=0)
    deadline_at: Mapped[float]
    cancel_requested: Mapped[bool] = mapped_column(default=False)


class ResearchEvent(Base):
    __tablename__ = "research_events"
    run_id: Mapped[str] = mapped_column(ForeignKey("workflow_runs.id", ondelete="CASCADE"), primary_key=True)
    sequence: Mapped[int] = mapped_column(primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)


class AppSettings(Base):
    __tablename__ = "settings"

    id: Mapped[int] = mapped_column(primary_key=True, default=1)

    model: Mapped[str]

    relevance_threshold: Mapped[float] = mapped_column(default=0.5)

    validate_answer: Mapped[bool] = mapped_column(default=True)

    max_retries: Mapped[int] = mapped_column(default=1)

    retrieval_count: Mapped[int] = mapped_column(default=6)


def create_database(data_dir, database_url=None, *, schema=None):
    """Initialize SQLite or PostgreSQL and return an engine and session factory."""

    data_dir.mkdir(parents=True, exist_ok=True)

    url = database_url or f"sqlite:///{(data_dir / 'aether.db').as_posix()}"
    if url.startswith(("postgres://", "postgresql://")):
        url = "postgresql+psycopg://" + url.split("://", 1)[1]
    is_sqlite = url.startswith("sqlite:")
    connect_args = (
        {"check_same_thread": False}
        if is_sqlite
        else {
            "connect_timeout": 10,
            "prepare_threshold": None,
        }
    )
    if schema is not None:
        if is_sqlite or schema != "crebry":
            raise ValueError("The private Crebry schema requires PostgreSQL.")
        connect_args.setdefault("sslmode", "require")
    engine = create_engine(
        url,
        connect_args=connect_args,
        pool_pre_ping=True,
        **({"pool_size": 3, "max_overflow": 2, "pool_recycle": 300} if not is_sqlite else {}),
        **({"execution_options": {"schema_translate_map": {None: schema}}} if schema is not None else {}),
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _):

        if not is_sqlite:
            if schema is not None:
                with connection.cursor() as cursor:
                    cursor.execute("SET search_path TO crebry, extensions, public")
                connection.commit()
            return

        connection.execute("PRAGMA foreign_keys=ON")

        connection.execute("PRAGMA journal_mode=WAL")

        connection.execute("PRAGMA busy_timeout=10000")

    with engine.begin() as connection:
        if schema is not None and connection.exec_driver_sql("SELECT current_schema()").scalar() != schema:
            raise RuntimeError(
                "Run backend/supabase/setup.sql in Supabase SQL Editor before starting the backend."
            )

        columns = connection.exec_driver_sql("PRAGMA table_info(workspaces)").fetchall() if is_sqlite else []

        # Legacy workspaces stay unowned until explicitly claimed.
        if columns and "owner_id" not in {row[1] for row in columns}:
            connection.exec_driver_sql("ALTER TABLE workspaces ADD COLUMN owner_id VARCHAR(200)")

            connection.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_workspaces_owner_id ON workspaces(owner_id)"
            )

    Base.metadata.create_all(engine)
    if schema is not None:
        # Keep browser roles out of the private schema even if exposed by mistake.
        with engine.begin() as connection:
            statements = []
            for table in Base.metadata.sorted_tables:
                quoted = engine.dialect.identifier_preparer.quote_identifier(table.name)
                statements.extend(
                    [
                        f'ALTER TABLE "crebry".{quoted} ENABLE ROW LEVEL SECURITY',
                        f'REVOKE ALL ON TABLE "crebry".{quoted} FROM PUBLIC, anon, authenticated',
                    ]
                )
            connection.exec_driver_sql("; ".join(statements), execution_options={"no_parameters": True})

    return engine, sessionmaker(engine, expire_on_commit=False)


def as_dict(record) -> dict:

    return {column.name: getattr(record, column.name) for column in record.__table__.columns}
