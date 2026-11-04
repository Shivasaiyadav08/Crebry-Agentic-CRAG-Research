"""Supabase pgvector adapter with the same evidence contract as Chroma."""
import math
from threading import RLock

from sqlalchemy import text

from app.core.config import config
from app.services.vector import VectorStore


def vector_literal(values):
    values = [float(value) for value in values]
    if len(values) != 768 or not all(math.isfinite(value) for value in values):
        raise ValueError("Supabase vectors require 768 finite Gemini embedding dimensions.")
    if not any(values):
        raise ValueError("A zero embedding cannot be used for cosine retrieval.")
    return "[" + ",".join(str(value) for value in values) + "]"


class PostgresVectorStore(VectorStore):
    def __init__(self, engine, data_dir, model_name, embedder=None):
        # Reuse embedding methods without constructing a local Chroma client.
        self.engine = engine
        self.model_name = model_name
        self.model_cache = str(data_dir / "models")
        self.model = embedder
        self.lock = RLock()
        with engine.connect() as connection:
            if not connection.execute(text("SELECT to_regclass('crebry.document_chunks')")).scalar():
                raise RuntimeError("Run backend/supabase/setup.sql in Supabase SQL Editor to create the vector table.")

    def collection(self, workspace_id):
        raise NotImplementedError("Supabase uses workspace-scoped SQL queries, not Chroma collections.")

    def index(self, workspace_id, document_id, name, chunks, version):
        statement = text("""
            INSERT INTO crebry.document_chunks
                (workspace_id, document_id, version, chunk, name, page, content, embedding)
            VALUES (:workspace, :document, :version, :chunk, :name, :page, :content, CAST(:embedding AS vector))
            ON CONFLICT (workspace_id, document_id, version, chunk) DO NOTHING
        """)
        with self.lock:
            for offset in range(0, len(chunks), config.embedding_batch_size):
                batch = chunks[offset:offset + config.embedding_batch_size]
                embeddings = self.embed([item["text"] for item in batch])
                if len(embeddings) != len(batch):
                    raise ValueError("The embedding provider returned an incomplete batch.")
                rows = [
                    {"workspace": workspace_id, "document": document_id, "version": version,
                     "chunk": offset + i, "name": name, "page": item["page"], "content": item["text"],
                     "embedding": vector_literal(embedding)}
                    for i, (item, embedding) in enumerate(zip(batch, embeddings))
                ]
                with self.engine.begin() as connection:
                    connection.execute(statement, rows)

    def retrieve(self, workspace_id, question, count=6):
        # Exact cosine search keeps recall on a small free database. Only published
        # document versions are visible, including during a staged reindex/crash.
        statement = text("""
            SELECT c.document_id, c.name, c.page, c.version, c.chunk, c.content,
                   c.embedding <=> CAST(:embedding AS vector) AS distance
            FROM crebry.document_chunks c
            JOIN crebry.documents d ON d.id = c.document_id AND d.workspace_id = c.workspace_id
                                  AND d.index_version = c.version
            WHERE c.workspace_id = :workspace
            ORDER BY distance, c.document_id, c.chunk
            LIMIT :count
        """)
        # Read-only retrieval need not wait behind another workspace's indexing
        # or network embedding call. The published-version join remains authoritative.
        with self.engine.connect() as connection:
            present = connection.execute(text("""
                    SELECT 1 FROM crebry.document_chunks c JOIN crebry.documents d
                      ON d.id = c.document_id AND d.workspace_id = c.workspace_id AND d.index_version = c.version
                    WHERE c.workspace_id = :workspace LIMIT 1
                """), {"workspace": workspace_id}).scalar()
        if not present:
            return []
        embedding = vector_literal(self.embed_query(question)[0])
        with self.engine.connect() as connection:
            rows = connection.execute(statement, {"workspace": workspace_id, "embedding": embedding,
                "count": max(1, min(int(count), 100))}).mappings().all()
        return [
                {"id": f"D{i + 1}", "kind": "document", "text": row["content"],
                 "similarity": round(max(0, min(1, 1 - float(row["distance"]))), 4),
                 **{field: row[field] for field in ("document_id", "name", "page", "version", "chunk")}}
                for i, row in enumerate(rows)
            ]

    def delete_document(self, workspace_id, document_id, version=None):
        statement = "DELETE FROM crebry.document_chunks WHERE workspace_id = :workspace AND document_id = :document"
        parameters = {"workspace": workspace_id, "document": document_id}
        if version is not None:
            statement += " AND version = :version"
            parameters["version"] = version
        with self.lock, self.engine.begin() as connection:
            connection.execute(text(statement), parameters)

    def delete_workspace(self, workspace_id):
        with self.lock, self.engine.begin() as connection:
            connection.execute(text("DELETE FROM crebry.document_chunks WHERE workspace_id = :workspace"),
                               {"workspace": workspace_id})
