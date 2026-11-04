from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import sys
from types import SimpleNamespace
import numpy as np
import pytest

from app.core.config import config
from app.core.memory import (
    check_and_collect_garbage,
    configure_process_memory_environment,
    get_process_memory_mb,
    log_stage_memory,
)
from app.services.documents import extract_text
from app.services.vector import VectorStore


@pytest.fixture
def local_embedding_modules(monkeypatch):
    """Unit tests use fake embedders and do not need optional PyTorch packages."""
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            set_num_threads=lambda _: None,
            set_num_interop_threads=lambda _: None,
            inference_mode=nullcontext,
        ),
    )


def test_memory_diagnostics_functions():
    configure_process_memory_environment()
    rss = get_process_memory_mb()
    assert isinstance(rss, (int, float))
    assert rss >= 0.0

    logged_rss = log_stage_memory("test_stage", test_key="test_value")
    assert logged_rss == rss or isinstance(logged_rss, (int, float))

    after_gc = check_and_collect_garbage(force=True)
    assert isinstance(after_gc, (int, float))


def test_health_endpoint_includes_memory_diagnostics(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "memory_rss_mb" in data
    assert "embedding_batch_size" in data
    assert data["embedding_batch_size"] == config.embedding_batch_size
    assert isinstance(data["memory_rss_mb"], (int, float))


def test_embedding_model_singleton_and_thread_safe_lazy_loading(
    tmp_path, monkeypatch, local_embedding_modules
):
    import app.services.vector as vector_module

    # Reset shared singleton for test
    monkeypatch.setattr(vector_module, "_SHARED_EMBEDDER", None)
    monkeypatch.setattr(config, "embedding_provider", "local")

    class MockST:
        init_count = 0

        def __init__(self, model_name, **kwargs):
            MockST.init_count += 1
            self.model_name = model_name

        def eval(self):
            pass

        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 384), dtype=np.float32)

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=MockST))

    store1 = VectorStore(tmp_path / "1", "mock-model")
    store2 = VectorStore(tmp_path / "2", "mock-model")

    # Before embed, model is None
    assert store1.model is None
    assert store2.model is None

    # Concurrently call embed across multiple threads
    def call_embed(store):
        return store.embed(["Test sentence"])

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(call_embed, [store1, store2, store1, store2]))

    # Model was only initialized ONCE across all instances and threads
    assert MockST.init_count == 1
    assert store1.model is store2.model
    assert len(results) == 4


def test_bounded_embedding_batches(tmp_path, monkeypatch, local_embedding_modules):
    calls = []

    class MockBatchST:
        def eval(self):
            pass

        def encode(self, texts, batch_size=32, **kwargs):
            calls.append((len(texts), batch_size))
            return np.zeros((len(texts), 384), dtype=np.float32)

    monkeypatch.setattr(config, "embedding_batch_size", 16)
    store = VectorStore(tmp_path, "mock-model", MockBatchST())

    # Create 35 chunks
    chunks = [{"text": f"Chunk text {i}", "page": 1} for i in range(35)]
    store.index("ws_test", "doc1", "test.txt", chunks, "v1")

    # With batch size 16: batches should be 16, 16, 3
    assert len(calls) == 3
    assert calls[0][0] == 16
    assert calls[1][0] == 16
    assert calls[2][0] == 3
    assert all(batch_size <= 16 for _, batch_size in calls)


def test_streaming_upload_duplicate_and_temp_cleanup(client, workspace, tmp_path):
    temp_dir = config.data_dir / ".tmp"

    # Upload valid document
    content = b"Unique content for streaming upload test."
    res = client.post(
        f"/api/workspaces/{workspace}/documents",
        files=[("files", ("stream_test.txt", content, "text/plain"))],
    )
    assert res.status_code == 201
    assert len(res.json()) == 1
    assert not res.json()[0]["duplicate"]

    # Verify no leaked temporary files in .tmp
    if temp_dir.exists():
        assert len(list(temp_dir.iterdir())) == 0

    # Upload duplicate
    res_dup = client.post(
        f"/api/workspaces/{workspace}/documents",
        files=[("files", ("stream_test.txt", content, "text/plain"))],
    )
    assert res_dup.status_code == 201
    assert res_dup.json()[0]["duplicate"]

    # Verify no leaked temporary files after duplicate
    if temp_dir.exists():
        assert len(list(temp_dir.iterdir())) == 0


def test_oversized_file_rejected_without_leaks(client, workspace, monkeypatch):
    monkeypatch.setattr(config, "max_upload_mb", 1)
    temp_dir = config.data_dir / ".tmp"

    # 1 MB + 1 byte
    oversized = b"x" * (1024 * 1024 + 1)
    res = client.post(
        f"/api/workspaces/{workspace}/documents",
        files=[("files", ("oversized.txt", oversized, "text/plain"))],
    )
    assert res.status_code == 201
    result = res.json()[0]
    assert "error" in result
    assert "exceeds the effective 1 MB upload limit" in result["error"]

    # No files leaked in .tmp
    if temp_dir.exists():
        assert len(list(temp_dir.iterdir())) == 0


def test_early_abort_on_extracted_char_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "max_extracted_chars", 500)

    # 1000 characters
    long_text = b"A" * 1000
    with pytest.raises(ValueError, match="exceeds the"):
        extract_text(long_text, ".txt")


def test_list_documents_defers_text_column(client, workspace):
    # Upload a document
    content = b"Sample text for document listing verification."
    res = client.post(
        f"/api/workspaces/{workspace}/documents",
        files=[("files", ("doc_list.txt", content, "text/plain"))],
    )
    assert res.status_code == 201

    # List documents
    docs = client.get(f"/api/workspaces/{workspace}/documents").json()
    assert len(docs) == 1
    # text column must not be present in document summary
    assert "text" not in docs[0]
    assert "content_hash" not in docs[0]
    assert docs[0]["name"] == "doc_list.txt"


def test_query_excludes_embedding_vectors(vectors):
    chunks = [{"text": "Query include test evidence text.", "page": 1}]
    vectors.index("ws_include", "doc_inc", "inc.txt", chunks, "v1")

    # Retrieve chunks
    results = vectors.retrieve("ws_include", "evidence text", 1)
    assert len(results) == 1
    assert "text" in results[0]
    assert "similarity" in results[0]
    assert "document_id" in results[0]
    # No raw embeddings exposed in result dictionary
    assert "embeddings" not in results[0]
