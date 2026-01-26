import os
import shutil
import pytest
import time
import numpy as np
from src.vector_store.embeddings_manager import EmbeddingManager, EmbeddingModelType
from src.vector_store.chroma_db import ChromaDBVectorStore

PERSIST_DIR = "./data/test_chroma_robust"
COLLECTION_NAME = "test_collection_robust"


@pytest.fixture(scope="session")
def manager():
    manager = EmbeddingManager(model_type=EmbeddingModelType.MULTILINGUAL_MINILM, device="cpu")
    manager.load_model()
    return manager


@pytest.fixture(scope="function")
def store(manager):
    test_dir = f"{PERSIST_DIR}_{int(time.time() * 1000)}"

    vector_store = ChromaDBVectorStore(
        collection_name=COLLECTION_NAME,
        persist_directory=test_dir,
        embedding_function=manager.chroma_embedding_function(),
        reset_on_start=True
    )

    yield vector_store

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir, ignore_errors=True)

def test_embedding_normalization(manager):
    texts = ["Test normalization", "next text"]
    embeddings = manager.encode(texts, normalize_embeddings=True)

    for emb in embeddings:
        norm = np.linalg.norm(emb)
        assert pytest.approx(norm, rel=1e-5) == 1.0


def test_embedding_empty_input(manager):
    assert manager.encode([]).size == 0


def test_document_full_lifecycle(store):
    doc_id = "unique_id_99"
    doc_content = "This is a test document for the lifecycle."

    # 1. Add
    store.add_documents([{"id": doc_id, "text": doc_content, "metadata": {"status": "old"}}])

    # 2. Get
    retrieved = store.get_document(doc_id)
    assert retrieved["document"] == doc_content

    # 3. Update
    new_text = "Updated document content."
    store.update_document(doc_id, text=new_text, metadata={"status": "new"})
    updated = store.get_document(doc_id)
    assert updated["document"] == new_text
    assert updated["metadata"]["status"] == "new"

    # 4. Delete
    store.delete_document(doc_id)
    assert store.get_document(doc_id) is None


def test_metadata_filtering(store):
    docs = [
        {"id": "a", "text": "Red apple", "metadata": {"color": "red", "type": "fruit"}},
        {"id": "b", "text": "Blue bird", "metadata": {"color": "blue", "type": "animal"}},
        {"id": "c", "text": "Red pepper", "metadata": {"color": "red", "type": "vegetable"}},
    ]
    store.add_documents(docs)

    results = store.search(query="food", n_results=5, where={"color": "red"})

    assert len(results["ids"]) == 2
    assert "b" not in results["ids"]
    assert all(m["color"] == "red" for m in results["metadatas"])


def test_batch_metadata_enrichment(store):
    store.add_documents([{"id": "meta_test", "text": "short text"}])
    doc = store.get_document("meta_test")

    assert "timestamp" in doc["metadata"]
    assert doc["metadata"]["text_length"] == len("short text")
    assert "source" in doc["metadata"]


def test_collection_stats_and_empty(store):
    stats_empty = store.get_collection_stats()
    assert stats_empty["total_documents"] == 0

    store.add_documents([{"text": "Sample"}] * 5)
    stats_full = store.get_collection_stats()
    assert stats_full["total_documents"] == 5
    assert "metadata_fields" in stats_full