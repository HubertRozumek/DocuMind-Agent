import os
import shutil
import time
import pytest
from src.vector_store.embeddings_manager import EmbeddingManager, EmbeddingModelType
from src.vector_store.chroma_db import ChromaDBVectorStore

PERSIST_DIR = "./data/test_chroma_temp"
COLLECTION_NAME = "test_collection_integration"


@pytest.fixture(scope="session")
def embeddings_manager():
    manager = EmbeddingManager(
        model_type=EmbeddingModelType.MULTILINGUAL_MINILM,
        device="cpu"
    )
    manager.load_model()
    return manager


@pytest.fixture(scope="module")
def sample_documents():
    return [
        {"id": "1", "text": "Rust ownership system ensures memory safety without garbage collector.",
         "metadata": {"source": "doc1"}},
        {"id": "2", "text": "Borrowing allows accessing data without taking ownership.",
         "metadata": {"source": "doc1"}},
        {"id": "3", "text": "Python uses reference counting and garbage collection.",
         "metadata": {"source": "doc2"}},
        {"id": "4", "text": "Java runs on Virtual Machine.",
         "metadata": {"source": "doc3"}},
    ]


@pytest.fixture(scope="function")
def chroma_store(embeddings_manager):
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)

    store = ChromaDBVectorStore(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings_manager.chroma_embedding_function(),
        reset_on_start=True
    )

    yield store

    try:
        store.client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    time.sleep(0.2)
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)


def test_chromadb_integration_flow(chroma_store, sample_documents, embeddings_manager):
    added_count = chroma_store.add_documents(documents=sample_documents, batch_size=2)
    assert added_count == len(sample_documents)

    query = "memory safety ownership"
    results = chroma_store.search(query=query, n_results=2)

    assert len(results["ids"]) == 2
    assert "1" in results["ids"]

    sims = results["similarities"]
    assert sims[0] >= sims[1], f"Results not sorted: {sims}"

    query_vec = embeddings_manager.encode(["garbage collection"])[0]

    emb_results = chroma_store.search_with_embeddings(
        query_embedding=query_vec,
        n_results=2
    )

    assert "3" in emb_results["ids"]
    assert "distance" in emb_results or "similarities" in emb_results

    doc = chroma_store.get_document("2")
    assert doc is not None
    assert "Borrowing" in doc["document"]

    stats = chroma_store.get_collection_stats()
    assert stats["total_documents"] == 4
    assert "metadata_fields" in stats