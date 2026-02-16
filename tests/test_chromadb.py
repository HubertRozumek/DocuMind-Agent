"""
Tests for ChromaDB Vector Store component.

Tests vector store operations including initialization,
document addition, search, persistence, and cleanup.
"""

import sys
from pathlib import Path

import chromadb
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ==================== INITIALIZATION TESTS ====================


def test_chromadb_initialization(vector_store_config, mock_embedding_function):
    """Test ChromaDB initialization."""
    from src.vector_store.chroma_db import ChromaDBVectorStore

    store = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
        client=chromadb.PersistentClient(path=str(vector_store_config["persist_directory"])),
    )

    assert store.collection_name == vector_store_config["collection_name"]
    assert store.client is not None
    assert store.collection is not None


def test_chromadb_with_reset(vector_store_config, mock_embedding_function):
    """Test ChromaDB with reset_on_start."""
    from src.vector_store.chroma_db import ChromaDBVectorStore

    # Create first instance
    store1 = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
        client=chromadb.PersistentClient(path=str(vector_store_config["persist_directory"])),
    )
    docs_to_add = [{"id": "01", "text": "text", "metadata": {}}]
    # Add some data
    store1.add_documents(docs_to_add)

    # Create second instance with reset
    store2 = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
        client=chromadb.PersistentClient(path=str(vector_store_config["persist_directory"])),
    )

    # Collection should be empty
    count = store2.get_collection_stats()
    assert count.get("total_documents") == 0


def test_chromadb_without_reset(vector_store_config, mock_embedding_function):
    """Test ChromaDB persistence without reset."""
    from src.vector_store.chroma_db import ChromaDBVectorStore

    # First instance
    store1 = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=False,
        client=chromadb.PersistentClient(path=str(vector_store_config["persist_directory"])),
    )

    docs_to_add = [{"id": "01", "text": "text", "metadata": {}}]
    # Add some data
    store1.add_documents(docs_to_add)

    initial_count = store1.get_collection_stats().get("total_documents")

    # Second instance without reset
    store2 = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=False,
        client=chromadb.PersistentClient(path=str(vector_store_config["persist_directory"])),
    )

    # Should have same count
    assert store2.get_collection_stats().get("total_documents") == initial_count


# ==================== ADD OPERATIONS TESTS ====================


def test_add_single_text(vector_store_config, mock_embedding_function):
    """Test adding single text."""
    from src.vector_store.chroma_db import ChromaDBVectorStore

    store = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
        client=chromadb.PersistentClient(path=str(vector_store_config["persist_directory"])),
    )

    docs_to_add = [{"id": "01", "text": "text", "metadata": {}}]
    # Add some data
    store.add_documents(docs_to_add)

    stats = store.get_collection_stats()

    assert stats.get("sample_size") == 1
    assert stats.get("total_documents") == 1


def test_add_multiple_texts(vector_store_config, mock_embedding_function):
    """Test adding multiple texts."""
    from src.vector_store.chroma_db import ChromaDBVectorStore

    store = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
        client=chromadb.PersistentClient(path=str(vector_store_config["persist_directory"])),
    )

    docs_to_add = [
        {"id": "01", "text": "text", "metadata": {}},
        {"id": "02", "text": "text", "metadata": {}},
    ]
    # Add some data
    store.add_documents(docs_to_add)

    stats = store.get_collection_stats()

    assert stats.get("sample_size") == 2
    assert stats.get("total_documents") == 2


def test_add_texts_without_metadata(vector_store_config, mock_embedding_function):
    """Test adding texts without metadata."""
    from src.vector_store.chroma_db import ChromaDBVectorStore

    store = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
        client=chromadb.PersistentClient(path=str(vector_store_config["persist_directory"])),
    )

    docs_to_add = [{"id": "01", "text": "text", "metadata": {}}]
    store.add_documents(docs_to_add)

    stats = store.get_collection_stats()

    assert stats.get("sample_size") == 1


# ==================== SEARCH TESTS ====================


def test_similarity_search(vector_store_config, mock_embedding_function):
    """Test similarity search."""
    from src.vector_store.chroma_db import ChromaDBVectorStore

    store = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
        client=chromadb.PersistentClient(path=str(vector_store_config["persist_directory"])),
    )

    # Add documents
    texts = [
        "Python is a programming language",
        "Machine learning uses Python",
        "The weather is nice today",
    ]
    docs_to_add = [{"id": i, "text": t, "metadata": {}} for i in range(len(texts)) for t in texts]

    store.add_documents(docs_to_add)
    # Search
    results = store.search(query="Python programming", n_results=2)

    assert len(results.get("ids")) <= 2


def test_search_empty_collection(vector_store_config, mock_embedding_function):
    """Test search on empty collection."""
    from src.vector_store.chroma_db import ChromaDBVectorStore

    store = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
        client=None,
    )

    results = store.search(query="test", n_results=5)

    assert len(results.get("ids")) == 0


# ==================== METADATA TESTS ====================


def test_search_returns_expected_structure(vector_store_config, mock_embedding_function):
    from src.vector_store.chroma_db import ChromaDBVectorStore

    store = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
        client=None,
    )

    store.add_documents([{"id": "1", "text": "hello world", "metadata": {"a": 1}}])

    results = store.search("hello")

    assert "ids" in results
    assert "documents" in results
    assert "metadatas" in results
    assert "distances" in results
    assert "similarities" in results


# ==================== PERFORMANCE TESTS ====================


def test_add_performance(vector_store_config, mock_embedding_function, performance_timer):
    from src.vector_store.chroma_db import ChromaDBVectorStore

    store = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
        client=None,
    )

    docs = [{"id": str(i), "text": f"text {i}", "metadata": {}} for i in range(100)]

    with performance_timer() as timer:
        store.add_documents(docs)

    assert timer.elapsed < 5.0


def test_search_performance(vector_store_config, mock_embedding_function, performance_timer):
    from src.vector_store.chroma_db import ChromaDBVectorStore

    store = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
        client=None,
    )

    docs = [{"id": str(i), "text": f"topic {i%5}", "metadata": {}} for i in range(300)]

    store.add_documents(docs)

    with performance_timer() as timer:
        results = store.search("topic", n_results=5)

    assert timer.elapsed < 1.0
    assert len(results["ids"]) <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
