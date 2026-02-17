"""
Tests for Embeddings Manager and Retriever Node.

Tests embedding generation, retriever functionality,
and document retrieval with scoring.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ==================== EMBEDDINGS MANAGER TESTS ====================


def test_embedding_manager_initialization():
    """Test EmbeddingManager initialization."""
    from src.vector_store.embeddings_manager import EmbeddingManager

    manager = EmbeddingManager()
    assert manager is not None


def test_chroma_embedding_function():
    """Test getting ChromaDB embedding function."""
    from src.vector_store.embeddings_manager import EmbeddingManager

    manager = EmbeddingManager()
    embedding_fn = manager.chroma_embedding_function()

    assert embedding_fn is not None
    assert callable(embedding_fn)


def test_embedding_function_produces_vectors():
    """Test that embedding function produces vectors."""
    from src.vector_store.embeddings_manager import EmbeddingManager

    manager = EmbeddingManager()
    embedding_fn = manager.chroma_embedding_function()

    # Test with single text
    result = embedding_fn(["test text"])

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert len(result[0]) > 0  # Should have dimensions


def test_embedding_function_batch():
    """Test embedding function with multiple texts."""
    from src.vector_store.embeddings_manager import EmbeddingManager

    manager = EmbeddingManager()
    embedding_fn = manager.chroma_embedding_function()

    texts = ["text one", "text two", "text three"]
    results = embedding_fn(texts)

    assert len(results) == 3
    assert all(isinstance(r, list) for r in results)


# ==================== RETRIEVER NODE TESTS ====================


def test_retriever_node_initialization(vector_store_config, mock_embedding_function):
    """Test RetrieverNode initialization."""
    from src.agent.nodes.retriever_node import RetrieverNode
    from src.vector_store.chroma_db import ChromaDBVectorStore

    vector_store = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
    )

    retriever = RetrieverNode(vector_store=vector_store, search_config={"k": 3, "score_threshold": 0.5})

    assert retriever.vector_store is not None
    assert retriever.search_config["k"] == 3


def test_retriever_retrieve_documents(vector_store_config, mock_embedding_function):
    """Test document retrieval."""
    from src.agent.graph_state import GraphState
    from src.agent.nodes.retriever_node import RetrieverNode
    from src.vector_store.chroma_db import ChromaDBVectorStore

    # Setup vector store with documents
    vector_store = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
        client=None,
    )
    texts = [
        "Python is a programming language",
        "Machine learning is fascinating",
        "The weather is nice",
    ]
    docs_to_add = [{"id": i, "text": t, "metadata": {}} for i, t in enumerate(texts)]
    # Add some data
    vector_store.add_documents(docs_to_add)

    retriever = RetrieverNode(vector_store=vector_store, search_config={"k": 2})

    # Create state
    state = GraphState(
        question="What is Python?",
        search_query="Python programming",
        documents=[],
        relevant_docs=[],
        iterations=0,
        needs_rewrite=False,
        confidence=0.0,
        history=[],
        metadata={},
    )

    # Retrieve
    result_state = retriever.retrieve("", state)

    assert "documents" in result_state
    assert len(result_state["documents"]) <= 2


def test_retriever_empty_collection(vector_store_config, mock_embedding_function):
    """Test retrieval from empty collection."""
    from src.agent.graph_state import GraphState
    from src.agent.nodes.retriever_node import RetrieverNode
    from src.vector_store.chroma_db import ChromaDBVectorStore

    vector_store = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
        client=None,
    )

    retriever = RetrieverNode(vector_store=vector_store)

    state = GraphState(
        question="Test query",
        search_query="Test query",
        documents=[],
        relevant_docs=[],
        iterations=0,
        needs_rewrite=False,
        confidence=0.0,
        history=[],
        metadata={},
    )

    result_state = retriever.retrieve("", state=state)

    assert result_state["documents"] == []


def test_retriever_with_score_threshold(vector_store_config, mock_embedding_function):
    """Test retrieval with score threshold."""
    from src.agent.graph_state import GraphState
    from src.agent.nodes.retriever_node import RetrieverNode
    from src.vector_store.chroma_db import ChromaDBVectorStore

    vector_store = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
        client=None,
    )

    texts = ["Relevant doc", "Another doc"]
    docs_to_add = [{"id": i, "text": t, "metadata": {}} for i, t in enumerate(texts)]
    # Add some data
    vector_store.add_documents(docs_to_add)

    retriever = RetrieverNode(vector_store=vector_store, search_config={"k": 5, "score_threshold": 0.8})

    state = GraphState(
        question="Query",
        search_query="Query",
        documents=[],
        relevant_docs=[],
        iterations=0,
        needs_rewrite=False,
        confidence=0.0,
        history=[],
        metadata={},
    )

    result_state = retriever.retrieve("", state)

    # Results depend on mock embeddings, just verify it runs
    assert "documents" in result_state


def test_retriever_updates_state_metadata(vector_store_config, mock_embedding_function):
    """Test that retriever updates state metadata."""
    from src.agent.graph_state import GraphState
    from src.agent.nodes.retriever_node import RetrieverNode
    from src.vector_store.chroma_db import ChromaDBVectorStore

    vector_store = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
        client=None,
    )

    texts = ["Test document"]
    docs_to_add = [{"id": i, "text": t, "metadata": {}} for i, t in enumerate(texts)]
    # Add some data
    vector_store.add_documents(docs_to_add)

    retriever = RetrieverNode(vector_store=vector_store)

    state = GraphState(
        question="Test",
        search_query="Test",
        documents=[],
        relevant_docs=[],
        iterations=0,
        needs_rewrite=False,
        confidence=0.0,
        history=[],
        metadata={},
    )

    result_state = retriever.retrieve("", state)

    # Should have updated metadata
    assert "search_history" in result_state or "metadatas" in result_state


def test_retriever_performance(vector_store_config, mock_embedding_function, performance_timer):
    """Test retriever performance."""
    from src.agent.graph_state import GraphState
    from src.agent.nodes.retriever_node import RetrieverNode
    from src.vector_store.chroma_db import ChromaDBVectorStore

    vector_store = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
        client=None,
    )

    # Add many documents
    texts = [f"Document {i}" for i in range(200)]
    docs_to_add = [{"id": i, "text": t, "metadata": {}} for i, t in enumerate(texts)]
    # Add some data
    vector_store.add_documents(docs_to_add)
    retriever = RetrieverNode(vector_store=vector_store, search_config={"k": 5})

    state = GraphState(
        question="Test",
        search_query="Test",
        documents=[],
        relevant_docs=[],
        iterations=0,
        needs_rewrite=False,
        confidence=0.0,
        history=[],
        metadata={},
    )

    with performance_timer() as timer:
        result_state = retriever.retrieve("", state)

    assert result_state is not None
    # Should be fast
    assert timer.elapsed < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
