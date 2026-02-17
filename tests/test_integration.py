"""
Integration Tests - End-to-End Pipeline.

Tests complete RAG pipeline from PDF loading through
text splitting, embedding, retrieval, grading, and answer generation.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ==================== FULL PIPELINE TESTS ====================


def test_pdf_to_vector_store_pipeline(create_test_pdf, vector_store_config, mock_embedding_function):
    """Test complete pipeline: PDF -> Chunks -> Vector Store."""
    from src.document_processor.pdf_loader import PDFLoader
    from src.document_processor.text_splitter import TextSplitter
    from src.vector_store.chroma_db import ChromaDBVectorStore

    # 1. Load PDF
    content = """
    Machine Learning Introduction

    Machine learning is a method of data analysis that automates analytical
    model building. It is a branch of artificial intelligence based on the idea
    that systems can learn from data, identify patterns and make decisions.

    Key Concepts:
    - Supervised learning uses labeled data
    - Unsupervised learning finds hidden patterns
    - Deep learning uses neural networks
    """

    pdf_path = create_test_pdf(content, num_pages=2, filename="ml_intro.pdf")

    loader = PDFLoader()
    documents = loader.load_pdf(str(pdf_path))

    assert len(documents) >= 1

    # 2. Split into chunks
    splitter = TextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    assert len(chunks) >= 1

    # 3. Add to vector store
    vector_store = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
    )

    # Convert chunks to documents format
    docs_to_add = []
    for i, chunk in enumerate(chunks):
        docs_to_add.append({"id": chunk.chunk_id, "text": chunk.text, "metadata": chunk.metadata})

    count = vector_store.add_documents(docs_to_add)

    assert count == len(chunks)


def test_retrieval_pipeline(vector_store_config, mock_embedding_function):
    """Test retrieval pipeline with search."""
    from src.agent.graph_state import GraphState
    from src.agent.nodes.retriever_node import RetrieverNode
    from src.vector_store.chroma_db import ChromaDBVectorStore

    # Setup vector store with documents
    vector_store = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
    )

    docs = [
        {
            "id": "1",
            "text": "Python is a high-level programming language known for its simplicity.",
            "metadata": {"topic": "programming"},
        },
        {
            "id": "2",
            "text": "Machine learning models require training data to learn patterns.",
            "metadata": {"topic": "ml"},
        },
        {
            "id": "3",
            "text": "Neural networks are inspired by biological neural networks.",
            "metadata": {"topic": "ml"},
        },
        {
            "id": "4",
            "text": "Data preprocessing is crucial for machine learning success.",
            "metadata": {"topic": "ml"},
        },
    ]

    vector_store.add_documents(docs)

    # Create retriever
    retriever = RetrieverNode(vector_store=vector_store, search_config={"k": 3, "score_threshold": 0.5})

    # Create state
    state = GraphState(
        question="What is machine learning?",
        search_query="machine learning",
        documents=[],
        relevant_docs=[],
        iterations=0,
        max_iterations=3,
        needs_rewrite=False,
        confidence=0.0,
        history=[],
        metadata={},
    )

    # Retrieve using runnable
    runnable = retriever.as_runnable()
    result_state = runnable.invoke(state)

    assert "documents" in result_state
    assert len(result_state["documents"]) <= 3


def test_retrieval_and_grading_pipeline(vector_store_config, mock_embedding_function):
    """Test retrieval followed by grading."""
    from src.agent.graph_state import GraphState
    from src.agent.nodes.grader_node import GraderNode
    from src.agent.nodes.retriever_node import RetrieverNode
    from src.agent.nodes.robust_grader import GradingResult, RelevanceScore
    from src.vector_store.chroma_db import ChromaDBVectorStore

    # Setup vector store
    vector_store = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
    )

    docs = [
        {"id": "1", "text": "Python is a programming language", "metadata": {}},
        {"id": "2", "text": "Machine learning uses Python", "metadata": {}},
        {"id": "3", "text": "The weather is sunny today", "metadata": {}},
    ]
    vector_store.add_documents(docs)

    # Create retriever and grader
    retriever = RetrieverNode(vector_store=vector_store)
    grader = GraderNode(confidence_threshold=0.6)

    # Initial state
    state = GraphState(
        question="What is Python?",
        search_query="Python programming",
        documents=[],
        relevant_docs=[],
        iterations=0,
        max_iterations=3,
        needs_rewrite=False,
        confidence=0.0,
        history=[],
        metadata={},
    )

    # Step 1: Retrieve
    retriever_runnable = retriever.as_runnable()
    state = retriever_runnable.invoke(state)

    assert len(state["documents"]) > 0

    # Step 2: Grade
    mock_results = [GradingResult(RelevanceScore.RELEVANT, 0.9, "Relevant", "llm") for _ in range(len(state["documents"]))]

    with patch.object(grader.grader, "grade_batch", return_value=mock_results):
        grader_runnable = grader.as_runnable()
        state = grader_runnable.invoke(state)

    assert "relevant_docs" in state
    assert len(state["relevant_docs"]) > 0


# ==================== MULTI-DOCUMENT TESTS ====================


def test_multi_document_processing(multiple_pdfs, vector_store_config, mock_embedding_function):
    """Test processing multiple PDFs through pipeline."""
    from src.document_processor.pdf_loader import PDFLoader
    from src.document_processor.text_splitter import TextSplitter
    from src.vector_store.chroma_db import ChromaDBVectorStore

    # Load all PDFs
    loader = PDFLoader()
    all_documents = []

    for pdf_path in multiple_pdfs:
        docs = loader.load_pdf(str(pdf_path))
        all_documents.extend(docs)

    assert len(all_documents) >= len(multiple_pdfs)

    # Split all
    splitter = TextSplitter(chunk_size=200, chunk_overlap=30)
    all_chunks = splitter.split_documents(all_documents)

    assert len(all_chunks) >= len(all_documents)

    # Store all
    vector_store = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
    )

    docs_to_add = [{"id": chunk.chunk_id, "text": chunk.text, "metadata": chunk.metadata} for chunk in all_chunks]

    count = vector_store.add_documents(docs_to_add)

    assert count == len(all_chunks)


# ==================== ERROR RECOVERY TESTS ====================


def test_pipeline_handles_empty_results(vector_store_config, mock_embedding_function):
    """Test pipeline handles empty search results."""
    from src.agent.graph_state import GraphState
    from src.agent.nodes.retriever_node import RetrieverNode
    from src.vector_store.chroma_db import ChromaDBVectorStore

    # Empty vector store
    vector_store = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
    )

    retriever = RetrieverNode(vector_store=vector_store)

    state = GraphState(
        question="Test query",
        search_query="Test query",
        documents=[],
        relevant_docs=[],
        iterations=0,
        max_iterations=3,
        needs_rewrite=False,
        confidence=0.0,
        history=[],
        metadata={},
    )

    runnable = retriever.as_runnable()
    result_state = runnable.invoke(state)

    # Should handle gracefully
    assert result_state["documents"] == []
    assert result_state["confidence"] == 0.0


def test_grader_handles_empty_documents():
    """Test grader handles empty document list."""
    from src.agent.graph_state import GraphState
    from src.agent.nodes.grader_node import GraderNode

    grader = GraderNode()

    state = GraphState(
        question="Test",
        search_query="test",
        documents=[],
        relevant_docs=[],
        iterations=0,
        max_iterations=3,
        needs_rewrite=False,
        confidence=0.0,
        history=[],
        metadata={},
    )

    runnable = grader.as_runnable()
    result = runnable.invoke(state)

    assert result["relevant_docs"] == []
    assert result["confidence"] == 0.0


# ==================== PERFORMANCE TESTS ====================


def test_full_pipeline_performance(create_test_pdf, vector_store_config, mock_embedding_function, performance_timer):
    """Test full pipeline performance."""
    from src.agent.graph_state import GraphState
    from src.agent.nodes.retriever_node import RetrieverNode
    from src.document_processor.pdf_loader import PDFLoader
    from src.document_processor.text_splitter import TextSplitter
    from src.vector_store.chroma_db import ChromaDBVectorStore

    # Create moderate-sized document
    content = "Test content paragraph. " * 100
    pdf_path = create_test_pdf(content, num_pages=3)

    with performance_timer() as timer:
        # Load
        loader = PDFLoader()
        documents = loader.load_pdf(str(pdf_path))

        # Split
        splitter = TextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        # Store
        vector_store = ChromaDBVectorStore(
            collection_name=vector_store_config["collection_name"],
            persist_directory=vector_store_config["persist_directory"],
            embedding_function=mock_embedding_function,
            reset_on_start=True,
        )

        docs_to_add = [{"id": chunk.chunk_id, "text": chunk.text, "metadata": chunk.metadata} for chunk in chunks]
        vector_store.add_documents(docs_to_add)

        # Retrieve
        retriever = RetrieverNode(vector_store=vector_store)
        state = GraphState(
            question="Test",
            search_query="Test",
            documents=[],
            relevant_docs=[],
            iterations=0,
            max_iterations=3,
            needs_rewrite=False,
            confidence=0.0,
            history=[],
            metadata={},
        )
        runnable = retriever.as_runnable()
        runnable.invoke(state)

    # Should complete in reasonable time
    assert timer.elapsed < 10.0


# ==================== DATA CONSISTENCY TESTS ====================


def test_metadata_consistency_through_pipeline(create_test_pdf, vector_store_config, mock_embedding_function):
    """Test that metadata is preserved through entire pipeline."""
    from src.document_processor.pdf_loader import PDFLoader
    from src.document_processor.text_splitter import TextSplitter
    from src.vector_store.chroma_db import ChromaDBVectorStore

    pdf_path = create_test_pdf("Test content", num_pages=1, filename="metadata_test.pdf")

    # Load with metadata
    loader = PDFLoader()
    documents = loader.load_pdf(str(pdf_path))

    # Split (preserves metadata)
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_documents(documents)

    # Verify metadata in chunks
    assert all("source" in chunk.metadata for chunk in chunks)

    # Store and retrieve
    vector_store = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
    )

    docs_to_add = [{"id": chunk.chunk_id, "text": chunk.text, "metadata": chunk.metadata} for chunk in chunks]
    vector_store.add_documents(docs_to_add)

    results = vector_store.search("Test", n_results=1)

    # Metadata should be preserved
    if len(results["metadatas"]) > 0:
        assert "source" in results["metadatas"][0]


# ==================== STRESS TESTS ====================


def test_large_batch_processing(vector_store_config, mock_embedding_function, performance_timer):
    """Test processing large batch of documents."""
    from src.vector_store.chroma_db import ChromaDBVectorStore

    vector_store = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
    )

    # Create 500 documents
    docs = [
        {
            "id": str(i),
            "text": f"Document {i} with content about topic {i % 20}",
            "metadata": {"index": i, "topic": i % 20},
        }
        for i in range(500)
    ]

    with performance_timer() as timer:
        count = vector_store.add_documents(docs)

    assert count == 500
    assert timer.elapsed < 30.0  # Should handle batch efficiently


# ==================== RECOVERY TESTS ====================


def test_pipeline_recovery_after_failure(vector_store_config, mock_embedding_function):
    """Test that pipeline can recover after component failure."""
    from src.vector_store.chroma_db import ChromaDBVectorStore

    vector_store = ChromaDBVectorStore(
        collection_name=vector_store_config["collection_name"],
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=mock_embedding_function,
        reset_on_start=True,
    )

    # Add some documents
    docs1 = [{"id": "1", "text": "Doc 1", "metadata": {}}]
    vector_store.add_documents(docs1)

    # Simulate failure and recovery
    try:
        # Force an error (documents without text)
        bad_docs = [{"id": "2", "metadata": {}}]
        vector_store.add_documents(bad_docs)
    except Exception:
        pass

    # Should still work after error
    docs2 = [{"id": "3", "text": "Doc 3", "metadata": {}}]
    count = vector_store.add_documents(docs2)
    assert count == 1


def test_retriever_with_various_query_types():
    """Test retriever handles different query types."""

    from src.agent.nodes.retriever_node import RetrieverNode
    from src.vector_store.chroma_db import ChromaDBVectorStore

    # Mock vector store
    mock_store = Mock(spec=ChromaDBVectorStore)
    mock_store.search.return_value = {
        "ids": ["1"],
        "documents": ["Test doc"],
        "metadatas": [{}],
        "distances": [0.5],
        "similarities": [0.5],
        "query": "test",
    }

    retriever = RetrieverNode(vector_store=mock_store)

    # Test different query types
    queries = [
        "What is Python?",
        "How to learn machine learning?",
        "Python vs Java comparison",
        "Short",
        "Very long query with many words to test preprocessing and truncation behavior",
    ]

    for query in queries:
        result = retriever.retrieve(query)
        assert "documents" in result
        assert "confidence" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
