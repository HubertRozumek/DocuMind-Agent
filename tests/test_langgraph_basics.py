import os
import sys
from datetime import datetime

import numpy as np
import pytest
from nltk.corpus.reader import documents
from oauthlib.uri_validate import query

from src.agent.graph_state import StateManager, serialize_state
from src.agent.nodes.retriever_node import RetrieverFactory
from src.agent.nodes.similarity_search import SimilaritySearch, SimilarityConfig, SimilarityMetric

def test_graph_state_lifecycle():

    initial_state = StateManager.create_initial_state(
        question="What are password security rules?",
        search_threshold=0.7,
        max_iterations=3
    )

    assert initial_state["question"] == "What are password security rules?"
    assert initial_state["search_threshold"] == 0.7
    assert initial_state["max_iterations"] == 3
    assert initial_state["iterations"] == 0

    updated_state = StateManager.update_state(
        initial_state,
        documents = [
            "Doc 1: Password rules...",
            "Doc 2: Security best practices...",
        ],
        relevant_docs = ["Doc 1"],
        confidence = 0.85,
        answer="Password must be at least 12 characters long.",
        increment_iterations = True
    )

    assert updated_state["iterations"] == 1
    assert len(updated_state["documents"]) == 2
    assert len(updated_state["relevant_docs"]) == 1
    assert updated_state["confidence"] == pytest.approx(0.85)

    validation = StateManager.validate_state(updated_state)
    assert validation["is_valid"] is True
    assert validation["errors"] == {}

    state_json = serialize_state(updated_state)
    assert isinstance(state_json, str)
    assert len(state_json) > 0

def test_chromadb_retriever_execution():

    try:
        retriever = RetrieverFactory.create_retriever(
            collection_name="lang_test",
            persist_directory="data/vector_store/chroma",
            search_config={
                "k": 5,
                "score_threshold": 0.7,
                "include_metadata": True,
                "rerank": True,
            },
        )
    except Exception:
        pytest.skip("ChromaDB vector store not available")

    state = StateManager.create_initial_state(
        question="information security",
    )

    retriever_fn = retriever.as_runnable()
    result_state = retriever_fn.invoke(state)

    assert "documents" in result_state
    assert "confidence" in result_state
    assert isinstance(result_state["documents"], list)
    assert 0.0 <= result_state["confidence"] <= 1.0

def test_similarity_search_with_threshold():

    config = SimilarityConfig(
        metric = SimilarityMetric.COSINE,
        threshold = 0.7,
        k=3,
        normalize = True,
        boost_recent=True,
        diversity_penalty=0.3,
    )

    search = SimilaritySearch(config)

    query_vector = np.array([0.1, 0.2, 0.3, 0.4])
    document_vectors = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 0.8, 0.7, 0.6],
        [-0.1, -0.2, -0.3, -0.4],
    ])

    documents = [
        "Very similar document",
        "Moderately similar document",
        "Low similarity document",
        "Opposite document",
    ]

    metadatas = [
        {"source": "doc1", "timestamp": "2026-01-15T10:00:00"},
        {"source": "doc2", "timestamp": "2026-01-10T10:00:00"},
        {"source": "doc3", "timestamp": "2026-01-05T10:00:00"},
        {"source": "doc4", "timestamp": "2026-01-01T10:00:00"},
    ]

    similarities = search.calculate_similarity(query_vector, document_vectors)

    assert len(similarities) == 4
    assert similarities[0] > similarities[-1]

    filtered_sim, filtered_docs, filtered_meta = search.apply_threshold(
        similarities, documents, metadatas
    )

    assert all(sim >= config.threshold for sim in filtered_sim)
    assert len(filtered_docs) == len(filtered_sim)

    results = search.search(
        query_vector=query_vector,
        document_vectors=document_vectors,
        documents=documents,
        metadatas=metadatas,
    )

    assert "documents" in results
    assert "stats" in results
    assert results["stats"]["avg_similarity"] >= 0.0

def test_integrated_workflow_simulation():
    """Test full workflow using simulated retriever and grading."""

    state = StateManager.create_initial_state(
        question="How to report a security incident?",
        vector_store_type="chromadb",
        search_threshold=0.65,
        max_iterations=2,
    )

    simulated_documents = [
        "Incident reporting procedure...",
        "Incident form available in HelpDesk...",
        "Security contact details...",
        "Incident response time SLA...",
    ]

    similarities = [0.85, 0.78, 0.72, 0.65]

    state = StateManager.update_state(
        state,
        documents=simulated_documents,
        confidence=float(np.mean(similarities)),
        metadata={
            "simulated_search": True,
            "similarities": similarities,
        },
        increment_iterations=True,
    )

    assert len(state["documents"]) == 4
    assert state["iterations"] == 1

    relevant_docs = simulated_documents[:3]

    state = StateManager.update_state(
        state,
        relevant_docs=relevant_docs,
        confidence=0.8,
        metadata={
            "grader_results": {
                "documents_evaluated": 4,
                "relevant_found": 3,
                "rejection_rate": 0.25,
            }
        },
    )

    assert len(state["relevant_docs"]) == 3

    answer = (
        "To report a security incident, use the HelpDesk form "
        "or contact the security team directly."
    )

    state = StateManager.update_state(
        state,
        answer=answer,
        confidence=0.85,
        metadata={
            "answer_generated": True,
            "timestamp": datetime.now().isoformat(),
        },
    )

    assert state["answer"] == answer
    assert state["confidence"] == pytest.approx(0.85)