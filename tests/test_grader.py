"""
Tests for Grader Node and Robust Grader components.

Tests document relevance grading, scoring mechanisms,
and grading strategies.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ==================== GRADER NODE TESTS ====================


def test_grader_node_initialization():
    """Test GraderNode initialization."""
    from src.agent.nodes.grader_node import GraderNode

    grader = GraderNode(grader_type="robust", confidence_threshold=0.7, model_name="phi3:mini")

    assert grader.confidence_threshold == 0.7
    assert grader.model_name == "phi3:mini"
    assert grader.grader is not None


def test_grader_node_default_values():
    """Test GraderNode with default values."""
    from src.agent.nodes.grader_node import GraderNode

    grader = GraderNode()

    assert grader.confidence_threshold > 0
    assert grader.grader is not None


def test_grade_single_document():
    """Test grading a single document."""
    from src.agent.nodes.grader_node import GraderNode
    from src.agent.nodes.robust_grader import GradingResult, RelevanceScore

    grader = GraderNode(model_name="phi3:mini")

    # Mock the internal grader
    mock_result = GradingResult(
        score=RelevanceScore.RELEVANT,
        confidence=0.85,
        reason="Document discusses Python",
        method="llm",
    )

    with patch.object(grader.grader, "grade", return_value=mock_result):
        result = grader.grade_document(
            question="What is Python?", document="Python is a programming language"
        )

    assert isinstance(result, dict)
    assert "confidence" in result
    assert "relevant" in result
    assert result["confidence"] == 0.85


def test_grade_multiple_documents():
    """Test grading multiple documents."""
    from src.agent.nodes.grader_node import GraderNode
    from src.agent.nodes.robust_grader import GradingResult, RelevanceScore

    grader = GraderNode(confidence_threshold=0.6)

    documents = [
        "Python is a programming language",
        "The weather is sunny today",
        "Python is used for machine learning",
    ]

    # Mock grade_batch to return mixed results
    mock_results = [
        GradingResult(RelevanceScore.RELEVANT, 0.9, "Relevant", "llm"),
        GradingResult(RelevanceScore.NOT_RELEVANT, 0.2, "Not relevant", "llm"),
        GradingResult(RelevanceScore.RELEVANT, 0.8, "Relevant", "llm"),
    ]

    with patch.object(grader.grader, "grade_batch", return_value=mock_results):
        result = grader.grade_documents(question="What is Python?", documents=documents)

    assert "relevant_documents" in result
    assert "relevant_count" in result
    assert "total_count" in result
    assert "avg_confidence" in result
    assert result["total_count"] == 3
    assert result["relevant_count"] == 2  # 2 out of 3 relevant


def test_grade_documents_empty_list():
    """Test grading empty document list."""
    from src.agent.nodes.grader_node import GraderNode

    grader = GraderNode()

    with patch.object(grader.grader, "grade_batch", return_value=[]):
        result = grader.grade_documents(question="Test question", documents=[])

    assert result["relevant_count"] == 0
    assert result["total_count"] == 0


def test_grader_confidence_threshold_filtering():
    """Test that confidence threshold filters documents."""
    from src.agent.nodes.grader_node import GraderNode
    from src.agent.nodes.robust_grader import GradingResult, RelevanceScore

    grader = GraderNode(confidence_threshold=0.7)

    documents = ["Doc 1", "Doc 2", "Doc 3"]

    # Mock results with varying confidence
    mock_results = [
        GradingResult(RelevanceScore.RELEVANT, 0.9, "High conf", "llm"),
        GradingResult(RelevanceScore.RELEVANT, 0.5, "Low conf", "llm"),
        GradingResult(RelevanceScore.RELEVANT, 0.8, "High conf", "llm"),
    ]

    # is_relevant checks against threshold
    for i, result in enumerate(mock_results):
        result.is_relevant = lambda thresh=0.7, conf=result.confidence: conf >= thresh

    with patch.object(grader.grader, "grade_batch", return_value=mock_results):
        result = grader.grade_documents(question="Test", documents=documents)

    # Should filter based on threshold
    assert "relevant_documents" in result


def test_grader_runnable_with_state():
    """Test grader runnable with GraphState."""
    from src.agent.graph_state import GraphState
    from src.agent.nodes.grader_node import GraderNode
    from src.agent.nodes.robust_grader import GradingResult, RelevanceScore

    grader = GraderNode(confidence_threshold=0.6)

    state = GraphState(
        question="What is Python?",
        search_query="Python",
        documents=["Python is a programming language", "Weather is sunny"],
        relevant_docs=[],
        iterations=0,
        max_iterations=3,
        needs_rewrite=False,
        confidence=0.0,
        history=[],
        metadata={},
    )

    # Mock the grading
    mock_results = [
        GradingResult(RelevanceScore.RELEVANT, 0.9, "Relevant", "llm"),
        GradingResult(RelevanceScore.NOT_RELEVANT, 0.2, "Not relevant", "llm"),
    ]

    with patch.object(grader.grader, "grade_batch", return_value=mock_results):
        runnable = grader.as_runnable()
        result_state = runnable.invoke(state)

    assert "relevant_docs" in result_state
    assert "confidence" in result_state
    assert len(result_state["relevant_docs"]) >= 0


# ==================== ROBUST GRADER TESTS ====================


def test_grading_result_dataclass():
    """Test GradingResult dataclass."""
    from src.agent.nodes.robust_grader import GradingResult, RelevanceScore

    result = GradingResult(
        score=RelevanceScore.RELEVANT, confidence=0.85, reason="Test reason", method="llm"
    )

    assert result.score == RelevanceScore.RELEVANT
    assert result.confidence == 0.85
    assert result.method == "llm"


def test_grading_result_is_relevant():
    """Test is_relevant method."""
    from src.agent.nodes.robust_grader import GradingResult, RelevanceScore

    # High confidence relevant
    result1 = GradingResult(RelevanceScore.RELEVANT, 0.9, "Test", "llm")
    assert result1.is_relevant(threshold=0.7) is True

    # Low confidence
    result2 = GradingResult(RelevanceScore.RELEVANT, 0.3, "Test", "llm")
    assert result2.is_relevant(threshold=0.7) is False

    # Not relevant score
    result3 = GradingResult(RelevanceScore.NOT_RELEVANT, 0.9, "Test", "llm")
    assert result3.is_relevant(threshold=0.6) is False


def test_grading_result_to_dict():
    """Test converting GradingResult to dict."""
    from src.agent.nodes.robust_grader import GradingResult, RelevanceScore

    result = GradingResult(
        score=RelevanceScore.HIGHLY_RELEVANT, confidence=0.95, reason="Very relevant", method="llm"
    )

    d = result.to_dict()

    assert isinstance(d, dict)
    assert "confidence" in d
    assert "relevant" in d
    assert "reason" in d
    assert "score" in d


def test_robust_grader_initialization():
    """Test RobustGrader initialization."""
    from src.agent.nodes.robust_grader import RobustGrader

    grader = RobustGrader(model_name="phi3:mini")

    assert grader.model_name == "phi3:mini"


@patch("src.agent.nodes.robust_grader.get_semantic_model")
def test_robust_grader_grade_document(mock_semantic_model):
    """Test grading a single document."""
    import numpy as np

    from src.agent.nodes.robust_grader import RobustGrader

    # Mock semantic model
    mock_model = Mock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]])
    mock_semantic_model.return_value = mock_model

    grader = RobustGrader(model_name="phi3:mini")

    # Mock LLM grading
    with patch.object(grader, "_grade_with_llm") as mock_llm:
        from src.agent.nodes.robust_grader import RelevanceScore

        mock_llm.return_value = RelevanceScore.RELEVANT

        result = grader.grade(
            question="What is Python?", document="Python is a programming language"
        )

    assert result is not None
    assert hasattr(result, "score")
    assert hasattr(result, "confidence")


@patch("src.agent.nodes.robust_grader.get_semantic_model")
def test_robust_grader_batch_grading(mock_semantic_model):
    """Test batch grading multiple documents."""
    import numpy as np

    from src.agent.nodes.robust_grader import RelevanceScore, RobustGrader

    # Mock semantic model
    mock_model = Mock()
    mock_model.encode.return_value = np.random.rand(3, 384)
    mock_semantic_model.return_value = mock_model

    grader = RobustGrader(model_name="phi3:mini")

    documents = [
        "Python is a programming language",
        "Weather is sunny",
        "Machine learning uses Python",
    ]

    with patch.object(grader, "_grade_with_llm", return_value=RelevanceScore.RELEVANT):
        results = grader.grade_batch("What is Python?", documents)

    assert len(results) == 3
    assert all(hasattr(r, "confidence") for r in results)


def test_robust_grader_fallback_on_llm_failure():
    """Test fallback mechanism when LLM fails."""
    from src.agent.nodes.robust_grader import RobustGrader

    grader = RobustGrader(model_name="phi3:mini")

    # Mock LLM to fail
    with (
        patch.object(grader, "_grade_with_llm", side_effect=Exception("LLM Error")),
        patch.object(grader, "_semantic_similarity", return_value=0.7),
        patch.object(grader, "_keyword_fallback", return_value=0.6),
        patch("src.agent.nodes.robust_grader.get_semantic_model"),
    ):
        result = grader.grade("test question", "test document")

        # Should fall back to semantic/keyword
        assert result is not None


# ==================== INTEGRATION TESTS ====================


def test_grader_node_with_state_no_documents():
    """Test grader handling state with no documents."""
    from src.agent.graph_state import GraphState
    from src.agent.nodes.grader_node import GraderNode

    grader = GraderNode()

    state = GraphState(
        question="Test",
        search_query="test",
        documents=[],  # Empty
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

    # Should handle empty documents gracefully
    assert result["relevant_docs"] == []
    assert result["confidence"] == 0.0


def test_grader_sets_needs_rewrite():
    """Test that grader sets needs_rewrite flag."""
    from src.agent.graph_state import GraphState
    from src.agent.nodes.grader_node import GraderNode
    from src.agent.nodes.robust_grader import GradingResult, RelevanceScore

    grader = GraderNode(confidence_threshold=0.7)

    state = GraphState(
        question="Test",
        search_query="test",
        documents=["Irrelevant document"],
        relevant_docs=[],
        iterations=0,
        max_iterations=3,
        needs_rewrite=False,
        confidence=0.0,
        history=[],
        metadata={},
    )

    # Mock all documents as not relevant
    mock_results = [GradingResult(RelevanceScore.NOT_RELEVANT, 0.2, "Not relevant", "llm")]

    with patch.object(grader.grader, "grade_batch", return_value=mock_results):
        runnable = grader.as_runnable()
        result = runnable.invoke(state)

    # Should set needs_rewrite if no relevant docs and iterations < max
    assert "needs_rewrite" in result


def test_grader_updates_history():
    """Test that grader updates history."""
    from src.agent.graph_state import GraphState
    from src.agent.nodes.grader_node import GraderNode
    from src.agent.nodes.robust_grader import GradingResult, RelevanceScore

    grader = GraderNode()

    state = GraphState(
        question="Test",
        search_query="test",
        documents=["Test doc"],
        relevant_docs=[],
        iterations=0,
        max_iterations=3,
        needs_rewrite=False,
        confidence=0.0,
        history=[],
        metadata={},
    )

    mock_results = [GradingResult(RelevanceScore.RELEVANT, 0.8, "Relevant", "llm")]

    with patch.object(grader.grader, "grade_batch", return_value=mock_results):
        runnable = grader.as_runnable()
        result = runnable.invoke(state)

    # Should add to history
    assert len(result["history"]) > 0
    assert result["history"][-1]["action"] == "grading"


# ==================== PERFORMANCE TESTS ====================


def test_grader_performance(performance_timer):
    """Test grader performance with multiple documents."""
    from src.agent.nodes.grader_node import GraderNode
    from src.agent.nodes.robust_grader import GradingResult, RelevanceScore

    grader = GraderNode()

    documents = [f"Document {i}" for i in range(20)]

    mock_results = [GradingResult(RelevanceScore.RELEVANT, 0.7, "Test", "llm") for _ in range(20)]

    with patch.object(grader.grader, "grade_batch", return_value=mock_results):
        with performance_timer() as timer:
            result = grader.grade_documents(question="Test", documents=documents)

        # Mocked should be fast
        assert result is not None
        assert timer.elapsed < 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
