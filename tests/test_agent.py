import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent.agent_builder import create_agent
from src.agent.edges import should_continue_to_rewriter

@pytest.mark.parametrize(
    "question,expected_max_iterations",
    [
        ("What is the company vacation policy?", 1),
        ("How time off?", 2),
        ("What is the meaning of life?", 3),
    ],
)
def test_self_correction_iterations(question, expected_max_iterations):
    agent = create_agent(
        max_iterations=3,
        grader_config={
            "grader_type": "mock",
            "confidence_threshold": 0.6,
        },
    )

    response = agent.invoke(question)

    assert "iterations_used" in response
    assert response["iterations_used"] <= expected_max_iterations
    assert response["iterations_used"] <= agent.max_iterations

    assert "answer" in response
    assert isinstance(response["answer"], str)

    assert "confidence" in response
    assert 0.0 <= response["confidence"] <= 1.0

    assert "state_summary" in response
    summary = response["state_summary"]

    assert "documents_found" in summary
    assert "relevant_documents" in summary


def test_self_correction_activates_rewrite_when_needed():
    agent = create_agent(
        max_iterations=3,
        grader_config={
            "grader_type": "mock",
            "confidence_threshold": 0.6,
        },
    )

    response = agent.invoke("How time off?")

    assert response["iterations_used"] >= 2

    summary = response["state_summary"]
    rewritten = summary.get("rewritten_questions", [])

    assert isinstance(rewritten, list)
    assert len(rewritten) >= 1


def test_iteration_limit_is_enforced():
    agent = create_agent(
        max_iterations=2,
        grader_config={
            "grader_type": "mock",
        },
    )

    response = agent.invoke("What is the meaning of life?")

    assert response["iterations_used"] == agent.max_iterations

def test_edge_statistics_are_present_when_available():
    agent = create_agent(
        max_iterations=3,
        grader_config={"grader_type": "mock"},
    )

    response = agent.invoke("How time off?")

    edge_stats = response.get("edge_statistics")

    if edge_stats:
        assert edge_stats["total_decisions"] > 0
        assert sum(edge_stats["decision_counts"].values()) == edge_stats["total_decisions"]
        assert all(isinstance(v, int) for v in edge_stats["decision_counts"].values())

# def test_graph_visualization_exists():
#     agent = create_agent(max_iterations=2)
#
#     visualization = agent.get_graph_visualization()
#
#     assert isinstance(visualization, str)
#     assert len(visualization) > 0


@pytest.mark.parametrize(
    "state,expected",
    [
        (
            {
                "iterations": 0,
                "max_iterations": 3,
                "relevant_docs": [],
                "documents": ["doc1", "doc2"],
                "confidence": 0.3,
            },
            "rewrite",
        ),
        (
            {
                "iterations": 1,
                "max_iterations": 3,
                "relevant_docs": ["doc1"],
                "documents": ["doc1", "doc2"],
                "confidence": 0.8,
            },
            "generate",
        ),
        (
            {
                "iterations": 3,
                "max_iterations": 3,
                "relevant_docs": [],
                "documents": ["doc1", "doc2"],
                "confidence": 0.3,
            },
            "end",
        ),
    ],
)
def test_should_continue_to_rewriter(state, expected):
    decision = should_continue_to_rewriter(state)
    assert decision == expected