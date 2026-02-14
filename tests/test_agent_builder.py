"""
Tests for Agent Builder and DocuMindAgent.

Tests agent initialization, graph building, invocation,
and end-to-end RAG pipeline execution.
"""

import pytest
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


# ==================== AGENT INITIALIZATION TESTS ====================

def test_agent_initialization():
    """Test DocuMindAgent initialization."""
    from src.agent.agent_builder import DocuMindAgent

    agent = DocuMindAgent(
        max_iterations=2,
        search_threshold=0.7,
        use_tools=False
    )

    assert agent.max_iterations == 2
    assert agent.search_threshold == 0.7
    assert agent.use_tools is False


def test_agent_initialization_with_configs(agent_config):
    """Test agent initialization with full configs."""
    from src.agent.agent_builder import DocuMindAgent

    agent = DocuMindAgent(
        vector_store_config=agent_config["vector_store_config"],
        grader_config=agent_config["grader_config"],
        generator_config=agent_config["generator_config"],
        max_iterations=agent_config["max_iterations"],
        use_tools=agent_config["use_tools"]
    )

    assert agent.vector_store_config == agent_config["vector_store_config"]
    assert agent.grader_config == agent_config["grader_config"]


def test_agent_default_configs():
    """Test agent uses sensible defaults."""
    from src.agent.agent_builder import DocuMindAgent

    agent = DocuMindAgent()

    assert agent.vector_store_config is not None
    assert agent.grader_config is not None
    assert agent.generator_config is not None
    assert agent.max_iterations > 0


# ==================== COMPONENT BUILDING TESTS ====================

@patch('src.agent.agent_builder.ChromaDBVectorStore')
@patch('src.agent.agent_builder.EmbeddingManager')
def test_build_components(mock_embedding, mock_vector_store, agent_config):
    """Test building agent components."""
    from src.agent.agent_builder import DocuMindAgent

    # Setup mocks
    mock_embedding.return_value.chroma_embedding_function.return_value = Mock()
    mock_vector_store.return_value = Mock()

    agent = DocuMindAgent(**agent_config)

    # Mock the internal components that require LLM
    with patch.object(agent, 'retriever'), \
            patch.object(agent, 'grader'), \
            patch.object(agent, 'rewriter'), \
            patch.object(agent, 'generator'):
        result = agent.build_components()

        assert result is True


@patch('src.agent.agent_builder.EmbeddingManager')
def test_build_components_creates_retriever(mock_embedding, agent_config):
    """Test that build_components creates retriever."""
    from src.agent.agent_builder import DocuMindAgent

    mock_embedding.return_value.chroma_embedding_function.return_value = Mock()

    agent = DocuMindAgent(**agent_config)

    with patch('src.agent.agent_builder.ChromaDBVectorStore'), \
            patch('src.agent.agent_builder.RetrieverNode'), \
            patch('src.agent.agent_builder.GraderNode'), \
            patch('src.agent.agent_builder.QueryRewriter'), \
            patch('src.agent.agent_builder.GeneratorNode'):
        agent.build_components()

        assert agent.retriever is not None


def test_build_components_failure_handling(agent_config):
    """Test that build_components handles failures."""
    from src.agent.agent_builder import DocuMindAgent

    agent = DocuMindAgent(**agent_config)

    # Force an error
    with patch('src.agent.agent_builder.EmbeddingManager', side_effect=Exception("Test error")):
        with pytest.raises(Exception):
            agent.build_components()


# ==================== GRAPH BUILDING TESTS ====================

@patch('src.agent.agent_builder.DocuMindAgent.build_components')
def test_build_graph(mock_build_components, agent_config):
    """Test graph building."""
    from src.agent.agent_builder import DocuMindAgent

    mock_build_components.return_value = True

    agent = DocuMindAgent(**agent_config)

    # Mock components
    agent.retriever = Mock()
    agent.grader = Mock()
    agent.rewriter = Mock()
    agent.generator = Mock()

    with patch('src.agent.agent_builder.StateGraph'):
        agent.build_graph()

        assert agent.graph is not None


# ==================== INVOCATION TESTS ====================

def test_agent_invoke_basic():
    """Test basic agent invocation."""
    from src.agent.agent_builder import DocuMindAgent

    agent = DocuMindAgent(use_tools=False)

    # Mock the graph
    mock_graph = Mock()
    mock_graph.invoke.return_value = {
        "question": "test",
        "answer": "Test answer",
        "confidence": 0.8,
        "iterations": 1,
        "relevant_docs": ["doc1"],
        "metadata": {}
    }

    agent.compiled_graph = mock_graph

    result = agent.invoke("Test question")

    assert "answer" in result
    assert "confidence" in result
    assert result["answer"] == "Test answer"


def test_agent_invoke_with_no_answer():
    """Test invocation when no answer is generated."""
    from src.agent.agent_builder import DocuMindAgent

    agent = DocuMindAgent()

    mock_graph = Mock()
    mock_graph.invoke.return_value = {
        "question": "test",
        "answer": "",  # Empty answer
        "confidence": 0.0,
        "iterations": 1,
        "metadata": {}
    }

    agent.compiled_graph = mock_graph

    result = agent.invoke("Test question")

    # Should have fallback answer
    assert "answer" in result
    assert len(result["answer"]) > 0


def test_agent_invoke_handles_errors():
    """Test that invoke handles errors gracefully."""
    from src.agent.agent_builder import DocuMindAgent

    agent = DocuMindAgent()

    mock_graph = Mock()
    mock_graph.invoke.side_effect = Exception("Graph error")

    agent.compiled_graph = mock_graph

    result = agent.invoke("Test question")

    # Should return error result
    assert "error" in result or "Error" in result["answer"]


def test_agent_invoke_returns_metadata():
    """Test that invoke returns comprehensive metadata."""
    from src.agent.agent_builder import DocuMindAgent

    agent = DocuMindAgent()

    mock_graph = Mock()
    mock_graph.invoke.return_value = {
        "question": "test",
        "answer": "Answer",
        "confidence": 0.9,
        "iterations": 2,
        "relevant_docs": ["doc1", "doc2"],
        "metadata": {"grading_result": {"relevant_count": 2}}
    }

    agent.compiled_graph = mock_graph
    agent.edge_router = Mock()
    agent.edge_router.get_edge_statistics.return_value = {}

    result = agent.invoke("Test question")

    assert "answer" in result
    assert "confidence" in result
    assert "iterations_used" in result
    assert "relevant_documents" in result
    assert "metadata" in result


# ==================== TOOL ROUTING TESTS ====================

def test_agent_with_tools_disabled():
    """Test agent behavior with tools disabled."""
    from src.agent.agent_builder import DocuMindAgent

    agent = DocuMindAgent(use_tools=False)

    assert agent.use_tools is False
    assert len(agent.tools) == 0


@patch('src.agent.agent_builder.create_all_tools')
def test_agent_with_tools_enabled(mock_tools, agent_config):
    """Test agent with tools enabled."""
    from src.agent.agent_builder import DocuMindAgent

    mock_tools.return_value = [Mock(), Mock()]

    agent = DocuMindAgent(use_tools=True)

    assert agent.use_tools is True


def test_tool_routing_node():
    """Test tool routing logic."""
    from src.agent.agent_builder import DocuMindAgent
    from src.agent.graph_state import GraphState

    agent = DocuMindAgent(use_tools=True)
    agent.tool_router = Mock()

    # Mock tool routing result
    agent.tool_router.route_query.return_value = {
        "success": True,
        "type": "tool",
        "tool_name": "test_tool",
        "result": "Tool result"
    }

    state = GraphState(
        question="Test query",
        search_query="test",
        documents=[],
        relevant_docs=[],
        iterations=0,
        needs_rewrite=False,
        confidence=0.0,
        history=[],
        metadata={}
    )

    result_state = agent._tool_routing_node(state)

    assert result_state.get("tool_used") is not None


# ==================== STATISTICS TESTS ====================

def test_get_agent_info():
    """Test getting agent information."""
    from src.agent.agent_builder import DocuMindAgent

    agent = DocuMindAgent(
        max_iterations=3,
        search_threshold=0.7,
        use_tools=True
    )

    info = agent.get_agent_info()

    assert "agent_name" in info
    assert "version" in info
    assert "features" in info
    assert "config" in info
    assert info["config"]["max_iterations"] == 3


def test_reset_statistics():
    """Test resetting agent statistics."""
    from src.agent.agent_builder import DocuMindAgent

    agent = DocuMindAgent()
    agent.edge_router = Mock()
    agent.tool_error_handler = Mock()

    agent.reset_statistics()

    agent.edge_router.edge_history.clear.assert_called_once()
    agent.tool_error_handler.reset_stats.assert_called_once()


# ==================== FACTORY FUNCTION TESTS ====================

@patch('src.agent.agent_builder.DocuMindAgent.build_graph')
@patch('src.agent.agent_builder.DocuMindAgent.build_components')
def test_create_agent_factory(mock_build_components, mock_build_graph, agent_config):
    """Test create_agent factory function."""
    from src.agent.agent_builder import create_agent

    mock_build_components.return_value = True

    agent = create_agent(**agent_config)

    assert agent is not None
    mock_build_graph.assert_called_once()


# ==================== INTEGRATION TESTS ====================

@patch('src.agent.agent_builder.ChromaDBVectorStore')
@patch('src.agent.agent_builder.EmbeddingManager')
def test_agent_end_to_end_mock(mock_embedding, mock_vector_store, agent_config):
    """Test end-to-end agent execution with mocks."""
    from src.agent.agent_builder import DocuMindAgent

    # Setup mocks
    mock_embedding.return_value.chroma_embedding_function.return_value = Mock()
    mock_vs_instance = Mock()
    mock_vs_instance.similarity_search.return_value = [
        {"text": "Python is a language", "metadata": {}}
    ]
    mock_vector_store.return_value = mock_vs_instance

    agent = DocuMindAgent(**agent_config)

    # Mock all components
    agent.retriever = Mock()
    agent.retriever.retrieve = Mock(return_value={
        "documents": ["Python is a language"],
        "metadata": {}
    })

    agent.grader = Mock()
    agent.grader.grade = Mock(return_value={
        "relevant_docs": ["Python is a language"],
        "confidence": 0.8
    })

    agent.generator = Mock()
    agent.generator.generate = Mock(return_value={
        "answer": "Python is a programming language",
        "confidence": 0.9
    })

    # Mock graph
    mock_graph = Mock()
    mock_graph.invoke.return_value = {
        "question": "What is Python?",
        "answer": "Python is a programming language",
        "confidence": 0.9,
        "iterations": 1,
        "relevant_docs": ["Python is a language"],
        "metadata": {}
    }

    agent.compiled_graph = mock_graph
    agent.edge_router = Mock()
    agent.edge_router.get_edge_statistics.return_value = {}

    # Invoke agent
    result = agent.invoke("What is Python?")

    # Verify result
    assert result["answer"]
    assert result["confidence"] > 0
    assert "relevant_documents" in result


def test_agent_handles_empty_documents():
    """Test agent behavior with no documents found."""
    from src.agent.agent_builder import DocuMindAgent

    agent = DocuMindAgent()

    mock_graph = Mock()
    mock_graph.invoke.return_value = {
        "question": "test",
        "answer": "No relevant documents found",
        "confidence": 0.0,
        "iterations": 1,
        "documents": [],
        "relevant_docs": [],
        "metadata": {}
    }

    agent.compiled_graph = mock_graph
    agent.edge_router = Mock()
    agent.edge_router.get_edge_statistics.return_value = {}

    result = agent.invoke("Test question")

    assert result["answer"]
    assert len(result["relevant_documents"]) == 0


def test_agent_iteration_limiting():
    """Test that agent respects max iterations."""
    from src.agent.agent_builder import DocuMindAgent

    agent = DocuMindAgent(max_iterations=2)

    mock_graph = Mock()
    mock_graph.invoke.return_value = {
        "question": "test",
        "answer": "Answer",
        "confidence": 0.7,
        "iterations": 2,
        "metadata": {}
    }

    agent.compiled_graph = mock_graph
    agent.edge_router = Mock()
    agent.edge_router.get_edge_statistics.return_value = {}

    result = agent.invoke("Test question")

    # Should not exceed max iterations
    assert result["iterations_used"] <= agent.max_iterations


# ==================== PERFORMANCE TESTS ====================

def test_agent_invocation_performance(performance_timer):
    """Test agent invocation performance."""
    from src.agent.agent_builder import DocuMindAgent

    agent = DocuMindAgent()

    # Mock quick graph
    mock_graph = Mock()
    mock_graph.invoke.return_value = {
        "question": "test",
        "answer": "Quick answer",
        "confidence": 0.8,
        "iterations": 1,
        "relevant_docs": ["doc"],
        "metadata": {}
    }

    agent.compiled_graph = mock_graph
    agent.edge_router = Mock()
    agent.edge_router.get_edge_statistics.return_value = {}

    with performance_timer() as timer:
        result = agent.invoke("Test question")

    # Mocked should be fast
    assert timer.elapsed < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])