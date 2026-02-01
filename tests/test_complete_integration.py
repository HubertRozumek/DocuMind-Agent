"""
Comprehensive integration tests for the complete agent system.
Task 10.2: Integration tests for different scenarios.
"""

import pytest
import os
import sys
from typing import Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tools.ticket_checker import MockTicketAPI, check_ticket_status
from src.tools.document_tool import create_all_tools, ToolRouter, ToolErrorHandler
from src.agent.enhanced_agent_builder import create_enhanced_agent


class TestTicketAPIIntegration:
    """Integration tests for ticket API"""

    def test_ticket_api_lifecycle(self):
        """Test complete ticket lifecycle"""
        api = MockTicketAPI()

        # Test 1: Get existing ticket
        ticket = api.get_ticket("TICKET-001")
        assert ticket is not None
        assert ticket["id"] == "TICKET-001"
        assert "status" in ticket

        # Test 2: Get ticket status
        status = api.get_ticket_status("TICKET-001")
        assert status["found"] is True
        assert status["status"] == "resolved"

        # Test 3: Search tickets
        open_tickets = api.search_tickets(status="open")
        assert len(open_tickets) >= 1
        assert all(t["status"] == "open" for t in open_tickets)

        # Test 4: Get user tickets
        user_tickets = api.get_my_tickets("Jan Kowalski")
        assert len(user_tickets) >= 1
        assert all(t["assigned_to"] == "Jan Kowalski" for t in user_tickets)

        # Test 5: Create new ticket
        new_ticket = api.create_ticket(
            title="Test ticket",
            description="This is a test",
            priority="high",
        )
        assert new_ticket["id"] is not None
        assert new_ticket["status"] == "open"
        assert new_ticket["priority"] == "high"

        # Test 6: Get statistics
        stats = api.get_statistics()
        assert stats["total_tickets"] >= 5
        assert "by_status" in stats
        assert "by_priority" in stats

    def test_ticket_convenience_functions(self):
        """Test convenience functions for LangChain"""
        # Test check_ticket_status
        result = check_ticket_status("TICKET-001")
        assert "TICKET-001" in result
        assert "Status:" in result
        assert "Priority:" in result

        # Test non-existent ticket
        result = check_ticket_status("TICKET-999")
        assert "not found" in result.lower()


class TestToolsIntegration:
    """Integration tests for tools system"""

    def test_all_tools_creation(self):
        """Test creating all tools"""
        tools = create_all_tools(vector_store=None)

        assert len(tools) > 0

        # Check ticket tools
        tool_names = [t.name for t in tools]
        assert "check_ticket_status" in tool_names
        assert "search_user_tickets" in tool_names
        assert "get_open_tickets" in tool_names

        # Check document tools
        assert "search_documents" in tool_names
        assert "get_document_count" in tool_names

    def test_tool_router_ticket_detection(self):
        """Test tool router correctly detects ticket queries"""
        tools = create_all_tools()
        router = ToolRouter(tools)

        result = router.should_use_tool("What is the status of TICKET-001?")
        assert result is not None
        tool_name, kwargs = result
        assert tool_name == "check_ticket_status"
        assert "ticket_id" in kwargs

        result = router.should_use_tool("Show me all open tickets")
        assert result is not None
        tool_name, kwargs = result
        assert tool_name == "get_open_tickets"

        result = router.should_use_tool("What are my tickets?")
        assert result is not None
        tool_name, kwargs = result
        assert tool_name == "search_user_tickets"
        assert kwargs.get("user") == "current_user"

        result = router.should_use_tool("What is the password policy?")
        assert result is None

    def test_tool_router_execution(self):
        """Test tool router can execute queries"""
        tools = create_all_tools()
        router = ToolRouter(tools)

        # Test routing to ticket status
        result = router.route_query("Check TICKET-001", ticket_id="TICKET-001")

        assert result["success"] is True
        assert result["type"] == "tool"
        assert result["tool_name"] == "check_ticket_status"
        assert result["result"] is not None

    def test_tool_error_handler(self):
        """Test error handler tracks errors"""
        handler = ToolErrorHandler()

        # Simulate error
        error = ValueError("Test error")
        result = handler.handle_tool_error(
            error=error,
            tool_name="test_tool",
            query="test query",
        )

        assert result["success"] is False
        assert "error" in result

        # Check statistics
        stats = handler.get_error_statistics()
        assert stats["total_errors"] == 1
        assert "test_tool" in stats["errors_by_tool"]


class TestEnhancedAgentIntegration:
    """Integration tests for enhanced agent"""

    @pytest.fixture
    def mock_agent(self):
        """Create agent with mock components"""
        return create_enhanced_agent(
            vector_store_config={
                "collection_name": "test_collection",
                "persist_directory": "data/vector_store/chroma",
            },
            grader_config={
                "grader_type": "mock",
                "confidence_threshold": 0.6,
            },
            generator_config={
                "model_name": "phi3:mini",
                "temperature": 0.1,
            },
            max_iterations=2,
            use_tools=True,
        )

    def test_agent_with_tool_query(self, mock_agent):
        """Test agent handles tool queries"""
        response = mock_agent.invoke("What is the status of TICKET-001?")

        assert "answer" in response
        assert response["confidence"] >= 0.0

        # Check if tool was used
        if "tool_used" in response:
            assert response["tool_used"] is not None

    def test_agent_with_document_query(self, mock_agent):
        """Test agent handles document queries"""
        response = mock_agent.invoke("What is the password policy?")

        assert "answer" in response
        assert "confidence" in response
        assert "iterations_used" in response

    def test_agent_info_complete(self, mock_agent):
        """Test agent info contains all expected fields"""
        info = mock_agent.get_agent_info()

        assert "agent_name" in info
        assert "version" in info
        assert "features" in info
        assert "config" in info
        assert "components" in info

        # Check tools info
        if info["config"]["tools_enabled"]:
            assert "tool_statistics" in info or "tools" in info["components"]


class TestCompleteScenarios:
    """
    End-to-end scenario tests.
    These test complete workflows from question to answer.
    """

    @pytest.fixture
    def agent(self):
        """Create fully configured agent"""
        return create_enhanced_agent(
            grader_config={"grader_type": "mock"},
            max_iterations=3,
            use_tools=True,
        )

    def test_scenario_1_ticket_status_check(self, agent):
        """
        Scenario 1: User asks about ticket status
        Expected: Tool is used, direct answer provided
        """
        question = "What is the status of TICKET-001?"
        response = agent.invoke(question)

        assert response["confidence"] > 0.5
        assert "answer" in response
        assert response["answer"] != ""

        # Should use tool
        summary = response.get("state_summary", {})
        if "tool_was_used" in summary:
            assert summary["tool_was_used"] is True

    def test_scenario_2_document_search_with_good_results(self, agent):
        """
        Scenario 2: User asks about policy (documents exist)
        Expected: Documents found, answer generated
        """
        question = "What is the company password policy?"
        response = agent.invoke(question)

        assert "answer" in response
        assert response["iterations_used"] <= agent.max_iterations

        # Check state
        summary = response.get("state_summary", {})
        assert "documents_found" in summary

    def test_scenario_3_no_documents_with_rewrite(self, agent):
        """
        Scenario 3: Initial search fails, requires rewrite
        Expected: Question rewritten, retry attempted
        """
        question = "How xyz abc?"  # Vague question
        response = agent.invoke(question)

        assert "answer" in response

        # Should attempt rewrites if no documents found
        summary = response.get("state_summary", {})
        if summary.get("documents_found", 0) == 0:
            assert (
                response["iterations_used"] > 0
                or summary.get("needed_rewrite") is True
            )

    def test_scenario_4_iteration_limit(self, agent):
        """
        Scenario 4: Max iterations reached
        Expected: Agent stops at max iterations
        """
        question = "Tell me about something completely unknown"
        response = agent.invoke(question)

        assert response["iterations_used"] <= agent.max_iterations
        assert "answer" in response  # Should still provide some answer

    def test_scenario_5_tool_and_document_combination(self, agent):
        """
        Scenario 5: Query could use both tool and documents
        Expected: Agent makes intelligent routing decision
        """
        question = "Are there any open security incident tickets?"
        response = agent.invoke(question)

        assert response["confidence"] >= 0.0
        assert "answer" in response

        # Either tool or document search should work
        summary = response.get("state_summary", {})
        assert (
            summary.get("tool_was_used") is True
            or summary.get("documents_found", 0) > 0
        )


class TestAgentStatistics:
    """Test statistics tracking across scenarios"""

    def test_edge_statistics_tracking(self):
        """Test edge routing statistics are tracked"""
        agent = create_enhanced_agent(
            grader_config={"grader_type": "mock"},
            max_iterations=2,
        )

        # Run multiple queries
        agent.invoke("Question 1")
        agent.invoke("Question 2")

        info = agent.get_agent_info()
        edge_stats = info.get("edge_statistics", {})

        if edge_stats:
            assert "total_decisions" in edge_stats
            assert edge_stats["total_decisions"] > 0

    def test_tool_usage_statistics(self):
        """Test tool usage is tracked"""
        agent = create_enhanced_agent(
            grader_config={"grader_type": "mock"},
            use_tools=True,
        )

        # Use tools
        agent.invoke("Check TICKET-001")
        agent.invoke("Show open tickets")

        info = agent.get_agent_info()

        if info["config"]["tools_enabled"]:
            tool_stats = info.get("tool_statistics", {})
            assert "tool_usage" in tool_stats or "total_tool_calls" in tool_stats


class TestErrorRecovery:
    """Test error handling and recovery"""

    def test_agent_handles_invalid_config_gracefully(self):
        """Test agent handles invalid configuration"""
        with pytest.raises(Exception):
            # This should fail during build
            agent = create_enhanced_agent(
                vector_store_config={
                    "collection_name": "",  # Invalid
                    "persist_directory": "",  # Invalid
                }
            )

    def test_agent_handles_query_errors(self):
        """Test agent handles errors during query processing"""
        agent = create_enhanced_agent(
            grader_config={"grader_type": "mock"},
            max_iterations=1,
        )

        # Should not crash on edge cases
        response = agent.invoke("")
        assert "error" in response or "answer" in response


# Performance benchmarks
class TestPerformance:
    """Basic performance tests"""

    def test_agent_response_time(self):
        """Test agent responds in reasonable time"""
        import time

        agent = create_enhanced_agent(
            grader_config={"grader_type": "mock"},
            max_iterations=2,
        )

        start = time.time()
        response = agent.invoke("Quick test question")
        duration = time.time() - start

        assert response is not None
        # Should complete within 30 seconds (generous for local LLM)
        assert duration < 30.0

    def test_tool_routing_is_fast(self):
        """Test tool routing doesn't add significant overhead"""
        import time

        tools = create_all_tools()
        router = ToolRouter(tools)

        start = time.time()
        for _ in range(100):
            router.should_use_tool("Test query about tickets")
        duration = time.time() - start

        # 100 routing decisions should be very fast
        assert duration < 1.0  # Under 1 second for 100 decisions


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])