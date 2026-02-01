"""
Enhanced agent builder with external tools integration.
This is the complete graph implementation for Day 10.
"""

import logging
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.graph_state import GraphState
from src.agent.nodes.retriever_node import RetrieverNode, RetrieverFactory
from src.agent.nodes.grader_node import GraderNode
from src.agent.nodes.query_rewriter import QueryRewriter
from src.agent.edges import EdgeRouter
from src.agent.nodes.generator_node import GeneratorNode
from src.tools.document_tool import (
    create_all_tools,
    ToolRouter,
    ToolErrorHandler,
)

logger = logging.getLogger(__name__)


class EnhancedDocuMindAgent:
    """
    Enhanced RAG agent with external tools integration.
    Complete implementation for Day 10.
    """

    def __init__(
        self,
        vector_store_config: Dict[str, Any] = None,
        grader_config: Dict[str, Any] = None,
        generator_config: Dict[str, Any] = None,
        max_iterations: int = 3,
        search_threshold: float = 0.7,
        use_tools: bool = True,
    ):
        """
        Initialize enhanced agent.

        Args:
            vector_store_config: Configuration for vector store
            grader_config: Configuration for grader
            generator_config: Configuration for answer generator
            max_iterations: Maximum number of iterations
            search_threshold: Similarity threshold for retrieval
            use_tools: Whether to enable external tools
        """
        self.vector_store_config = vector_store_config or {
            "collection_name": "documents",
            "persist_directory": "data/vector_store/chroma",
        }

        self.grader_config = grader_config or {
            "grader_type": "hybrid",
            "grading_strategy": "confidence",
            "confidence_threshold": 0.6,
            "use_fallback": True,
        }

        self.generator_config = generator_config or {
            "model_name": "phi3:mini",
            "temperature": 0.1,
        }

        self.max_iterations = max_iterations
        self.search_threshold = search_threshold
        self.use_tools = use_tools

        # Core components
        self.retriever = None
        self.grader = None
        self.rewriter = None
        self.generator = None
        self.edge_router = EdgeRouter()

        # Tool components
        self.tools = []
        self.tool_router = None
        self.tool_error_handler = ToolErrorHandler()

        # Graph
        self.graph = None
        self.compiled_graph = None

        logger.info(
            f"EnhancedDocuMindAgent initialized (tools={'enabled' if use_tools else 'disabled'})"
        )

    def build_components(self):
        """Build all agent components including tools."""
        try:
            # Build retriever
            self.retriever = RetrieverFactory.create_retriever(
                collection_name=self.vector_store_config["collection_name"],
                persist_directory=self.vector_store_config["persist_directory"],
                search_config={
                    "k": 5,
                    "score_threshold": self.search_threshold,
                    "include_metadata": True,
                },
            )
            logger.info("Retriever built successfully")

            # Build grader
            self.grader = GraderNode(**self.grader_config)
            logger.info("Grader built successfully")

            # Build rewriter
            self.rewriter = QueryRewriter(model_name="phi3:mini")
            logger.info("Query rewriter built successfully")

            # Build generator
            self.generator = GeneratorNode(**self.generator_config)
            logger.info("Answer generator built successfully")

            # Build tools if enabled
            if self.use_tools:
                vector_store = None
                try:
                    from src.vector_store.chroma_db import ChromaDBVectorStore

                    vector_store = ChromaDBVectorStore(
                        collection_name=self.vector_store_config["collection_name"],
                        persist_directory=self.vector_store_config[
                            "persist_directory"
                        ],
                    )
                except Exception as e:
                    logger.warning(f"Could not initialize vector store for tools: {e}")

                self.tools = create_all_tools(vector_store)
                self.tool_router = ToolRouter(self.tools, vector_store)
                logger.info(f"Built {len(self.tools)} tools")

            return True

        except Exception as e:
            logger.error(f"Failed to build agent components: {e}")
            raise

    def _tool_routing_node(self, state: GraphState) -> GraphState:
        """
        Node that routes to tools or document search.
        """
        logger.info("[Tool Routing] Analyzing query")

        question = state.get("question", "")

        # Check if we should use a tool
        if self.tool_router:
            routing_result = self.tool_router.route_query(question)

            if routing_result["success"]:
                if routing_result["type"] == "tool":
                    # Tool was used
                    logger.info(
                        f"[Tool Routing] Used tool: {routing_result['tool_name']}"
                    )

                    # Update state with tool result
                    state["tool_result"] = routing_result["result"]
                    state["tool_used"] = routing_result["tool_name"]
                    state["skip_retrieval"] = True

                    if routing_result["result"]:
                        state["answer"] = str(routing_result["result"])
                        state["confidence"] = 0.9

                    return state

        # No tool used - proceed with normal document retrieval
        logger.info("[Tool Routing] No tool used, proceeding to retrieval")
        state["skip_retrieval"] = False
        state["tool_used"] = None

        return state

    def build_graph(self) -> StateGraph:
        """
        Build complete LangGraph with tools integration.
        This is Task 10.1: Complete graph connection.
        """
        if not all([self.retriever, self.grader, self.rewriter, self.generator]):
            self.build_components()

        workflow = StateGraph(GraphState)

        # Add all nodes
        if self.use_tools:
            workflow.add_node("tool_routing", self._tool_routing_node)

        workflow.add_node("retrieve", self.retriever.as_runnable())
        workflow.add_node("grade", self.grader.as_runnable())
        workflow.add_node("rewrite", self.rewriter.rewrite_question)
        workflow.add_node("generate", self.generator.as_runnable())

        # Set entry point
        if self.use_tools:
            workflow.set_entry_point("tool_routing")

            # From tool_routing, either skip to generate or go to retrieve
            workflow.add_conditional_edges(
                "tool_routing",
                self._route_from_tool_routing,
                {"retrieve": "retrieve", "generate": "generate"},
            )
        else:
            workflow.set_entry_point("retrieve")

        # Standard edges
        workflow.add_edge("retrieve", "grade")

        workflow.add_conditional_edges(
            "grade",
            self.edge_router.route_to_rewriter,
            {"rewrite": "rewrite", "generate": "generate", "end": END},
        )

        workflow.add_conditional_edges(
            "rewrite",
            self.edge_router.route_after_rewrite,
            {"retrieve": "retrieve", "generate": "generate", "end": END},
        )

        workflow.add_conditional_edges(
            "generate",
            self.edge_router.route_from_generator,
            {END: END, "rewrite": "rewrite"},
        )

        # Compile
        checkpointer = MemorySaver()
        self.graph = workflow
        self.compiled_graph = workflow.compile(checkpointer=checkpointer)

        logger.info("Complete graph built with tools integration")
        return workflow

    def _route_from_tool_routing(self, state: GraphState) -> str:
        """Route from tool_routing node"""
        if state.get("tool_used") and state.get("tool_result"):
            if state.get("answer"):
                return "generate"
            return "generate"
        return "retrieve"

    def invoke(self, question: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Invoke the enhanced agent.

        Args:
            question: User question
            config: Optional configuration

        Returns:
            Complete response with metadata
        """
        if self.compiled_graph is None:
            self.build_graph()

        initial_state = {
            "question": question,
            "search_query": question,
            "iterations": 0,
            "max_iterations": self.max_iterations,
            "search_threshold": self.search_threshold,
            "history": [],
            "documents": [],
            "relevant_docs": [],
            "confidence": 0.0,
            "needs_rewrite": False,
            "rewritten_questions": [],
            "current_rewrite_index": 0,
            "search_history": [],
            "decision_log": [],
            "tool_used": None,
            "tool_result": None,
            "skip_retrieval": False,
            "metadata": {
                "agent_version": "2.0-enhanced",
                "max_iterations": self.max_iterations,
                "tools_enabled": self.use_tools,
            },
        }

        if config:
            initial_state.update(config)

        try:
            logger.info(f"Invoking enhanced agent with question: {question}")

            result = self.compiled_graph.invoke(
                initial_state, config={"configurable": {"thread_id": "user_session"}}
            )

            # Gather statistics
            edge_stats = self.edge_router.get_edge_statistics()
            tool_stats = (
                self.tool_router.get_usage_statistics()
                if self.tool_router
                else {"tool_usage": {}, "total_tool_calls": 0}
            )

            return {
                "answer": result.get(
                    "answer", "I couldn't find an answer to your question."
                ),
                "confidence": float(result.get("confidence", 0.0)),
                "iterations_used": result.get("iterations", 0),
                "max_iterations": self.max_iterations,
                "tool_used": result.get("tool_used"),
                "tool_result": result.get("tool_result"),
                "relevant_documents": result.get("relevant_docs", []),
                "metadata": result.get("metadata", {}),
                "edge_statistics": edge_stats,
                "tool_statistics": tool_stats,
                "state_summary": {
                    "question": result.get("question", question),
                    "has_answer": bool(result.get("answer")),
                    "documents_found": len(result.get("documents", [])),
                    "relevant_documents": len(result.get("relevant_docs", [])),
                    "needed_rewrite": result.get("needs_rewrite", False),
                    "rewritten_questions": result.get("rewritten_questions", []),
                    "tool_was_used": result.get("tool_used") is not None,
                },
            }

        except Exception as e:
            logger.error(f"Enhanced agent invocation failed: {e}")

            # Record error
            if self.use_tools:
                self.tool_error_handler.handle_tool_error(
                    e, "agent", question, fallback_to_search=False
                )

            return {
                "answer": f"An error occurred: {str(e)}",
                "confidence": 0.0,
                "iterations_used": 0,
                "tool_used": None,
                "error": str(e),
                "metadata": {"error_type": type(e).__name__},
            }

    def get_agent_info(self) -> Dict[str, Any]:
        """Get comprehensive agent information"""
        base_info = {
            "agent_name": "DocuMind Enhanced Agent",
            "version": "2.0",
            "features": [
                "Self-correction mechanism",
                "Question rewriting",
                "Document relevance grading",
                "Fallback strategies",
                "Iteration limiting",
                "External tools integration",
                "Tool routing",
                "Error recovery",
            ],
            "config": {
                "max_iterations": self.max_iterations,
                "search_threshold": self.search_threshold,
                "grader_type": self.grader_config.get("grader_type"),
                "generator_model": self.generator_config.get("model_name"),
                "tools_enabled": self.use_tools,
            },
            "components": {
                "retriever": "ChromaDB with similarity search",
                "grader": f"{self.grader_config.get('grader_type')} with fallback",
                "rewriter": "Query rewriting with phi3:mini",
                "generator": "Answer generation with LLM",
                "tools": f"{len(self.tools)} tools available" if self.tools else "No tools",
            },
        }

        if self.use_tools and self.tool_router:
            base_info["tool_statistics"] = self.tool_router.get_usage_statistics()
            base_info["error_statistics"] = self.tool_error_handler.get_error_statistics()

        base_info["edge_statistics"] = (
            self.edge_router.get_edge_statistics() if self.edge_router else {}
        )

        return base_info

    def reset_statistics(self):
        """Reset all statistics"""
        if self.edge_router:
            self.edge_router.edge_history.clear()
        if self.tool_error_handler:
            self.tool_error_handler.reset_stats()
        logger.info("All statistics reset")


def create_enhanced_agent(
    vector_store_config: Dict[str, Any] = None,
    grader_config: Dict[str, Any] = None,
    generator_config: Dict[str, Any] = None,
    max_iterations: int = 3,
    use_tools: bool = True,
) -> EnhancedDocuMindAgent:
    """
    Factory function to create enhanced agent.

    Args:
        vector_store_config: Vector store configuration
        grader_config: Grader configuration
        generator_config: Generator configuration
        max_iterations: Maximum iterations
        use_tools: Enable external tools

    Returns:
        Configured enhanced agent
    """
    agent = EnhancedDocuMindAgent(
        vector_store_config=vector_store_config,
        grader_config=grader_config,
        generator_config=generator_config,
        max_iterations=max_iterations,
        use_tools=use_tools,
    )

    agent.build_graph()
    return agent