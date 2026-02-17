import logging
from typing import Any, Dict

import chromadb
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agent.edges import EdgeRouter, should_rewrite_again
from src.agent.graph_state import GraphState
from src.agent.nodes.generator_node import GeneratorNode
from src.agent.nodes.grader_node import GraderNode
from src.agent.nodes.query_rewriter import QueryRewriter
from src.agent.nodes.retriever_node import RetrieverNode
from src.tools.document_tool import ToolErrorHandler, ToolRouter, create_all_tools
from src.vector_store.chroma_db import ChromaDBVectorStore
from src.vector_store.embeddings_manager import EmbeddingManager

logger = logging.getLogger(__name__)

_chroma_client = None


class DocuMindAgent:
    """
    RAG agent with external tools integration.

    Implements a complete RAG pipeline with self-correction, query rewriting,
    robust document grading, and optional external tool integration.

    Attributes:
        vector_store_config: Configuration for vector store connection
        grader_config: Configuration for document grading
        generator_config: Configuration for answer generation
        max_iterations: Maximum query rewrite iterations
        search_threshold: Minimum similarity threshold for retrieval
        use_tools: Whether to enable external tools
        retriever: Document retriever instance
        grader: Document grader instance
        rewriter: Query rewriter instance
        generator: Answer generator instance
        edge_router: Graph edge routing logic
        tools: List of available external tools
        tool_router: Tool routing and execution logic
        tool_error_handler: Tool error handling
        graph: LangGraph StateGraph instance
        compiled_graph: Compiled graph with checkpointing
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
        Initialize the RAG agent.

        Args:
            vector_store_config: Vector store configuration (collection, persist_dir)
            grader_config: Grader configuration (type, threshold, model)
            generator_config: Generator configuration (model, temperature)
            max_iterations: Maximum number of rewrite iterations
            search_threshold: Minimum similarity score for document retrieval
            use_tools: Enable external tool integration
        """
        self.vector_store_config = vector_store_config or {
            "collection_name": "documents",
            "persist_directory": "data/vector_store/chroma",
        }

        self.grader_config = grader_config or {
            "grader_type": "robust",
            "confidence_threshold": 0.6,
            "model_name": "phi3:mini",
        }

        self.generator_config = generator_config or {
            "model_name": "llama3.1:8b",
            "temperature": 0.1,
        }

        self.max_iterations = max_iterations
        self.search_threshold = search_threshold
        self.use_tools = use_tools

        self.retriever = None
        self.grader = None
        self.rewriter = None
        self.generator = None
        self.edge_router = EdgeRouter()

        self.tools = []
        self.tool_router = None
        self.tool_error_handler = ToolErrorHandler()

        self.graph = None
        self.compiled_graph = None

        self._embedding_cache = {}
        self._cache_max_size = 100

        logger.info(f"DocuMindAgent initialized (tools={'enabled' if use_tools else 'disabled'})")

    def get_chroma_client(self):
        global _chroma_client

        if _chroma_client is None:
            path = "data/vector_store/chroma"
            _chroma_client = chromadb.PersistentClient(path=str(path))

        return _chroma_client

    def build_components(self):
        """
        Build all agent components including vector store, retriever, grader,
        rewriter, generator, and optional tools.

        Returns:
            True if all components built successfully

        Raises:
            Exception: If component initialization fails
        """
        try:
            embedding_manager = EmbeddingManager()
            embedding_function = embedding_manager.chroma_embedding_function()
            logger.info("Embedding function created")

            vector_store = ChromaDBVectorStore(
                collection_name=self.vector_store_config["collection_name"],
                persist_directory=self.vector_store_config["persist_directory"],
                embedding_function=embedding_function,
                reset_on_start=False,
                client=self.get_chroma_client(),
            )

            logger.info(f"Vector store created with embedding_function: {vector_store.embedding_function is not None}")

            self.retriever = RetrieverNode(
                vector_store=vector_store,
                search_config={
                    "k": 3,
                    "score_threshold": self.search_threshold,
                    "include_metadata": True,
                },
            )
            logger.info("Retriever built successfully")

            self.grader = GraderNode(**self.grader_config)
            logger.info("Grader built successfully")

            self.rewriter = QueryRewriter(model_name="mistral:7b")
            logger.info("Query rewriter built successfully")

            self.generator = GeneratorNode(**self.generator_config)
            logger.info("Answer generator built successfully")

            if self.use_tools:
                self.tools = create_all_tools(vector_store)
                self.tool_router = ToolRouter(self.tools, vector_store)
                logger.info(f"Built {len(self.tools)} tools")

            return True

        except Exception as e:
            logger.error(f"Failed to build agent components: {e}")
            raise

    def _tool_routing_node(self, state: GraphState) -> GraphState:
        """
        Route queries to appropriate tools or document search.

        Analyzes the query and either executes a matching tool or
        proceeds to document retrieval.

        Args:
            state: Current graph state containing the question

        Returns:
            Updated state with tool results or routing flags
        """
        logger.info("[Tool Routing] Analyzing query")

        question = state.get("question", "")

        cache_key = f"query:{question}"
        if cache_key not in self._embedding_cache:
            if self.tool_router:
                routing_result = self.tool_router.route_query(question)
                if len(self._embedding_cache) >= self._cache_max_size:
                    oldest = next(iter(self._embedding_cache))
                    del self._embedding_cache[oldest]
                self._embedding_cache[cache_key] = True

        if self.tool_router:
            routing_result = self.tool_router.route_query(question)

            if routing_result["type"] == "error":
                logger.error(f"[Tool Routing] Tool error: {routing_result.get('error')}")
                state["tool_error"] = routing_result.get("error")
                state["skip_retrieval"] = False

            elif routing_result["success"]:
                if routing_result["type"] == "tool":
                    logger.info(f"[Tool Routing] Used tool: {routing_result['tool_name']}")

                    state["tool_result"] = routing_result["result"]
                    state["tool_used"] = routing_result["tool_name"]
                    state["skip_retrieval"] = True

                    if routing_result["result"]:
                        state["answer"] = str(routing_result["result"])
                        state["confidence"] = 0.9

                    return state

            else:
                logger.warning(f"[Tool Routing] Unknown result type: {routing_result.get('type')}")
                state["skip_retrieval"] = False

        logger.info("[Tool Routing] No tool used, proceeding to retrieval")
        state["skip_retrieval"] = False
        state["tool_used"] = None
        state["tool_error"] = None

        return state

    def build_graph(self) -> StateGraph:
        """
        Build complete LangGraph workflow with tool integration.

        Constructs the state graph with nodes for tool routing, retrieval,
        grading, rewriting, and generation, connected by conditional edges.

        Returns:
            Configured StateGraph instance
        """
        if not all([self.retriever, self.grader, self.rewriter, self.generator]):
            self.build_components()

        workflow = StateGraph(GraphState)

        if self.use_tools:
            workflow.add_node("tool_routing", self._tool_routing_node)

        workflow.add_node("retrieve", self.retriever.as_runnable())
        workflow.add_node("grade", self.grader.as_runnable())
        workflow.add_node("rewrite", self.rewriter.rewrite_question)
        workflow.add_node("generate", self.generator.as_runnable())

        if self.use_tools:
            workflow.set_entry_point("tool_routing")
            workflow.add_conditional_edges(
                "tool_routing",
                self._route_from_tool_routing,
                {"retrieve": "retrieve", "generate": "generate"},
            )
        else:
            workflow.set_entry_point("retrieve")

        workflow.add_edge("retrieve", "grade")

        workflow.add_conditional_edges(
            "grade",
            self.edge_router.route_to_rewriter,
            {"rewrite": "rewrite", "generate": "generate", "end": END},
        )

        workflow.add_conditional_edges(
            "rewrite",
            should_rewrite_again,
            {"retrieve": "retrieve", "generate": "generate"},
        )

        workflow.add_conditional_edges(
            "generate",
            self.edge_router.route_from_generator,
            {END: END, "rewrite": "rewrite"},
        )

        checkpointer = MemorySaver()
        self.graph = workflow
        self.compiled_graph = workflow.compile(checkpointer=checkpointer)

        logger.info("Complete graph built successfully")
        return workflow

    def _route_from_tool_routing(self, state: GraphState) -> str:
        """
        Determine next node after tool routing.

        Routes to "generate" if tool succeeded, "retrieve" otherwise.

        Args:
            state: Current graph state

        Returns:
            Next node name ("retrieve" or "generate")
        """
        if state.get("tool_error"):
            return "retrieve"

        if state.get("tool_used") and state.get("tool_result"):
            result = state["tool_result"]
            if isinstance(result, str) and result.startswith("Error"):
                logger.warning(f"Tool returned error string: {result[:100]}")
                return "retrieve"
            return "generate"
        return "retrieve"

    def invoke(self, question: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the RAG pipeline with the given question.

        Initializes state, executes the compiled graph, and returns
        the final answer with metadata and statistics.

        Args:
            question: User question to answer
            config: Optional configuration overrides

        Returns:
            Dictionary containing:
                - answer: Generated answer string
                - confidence: Answer confidence score
                - iterations_used: Number of rewrite iterations
                - tool_used: Name of tool used (if any)
                - relevant_documents: List of relevant document strings
                - metadata: Execution metadata
                - state_summary: Summary of execution state
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
            "answer": None,
            "metadata": {
                "agent_version": "3.1-robust-grader",
                "max_iterations": self.max_iterations,
                "tools_enabled": self.use_tools,
            },
        }

        if config:
            initial_state.update(config)

        try:
            logger.info(f"Invoking agent with question: {question}")

            result = self.compiled_graph.invoke(initial_state, config={"configurable": {"thread_id": "user_session"}})

            if result is None:
                logger.error("Graph returned None result")
                return {
                    "answer": "Error: Graph failed to return state",
                    "confidence": 0.0,
                    "error": "Null result from graph",
                }

            answer = result.get("answer")

            if not answer or not answer.strip():
                logger.warning("No answer generated, using fallback")
                answer = "I couldn't find a relevant answer to your question. " "The search didn't yield sufficient information."

            logger.info(f"[Agent Invoke] Final state keys: {list(result.keys())}")

            edge_stats = self.edge_router.get_edge_statistics()
            tool_stats = self.tool_router.get_usage_statistics() if self.tool_router else {"tool_usage": {}, "total_tool_calls": 0}

            grading_result = result.get("metadata", {}).get("grading_result", {})

            return {
                "answer": answer,
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
                    "relevant_documents": grading_result.get("relevant_count", len(result.get("relevant_docs", []))),
                    "needed_rewrite": result.get("needs_rewrite", False),
                    "rewritten_questions": result.get("rewritten_questions", []),
                    "tool_was_used": result.get("tool_used") is not None,
                    "grading_confidence": float(grading_result.get("avg_confidence", result.get("confidence", 0.0))),
                    "relevance_ratio": grading_result.get("relevance_ratio", 0.0),
                },
            }

        except Exception as e:
            logger.error(f"Agent invocation failed: {e}", exc_info=True)

            if self.use_tools:
                self.tool_error_handler.handle_tool_error(e, "agent", question, fallback_to_search=False)

            return {
                "answer": f"An error occurred: {str(e)}",
                "confidence": 0.0,
                "iterations_used": 0,
                "tool_used": None,
                "error": str(e),
                "metadata": {"error_type": type(e).__name__},
            }

    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get comprehensive agent information and configuration.

        Returns:
            Dictionary with agent name, version, features, and component details
        """
        return {
            "agent_name": "DocuMind Agent",
            "version": "3.1",
            "features": [
                "Self-correction mechanism",
                "Question rewriting",
                "Robust document relevance grading (v2)",
                "Triple-layer fallback strategies",
                "Iteration limiting",
                "External tools integration",
            ],
            "config": {
                "max_iterations": self.max_iterations,
                "search_threshold": self.search_threshold,
                "grader_type": "robust",
                "generator_model": self.generator_config.get("model_name"),
                "tools_enabled": self.use_tools,
            },
            "components": {
                "retriever": "ChromaDB with similarity search",
                "grader": "RobustGrader (LLM + Semantic + Keyword)",
                "rewriter": "Query rewriting with mistral:7b",
                "generator": f"Answer generation with {self.generator_config.get('model_name')}",
                "tools": f"{len(self.tools)} tools available" if self.tools else "No tools",
            },
        }

    def reset_statistics(self):
        """Reset all execution statistics and history."""
        if self.edge_router:
            self.edge_router.edge_history.clear()
        if self.tool_error_handler:
            self.tool_error_handler.reset_stats()
        logger.info("All statistics reset")


def create_agent(
    vector_store_config: Dict[str, Any] = None,
    grader_config: Dict[str, Any] = None,
    generator_config: Dict[str, Any] = None,
    max_iterations: int = 3,
    use_tools: bool = True,
) -> DocuMindAgent:
    """
    Factory function to create and initialize an agent.

    Args:
        vector_store_config: Vector store configuration
        grader_config: Grader configuration
        generator_config: Generator configuration
        max_iterations: Maximum rewrite iterations
        use_tools: Enable tool integration

    Returns:
        Initialized DocuMindAgent with built graph
    """
    agent = DocuMindAgent(
        vector_store_config=vector_store_config,
        grader_config=grader_config,
        generator_config=generator_config,
        max_iterations=max_iterations,
        use_tools=use_tools,
    )

    agent.build_graph()
    return agent
