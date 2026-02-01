import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.graph_state import GraphState
from src.agent.nodes.retriever_node import RetrieverNode, RetrieverFactory
from src.agent.nodes.grader_node import GraderNode
from src.agent.nodes.query_rewriter import QueryRewriter
from src.agent.edges import EdgeRouter, create_conditional_edges
from src.agent.nodes.generator_node import GeneratorNode

logger = logging.getLogger(__name__)


class DocuMindAgentBuilder:
    """
    Builds the complete RAG agent with self-correction mechanism.
    """

    def __init__(
            self,
            vector_store_config: Dict[str, Any] = None,
            grader_config: Dict[str, Any] = None,
            generator_config: Dict[str, Any] = None,
            max_iterations: int = 3,
            search_threshold: float = 0.7
    ):
        """
        Initialize the agent builder.

        Args:
            vector_store_config: Configuration for vector store
            grader_config: Configuration for grader
            generator_config: Configuration for answer generator
            max_iterations: Maximum number of iterations for self-correction
            search_threshold: Similarity threshold for document retrieval
        """
        self.vector_store_config = vector_store_config or {
            "collection_name": "documents",
            "persist_directory": "data/vector_store/chroma"
        }

        self.grader_config = grader_config or {
            "grader_type": "hybrid",
            "grading_strategy": "confidence",
            "confidence_threshold": 0.6,
            "use_fallback": True
        }

        self.generator_config = generator_config or {
            "model_name": "phi3:mini",
            "temperature": 0.1
        }

        self.max_iterations = max_iterations
        self.search_threshold = search_threshold

        self.retriever = None
        self.grader = None
        self.rewriter = None
        self.generator = None
        self.edge_router = EdgeRouter()

        self.graph = None
        self.compiled_graph = None

        logger.info(f"Initialized DocuMindAgentBuilder with max_iterations={max_iterations}")

    def build_components(self):
        """
        Build all agent components.
        """
        try:
            self.retriever = RetrieverFactory.create_retriever(
                collection_name=self.vector_store_config["collection_name"],
                persist_directory=self.vector_store_config["persist_directory"],
                search_config={
                    "k": 5,
                    "score_threshold": self.search_threshold,
                    "include_metadata": True
                }
            )
            logger.info("Retriever built successfully")

            self.grader = GraderNode(**self.grader_config)
            logger.info("Grader built successfully")

            self.rewriter = QueryRewriter(model_name="phi3:mini")
            logger.info("Query rewriter built successfully")

            self.generator = GeneratorNode(**self.generator_config)
            logger.info("Answer generator built successfully")

            return True

        except Exception as e:
            logger.error(f"Failed to build agent components: {e}")
            raise

    def build_graph(self) -> StateGraph:
        """
        Build the complete LangGraph with self-correction mechanism.

        Returns:
            Configured StateGraph
        """
        if not all([self.retriever, self.grader, self.rewriter, self.generator]):
            self.build_components()

        workflow = StateGraph(GraphState)

        workflow.add_node("retrieve", self.retriever.as_runnable())
        workflow.add_node("grade", self.grader.as_runnable())
        workflow.add_node("rewrite", self.rewriter.rewrite_question)
        workflow.add_node("generate", self.generator.as_runnable())

        workflow.set_entry_point("retrieve")

        workflow.add_edge("retrieve", "grade")

        workflow.add_conditional_edges(
            "grade",
            self.edge_router.route_to_rewriter,
            {
                "rewrite": "rewrite",
                "generate": "generate",
                "end": END
            }
        )

        workflow.add_conditional_edges(
            "rewrite",
            self.edge_router.route_after_rewrite,
            {
                "retrieve": "retrieve",
                "generate": "generate",
                "end": END
            }
        )

        workflow.add_conditional_edges(
            "generate",
            self.edge_router.route_from_generator,
            {
                END: END,
                "rewrite": "rewrite"
            }
        )

        checkpointer = MemorySaver()

        self.graph = workflow
        self.compiled_graph = workflow.compile(checkpointer=checkpointer)

        logger.info("Graph built successfully with self-correction mechanism")
        return workflow

    def invoke(
            self,
            question: str,
            config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Invoke the agent with a question.

        Args:
            question: User question
            config: Optional configuration for the run

        Returns:
            Agent response with full trace
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
            "metadata": {
                "agent_version": "1.0",
                "max_iterations": self.max_iterations
            },
            "search_history": [],
            "decision_log": [],
            "current_rewrite_index": 0,
        }

        if config:
            initial_state.update(config)

        try:
            logger.info(f"Invoking agent with question: {question}")

            result = self.compiled_graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": "user_session"}}
            )

            edge_stats = self.edge_router.get_edge_statistics()

            iterations_used = result.get("iterations", 0)

            return {
                "answer": result.get(
                    "answer",
                    "I couldn't find an answer to your question."
                ),
                "confidence": float(result.get("confidence", 0.0)),
                "iterations_used": iterations_used,
                "max_iterations": self.max_iterations,
                "search_threshold": self.search_threshold,
                "relevant_documents": result.get("relevant_docs", []),
                "metadata": result.get("metadata", {}),
                "edge_statistics": edge_stats,
                "state_summary": {
                    "question": result.get("question", question),
                    "has_answer": bool(result.get("answer")),
                    "documents_found": len(result.get("documents", [])),
                    "relevant_documents": len(result.get("relevant_docs", [])),
                    "needed_rewrite": result.get("needs_rewrite", False),
                    "rewritten_questions": result.get("rewritten_questions", result.get("metadata", {}).get("rewritten_questions", [])),
                }
            }

        except Exception as e:

            logger.error(f"Agent invocation failed: {e}")

            return {

                "answer": f"An error occurred while processing your question: {str(e)}",
                "confidence": 0.0,
                "iterations_used": initial_state["iterations"],
                "max_iterations": self.max_iterations,
                "search_threshold": self.search_threshold,
                "relevant_documents": [],
                "metadata": {
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                "edge_statistics": self.edge_router.get_edge_statistics(),
                "state_summary": {
                    "question": question,
                    "has_answer": False,
                    "documents_found": 0,
                    "relevant_documents": 0,
                    "needed_rewrite": False,
                    "rewritten_questions": []
                }
            }

    def get_graph_visualization(self) -> str:
        """
        Get a visualization of the graph structure.

        Returns:
            Graph visualization as string
        """
        if self.graph is None:
            return "Graph not built yet"

        visualization = """
        DocuMind Agent Graph Structure:
        ================================

        START → retrieve → grade
                    |
                    v
                (conditional)
                    |
            +-------+-------+-------+
            |       |       |       |
            v       v       v       v
          rewrite generate   end    (if no docs)
            |       |
            v       v
        (conditional) → (if no answer)
            |               |
            v               v
        retrieve         rewrite
            |
            v
          grade
            |
            v
        (repeat until answer found or max iterations)

        Conditional Logic:
        1. From grade:
           - If relevant docs found → generate
           - If no relevant docs & iterations < max → rewrite
           - If max iterations reached → end

        2. From rewrite:
           - If more rewrites available → retrieve
           - If no more rewrites → generate
           - If max iterations → end

        3. From generate:
           - If answer generated → end
           - If no answer & can rewrite → rewrite
           - If max iterations → end
        """

        return visualization

    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about the built agent.

        Returns:
            Dictionary with agent information
        """
        return {
            "agent_name": "DocuMind Agent",
            "version": "1.0",
            "features": [
                "Self-correction mechanism",
                "Question rewriting",
                "Document relevance grading",
                "Fallback strategies",
                "Iteration limiting"
            ],
            "config": {
                "max_iterations": self.max_iterations,
                "search_threshold": self.search_threshold,
                "grader_type": self.grader_config.get("grader_type"),
                "generator_model": self.generator_config.get("model_name")
            },
            "components": {
                "retriever": "ChromaDB with similarity search",
                "grader": f"{self.grader_config.get('grader_type')} with fallback",
                "rewriter": "Query rewriting with phi3:mini",
                "generator": "Answer generation with LLM"
            },
            "edge_statistics": self.edge_router.get_edge_statistics() if self.edge_router else {}
        }


def create_agent(
        vector_store_config: Dict[str, Any] = None,
        grader_config: Dict[str, Any] = None,
        generator_config: Dict[str, Any] = None,
        max_iterations: int = 3
) -> DocuMindAgentBuilder:
    """
    Factory function to create a DocuMind agent.

    Args:
        vector_store_config: Vector store configuration
        grader_config: Grader configuration
        generator_config: Generator configuration
        max_iterations: Maximum iterations for self-correction

    Returns:
        Configured DocuMindAgentBuilder
    """
    agent = DocuMindAgentBuilder(
        vector_store_config=vector_store_config,
        grader_config=grader_config,
        generator_config=generator_config,
        max_iterations=max_iterations
    )

    agent.build_graph()
    return agent