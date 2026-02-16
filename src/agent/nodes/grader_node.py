import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.runnables import RunnableLambda

from src.agent.graph_state import GraphState, StateManager
from src.agent.nodes.robust_grader import RobustGrader

logger = logging.getLogger(__name__)


class GraderNode:
    """
    Document relevance grading node for RAG-based systems.

    Evaluates document relevance to user queries using a multi-layer grading
    strategy with confidence scoring and fallback mechanisms.

    Attributes:
        confidence_threshold: Minimum confidence score for relevance classification
        model_name: Name of the LLM model used for grading
        grader: Configured RobustGrader instance
    """

    def __init__(
        self,
        grader_type: str = "robust",
        confidence_threshold: float = 0.7,
        model_name: str = "phi3:mini",
        **kwargs,
    ):
        """
        Initialize the grader node with grading configuration.

        Args:
            grader_type: Type of grader to use (only 'robust' is supported)
            confidence_threshold: Minimum confidence score (0.0-1.0) for relevance
            model_name: Name of the Ollama model for grading
            **kwargs: Additional configuration parameters (ignored, for compatibility)

        Raises:
            No explicit exceptions, falls back to 'robust' grader for unsupported types
        """
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name

        if grader_type != "robust":
            logger.warning(f"Grader type '{grader_type}' not supported, using 'robust'")

        self.grader = RobustGrader(model_name=model_name)

        logger.info(f"Initialized GraderNode with type=robust, threshold={confidence_threshold}")

    def grade_document(
        self, question: str, document: str, metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Grade a single document for relevance to the question.

        Args:
            question: User query to evaluate against
            document: Document content to grade
            metadata: Optional document metadata (API compatibility, not used by RobustGrader)

        Returns:
            Dictionary containing grading results with confidence scores
        """
        result = self.grader.grade(question, document, metadata)
        return result.to_dict()

    def grade_documents(
        self, question: str, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Grade multiple documents and aggregate results.

        Processes documents in batch, calculates relevance metrics, and returns
        aggregated statistics including average confidence and relevance ratio.

        Args:
            question: User query to evaluate documents against
            documents: List of document strings to grade
            metadatas: Optional list of metadata dictionaries (API compatibility)

        Returns:
            Dictionary containing:
                - individual_results: List of per-document grading results
                - relevant_documents: List of documents passing threshold
                - relevant_count: Number of relevant documents
                - total_count: Total documents graded
                - avg_confidence: Mean confidence across all documents
                - relevance_ratio: Proportion of relevant documents (0.0-1.0)
                - is_any_relevant: Boolean indicating if any document passed
        """
        logger.info(f"[grade_documents] Starting with {len(documents)} documents")

        results = self.grader.grade_batch(question, documents)

        relevant_docs = []
        individual_results = []

        for i, (doc, result) in enumerate(zip(documents, results)):
            result_dict = result.to_dict()
            logger.info(
                f"Doc {i}: LLM_conf={result_dict.get('llm_confidence', 'N/A')}, "
                f"final_conf={result.confidence}, method={result.method}"
            )
            result_dict["document_index"] = i

            doc_preview = doc[:200] + "..." if len(doc) > 200 else doc
            result_dict["document_preview"] = doc_preview

            individual_results.append(result_dict)

            if result.is_relevant(self.confidence_threshold):
                relevant_docs.append(doc)
                logger.debug(
                    f"Doc {i}: RELEVANT (confidence={result.confidence:.2f}, score={result.score.name})"
                )
            else:
                logger.debug(
                    f"Doc {i}: NOT RELEVANT (confidence={result.confidence:.2f}, score={result.score.name})"
                )

        confidences = [r.confidence for r in results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        relevance_ratio = len(relevant_docs) / len(documents) if documents else 0.0

        result_dict = {
            "individual_results": individual_results,
            "relevant_documents": relevant_docs,
            "relevant_count": len(relevant_docs),
            "total_count": len(documents),
            "avg_confidence": float(avg_confidence),
            "relevance_ratio": float(relevance_ratio),
            "is_any_relevant": len(relevant_docs) > 0,
        }

        logger.info(
            f"[grade_documents] Graded {len(documents)} docs, {len(relevant_docs)} relevant, "
            f"avg_confidence={avg_confidence:.2f}"
        )

        for i, (doc, result) in enumerate(zip(documents, results)):
            is_rel = result.is_relevant(self.confidence_threshold)
            logger.info(
                f"Doc {i}: score={result.score.name}, conf={result.confidence:.2f}, is_relevant={is_rel}, threshold={self.confidence_threshold}"
            )
        return result_dict

    def as_runnable(self) -> Callable:
        """
        Convert to LangGraph-compatible runnable function.

        Returns:
            RunnableLambda wrapping the grader function for LangGraph integration
        """

        def grader_function(state: GraphState) -> GraphState:
            """
            LangGraph-compatible grader node function.

            Args:
                state: Current graph state containing documents and iteration info

            Returns:
                Updated state with grading results, relevant documents, and routing flags
            """
            logger.info(f"[Grader Function] Iteration {state.get('iterations', 0)}")
            logger.info(f"[Grader Function] Grading {len(state.get('documents', []))} documents")

            documents = state.get("documents", [])

            if not documents:
                logger.warning("No documents to grade")
                should_rewrite = state["iterations"] < state["max_iterations"]

                return StateManager.update_state(
                    state,
                    relevant_docs=[],
                    confidence=0.0,
                    needs_rewrite=should_rewrite,
                    iterations=state["iterations"] + 1,
                    metadata={
                        **state.get("metadata", {}),
                        "grading_result": {
                            "error": "No documents to grade",
                            "relevant_count": 0,
                            "total_count": 0,
                            "relevance_ratio": 0.0,
                            "is_any_relevant": False,
                        },
                    },
                )

            retrieval_results = state.get("metadata", {}).get("retrieval_results", {})
            metadatas = retrieval_results.get("metadatas", [])

            grading_result = self.grade_documents(
                question=state["question"],
                documents=documents,
                metadatas=metadatas[: len(documents)] if metadatas else None,
            )

            is_any_relevant = grading_result["is_any_relevant"]

            needs_rewrite = not is_any_relevant and state["iterations"] < state["max_iterations"]

            relevant_confidences = [
                r["confidence"]
                for r in grading_result["individual_results"]
                if r.get("relevant", False)
            ]
            logger.info(f"[Grader Function] Relevant docs found: {len(relevant_confidences)}")
            if relevant_confidences:
                confidence = float(max(relevant_confidences))
                logger.info(f"[Grader Function] Confidences: {relevant_confidences}")
            else:
                confidence = float(grading_result["avg_confidence"])

            updated_state = StateManager.update_state(
                state,
                relevant_docs=grading_result["relevant_documents"],
                confidence=confidence,
                needs_rewrite=needs_rewrite,
                increment_iterations=True,
                metadata={
                    **state.get("metadata", {}),
                    "grading_result": grading_result,
                    "current_iteration": state["iterations"],
                },
            )

            history_entry = {
                "role": "system",
                "action": "grading",
                "content": f"Graded {len(documents)} documents: {grading_result['relevant_count']} relevant",
                "confidence": grading_result["avg_confidence"],
                "details": {
                    "relevant_count": grading_result["relevant_count"],
                    "total_count": grading_result["total_count"],
                    "needs_rewrite": needs_rewrite,
                    "relevance_ratio": grading_result["relevance_ratio"],
                },
            }

            updated_state = StateManager.add_to_history(updated_state, history_entry)

            logger.info(
                f"[Grader Function] Found {grading_result['relevant_count']} relevant documents"
            )
            logger.info(
                f"[Grader Function] Confidence: {confidence:.2f}, Needs rewrite: {needs_rewrite}"
            )

            return updated_state

        return RunnableLambda(grader_function)


def grader_node(state: GraphState) -> GraphState:
    """
    Standalone grader node function for LangGraph.

    Factory function that creates a GraderNode instance and executes
    the grading pipeline. Suitable for direct use in LangGraph graphs.

    Args:
        state: Current graph state with documents and question

    Returns:
        Updated state containing grading results and routing decisions
    """
    grader = GraderNode()
    return grader.as_runnable().invoke(state)
