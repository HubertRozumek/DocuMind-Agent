import logging
from typing import Dict, Any, List, Optional, Callable
from langchain_core.runnables import RunnableLambda

from src.agent.graph_state import GraphState, StateManager
from src.agent.nodes.grader_model import BaseGraderModel, GraderFactory, HybridGrader
from src.agent.nodes.grader_prompts import PromptFactory, GradingStrategy
from src.agent.nodes.grader_fallback import FallbackManager

logger = logging.getLogger(__name__)


class GraderNode:
    """
    Node that assesses document relevance to the user's question.
    Integrates LLM grading with fallback strategies.
    """

    def __init__(
            self,
            grader_type: str = "hybrid",
            grading_strategy: GradingStrategy = GradingStrategy.CONFIDENCE,
            use_fallback: bool = True,
            confidence_threshold: float = 0.6,
            **kwargs
    ):
        """
        Initialize the grader node.

        Args:
            grader_type: Type of grader to use ('ollama', 'transformers', 'llama_cpp', 'mock', 'hybrid')
            grading_strategy: Strategy for grading (binary, confidence, multi_criteria, reasoning)
            use_fallback: Whether to use fallback strategies
            confidence_threshold: Threshold for considering a document relevant
            **kwargs: Additional configuration for the grader
        """
        self.grader_type = grader_type
        self.grading_strategy = grading_strategy
        self.use_fallback = use_fallback
        self.confidence_threshold = confidence_threshold
        self.config = kwargs

        self.grader = self._initialize_grader()

        self.fallback_manager = FallbackManager() if use_fallback else None

        self.prompt_factory = PromptFactory()

        logger.info(f"Initialized GraderNode with type={grader_type}, strategy={grading_strategy}")

    def _initialize_grader(self) -> BaseGraderModel:
        """
        Initialize the grader model based on configuration.
        """
        try:
            if self.grader_type == "hybrid":
                return HybridGrader(**self.config)
            else:
                return GraderFactory.create_grader(self.grader_type, **self.config)
        except Exception as e:
            logger.error(f"Failed to initialize grader {self.grader_type}: {e}")
            logger.warning("Falling back to mock grader")
            return GraderFactory.create_grader("mock", **self.config)

    def _create_grading_prompt(
            self,
            question: str,
            document: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a prompt for grading based on the selected strategy.

        Args:
            question: User question
            document: Document text to evaluate
            metadata: Optional document metadata

        Returns:
            Formatted prompt for the grader
        """
        if self.grading_strategy == GradingStrategy.BINARY:
            prompt_template = self.prompt_factory.create_binary_grading_prompt()
        elif self.grading_strategy == GradingStrategy.CONFIDENCE:
            prompt_template = self.prompt_factory.create_confidence_grading_prompt()
        elif self.grading_strategy == GradingStrategy.MULTI_CRITERIA:
            prompt_template = self.prompt_factory.create_multi_criteria_prompt()
        elif self.grading_strategy == GradingStrategy.REASONING:
            prompt_template = self.prompt_factory.create_reasoning_prompt()
        else:
            prompt_template = self.prompt_factory.create_binary_grading_prompt()

        prompt = prompt_template.user_template.format(
            question=question,
            document=document,
            metadata=metadata or {}
        )

        if hasattr(self.grader, 'needs_system_prompt') and self.grader.needs_system_prompt:
            full_prompt = f"{prompt_template.system_template}\n\n{prompt}"
        else:
            full_prompt = prompt

        return full_prompt

    def grade_document(
            self,
            question: str,
            document: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Grade a single document for relevance to the question.

        Args:
            question: User question
            document: Document text to evaluate
            metadata: Optional document metadata

        Returns:
            Dictionary with grading results
        """
        try:
            prompt = self._create_grading_prompt(question, document, metadata)

            llm_response = self.grader.grade(prompt)

            parsed_result = self.grader.validate_response(
                llm_response,
                expected_format="JSON" if self.grading_strategy != GradingStrategy.BINARY else "YES/NO"
            )

            if (
                    self.fallback_manager
                    and parsed_result.get("confidence", 0) < self.confidence_threshold * 0.7
            ):
                fallback_result = self.fallback_manager.execute_fallback(
                    question=question,
                    document=document,
                    metadata=metadata,
                    llm_grade=parsed_result
                )

                if fallback_result["fallback_used"]:
                    logger.info(f"Used fallback: {fallback_result.get('fallback_method')}")
                    parsed_result.update(fallback_result)

            if "confidence" not in parsed_result:
                parsed_result["confidence"] = 0.5

            is_relevant = parsed_result.get("relevant", False)
            confidence = parsed_result.get("confidence", 0.0)

            parsed_result["is_relevant"] = is_relevant and confidence >= self.confidence_threshold
            parsed_result["final_confidence"] = confidence

            return parsed_result

        except Exception as e:
            logger.error(f"Error grading document: {e}")

            from src.agent.nodes.grader_fallback import KeywordFallbackGrader
            keyword_grader = KeywordFallbackGrader(min_keyword_match=1)
            fallback_result = keyword_grader.grade(question, document, metadata)

            return {
                "relevant": fallback_result.relevant,
                "confidence": fallback_result.confidence,
                "reason": f"Error occurred, used keyword fallback: {str(e)}",
                "is_relevant": fallback_result.relevant and fallback_result.confidence >= 0.5,
                "final_confidence": fallback_result.confidence,
                "fallback_used": True,
                "error": str(e)
            }

    def grade_documents(
            self,
            question: str,
            documents: List[str],
            metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Grade multiple documents for relevance.

        Args:
            question: User question
            documents: List of document texts
            metadatas: Optional list of metadata dictionaries

        Returns:
            Dictionary with grading results for all documents
        """
        if metadatas is None:
            metadatas = [{} for _ in documents]

        results = []
        relevant_docs = []

        for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
            try:
                result = self.grade_document(question, doc, metadata)
                result["document_index"] = i
                result["document_preview"] = doc[:200] + "..." if len(doc) > 200 else doc

                results.append(result)

                if result.get("is_relevant", False):
                    relevant_docs.append(doc)

            except Exception as e:
                logger.error(f"Error grading document {i}: {e}")
                results.append({
                    "document_index": i,
                    "relevant": False,
                    "confidence": 0.0,
                    "reason": f"Error during grading: {str(e)}",
                    "is_relevant": False,
                    "error": str(e)
                })

        if results:
            confidences = [r.get("confidence", 0.0) for r in results]
            relevant_count = len([r for r in results if r.get("is_relevant", False)])

            overall_result = {
                "individual_results": results,
                "relevant_documents": relevant_docs,
                "relevant_count": relevant_count,
                "total_count": len(documents),
                "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
                "relevance_ratio": relevant_count / len(documents) if documents else 0.0,
                "is_any_relevant": relevant_count > 0
            }
        else:
            overall_result = {
                "individual_results": [],
                "relevant_documents": [],
                "relevant_count": 0,
                "total_count": 0,
                "avg_confidence": 0.0,
                "relevance_ratio": 0.0,
                "is_any_relevant": False
            }

        return overall_result

    def as_runnable(self) -> Callable:
        """
        Convert the grader to a LangGraph compatible runnable.

        Returns:
            Function that takes a state and returns updated state
        """

        def grader_function(state: GraphState) -> GraphState:
            """
            Grader function for LangGraph.

            Args:
                state: Current graph state

            Returns:
                Updated state with grading results
            """
            logger.info(f"[Grader Function] Iteration {state['iterations']}")
            logger.info(f"[Grader Function] Grading {len(state.get('documents', []))} documents")

            documents = state.get("documents", [])

            if not documents:
                logger.warning("No documents to grade")
                should_rewrite = (state["iterations"] < state["max_iterations"])
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
                            "relevant_count": 0
                        }
                    }
                )

            metadatas = state.get("metadata", {}).get("retrieval_results", {}).get("metadatas", [])

            grading_result = self.grade_documents(
                question=state["question"],
                documents=documents,
                metadatas=metadatas[:len(documents)] if metadatas else None
            )

            needs_rewrite = (
                    not grading_result["is_any_relevant"] and
                    state["iterations"] < state["max_iterations"]
            )

            relevant = [
                r.get("final_confidence", 0.0)
                for r in grading_result["individual_results"]
                if r.get("is_relevant")
            ]
            confidence = max(relevant) if relevant else grading_result["avg_confidence"]

            updated_state = StateManager.update_state(
                state,
                relevant_docs=grading_result["relevant_documents"],
                confidence=confidence,
                needs_rewrite=needs_rewrite,
                increment_iterations=True,
                metadata={
                    **state.get("metadata", {}),
                    "grading_result": grading_result,
                    "current_iteration": state["iterations"]
                }
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
                    "relevance_ratio": grading_result["relevance_ratio"]
                }
            }

            updated_state = StateManager.add_to_history(updated_state, history_entry)

            logger.info(f"[Grader Function] Found {grading_result['relevant_count']} relevant documents")
            logger.info(f"[Grader Function] Needs rewrite: {needs_rewrite}")

            return updated_state

        return RunnableLambda(grader_function)


def grader_node(state: GraphState) -> GraphState:
    """
    Node function for document grading.

    Args:
        state: Current graph state

    Returns:
        Updated state
    """
    grader = GraderNode()
    return grader.as_runnable().invoke(state)