import logging
from datetime import datetime
from typing import Any, Callable, Dict, Literal

from langgraph.graph import END

from src.agent.graph_state import GraphState

logger = logging.getLogger(__name__)


def should_continue_to_rewriter(state: GraphState) -> Literal["rewrite", "generate", "end"]:
    """
    Determine next step in the RAG pipeline based on current state.

    Decision hierarchy:
        1. Max iterations reached with docs → generate
        2. Max iterations reached without docs → end
        3. No documents found → end
        4. Relevant docs with confidence >= 0.6 → generate
        5. Relevant docs with low confidence, can rewrite → rewrite
        6. Relevant docs with low confidence, last iteration → generate
        7. No relevant docs, can rewrite → rewrite
        8. No relevant docs, last iteration → end

    Args:
        state: Current graph state with iteration and document info

    Returns:
        Next node: "rewrite", "generate", or "end"
    """
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", 3)
    relevant_docs = state.get("relevant_docs", [])
    documents = state.get("documents", [])
    confidence = state.get("confidence", 0.0)

    logger.info(f"[Routing Decision] Iteration {iterations}/{max_iterations}")
    logger.info(f"[Routing Decision] Docs: {len(documents)}, Relevant: {len(relevant_docs)}, Confidence: {confidence:.2f}")

    if iterations >= max_iterations:
        if documents or relevant_docs:
            logger.info("[Routing] Max iterations reached, generating with available docs")
            return "generate"
        else:
            logger.info("[Routing] Max iterations reached, no docs")
            return "end"

    if not documents:
        logger.info("[Routing] No documents found")
        return "end"

    if relevant_docs:
        if confidence >= 0.6:
            logger.info("[Routing] High confidence with relevant docs")
            return "generate"
        elif iterations < max_iterations - 1:
            logger.info("[Routing] Low confidence, will rewrite")
            return "rewrite"
        else:
            logger.info("[Routing] Low confidence, last iteration")
            return "generate"
    else:
        if iterations < max_iterations:
            logger.info("[Routing] No relevant docs, will rewrite")
            return "rewrite"
        else:
            logger.info("[Routing] No relevant docs, max iterations")
            return "end"


def should_rewrite_again(state: GraphState) -> Literal["retrieve", "generate"]:
    """
    Determine next step after question rewriting.

    Args:
        state: Current graph state with rewrite information

    Returns:
        Next node: "retrieve" to search again, or "generate" to produce answer
    """
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", 3)
    rewritten_questions = state.get("rewritten_questions", [])
    current_rewrite_index = state.get("current_rewrite_index", 0)

    logger.info(f"[Rewrite Check] Iteration {iterations}/{max_iterations}")
    logger.info(f"[Rewrite Check] Remaining rewrites: {len(rewritten_questions) - current_rewrite_index}")

    if iterations >= max_iterations:
        logger.info("[Rewrite Check] Max iterations")
        return "generate"

    if current_rewrite_index < len(rewritten_questions):
        logger.info("[Rewrite Check] More rewrites available")
        return "retrieve"

    logger.info("[Rewrite Check] No more rewrites")
    return "generate"


def route_after_generation(state: GraphState) -> Literal[END, "rewrite"]:
    """
    Determine next step after answer generation.

    Routes to END if answer exists or max iterations reached,
    otherwise routes back to rewrite for another attempt.

    Args:
        state: Current graph state

    Returns:
        END or "rewrite"
    """
    answer = state.get("answer")
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", 3)

    logger.info(f"[Route After Generation] Iteration {iterations}/{max_iterations}")
    logger.debug(f"[Route After Generation] Answer present: {answer is not None}")
    logger.debug(f"[Route After Generation] Answer value: {repr(answer)[:200] if answer else 'None'}")

    if answer and str(answer).strip():
        logger.info("[Route After Generation] Answer generated")
        return END

    if iterations < max_iterations:
        logger.info("[Route After Generation] No answer, can rewrite")
        return "rewrite"

    logger.info("[Route After Generation] No answer, max iterations")
    return END


def create_conditional_edges() -> Dict[str, Callable]:
    """
    Create all conditional edge functions for the graph.

    Returns:
        Dictionary mapping edge names to routing functions
    """
    return {
        "should_continue": should_continue_to_rewriter,
        "should_rewrite_again": should_rewrite_again,
        "route_after_generation": route_after_generation,
    }


class EdgeRouter:
    """
    Router for conditional edges with logging and decision tracking.

    Wraps routing functions to provide consistent logging and
    maintain history of routing decisions for debugging.

    Attributes:
        edge_history: List of routing decisions with timestamps
    """

    def __init__(self):
        self.edge_history = []

    def route_to_rewriter(self, state: GraphState) -> str:
        """
        Route from grader to next node with logging.

        Args:
            state: Current graph state

        Returns:
            Next node name
        """
        decision = should_continue_to_rewriter(state)

        self.edge_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "from_node": "grader",
                "to_node": decision,
                "state_snapshot": {
                    "iterations": state.get("iterations", 0),
                    "relevant_docs_count": len(state.get("relevant_docs", [])),
                    "confidence": state.get("confidence", 0.0),
                    "needs_rewrite": state.get("needs_rewrite", False),
                },
            }
        )

        return decision

    def route_after_rewrite(self, state: GraphState) -> str:
        """
        Route from rewriter to next node with logging.

        Args:
            state: Current graph state

        Returns:
            Next node name
        """
        decision = should_rewrite_again(state)

        self.edge_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "from_node": "rewriter",
                "to_node": decision,
                "state_snapshot": {
                    "iterations": state.get("iterations", 0),
                    "rewrite_index": state.get("current_rewrite_index", 0),
                    "total_rewrites": len(state.get("rewritten_questions", [])),
                },
            }
        )

        return decision

    def route_from_generator(self, state: GraphState) -> str:
        """
        Route from generator to next node with logging.

        Args:
            state: Current graph state

        Returns:
            END or "rewrite"
        """
        decision = route_after_generation(state)

        self.edge_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "from_node": "generator",
                "to_node": "END" if decision == END else "rewrite",
                "state_snapshot": {
                    "answer_present": state.get("answer") is not None,
                    "iterations": state.get("iterations", 0),
                },
            }
        )

        return decision

    def get_edge_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about routing decisions.

        Returns:
            Dictionary with total decisions, counts by type, and recent history
        """
        if not self.edge_history:
            return {"total_decisions": 0}

        decisions = [h["to_node"] for h in self.edge_history]

        return {
            "total_decisions": len(self.edge_history),
            "decision_counts": {
                "rewrite": decisions.count("rewrite"),
                "generate": decisions.count("generate"),
                "retrieve": decisions.count("retrieve"),
                "end": decisions.count("end") + decisions.count(END),
            },
            "recent_decisions": (self.edge_history[-5:] if len(self.edge_history) > 5 else self.edge_history),
        }
