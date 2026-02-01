from typing import Literal, Dict, Any, Callable
from langgraph.graph import END
import logging
from datetime import datetime

from src.agent.graph_state import GraphState

logger = logging.getLogger(__name__)


def should_continue_to_rewriter(state: GraphState) -> Literal["rewrite", "generate", "end"]:
    """
    Determine whether to rewrite the question, generate answer, or end.

    Conditions:
    1. If we have relevant documents and confidence is high → generate answer
    2. If we have no relevant documents AND iterations < max_iterations → rewrite question
    3. If max iterations reached OR no documents at all → end

    Args:
        state: Current graph state

    Returns:
        Next node to transition to
    """
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", 3)
    relevant_docs = state.get("relevant_docs", [])
    documents = state.get("documents", [])
    confidence = state.get("confidence", 0.0)

    logger.info(f"[Conditional Edge] Iteration {iterations}/{max_iterations}")
    logger.info(f"[Conditional Edge] Relevant docs: {len(relevant_docs)}, Total docs: {len(documents)}")
    logger.info(f"[Conditional Edge] Confidence: {confidence:.2f}")

    # Case 1: Max iterations reached
    if iterations >= max_iterations:
        logger.info("[Conditional Edge] Max iterations reached → END")
        return "end"

    # Case 2: No documents at all (search failed)
    if not documents:
        logger.info("[Conditional Edge] No documents found → END")
        return "end"

    # Case 3: We have relevant documents with sufficient confidence
    if relevant_docs and confidence >= 0.6:
        logger.info("[Conditional Edge] Relevant documents found → GENERATE")
        return "generate"

    # Case 4: No relevant documents but can try again
    if not relevant_docs:
        logger.info("[Conditional Edge] No relevant docs, need rewrite → REWRITE")
        return "rewrite"

    # Case 5: Low confidence with relevant docs - still try to generate
    if relevant_docs and confidence < 0.6:
        logger.info("[Conditional Edge] Low confidence but has docs → GENERATE")
        return "generate"

    # Default fallback
    logger.info("[Conditional Edge] Default fallback → END")
    return "end"


def should_rewrite_again(state: GraphState) -> Literal["retrieve", "generate", "end"]:
    """
    Determine what to do after question rewriting.

    Conditions:
    1. If we have rewritten questions left to try → retrieve again
    2. If no more rewritten questions → generate with what we have
    3. If max iterations reached → end

    Args:
        state: Current graph state

    Returns:
        Next node to transition to
    """
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", 3)
    rewritten_questions = state.get("rewritten_questions", [])
    current_rewrite_index = state.get("current_rewrite_index", 0)

    logger.info(f"[Rewrite Check] Iteration {iterations}/{max_iterations}")
    logger.info(f"[Rewrite Check] Remaining rewrites: {len(rewritten_questions) - current_rewrite_index}")

    # Case 1: Max iterations reached
    if iterations >= max_iterations:
        logger.info("[Rewrite Check] Max iterations → END")
        return "end"

    # Case 2: More rewritten questions to try
    if current_rewrite_index < len(rewritten_questions) - 1:
        logger.info("[Rewrite Check] More rewrites available → RETRIEVE")
        return "retrieve"

    # Case 3: No more rewrites, generate with what we have
    logger.info("[Rewrite Check] No more rewrites → GENERATE")
    return "generate"


def route_after_generation(state: GraphState) -> Literal[END, "rewrite"]:
    """
    Determine what to do after answer generation.

    Conditions:
    1. If answer is generated successfully → END
    2. If generation failed and we can still rewrite → rewrite
    3. Otherwise → END

    Args:
        state: Current graph state

    Returns:
        Next node to transition to
    """
    answer = state.get("answer")
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", 3)

    logger.info(f"[Route After Generation] Iteration {iterations}/{max_iterations}")
    logger.info(f"[Route After Generation] Answer present: {answer is not None}")

    # Case 1: Answer successfully generated
    if answer and answer.strip():
        logger.info("[Route After Generation] Answer generated → END")
        return END

    # Case 2: No answer but can still rewrite
    if not answer and iterations < max_iterations:
        logger.info("[Route After Generation] No answer, can rewrite → REWRITE")
        return "rewrite"

    # Case 3: No answer and max iterations reached
    logger.info("[Route After Generation] No answer, max iterations → END")
    return END


def create_conditional_edges() -> Dict[str, Callable]:
    """
    Create all conditional edge functions for the graph.

    Returns:
        Dictionary of conditional edge functions
    """
    return {
        "should_continue": should_continue_to_rewriter,
        "should_rewrite_again": should_rewrite_again,
        "route_after_generation": route_after_generation,
    }


class EdgeRouter:
    """
    Router for conditional edges with logging and monitoring.
    """

    def __init__(self):
        self.edge_history = []

    def route_to_rewriter(self, state: GraphState) -> str:
        """
        Route to question rewriter with logging.
        """
        decision = should_continue_to_rewriter(state)

        self.edge_history.append({
            "timestamp": datetime.now().isoformat(),
            "from_node": "grader",
            "to_node": decision,
            "state_snapshot": {
                "iterations": state.get("iterations", 0),
                "relevant_docs_count": len(state.get("relevant_docs", [])),
                "confidence": state.get("confidence", 0.0),
                "needs_rewrite": state.get("needs_rewrite", False)
            }
        })

        return decision

    def route_after_rewrite(self, state: GraphState) -> str:
        """
        Route after question rewriting.
        """
        decision = should_rewrite_again(state)

        self.edge_history.append({
            "timestamp": datetime.now().isoformat(),
            "from_node": "rewriter",
            "to_node": decision,
            "state_snapshot": {
                "iterations": state.get("iterations", 0),
                "rewrite_index": state.get("current_rewrite_index", 0),
                "total_rewrites": len(state.get("rewritten_questions", []))
            }
        })

        return decision

    def route_from_generator(self, state: GraphState) -> str:
        """
        Route from answer generator.
        """
        decision = route_after_generation(state)

        self.edge_history.append({
            "timestamp": datetime.now().isoformat(),
            "from_node": "generator",
            "to_node": "END" if decision == END else "rewrite",
            "state_snapshot": {
                "answer_present": state.get("answer") is not None,
                "iterations": state.get("iterations", 0)
            }
        })

        return decision

    def get_edge_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about edge routing decisions.
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
                "end": decisions.count("end") + decisions.count(END)
            },
            "recent_decisions": self.edge_history[-5:] if len(self.edge_history) > 5 else self.edge_history
        }