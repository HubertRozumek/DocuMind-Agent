import copy
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    """
    Typed dictionary representing the complete state of a RAG conversation.

    Tracks all information needed for multi-turn retrieval, grading,
    rewriting, and generation pipeline.

    Attributes:
        question: Original user question
        documents: Retrieved documents from vector store
        relevant_docs: Documents passing relevance grading
        rewritten_question: Single rewritten question (legacy)
        rewritten_questions: List of query variations for rewriting
        current_rewrite_index: Index of current rewrite being used
        answer: Generated final answer
        iterations: Current iteration count
        search_query: Active search query (may be rewritten)
        current_document_index: Index for document iteration
        needs_rewrite: Flag indicating if rewrite is needed
        confidence: Overall confidence score (0.0-1.0)
        history: Conversation history with timestamps
        vector_store_type: Type of vector store (e.g., "chromadb")
        search_threshold: Minimum similarity threshold for retrieval
        max_iterations: Maximum allowed iterations
        metadata: Additional execution metadata
        error: Error message if occurred
        search_history: History of search queries
        decision_log: Log of agent decisions
        tool_used: Name of external tool used (if any)
        tool_result: Result from external tool (if any)
        skip_retrieval: Flag to skip retrieval when tool provides answer
    """

    question: str
    documents: List[str]
    relevant_docs: List[str]
    rewritten_question: Optional[str]
    rewritten_questions: List[str]
    current_rewrite_index: int
    answer: Optional[str]
    iterations: int
    search_query: Optional[str]
    current_document_index: int
    needs_rewrite: bool
    confidence: float
    history: List[Dict[str, Any]]
    vector_store_type: str
    search_threshold: float
    max_iterations: int
    metadata: Dict[str, Any]
    error: Optional[str]
    search_history: List[str]
    decision_log: List[Dict[str, Any]]
    tool_used: Optional[str]
    tool_result: Optional[Any]
    skip_retrieval: bool


def _sanitize_value(value):
    """Convert numpy types to native Python types."""
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(v) for v in value]
    return value


def sanitize_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types.

    Handles numpy scalars, arrays, and nested structures.

    Args:
        obj: Object potentially containing numpy types

    Returns:
        Object with numpy types converted to Python natives
    """
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()

    if isinstance(obj, np.str_):
        return str(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, dict):
        return {key: sanitize_numpy_types(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [sanitize_numpy_types(item) for item in obj]

    if isinstance(obj, tuple):
        return tuple(sanitize_numpy_types(item) for item in obj)

    return obj


class StateManager:
    """
    Manager for graph state operations.

    Provides static methods for creating, updating, validating,
    and serializing graph states with proper type handling.
    """

    @staticmethod
    def create_initial_state(
        question: str,
        vector_store_type: str = "chromadb",
        search_threshold: float = 0.7,
        max_iterations: int = 3,
    ) -> GraphState:
        """
        Create initial state for new conversation.

        Args:
            question: User question
            vector_store_type: Vector store backend type
            search_threshold: Minimum similarity for retrieval
            max_iterations: Maximum rewrite iterations

        Returns:
            Initialized GraphState
        """
        return {
            "question": question,
            "documents": [],
            "relevant_docs": [],
            "rewritten_question": None,
            "answer": None,
            "iterations": 0,
            "search_query": question,
            "current_document_index": 0,
            "needs_rewrite": False,
            "confidence": 0.0,
            "history": [{"role": "user", "content": question, "timestamp": datetime.now().isoformat()}],
            "vector_store_type": vector_store_type,
            "search_threshold": search_threshold,
            "max_iterations": max_iterations,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "vector_store": vector_store_type,
                "search_params": {"threshold": search_threshold, "max_results": 5},
            },
            "error": None,
            "rewritten_questions": [],
            "current_rewrite_index": 0,
            "search_history": [],
            "decision_log": [],
            "tool_used": None,
            "tool_result": None,
            "skip_retrieval": False,
        }

    @staticmethod
    def update_state(state: GraphState, **kwargs) -> GraphState:
        """
        Update graph state while preserving critical keys.

        Sanitizes numpy types and ensures preserved keys are not lost.
        Supports special keys: "increment_iterations" and "metadata".

        Args:
            state: Current state
            **kwargs: Updates to apply

        Returns:
            Updated state
        """
        updated_state = copy.deepcopy(state)
        kwargs = sanitize_numpy_types(kwargs)

        preserved_keys = [
            "rewritten_questions",
            "current_rewrite_index",
            "search_history",
            "decision_log",
            "tool_used",
            "tool_result",
        ]

        for key, value in kwargs.items():
            if key == "increment_iterations":
                if value:
                    updated_state["iterations"] = updated_state.get("iterations", 0) + 1
            elif key == "metadata":
                existing_metadata = copy.deepcopy(updated_state.get("metadata", {}))
                updated_state["metadata"] = {**existing_metadata, **copy.deepcopy(value)}
            else:
                updated_state[key] = copy.deepcopy(value)

        for key in preserved_keys:
            if key not in updated_state and key in state:
                updated_state[key] = copy.deepcopy(state[key])
                logger.debug(f"Preserved state key: {key}")

        updated_state = sanitize_numpy_types(updated_state)

        if "rewritten_questions" in kwargs:
            logger.info(f"[StateUpdate] rewritten_questions set: {len(kwargs['rewritten_questions'])} items")
        if "current_rewrite_index" in kwargs:
            logger.info(f"[StateUpdate] current_rewrite_index: {kwargs['current_rewrite_index']}")
        if "confidence" in kwargs:
            logger.info(f"[StateUpdate] confidence updated: {kwargs['confidence']}")

        return updated_state

    @staticmethod
    def _log_state_change(old_state: GraphState, new_state: GraphState, changes: Dict):
        """Log state changes for debugging."""
        if old_state.get("metadata", {}).get("debug", False):
            print(f"\n[State Change] Iteration {new_state['iterations']}")
            for key, value in changes.items():
                if key in old_state:
                    old_val = old_state[key]
                    print(f" {key}: {old_val} -> {value}")

    @staticmethod
    def validate_state(state: GraphState) -> Dict[str, Any]:
        """
        Validate state and return validation results.

        Checks for constraint violations and unsupported configurations.

        Args:
            state: State to validate

        Returns:
            Dictionary with is_valid flag, errors, warnings, and summary
        """
        errors = []
        warnings = []

        if state["iterations"] > state["max_iterations"]:
            errors.append(f"Max iterations reached: {state['iterations']}/{state['max_iterations']}")

        if not state["question"] or not state["question"].strip():
            errors.append("Question cannot be empty")

        if not 0 <= state["search_threshold"] <= 1:
            errors.append(f"Search threshold must be between 0 and 1: {state['search_threshold']}")

        if not 0 <= state["confidence"] <= 1:
            warnings.append(f"Confidence must be between 0 and 1: {state['confidence']}")

        supported_stores = ["chromadb"]
        if state["vector_store_type"] not in supported_stores:
            warnings.append(f"Vector store type '{state['vector_store_type']}' is not supported")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "state_summary": StateManager.get_state_summary(state),
        }

    @staticmethod
    def get_state_summary(state: GraphState) -> Dict[str, Any]:
        """
        Get summary of current state.

        Args:
            state: Current state

        Returns:
            Dictionary with key state metrics
        """
        return {
            "question": state["question"],
            "iterations": state["iterations"],
            "documents_found": len(state["documents"]),
            "relevant_documents": len(state["relevant_docs"]),
            "confidence": state["confidence"],
            "needs_rewrite": state["needs_rewrite"],
            "has_answer": state["answer"] is not None,
            "vector_store": state["vector_store_type"],
            "search_threshold": state["search_threshold"],
        }

    @staticmethod
    def add_to_history(state: GraphState, entry: Dict[str, Any]) -> GraphState:
        """
        Add entry to conversation history.

        Args:
            state: Current state
            entry: History entry to add

        Returns:
            Updated state with new history entry
        """
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now().isoformat()

        new_state = state.copy()
        new_state["history"] = state["history"] + [entry]
        return new_state

    @staticmethod
    def get_conversation_history(state: GraphState, max_entries: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent conversation history.

        Args:
            state: Current state
            max_entries: Maximum number of entries to return

        Returns:
            List of recent history entries
        """
        return state["history"][-max_entries:]


class StateEncoder(json.JSONEncoder):
    """
    JSON encoder for graph state serialization.

    Handles datetime objects and custom objects with __dict__.
    """

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)


def serialize_state(state: GraphState) -> str:
    """
    Serialize state to JSON string.

    Args:
        state: State to serialize

    Returns:
        JSON string representation
    """
    return json.dumps(state, cls=StateEncoder, ensure_ascii=False, indent=2)


def deserialize_state(state: str) -> GraphState:
    """
    Deserialize state from JSON string.

    Args:
        state: JSON string

    Returns:
        Deserialized GraphState
    """
    data = json.loads(state)

    for entry in data.get("history", []):
        if "timestamp" in entry and isinstance(entry["timestamp"], str):
            try:
                entry["timestamp"] = datetime.fromisoformat(entry["timestamp"])
            except Exception as e:
                logger.info(f"[Deserialize State]Failed to convert timestamp {e}")
    return data
