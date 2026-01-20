from typing import List, Dict, Any, Optional, Annotated, TypedDict
from typing_extensions import TypedDict
import operator
from datetime import datetime
import json
import copy

class GraphState(TypedDict, total=False):
    """
    A state graph that stores all information about the course of a conversation

    Attributes:
        question: orginal user question
        documents: lsit of documents
        relevant_docs: lsit of relevant documents
        rewritten_question: rewritten question
        answear: Definitive answer
        iterations: number of iterations
        search_query: search query
        current_document_index: current document index
        needs_rewrite: does the question needs to be rewritten
        confidence: confidence score (0-1)
        history: conversation history
        vector_store_type: vector store type
        search_threshold: similarity threshold for search
        max_iterations: maximum number of iterations
        metadata: additional metadata
        error: error if occurred
    """

    question: str
    documents: List[str]
    relevant_docs: List[str]
    rewritten_question: Optional[str]
    answear: Optional[str]
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

class StateManager:
    """
    Manager class that manages state graphs
    """

    @staticmethod
    def create_initial_state(
            question: str,
            vector_store_type: str = "chromadb",
            search_threshold: float = 0.7,
            max_iterations: int = 3
    ) -> GraphState:
        """
        Create initial state graph

        Args:
            question: question text
            vector_store_type: vector store type
            search_threshold: similarity threshold for search
            max_iterations: maximum number of iterations
        Returns:
            Initial state graph
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

            "history": [{
                "role": "user",
                "content": question,
                "timestamp": datetime.now().isoformat()
            }],
            "vector_store_type": vector_store_type,
            "search_threshold": search_threshold,
            "max_iterations": max_iterations,

            "metadata": {
                "created_at": datetime.now().isoformat(),
                "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "vector_store": vector_store_type,
                "search_params": {
                    "threshold": search_threshold,
                    "max_results": 5
                },
            },
            "error": None
        }

    @staticmethod
    def update_state(state: GraphState, **kwargs) -> GraphState:
        """
        Update graph state

        Args:
            state: current state
            kwargs: fields to update
        Returns:
            updated state
        """
        updated_state = copy.deepcopy(state)

        for key, value in kwargs.items():
            if key in updated_state:
                updated_state[key] = value
            elif key in updated_state["metadata"]:
                updated_state["metadata"][key] = value

        if "answer" in kwargs and kwargs["answer"]:
            updated_state["history"].append({
                "role": "assistant",
                "content": kwargs["answer"],
                "timestamp": datetime.now().isoformat(),
                "confidence": updated_state.get("confidence", 0.0),
                "document_used": len(updated_state.get("relevant_docs", [])),
            })

        if "iterations" in kwargs:
            updated_state["iterations"] = kwargs["iterations"]
        elif kwargs.get("increment_iterations", False):
            updated_state["iterations"] += 1

        StateManager._log_state_change(state, updated_state, kwargs)

        return updated_state

    @staticmethod
    def _log_state_change(old_state: GraphState, new_state: GraphState, changes: Dict):
        """
        Log changes in state
        :param old_state:
        :param new_state:
        :param changes:
        :return:
        """
        if old_state.get("metadata", {}).get("debug", False):
            print(f"\n[State Change] Iteration {new_state['iterations']}")
            for key, value in changes.items():
                if key in old_state:
                    old_val = old_state[key]
                    print(f" {key}: {old_val} -> {value}")

    @staticmethod
    def validate_state(state: GraphState) -> Dict[str, Any]:
        """
        Validate state and return information about errors

        Returns:
            Dictionary of errors
        """
        errors = {}
        warnings = {}

        if state["iterations"] > state["max_iterations"]:
            errors.append(f"Max iterations reached: {state['iterations']}/{state['max_iterations']}")

        if not state["question"] or not state["question"].strip():
            errors.append("Question cannot be empty")

        if not 0 <= state["search_threshold"] <= 1:
            errors.append(f"Search threshold must be between 0 and 1: {state['search_threshold']}")

        if not 0 <= state["confidence"] <= 1:
            warnings.append(f"Confidence must be between 0 and 1: {state['confidence']}")

        supported_stores = ['chromadb']
        if state["vector_store_type"] not in supported_stores:
            warnings.append(f"Vector store type '{state['vector_store_type']}' is not supported")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "state_summary": StateManager.get_state_summary(state)
        }

    @staticmethod
    def get_state_summary(state: GraphState) -> Dict[str, Any]:
        """
        Return summary of state
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
        Add state to history
        """
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now().isoformat()

        new_state = state.copy()
        new_state["history"] = state["history"] + [entry]
        return new_state

    @staticmethod
    def get_conversation_history(state: GraphState, max_entries: int = 10) -> List[Dict[str, Any]]:
        """
        Return conversation history
        """
        return state["history"][-max_entries:]


class StateEncoder(json.JSONEncoder):
    """
    Class to convert state to JSON
    """
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

def serialize_state(state: GraphState) -> str:
    """
    Serialize state to JSON
    """
    return json.dumps(state, cls=StateEncoder, ensure_ascii=False, indent=2)

def deserialize_state(state: str) -> GraphState:
    """
    Deserialize state from JSON
    """
    data = json.loads(state)

    for entry in data.get("history", []):
        if "timestamp" in entry and isinstance(entry["timestamp"], str):
            try:
                entry["timestamp"] = datetime.fromisoformat(entry["timestamp"])
            except:
                pass
    return data