from typing import Dict, Any, List, Optional, Callable
from langchain_core.runnables import RunnableLambda
import numpy as np
import logging
from datetime import datetime

from ..graph_state import GraphState

logger = logging.getLogger(__name__)

class RetrieverNode:
    """
    The node responsible for searching documents in the vector store.
    """

    def __init__(self,
                 vector_store,
                 embedding_model = None,
                 search_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            vector_store: vector store
            embedding_model: Model to create embeddings
            search_config: search config
        """

        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.search_config = search_config or self._default_search_config()

        logger.info(f"Initializing retriever node")
        logger.info(f"Search config: {self.search_config}")

    def _default_search_config(self) -> Dict[str, Any]:
        """
        Default search config.
        """

        return {
            "k": 5,
            "score_threshold": 0.7,
            "filter_metadata": None,
            "include_metadata": True,
            "include_embeddings": False,
            "rerank": False,
            "diversity": False
        }

    def retrieve(self,
                  query: str,
                  state: Optional[GraphState] = None) -> Dict[str, Any]:
        """
        Main function for retrieving documents.

        Args:
            query: query to search for
            state: graph state
        Returns:
            Dictionary with search results
        """
        logger.info("Retrieving documents for query %s", query)

        try:
            #[1] - Prepare query
            processed_query = self._preprocess_query(query, state)

            #[2] - Search the vector store
            search_results = self._search_vector_store(processed_query)

            #[3] - Filter results based on threshold
            filtered_results = self._filter_results(search_results)

            #[4] - Process results
            processed_results = self._process_results(filtered_results, query)

            logger.info(f"Retriever: find {len(processed_results['documents'])} documents")

            return processed_results

        except Exception as e:
            logger.error(f"Retriever: Search failed: {e}")
            return self._create_error_result(e,query)

    def _preprocess_query(self, query: str, state: Optional[GraphState]) -> str:
        """
        Intelligent query preprocessing for RAG retrieval.
        Goal: maximize recall without hurting precision.
        """

        # normalize
        query = " ".join(query.strip().split())

        if not state:
            return query

        enriched_parts = []

        # rewritten / canonical query
        rewritten = state.get("rewritten_question")
        if rewritten and rewritten.strip() and rewritten.lower() != query.lower():
            enriched_parts.append(rewritten.strip())

        # Always include current query
        enriched_parts.append(query)

        # recent user context
        history = state.get("history", [])
        user_turns = [
            h["content"].strip()
            for h in history
            if h.get("role") == "user" and h.get("content")
        ]

        # Take last 2 user questions (excluding current one)
        context_turns = user_turns[-3:-1]

        if context_turns:
            context = " ".join(context_turns)
            enriched_parts.append(context)

        # light intent boost
        intent_hint = self._infer_intent_hint(query)
        if intent_hint:
            enriched_parts.append(intent_hint)

        # deduplicate + cap length
        final_query = " ".join(dict.fromkeys(enriched_parts))

        # Hard safety cap (vector stores like short queries)
        MAX_CHARS = 512
        return final_query[:MAX_CHARS]

    def _infer_intent_hint(self, query: str) -> Optional[str]:
        q = query.lower()

        if q.startswith(("what is", "define", "explain")):
            return "definition explanation overview"

        if q.startswith(("how to", "how do i", "steps", "guide")):
            return "step by step instructions example"

        if "compare" in q or "difference" in q:
            return "comparison pros cons differences"

        if q.endswith("?"):
            return "factual answer"

        return None

    def _search_vector_store(self, query: str) -> Dict[str, Any]:
        """
        Searches in the vector store
        """
        search_kwargs = {
            "k": self.search_config["k"] * 2,
        }

        if self.search_config["filter_metadata"]:
            search_kwargs["where"] = self.search_config["filter_metadata"]

        try:

            results = self.vector_store.search(
                query = query,
                n_results = search_kwargs["k"],
                where = search_kwargs.get("where")
            )

            processed = {
                "documents": results.get("documents", []),
                "metadatas": results.get("metadatas", []),
                "ids": results.get("ids", []),
                "distances": results.get("distances", []),
                "similarities": results.get("similarities", []),
                "query": query,
            }
            return processed

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def _filter_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter results by similarity threshold.
        """
        threshold = self.search_config["score_threshold"]

        if not results.get("similarities") or not results.get("documents"):
            return {
                "documents": [],
                "metadatas": [],
                "ids": [],
                "distances": [],
                "similarities": [],
                "query": results.get("query", "")
            }
        filtered_indices = [
            i for i, similarity in enumerate(results["similarities"])
            if similarity >= threshold
        ]
        filtered_results = {
            "documents": [results["documents"][i] for i in filtered_indices],
            "metadatas": [results["metadatas"][i] for i in filtered_indices] if results.get("metadatas") else [],
            "ids": [results["ids"][i] for i in filtered_indices] if results.get("ids") else [],
            "distances": [results["distances"][i] for i in filtered_indices] if results.get("distances") else [],
            "similarities": [results["similarities"][i] for i in filtered_indices],
            "query": results.get("query", "")
        }

        if len(filtered_results["documents"]) < 3 and results["documents"]:
            logger.warning(f"Too few results below threshold: {threshold}")

            n_best = min(3, len(results["documents"]))
            filtered_results = {
                "documents": results["documents"][:n_best],
                "metadatas": results["metadatas"][:n_best] if results.get("metadatas") else [],
                "ids": results["ids"][:n_best] if results.get("ids") else [],
                "distances": results["distances"][:n_best] if results.get("distances") else [],
                "similarities": results["similarities"][:n_best],
                "query": results.get("query", "")
            }

        if self.search_config["rerank"] and len(filtered_results["documents"]) > 1:
            filtered_results = self._rerank_results(filtered_results)

        if self.search_config["diversity"] and len(filtered_results["documents"]) > 3:
            filtered_results = self._diversify_results(filtered_results)

        return filtered_results

    def _rerank_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rerank results based on additional criteria.
        Can add rerank based on:
        - document length
        """

        # Example: prefer shorter docs
        if results.get("documents"):
            docs_with_indices = list(enumerate(results["documents"]))
            docs_with_indices.sort(key = lambda x: len(x[1]))

            new_order = [i for i,_ in docs_with_indices]

            reranked_results = {
                "documents": [results["documents"][i] for i in new_order],
                "metadatas": [results["metadatas"][i] for i in new_order] if results.get("metadatas") else [],
                "ids": [results["ids"][i] for i in new_order] if results.get("ids") else [],
                "distances": [results["distances"][i] for i in new_order] if results.get("distances") else [],
                "similarities": [results["similarities"][i] for i in new_order],
                "query": results["query"]
            }

            return reranked_results

        return results

    def _diversify_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Increases the variety of results.
        Avoids returning documents that are too similar to each other.
        """

        if len(results["documents"]) < 3:
            return results

        selected_indices = [0]

        for i in range(1, len(results["documents"])):
            too_similar = False

            for selected_idx in selected_indices:
                doc1 = results["documents"][selected_idx].lower()
                doc2 = results["documents"][i].lower()

                words1 = set(doc1.split()[:50])
                words2 = set(doc2.split()[:50])

                if words1 and words2:
                    similarity = len(words1 & words2) / len(words1 | words2)
                    if similarity > 0.7:
                        too_similar = True
                        break

            if not too_similar and len(selected_indices) < 5:
                selected_indices.append(i)

        diversified_results = {
            "documents": [results["documents"][i] for i in selected_indices],
            "metadatas": [results["metadatas"][i] for i in selected_indices] if results.get("metadatas") else [],
            "ids": [results["ids"][i] for i in selected_indices] if results.get("ids") else [],
            "distances": [results["distances"][i] for i in selected_indices] if results.get("distances") else [],
            "similarities": [results["similarities"][i] for i in selected_indices],
            "query": results["query"]
        }
        return diversified_results

    def _process_results(self, results: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """
        Processes and formats search results.
        """
        if not results["documents"]:
            return {
                "documents": [],
                "metadatas": [],
                "ids": [],
                "distances": [],
                "similarities": [],
                "query": original_query,
                "summary": "No documents meeting the criteria.",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat()
            }

        avg_similarity = np.mean(results["similarities"]) if results["similarities"] else 0.0

        summary = f"Found {len(results['documents'])} similar documents with average similarity: {avg_similarity:.2f}"

        sources = []
        if results.get("metadatas"):
            for metadata in results["metadatas"]:
                if isinstance(metadata, dict):
                    source = metadata.get("source", "unknown")
                    doc_id = metadata.get("doc_id", "unknown")
                    sources.append(f"{source}/{doc_id}")

        processed_results = {
            "documents": results["documents"],
            "metadatas": results["metadatas"],
            "ids": results["ids"],
            "distances": results["distances"],
            "similarities": results["similarities"],
            "query": original_query,
            "summary": summary,
            "confidence": float(avg_similarity),
            "timestamp": datetime.now().isoformat(),
            "sources": list(set(sources))[:5],
            "stats": {
                "total_documents": len(results["documents"]),
                "avg_similarity": float(avg_similarity),
                "min_similarity": float(min(results["similarities"])) if results["similarities"] else 0.0,
                "max_similarity": float(max(results["similarities"])) if results["similarities"] else 0.0,
                "threshold_used": self.search_config["score_threshold"]
            }
        }

        return processed_results

    def _create_error_result(self, error: Exception, query: str) -> Dict[str, Any]:
        """
        Creates error result
        """
        return {
            "documents": [],
            "metadatas": [],
            "ids": [],
            "distances": [],
            "similarities": [],
            "query": query,
            "summary": f"Search Error: {str(error)}",
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
            "error": str(error),
            "error_type": type(error).__name__
        }

    def as_runnable(self) -> Callable:
        """
        Return a LangChain Runnable compatible function
        Returns:
            A function that takes a state and returns the updated state.
        """
        def retriever_function(state: GraphState) -> GraphState:
            """
            Retriever function for LangGraph

            Args:
                 state: Current state.

            Returns:
                Updated state with search results.
            """
            from ..graph_state import StateManager

            logger.info(f"[Retriever Function] Iteration {state['iterations']}")
            logger.info(f"[Retriever Function] Query {state['search_query']}")

            results = self.retrieve(state['search_query'], state)

            updated_state = StateManager.update_state(
                state,
                documents = results.get("documents", []),
                confidence = results.get("confidence", 0.0),
                metadata = {
                    **state.get("metadata", {}),
                    "retrieval_results": {
                        "summary": results.get("summary"),
                        "stats": results.get("stats", {}),
                        "timestamp": results.get("timestamp"),
                        "query_used": results.get("query")
                    }
                }
            )

            history_entry = {
                "role": "system",
                "action": "retrieval",
                "content": f"Retrieved {len(results.get('documents', []))} documents.",
                "confidence": results.get("confidence", 0.0),
                "details": {
                    "query": state['search_query'],
                    "documents_found": len(results.get("documents", [])),
                    "avg_similarity": results.get("stats", {}).get("avg_similarity", 0.0)
                }
            }

            updated_state = StateManager.add_to_history(updated_state, history_entry)

            logger.info(f"[Retriever Function] Found {len(results.get('documents', []))} documents.")
            logger.info(f"[Retriever Function] Confidence {results.get('confidence', 0.0):.2f}")

            return updated_state

        return RunnableLambda(retriever_function)


class RetrieverFactory:
    """
    Factory class to create retriever for vector stores
    """
    @staticmethod
    def create_retriever(
            collection_name: str = "documents",
            persist_directory: str = "data/vector_store/chroma",
            search_config: Optional[Dict[str, Any]] = None
    ) -> RetrieverNode:
        """
        Create retriever for ChromaDB

        Args:
            collection_name: Collection name in ChromaDB
            persist_directory: Directory to save data
            search_config: Search configuration
        Returns:
            RetrieverNode with ChromaDB
        """
        from src.vector_store.chroma_db import ChromaDBVectorStore

        try:
            vector_store = ChromaDBVectorStore(
                collection_name = collection_name,
                persist_directory = persist_directory,
                reset_on_start = False
            )

            logger.info(f"Created ChromaDB retriever for: {collection_name}")
            return RetrieverNode(vector_store, search_config=search_config)

        except Exception as e:
            logger.error(f"Failed to creating ChromaDB retriever: {e}")
            raise