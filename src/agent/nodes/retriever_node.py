import logging
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import numpy as np
from langchain_core.runnables import RunnableLambda

from ..graph_state import GraphState

logger = logging.getLogger(__name__)


class RetrieverNode:
    """
    Document retrieval node for RAG-based systems.

    Searches vector stores with intelligent query preprocessing, result filtering,
    and post-processing capabilities including reranking and diversification.

    Attributes:
        vector_store: Vector store instance for similarity search
        embedding_model: Optional embedding model for query encoding
        search_config: Configuration dict for search parameters
        _query_cache: LRU cache for preprocessed queries
        _cache_max_size: Maximum cache size for query preprocessing
    """

    def __init__(self, vector_store, embedding_model=None, search_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the retriever node.

        Args:
            vector_store: Vector store instance (e.g., ChromaDB)
            embedding_model: Optional model for creating query embeddings
            search_config: Search configuration dictionary with keys:
                - k: Number of results to retrieve
                - score_threshold: Minimum similarity score
                - filter_metadata: Optional metadata filters
                - rerank: Enable result reranking
                - diversity: Enable result diversification
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.search_config = search_config or self._default_search_config()

        self._query_cache = {}
        self._cache_max_size = 100

        logger.info("Initializing retriever node")
        logger.info(f"Search config: {self.search_config}")

    def _default_search_config(self) -> Dict[str, Any]:
        """Return default search configuration."""
        return {
            "k": 3,
            "score_threshold": 0.5,
            "filter_metadata": None,
            "include_metadata": True,
            "include_embeddings": False,
            "rerank": False,
            "diversity": False,
        }

    def retrieve(self, query: str, state: Optional[GraphState] = None) -> Dict[str, Any]:
        """
        Execute document retrieval pipeline.

        Performs query preprocessing, vector search, result filtering, and
        post-processing to return relevant documents.

        Args:
            query: Search query string
            state: Optional graph state for query enrichment and context

        Returns:
            Dictionary containing:
                - documents: List of retrieved document strings
                - metadatas: Document metadata
                - similarities: Similarity scores
                - confidence: Average similarity score
                - summary: Human-readable result summary
                - stats: Detailed statistics
                - timestamp: ISO format timestamp
                - error: Error message if failed (optional)
        """
        logger.info("Retrieving documents for query %s", query)

        if not query or not query.strip():
            logger.warning("Empty query provided")
            return self._create_error_result(ValueError("Query cannot be empty"), query)

        try:
            processed_query = self._preprocess_query(query, state)

            if state and "search_threshold" in state:
                self.search_config["score_threshold"] = state["search_threshold"]

            search_results = self._search_vector_store(processed_query)
            filtered_results = self._filter_results(search_results)
            processed_results = self._process_results(filtered_results, query)

            logger.info(f"Retriever: found {len(processed_results['documents'])} documents")

            return processed_results

        except Exception as e:
            logger.error(f"Retriever: Search failed: {e}")
            return self._create_error_result(e, query)

    def _preprocess_query(self, query: str, state: Optional[GraphState]) -> str:
        """
        Intelligent query preprocessing with context enrichment.

        Enriches queries using:
        - Query rewriting from state
        - Conversation history context
        - Intent-based keyword hints
        - Deduplication and length limiting

        Args:
            query: Raw user query
            state: Graph state containing rewrite history and context

        Returns:
            Optimized query string for vector search
        """
        if not state:
            return " ".join(query.strip().split())

        query = " ".join(query.strip().split())

        cache_key = self._create_cache_key(query, state)

        if cache_key in self._query_cache:
            logger.debug("Using cached preprocessed query")
            return self._query_cache[cache_key]

        enriched_parts = []

        rewritten_questions = state.get("rewritten_questions", [])
        current_rewrite_index = state.get("current_rewrite_index", 0)

        if rewritten_questions and current_rewrite_index < len(rewritten_questions):
            current_rewrite = rewritten_questions[current_rewrite_index]
            if current_rewrite and current_rewrite.strip() and current_rewrite.lower() != query.lower():
                enriched_parts.append(current_rewrite.strip())
                logger.info(f"[Query Preprocessing] Using rewrite #{current_rewrite_index}: {current_rewrite[:50]}...")

        enriched_parts.append(query)

        history = state.get("history", [])
        user_turns = [h["content"].strip() for h in history if h.get("role") == "user" and h.get("content")]

        context_turns = user_turns[-2:-1]

        if context_turns:
            context = " ".join(context_turns)
            enriched_parts.append(context)

        if len(query) < 50:
            intent_hint = self._infer_intent_hint(query)
            if intent_hint:
                enriched_parts.append(intent_hint)

        seen = set()
        final_parts = []
        for part in enriched_parts:
            part_lower = part.lower()
            if part_lower not in seen:
                seen.add(part_lower)
                final_parts.append(part)

        final_query = " ".join(final_parts)
        final_query = final_query[:512]

        if len(self._query_cache) >= self._cache_max_size:
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]

        self._query_cache[cache_key] = final_query
        return final_query

    def _create_cache_key(self, query: str, state: Optional[GraphState]) -> str:
        """Create cache key from query and relevant state."""
        if not state:
            return query

        rewritten_idx = state.get("current_rewrite_index", 0)
        rewrites = state.get("rewritten_questions", [])
        current_rewrite = rewrites[rewritten_idx] if rewritten_idx < len(rewrites) else ""

        return f"{query}|{current_rewrite}|{rewritten_idx}"

    def _infer_intent_hint(self, query: str) -> Optional[str]:
        """Infer search intent hints based on query patterns."""
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
        """Execute vector similarity search."""
        search_kwargs = {
            "k": self.search_config["k"],
        }

        if self.search_config.get("filter_metadata"):
            search_kwargs["where"] = self.search_config["filter_metadata"]

        try:
            results = self.vector_store.search(query=query, n_results=search_kwargs["k"], where=search_kwargs.get("where"))

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

    def _empty_results(self, query: str) -> Dict[str, Any]:
        """Return empty results structure."""
        return {
            "documents": [],
            "metadatas": [],
            "ids": [],
            "distances": [],
            "similarities": [],
            "query": query,
        }

    def _filter_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter and post-process search results.

        Applies similarity threshold filtering with fallback to absolute minimum,
        optional reranking, and optional diversification.

        Args:
            results: Raw search results from vector store

        Returns:
            Filtered and optionally reranked/diversified results
        """
        threshold = self.search_config["score_threshold"]
        ABSOLUTE_MIN_SIMILARITY = 0.25

        if not results.get("similarities") or not results.get("documents"):
            return self._empty_results(results.get("query", ""))

        filtered_indices = [i for i, similarity in enumerate(results["similarities"]) if similarity >= threshold]

        if not filtered_indices:
            logger.warning(f"No documents above threshold {threshold}")

            valid_docs = [(i, sim) for i, sim in enumerate(results["similarities"]) if sim >= ABSOLUTE_MIN_SIMILARITY]

            if not valid_docs:
                logger.error("No documents meet minimum quality threshold")
                return self._empty_results(results.get("query", ""))

            valid_docs.sort(key=lambda x: x[1], reverse=True)
            filtered_indices = [i for i, _ in valid_docs[:3]]

            logger.info(f"Using top {len(filtered_indices)} documents above minimum threshold")

        filtered_results = {
            "documents": [results["documents"][i] for i in filtered_indices],
            "metadatas": ([results["metadatas"][i] for i in filtered_indices] if results.get("metadatas") else []),
            "ids": [results["ids"][i] for i in filtered_indices] if results.get("ids") else [],
            "distances": ([results["distances"][i] for i in filtered_indices] if results.get("distances") else []),
            "similarities": [results["similarities"][i] for i in filtered_indices],
            "query": results.get("query", ""),
        }

        if len(filtered_results["documents"]) < 3 and results["documents"]:
            logger.warning(f"Only {len(filtered_results['documents'])} docs above threshold {threshold}")

            valid_docs = [(i, sim) for i, sim in enumerate(results["similarities"]) if sim >= ABSOLUTE_MIN_SIMILARITY]

            valid_docs.sort(key=lambda x: x[1], reverse=True)

            if valid_docs:
                n_best = min(3, len(valid_docs))
                best_indices = [i for i, sim in valid_docs[:n_best]]

                logger.info(f"Using top {n_best} results (above absolute min {ABSOLUTE_MIN_SIMILARITY})")

                filtered_results = {
                    "documents": [results["documents"][i] for i in best_indices],
                    "metadatas": ([results["metadatas"][i] for i in best_indices] if results.get("metadatas") else []),
                    "ids": [results["ids"][i] for i in best_indices] if results.get("ids") else [],
                    "distances": ([results["distances"][i] for i in best_indices] if results.get("distances") else []),
                    "similarities": [results["similarities"][i] for i in best_indices],
                    "query": results.get("query", ""),
                }
            else:
                logger.warning(f"No documents meet absolute minimum threshold {ABSOLUTE_MIN_SIMILARITY}")
                filtered_results = self._empty_results(results.get("query", ""))

        if self.search_config.get("rerank") and len(filtered_results["documents"]) > 1:
            try:
                filtered_results = self._rerank_results(filtered_results)
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")

        if self.search_config.get("diversity") and len(filtered_results["documents"]) > 3:
            try:
                filtered_results = self._diversify_results(filtered_results)
            except Exception as e:
                logger.warning(f"Diversification failed: {e}")

        return filtered_results

    def _rerank_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rerank results by document length (prefer shorter documents).

        Args:
            results: Filtered search results

        Returns:
            Reranked results sorted by document length
        """
        if results.get("documents"):
            docs_with_indices = list(enumerate(results["documents"]))
            docs_with_indices.sort(key=lambda x: len(x[1]))

            new_order = [i for i, _ in docs_with_indices]

            reranked_results = {
                "documents": [results["documents"][i] for i in new_order],
                "metadatas": ([results["metadatas"][i] for i in new_order] if results.get("metadatas") else []),
                "ids": [results["ids"][i] for i in new_order] if results.get("ids") else [],
                "distances": ([results["distances"][i] for i in new_order] if results.get("distances") else []),
                "similarities": [results["similarities"][i] for i in new_order],
                "query": results["query"],
            }

            return reranked_results

        return results

    def _diversify_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Diversify results to avoid near-duplicate content.

        Uses Jaccard similarity on word sets to detect and filter
        overly similar documents.

        Args:
            results: Search results to diversify

        Returns:
            Diversified results with reduced content overlap
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
            "metadatas": ([results["metadatas"][i] for i in selected_indices] if results.get("metadatas") else []),
            "ids": [results["ids"][i] for i in selected_indices] if results.get("ids") else [],
            "distances": ([results["distances"][i] for i in selected_indices] if results.get("distances") else []),
            "similarities": [results["similarities"][i] for i in selected_indices],
            "query": results["query"],
        }
        return diversified_results

    def _process_results(self, results: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """
        Format and enrich search results with metadata.

        Args:
            results: Filtered search results
            original_query: Original user query

        Returns:
            Enriched results with summary, statistics, and source information
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
                "confidence": float(0.0),
                "timestamp": datetime.now().isoformat(),
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
                "min_similarity": (float(min(results["similarities"])) if results["similarities"] else 0.0),
                "max_similarity": (float(max(results["similarities"])) if results["similarities"] else 0.0),
                "threshold_used": self.search_config["score_threshold"],
            },
        }

        return processed_results

    def _create_error_result(self, error: Exception, query: str) -> Dict[str, Any]:
        """Create standardized error result structure."""
        return {
            "documents": [],
            "metadatas": [],
            "ids": [],
            "distances": [],
            "similarities": [],
            "query": query,
            "summary": f"Search Error: {str(error)}",
            "confidence": float(0.0),
            "timestamp": datetime.now().isoformat(),
            "error": str(error),
            "error_type": type(error).__name__,
        }

    def as_runnable(self) -> Callable:
        """
        Convert to LangGraph-compatible runnable function.

        Returns:
            RunnableLambda wrapping the retriever function for LangGraph integration
        """

        def retriever_function(state: GraphState) -> GraphState:
            """
            LangGraph-compatible retriever node function.

            Args:
                state: Current graph state containing query and iteration info

            Returns:
                Updated state with retrieved documents and metadata
            """
            from ..graph_state import StateManager

            logger.info(f"[Retriever Function] Iteration {state['iterations']}")
            logger.info(f"[Retriever Function] Query: {state['search_query']}")

            query = state.get("search_query") or state.get("question")
            results = self.retrieve(query, state)

            new_state = dict(state)

            new_state = StateManager.update_state(
                new_state,
                documents=results.get("documents", []),
                confidence=float(results.get("confidence", 0.0)),
                iterations=state.get("iterations", 0),
                metadata={
                    **state.get("metadata", {}),
                    "retrieval_results": {
                        "summary": results.get("summary"),
                        "stats": results.get("stats", {}),
                        "timestamp": results.get("timestamp"),
                        "query_used": results.get("query"),
                        "metadatas": results.get("metadatas", []),
                    },
                },
            )

            history_entry = {
                "role": "system",
                "action": "retrieval",
                "content": f"Retrieved {len(results.get('documents', []))} documents.",
                "confidence": float(results.get("confidence", 0.0)),
                "details": {
                    "query": state.get("search_query"),
                    "documents_found": len(results.get("documents", [])),
                    "avg_similarity": results.get("stats", {}).get("avg_similarity", 0.0),
                },
            }

            history = list(new_state.get("history", []))
            history.append(history_entry)
            new_state["history"] = history

            logger.info(f"[Retriever Function] Found {len(results.get('documents', []))} documents")
            logger.info(f"[Retriever Function] Confidence: {results.get('confidence', 0.0):.2f}")

            return new_state

        return RunnableLambda(retriever_function)


class RetrieverFactory:
    """
    Factory for creating configured RetrieverNode instances.

    Provides convenient factory methods for common vector store backends.
    """

    @staticmethod
    def create_retriever(
        collection_name: str = "documents",
        persist_directory: str = "data/vector_store/chroma",
        search_config: Optional[Dict[str, Any]] = None,
    ) -> RetrieverNode:
        """
        Create a RetrieverNode configured for ChromaDB.

        Args:
            collection_name: ChromaDB collection name
            persist_directory: Directory for persistent storage
            search_config: Optional search configuration override

        Returns:
            Configured RetrieverNode instance

        Raises:
            Exception: If ChromaDB initialization fails
        """
        from src.vector_store.chroma_db import ChromaDBVectorStore

        try:
            vector_store = ChromaDBVectorStore(
                collection_name=collection_name,
                persist_directory=persist_directory,
                reset_on_start=False,
            )

            logger.info(f"Created ChromaDB retriever for: {collection_name}")
            return RetrieverNode(vector_store=vector_store, search_config=search_config)

        except Exception as e:
            logger.error(f"Failed to create ChromaDB retriever: {e}")
            raise
