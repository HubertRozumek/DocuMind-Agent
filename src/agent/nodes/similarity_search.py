import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import logging

from langsmith import expect

logger = logging.getLogger(__name__)

class SimilarityMetric(Enum):
    """
    Enum class for similarity metrics
    """
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"

@dataclass
class SimilarityConfig:
    """
    Configuration for similarity search

    Attributes:
        metric: Similarity metric
        threshold: Minimum similarity threshold (0-1)
        k: Number of nearest neighbors
        normalize: Whether to normalize vectors
        use_filter: Whether to use filtering metadata
        filters: Dictionary filters to use
        boost_recent: Whether to boost newer document weight
        diversity_penalty: Penalty for too similar documents (0-1)
    """
    metric: SimilarityMetric = SimilarityMetric.COSINE
    threshold: float = 0.7
    k: int = 5
    normalize: bool = True
    use_filter: bool = False
    filters: Optional[Dict[str, Any]] = None
    boost_recent: bool = False
    diversity_penalty: float = 0.3

    def validate(self) -> List[str]:
        """
        Validate configuration and return a list of errors
        """
        errors = []
        if not 0 <= self.threshold <= 1:
            errors.append("Threshold must be between 0 and 1.")
        if self.k <= 0:
            errors.append("k must be greater than 0.")
        if not 0 <= self.diversity_penalty <= 1:
            errors.append("diversity_penalty must be between 0 and 1.")
        return errors


class SimilaritySearch:
    """
    Advanced class for similarity search with configurable parameters.

    Features:
        - Various similarity metrics
        - Threshold-based filtering
        - Increasing the diversity of results
        - Support for metadata filters
        - Vector normalization
    """

    def __init__(self, config: Optional[SimilarityConfig] = None):
        self.config = config or SimilarityConfig()

        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid similarity config: {errors}")

        logger.info(f"Similarity search initialized with config: {self.config}")

    def calculate_similarity(self, query_vector: np.ndarray, documents_vectors: np.ndarray) -> np.ndarray:
        """
        Calculate similarity between query and documents vectors

        Args:
            query_vector: Query vector (1D array)
            documents_vectors: Documents vectors (2D array, n_docs x dim)
        Returns:
            Similarity matrix (1D array, length = n_docs)
        """
        if documents_vectors.shape[0] == 0:
            return np.array([])

        if self.config.normalize:
            query_norm = np.linalg.norm(query_vector)
            if query_norm > 0:
                query_vector = query_vector / query_norm

            doc_norms = np.linalg.norm(documents_vectors, axis=1, keepdims=True)
            doc_norms[doc_norms == 0] = 1
            document_vectors = documents_vectors / doc_norms

        if self.config.metric == SimilarityMetric.COSINE:
            similarities = np.dot(document_vectors, query_vector)

        elif self.config.metric == SimilarityMetric.DOT_PRODUCT:
            similarities = np.dot(document_vectors, query_vector)

        elif self.config.metric == SimilarityMetric.EUCLIDEAN:
            distances = np.linalg.norm(documents_vectors - query_vector, axis=1)
            similarities = 1 / (1 + distances)

        elif self.config.metric == SimilarityMetric.MANHATTAN:
            distances = np.sum(np.abs(document_vectors - query_vector), axis=1)
            similarities = 1 / (1 + distances)

        else:
            raise ValueError(f"Unknown similarity metric: {self.config.metric}")

        return similarities

    def apply_threshold(self,
                        similarities: np.ndarray,
                        documents: List[Any],
                        metadatas: Optional[List[Dict]] = None) -> Tuple[np.ndarray, List[Any], List[Dict]]:
        """
        Filter documents based on similarity threshold.

        Args:
            similarities: Similarity matrix
            documents: List of documents
            metadatas: List of metadatas

        Returns:
            Tuple of filtered similarities, documents, metadata
        """

        if len(similarities) != len(documents):
            raise ValueError(f"Length mismatch: similarities={len(similarities)} documents={len(documents)}")

        above_threshold = similarities >= self.config.threshold

        if not np.any(above_threshold) and len(similarities) > 0:
            logger.warning(f"No documents above threshold: {self.config.threshold}, using top {self.config.k}")
            top_k = min(self.config.k, len(similarities))
            top_indices = np.argpartition(similarities)[-top_k:][::-1]

            filtered_similarities = similarities[top_indices]
            filtered_documents = [documents[i] for i in top_indices]
            filtered_metadatas = [metadatas[i] for i in top_indices] if metadatas else []


        else:

            filtered_similarities = similarities[above_threshold]
            filtered_documents = [documents[i] for i in range(len(documents)) if above_threshold[i]]
            filtered_metadatas = [metadatas[i] for i in range(len(documents)) if above_threshold[i]] if metadatas else []

        return filtered_similarities, filtered_documents, filtered_metadatas

    def apply_diversity(self,
                        similarities: np.ndarray,
                        documents: List[Any],
                        document_vectors: np.ndarray,
                        metadatas: Optional[List[Dict]] = None) -> Tuple[np.ndarray, List[Any], List[Dict]]:
        """
        Increasing the diversity of results by penalizing documents that are too similar.

        Args:
            similarities: Similarity matrix
            documents: List of documents
            document_vectors: Document vectors
            metadatas: List of metadatas

        Returns:
            Different results
        """
        if len(similarities) <= 1 or self.config.diversity_penalty == 0:
            return similarities, documents, metadatas or []

        sorted_indices = np.argsort(similarities)[::-1]

        selected_indices = []
        selected_vectors = []

        for idx in sorted_indices:
            current_vector = document_vectors[idx]
            too_similar = False

            for selected_idx, selected_vector in zip(selected_indices, selected_vectors):
                if self.config.normalize:
                    doc_similarity = np.dot(current_vector, selected_vector)
                else:
                    norm1 = np.linalg.norm(current_vector)
                    norm2 = np.linalg.norm(selected_vector)
                    if norm1 > 0 and norm2 > 0:
                        doc_similarity = np.dot(current_vector, selected_vector) / norm1 * norm2
                    else:
                        doc_similarity = 0

                if doc_similarity > (1 - self.config.diversity_penalty):
                    too_similar = True
                    break

            if not too_similar:
                selected_indices.append(idx)
                selected_vectors.append(current_vector)

            if len(selected_indices) >= self.config.k:
                break

        diversified_similarities = similarities[selected_indices]
        diversified_documents = [documents[i] for i in selected_indices]
        diversified_metadatas = [metadatas[i] for i in selected_indices] if metadatas else []

        return diversified_similarities, diversified_documents, diversified_metadatas

    def boost_recent_documents(self,
                               similarities: np.ndarray,
                               metadatas: List[Dict]) -> np.ndarray:
        """
        Increasing new documents weight

        Args:
            similarities: Similarity matrix
            metadatas: List of metadatas

        Returns:
            Modified similarities with boost for new documents
        """
        if not self.config.boost_recent:
            return similarities

        boosted_similarities = np.copy(similarities)

        for i, metadata in enumerate(metadatas):
            if i >= len(boosted_similarities):
                break

            timestamp = metadata.get('timestamp')
            if timestamp:
                try:
                    from datetime import datetime
                    if isinstance(timestamp, str):
                        doc_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        doc_date = timestamp

                    current_date = datetime.now()
                    age_days = (current_date - doc_date).days

                    boost = np.exp(-age_days / 30)
                    boosted_similarities[i] *= (1 + boost * 0.2)
                    boosted_similarities[i] = min(boosted_similarities[i], 1.0)

                except Exception as e:
                    logger.debug(f"Could not parse timestamp for boost: {e}")

        return boosted_similarities

    def filter_by_metadata(self,
                           similarities: np.ndarray,
                           documents: List[Any],
                           metadatas: List[Dict]) -> Tuple[np.ndarray, List[Any], List[Dict]]:
        """
        Filter documents based on metadata

        Args:
            similarities: Similarity matrix
            documents: List of documents
            metadatas: List of metadatas

        Returns:
            Tuple of filtered similarities, documents, metadata
        """
        if not self.config.use_filter or not self.config.filters:
            return similarities, documents, metadatas

        filtered_indices = []

        for i, metadata in enumerate(metadatas):
            if i >= len(similarities):
                break

            matches_all = True

            for key, value in self.config.filters.items():
                if key not in metadata:
                    matches_all = False
                    break

                if isinstance(value, (list, tuple)):
                    if metadata[key] not in value:
                        matches_all = False
                        break

                elif callable(value):
                    if not value(metadata[key]):
                        matches_all = False
                        break
                else:
                    if metadata[key] != value:
                        matches_all = False
                        break

            if matches_all:
                filtered_indices.append(i)

        filtered_similarities = similarities[filtered_indices]
        filtered_documents = [documents[i] for i in filtered_indices]
        filtered_metadatas = [metadatas[i] for i in filtered_indices]

        return filtered_similarities, filtered_documents, filtered_metadatas

    def search(self,
               query_vector: np.ndarray,
               document_vectors: np.ndarray,
               documents: List[Any],
               metadatas: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Complete similarity search process with all features.

        Args:
            query_vector: Query vector
            document_vectors: Document vectors
            documents: List of documents
            metadatas: List of metadatas

        Returns:
            Search results
        """
        if len(document_vectors) == 0:
            return {
                "documents": [],
                "metadatas": [],
                "similarities": [],
                "filtered": False,
                "threshold_used": self.config.threshold,
            }

        similarities = self.calculate_similarity(query_vector, document_vectors)

        if metadatas is None:
            metadatas = [{} for _ in range(len(documents))]

        if self.config.use_filter and self.config.filters:
            similarities, documents, metadatas = self.filter_by_metadata(similarities, documents, metadatas)

        if self.config.boost_recent and metadatas:
            similarities = self.boost_recent_documents(similarities, metadatas)

        similarities, docuemnts, metadatas = self.apply_threshold(similarities, documents, metadatas)

        if len(documents) > 1 and self.config.diversity_penalty > 0:
            similarities, docuemnts, metadatas = self.apply_diversity(similarities, docuemnts, document_vectors, metadatas)

        if len(similarities) > self.config.k:
            top_indices = np.argpartition(similarities)[-self.config.k][::-1]
            similarities = similarities[top_indices]
            documents = [documents[i] for i in top_indices]
            metadatas = [metadatas[i] for i in top_indices]

        return {
            "documents": documents,
            "metadatas": metadatas,
            "similarities": similarities.tolist(),
            "filtered": True,
            "threshold_used": self.config.threshold,
            "metric_used": self.config.metric.value,
            "stats": {
                "total_before_filtering": len(document_vectors),
                "total_after_filtering": len(documents),
                "avg_similarity": float(np.mean(similarities)) if len(similarities) > 0 else 0.0,
                "max_similarity": float(np.max(similarities)) if len(similarities) > 0 else 0.0,
                "min_similarity": float(np.min(similarities)) if len(similarities) > 0 else 0.0
            }
        }


class ThresholdOptimizer:
    """
    Class to optimize similarity threshold based on data
    """
    def __init__(self, validation_data: List[Dict[str, Any]]):
        """
        Args:
            validation_data: Validation data in format:
            [
                {
                "query": "query",
                "relevant_docs": ["doc_id1", "doc_id2"],
                "irrelevant_docs": ["doc_id3", "doc_id4"],
                },
                ...
            ]
        """
        self.validation_data = validation_data

    def find_optimal_threshold(self, thresholds: List[float] = None) -> Dict[str, Any]:
        """
        Finds the optimal threshold based on quality metrics.
        Args:
            thresholds: List of thresholds
        Returns:
            Dictionary of optimal thresholds
        """
        if thresholds is None:
            thresholds = [i/10 for i in range(1, 10)]

        results = []

        for threshold in thresholds:
            metrics = self.evaluate_threshold(threshold)
            results.append({
                "threshold": threshold,
                "metric": metrics,
                "score": self._calculate_score(metrics)
            })

        best_result = max(results, key=lambda r: r["score"])

        return {
            "optimal_threshold": best_result["threshold"],
            "optimal_metric": best_result["metric"],
            "all_results": results,
            "recommendation": self._generate_recommendation(best_result)
        }

    def evaluate_threshold(self, threshold: float) -> Dict[str, Any]:
        """
        Evaluate threshold based on validation data.
        Returns: Dictionary with quality metrics
        """
        precisions = []
        recalls = []
        f1_scores = []

        for data_point in self.validation_data:
            relevant_docs = set(data_point.get("relevant_docs", []))
            retrieved_docs = set()

            similarities = data_point.get("similarities", {})

            for doc_id, similarity in similarities.items():
                if similarity >= threshold:
                    retrieved_docs.add(doc_id)

            true_positives = len(relevant_docs.intersection(retrieved_docs))
            false_positives = len(retrieved_docs - relevant_docs)
            false_negatives = len(relevant_docs - retrieved_docs)

            if true_positives + false_positives > 0:
                recall = true_positives / (true_positives + false_positives)
            else:
                precision = 0.0

            if true_positives + false_negatives > 0:
                recall = true_positives / (true_positives + false_negatives)
            else:
                recall = 0.0

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        avg_precision = np.mean(precisions) if precisions else 0.0
        avg_recall = np.mean(recalls) if recalls else 0.0
        avg_f1_score = np.mean(f1_scores) if f1_scores else 0.0

        return {
            "precision": float(avg_precision),
            "recall": float(avg_recall),
            "f1_score": float(avg_f1_score),
            "std_precision": float(np.std(precisions)) if precisions else 0.0,
            "std_recall": float(np.std(recalls)) if recalls else 0.0,
        }

    def _calculate_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate score based on metrics
        """
        return (
            0.4 * metrics["precision"] +
            0.4 * metrics["recall"] +
            0.2 * metrics["f1_score"]
        )

    def _generate_recommendation(self, best_result: Dict[str, Any]) -> str:
        """
        Generate recommendation based on best result
        """
        threshold = best_result["threshold"]
        metrics = best_result["metrics"]

        if metrics["precision"] > 0.8 and metrics["recall"] > 0.8:
            return f"Threshold {threshold} gives perfect quality (precision={metrics['precision']}, recall={metrics['recall']})"
        elif metrics["precision"] > 0.7:
            return f"Threshold {threshold} gives good precision (precision={metrics['precision']} at the expense recall={metrics['recall']})"
        elif metrics["recall"] > 0.7:
            return f"Threshold {threshold} gives good recall (recall={metrics['recall']} at the expense precision={metrics['precision']})"
        else:
            return f"Threshold {threshold} gives moderate results (precision={metrics['precision']}, recall={metrics['recall']})"

    def create_similarity_search_runnable(config: Optional[SimilarityConfig] = None) -> Callable:
        """
        Creates a LangChain compatible runnable function for similarity search.

        Returns:
            A function that accepts a state and returns a updated state.
        :return:
        """
        similarity_search = SimilaritySearch(config)

        def similarity_search_function(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Similarity search function for LangGraph

            Requires:
                -query_embedding: query embedding
                -document_embedding: document embedding
                -documents: list of documents
                -metadatas: list of metadatas
            Returns:
                State with similarity results
            """
            from src.agent.graph_state import  StateManager

            query_embedding = state.get("query_embedding")
            document_embedding = state.get("document_embeddings")
            documents = state.get("documents", [])
            metadatas = state.get("metadatas", [])

            if query_embedding is None or document_embedding is None:
                logger.error("Missing embeddings in state")
                return StateManager.update_state(
                    state,
                    error = 'Missing embeddings for similarity search',
                    confidence = 0.0
                )

            try:
                results = similarity_search.search(
                    query_vector = np.array(query_embedding),
                    document_vectors = np.array(document_embedding),
                    documents = documents,
                    metadatas = metadatas
                )

                updated_state = StateManager.update_state(
                    state,
                    documents = results["documents"],
                    confidence = results["stats"]["avg_similarity"],
                    metadata = {
                        **state.get("metadata", {}),
                        "similarity_search": {
                            "stats": results["stats"],
                            "threshold_used": results["threshold_used"],
                            "metric_used": results["metric_used"],
                            "filtered": results["filtered"],
                        }
                    }
                )
                logger.info(f"[SimilaritySearch] Found {len(results['documents'])} documents")
                logger.info(f"[SimilaritySearch] Avg similarity: {results['stats']['avg_similarity']:.2f}")
                return updated_state

            except Exception as e:
                logger.error(f"[SimilaritySearch] Error occured: {e}")
                return StateManager.update_state(
                    state,
                    error = f'Similarity search failed with error: {e}',
                    confidence = 0.0
                )

        return similarity_search_function
