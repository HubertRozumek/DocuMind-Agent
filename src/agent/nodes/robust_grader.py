import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_semantic_model_instance = None
_model_lock = threading.Lock()
_model_loading = False


def get_semantic_model():
    """
    Thread-safe singleton for sentence transformer model.

    Returns:
        SentenceTransformer: Loaded model instance

    Raises:
        Exception: If model loading fails
    """
    global _semantic_model_instance, _model_loading

    if _semantic_model_instance is not None:
        return _semantic_model_instance

    with _model_lock:
        if _semantic_model_instance is not None:
            return _semantic_model_instance

        if _model_loading:
            logger.info("Model already being loaded by another thread, waiting...")
            while _model_loading:
                time.sleep(0.1)
            return _semantic_model_instance

        _model_loading = True

        try:
            import torch
            from sentence_transformers import SentenceTransformer

            logger.info("Loading semantic model (singleton)...")

            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

            logger.info(f"Using device: {device}")

            _semantic_model_instance = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)

            logger.info("Semantic model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load semantic model: {e}")
            _model_loading = False
            raise
        finally:
            _model_loading = False

    return _semantic_model_instance


class RelevanceScore(Enum):
    """
    Document relevance classification levels.

    Attributes:
        HIGHLY_RELEVANT: Document directly answers the question
        RELEVANT: Document partially answers the question
        SOMEWHAT_RELEVANT: Topically related but not directly answering
        WEAKLY_RELEVANT: Marginally related
        NOT_RELEVANT: Unrelated to the question
    """

    HIGHLY_RELEVANT = 5
    RELEVANT = 4
    SOMEWHAT_RELEVANT = 3
    WEAKLY_RELEVANT = 2
    NOT_RELEVANT = 1


@dataclass
class GradingResult:
    """
    Result of document grading operation.

    Attributes:
        score: Relevance classification level
        confidence: Confidence score (0.0-1.0)
        reason: Human-readable explanation of the grading
        method: Grading method used ("llm", "fallback_keyword", "fallback_semantic")
    """

    score: RelevanceScore
    confidence: float
    reason: str
    method: str

    def is_relevant(self, threshold: Optional[float] = None) -> bool:
        """
        Determine if document meets relevance threshold.

        Applies method-specific thresholds with stricter requirements for
        fallback methods to reduce false positives.

        Args:
            threshold: Minimum confidence threshold (default 0.6)

        Returns:
            True if document is considered relevant
        """
        if threshold is None:
            threshold = 0.6

        if self.method.startswith("fallback"):
            threshold = max(threshold, 0.4)
            return self.confidence >= threshold and self.score.value >= 3

        min_confidence_for_score = {
            RelevanceScore.HIGHLY_RELEVANT: 0.7,
            RelevanceScore.RELEVANT: 0.5,
            RelevanceScore.SOMEWHAT_RELEVANT: 0.3,
            RelevanceScore.WEAKLY_RELEVANT: 0.4,
            RelevanceScore.NOT_RELEVANT: 1.0,
        }

        min_required = min_confidence_for_score.get(self.score, 0.5)
        effective_threshold = max(threshold, min_required)

        return self.confidence >= effective_threshold and self.score.value >= 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "relevant": self.is_relevant(),
            "confidence": round(self.confidence, 2),
            "reason": self.reason,
            "score": self.score.name,
            "score_value": self.score.value,
            "method": self.method,
        }


@dataclass
class GraderConfig:
    """
    Centralized configuration for grading thresholds and behavior.

    Attributes:
        llm_min_confidence: Minimum confidence for accepting LLM results
        llm_retry_count: Number of retries for LLM grading
        semantic_highly_relevant: Cosine similarity threshold for high relevance
        semantic_relevant: Cosine similarity threshold for relevance
        semantic_somewhat_relevant: Cosine similarity threshold for partial relevance
        semantic_weakly_relevant: Cosine similarity threshold for weak relevance
        keyword_highly_relevant: Keyword overlap threshold for high relevance
        keyword_relevant: Keyword overlap threshold for relevance
        keyword_somewhat_relevant: Keyword overlap threshold for partial relevance
        fallback_max_confidence: Maximum confidence cap for fallback methods
        fallback_min_threshold: Absolute minimum threshold for fallback acceptance
        relevance_threshold: Default threshold for is_relevant() method
        skip_fallback_on_clear_verdict: Skip fallbacks if LLM confidence is clear
    """

    llm_min_confidence: float = 0.6
    llm_retry_count: int = 2
    semantic_highly_relevant: float = 0.75
    semantic_relevant: float = 0.55
    semantic_somewhat_relevant: float = 0.40
    semantic_weakly_relevant: float = 0.25
    keyword_highly_relevant: float = 0.60
    keyword_relevant: float = 0.40
    keyword_somewhat_relevant: float = 0.25
    fallback_max_confidence: float = 0.60
    fallback_min_threshold: float = 0.30
    relevance_threshold: float = 0.60
    skip_fallback_on_clear_verdict: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if not (0 <= self.relevance_threshold <= 1):
            raise ValueError("relevance_threshold must be between 0 and 1")

        if not self.semantic_highly_relevant > self.semantic_relevant:
            raise ValueError("semantic_highly_relevant must be greater than semantic_relevant")

        if not self.semantic_relevant > self.semantic_somewhat_relevant:
            raise ValueError("semantic_relevant must be greater than semantic_somewhat_relevant")


class RobustGrader:
    """
    Multi-layer document relevance grader with fallback mechanisms.

    Implements a three-tier grading strategy:
    1. LLM-based grading (primary)
    2. Semantic similarity fallback (embedding-based)
    3. Keyword matching fallback (final)

    Attributes:
        model_name: Name of the Ollama model for LLM grading
        config: GraderConfig instance with thresholds
    """

    def __init__(self, model_name: str = "phi3:mini", config: Optional[GraderConfig] = None):
        """
        Initialize the robust grader.

        Args:
            model_name: Ollama model name for LLM grading
            config: Optional custom configuration
        """
        self.model_name = model_name
        self.config = config or GraderConfig()

    def grade(self, question: str, document: str, metadata: Optional[Dict] = None) -> GradingResult:
        """
        Grade document relevance using multi-layer strategy.

        Attempts LLM grading first, falls back to semantic similarity,
        then keyword matching if needed.

        Args:
            question: User query to evaluate against
            document: Document content to grade
            metadata: Optional document metadata for confidence boosting

        Returns:
            GradingResult with score, confidence, and method used
        """
        logger.info(f"Grading document with metadata: {metadata is not None}")

        llm_result = None
        try:
            llm_result = self._grade_with_llm(question, document)

            if self.config.skip_fallback_on_clear_verdict:
                if llm_result.confidence >= 0.85 or llm_result.confidence <= 0.15:
                    logger.info(f"LLM confidence clear ({llm_result.confidence:.2f}), skipping fallback")
                    return llm_result

            if llm_result.confidence >= self.config.llm_min_confidence:
                logger.info(f"LLM confidence sufficient ({llm_result.confidence:.2f}), using LLM result")
                return llm_result

            if metadata and metadata.get("source_type") == "policy":
                boosted_confidence = min(0.9, llm_result.confidence * 1.2)
                if boosted_confidence >= self.config.llm_min_confidence:
                    from dataclasses import replace

                    boosted_result = replace(llm_result, confidence=boosted_confidence)
                    logger.info(f"Boosted confidence for policy: {boosted_confidence:.2f}")
                    return boosted_result

            logger.info(f"LLM confidence low ({llm_result.confidence:.2f}), trying fallbacks")

        except Exception as e:
            logger.warning(f"LLM grading failed: {e}")

        try:
            semantic_result = self._semantic_similarity(question, document)
            if semantic_result.confidence >= self.config.fallback_min_threshold:
                logger.info(f"Using semantic fallback with confidence {semantic_result.confidence:.2f}")
                return semantic_result
        except Exception as e:
            logger.warning(f"Semantic fallback failed: {e}")

        keyword_result = self._keyword_fallback(question, document)
        logger.info(f"Using keyword fallback with confidence {keyword_result}")
        return keyword_result

    def _create_ultra_precise_prompt(self, question: str, document: str) -> Tuple[str, str]:
        """
        Create evaluation prompt optimized for consistent LLM responses.

        Args:
            question: User question to evaluate against
            document: Document content to evaluate

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = """You are a DOCUMENT EVALUATOR. Your ONLY job is to rate DOCUMENT quality for answering QUESTION.

        CRITICAL RULES:
        1. DO NOT answer the QUESTION
        2. DO NOT explain the topic
        3. ONLY evaluate if DOCUMENT contains information to answer QUESTION
        4. Output STRICTLY valid JSON

        Rating scale:
        5 = Document perfectly answers the question
        4 = Document partially answers the question
        3 = Document is related but doesn't directly answer
        2 = Document is weakly related
        1 = Document is unrelated

        OUTPUT FORMAT (copy exactly, fill values):
        {
          "rating": 5,
          "confidence": 0.95,
          "explanation": "One sentence why this rating"
        }"""

        doc_preview = document[:1500].replace('"', '\\"').replace("\n", " ")

        user_prompt = f"""QUESTION TO EVALUATE AGAINST: "{question}"

        DOCUMENT TO EVALUATE: "{doc_preview}"

        Remember: You are EVALUATING, not ANSWERING.
        Rate how well DOCUMENT helps answer QUESTION.

        JSON output:"""

        return system_prompt, user_prompt

    def _grade_with_llm(self, question: str, document: str) -> GradingResult:
        """
        Grade document using LLM with robust JSON parsing.

        Args:
            question: User question
            document: Document to grade

        Returns:
            GradingResult from LLM evaluation

        Raises:
            Exception: If LLM grading or parsing fails
        """
        system, user = self._create_ultra_precise_prompt(question, document)

        try:
            from src.agent.nodes.grader_model import OllamaGrader

            ollama = OllamaGrader(model_name=self.model_name)

            response = ollama.grade(user, system_prompt=system)

            logger.debug(f"[LLM Response] First 300 chars: {response[:300]}")

            parsed = None

            try:
                parsed = json.loads(response.strip())
                logger.debug(f"[LLM Parsed] Direct JSON success: {parsed}")
            except json.JSONDecodeError:
                pass

            if not parsed:
                parsed = self._extract_json_aggressive(response)
                logger.debug(f"[LLM Parsed] Aggressive extraction: {parsed}")

            if not parsed:
                raise ValueError("Could not extract JSON from LLM response")

            rating = parsed.get("rating")
            relevant = parsed.get("relevant")
            confidence = parsed.get("confidence", 0.0)

            if isinstance(relevant, bool):
                rating = 5 if relevant else 1
            elif isinstance(rating, bool):
                rating = 5 if rating else 1
            elif rating is None:
                if "relevant" in str(response).lower():
                    rating = 5
                    confidence = max(confidence, 0.7)
                else:
                    rating = 3
                    confidence = 0.5

            try:
                rating = int(rating)
                confidence = float(confidence)
            except (ValueError, TypeError):
                rating = 3
                confidence = 0.5

            score_map = {
                5: RelevanceScore.HIGHLY_RELEVANT,
                4: RelevanceScore.RELEVANT,
                3: RelevanceScore.SOMEWHAT_RELEVANT,
                2: RelevanceScore.WEAKLY_RELEVANT,
                1: RelevanceScore.NOT_RELEVANT,
            }
            score = score_map.get(rating, RelevanceScore.SOMEWHAT_RELEVANT)

            logger.info(f"[LLM Result] rating={rating}, confidence={confidence}, score={score.name}")

            return GradingResult(
                score=score,
                confidence=confidence,
                reason=parsed.get("explanation", parsed.get("reason", "LLM graded")),
                method="llm",
            )

        except Exception as e:
            logger.error(f"LLM grading failed: {e}")
            raise

    def _extract_json_aggressive(self, text: str) -> Optional[Dict]:
        """
        Multi-strategy JSON extractor for robust LLM response parsing.

        Attempts multiple extraction strategies:
        1. Markdown JSON blocks
        2. Generic code blocks
        3. Objects with specific fields
        4. Any JSON-like structure
        5. Linear key-value extraction

        Args:
            text: Raw LLM response text

        Returns:
            Parsed dictionary or None if extraction fails
        """
        patterns = [
            r"```json\s*(\{.*?\})\s*```",
            r"```\s*(\{.*?\})\s*```",
            r'(\{[^{}]*"rating"[^{}]*\})',
            r'(\{[^{}]*"relevant"[^{}]*\})',
            r"(\{.*\})",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    cleaned = match.strip()
                    cleaned = re.sub(r",\s*}", "}", cleaned)
                    cleaned = re.sub(r"\n\s*", " ", cleaned)
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    continue

        result = {}

        rating_match = re.search(r'"rating"\s*:\s*(\d)', text)
        if rating_match:
            result["rating"] = int(rating_match.group(1))

        relevant_match = re.search(r'"relevant"\s*:\s*(true|false)', text, re.I)
        if relevant_match:
            result["relevant"] = relevant_match.group(1).lower() == "true"

        conf_match = re.search(r'"confidence"\s*:\s*(0?\.\d+|1\.0)', text)
        if conf_match:
            result["confidence"] = float(conf_match.group(1))

        exp_match = re.search(r'"explanation"\s*:\s*"([^"]+)"', text)
        if exp_match:
            result["explanation"] = exp_match.group(1)

        return result if result else None

    def _semantic_similarity(self, question: str, document: str) -> GradingResult:
        """
        Calculate relevance using sentence transformer embeddings.

        Uses cosine similarity between question and document embeddings
        with configured thresholds for relevance classification.

        Args:
            question: User question
            document: Document to evaluate

        Returns:
            GradingResult with semantic similarity score
        """
        import numpy as np

        try:
            model = get_semantic_model()
            if not hasattr(self, "_question_cache"):
                self._question_cache = {}

            if question not in self._question_cache:
                self._question_cache[question] = model.encode([question])[0]

            q_embedding = self._question_cache[question]
            d_embedding = model.encode([document])[0]
            similarity = np.dot(q_embedding, d_embedding) / (np.linalg.norm(q_embedding) * np.linalg.norm(d_embedding))

            if similarity > self.config.semantic_highly_relevant:
                score = RelevanceScore.HIGHLY_RELEVANT
            elif similarity > self.config.semantic_relevant:
                score = RelevanceScore.RELEVANT
            elif similarity > self.config.semantic_somewhat_relevant:
                score = RelevanceScore.SOMEWHAT_RELEVANT
            elif similarity > self.config.semantic_weakly_relevant:
                score = RelevanceScore.WEAKLY_RELEVANT
            else:
                score = RelevanceScore.NOT_RELEVANT
                similarity = max(0.0, similarity)

            adjusted_confidence = min(float(similarity), self.config.fallback_max_confidence)

            return GradingResult(
                score=score,
                confidence=adjusted_confidence,
                reason=f"Semantic similarity: {similarity:.2f} (fallback)",
                method="fallback_semantic",
            )

        except Exception as e:
            logger.error(f"Semantic similarity failed: {e}")
            return self._keyword_fallback(question, document)

    def _keyword_fallback(self, question: str, document: str) -> GradingResult:
        """
        Calculate relevance using keyword overlap analysis.

        Extracts keywords from question and document, calculates overlap
        with penalty for missing question keywords.

        Args:
            question: User question
            document: Document to evaluate

        Returns:
            GradingResult based on keyword coverage
        """
        import re

        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}

        def extract_keywords(text: str) -> set:
            words = re.findall(r"\b\w{4,}\b", text.lower())
            return set(w for w in words if w not in stop_words)

        q_keywords = extract_keywords(question)
        d_keywords = extract_keywords(document)

        if not q_keywords:
            return GradingResult(
                score=RelevanceScore.NOT_RELEVANT,
                confidence=0.1,
                reason="No valid keywords in question",
                method="fallback_keyword",
            )

        overlap = q_keywords & d_keywords
        coverage = len(overlap) / len(q_keywords) if q_keywords else 0

        q_unique = q_keywords - d_keywords
        missing_penalty = len(q_unique) / len(q_keywords) if q_keywords else 0
        adjusted_coverage = coverage * (1 - missing_penalty)

        if adjusted_coverage > self.config.keyword_highly_relevant:
            score = RelevanceScore.HIGHLY_RELEVANT
        elif adjusted_coverage > self.config.keyword_relevant:
            score = RelevanceScore.RELEVANT
        elif adjusted_coverage > self.config.keyword_somewhat_relevant:
            score = RelevanceScore.SOMEWHAT_RELEVANT
        elif adjusted_coverage > 0.1:
            score = RelevanceScore.WEAKLY_RELEVANT
        else:
            score = RelevanceScore.NOT_RELEVANT

        return GradingResult(
            score=score,
            confidence=float(adjusted_coverage),
            reason=f"Keyword coverage: {len(overlap)}/{len(q_keywords)} words (adjusted: {adjusted_coverage:.2f})",
            method="fallback_keyword",
        )

    def grade_batch(self, question: str, documents: List[str]) -> List[GradingResult]:
        """
        Grade multiple documents with early exit optimization.

        Processes documents sequentially, stopping early if a highly
        relevant document is found (confidence > 0.9).

        Args:
            question: User question
            documents: List of documents to grade

        Returns:
            List of GradingResult objects
        """
        results = []

        for i, doc in enumerate(documents):
            logger.info(f"Grading document {i + 1}/{len(documents)}")
            result = self.grade(question, doc)
            results.append(result)

            if result.score == RelevanceScore.HIGHLY_RELEVANT and result.confidence > 0.9:
                logger.info("Found highly relevant document, skipping remaining")
                for remaining_doc in documents[i + 1 :]:
                    quick = self._keyword_fallback(question, remaining_doc)
                    results.append(quick)
                break

        return results
