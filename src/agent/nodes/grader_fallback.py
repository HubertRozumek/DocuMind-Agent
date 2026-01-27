import logging
import time
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class FallbackStrategy(Enum):
    """
    Fallback strategies for a grader.
    """
    KEYWORD_MATCH = "keyword_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    RULE_BASED = "rule_based"
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    HYBRID = "hybrid"


class FallbackResult:
    """
    Fallback result with confidence score.
    """

    def __init__(self, relevant: bool, confidence: float, method: str, reason: str):
        self.relevant = relevant
        self.confidence = confidence
        self.method = method
        self.reason = reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relevant": self.relevant,
            "confidence": self.confidence,
            "method": self.method,
            "reason": self.reason,
            "timestamp": datetime.now().isoformat()
        }


class BaseFallbackGrader:
    """
    Base class for grader fallback strategies.
    """

    def __init__(self, name: str, confidence_weight: float = 1.0):
        self.name = name
        self.confidence_weight = confidence_weight

    def grade(self, question: str, document: str, metadata: Optional[Dict] = None) -> FallbackResult:
        raise NotImplementedError

    def get_weighted_confidence(self, confidence: float) -> float:
        return confidence * self.confidence_weight


class KeywordFallbackGrader(BaseFallbackGrader):
    """
    Fallback based on keywords.
    """

    def __init__(self, min_keyword_match: int = 2, **kwargs):
        super().__init__("keyword_matcher", **kwargs)
        self.min_keyword_match = min_keyword_match
        self.stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]


    def grade(self, question: str, document: str, metadata: Optional[Dict] = None) -> FallbackResult:
        """
        Keyword Matching Rating.
        """
        try:
            question_keywords = self._extract_keywords(question)
            document_keywords = self._extract_keywords(document)

            common_keywords = set(question_keywords) & set(document_keywords)
            match_count = len(common_keywords)

            max_possible = min(len(question_keywords), len(document_keywords))
            if max_possible > 0:
                confidence = match_count / max_possible
            else:
                confidence = 0.0

            relevant = match_count >= self.min_keyword_match

            reason = f"Keyword match: {match_count} common keywords ({common_keywords})"

            return FallbackResult(
                relevant=relevant,
                confidence=confidence,
                method=self.name,
                reason=reason
            )

        except Exception as e:
            logger.error(f"Keyword fallback failed: {e}")
            return self._error_result(f"Keyword fallback error: {e}")

    def _extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
        """
        Extract keywords from text.
        """

        words = text.lower().split()

        keywords = []
        for word in words:
            word = ''.join(c for c in word if c.isalnum())

            if (len(word) > 3 and
                    word not in self.stop_words and
                    not word.isdigit()):
                keywords.append(word)

        unique_keywords = list(set(keywords))

        return unique_keywords[:max_keywords]

    def _error_result(self, error_msg: str) -> FallbackResult:
        """
        Return error result.
        """
        return FallbackResult(
            relevant=False,
            confidence=0.1,
            method=self.name,
            reason=f"Error: {error_msg}"
        )


class SemanticFallbackGrader(BaseFallbackGrader):
    """
    Fallback based on semantic similarity.
    """

    def __init__(self, similarity_threshold: float = 0.7, **kwargs):
        super().__init__("semantic_matcher", **kwargs)
        self.similarity_threshold = similarity_threshold

        self.embedding_model = None
        self._load_embedding_model()

    def _load_embedding_model(self):
        """
        Load embedding model.
        """
        try:
            from sentence_transformers import SentenceTransformer

            self.embedding_model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                device="cpu"
            )
            logger.info("Semantic fallback: Embedding model loaded")

        except ImportError:
            logger.warning("SentenceTransformers not installed for semantic fallback")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")

    def grade(self, question: str, document: str, metadata: Optional[Dict] = None) -> FallbackResult:
        """
        Grade based on semantic similarity.
        """
        if self.embedding_model is None:
            return self._error_result("Embedding model not available")

        try:
            embeddings = self.embedding_model.encode([question, document])
            question_embedding = embeddings[0]
            document_embedding = embeddings[1]

            similarity = np.dot(question_embedding, document_embedding) / (
                    np.linalg.norm(question_embedding) * np.linalg.norm(document_embedding)
            )

            relevant = similarity >= self.similarity_threshold

            confidence = self._similarity_to_confidence(similarity)

            reason = f"Semantic similarity: {similarity:.3f} (threshold: {self.similarity_threshold})"

            return FallbackResult(
                relevant=relevant,
                confidence=confidence,
                method=self.name,
                reason=reason
            )

        except Exception as e:
            logger.error(f"Semantic fallback failed: {e}")
            return self._error_result(f"Semantic fallback error: {e}")

    def _similarity_to_confidence(self, similarity: float) -> float:
        """
        Mapping similarity to confidence score.
        """
        if similarity < 0:
            return 0.1
        elif similarity > 1:
            return 0.9

        return 1 / (1 + np.exp(-10 * (similarity - 0.5)))


class RuleBasedFallbackGrader(BaseFallbackGrader):
    """
    Rule-based fallback.
    """

    def __init__(self, **kwargs):
        super().__init__("rule_based", **kwargs)
        self.rules = self._initialize_rules()

    def _initialize_rules(self) -> List[Dict[str, Any]]:
        """
        Initializes evaluation rules
        """
        return [
            {
                "name": "negative_phrases",
                "pattern": ["no information", "i don't know", "not applicable", "there is no"],
                "action": "not_relevant",
                "weight": 0.8
            },
            {
                "name": "answer_phrases",
                "pattern": ["answer:", "information:", "procedure:", "step"],
                "action": "relevant",
                "weight": 0.7
            },
            {
                "name": "question_words",
                "condition": lambda q, d: any(word in q.lower() for word in ["how", "why", "when"]),
                "action": "check_procedure",
                "weight": 0.6
            },
            {
                "name": "document_length",
                "condition": lambda q, d: len(d) < 50,
                "action": "not_relevant",
                "weight": 0.9
            },
            {
                "name": "direct_match",
                "condition": lambda q, d: any(q_word in d.lower() for q_word in q.lower().split()[:5]),
                "action": "relevant",
                "weight": 0.5
            }
        ]

    def grade(self, question: str, document: str, metadata: Optional[Dict] = None) -> FallbackResult:
        """
        Grade based on rule-based fallback.
        """
        try:
            scores = []
            reasons = []

            question_lower = question.lower()
            document_lower = document.lower()

            for rule in self.rules:
                rule_applies = False

                if "pattern" in rule:
                    for pattern in rule["pattern"]:
                        if pattern in document_lower:
                            rule_applies = True
                            reasons.append(f"Rule '{rule['name']}': pattern '{pattern}' found")
                            break

                elif "condition" in rule:
                    if rule["condition"](question, document):
                        rule_applies = True
                        reasons.append(f"Rule '{rule['name']}': condition met")

                if rule_applies:
                    if rule["action"] == "relevant":
                        scores.append(rule["weight"])
                    elif rule["action"] == "not_relevant":
                        scores.append(-rule["weight"])
                    elif rule["action"] == "check_procedure":
                        if any(word in document_lower for word in ["procedure", "step", "instruction"]):
                            scores.append(0.6)
                            reasons.append(f"Rule '{rule['name']}': procedure found")

            if scores:
                avg_score = sum(scores) / len(scores)
                relevant = avg_score > 0
                confidence = abs(avg_score)
            else:
                relevant = False
                confidence = 0.3
                reasons.append("No rules applied")

            reason = f"Rule-based: {len([s for s in scores if s > 0])} pro, {len([s for s in scores if s < 0])} con. " + "; ".join(
                reasons[:3])

            return FallbackResult(
                relevant=relevant,
                confidence=confidence,
                method=self.name,
                reason=reason
            )

        except Exception as e:
            logger.error(f"Rule-based fallback failed: {e}")
            return self._error_result(f"Rule-based fallback error: {e}")


class FallbackManager:
    """
    Fallback Strategy Manager.
    """

    def __init__(self, strategies: Optional[List[FallbackStrategy]] = None):
        self.strategies = strategies or [
            FallbackStrategy.SEMANTIC_SIMILARITY,
            FallbackStrategy.KEYWORD_MATCH,
            FallbackStrategy.RULE_BASED,
            FallbackStrategy.CONFIDENCE_THRESHOLD
        ]

        self.fallback_graders = self._initialize_fallback_graders()

        self.min_confidence = 0.6
        self.retry_count = 2
        self.timeout = 10

        logger.info(f"FallbackManager initialized with strategies: {[s.value for s in self.strategies]}")

    def _initialize_fallback_graders(self) -> Dict[str, BaseFallbackGrader]:
        """
        Initializes fallback graders
        """
        graders = {}

        if FallbackStrategy.KEYWORD_MATCH in self.strategies:
            graders["keyword"] = KeywordFallbackGrader(min_keyword_match=2, confidence_weight=0.8)

        if FallbackStrategy.SEMANTIC_SIMILARITY in self.strategies:
            try:
                graders["semantic"] = SemanticFallbackGrader(similarity_threshold=0.65, confidence_weight=0.9)
            except:
                logger.warning("Semantic fallback not available")

        if FallbackStrategy.RULE_BASED in self.strategies:
            graders["rule"] = RuleBasedFallbackGrader(confidence_weight=0.7)

        return graders

    def execute_fallback(self,
                         question: str,
                         document: str,
                         metadata: Optional[Dict] = None,
                         llm_grade: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Executes the fallback strategy

        Args:
            question: User question
            document: Document to be graded
            metadata: Document metadata
            llm_grade: LLM grade, if available

        Returns:
            Final fallback grade
        """
        start_time = time.time()

        if llm_grade and self._is_confident_llm_grade(llm_grade):
            logger.info("LLM grade confident, no fallback needed")
            return {
                **llm_grade,
                "fallback_used": False,
                "fallback_method": "none",
                "processing_time": time.time() - start_time
            }

        fallback_results = []

        for name, grader in self.fallback_graders.items():
            try:
                result = grader.grade(question, document, metadata)
                weighted_confidence = grader.get_weighted_confidence(result.confidence)

                fallback_results.append({
                    "grader": name,
                    "result": result,
                    "weighted_confidence": weighted_confidence
                })

                logger.debug(f"Fallback {name}: relevant={result.relevant}, confidence={result.confidence:.2f}")

            except Exception as e:
                logger.error(f"Fallback grader {name} failed: {e}")

        final_result = self._combine_fallback_results(
            fallback_results, llm_grade, question, document
        )

        final_result["fallback_used"] = True
        final_result["processing_time"] = time.time() - start_time
        final_result["fallback_methods_used"] = [r["grader"] for r in fallback_results]

        logger.info(f"Fallback completed: relevant={final_result['relevant']}, "
                    f"confidence={final_result['confidence']:.2f}, "
                    f"methods={final_result['fallback_methods_used']}")

        return final_result

    def _is_confident_llm_grade(self, llm_grade: Dict[str, Any]) -> bool:
        """
        Checks whether the LLM assessment has sufficient certainty
        """

        confidence = llm_grade.get("confidence", 0)
        if confidence >= self.min_confidence:
            return True

        if "relevant" in llm_grade and llm_grade["relevant"] is not None:
            reason = llm_grade.get("reason", "").lower()
            if any(word in reason for word in ["directly", "definitely", "obvious"]):
                return True

        return False

    def _combine_fallback_results(self,
                                  fallback_results: List[Dict[str, Any]],
                                  llm_grade: Optional[Dict[str, Any]],
                                  question: str,
                                  document: str) -> Dict[str, Any]:
        """
        Combines results from different fallback strategies.
        """
        if not fallback_results:
            return self._default_fallback_result(question, document)

        relevant_results = [r for r in fallback_results if r["result"].relevant]
        not_relevant_results = [r for r in fallback_results if not r["result"].relevant]

        relevant_confidence = sum(r["weighted_confidence"] for r in relevant_results)
        not_relevant_confidence = sum(r["weighted_confidence"] for r in not_relevant_results)

        if relevant_confidence > not_relevant_confidence:
            relevant = True
            confidence = relevant_confidence / len(relevant_results) if relevant_results else 0.5
            method = "weighted_majority"
            reason = f"Weighted majority: relevant confidence {relevant_confidence:.2f} > not relevant {not_relevant_confidence:.2f}"
        else:
            relevant = False
            confidence = not_relevant_confidence / len(not_relevant_results) if not_relevant_results else 0.5
            method = "weighted_majority"
            reason = f"Weighted majority: not relevant confidence {not_relevant_confidence:.2f} >= relevant {relevant_confidence:.2f}"

        if llm_grade and "relevant" in llm_grade:
            llm_weight = 0.3
            llm_confidence = llm_grade.get("confidence", 0.5) * llm_weight

            if llm_grade["relevant"] == relevant:
                confidence = (confidence + llm_confidence) / (1 + llm_weight)
            else:
                confidence = confidence * 0.8
                reason += f"; LLM disagrees (confidence reduced)"

        confidence = max(0.1, min(0.9, confidence))

        return {
            "relevant": relevant,
            "confidence": confidence,
            "reason": reason,
            "fallback_method": method,
            "detailed_results": [
                {
                    "method": r["grader"],
                    "relevant": r["result"].relevant,
                    "confidence": r["result"].confidence,
                    "reason": r["result"].reason
                }
                for r in fallback_results
            ]
        }

    def _default_fallback_result(self, question: str, document: str) -> Dict[str, Any]:
        """
        Default fallback result when other methods fail
        """
        doc_length = len(document)
        question_length = len(question)

        if doc_length < 100:
            relevant = False
            confidence = 0.7
            reason = "Default: document too short"
        elif question_length < 10:
            relevant = True
            confidence = 0.5
            reason = "Default: short question"
        else:
            relevant = False
            confidence = 0.3
            reason = "Default: insufficient information"

        return {
            "relevant": relevant,
            "confidence": confidence,
            "reason": reason,
            "fallback_method": "default",
            "detailed_results": []
        }

    def create_emergency_fallback(self) -> Callable:
        """
        Creates emergency fallback functionality for LangGraph integration.
        """

        def emergency_fallback_function(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Emergency fallback feature for LangGraph.
            Used when the primary grader fails completely.
            """
            from src.agent.graph_state import StateManager

            question = state.get("question", "")
            documents = state.get("documents", [])

            if not documents:
                return StateManager.update_state(
                    state,
                    relevant_docs=[],
                    confidence=0.0,
                    needs_rewrite=True,
                    metadata={
                        **state.get("metadata", {}),
                        "fallback_triggered": "no_documents",
                        "fallback_reason": "No documents to grade"
                    }
                )

            document_to_grade = documents[0] if documents else ""

            fallback_result = self.execute_fallback(question, document_to_grade)

            relevant_docs = [document_to_grade] if fallback_result["relevant"] else []

            updated_state = StateManager.update_state(
                state,
                relevant_docs=relevant_docs,
                confidence=fallback_result["confidence"],
                needs_rewrite=not fallback_result["relevant"] and state["iterations"] < state["max_iterations"],
                metadata={
                    **state.get("metadata", {}),
                    "fallback_triggered": True,
                    "fallback_result": fallback_result,
                    "emergency_fallback": True
                }
            )

            logger.warning(f"Emergency fallback triggered: relevant={fallback_result['relevant']}, "
                           f"confidence={fallback_result['confidence']:.2f}")

            return updated_state

        return emergency_fallback_function


class ErrorRecoverySystem:
    """
    Grader Error Recovery System.
    """

    def __init__(self):
        self.error_history = []
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3

    def handle_grader_error(self,
                            error: Exception,
                            question: str,
                            document: str,
                            grader_type: str) -> Dict[str, Any]:
        """
        Handles the grader error and attempts to recover.

        Returns:
            Recovery result or None if recovery failed.
        """

        error_record = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "grader_type": grader_type,
            "question_preview": question[:100],
            "document_preview": document[:100]
        }
        self.error_history.append(error_record)

        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]

        recovery_result = self._attempt_recovery(error, question, document)

        if recovery_result:
            self.recovery_attempts = 0
            return recovery_result
        else:
            self.recovery_attempts += 1

            if self.recovery_attempts >= self.max_recovery_attempts:
                logger.critical(f"Max recovery attempts reached ({self.recovery_attempts})")
                return self._final_fallback(question, document)

            return None

    def _attempt_recovery(self,
                          error: Exception,
                          question: str,
                          document: str) -> Optional[Dict[str, Any]]:
        """
        Trying to recover from a mistake.
        """
        error_type = type(error).__name__
        error_msg = str(error).lower()

        if "timeout" in error_msg or "connection" in error_msg:
            return self._local_fallback_recovery(question, document)

        elif "memory" in error_msg or "gpu" in error_msg:
            return self._lightweight_fallback_recovery(question, document)

        elif "json" in error_msg or "parsing" in error_msg:
            return self._simple_format_recovery(question, document)

        elif "model" in error_msg or "not found" in error_msg:
            return self._alternative_model_recovery(question, document)

        else:
            return self._general_recovery(question, document)

    def _local_fallback_recovery(self, question: str, document: str) -> Dict[str, Any]:
        """
        Recovery for connection errors
        """

        fallback_manager = FallbackManager([
            FallbackStrategy.KEYWORD_MATCH,
            FallbackStrategy.RULE_BASED
        ])

        result = fallback_manager.execute_fallback(question, document)

        return {
            **result,
            "recovery_method": "local_fallback",
            "recovery_reason": "Connection/timeout error"
        }

    def _lightweight_fallback_recovery(self, question: str, document: str) -> Dict[str, Any]:
        """
        Recovery for memory errors
        """

        keyword_grader = KeywordFallbackGrader(min_keyword_match=1)
        rule_grader = RuleBasedFallbackGrader()

        keyword_result = keyword_grader.grade(question, document)
        rule_result = rule_grader.grade(question, document)


        relevant_votes = [keyword_result.relevant, rule_result.relevant]
        relevant = sum(relevant_votes) >= 1

        confidence = (keyword_result.confidence + rule_result.confidence) / 2

        return {
            "relevant": relevant,
            "confidence": confidence,
            "reason": f"Lightweight recovery: keyword={keyword_result.relevant}, rule={rule_result.relevant}",
            "recovery_method": "lightweight_fallback",
            "recovery_reason": "Memory/GPU error"
        }

    def _simple_format_recovery(self, question: str, document: str) -> Dict[str, Any]:
        """
        Recovery for parsing errors.
        """

        prompt = f"P: {question}\nD: {document}\nDoes the document answer the question? Answer only YES or NO."

        rule_grader = RuleBasedFallbackGrader()
        result = rule_grader.grade(question, document)

        return {
            "relevant": result.relevant,
            "confidence": result.confidence * 0.8,
            "reason": f"Simple format recovery: {result.reason}",
            "recovery_method": "simple_format",
            "recovery_reason": "JSON/parsing error"
        }

    def _alternative_model_recovery(self, question: str, document: str) -> Dict[str, Any]:
        """
        Recovery when model is not available.
        """
        try:
            from src.agent.nodes.grader_model import LocalTransformersGrader
            grader = LocalTransformersGrader(model_name="all-MiniLM-L6-v2")

            prompt = f"QUESTION: {question}\nDOCUMENT: {document}\nIs the document relevant? Answer YES or NO."
            response = grader.grade(prompt)

            relevant = "yes" in response.lower() or "true" in response.lower()
            confidence = 0.6

            return {
                "relevant": relevant,
                "confidence": confidence,
                "reason": f"Alternative model recovery: response='{response[:50]}...'",
                "recovery_method": "alternative_model",
                "recovery_reason": "Model not found"
            }

        except:
            return self._local_fallback_recovery(question, document)

    def _general_recovery(self, question: str, document: str) -> Dict[str, Any]:
        """
        General recovery for unknown errors.
        """

        fallback_manager = FallbackManager()
        result = fallback_manager.execute_fallback(question, document)

        return {
            **result,
            "recovery_method": "general_fallback",
            "recovery_reason": "Unknown error"
        }

    def _final_fallback(self, question: str, document: str) -> Dict[str, Any]:
        """
        The ultimate fallback when all else fails.
        """

        return {
            "relevant": False,
            "confidence": 0.1,
            "reason": "All recovery attempts failed. Defaulting to not relevant.",
            "recovery_method": "final_fallback",
            "recovery_reason": "Max recovery attempts reached",
            "emergency": True
        }

    def get_error_stats(self) -> Dict[str, Any]:
        """
        Returns error statistics.
        """
        if not self.error_history:
            return {"total_errors": 0, "recovery_rate": 1.0}

        error_types = {}
        for record in self.error_history:
            error_type = record["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1

        total_errors = len(self.error_history)
        successful_recoveries = sum(1 for r in self.error_history
                                    if "recovery_method" in r)

        recovery_rate = successful_recoveries / total_errors if total_errors > 0 else 1.0

        return {
            "total_errors": total_errors,
            "error_types": error_types,
            "recovery_rate": recovery_rate,
            "recovery_attempts": self.recovery_attempts,
            "error_history_samples": self.error_history[-5:]
        }