import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import requests

logger = logging.getLogger(__name__)


class BaseGraderModel(ABC):
    """
    Abstract base class for grading models.

    Defines interface for document grading with support for single
    and batch grading, response validation, and fallback strategies.

    Attributes:
        model_name: Name of the model to use
        config: Configuration dictionary
        retry_count: Number of retries for failed requests
        timeout: Request timeout in seconds
    """

    def __init__(self, model_name: str, **kwargs) -> None:
        """
        Initialize base grader model.

        Args:
            model_name: Model identifier
            **kwargs: Configuration options including retry_count and timeout
        """
        self.model_name = model_name
        self.config = kwargs
        self.retry_count = kwargs.get("retry_count", 3)
        self.timeout = kwargs.get("timeout", 30)

    @abstractmethod
    def grade(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Perform single document grading.

        Args:
            prompt: Grading prompt with question and document
            system_prompt: Optional system instructions

        Returns:
            Model response as string
        """
        pass

    @abstractmethod
    def grade_batch(
        self, prompts: List[str], system_prompts: Optional[List[str]] = None
    ) -> List[str]:
        """
        Perform batch grading for multiple documents.

        Args:
            prompts: List of grading prompts
            system_prompts: Optional list of system prompts

        Returns:
            List of model responses
        """
        pass

    def validate_response(self, response: str, expected_format: str) -> Any:
        """
        Validate and parse model response.

        Args:
            response: Raw model response
            expected_format: Expected format ("json" or "YES/NO")

        Returns:
            Parsed response dictionary
        """
        try:
            if expected_format.lower() == "json":
                try:
                    return json.loads(response.strip())
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON response: {response[:100]}")
                    return self._fix_json_response(response)

            elif expected_format.lower() == "YES/NO":
                return {"relevant": self._parse_boolean(response)}

            else:
                return {"raw": response}

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON response: {response[:100]}")
            return self._fix_json_response(response)

    def _fallback_grade(self, prompt: str) -> str:
        """
        Fallback grading when primary model is unavailable.

        Uses simple keyword matching between question and document.

        Args:
            prompt: Grading prompt

        Returns:
            JSON string with relevance assessment
        """
        logger.warning("Using fallback grading (primary model not available)")

        question_keywords = self._extract_keywords(
            prompt.split("QUESTION:")[1].split("DOCUMENT:")[0]
        )
        document_text = prompt.split("DOCUMENT:")[1] if "DOCUMENT:" in prompt else ""
        document_keywords = self._extract_keywords(document_text)

        matches = len(set(question_keywords) & set(document_keywords))

        if matches >= 2:
            return '{"relevant": true, "reason": "Fallback: keyword match"}'
        else:
            return '{"relevant": false, "reason": "Fallback: insufficient keyword match"}'

    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text, filtering stop words.

        Args:
            text: Source text
            max_keywords: Maximum keywords to return

        Returns:
            List of keywords
        """
        stop_words = [
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "having",
            "do",
            "does",
            "did",
            "doing",
            "a",
            "an",
            "the",
            "and",
            "but",
            "if",
            "or",
            "because",
            "as",
            "until",
            "while",
            "of",
            "at",
            "by",
            "for",
            "with",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "to",
            "from",
            "up",
            "down",
            "in",
            "out",
            "on",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "s",
            "t",
            "can",
            "will",
            "just",
            "don",
            "should",
            "now",
        ]

        words = text.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        return list(set(keywords))[:max_keywords]

    def _parse_boolean(self, response: str) -> bool:
        """
        Parse text response into boolean relevance value.

        Args:
            response: Text response to parse

        Returns:
            True if relevant, False otherwise
        """
        response_lower = response.strip().lower()

        if any(
            pattern in response_lower for pattern in ["not relevant", "false", "no", "irrelevant"]
        ):
            return False

        if any(pattern in response_lower for pattern in ["yes", "true", "relevant"]):
            return True

        logger.warning(f"Could not parse boolean response: {response}")
        return False

    def _fix_json_response(self, response: str) -> Any:
        """
        Attempt to repair malformed JSON responses.

        Tries multiple strategies: regex extraction, line-based reconstruction,
        and heuristic keyword analysis as final fallback.

        Args:
            response: Malformed JSON string

        Returns:
            Parsed dictionary or heuristic result
        """
        json_match = re.search(r"\{[^}]+\}", response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        lines = response.strip().split("\n")
        json_lines = []

        for line in lines:
            if line.strip().startswith("{") or line.strip().endswith("}") or ":" in line:
                json_lines.append(line)

        cleaned = "\n".join(json_lines)

        if not cleaned.startswith("{"):
            cleaned = "{" + cleaned
        if not cleaned.endswith("}"):
            cleaned = cleaned + "}"

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON, using heuristic fallback")

            lower_resp = response.lower()
            relevant_keywords = ["create", "use", "example", "numpy", "array", "function"]
            irrelevant_keywords = ["cannot", "not found", "no information", "unclear"]

            relevant_score = sum(1 for kw in relevant_keywords if kw in lower_resp)
            irrelevant_score = sum(1 for kw in irrelevant_keywords if kw in lower_resp)

            is_relevant = relevant_score > irrelevant_score
            confidence = min(0.9, relevant_score * 0.15) if is_relevant else 0.2

            return {
                "relevant": is_relevant,
                "confidence": confidence,
                "reason": f"Heuristic: found {relevant_score} relevant keywords",
                "error": "json_parse_failed",
                "original_response": response[:100],
            }


class OllamaGrader(BaseGraderModel):
    """
    Grader implementation using Ollama API with local models.

    Connects to local Ollama instance for document grading with
    automatic fallback to keyword-based grading if unavailable.

    Attributes:
        base_url: Ollama API endpoint URL
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        available: Whether Ollama is available
    """

    def __init__(self, model_name: str = "phi3:mini", **kwargs):
        """
        Initialize Ollama grader and check connection.

        Args:
            model_name: Ollama model name
            **kwargs: Configuration including base_url, temperature, max_tokens
        """
        super().__init__(model_name, **kwargs)
        self.base_url = kwargs.get("base_url", "http://localhost:11434")
        self.temperature = kwargs.get("temperature", 0.0)
        self.max_tokens = kwargs.get("max_tokens", 200)

        self._check_connection()

    def _check_connection(self):
        """Verify Ollama connectivity and model availability."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]

                if self.model_name in model_names:
                    logger.info(f"Successfully connected to {self.model_name}")
                    self.available = True
                else:
                    logger.warning(f"Model {self.model_name} not found. Available: {model_names}")

                    if model_names:
                        self.model_name = model_names[0]
                        logger.info(f"Using fallback model: {self.model_name}")
                        self.available = True
                    else:
                        self.available = False
            else:
                logger.warning(f"Ollama API returned status code {response.status_code}")
                self.available = False

        except requests.exceptions.ConnectionError:
            logger.warning("Ollama not connected. Will use fallback strategies")
            self.available = False
        except Exception as e:
            logger.error(f"Error checking Ollama: {e}")
            self.available = False

    def grade(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Grade document using Ollama API.

        Args:
            prompt: Grading prompt
            system_prompt: Optional system instructions

        Returns:
            Model response or fallback result
        """
        if not getattr(self, "available", False):
            return self._fallback_grade(prompt)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }

        for attempt in range(self.retry_count):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "").strip()
                else:
                    logger.warning(f"Ollama API error: {response.status_code}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Ollama request failed (attempt {attempt + 1}): {e}")
                time.sleep(1)

            except Exception:
                wait_time = 2**attempt
                time.sleep(wait_time)

        return self._fallback_grade(prompt)

    def grade_batch(
        self, prompts: List[str], system_prompts: Optional[List[str]] = None
    ) -> List[str]:
        """
        Grade multiple documents sequentially.

        Args:
            prompts: List of grading prompts
            system_prompts: Optional list of system prompts

        Returns:
            List of model responses
        """
        results = []

        for i, prompt in enumerate(prompts):
            system_prompt = (
                system_prompts[i] if system_prompts and i < len(system_prompts) else None
            )
            result = self.grade(prompt, system_prompt=system_prompt)
            results.append(result)

            if i < len(prompts) - 1:
                time.sleep(1)

        return results


class GraderFactory:
    """
    Factory for creating grader model instances.
    """

    @staticmethod
    def create_grader(grader_type: str, **kwargs) -> BaseGraderModel:
        """
        Create grader of specified type.

        Args:
            grader_type: Type of grader ("ollama", etc.)
            **kwargs: Type-specific configuration

        Returns:
            Configured grader instance

        Raises:
            ValueError: If grader_type is not supported
        """
        if grader_type == "ollama":
            return OllamaGrader(**kwargs)
        else:
            raise ValueError(f"Unknown grader type: {grader_type}")
