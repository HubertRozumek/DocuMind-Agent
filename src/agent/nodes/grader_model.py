import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, Callable
import time
from abc import ABC, abstractmethod
import requests
import subprocess
import numpy as np

logger = logging.getLogger(__name__)

class BaseGraderModel(ABC):
    """
    Abstract base class for grading models.
    """
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
        self.retry_count = kwargs.get("retry_count", 3)
        self.timeout = kwargs.get("timeout", 30)

    @abstractmethod
    def grade(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Performs prompt-based assessment.
        """
        pass

    @abstractmethod
    def grade_batch(self, prompts: List[str], system_prompts: Optional[List[str]] = None) -> List[str]:
        """
        Performs evaluation for batch of prompts.
        """
        pass

    def validate_response(self, response: str, expected_format: str) -> Dict[str, Any]:
        """
        Validates model response.
        """
        try:
            if expected_format.lower() == "json":
                return json.loads(response.strip())
            elif expected_format.lower() == "YES/NO":
                return {"relevant": self._parse_boolean(response)}
            else:
                return {"raw": response}

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON response: {response[:100]}")
            return self._fix_json_response(response)

    def _parse_boolean(self, response: str) -> bool:
        """
        Parsing a text response into a boolean value.
        """
        response_lower = response.strip().lower()

        true_keywords = ['yes','true','y','relevant']
        false_keywords = ['no','false','n','not relevant']

        for keyword in true_keywords:
            if keyword in response_lower: return True

        for keyword in false_keywords:
            if keyword in response_lower: return False

        logger.warning(f"Could not parse boolean response: {response}")
        return False

    def _fix_json_response(self, response: str) -> Dict[str, Any]:
        """
        Attempts to repair a corrupted JSON response.
        """
        lines = response.strip().split('\n')
        json_lines = []

        for line in lines:
            if line.strip().startswith("{") or line.strip().endswith("}") or ':' in line:
                json_lines.append(line)

        cleaned = '\n'.join(json_lines)

        if not cleaned.startswith("{"):
            cleaned = "{" + cleaned
        if not cleaned.endswith("}"):
            cleaned = cleaned + "}"

        try:
            return json.loads(cleaned)
        except:
            return {
                "relevant": False,
                "reason": f"Could not parse JSON response: {response[:100]}",
                "error": "invalid_json"
            }


class OllamaGrader(BaseGraderModel):
    """
    Grader using Ollama with local model.
    """

    def __init__(self, model_name: str = "llama3.2:1b", **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = kwargs.get('base_url', 'http://localhost:11434')
        self.temperature = kwargs.get('temperature', 0.1)
        self.max_tokens = kwargs.get('max_tokens', 200)

        self._check_connection()

    def _check_connection(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]

                if self.model_name in model_names:
                    logger.info(f"Successfully connected to {self.model_name}")
                else:
                    logger.warning(f"Failed to connect to {self.model_name}. Available models: {model_names}")

                    if model_names:
                        self.model_name = model_names[0]
                        logger.info(f"Using fallback model: {self.model_name}")
            else:
                logger.warning(f"Ollama API returned status code {response.status_code}")
        except requests.exceptions.ConnectionError:
            logger.warning(f"Ollama not connected. Will use fallback strategies")
            self.available = False
        except Exception as e:
            logger.error(f"Error checking Ollama: {e}")
            self.available = False

    def grade(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Performs assessment by Ollama
        """

        if hasattr(self, 'available') or not self.available:
            return self._fallback_grade(prompt)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
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
                    return result.get('response', '').strip()
                else:
                    logger.warning(f"Ollama API error: {response.status_code}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Ollama request failed (attempt #{attempt+1}): {e}")
                time.sleep(1)

        return self._fallback_grade(prompt)

    def grade_batch(self, prompts: List[str], system_prompts: Optional[List[str]] = None) -> List[str]:
        """
        Performs evaluation for batch of prompts.
        """

        results = []

        for i, prompt in enumerate(prompts):
            system_prompt = system_prompts[i] if system_prompts and i < len(system_prompts) else None
            result = self.grade(prompt, system_prompt=system_prompt)
            results.append(result)

            if i < len(prompts) - 1:
                time.sleep(1)

        return results

    def _fallback_grade(self, prompt: str) -> str:
        """
        Fallback when Ollama is not available.
        Simple keyword-based fallback.
        """
        logger.warning("Using fallback grading (Ollama not available)")

        question_keywords = self._extract_keywords(prompt.split("QUESTION:")[1].split("DOCUMENT:")[0])
        document_text = prompt.split("DOCUMENT:")[1] if "DOCUMENT:" in prompt else ""
        document_keywords = self._extract_keywords(document_text)


        matches = len(set(question_keywords) & set(document_keywords))

        if matches >= 2:
            return '{"relevant": true, "reason": "Fallback: keyword match"}'
        else:
            return '{"relevant": false, "reason": "Fallback: insufficient keyword match"}'

    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extracts keywords from text.
        """
        stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

        words = text.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        return list(set(keywords))[:max_keywords]

class LocalTransformersGrader(BaseGraderModel):
    """
    Grader using local models.
    """
    def __init__(self, model_name: str = "distilbert/distilroberta-base", **kwargs):
        super().__init__(model_name, **kwargs)
        self.model = None
        self.tokenizer = None
        self.device = kwargs.get('device', 'cpu')

        self._load_model()

    def _load_model(self):
        """
        Loads model.
        """

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            logger.info(f"Loading transformers model: {self.model_name}")

            if "gpt2" in self.model_name:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    device_map="auto"
                )

                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.generator = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                )
            else:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                self.is_encoder_only = True

            logger.info(f"Model loaded: {self.model_name}")

        except ImportError:
            logger.error("Transformers not installed. Install with pip install transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def grade(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Performs evaluation through local model.
        """

        if self.model is None:
            return self._fallback_grade(prompt)

        try:
            if hasattr(self, "generator"):
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

                outputs = self.generator(
                    full_prompt,
                    max_length=self.config.get('max_length', 200),
                    temperature=self.config.get('temperature', 0.1),
                    do_sample=True,
                    num_return_sequences=1
                )

                return outputs[0]['generated_text'].replace(full_prompt, '').strip()

            elif hasattr(self, "is encoder_only"):
                return self._semantic_grade(prompt, system_prompt)

        except Exception as e:
            logger.error(f"Failed to evaluate model: {e}")
            return self._fallback_grade(prompt)

    def _semantic_grade(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Semantic similarity-based evaluation.
        (for encoder only)
        """

        if 'QUESTION' in prompt and 'DOCUMENT' in prompt:
            try:
                question_part = prompt.split("QUESTION:")[1].split("DOCUMENT:")[0]
                document_part = prompt.split("DOCUMENT:")[1]

                embeddings = self.model.encode([question_part, document_part])
                a = embeddings[0]
                b = embeddings[1]
                similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

                threshold = 0.7
                relevant = similarity > threshold

                return json.dumps({
                    "relevant": bool(relevant),
                    "confidence": float(similarity),
                    "reason": f"Semantic similarity: {similarity:.3f}"
                })
            except:
                pass

        return self._fallback_grade(prompt)

    def grade_batch(self, prompts: List[str], system_prompts: Optional[List[str]] = None) -> List[str]:
        """
        Performs batch evaluation.
        """

        results = []

        for i, prompt in enumerate(prompts):
            system_prompt = system_prompts if system_prompts and i < len(system_prompts) else None
            result = self.grade(prompt, system_prompt=system_prompt)
            results.append(result)

        return results

class LlamaCppGrader(BaseGraderModel):
    """
    Grader using Llama's C++ implementation.
    """

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        self.model_path = model_path
        self.n_gpu_layers = kwargs.get('n_gpu_layers', -1)
        self.n_ctx = kwargs.get('n_ctx', 2048)

        self._check_llama_cpp()

    def _check_llama_cpp(self):
        """

        :return:
        """
        try:
            from llama.cpp import Llama
            self.llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                verbose=False
            )
            logger.info(f"Llama.cpp model loaded: {self.model_path}")
            self.available = True
        except ImportError:
            logger.warning("Llama.cpp not installed. Install with: pip install llama-cpp-python")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to load llama.cpp model: {e}")
            self.available = False

    def grade(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Performs evaluation using llama.cpp.
        """
        if not hasattr(self, "available") or not self.available:
            return self._fallback_grade(prompt)

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        try:
            output = self.llm(
                full_prompt,
                max_tokens = self.config.get('max_tokens', 200),
                temperature=self.config.get('temperature', 0.1),
                echo=False
            )
            return output['choices'][0]['text'].strip()
        except Exception as e:
            logger.error(f"llama.cpp grading failed: {e}")
            return self._fallback_grade(prompt)

class MockGrader(BaseGraderModel):
    """
    Mock grader for tests.
    """

    def __init__(self, accuracy: float = 0.8, **kwargs):
        super().__init__("mock", **kwargs)
        self.accuracy = accuracy
        self.response_patterns = [
            '{"relevant": true, "reason": "The document answers the question directly"}',
            '{"relevant": false, "reason": "The document does not contain information related to the question"}',
            '{"relevant": true, "confidence": 0.85, "reason": "High semantic similarity"}',
            '{"relevant": false, "confidence": 0.25, "reason": "Low semantic similarity"}'
        ]

    def grade(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Return mock response.
        """

        import random
        if random.random() < self.accuracy:
            if "QUESTION" in prompt and "DOCUMENT" in prompt:
                question = prompt.split("QUESTION:")[1].split("DOCUMENT:")[0].lower()
                document = prompt.split("DOCUMENT:")[1].lower()

                common_words = set(question.split()) &set(document.split())
                if len(common_words) >= 2:
                    return self.response_patterns[0]
                else:
                    return self.response_patterns[1]

        return random.choice(self.response_patterns)

    def grade_batch(self, prompts: List[str], system_prompts: Optional[List[str]] = None) -> List[str]:
        return [self.grade(prompt) for prompt in prompts]

class GraderFactory:
    """
    Factory class for grading models.
    """

    @staticmethod
    def create_grader(grader_type: str, **kwargs) -> BaseGraderModel:
        """
        Creates a grader of the specified type.

        Args:
            grader_type: 'ollama', 'transformers', 'llama_cpp', 'mock', 'hybrid'
            **kwargs: type-specific configuration

        Returns:
            BaseGraderModel instance
        """

        if grader_type == 'ollama':
            return OllamaGrader(**kwargs)
        elif grader_type == "transformers":
            return LocalTransformersGrader(**kwargs)
        elif grader_type == "llama_cpp":
            return LlamaCppGrader(**kwargs)
        elif grader_type == "mock":
            return MockGrader(**kwargs)
        elif grader_type == "hybrid":
            return HybridGrader(**kwargs)
        else:
            raise ValueError(f"Unknown grader type: {grader_type}")

class HybridGrader(BaseGraderModel):
    """
    Hybrid grader combining different methods
    """

    def __init__(self, **kwargs):
        super().__init__("hybrid", **kwargs)
        self.graders = []
        self.fallbacks = []

        try:
            ollama_grader = OllamaGrader(**kwargs.get('ollama_config', {}))
            if hasattr(ollama_grader, 'available') and ollama_grader.available:
                self.graders.append(("ollama",ollama_grader))
                logger.info("Hybrid ollama available")
        except:
            pass

        try:
            transformers_grader = LocalTransformersGrader(**kwargs.get('transformers_config', {}))
            self.graders.append(("transformers", transformers_grader))
            logger.info("Hybrid transformers available")
        except:
            pass

        mocked_grader = MockGrader(**kwargs.get('mock_config', {}))
        self.fallbacks.append(("mock", mocked_grader))

        if not self.graders:
            logger.warning("No grader available, using mock grader.")
            self.graders = self.fallbacks

    def grade(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Performs evaluation using available graders.
        """
        results = []

        for name, grader in self.graders:
            try:
                result = grader.grade(prompt, system_prompt)
                validated = self._validate_grade(result, prompt)

                if validated.get('confidence', 0) > 0.5:
                    results.append((name, result, validated.get('confidence', 0)))
                    logger.info(f"Hybrid {name} graded with confidence {validated.get('confidence', 0):.2f}")

                    if validated.get('confidence', 0) > 0.8:
                        return result

            except Exception as e:
                logger.warning(f"Hybrid {name} failed with error: {e}")

        if results:
            results.sort(key=lambda x: x[2], reverse=True)
            best_name, best_result, best_confidence = results[0]
            logger.info(f"Hybrid {best_name} graded with confidence {best_confidence:.2f}")
            return best_result

        for name, fallback in self.fallbacks:
            try:
                result = fallback.grade(prompt, system_prompt)
                logger.warning(f"Hybrid using fallback {name}")
                return result
            except:
                continue

        return '{"relevant": false, "reason": "All graders failed", "confidence": 0.0}'

    def _validate_grade(self, response: str, original_prompt: str) -> Dict[str, Any]:
        """
        Validates the grade and returns confidence.
        """

        try:
            parsed = json.loads(response)

            has_relevant = 'relevant' in parsed
            has_reason = 'reason' in parsed

            confidence = 0.5

            if has_relevant and has_reason:
                confidence = 0.8

            if 'confidence' in parsed:
                try:
                    confidence = float(parsed['confidence'])
                except:
                    pass

            parsed['validation_confidence'] = confidence
            return parsed

        except json.JSONDecodeError:
            return {"relevant": False,"reason": "Invalid JSON", "confidence": 0.1}