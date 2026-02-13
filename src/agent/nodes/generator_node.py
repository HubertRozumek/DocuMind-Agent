import logging
import math
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Dict, Any, List, Optional, Callable

from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

from src.agent.graph_state import GraphState

logger = logging.getLogger(__name__)


class GeneratorNode:
    """
    Answer generation node for RAG-based systems.

    Generates final answers from retrieved documents using an LLM with
    robust error handling, retry mechanisms, and fallback strategies.

    Attributes:
        model_name: Name of the Ollama model to use
        temperature: Sampling temperature for generation
        max_tokens: Maximum tokens to generate per response
        llm: Configured Ollama LLM instance
        prompt: Answer generation prompt template
        chain: Compiled LangChain pipeline
    """

    def __init__(
            self,
            model_name: str = "mistral:7b",
            temperature: float = 0.1,
            max_tokens: int = 500
    ):
        """
        Initialize the answer generator with LLM configuration.

        Args:
            model_name: Name of the Ollama model to load
            temperature: Sampling temperature (0.0-1.0, lower is more deterministic)
            max_tokens: Maximum number of tokens to generate per response

        Raises:
            No explicit exceptions, falls back to default model on initialization failure
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            self.llm = Ollama(
                model=model_name,
                temperature=temperature,
                num_predict=max_tokens,
            )
            logger.info(f"Generator initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize generator with {model_name}: {e}")
            self.llm = Ollama(model="mistral:7b")
            logger.warning(f"Using fallback model: mistral:7b")

        self.prompt = self._create_answer_prompt()
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _create_answer_prompt(self) -> PromptTemplate:
        """Create the RAG answer generation prompt template."""
        template = """
        You are an expert assistant answering questions based on provided documents.

        DOCUMENTS:
        {documents}

        USER QUESTION:
        {question}

        INSTRUCTIONS:
        1. Answer the question based ONLY on the provided documents.
        2. If the documents don't contain the answer, say "I couldn't find the answer in the provided documents."
        3. Cite specific parts of the documents that support your answer.
        4. Be concise but comprehensive.
        5. Format your answer in a clear, professional manner.

        ANSWER:
        """

        return PromptTemplate(
            template=template,
            input_variables=["question", "documents"]
        )

    def generate_answer(
            self,
            question: str,
            documents: List[str],
            metadata: Optional[Dict[str, Any]] = None,
            retry_count: int = 0,
            max_retries: int = 2,
            timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Generate an answer from documents with retry and timeout handling.

        Implements a robust generation pipeline with:
        - Timeout protection via ThreadPoolExecutor
        - Exponential backoff retry for empty/timeout responses
        - Confidence scoring based on document coverage
        - Graceful fallback on persistent failures

        Args:
            question: User query to answer
            documents: Retrieved documents to base answer on
            metadata: Optional context metadata for tracking
            retry_count: Current retry attempt (internal use)
            max_retries: Maximum retry attempts before fallback
            timeout: Seconds to wait for LLM response

        Returns:
            Dictionary containing:
                - answer: Generated text or fallback message
                - confidence: Float score (0.0-0.9) based on document coverage
                - sources_used: Number of documents referenced
                - documents_considered: Total documents available
                - generation_successful: Boolean success flag
                - error: Error description if failed (optional)
                - retries_needed: Number of retries executed
        """
        if not documents:
            logger.warning("No documents provided for answer generation")
            return {
                "answer": "I couldn't find any relevant documents to answer your question.",
                "confidence": 0.0,
                "sources_used": 0,
                "documents_considered": 0,
                "generation_successful": False
            }

        try:
            combined_docs = "\n\n".join([
                f"Document {i + 1}:\n{doc[:2000]}"
                for i, doc in enumerate(documents[:3])
            ])

            logger.info(f"[generate_answer] Calling LLM (attempt {retry_count + 1}/{max_retries + 1})")
            logger.debug(f"[generate_answer] Question: {question}")

            answer = None
            timed_out = False

            def _generate():
                return self.chain.invoke({
                    "question": question,
                    "documents": combined_docs
                })

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_generate)
                try:
                    answer = future.result(timeout=timeout)
                    timed_out = False
                except TimeoutError:
                    logger.error(f"LLM generation timed out after {timeout}s")
                    timed_out = True

            logger.info(f"[generate_answer] Raw LLM response type: {type(answer)}")
            logger.debug(f"[generate_answer] Raw LLM response: {repr(answer)[:500] if answer else 'None (timeout)'}")

            if timed_out:
                if retry_count < max_retries:
                    logger.warning(f"[generate_answer] Timeout, retrying ({retry_count + 1}/{max_retries})...")
                    import time
                    time.sleep(1.0 * (retry_count + 1))
                    return self.generate_answer(question, documents, metadata, retry_count + 1, max_retries, timeout)
                else:
                    logger.error("[generate_answer] Max retries after timeout exceeded")
                    return {
                        "answer": self._generate_fallback_answer(question, documents, error="Generation timeout"),
                        "confidence": 0.0,
                        "sources_used": 0,
                        "documents_considered": len(documents),
                        "generation_successful": False,
                        "error": "timeout",
                        "retries_needed": retry_count
                    }

            answer_text = str(answer).strip() if answer else ""

            if not answer_text:
                if retry_count < max_retries:
                    logger.warning(f"[generate_answer] Empty response, retrying ({retry_count + 1}/{max_retries})...")
                    import time
                    time.sleep(0.5 * (retry_count + 1))
                    return self.generate_answer(question, documents, metadata, retry_count + 1, max_retries, timeout)
                else:
                    logger.error("[generate_answer] LLM returned empty response after all retries")
                    answer_text = self._generate_fallback_answer(question, documents)

            base_confidence = 0.3
            doc_bonus = 0.05 * math.log(len(documents) + 1)
            confidence = min(0.9, base_confidence + doc_bonus)

            logger.info(f"[generate_answer] Successfully generated answer (confidence: {confidence:.2f})")

            return {
                "answer": answer_text,
                "confidence": float(confidence),
                "sources_used": len(documents),
                "documents_considered": len(documents),
                "generation_successful": True,
                "retries_needed": retry_count
            }

        except Exception as e:
            logger.error(f"Answer generation failed: {e}", exc_info=True)

            if retry_count < max_retries:
                logger.warning(f"Retrying after exception ({retry_count + 1}/{max_retries})...")
                import time
                time.sleep(1.0 * (retry_count + 1))
                return self.generate_answer(question, documents, metadata, retry_count + 1, max_retries, timeout)

            logger.error("[generate_answer] Max retries after exception exceeded")
            return {
                "answer": self._generate_fallback_answer(question, documents, error=str(e)),
                "confidence": 0.0,
                "sources_used": 0,
                "documents_considered": len(documents),
                "generation_successful": False,
                "error": str(e),
                "retries_needed": retry_count
            }

    def _generate_fallback_answer(
            self,
            question: str,
            documents: List[str],
            error: Optional[str] = None
    ) -> str:
        """
        Generate a user-friendly fallback response when LLM generation fails.

        Provides document previews and actionable suggestions to help users
        understand what went wrong and how to proceed.

        Args:
            question: Original user question
            documents: Available documents (may be empty)
            error: Optional technical error message for debugging

        Returns:
            Formatted fallback message with document previews or guidance
        """
        if not documents:
            return (
                "I couldn't find any relevant documents to answer your question. "
                "Try:\n"
                "• Rephrasing your question\n"
                "• Using different keywords\n"
                "• Checking if documents are loaded correctly"
            )

        doc_previews = [doc[:200] + "..." for doc in documents[:2]]

        fallback = (
            "I encountered an issue generating a complete answer. "
            "However, I found these relevant documents:\n\n"
        )

        for i, preview in enumerate(doc_previews, 1):
            fallback += f"{i}. {preview}\n\n"

        fallback += (
            "Please try:\n"
            "• Asking a more specific question\n"
            "• Rephrasing your query\n"
            "• Checking the system logs for details"
        )

        if error:
            fallback += f"\n\nTechnical details: {error}"

        return fallback

    def as_runnable(self) -> Callable:
        """
        Convert to LangGraph-compatible runnable function.

        Returns:
            RunnableLambda wrapping the generator function for LangGraph integration
        """

        def generator_function(state: GraphState) -> GraphState:
            """
            LangGraph-compatible generator node function.

            Args:
                state: Current graph state containing question and documents

            Returns:
                Updated state with generated answer and metadata
            """
            logger.info("[Generator] Starting answer generation node")

            relevant_docs = state.get("relevant_docs", [])
            if not relevant_docs:
                relevant_docs = state.get("documents", [])[:2]
                logger.debug(f"[Generator] Using fallback documents (count: {len(relevant_docs)})")

            if not relevant_docs:
                logger.warning("[Generator] No documents available for generation")
                return {
                    **state,
                    "answer": "I couldn't find any documents to answer your question.",
                    "confidence": 0.0,
                    "history": [
                        *state.get("history", []),
                        {
                            "role": "assistant",
                            "action": "generation",
                            "content": "No documents available",
                            "confidence": 0.0
                        }
                    ]
                }

            result = self.generate_answer(
                question=state["question"],
                documents=relevant_docs,
                metadata=state.get("metadata", {})
            )

            answer_text = result.get("answer", "").strip()
            if not answer_text:
                logger.error("[Generator] Empty answer after generation")
                answer_text = "Error: empty response from LLM"

            logger.info(
                f"[Generator] Completed generation (success: {result['generation_successful']}, confidence: {result.get('confidence', 0):.2f})")
            logger.debug(f"[Generator] Generated answer preview: {answer_text[:100]}...")

            return {
                **state,
                "answer": answer_text,
                "confidence": result["confidence"],
                "metadata": {
                    **state.get("metadata", {}),
                    "generation_result": result
                },
                "history": [
                    *state.get("history", []),
                    {
                        "role": "assistant",
                        "action": "generation",
                        "content": answer_text,
                        "confidence": result["confidence"]
                    }
                ]
            }

        return RunnableLambda(generator_function)


def generator_node(state: GraphState) -> GraphState:
    """
    Standalone generator node function for LangGraph.

    Factory function that creates a GeneratorNode instance and executes
    the generation pipeline. Suitable for direct use in LangGraph graphs.

    Args:
        state: Current graph state with question and retrieved documents

    Returns:
        Updated state containing the generated answer and metadata
    """
    generator = GeneratorNode()
    return generator.as_runnable().invoke(state)