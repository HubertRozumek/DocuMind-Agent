import logging
from typing import Dict, Any, List, Optional, Callable
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

from src.agent.graph_state import GraphState, StateManager

logger = logging.getLogger(__name__)


class GeneratorNode:
    """
    Node that generates final answers based on relevant documents.
    """

    def __init__(
            self,
            model_name: str = "phi3:mini",
            temperature: float = 0.1,
            max_tokens: int = 1000
    ):
        """
        Initialize the answer generator.

        Args:
            model_name: Name of the LLM model
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            self.llm = Ollama(
                model=model_name,
                temperature=temperature,
                num_predict=max_tokens
            )
            logger.info(f"Generator initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize generator with {model_name}: {e}")
            self.llm = Ollama(model="llama3.2:1b")
            logger.warning(f"Using fallback model: llama3.2:1b")

        self.prompt = self._create_answer_prompt()
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _create_answer_prompt(self) -> PromptTemplate:
        """Create prompt template for answer generation."""
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
            metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate an answer based on documents.

        Args:
            question: User question
            documents: Relevant documents
            metadata: Optional metadata

        Returns:
            Dictionary with generated answer and metadata
        """
        if not documents:
            return {
                "answer": "I couldn't find any relevant documents to answer your question.",
                "confidence": 0.0,
                "sources_used": 0,
                "documents_considered": 0
            }

        try:
            # Combine documents
            combined_docs = "\n\n".join([
                f"Document {i + 1}:\n{doc[:2000]}"
                for i, doc in enumerate(documents[:5])
            ])

            answer = self.chain.invoke({
                "question": question,
                "documents": combined_docs
            })

            confidence = min(0.9, 0.3 + (len(documents) * 0.1))

            return {
                "answer": answer.strip(),
                "confidence": confidence,
                "sources_used": len(documents),
                "documents_considered": len(documents),
                "generation_successful": True
            }

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")

            return {
                "answer": f"I encountered an error while generating the answer: {str(e)}",
                "confidence": 0.0,
                "sources_used": 0,
                "documents_considered": len(documents),
                "generation_successful": False,
                "error": str(e)
            }

    def as_runnable(self) -> Callable:
        """
        Convert to LangGraph compatible runnable.

        Returns:
            Function that takes a state and returns updated state
        """

        def generator_function(state: GraphState) -> GraphState:
            """
            Generator function for LangGraph.

            Args:
                state: Current graph state

            Returns:
                Updated state with generated answer
            """
            logger.info(f"[Generator Function] Generating answer with {len(state.get('relevant_docs', []))} documents")

            if state.get("tool_result"):
                return StateManager.update_state(
                    state,
                    answer=state["tool_result"],
                    confidence=0.95,
                    metadata={
                        **state.get("metadata", {}),
                        "generation_result": {
                            "answer": state["tool_result"],
                            "confidence": 0.95,
                            "sources_used": 0,
                            "tool_used": state.get("tool_used"),
                        }
                    }
                )

            relevant_docs = state.get("relevant_docs", [])
            question = state.get("question", "")

            # If no relevant docs but we have some docs, use them with low confidence
            if not relevant_docs:
                all_docs = state.get("documents", [])
                if all_docs:
                    relevant_docs = all_docs[:2]
                    logger.warning("No relevant docs found, using retrieved docs with low confidence")
                else:
                    return StateManager.update_state(
                        state,
                        answer="I couldn't find any documents to answer your question.",
                        confidence=0.0
                    )

            generation_result = self.generate_answer(
                question=question,
                documents=relevant_docs,
                metadata=state.get("metadata", {})
            )

            updated_state = StateManager.update_state(
                state,
                answer=generation_result["answer"],
                confidence=generation_result["confidence"],
                metadata={
                    **state.get("metadata", {}),
                    "generation_result": generation_result
                }
            )

            history_entry = {
                "role": "assistant",
                "action": "generation",
                "content": generation_result["answer"],
                "confidence": generation_result["confidence"],
                "details": {
                    "sources_used": generation_result["sources_used"],
                    "generation_successful": generation_result.get("generation_successful", False)
                }
            }

            updated_state = StateManager.add_to_history(updated_state, history_entry)

            logger.info(f"[Generator Function] Answer generated with confidence: {generation_result['confidence']:.2f}")

            return updated_state

        return RunnableLambda(generator_function)


def generator_node(state: GraphState) -> GraphState:
    """
    Node function for answer generation.

    Args:
        state: Current graph state

    Returns:
        Updated state
    """
    generator = GeneratorNode()
    return generator.as_runnable().invoke(state)