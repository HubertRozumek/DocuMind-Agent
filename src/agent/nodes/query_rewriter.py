import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from src.agent.graph_state import GraphState, StateManager

logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    A node that rephrases the user's question for better relevance.
    """

    def __init__(self, model_name: str = "mistral:7b"):
        self.model_name = model_name
        self.llm = OllamaLLM(model=model_name)
        self.prompt = self._create_rewrite_prompt()
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _create_rewrite_prompt(self) -> PromptTemplate:
        template = """
            You are an assistant tasked with rewriting user questions to improve
            search results in corporate documents.

            Your task: Analyze the original question and suggest 1-3 alternative
            versions that are:
            1. More precise and detailed
            2. Use corporate terminology
            3. Preserve the original intent
            4. Suitable for searching in formal documents

            Context: The question was asked by an employee searching for information
            in company documents (policies, regulations, procedures).

            Original question: {question}

            Previous search attempts (if any): {search_history}

            Rewriting guidelines:
            - If the question is too general, add corporate context
            - Use technical synonyms for terms
            - Break complex questions into simpler parts
            - Add specifiers (e.g., "procedure", "policy", "form")

            Rewritten versions (one per line, no numbering):
        """
        return PromptTemplate(template=template, input_variables=["question", "search_history"])

    def rewrite_question(self, state: GraphState) -> GraphState:
        """ """
        logger.info(f"Rewriting question: {state['question']}")

        search_history = state.get("search_history", [])
        history_text = "\n".join(search_history[-3:]) if search_history else "No history"

        rewritten_questions_text = self.chain.invoke({"question": state["question"], "search_history": history_text})

        rewritten_questions = [
            q.strip() for q in rewritten_questions_text.split("\n") if q.strip() and q.strip().lower() != state["question"].lower()
        ]

        # Fallback if LLM returns empty or invalid
        if not rewritten_questions:
            logger.warning("No valid rewrites from LLM, using fallback")
            base_question = state["question"]
            rewritten_questions = [
                f"procedure {base_question}",
                f"policy regarding {base_question}",
                f"{base_question} documentation",
            ]

        logger.info(f"Generated {len(rewritten_questions)} rewrites: {rewritten_questions}")

        updated_state = StateManager.update_state(
            state,
            rewritten_questions=rewritten_questions,
            current_rewrite_index=0,
            search_query=rewritten_questions[0] if rewritten_questions else state["search_query"],
            metadata={
                **state.get("metadata", {}),
                "rewrite_action": "query_rewritten",
                "original_question": state["question"],
                "rewritten_count": len(rewritten_questions),
            },
        )

        # Add to decision log
        decision_log = updated_state.get("decision_log", [])
        decision_log.append(
            {
                "step": "query_rewriter",
                "original_question": state["question"],
                "rewritten_questions": rewritten_questions,
                "iteration": updated_state["iterations"],
            }
        )
        updated_state["decision_log"] = decision_log

        logger.info(f"Rewrite complete. State keys: {list(updated_state.keys())}")
        logger.info(f"rewritten_questions in state: {len(updated_state.get('rewritten_questions', []))}")

        return updated_state


def query_rewriter_node(state: GraphState) -> GraphState:
    """Node function to rewrite a question."""
    rewriter = QueryRewriter()
    return rewriter.rewrite_question(state)
