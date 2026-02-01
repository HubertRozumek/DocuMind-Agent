import logging
from typing import Dict, Any, List
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from src.agent.graph_state import GraphState

logger = logging.getLogger(__name__)

class QueryRewriter:
    """
    A node that rephrases the user's question for better relevance.
    """

    def __init__(self, model_name: str = "phi3:mini"):
        """
        Initialise the query rewriter with a language model.

        Args:
            model_name: The name of the language model.(default "phi3:mini")
        """

        self.model_name = model_name
        self.llm = Ollama(model=model_name)
        self.prompt = self._create_rewrite_prompt()
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _create_rewrite_prompt(self) -> PromptTemplate:
        """
        Creates a prompt template for rewriting questions.
        """
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
        """
        Rewrites a question based on search history.

        Args:
            state: Current graph state  # ← GraphState, nie Dict

        Returns:
            Updated state with rewritten questions
        """
        logger.info(f"Rewriting question: {state['question']}")

        search_history = state.get("search_history", [])
        history_text = "\n".join(search_history[-3:]) if search_history else "No history"

        rewritten_questions_text = self.chain.invoke({
            "question": state["question"],
            "search_history": history_text
        })

        rewritten_questions = [
            q.strip() for q in rewritten_questions_text.split("\n")
            if q.strip()
        ]

        if not rewritten_questions:
            base_question = state["question"]
            rewritten_questions = [
                f"procedure {base_question}",
                f"policy {base_question} in the company",
                f"how to {base_question} - workplace regulations"
            ]

        state["rewritten_questions"] = rewritten_questions
        state["current_rewrite_index"] = 0
        state["iterations"] = state.get("iterations", 0) + 1

        decision_log = state.get("decision_log", [])
        decision_log.append({
            "step": "query_rewriter",
            "original_question": state["question"],
            "rewritten_questions": rewritten_questions,
            "iteration": state["iterations"]
        })
        state["decision_log"] = decision_log

        logger.info(f"Generated {len(rewritten_questions)} rewrites")

        return state

    def query_rewriter_node(state: GraphState) -> GraphState:
        """
        Node function to rewrite a question.

        Args:
            state: Current graph state

        Returns:
            Updated state
        """
        rewriter = QueryRewriter()
        return rewriter.rewrite_question(state)