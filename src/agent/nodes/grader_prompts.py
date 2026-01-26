from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json

class GradingStrategy(Enum):
    """
    Document relevance grading strategies.
    """
    BINARY = "binary"
    CONFIDENCE = "confidence"
    MULTI_CRITERIA = "multi_criteria"
    REASONING = "reasoning"

@dataclass
class GradingPrompt:
    """
    Prompt template for relevance grading.
    """
    system_template: str
    user_template: str
    output_format: str
    strategy: GradingStrategy
    temperature: float = 0.1
    max_tokens: int = 500

class PromptFactory:
    """
    Factory for relevance grading prompts.
    """
    @staticmethod
    def crate_binary_grading_prompt() -> GradingPrompt:
        system_template ="""
        You are an expert in document relevance assessment.
        Your task is to determine whether the given document is relevant to the user's question.
        
        Evaluation rules:
        1. A document is RELEVANT if it directly answers the question
        2. A document is RELEVANT if it partially answers the question
        3. A document is NOT RELEVANT if it contains no information related to the question
        4. A document is NOT RELEVANT if it is too generic or vague
        
        Return the answer is JSON format with fields:
        - "relevant": true/false
        - "reason": short justification (1-2 sentences)
        """

        user_template = """
        QUESTION: {question}
        
        DOCUMENT: {document}
        Is the document relevant to the user's question?
        """
        return GradingPrompt(
            system_template=system_template,
            user_template=user_template,
            output_format='{"relevant": true/false, "reason": "justification"}',
            strategy=GradingStrategy.BINARY,
            max_tokens=200,
        )

    @staticmethod
    def create_confidence_grading_prompt() -> GradingPrompt:
        system_template = """
        You are an expert in document relevance assessment.
        Evaluate how relevant the document is to the user's question on a scale from 0 to 1.
        
        Scale:
        1.0 - Perfectly answers the question
        0.8-0.9 - Strongly answers the question
        0.6-0.7 - Partially answers the question
        0.4-0.5 - Weakly related
        0.0-0.3 - Not relevant
        
        Return JSON with:
        - "relevant": true/false (true if confidence >= 0.6)
        - "confidence": number between 0 and 1
        - "reason": justification
        """
        user_template = """
        QUESTION: {question}
        
        DOCUMENT: {document}
        Evaluate the document relevance and confidence.
        """

        return GradingPrompt(
            system_template=system_template,
            user_template=user_template,
            output_format='{"relevant": true/false, "confidence": 0-1, "reason": "justification"}',
            strategy=GradingStrategy.CONFIDENCE,
            max_tokens=250,
        )

    @staticmethod
    def create_multi_criteria_prompt() -> GradingPrompt:
        system_template = """
        You are an expert in document relevance assessment.
        Evaluate the document using the following criteria (0–5):

        1. TOPICALITY – Same topic as the question
        2. SPECIFICITY – Level of detail
        3. CURRENCY – Information freshness
        4. RELIABILITY – Source credibility
        5. COMPLETENESS – Coverage of all question aspects

        Return JSON with:
        - "relevant": true/false (true if total_score >= 15)
        - "scores": detailed scores
        - "total_score": 0–25
        - "reason": explanation
        """

        user_template = """
        QUESTION: {question}

        DOCUMENT: {document}

        METADATA (if available): {metadata}

        Evaluate according to the criteria.
        """

        return GradingPrompt(
            system_template=system_template,
            user_template=user_template,
            output_format="""{
                "relevant": true/false,
                "scores": {
                    "topicality": 0-5,
                    "specificity": 0-5,
                    "currency": 0-5,
                    "reliability": 0-5,
                    "completeness": 0-5
                },
                "total_score": 0-25,
                "reason": "explanation"
            }""",
            strategy=GradingStrategy.MULTI_CRITERIA,
            max_tokens=300
        )

    @staticmethod
    def create_reasoning_prompt() -> GradingPrompt:
        system_template = """
        You are an expert in document relevance assessment.
        Perform a deep analysis of the document in relation to the question.

        ANALYSIS STEPS:
        1. Identify key elements of the question
        2. Find corresponding information in the document
        3. Assess direct answers
        4. Assess indirect answers
        5. Identify missing critical information

        Return JSON with:
        - "relevant": true/false
        - "key_elements_found"
        - "key_elements_missing"
        - "evidence"
        - "reasoning"
        - "confidence": 0–1
        """

        user_template = """
        QUESTION: {question}

        DOCUMENT: {document}

        Perform a detailed relevance analysis.
        """

        return GradingPrompt(
            system_template=system_template,
            user_template=user_template,
            output_format="""{
                "relevant": true/false,
                "key_elements_found": [],
                "key_elements_missing": [],
                "evidence": [],
                "reasoning": "",
                "confidence": 0-1
            }""",
            strategy=GradingStrategy.REASONING,
            temperature=0.2,
            max_tokens=500
        )

    @staticmethod
    def create_fast_grading_prompt() -> GradingPrompt:
        system_template = "Determine whether the document answers the question. Reply only YES or NO."

        user_template = """
        Q: {question}
        
        D: {document}
        Answer:"""

        return GradingPrompt(
            system_template=system_template,
            user_template=user_template,
            output_format="YES or NO",
            strategy=GradingStrategy.BINARY,
            temperature=0.0,
            max_tokens=10
        )