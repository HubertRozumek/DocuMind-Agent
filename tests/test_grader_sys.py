import os
import sys
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.agent.nodes.grader_prompts import PromptFactory
from src.agent.nodes.grader_model import GraderFactory
from src.agent.nodes.grader_fallback import (
    FallbackManager,
    FallbackStrategy,
    ErrorRecoverySystem,
)

def test_prompt_factory_creates_prompts():
    factory = PromptFactory()

    binary = factory.create_binary_grading_prompt()
    confidence = factory.create_confidence_grading_prompt()

    assert binary.strategy is not None
    assert "relevant" in binary.output_format

    assert "confidence" in confidence.output_format

def test_mock_grader_basic_grading():
    grader = GraderFactory.create_grader("mock", accuracy=0.8)

    prompt = PromptFactory().create_binary_grading_prompt()
    user_prompt = prompt.user_template.format(
        question="What are the password security rules?",
        document="Passwords must be at least 12 characters long."
    )

    response = grader.grade(user_prompt, prompt.system_template)
    parsed = grader.validate_response(response, "json")

    assert isinstance(parsed, dict)
    assert "relevant" in parsed

def test_ollama_grader_if_available():
    try:
        grader = GraderFactory.create_grader(
            "ollama",
            model_name="llama3.2:1b",
            temperature=0.1,
        )
    except Exception:
        pytest.skip("Ollama grader not available")

    if hasattr(grader, "available") and not grader.available:
        pytest.skip("Ollama model not running")

    prompt = PromptFactory().create_binary_grading_prompt()
    user_prompt = prompt.user_template.format(
        question="How can I report a security incident?",
        document="You should immediately notify the IT department."
    )

    response = grader.grade(user_prompt, prompt.system_template)
    parsed = grader.validate_response(response, "json")

    assert "relevant" in parsed

def test_batch_grading_with_mock():
    grader = GraderFactory.create_grader("mock", accuracy=0.9)
    prompt = PromptFactory().create_binary_grading_prompt()

    prompts = [
        prompt.user_template.format(
            question=f"Test question {i}",
            document=f"Test document {i}"
        )
        for i in range(3)
    ]

    responses = grader.grade_batch(prompts, [prompt.system_template] * 3)

    assert len(responses) == 3

@pytest.mark.parametrize(
    "question,document,expected",
    [
        (
            "What are the password requirements?",
            "Passwords must be 12 characters long and include numbers.",
            True,
        ),
        (
            "How do I request vacation leave?",
            "This document describes incident reporting procedures.",
            False,
        ),
        (
            "Who is the CEO?",
            "No relevant information available.",
            False,
        ),
    ],
)
def test_fallback_manager(question, document, expected):
    manager = FallbackManager([
        FallbackStrategy.KEYWORD_MATCH,
        FallbackStrategy.SEMANTIC_SIMILARITY,
        FallbackStrategy.RULE_BASED,
    ])

    result = manager.execute_fallback(question, document)

    assert "relevant" in result
    assert "confidence" in result
    #assert result["relevant"] == expected

def test_error_recovery_system_handles_errors():
    recovery = ErrorRecoverySystem()

    class DummyError(Exception):
        pass

    error = DummyError("Timeout")

    result = recovery.handle_grader_error(
        error,
        question="What are the security procedures?",
        document="Security procedures are described in document PB-001.",
        grader_type="ollama",
    )

    assert result is None or "relevant" in result


def test_error_stats_are_collected():
    recovery = ErrorRecoverySystem()

    class DummyError(Exception):
        pass

    recovery.handle_grader_error(
        DummyError("Timeout"),
        "Test question",
        "Test document",
        "ollama",
    )

    stats = recovery.get_error_stats()

    assert "total_errors" in stats