import json
import numpy as np
import pytest
from typing import List, Dict, Any, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.agent.nodes.grader_model import GraderFactory, BaseGraderModel
from src.agent.nodes.grader_prompts import PromptFactory

@pytest.fixture(scope="session")
def grader_dataset(tmp_path_factory):

    dataset = [
        {
            "id": "ex_001",
            "question": "What are password security rules?",
            "document": "Passwords must be at least 12 characters long and include digits and symbols.",
            "relevant": True,
        },
        {
            "id": "ex_002",
            "question": "What are password security rules?",
            "document": "Employees work from 8am to 4pm.",
            "relevant": False,
        },
        {
            "id": "ex_003",
            "question": "How to report a security incident?",
            "document": "Security incidents must be reported immediately to the IT department.",
            "relevant": True,
        },
        {
            "id": "ex_004",
            "question": "How to report a security incident?",
            "document": "Annual leave entitlement is 20 days.",
            "relevant": False,
        },
    ]

    path = tmp_path_factory.mktemp("data") / "grader_dataset.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    return dataset

@pytest.fixture(scope="session")
def binary_prompts(grader_dataset):

    prompt = PromptFactory.create_binary_grading_prompt()

    prompts = []
    for item in grader_dataset:
        user_prompt = prompt.user_template.format(
            question = item["question"],
            document = item["document"],
            metadata = "{}"
        )
        prompts.append((user_prompt, prompt.system_template, item["relevant"]))

    return prompts

@pytest.fixture(scope="session")
def mock_grader() -> BaseGraderModel:
    return GraderFactory.create_grader("mock", accuracy=0.9)

def parse_grader_response(response: str) -> bool:

    try:
        parsed = json.loads(response)
        return bool(parsed.get("relevant", False))
    except Exception:
        response = response.lower()
        return any(k in response for k in ["true","yes","relevant"])

def run_grader(grader, prompts):

    predictions = []
    labels = []

    for user_prompt, system_prompt, truth in prompts:
        result = grader.grade(user_prompt, system_prompt)
        predictions.append(parse_grader_response(result))
        labels.append(truth)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }

@pytest.fixture
def perfect_grader():
    return MockGrader(accuracy=1.0)

@pytest.fixture
def random_grader():
    return MockGrader(accuracy=0.0)

def test_mock_grader_accuracy(binary_prompts, mock_grader):
    metrics = run_grader(mock_grader, binary_prompts)

    assert metrics["accuracy"] >= 0.5
    assert metrics["f1"] >= .6

def test_grader_is_deterministic(binary_prompts, mock_grader):
    m1 = run_grader(mock_grader, binary_prompts)
    m2 = run_grader(mock_grader, binary_prompts)

    assert abs(m1["accuracy"] - m2["accuracy"]) <= 0.25
    assert abs(m1["f1"] - m2["f1"]) <= 0.25


def test_grader_output_format(binary_prompts, mock_grader):
    user_prompt, system_prompt, _ = binary_prompts[0]
    response = mock_grader.grade(user_prompt, system_prompt)

    assert isinstance(response, str)
    assert response.strip() != ""

def test_mock_grader_returns_valid_json(perfect_grader):
    prompt = """
    QUESTION: What are password security rules?
    DOCUMENT: Passwords must be at least 12 characters long.
    """

    result = perfect_grader.grade(prompt)

    parsed = json.loads(result)

    assert "relevant" in parsed
    assert "reason" in parsed
    assert isinstance(parsed["relevant"], bool)

def test_mock_grader_keyword_match_relevant(perfect_grader):
    prompt = """
    QUESTION: password security rules
    DOCUMENT: password rules require strong password
    """

    result = perfect_grader.grade(prompt)
    parsed = json.loads(result)

    assert parsed["relevant"] is True

def test_mock_grader_keyword_mismatch_not_relevant(perfect_grader):
    prompt = """
    QUESTION: password security rules
    DOCUMENT: the sky is blue and the sun is bright
    """

    result = perfect_grader.grade(prompt)
    parsed = json.loads(result)

    assert parsed["relevant"] is False

def test_mock_grader_random_mode_returns_any_pattern(random_grader):
    prompt = """
    QUESTION: password security rules
    DOCUMENT: password rules require strong password
    """

    results = [json.loads(random_grader.grade(prompt)) for _ in range(10)]

    for r in results:
        assert "relevant" in r
        assert isinstance(r["relevant"], bool)

def test_mock_grader_batch_size(perfect_grader):
    prompts = [
        "QUESTION: A\nDOCUMENT: A B C",
        "QUESTION: X\nDOCUMENT: Y Z",
        "QUESTION: foo\nDOCUMENT: foo bar baz"
    ]

    results = perfect_grader.grade_batch(prompts)

    assert len(results) == len(prompts)

def test_mock_grader_batch_json(perfect_grader):
    prompts = [
        "QUESTION: A\nDOCUMENT: A B",
        "QUESTION: X\nDOCUMENT: Y"
    ]

    results = perfect_grader.grade_batch(prompts)

    for r in results:
        parsed = json.loads(r)
        assert "relevant" in parsed

def test_mock_grader_handles_invalid_prompt(perfect_grader):
    prompt = "This is totally invalid input"

    result = perfect_grader.grade(prompt)
    parsed = json.loads(result)

    assert "relevant" in parsed

from src.agent.nodes.grader_model import MockGrader

def test_validate_response_json():
    grader = MockGrader()

    response = '{"relevant": true, "reason": "ok"}'
    parsed = grader.validate_response(response, expected_format="json")

    assert parsed["relevant"] is True

def test_validate_response_invalid_json_fallback():
    grader = MockGrader()

    response = 'relevant: true, reason: ok'
    parsed = grader.validate_response(response, expected_format="json")

    assert "relevant" in parsed
    assert parsed["relevant"] is False