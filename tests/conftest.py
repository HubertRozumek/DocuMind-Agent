"""
Shared pytest fixtures for RAG project tests.

This file contains reusable fixtures for creating test data,
temporary files, mock objects, and test configurations.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
import logging

# Suppress warnings during tests
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


# ==================== PDF GENERATION FIXTURES ====================

@pytest.fixture
def temp_dir():
    """
    Create temporary directory that auto-cleans after test.

    Yields:
        Path: Temporary directory path
    """
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def create_test_pdf():
    """
    Factory fixture to create test PDFs with custom content.

    Returns:
        Callable that creates PDF and returns path

    Example:
        >>> pdf_path = create_test_pdf("Test content", num_pages=2)
    """
    created_files = []

    def _create_pdf(
            content: str = "Test document content",
            num_pages: int = 1,
            filename: str = "test.pdf"
    ) -> Path:
        """
        Create a PDF file with specified content.

        Args:
            content: Text content for each page
            num_pages: Number of pages to create
            filename: PDF filename

        Returns:
            Path to created PDF
        """
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch

        # Create in temp directory
        temp_dir = Path(tempfile.mkdtemp())
        pdf_path = temp_dir / filename

        # Create PDF
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        width, height = letter

        for page_num in range(num_pages):
            # Add page number to content
            page_content = f"{content}\n\nPage {page_num + 1} of {num_pages}"

            # Write text
            text_object = c.beginText(inch, height - inch)
            text_object.setFont("Helvetica", 12)

            # Split content into lines
            lines = page_content.split('\n')
            for line in lines:
                text_object.textLine(line)

            c.drawText(text_object)

            if page_num < num_pages - 1:
                c.showPage()

        c.save()

        created_files.append(pdf_path)
        return pdf_path

    yield _create_pdf

    # Cleanup all created PDFs
    for pdf_path in created_files:
        if pdf_path.exists():
            pdf_path.unlink()
        # Also cleanup parent temp directories
        if pdf_path.parent.exists():
            shutil.rmtree(pdf_path.parent, ignore_errors=True)


@pytest.fixture
def sample_pdf(create_test_pdf):
    """
    Pre-created sample PDF with standard content.

    Returns:
        Path: Path to sample PDF
    """
    content = """
    This is a sample document for testing purposes.

    It contains multiple paragraphs with various information.
    The content is designed to test PDF loading and text extraction.

    Key features to test:
    - Multi-line text
    - Paragraph breaks
    - Special characters: @#$%
    - Numbers: 123456789

    This document helps verify that the PDF loader works correctly.
    """
    return create_test_pdf(content, num_pages=2, filename="sample.pdf")


@pytest.fixture
def complex_pdf(create_test_pdf):
    """
    Complex PDF with varied formatting for advanced tests.

    Returns:
        Path: Path to complex PDF
    """
    content = """
    CHAPTER 1: INTRODUCTION

    This is a comprehensive test document that includes various elements.

    Section 1.1: Background
    The background section provides context about the testing framework.
    It includes multiple sentences to test chunking algorithms.

    Section 1.2: Methodology
    Our testing methodology follows industry best practices.
    We use pytest for unit testing and integration testing.

    Key Points:
    - Automated testing is essential
    - Code coverage should exceed 80%
    - Tests should be maintainable

    CHAPTER 2: IMPLEMENTATION DETAILS

    The implementation uses modern Python practices including:
    - Type hints for better code clarity
    - Docstrings for documentation
    - Logging for debugging

    Conclusion: This document tests advanced PDF processing capabilities.
    """
    return create_test_pdf(content, num_pages=3, filename="complex.pdf")


@pytest.fixture
def multiple_pdfs(create_test_pdf):
    """
    Create multiple test PDFs.

    Returns:
        List[Path]: List of PDF paths
    """
    pdfs = []

    for i in range(3):
        content = f"""
        Document {i + 1}

        This is test document number {i + 1}.
        It contains unique content for testing batch processing.

        Content includes: data_{i}, value_{i * 10}, score_{i * 100}
        """
        pdf = create_test_pdf(content, num_pages=1, filename=f"doc_{i}.pdf")
        pdfs.append(pdf)

    return pdfs


# ==================== TEXT FIXTURES ====================

@pytest.fixture
def sample_text():
    """Sample text for text splitter tests."""
    return """
    Introduction to Machine Learning

    Machine learning is a subset of artificial intelligence that focuses on 
    building systems that can learn from data. It has revolutionized many 
    industries including healthcare, finance, and technology.

    Types of Machine Learning:

    1. Supervised Learning: Uses labeled data to train models
    2. Unsupervised Learning: Finds patterns in unlabeled data
    3. Reinforcement Learning: Learns through interaction with environment

    Applications of machine learning are vast and growing. From recommendation 
    systems to autonomous vehicles, ML is transforming how we interact with 
    technology. The field continues to evolve with new algorithms and techniques.

    Deep learning, a subset of machine learning, uses neural networks with 
    multiple layers. These networks can learn hierarchical representations of 
    data, making them particularly effective for tasks like image recognition 
    and natural language processing.

    Conclusion: Machine learning will continue to shape the future of technology.
    """


@pytest.fixture
def long_text():
    """Long text for chunking tests."""
    paragraphs = []
    for i in range(20):
        paragraphs.append(
            f"Paragraph {i + 1}: This is a test paragraph with sufficient content "
            f"to test chunking behavior. It contains {50 * (i + 1)} characters of text. "
            f"The content is designed to verify that the text splitter handles long "
            f"documents correctly and creates appropriate chunks. Each paragraph has "
            f"unique markers like paragraph_{i} and value_{i * 10}."
        )
    return "\n\n".join(paragraphs)


@pytest.fixture
def code_text():
    """Sample code for code-specific tests."""
    return '''
def calculate_sum(numbers: List[int]) -> int:
    """Calculate sum of numbers."""
    return sum(numbers)

class DataProcessor:
    """Process data efficiently."""

    def __init__(self, config: Dict):
        self.config = config

    def process(self, data: List) -> List:
        """Process data."""
        results = []
        for item in data:
            results.append(self._transform(item))
        return results

    def _transform(self, item):
        """Transform single item."""
        return item * 2
'''


# ==================== DOCUMENT FIXTURES ====================

@pytest.fixture
def sample_documents():
    """Sample documents for retriever/grader tests."""
    return [
        {
            "text": "Python is a high-level programming language. It emphasizes code readability.",
            "metadata": {"source": "doc1", "page": 1, "topic": "programming"}
        },
        {
            "text": "Machine learning models require large amounts of training data to achieve good performance.",
            "metadata": {"source": "doc2", "page": 1, "topic": "ml"}
        },
        {
            "text": "The weather today is sunny with clear skies. Perfect for outdoor activities.",
            "metadata": {"source": "doc3", "page": 1, "topic": "weather"}
        },
        {
            "text": "Deep learning uses neural networks with multiple layers to learn representations.",
            "metadata": {"source": "doc4", "page": 1, "topic": "ml"}
        },
        {
            "text": "Python's simplicity and extensive libraries make it popular for data science.",
            "metadata": {"source": "doc5", "page": 1, "topic": "programming"}
        }
    ]


@pytest.fixture
def chunked_documents():
    """Pre-chunked documents for vector store tests."""
    from dataclasses import dataclass
    from typing import Dict, Any

    @dataclass
    class MockChunk:
        text: str
        metadata: Dict[str, Any]
        chunk_id: str

    return [
        MockChunk(
            text="Python is a versatile programming language",
            metadata={"source": "doc1", "chunk_index": 0},
            chunk_id="doc1_0"
        ),
        MockChunk(
            text="Machine learning is transforming technology",
            metadata={"source": "doc2", "chunk_index": 0},
            chunk_id="doc2_0"
        ),
        MockChunk(
            text="Data science requires statistical knowledge",
            metadata={"source": "doc3", "chunk_index": 0},
            chunk_id="doc3_0"
        )
    ]


# ==================== CONFIGURATION FIXTURES ====================

@pytest.fixture
def vector_store_config(temp_dir):
    """Configuration for vector store tests."""
    return {
        "collection_name": "test_collection",
        "persist_directory": str(temp_dir / "chroma_test"),
        "reset_on_start": True
    }


@pytest.fixture
def grader_config():
    """Configuration for grader tests."""
    return {
        "grader_type": "robust",
        "confidence_threshold": 0.6,
        "model_name": "phi3:mini"
    }


@pytest.fixture
def generator_config():
    """Configuration for generator tests."""
    return {
        "model_name": "phi3:mini",
        "temperature": 0.1
    }


@pytest.fixture
def agent_config(vector_store_config, grader_config, generator_config):
    """Complete agent configuration."""
    return {
        "vector_store_config": vector_store_config,
        "grader_config": grader_config,
        "generator_config": generator_config,
        "max_iterations": 2,
        "use_tools": False  # Disable for testing
    }


# ==================== MOCK FIXTURES ====================

@pytest.fixture
def mock_embedding_function():
    """Mock embedding function for tests."""

    from src.vector_store.embeddings_manager import EmbeddingManager
    embedding_manager = EmbeddingManager()

    return embedding_manager.chroma_embedding_function()


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""

    def _mock_response(prompt: str) -> str:
        if "relevant" in prompt.lower():
            return "yes"
        elif "rewrite" in prompt.lower():
            return "What is machine learning?"
        else:
            return "This is a mock response to the query."

    return _mock_response


# ==================== UTILITY FIXTURES ====================

@pytest.fixture
def capture_logs():
    """Capture logs during test execution."""
    import logging
    from io import StringIO

    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)

    logger = logging.getLogger()
    logger.addHandler(handler)

    yield log_capture

    logger.removeHandler(handler)


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration between tests."""
    yield
    logging.getLogger().handlers.clear()


# ==================== PERFORMANCE FIXTURES ====================

@pytest.fixture
def performance_timer():
    """Timer for performance tests."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, *args):
            self.end_time = time.time()

        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None

    return Timer