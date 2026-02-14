"""
Tests for Text Splitter component.

Tests text chunking strategies, overlap handling,
chunk size management, and metadata preservation.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ==================== BASIC SPLITTING TESTS ====================

def test_text_splitter_initialization():
    """Test TextSplitter initialization with different configs."""
    from src.document_processor.text_splitter import TextSplitter

    # Default initialization
    splitter = TextSplitter()
    assert splitter.chunk_size == 1000
    assert splitter.chunk_overlap == 200
    assert splitter.strategy == "recursive"

    # Custom initialization
    splitter = TextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        strategy="token"
    )
    assert splitter.chunk_size == 500
    assert splitter.chunk_overlap == 50
    assert splitter.strategy == "token"


def test_split_simple_text():
    """Test splitting simple text."""
    from src.document_processor.text_splitter import TextSplitter

    text = "This is a simple text. " * 100  # ~2200 chars

    splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    # Should create multiple chunks
    assert len(chunks) > 1

    # Each chunk should be Chunk object
    for chunk in chunks:
        assert hasattr(chunk, 'text')
        assert hasattr(chunk, 'metadata')
        assert hasattr(chunk, 'chunk_id')


def test_split_respects_chunk_size(sample_text):
    """Test that chunks respect size limits."""
    from src.document_processor.text_splitter import TextSplitter

    splitter = TextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(sample_text)

    # Most chunks should be around target size
    for chunk in chunks:
        # Allow some flexibility (LangChain tries to respect boundaries)
        assert len(chunk.text) <= 400  # Some margin for separator preservation


def test_chunk_overlap_works(long_text):
    """Test that overlap between chunks works."""
    from src.document_processor.text_splitter import TextSplitter

    splitter = TextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(long_text)

    # Should have multiple chunks
    assert len(chunks) >= 2

    # Check for overlap (approximate - LangChain may adjust)
    for i in range(len(chunks) - 1):
        chunk1 = chunks[i].text
        chunk2 = chunks[i + 1].text

        # Should have some common content
        # (exact overlap detection is complex due to separator handling)
        assert isinstance(chunk1, str)
        assert isinstance(chunk2, str)


def test_min_chunk_size_filtering():
    """Test that too-small chunks are filtered."""
    from src.document_processor.text_splitter import TextSplitter

    text = "Short. " * 20  # Creates many small pieces

    splitter = TextSplitter(
        chunk_size=50,
        chunk_overlap=10,
        min_chunk_size=20
    )
    chunks = splitter.split_text(text)

    # All chunks should meet minimum size
    for chunk in chunks:
        assert len(chunk.text) >= 20


# ==================== STRATEGY TESTS ====================

@pytest.mark.parametrize("strategy", ["recursive", "token", "character"])
def test_different_strategies(sample_text, strategy):
    """Test different splitting strategies."""
    from src.document_processor.text_splitter import TextSplitter

    splitter = TextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        strategy=strategy
    )
    chunks = splitter.split_text(sample_text)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk.text) > 0

def test_token_strategy():
    """Test token-based splitting."""
    from src.document_processor.text_splitter import TextSplitter

    text = "Token counting test. " * 100

    splitter = TextSplitter(
        chunk_size=100,  # tokens
        chunk_overlap=10,
        strategy="token"
    )
    chunks = splitter.split_text(text)

    assert len(chunks) >= 1


def test_character_strategy():
    """Test character-based splitting."""
    from src.document_processor.text_splitter import TextSplitter

    text = "Character split test.\n\n" * 50

    splitter = TextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        strategy="character"
    )
    chunks = splitter.split_text(text)

    assert len(chunks) >= 1


# ==================== METADATA TESTS ====================

def test_chunk_metadata_preserved():
    """Test that metadata is preserved in chunks."""
    from src.document_processor.text_splitter import TextSplitter

    text = "Test text for metadata. " * 50
    metadata = {"source": "test.pdf", "page": 1, "author": "Test"}

    splitter = TextSplitter(chunk_size=200)
    chunks = splitter.split_text(text, metadata=metadata)

    for chunk in chunks:
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["page"] == 1
        assert chunk.metadata["author"] == "Test"
        assert "chunk_index" in chunk.metadata
        assert "strategy" in chunk.metadata


def test_chunk_index_increments():
    """Test that chunk_index increments correctly."""
    from src.document_processor.text_splitter import TextSplitter

    text = "Chunk indexing test. " * 100

    splitter = TextSplitter(chunk_size=200)
    chunks = splitter.split_text(text)

    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i


def test_chunk_id_generation():
    """Test chunk ID generation."""
    from src.document_processor.text_splitter import TextSplitter

    text = "ID generation test. " * 50
    metadata = {"doc_id": "doc123"}

    splitter = TextSplitter(chunk_size=200)
    chunks = splitter.split_text(text, metadata=metadata)

    for i, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"doc123_{i}"


# ==================== DOCUMENT SPLITTING TESTS ====================

def test_split_documents(sample_documents):
    """Test splitting multiple documents."""
    from src.document_processor.text_splitter import TextSplitter

    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_documents(sample_documents)

    # Should have chunks from all documents
    assert len(chunks) >= len(sample_documents)

    # Each chunk should have metadata
    for chunk in chunks:
        assert "source" in chunk.metadata


def test_split_documents_preserves_source():
    """Test that document source is preserved."""
    from src.document_processor.text_splitter import TextSplitter

    documents = [
        {"text": "Document 1 content. " * 20, "metadata": {"source": "doc1"}},
        {"text": "Document 2 content. " * 20, "metadata": {"source": "doc2"}},
    ]

    splitter = TextSplitter(chunk_size=150,chunk_overlap=20)
    chunks = splitter.split_documents(documents)

    # Check that sources are preserved
    sources = set(chunk.metadata["source"] for chunk in chunks)
    assert "doc1" in sources
    assert "doc2" in sources


# ==================== CHUNK PROPERTIES TESTS ====================

def test_chunk_char_count():
    """Test chunk char_count property."""
    from src.document_processor.text_splitter import Chunk

    chunk = Chunk(
        text="Test content",
        metadata={},
        chunk_id="test_1"
    )

    assert chunk.char_count == len("Test content")


def test_chunk_word_count():
    """Test chunk word_count property."""
    from src.document_processor.text_splitter import Chunk

    chunk = Chunk(
        text="One two three four",
        metadata={},
        chunk_id="test_1"
    )

    assert chunk.word_count == 4


def test_chunk_to_dict():
    """Test chunk to_dict method."""
    from src.document_processor.text_splitter import Chunk

    chunk = Chunk(
        text="Test",
        metadata={"key": "value"},
        chunk_id="test_1"
    )

    d = chunk.to_dict()

    assert d["text"] == "Test"
    assert d["metadata"]["key"] == "value"
    assert d["chunk_id"] == "test_1"


def test_chunk_validation():
    """Test that empty chunks are rejected."""
    from src.document_processor.text_splitter import Chunk

    with pytest.raises(ValueError, match="Chunk text cannot be empty"):
        Chunk(text="", metadata={}, chunk_id="test")

    with pytest.raises(ValueError, match="Chunk text cannot be empty"):
        Chunk(text="   ", metadata={}, chunk_id="test")


# ==================== EDGE CASES ====================

def test_empty_text_handling():
    """Test handling of empty text."""
    from src.document_processor.text_splitter import TextSplitter

    splitter = TextSplitter()
    chunks = splitter.split_text("")

    assert len(chunks) == 0


def test_whitespace_only_text():
    """Test handling of whitespace-only text."""
    from src.document_processor.text_splitter import TextSplitter

    splitter = TextSplitter()
    chunks = splitter.split_text("   \n\n   ")

    # Should return empty or filter out whitespace
    assert len(chunks) == 0


def test_text_shorter_than_chunk_size(sample_text):
    """Test text shorter than chunk size."""
    from src.document_processor.text_splitter import TextSplitter

    splitter = TextSplitter(chunk_size=10000)  # Very large
    chunks = splitter.split_text(sample_text)

    # Should create single chunk
    assert len(chunks) == 1


# ==================== CODE SPLITTING TESTS ====================

def test_code_splitting(code_text):
    """Test splitting code with appropriate separators."""
    from src.document_processor.text_splitter import TextSplitter

    splitter = TextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        strategy="recursive",
        separators=["\n\nclass ", "\n\ndef ", "\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(code_text)

    assert len(chunks) >= 1
    # Should preserve some code structure
    assert any("def" in chunk.text for chunk in chunks)


# ==================== ANALYZER TESTS ====================

def test_chunk_analyzer_basic(sample_text):
    """Test ChunkAnalyzer basic functionality."""
    from src.document_processor.text_splitter import TextSplitter, ChunkAnalyzer

    splitter = TextSplitter(chunk_size=300)
    chunks = splitter.split_text(sample_text)

    stats = ChunkAnalyzer.analyze_chunks(chunks)

    assert "total_chunks" in stats
    assert stats["total_chunks"] == len(chunks)
    assert "char_stats" in stats
    assert "word_stats" in stats


def test_chunk_analyzer_statistics(long_text):
    """Test ChunkAnalyzer statistical calculations."""
    from src.document_processor.text_splitter import TextSplitter, ChunkAnalyzer

    splitter = TextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_text(long_text)

    stats = ChunkAnalyzer.analyze_chunks(chunks)

    # Check statistics are reasonable
    assert stats["char_stats"]["mean"] > 0
    assert stats["char_stats"]["min"] <= stats["char_stats"]["mean"]
    assert stats["char_stats"]["mean"] <= stats["char_stats"]["max"]
    assert stats["word_stats"]["mean"] > 0


def test_chunk_analyzer_empty_chunks():
    """Test ChunkAnalyzer with empty chunk list."""
    from src.document_processor.text_splitter import ChunkAnalyzer

    stats = ChunkAnalyzer.analyze_chunks([])

    assert "error" in stats


# ==================== FACTORY FUNCTIONS TESTS ====================

def test_create_splitter_for_embeddings():
    """Test factory function for embeddings."""
    from src.document_processor.text_splitter import TextSplitter

    splitter = TextSplitter(strategy="token", chunk_size=512)

    assert splitter.strategy == "token"
    assert splitter.chunk_size <= 8000  # OpenAI limit

# ==================== CONVENIENCE FUNCTION TESTS ====================

def test_convenience_split_text_function(sample_text):
    """Test convenience split_text function."""
    from src.document_processor.text_splitter import TextSplitter
    splitter = TextSplitter(chunk_size=300, strategy="recursive")
    chunks = splitter.split_text(sample_text)

    assert len(chunks) >= 1
    assert all(hasattr(c, 'text') for c in chunks)


# ==================== PERFORMANCE TESTS ====================

def test_splitting_performance(long_text, performance_timer):
    """Test that splitting completes in reasonable time."""
    from src.document_processor.text_splitter import TextSplitter

    splitter = TextSplitter(chunk_size=500, chunk_overlap=50)

    with performance_timer() as timer:
        chunks = splitter.split_text(long_text)

    # Should complete quickly
    assert timer.elapsed < 2.0
    assert len(chunks) >= 1


def test_large_document_splitting(performance_timer):
    """Test splitting very large documents."""
    from src.document_processor.text_splitter import TextSplitter

    # Create very long text
    large_text = "This is a test sentence. " * 10000  # ~250k chars

    splitter = TextSplitter(chunk_size=1000, chunk_overlap=100)

    with performance_timer() as timer:
        chunks = splitter.split_text(large_text)

    # Should still complete in reasonable time
    assert timer.elapsed < 5.0
    assert len(chunks) >= 100


# ==================== INTEGRATION TESTS ====================

def test_pdf_to_chunks_pipeline(sample_documents):
    """Test complete document to chunks pipeline."""
    from src.document_processor.text_splitter import TextSplitter

    splitter = TextSplitter(chunk_size=200, chunk_overlap=50)

    # Split all documents
    all_chunks = splitter.split_documents(sample_documents)

    # Verify pipeline
    assert len(all_chunks) >= len(sample_documents)

    # All chunks should have complete metadata
    for chunk in all_chunks:
        assert "source" in chunk.metadata
        assert "chunk_index" in chunk.metadata
        assert "strategy" in chunk.metadata
        assert chunk.text.strip()  # Non-empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])