"""
Tests for PDF Loader component.

Tests PDF loading functionality with dynamic PDF creation,
fallback mechanisms, validation, and error handling.
"""

import sys
from pathlib import Path

import pytest

# Add src to path (adjust based on your structure)
sys.path.insert(0, str(Path(__file__).parent.parent))


# ==================== BASIC LOADING TESTS ====================


def test_pdf_loader_initialization():
    """Test PDFLoader can be initialized with different configs."""
    from src.document_processor.pdf_loader import PDFLoader

    # Default initialization
    loader = PDFLoader()
    assert loader.loader_type == "auto"
    assert loader.extract_images is False

    # Custom initialization
    loader = PDFLoader(loader_type="pymupdf", extract_images=True)
    assert loader.loader_type == "pymupdf"
    assert loader.extract_images is True


def test_load_single_page_pdf(create_test_pdf):
    """Test loading a simple single-page PDF."""
    from src.document_processor.pdf_loader import PDFLoader

    # Create test PDF
    pdf_path = create_test_pdf("Simple test content", num_pages=1)

    # Load it
    loader = PDFLoader()
    documents = loader.load_pdf(str(pdf_path))

    # Assertions
    assert len(documents) == 1
    assert "text" in documents[0]
    assert "metadata" in documents[0]
    assert len(documents[0]["text"]) > 0
    assert "Simple test content" in documents[0]["text"]


def test_load_multi_page_pdf(create_test_pdf):
    """Test loading a multi-page PDF."""
    from src.document_processor.pdf_loader import PDFLoader

    # Create 3-page PDF
    pdf_path = create_test_pdf("Multi page content", num_pages=3)

    loader = PDFLoader()
    documents = loader.load_pdf(str(pdf_path))

    # Should have 3 pages
    assert len(documents) == 3

    # Each page should have content
    for i, doc in enumerate(documents):
        assert "text" in doc
        assert f"Page {i + 1}" in doc["text"]
        assert doc["metadata"]["page"] == i


def test_load_pdf_with_special_characters(create_test_pdf):
    """Test loading PDF with special characters."""
    from src.document_processor.pdf_loader import PDFLoader

    content = """
    Testing special characters:
    Symbols: @#$%^&*()
    Accents: café, naïve, résumé
    Math: ∑, ∫, ∞, ≈, ≠
    Quotes: "double", 'single'
    """

    pdf_path = create_test_pdf(content, num_pages=1)

    loader = PDFLoader()
    documents = loader.load_pdf(str(pdf_path))

    assert len(documents) == 1
    # Basic check that content was loaded
    assert len(documents[0]["text"]) > 0


def test_validate_nonexistent_file():
    """Test that loading nonexistent file raises error."""
    from src.document_processor.pdf_loader import PDFLoader

    loader = PDFLoader()

    with pytest.raises(FileNotFoundError):
        loader.load_pdf("/nonexistent/file.pdf")


def test_validate_non_pdf_file(temp_dir):
    """Test that loading non-PDF file raises error."""
    from src.document_processor.pdf_loader import PDFLoader

    # Create a text file
    txt_path = temp_dir / "test.txt"
    txt_path.write_text("Not a PDF")

    loader = PDFLoader()

    with pytest.raises(ValueError, match="Not a PDF file"):
        loader.load_pdf(str(txt_path))


def test_validate_directory_path(temp_dir):
    """Test that passing directory raises error."""
    from src.document_processor.pdf_loader import PDFLoader

    loader = PDFLoader()

    with pytest.raises(ValueError, match="Not a file"):
        loader.load_pdf(str(temp_dir))


# ==================== LOADER TYPE TESTS ====================


@pytest.mark.parametrize("loader_type", ["pymupdf"])
def test_different_loader_types(create_test_pdf, loader_type):
    """Test that different loader types work."""
    from src.document_processor.pdf_loader import PDFLoader

    pdf_path = create_test_pdf("Test content for loaders", num_pages=1)

    loader = PDFLoader(loader_type=loader_type)
    documents = loader.load_pdf(str(pdf_path))

    assert len(documents) >= 1
    assert len(documents[0]["text"]) > 0


def test_auto_loader_fallback(create_test_pdf):
    """Test that auto loader tries fallback methods."""
    from src.document_processor.pdf_loader import PDFLoader

    pdf_path = create_test_pdf("Fallback test", num_pages=1)

    # Auto should work even if one loader fails
    loader = PDFLoader(loader_type="auto")
    documents = loader.load_pdf(str(pdf_path))

    assert len(documents) >= 1


# ==================== METADATA TESTS ====================


def test_metadata_preservation(create_test_pdf):
    """Test that metadata is properly preserved."""
    from src.document_processor.pdf_loader import PDFLoader

    pdf_path = create_test_pdf("Metadata test", num_pages=2)

    loader = PDFLoader()
    documents = loader.load_pdf(str(pdf_path))

    # Check metadata exists
    for doc in documents:
        assert "metadata" in doc
        assert "source" in doc["metadata"]
        assert "page" in doc["metadata"]
        assert doc["metadata"]["source"] == str(pdf_path)


def test_page_numbering(create_test_pdf):
    """Test that pages are numbered correctly."""
    from src.document_processor.pdf_loader import PDFLoader

    pdf_path = create_test_pdf("Page numbering", num_pages=5)

    loader = PDFLoader()
    documents = loader.load_pdf(str(pdf_path))

    # Check page numbers
    for i, doc in enumerate(documents):
        assert doc["metadata"]["page"] == i
        assert f"Page {i + 1}" in doc["text"]


# ==================== MULTIPLE FILES TESTS ====================


def test_load_multiple_pdfs(multiple_pdfs):
    """Test loading multiple PDF files."""
    from src.document_processor.pdf_loader import PDFLoader

    loader = PDFLoader()
    results = loader.load_multiple([str(p) for p in multiple_pdfs])

    # Should have 3 files
    assert len(results) == 3

    # Each file should have documents
    for filename, documents in results.items():
        assert len(documents) >= 1
        assert "text" in documents[0]


def test_load_multiple_with_failures(multiple_pdfs, temp_dir):
    """Test that load_multiple handles failures gracefully."""
    from src.document_processor.pdf_loader import PDFLoader

    # Add an invalid path
    pdf_paths = [str(p) for p in multiple_pdfs]
    pdf_paths.append(str(temp_dir / "nonexistent.pdf"))

    loader = PDFLoader()
    results = loader.load_multiple(pdf_paths)

    # Should still have successful loads
    assert len(results) >= 2
    # Should not include the failed file
    assert "nonexistent.pdf" not in results


# ==================== DOCUMENT PROCESSOR TESTS ====================


def test_document_processor_initialization(temp_dir):
    """Test DocumentProcessor initialization."""
    from src.document_processor.pdf_loader import DocumentProcessor

    processor = DocumentProcessor(data_dir=str(temp_dir))
    assert processor.data_dir == str(temp_dir)
    assert processor.loader is not None


def test_get_all_pdfs_empty_directory(temp_dir):
    """Test getting PDFs from empty directory."""
    from src.document_processor.pdf_loader import DocumentProcessor

    processor = DocumentProcessor(data_dir=str(temp_dir))
    pdfs = processor.get_all_pdfs()

    assert len(pdfs) == 0


def test_get_all_pdfs_with_files(temp_dir, create_test_pdf):
    """Test getting PDFs from directory with files."""
    from src.document_processor.pdf_loader import DocumentProcessor

    # Create PDFs in temp_dir
    pdf1 = create_test_pdf("Doc 1", num_pages=1, filename="doc1.pdf")
    pdf2 = create_test_pdf("Doc 2", num_pages=1, filename="doc2.pdf")

    # Move to temp_dir
    import shutil

    shutil.move(str(pdf1), str(temp_dir / "doc1.pdf"))
    shutil.move(str(pdf2), str(temp_dir / "doc2.pdf"))

    processor = DocumentProcessor(data_dir=str(temp_dir))
    pdfs = processor.get_all_pdfs()

    assert len(pdfs) == 2
    assert all(p.endswith(".pdf") for p in pdfs)


def test_process_all_documents(temp_dir, create_test_pdf):
    """Test processing all documents in directory."""
    from src.document_processor.pdf_loader import DocumentProcessor

    # Create and move PDFs
    pdf1 = create_test_pdf("Document 1", num_pages=1, filename="test1.pdf")
    pdf2 = create_test_pdf("Document 2", num_pages=2, filename="test2.pdf")

    import shutil

    shutil.move(str(pdf1), str(temp_dir / "test1.pdf"))
    shutil.move(str(pdf2), str(temp_dir / "test2.pdf"))

    processor = DocumentProcessor(data_dir=str(temp_dir))
    results = processor.process_all_documents()

    assert len(results) == 2
    assert "test1.pdf" in results
    assert "test2.pdf" in results


# ==================== EDGE CASES ====================


def test_empty_pdf_handling(create_test_pdf):
    """Test handling of PDF with minimal content."""
    from src.document_processor.pdf_loader import PDFLoader

    pdf_path = create_test_pdf("", num_pages=1)  # Empty content

    loader = PDFLoader()
    documents = loader.load_pdf(str(pdf_path))

    # Should still load, even if empty
    assert isinstance(documents, list)


def test_pdf_with_only_whitespace(create_test_pdf):
    """Test PDF with only whitespace."""
    from src.document_processor.pdf_loader import PDFLoader

    pdf_path = create_test_pdf("   \n\n   ", num_pages=1)

    loader = PDFLoader()
    documents = loader.load_pdf(str(pdf_path))

    assert isinstance(documents, list)


# ==================== PERFORMANCE TESTS ====================


def test_loading_performance(create_test_pdf, performance_timer):
    """Test that PDF loading completes in reasonable time."""
    from src.document_processor.pdf_loader import PDFLoader

    # Create larger PDF
    content = "Performance test content. " * 1000
    pdf_path = create_test_pdf(content, num_pages=5)

    loader = PDFLoader()

    with performance_timer() as timer:
        documents = loader.load_pdf(str(pdf_path))

    # Should complete in under 5 seconds
    assert timer.elapsed < 5.0
    assert len(documents) == 5


def test_batch_loading_performance(multiple_pdfs, performance_timer):
    """Test batch loading performance."""
    from src.document_processor.pdf_loader import PDFLoader

    loader = PDFLoader()

    with performance_timer() as timer:
        results = loader.load_multiple([str(p) for p in multiple_pdfs])

    # Should complete in reasonable time
    assert timer.elapsed < 10.0
    assert len(results) >= 2


# ==================== INTEGRATION TESTS ====================


def test_pdf_to_text_pipeline(create_test_pdf):
    """Test complete PDF loading pipeline."""
    from src.document_processor.pdf_loader import PDFLoader

    # Create realistic PDF
    content = """
    CHAPTER 1: INTRODUCTION

    This document demonstrates the complete PDF loading pipeline.
    It includes various elements like headers, paragraphs, and lists.

    Key Points:
    - Point one
    - Point two
    - Point three

    The content is structured to test real-world scenarios.
    """

    pdf_path = create_test_pdf(content, num_pages=2)

    loader = PDFLoader()
    documents = loader.load_pdf(str(pdf_path))

    # Verify complete pipeline
    assert len(documents) == 2
    assert all("INTRODUCTION" in doc["text"] or "Key Points" in doc["text"] for doc in documents)
    assert all("metadata" in doc for doc in documents)
    assert all(doc["metadata"]["source"] == str(pdf_path) for doc in documents)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
