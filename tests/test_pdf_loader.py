from pathlib import Path
import tempfile
import shutil

from src.document_processor.pdf_loader import PDFLoader, DocumentProcessor


def test_pdf_loader():
    test_pdf_dir = Path("../src/data/test_doc")
    pdf_files = list(test_pdf_dir.glob("*.pdf"))

    assert pdf_files, "No test PDF files found"

    loader = PDFLoader()

    for pdf in pdf_files:
        result = loader.load_pdf(str(pdf))

        assert "text" in result
        assert "pages" in result
        assert "metadata" in result
        assert "page_count" in result

        assert isinstance(result["text"], str)
        assert isinstance(result["pages"], list)
        assert result["page_count"] == len(result["pages"])

        print(f"{pdf.name}: {result['page_count']} pages, {len(result['text'])} chars")


def test_document_processor():
    # create temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # copy test PDFs into temp dir
        source_dir = Path("../src/data/test_doc")
        pdf_files = list(source_dir.glob("*.pdf"))
        assert pdf_files, "No test PDFs to copy"

        for pdf in pdf_files:
            shutil.copy(pdf, tmp_path / pdf.name)

        processor = DocumentProcessor(data_dir=str(tmp_path))
        documents = processor.process_all_documents()

        assert documents
        assert len(documents) == len(pdf_files)

        for name, doc in documents.items():
            assert "text" in doc
            assert "pages" in doc
            assert doc["page_count"] > 0

def run_test():
    print("Running tests")
    try:
        test_pdf_loader()
        test_document_processor()
        print("All tests passed")

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()