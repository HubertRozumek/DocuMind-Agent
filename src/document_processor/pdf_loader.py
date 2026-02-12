import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

try:
    from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader, PyMuPDFLoader
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning(
        "LangChain not available. Install with: "
        "pip install langchain langchain-community"
    )


class PDFLoader:
    """
    Modern PDF loader using LangChain.

    Features:
    - PyMuPDF as primary loader (best quality)
    - Automatic fallback to other loaders
    - Preserves text formatting and spaces
    - Returns LangChain Document objects
    - Much better than pdfplumber for complex PDFs
    """

    def __init__(
        self,
        loader_type: str = "auto",
        extract_images: bool = False
    ):
        """
        Initialize PDF loader.

        Args:
            loader_type: "pymupdf", "unstructured", "pypdf", or "auto" (default)
            extract_images: Whether to extract images (only for unstructured)
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required. Install with: "
                "pip install langchain langchain-community"
            )

        self.loader_type = loader_type
        self.extract_images = extract_images

        logger.info(f"PDFLoader initialized (type={loader_type})")

    def load_pdf(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load PDF and return documents.

        Args:
            filepath: Path to PDF file

        Returns:
            List of document dictionaries with:
                - page_content: Text content
                - metadata: Page number, source, etc.

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a PDF
        """
        # Validate
        self._validate_file(filepath)

        # Load based on type
        if self.loader_type == "auto":
            documents = self._load_with_fallback(filepath)
        elif self.loader_type == "pymupdf":
            documents = self._load_with_pymupdf(filepath)
        elif self.loader_type == "unstructured":
            documents = self._load_with_unstructured(filepath)
        elif self.loader_type == "pypdf":
            documents = self._load_with_pypdf(filepath)
        else:
            raise ValueError(f"Unknown loader type: {self.loader_type}")

        # Convert to our format
        result = []
        for doc in documents:
            result.append({
                "text": doc.page_content,
                "metadata": doc.metadata
            })

        logger.info(
            f"Loaded {os.path.basename(filepath)}: "
            f"{len(result)} pages/chunks"
        )

        return result

    def _validate_file(self, filepath: str):
        """Validate PDF file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        if not os.path.isfile(filepath):
            raise ValueError(f"Not a file: {filepath}")

        if not filepath.lower().endswith('.pdf'):
            raise ValueError(f"Not a PDF file: {filepath}")

        # Check size (max 500MB)
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        if size_mb > 500:
            logger.warning(f"Large PDF: {size_mb:.1f}MB")

    def _load_with_pymupdf(self, filepath: str) -> List:
        """
        Load with PyMuPDF (best quality, fastest).

        PyMuPDF is the best choice for most PDFs:
        - Excellent text extraction
        - Preserves spaces and formatting
        - Fast (C library)
        - Handles complex layouts
        """
        try:
            loader = PyMuPDFLoader(filepath)
            documents = loader.load()
            logger.info(f"Loaded with PyMuPDF: {len(documents)} pages")
            return documents
        except Exception as e:
            logger.error(f"PyMuPDF failed: {e}")
            raise

    def _load_with_unstructured(self, filepath: str) -> List:
        """
        Load with Unstructured (most intelligent).

        Unstructured uses AI to understand document structure:
        - Detects headers, paragraphs, tables
        - Better for complex layouts
        - Can extract images
        - Slower than PyMuPDF
        """
        try:
            loader = UnstructuredPDFLoader(
                filepath,
                mode="elements" if self.extract_images else "single"
            )
            documents = loader.load()
            logger.info(f"Loaded with Unstructured: {len(documents)} elements")
            return documents
        except Exception as e:
            logger.error(f"Unstructured failed: {e}")
            raise

    def _load_with_pypdf(self, filepath: str) -> List:
        """
        Load with PyPDF (simple, reliable fallback).

        PyPDF is a simple, pure-Python loader:
        - Lightweight
        - Reliable for simple PDFs
        - Good fallback option
        """
        try:
            loader = PyPDFLoader(filepath)
            documents = loader.load()
            logger.info(f"Loaded with PyPDF: {len(documents)} pages")
            return documents
        except Exception as e:
            logger.error(f"PyPDF failed: {e}")
            raise

    def _load_with_fallback(self, filepath: str) -> List:
        """
        Try loaders in order until one succeeds.

        Order:
        1. PyMuPDF (best quality)
        2. Unstructured (most intelligent)
        3. PyPDF (simple fallback)
        """
        errors = []

        # Try PyMuPDF first
        try:
            return self._load_with_pymupdf(filepath)
        except Exception as e:
            errors.append(f"PyMuPDF: {e}")
            logger.warning(f"PyMuPDF failed, trying Unstructured...")

        # Try Unstructured
        try:
            return self._load_with_unstructured(filepath)
        except Exception as e:
            errors.append(f"Unstructured: {e}")
            logger.warning(f"Unstructured failed, trying PyPDF...")

        # Try PyPDF as last resort
        try:
            return self._load_with_pypdf(filepath)
        except Exception as e:
            errors.append(f"PyPDF: {e}")

        # All failed
        error_msg = "All loaders failed:\n" + "\n".join(errors)
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    def load_multiple(self, filepaths: List[str]) -> Dict[str, List[Dict]]:
        """
        Load multiple PDF files.

        Args:
            filepaths: List of PDF file paths

        Returns:
            Dictionary mapping filenames to documents
        """
        results = {}

        for filepath in filepaths:
            try:
                filename = os.path.basename(filepath)
                documents = self.load_pdf(filepath)
                results[filename] = documents
                logger.info(f"✓ {filename}: {len(documents)} pages")
            except Exception as e:
                logger.error(f"✗ {filepath}: {e}")

        return results


class DocumentProcessor:
    """
    Batch document processor for directories.
    """

    def __init__(
        self,
        data_dir: str = "data/raw_documents",
        loader_type: str = "auto"
    ):
        """
        Initialize processor.

        Args:
            data_dir: Directory containing PDFs
            loader_type: Loader type to use
        """
        self.data_dir = data_dir
        self.loader = PDFLoader(loader_type=loader_type)

        logger.info(f"DocumentProcessor initialized (dir={data_dir})")

    def get_all_pdfs(self) -> List[str]:
        """Get all PDF files in directory."""
        if not os.path.exists(self.data_dir):
            logger.warning(f"Directory not found: {self.data_dir}")
            return []

        pdf_files = []
        for file in os.listdir(self.data_dir):
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(self.data_dir, file))

        return sorted(pdf_files)

    def process_all_documents(self) -> Dict[str, List[Dict]]:
        """
        Process all PDFs in directory.

        Returns:
            Dictionary of processed documents
        """
        pdf_files = self.get_all_pdfs()

        if not pdf_files:
            logger.info(f"No PDFs found in {self.data_dir}")
            return {}

        logger.info(f"Found {len(pdf_files)} PDFs")

        results = self.loader.load_multiple(pdf_files)

        # Statistics
        total_pages = sum(len(docs) for docs in results.values())
        logger.info(f"Processed: {len(results)} files, {total_pages} pages")

        return results