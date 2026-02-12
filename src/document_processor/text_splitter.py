import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter, CharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning(
        "LangChain not available. Install with: "
        "pip install langchain tiktoken"
    )

try:
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai import OpenAIEmbeddings
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False


@dataclass
class Chunk:
    """
    Chunk dataclass compatible with LangChain.
    """
    text: str
    metadata: Dict[str, Any]
    chunk_id: str

    def __post_init__(self):
        """Validate chunk."""
        if not self.text or not self.text.strip():
            raise ValueError("Chunk text cannot be empty")

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id
        }


class TextSplitter:
    """
    Modern text splitter using LangChain.

    Strategies:
    - "recursive" (default): Smart hierarchical splitting
    - "token": Token-aware splitting (best for embeddings)
    - "semantic": AI-powered semantic splitting (experimental)
    - "character": Simple character-based splitting
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: str = "recursive",
        length_function: str = "len",
        separators: Optional[List[str]] = None,
        min_chunk_size: int = 50
    ):
        """
        Initialize text splitter.

        Args:
            chunk_size: Target chunk size (characters or tokens)
            chunk_overlap: Overlap between chunks
            strategy: "recursive", "token", "semantic", or "character"
            length_function: "len" or "tiktoken" (for token counting)
            separators: Custom separators (only for recursive)
            min_chunk_size: Minimum chunk size to keep
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain required. Install: pip install langchain tiktoken"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.min_chunk_size = min_chunk_size

        self.splitter = self._create_splitter(
            strategy, chunk_size, chunk_overlap, length_function, separators
        )

        logger.info(
            f"TextSplitter initialized: strategy={strategy}, "
            f"size={chunk_size}, overlap={chunk_overlap}"
        )

    def _create_splitter(
        self,
        strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        length_function: str,
        separators: Optional[List[str]]
    ):
        """Create appropriate LangChain splitter."""

        if strategy == "recursive":
            # Hierarchical splitting: \n\n -> \n -> . -> space

            if separators is None:
                # Default separators (optimized)
                separators = [
                    "\n\n",  # Paragraphs
                    "\n",    # Lines
                    ". ",    # Sentences
                    "? ",
                    "! ",
                    "; ",
                    ": ",
                    ", ",
                    " ",     # Words
                    ""       # Characters (fallback)
                ]

            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len if length_function == "len" else self._token_length,
                separators=separators,
                keep_separator=True
            )

        elif strategy == "token":
            # Token-aware splitting (best for embeddings)
            # Uses tiktoken to count tokens accurately

            return TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        elif strategy == "semantic":
            # AI-powered semantic splitting
            # Splits based on meaning, not just characters

            if not SEMANTIC_AVAILABLE:
                raise ImportError(
                    "Semantic chunking requires: "
                    "pip install langchain-experimental langchain-openai"
                )

            logger.warning(
                "Semantic chunking requires OpenAI API key. "
                "Set OPENAI_API_KEY environment variable."
            )

            return SemanticChunker(
                OpenAIEmbeddings(),
                breakpoint_threshold_type="percentile"
            )

        elif strategy == "character":
            # Simple character-based splitting

            return CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n\n"
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _token_length(self, text: str) -> int:
        """Count tokens using tiktoken."""
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using len()")
            return len(text)

    def split_text(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Split text into chunks.

        Args:
            text: Text to split
            metadata: Metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            logger.warning("Empty text provided")
            return []

        if metadata is None:
            metadata = {}

        # Split using LangChain
        try:
            langchain_docs = self.splitter.split_text(text)
        except Exception as e:
            logger.error(f"Splitting failed: {e}")
            raise

        # Convert to our Chunk format
        chunks = []
        for i, doc_text in enumerate(langchain_docs):
            # Skip too small chunks
            if len(doc_text.strip()) < self.min_chunk_size:
                logger.debug(f"Skipping small chunk {i} ({len(doc_text)} chars)")
                continue

            chunk = Chunk(
                text=doc_text.strip(),
                metadata={
                    **metadata,
                    "chunk_index": i,
                    "strategy": self.strategy,
                    "chunk_size_config": self.chunk_size,
                    "overlap_config": self.chunk_overlap
                },
                chunk_id=f"{metadata.get('doc_id', 'doc')}_{i}"
            )

            chunks.append(chunk)

        logger.info(
            f"Split into {len(chunks)} chunks "
            f"(filtered from {len(langchain_docs)})"
        )

        return chunks

    def split_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Chunk]:
        """
        Split multiple documents.

        Args:
            documents: List of dicts with 'text' and 'metadata'

        Returns:
            List of chunks from all documents
        """
        all_chunks = []

        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})

            chunks = self.split_text(text, metadata)
            all_chunks.extend(chunks)

        logger.info(f"Split {len(documents)} docs into {len(all_chunks)} chunks")

        return all_chunks


class ChunkAnalyzer:
    """
    Analyze chunk quality.
    """

    @staticmethod
    def analyze_chunks(chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Analyze chunk statistics.

        Args:
            chunks: List of chunks

        Returns:
            Statistics dictionary
        """
        if not chunks:
            return {"error": "No chunks to analyze"}

        lengths = [len(chunk.text) for chunk in chunks]
        word_counts = [chunk.word_count for chunk in chunks]

        import numpy as np

        stats = {
            "total_chunks": len(chunks),
            "char_stats": {
                "mean": float(np.mean(lengths)),
                "median": float(np.median(lengths)),
                "std": float(np.std(lengths)),
                "min": int(np.min(lengths)),
                "max": int(np.max(lengths))
            },
            "word_stats": {
                "mean": float(np.mean(word_counts)),
                "median": float(np.median(word_counts)),
                "std": float(np.std(word_counts)),
                "min": int(np.min(word_counts)),
                "max": int(np.max(word_counts))
            },
            "total_chars": int(sum(lengths)),
            "total_words": int(sum(word_counts))
        }

        return stats