from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import warnings

warnings.filterwarnings("ignore")

@dataclass
class Chunk:
    """
    Class representing a chunk of text.
    """
    text: str
    metadata: Dict[str, Any]
    chunk_id: str
    embeddings: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id,
            "embeddings": self.embeddings,
        }

class TextSplitter:
    """
    Main class for splitting text into chunks with overlap.
    """

    def __init__(self,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 strategy: str = "recursive",
                 language: str = "en"):
        """
        Args:
        chunk_size (int, optional): The size of the chunk to split. Defaults to 500.
        chunk_overlap (int, optional): The overlap between chunks. Defaults to 50.
        strategy: Splitting strategy ("recursive", "sentences", "paragraphs", "semantic"). Defaults to "recursive". )
        language: text language. Defaults to "en".
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.language = language

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.embedding_model = None
        self._initialize_nltk()

    def set_embedding_model(self, model):
        self.embedding_model = model

    def _initialize_nltk(self):
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

    def split_text(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """
        Split text into chunks according to the strategy.
        """

        if metadata is None:
            metadata = {}

        text = self._preprocess_text(text)

        if self.strategy == "recursive":
            return self._recursive_split(text, metadata)
        if self.strategy == "sentences":
            return self._split_by_sentences(text, metadata)
        if self.strategy == "semantic":
            return self._semantic_split(text, metadata)
        if self.strategy == "fixed":
            return self._fixed_size_split(text, metadata)

    def _preprocess_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _recursive_split(self, text: str, metadata: Dict) -> List[Chunk]:
        separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", " "]
        chunks = []
        chunk_id = 0

        def split_recursively(t: str, seps: List[str]):
            nonlocal chunk_id

            if len(t) <= self.chunk_size:
                chunks.append(self._make_chunk(t, metadata, chunk_id))
                chunk_id += 1
                return

            for sep in seps:
                if sep in t:
                    parts = t.split(sep)
                    current = ""

                    for part in parts:
                        candidate = current + part + sep
                        if len(candidate) <= self.chunk_size:
                            current = candidate
                        else:
                            chunks.append(self._make_chunk(current, metadata, chunk_id))
                            chunk_id += 1
                            current = current[-self.chunk_overlap:] + part + sep

                    if current.strip():
                        chunks.append(self._make_chunk(current, metadata, chunk_id))
                        chunk_id += 1
                    return
            # fallback
            for i in range(0, len(t), self.chunk_size - self.chunk_overlap):
                chunks.append(
                    self._make_chunk(t[i:i + self.chunk_size], metadata, chunk_id)
                )
                chunk_id += 1

        split_recursively(text, separators)
        return chunks

    def _split_by_sentences(self, text: str, metadata: Dict) -> List[Chunk]:
        sentences = sent_tokenize(text)
        chunks = []
        buffer = []
        length = 0
        chunk_id = 0

        for s in sentences:
            if length + len(s) <= self.chunk_size:
                buffer.append(s)
                length += len(s)
            else:
                chunks.append(self._make_chunk(" ".join(buffer), metadata, chunk_id))
                chunk_id += 1
                buffer = buffer[-1:] if self.chunk_overlap > 0 else []
                buffer.append(s)
                length = len(" ".join(buffer))

        if buffer:
            chunks.append(self._make_chunk(" ".join(buffer), metadata, chunk_id))

        return chunks

    def _semantic_split(self, text: str, metadata: Dict) -> List[Chunk]:
        if not self.embedding_model:
            return self._split_by_sentences(text, metadata)

        sentences = sent_tokenize(text)
        embeddings = self.embedding_model.encode(sentences)

        chunks = []
        current = []
        current_embs = []
        chunk_id = 0

        for sentence, emb in zip(sentences, embeddings):
            current.append(sentence)
            current_embs.append(emb)

            if len(" ".join(current)) > self.chunk_size:
                chunks.append(
                    self._make_chunk(" ".join(current[:-1]), metadata, chunk_id)
                )
                chunk_id += 1
                current = current[-1:]
                current_embs = current_embs[-1:]

        if current:
            chunks.append(self._make_chunk(" ".join(current), metadata, chunk_id))

        return chunks

    def _fixed_size_split(self, text: str, metadata: Dict) -> List[Chunk]:
        chunks = []
        chunk_id = 0

        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(
                self._make_chunk(text[i:i + self.chunk_size], metadata, chunk_id)
            )
            chunk_id += 1

        return chunks

    def _make_chunk(self, text: str, metadata: Dict, chunk_id: int) -> Chunk:
        text = text.strip()

        if len(text) > self.chunk_size:
            text = text[:self.chunk_size]

        return Chunk(
            text=text.strip(),
            metadata={**metadata, "strategy": self.strategy},
            chunk_id=f"{metadata.get('doc_id', 'doc')}_{chunk_id}")

class ChunkAnalyzer:
    """
    Chunking quality analysis
    """

    @staticmethod
    def analyze_chunks(chunks: List[Chunk]) -> Dict[str, Any]:
        if not chunks:
            return {}

        texts = [chunk.text for chunk in chunks]
        lengths = [len(text) for text in texts]

        stats = {
            "total_chunks": len(chunks),
            "avg_length": np.mean(lengths),
            "min_length": np.min(lengths),
            "max_length": np.max(lengths),
            "std": np.std(lengths),
            "total_characters": sum(lengths),
            "chunk_size_distribution": {
                "small": len([l for l in lengths if l < 100]),
                "medium": len([l for l in lengths if 100 <= l <= 500]),
                "large": len([l for l in lengths if l > 500])
            }
        }

        overlaps = []
        for i in range(len(chunks) - 1):
            chunk1 = chunks[i].text
            chunk2 = chunks[i + 1].text

            for overlap_size in range(min(len(chunk1), len(chunk2), 100), 0, -1):
                if chunk1.endswith(chunk2[:overlap_size]):
                    overlaps.append(overlap_size)
                    break

        if overlaps:
            stats["avg_overlap"] = np.mean(overlaps)
            stats["min_overlap"] = np.min(overlaps)
            stats["max_overlap"] = np.max(overlaps)
        else:
            stats["avg_overlap"] = 0
            stats["min_overlap"] = 0
            stats["max_overlap"] = 0

        return stats

    @staticmethod
    def find_optimal_chunk_size(texts: List[str],
                                chunk_sizes: List[int] = [100, 250, 500, 1000],
                                overlap_ratios: List[float] = [0.0, 0.1, 0.2]) -> Dict:
        """
        """
        results = []

        for chunk_size in chunk_sizes:
            for overlap_ratio in overlap_ratios:
                chunk_overlap = int(chunk_size * overlap_ratio)

                splitter = TextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    strategy="recursive"
                )

                all_chunks = []
                for text in texts:
                    chunks = splitter.split_text(text, {})
                    all_chunks.extend(chunks)

                stats = ChunkAnalyzer.analyze_chunks(all_chunks)

                score = 0
                if stats["std_length"] < chunk_size * 0.5:
                    score += 1
                if stats["avg_overlap"] > 0:
                    score += 1

                results.append({
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "overlap_ratio": overlap_ratio,
                    "stats": stats,
                    "score": score
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

