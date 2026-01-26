import torch
from sentence_transformers import SentenceTransformer
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from typing import List, Dict, Any, Optional, Union
import numpy as np
import time
import logging
from enum import Enum
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModelType(Enum):
    """
    Embedding model types
    """
    MULTILINGUAL_MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ADA_002 = "text-embedding-ada-002"
    E5_SMALL = "intfloat/multilingual-e5-small"
    E5_LARGE = "intfloat/multilingual-e5-large"

class EmbeddingManager:
    """
    Manager class for embedding models
    """

    def __init__(self,model_type: Union[EmbeddingModelType, str] = EmbeddingModelType.MULTILINGUAL_MINILM, device: Optional[str] = None, cache_dir: str ="models/cache"):
        """
        Args:
            model_type: model type
            device: device to use('cuda', 'cpu')
            cache_dir: cache directory
        """
        self.model_type = model_type if isinstance(model_type, EmbeddingModelType) else EmbeddingModelType(model_type)
        self.device = self._determine_device(device)
        self.model = None
        self.model_name = self.model_type.value
        self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)

    def _determine_device(self, device: Optional[str]) -> str:
        """
        Automatically determine device
        """
        if device:
            return device

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def load_model(self):
        """
        Embeddings model loading
        """

        if self.model is not None:
            return self.model

        logger.info(f"Loading {self.model_name} model")
        logger.info(f"Device: {self.device}")
        logger.info(f"Cache dir: {self.cache_dir}")

        try:
            start_time = time.time()

            if self.model_type == EmbeddingModelType.ADA_002:
                logger.warning(f"OpenAI ADA-002 requires API key")
                self.model_type = EmbeddingModelType.MULTILINGUAL_MINILM
                self.model_name = self.model_type.name

            self.model = SentenceTransformer(
                model_name_or_path=self.model_name,
                device=self.device,
                cache_folder=self.cache_dir,
            )

            test_embedding = self.model.encode(['test'])
            embedding_dim = len(test_embedding[0])

            load_time = time.time() - start_time

            logger.info(f"Model loaded in {load_time:.2f} seconds")
            logger.info(f"Embeddings dimension: {embedding_dim}")
            logger.info(f"Max sequence length: {self.model.max_seq_length}")

            return self.model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def encode(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = True, normalize_embeddings: bool = True) -> np.ndarray:
        """
        Creates embeddings for a list of texts

        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            show_progress_bar: Whether to show a progress bar
            normalize_embeddings: Whether to normalize embeddings to a length of 1

        Returns:
        Numpy array of embeddings (n_texts x embedding_dim)
        """

        if self.model is None:
            self.load_model()

        if not texts:
            return np.array([])

        logger.info(f"Encoding {len(texts)} texts")

        start_time = time.time()

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
            )

            encode_time = time.time() - start_time

            logger.info(f"Embeddings encoded in {encode_time:.2f} seconds")
            logger.info(f"{len(texts)/encode_time:.2f} texts/second")
            logger.info(f" Shape of embeddings: {embeddings.shape}")

            return embeddings

        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise

    def encode_single(self, text: str) -> np.ndarray:
        """
        Encodes a single text
        """
        return self.model.encode(text)

    def get_embedding_dimension(self) -> int:
        """
        returns the embedding dimension
        """
        if self.model is None:
            self.load_model()

        test_embedding = self.encode_single("test")
        return len(test_embedding)

    def compare_embeddings(self,
                           embedding1: np.ndarray,
                           embedding2: np.ndarray,
                           metric: str = "cosine") -> np.ndarray:
        """
        Compares two embeddings

        Args:
            embedding1: embedding
            embedding2: embedding
            metric: metric to use("cosine", "euclidean", "dot)

        Returns:
            Similarity matrix
        """
        if metric == "cosine":
            norms1 = np.linalg.norm(embedding1, axis=1, keepdims=True)
            norms2 = np.linalg.norm(embedding2, axis=1, keepdims=True)
            normalized1 = embedding1 / norms1
            normalized2 = embedding2 / norms2
            similarity = np.dot(normalized1, normalized2.T)

        elif metric == "dot":
            similarity = np.dot(embedding1, embedding2.T)

        elif metric == "euclidean":
            n1, n2 = len(embedding1), len(embedding2)
            similarity = np.zeros([n1, n2])

        else:
            raise ValueError(f"Unknown metric {metric}")

        return similarity

    def chroma_embedding_function(self):
        """
        Returns a class instance compatible with ChromaDB
        """
        return ChromaEmbeddingAdapter(self)

class ChromaEmbeddingAdapter(EmbeddingFunction):
    def __init__(self, manager: 'EmbeddingManager'):
        self.manager = manager

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.manager.encode(
            list(input),
            show_progress_bar=False
        )
        return embeddings.tolist()

class EmbeddingBenchmark:

    def benchmarkmodel(self):
        pass

    def quality_evaluation(self):
        pass
