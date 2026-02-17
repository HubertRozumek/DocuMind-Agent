"""
Centralized configuration management for DocuMind-Agent.

This module loads configuration from environment variables and provides
a single source of truth for all application settings.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent if Path(__file__).parent.name == "src" else Path(__file__).parent


@dataclass
class OllamaConfig:
    """Ollama LLM configuration."""

    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    grader_model: str = os.getenv("OLLAMA_GRADER_MODEL", "phi3:mini")
    generator_model: str = os.getenv("OLLAMA_GENERATOR_MODEL", "llama3.1:8b")
    rewriter_model: str = os.getenv("OLLAMA_REWRITER_MODEL", "mistral:7b")


@dataclass
class VectorStoreConfig:
    """Vector store configuration."""

    persist_dir: Path = field(default_factory=lambda: Path(os.getenv("CHROMA_PERSIST_DIR", "data/vector_store/chroma")))
    default_collection: str = os.getenv("CHROMA_DEFAULT_COLLECTION", "documents")
    host: str = os.getenv("CHROMA_HOST", "localhost")
    port: int = int(os.getenv("CHROMA_PORT", "8000"))

    def __post_init__(self):
        """Ensure persist directory exists."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""

    model_type: str = os.getenv("EMBEDDING_MODEL_TYPE", "MPNET")
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("EMBEDDING_CACHE_DIR", "models/cache")))
    device: Optional[str] = os.getenv("EMBEDDING_DEVICE", None)

    def __post_init__(self):
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class AgentConfig:
    """RAG Agent configuration."""

    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "3"))
    search_threshold: float = float(os.getenv("SEARCH_THRESHOLD", "0.7"))
    grader_confidence_threshold: float = float(os.getenv("GRADER_CONFIDENCE_THRESHOLD", "0.6"))
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    enable_tools: bool = os.getenv("ENABLE_TOOLS", "true").lower() == "true"


@dataclass
class DocumentProcessingConfig:
    """Document processing configuration."""

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "400"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    chunking_strategy: str = os.getenv("CHUNKING_STRATEGY", "recursive")
    pdf_loader_type: str = os.getenv("PDF_LOADER_TYPE", "auto")
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))


@dataclass
class StreamlitConfig:
    """Streamlit app configuration."""

    title: str = os.getenv("APP_TITLE", "DocuMind-Agent")
    theme: str = os.getenv("APP_THEME", "light")
    layout: str = os.getenv("APP_LAYOUT", "wide")


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = os.getenv("LOG_LEVEL", "INFO")
    file: Optional[str] = os.getenv("LOG_FILE", None)
    format: str = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    @property
    def level_int(self) -> int:
        """Convert log level string to logging constant."""
        return getattr(logging, self.level.upper(), logging.INFO)


@dataclass
class PerformanceConfig:
    """Performance and caching configuration."""

    enable_query_cache: bool = os.getenv("ENABLE_QUERY_CACHE", "true").lower() == "true"
    cache_size: int = int(os.getenv("CACHE_SIZE", "100"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "100"))


@dataclass
class SecurityConfig:
    """Security configuration."""

    api_key: Optional[str] = os.getenv("API_KEY", None)
    enable_auth: bool = os.getenv("ENABLE_AUTH", "false").lower() == "true"
    allowed_origins: list = field(default_factory=lambda: os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(","))


@dataclass
class TestingConfig:
    """Testing configuration."""

    collection_name: str = os.getenv("TEST_COLLECTION_NAME", "test_collection")
    persist_dir: Path = field(default_factory=lambda: Path(os.getenv("TEST_PERSIST_DIR", "tests/data/vector_store/chroma")))

    def __post_init__(self):
        """Ensure test persist directory exists."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Main application configuration."""

    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    document_processing: DocumentProcessingConfig = field(default_factory=DocumentProcessingConfig)
    streamlit: StreamlitConfig = field(default_factory=StreamlitConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "debug": self.debug,
            "ollama": {
                "base_url": self.ollama.base_url,
                "grader_model": self.ollama.grader_model,
                "generator_model": self.ollama.generator_model,
                "rewriter_model": self.ollama.rewriter_model,
            },
            "vector_store": {
                "persist_dir": str(self.vector_store.persist_dir),
                "default_collection": self.vector_store.default_collection,
            },
            "agent": {
                "max_iterations": self.agent.max_iterations,
                "search_threshold": self.agent.search_threshold,
                "enable_tools": self.agent.enable_tools,
            },
        }

    def validate(self) -> bool:
        """Validate configuration."""
        errors = []

        # Validate numeric ranges
        if not 1 <= self.agent.max_iterations <= 10:
            errors.append("max_iterations must be between 1 and 10")

        if not 0.0 <= self.agent.search_threshold <= 1.0:
            errors.append("search_threshold must be between 0.0 and 1.0")

        if not 0.0 <= self.agent.grader_confidence_threshold <= 1.0:
            errors.append("grader_confidence_threshold must be between 0.0 and 1.0")

        if self.document_processing.chunk_size < 100:
            errors.append("chunk_size must be at least 100")

        if self.document_processing.chunk_overlap >= self.document_processing.chunk_size:
            errors.append("chunk_overlap must be less than chunk_size")

        if errors:
            raise ValueError(f"Configuration validation failed: {', '.join(errors)}")

        return True


# Global configuration instance
config = Config()

# Validate on import
try:
    config.validate()
except ValueError as e:
    logging.warning(f"Configuration validation warning: {e}")


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config():
    """Reload configuration from environment variables."""
    global config
    load_dotenv(override=True)
    config = Config()
    config.validate()
    return config


# Convenience functions for common configuration access
def get_vector_store_config() -> Dict[str, Any]:
    """Get vector store configuration as dictionary."""
    return {
        "collection_name": config.vector_store.default_collection,
        "persist_directory": str(config.vector_store.persist_dir),
    }


def get_grader_config() -> Dict[str, Any]:
    """Get grader configuration as dictionary."""
    return {
        "grader_type": "robust",
        "confidence_threshold": config.agent.grader_confidence_threshold,
        "model_name": config.ollama.grader_model,
    }


def get_generator_config() -> Dict[str, Any]:
    """Get generator configuration as dictionary."""
    return {
        "model_name": config.ollama.generator_model,
        "temperature": 0.1,
    }


if __name__ == "__main__":
    """Print current configuration."""
    import json

    print(json.dumps(config.to_dict(), indent=2))
