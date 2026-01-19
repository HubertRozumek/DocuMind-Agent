import os
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from datetime import datetime
import json
import hashlib
import numpy as np
import torch

from src.document_processor.pdf_loader import DocumentProcessor as dp, PDFLoader
from src.document_processor.text_splitter import TextSplitter, Chunk, ChunkAnalyzer
from src.vector_store.embeddings_manager import EmbeddingManager, EmbeddingModelType
from src.vector_store.chroma_db import ChromaDBVectorStore
from src.document_processor.data_persister import DataPersister

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentPipeline:
    """
    The main pipeline class that integrates all components
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.stats = {}
        self.timings = {}

        self._initialize_components()

    def _default_config(self) -> Dict[str, Any]:
        return {
            "pipeline": {
                "name": "DocuMind Pipeline",
                "version": "1.0.0",
                "description": "End-to-end document processing pipeline"
            },
            "processing": {
                "chunk_size": 500,
                "chunk_overlap": 50,
                "chunking_strategy": "recursive",
            },
            "embeddings": {
                "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "device": "cpu",
                "normalize": True,
                "batch_size": 32
            },
            "vector_stores": {
                "chromadb": {
                    "enabled": True,
                    "collection_name": "documents",
                    "persist_directory": "data/vector_store/chroma"
                },
            },
            "output": {
                "save_intermediate": True,
                "intermediate_dir": "data/processed/pipeline_runs",
                "export_formats": ["json", "csv"]
            },
            "logging": {
                "level": "INFO",
                "save_logs": True
            }
        }

    def _initialize_components(self):
        """
        Initialize all pipeline components
        """
        logger.info("Initializing components")

        self.pdf_loader = PDFLoader()

        processing_config = self.config["processing"]
        self.text_splitter = TextSplitter(
            chunk_size=processing_config["chunk_size"],
            chunk_overlap=processing_config["chunk_overlap"],
            strategy=processing_config["chunking_strategy"],
        )

        embeddings_config = self.config["embeddings"]
        self.embeddings_manager = EmbeddingManager(
            model_type=EmbeddingModelType(embeddings_config["model"]),
            device=embeddings_config["device"],
        )

        self.vector_store = {}

        vs_config = self.config["vector_stores"]

        if vs_config["chromadb"]["enabled"]:
            try:
                self.vector_store["chromadb"] = ChromaDBVectorStore(
                    collection_name=vs_config["chromadb"]["collection_name"],
                    persist_directory=vs_config["chromadb"]["persist_directory"],
                    reset_on_start=False
                )
                logger.info("Chromadb initialized")
            except Exception as e:
                logger.error(f"ChromaDB initialized error: {e}")

        output_config = self.config["output"]
        self.data_persister = DataPersister(
            output_dir=output_config["intermediate_dir"],
        )

        logger.info("All components initialized")

    def _generate_run_id(self) -> str:
        """
        Generate unique id for pipeline run
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        random_str = hashlib.md5(timestamp.encode()).hexdigest()[:8]
        return f"run_{timestamp}_{random_str}"

    def process_documents(self,
                          input_paths: Union[str, List[str]],
                          run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main method that processes documents

        Args:
            input_paths: path to pdf file or list of paths
            run_id: run id (if None, will be generated)

        Returns:
            Dictionary with process results
        """

        if run_id is None:
            run_id = self._generate_run_id()

        self.stats = {
            "run_id": run_id,
            "start_time": datetime.now().isoformat(),
            "input_paths": input_paths if isinstance(input_paths, list) else [input_paths],
            "stages": {}
        }

        logger.info(f"Starting pipeline run {run_id}")
        logger.info(f"Input paths: {input_paths}")

        try:
            documents = self._load_documents(input_paths)
            if not documents:
                raise ValueError("Could not load any documents")

            chunks = self._chunk_documents(documents)
            if not chunks:
                raise ValueError("Could not generate chunks")

            chunk_embeddings = self._create_embeddings(chunks)

            vector_store_results = self._add_to_vector_stores(chunks, chunk_embeddings)

            saved_files = self._save_results(documents, chunks, chunk_embeddings, run_id)

            self.stats["end_time"] = datetime.now().isoformat()
            self.stats["success"] = True
            self.stats["total_documents"] = len(documents)
            self.stats["total_chunks"] = len(chunks)
            self.stats["vector_store_results"] = vector_store_results
            self.stats["saved_files"] = saved_files

            validation_results = self._validate_pipeline(chunks, chunk_embeddings)
            self.stats["validation"] = validation_results

            logger.info(f"Finished pipeline run {run_id}")

            return self.stats

        except Exception as e:
            logger.error(f"Error processing {run_id}: {e}")
            self.stats["error"] = str(e)
            self.stats["success"] = False
            self.stats["end_time"] = datetime.now().isoformat()

            self._save_error_report(e, run_id)

            raise

    def _load_documents(self, input_paths: Union[str, List[str]]) -> Dict[str, Any]:
        """
        load documents
        """
        logger.info(f"Loading documents from {input_paths}")
        start_time = datetime.now()

        try:
            if isinstance(input_paths, str):
                input_paths = [input_paths]

            valid_paths = []
            for path in input_paths:
                if os.path.exists(path):
                    valid_paths.append(path)
                else:
                    logger.warning(f"file {path} does not exist")

            if not valid_paths:
                raise FileNotFoundError("No valid paths found")

            documents = {}
            for path in valid_paths:
                try:
                    filename = os.path.basename(path)
                    logger.info(f"Load {filename}")

                    result = self.pdf_loader.load_pdf(path)

                    result["filepath"] = path
                    result["filename"] = filename
                    result["file_size"] = os.path.getsize(path)

                    documents[filename] = result

                except Exception as e:
                    logger.error(f"Error loading {path}: {e}")

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Finished loading {len(documents)} documents in {elapsed:.2f} seconds")

            self.stats["stages"]["load_documents"] = {
                "duration_seconds": elapsed,
                "documents_loaded": len(documents),
                "document_names": list(documents.keys()),
            }

            return documents

        except Exception as e:
            logger.error(f"Error loading {input_paths}: {e}")
            raise

    def _chunk_documents(self, documents: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk documents
        """
        logger.info(f"Chunking documents")
        start_time = datetime.now()

        try:
            all_chunks = []

            for doc_name, doc_data in documents.items():
                logger.info(f"Chunking {doc_name}")

                if "pages" in doc_data and doc_data["page"]:
                    for page_num, page_text in enumerate(doc_data["pages"],1):
                        chunks = self.text_splitter.split_text(
                            page_text,
                            {
                                "doc_id": doc_name,
                                "page": page_num,
                                "total_pages": doc_data["total_pages"],
                                "source": "pdf",
                                "filename": doc_name,
                                "chunking_strategy": self.config["processing"]["chunking_strategy"],
                            }
                        )
                        all_chunks.extend(chunks)
                else:
                    chunks = self.text_splitter.split_text(
                        doc_data["text"],
                        {
                            "doc_id": doc_name,
                            "page": 1,
                            "total_pages": doc_data["num_pages"],
                            "source": "pdf",
                            "filename": doc_name,
                            "chunking_strategy": self.config["processing"]["chunking_strategy"],
                        }
                    )
                    all_chunks.extend(chunks)

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Generated {len(all_chunks)} chunks in {elapsed:.2f} seconds")

            chunk_stats = ChunkAnalyzer.analyze_chunks(all_chunks)

            self.stats["stages"]["chunk_documents"] = {
                "duration_seconds": elapsed,
                "total_chunks": len(all_chunks),
                "chunk_stats": chunk_stats,
            }
            return all_chunks

        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            raise

    def _create_embeddings(self,
                           chunks: List[Chunk]) -> np.ndarray:

        logger.info("Creating embeddings")
        start_time = datetime.now()

        try:
            self.embeddings_manager.load_model()

            texts = [chunk.text for chunk in chunks]

            embeddings_config = self.config["embeddings"]
            embeddings = self.embeddings_manager.encode(
                texts,
                batch_size=embeddings_config["batch_size"],
                show_progress_bar=True,
                normalize_embeddings=embeddings_config["normalize"],
            )

            for i, chunk in enumerate(chunks):
                chunk.embeddings = embeddings[i].tolist()

            elapsed = (datetime.now() - start_time).total_seconds()
            embedding_dim = embeddings.shape[1]
            logger.info(f"Created embeddings for {len(chunks)} chunks in {elapsed:.2f} seconds")
            logger.info(f"Embedding dimension: {embedding_dim}")
            logger.info(f"Time per chunk: {elapsed/len(chunks):.2f} seconds")

            self.stats["stages"]["create_embeddings"] = {
                "duration_seconds": elapsed,
                "embedding_dimension": embedding_dim,
                "throughput_chunks_per_second": len(chunks) / elapsed if elapsed > 0 else 0,
            }

            return embeddings

        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise

    def _add_to_vector_stores(self,
                              chunks: List[Chunk],
                              embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Adding embeddings to vector store
        """
        logger.info(f"Adding embeddings to vector store")
        start_time = datetime.now()

        try:
            documents = []
            for chunk in chunks:
                documents.append({
                    "id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "source": "pipeline"
                })

            results = {}

            logger.info(f"Adding to ChromaDB")
            chroma_start = datetime.now()
            chroma_store = self.vector_store["chroma"]
            added = chroma_store.add_documents(
                documents=documents,
                embeddings=embeddings,
                batch_size=self.config["embeddings"]["batch_size"],
            )

            chroma_elapsed = (datetime.now() - chroma_start).total_seconds()
            results["chromadb"] = {
                "documents_added": added,
                "duration_seconds": chroma_elapsed,
                "collection_name": chroma_store.collection_name,
            }
            logger.info(f"Added {added} documents in {chroma_elapsed:.2f} seconds")

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Finish adding to vector store in {elapsed:.2f} seconds")

            self.stats["stages"]["add_to_vector_stores"] = {
                "duration_seconds": elapsed,
                "results": results,
            }

            return results

        except Exception as e:
            logger.error(f"Error adding to vector store: {e}")
            raise

    def _save_results(self,
                      documents: Dict[str, Any],
                      chunks: List[Chunk],
                      embeddings: np.ndarray,
                      run_id: str) -> Dict[str, str]:
        """
        Save results
        """
        logger.info(f"Saving results to {run_id}")
        start_time = datetime.now()

        try:
            saved_files = {}

            pipeline_config = {
                "run_id": run_id,
                "pipeline_config": self.config,
                "processing_stats": self.stats["stages"],
                "timestamp": datetime.now().isoformat(),
            }

            saved_files.update(
                self.data_persister.save_full_pipeline_output(
                    documents=documents,
                    chunks=chunks,
                    pipeline_config=pipeline_config,
                )
            )

            embeddings_file = os.path.join(
                self.data_persister.output_dir,
                f"{run_id}_embeddings.npy",
            )
            np.save(embeddings_file, embeddings)
            saved_files["embeddings"] = embeddings_file
            logger.info(f"Saved embeddings to {embeddings_file}")

            stats_file = os.path.join(
                self.data_persister.output_dir,
                f"{run_id}_pipeline_stats.json"
            )
            with open(stats_file, "w", encoding='utf-8') as f:
                json.dump(self.stats, f,ensure_ascii=False, indent=2)
            saved_files["pipeline_stats"] = stats_file

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Finished saving results in {elapsed:.2f} seconds")

            self.stats["stages"]["save_results"] = {
                "duration_seconds": elapsed,
                "files_saved": list(saved_files.keys()),
            }

            return saved_files

        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

    def _validate_pipeline(self,
                           chunks: List[Chunk],
                           embeddings: np.ndarray) -> Dict[str, Any]:
        """
        quality validation
        """
        logger.info("Quality validation")
        start_time = datetime.now()

        try:
            validation_results = {}

            chunk_stats = ChunkAnalyzer.analyze_chunks(chunks)
            validation_results["chunk_quality"] = {
                "avg_length": chunk_stats["avg_length"],
                "length_std": chunk_stats["std_length"],
                "overlap_present": chunk_stats.get("avg_overlap", 0) > 0
            }

            embedding_quality = self._validate_embeddings(embeddings)
            validation_results["embedding_quality"] = embedding_quality

            if self.vector_store:
                search_quality = self._test_search_quality(chunks[:5])
                validation_results["search_quality"] = search_quality

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Finished quality validation in {elapsed:.2f} seconds")

            return validation_results

        except Exception as e:
            logger.error(f"Error validating results: {e}")
            return {"error": str(e)}

    def _validate_embeddings(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Embedding validation quality
        """
        try:
            has_nan = np.isnan(embeddings).any()
            has_inf = np.isinf(embeddings).any()

            norms = np.linalg.norm(embeddings, axis=1)
            avg_norm = np.mean(norms)
            norm_std = np.std(norms)

            n_samples = min(100, len(embeddings))
            indices = np.random.choice(len(embeddings), n_samples, replace=False)
            sampled_embeddings = embeddings[indices]

            similarities = []
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    sim = np.dot(sampled_embeddings[i], sampled_embeddings[j])
                    similarities.append(sim)

            avg_similarity = np.mean(similarities) if similarities else 0
            similarity_std = np.std(similarities) if similarities else 0

            return {
                "has_nan": bool(has_nan),
                "has_inf": bool(has_inf),
                "avg_norm": float(avg_norm),
                "norm_std": float(norm_std),
                "avg_similarity": float(avg_similarity),
                "similarity_std": float(similarity_std),
                "embedding_shape": embeddings.shape,
                "validation_passed": not (has_nan or has_inf) and avg_norm > 0.5
            }

        except Exception as e:
            logger.error(f"Error validating embeddings: {e}")
            return {"error": str(e)}

    def _test_search_quality(self, test_chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Test search quality
        """
        try:
            results = {}

            for store_name, store in self.vector_store.items():
                if store is None:
                    continue

                store_results = []

                for chunk in test_chunks[:3]:
                    search_results = store.search(query=chunk.text[:50], n_results=3)

                    orginal_found = chunk.chunk_id in search_results.get("ids", [])

                    store_results.append({
                        "chunk_id": chunk.chunk_id,
                        "original_found": orginal_found,
                        "top_similarity": search_results.get("similarities", [0])[0] if search_results.get("similarities") else 0,
                    })

                if store_results:
                    recall = sum(1 for r in store_results if r["original_found"]) / len(store_results)
                    avg_similarity = sum(r["top_similarity"] for r in store_results) / len(store_results)

                    results[store_name] = {
                        "recall": recall,
                        "avg_top_similarity": avg_similarity,
                        "test_samples": len(store_results)
                    }

            return results

        except Exception as e:
            logger.error(f"Error testing search quality: {e}")
            return {"error": str(e)}

    def _save_error_report(self, error: Exception, run_id: str):
        """
        save error report
        """
        try:
            error_dir = os.path.join(self.data_persister.output_dir, "errors")
            os.makedirs(error_dir, exist_ok=True)

            error_file = os.path.join(error_dir, f"{run_id}_error.json")

            error_report = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "pipeline_stats": self.stats,
                "config": self.config,
            }

            with open(error_file, "w", encoding='utf-8') as f:
                json.dump(error_report, f, ensure_ascii=False, indent=2)

            logger.info(f"Error report saved in {error_file}")

        except Exception as e:
            logger.error(f"Error saving error report: {e}")

    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Return pipeline info
        """
        return {
            "config": self.config,
            "components": {
                "pdf_loader": "+" if self.pdf_loader else "-",
                "text_splitter": "+" if self.text_splitter else "-",
                "embeddings_manager": "+" if self.embeddings_manager else "-",
                "vector_store": list(self.vector_store.values()),
                "data_persister": "+" if self.data_persister else "-",
            },
            "stats": self.stats if self.stats else "No stats",
        }

    def search(self,
               query: str,
               n_results: int = 5,
               vector_store: str = "chromadb") -> Dict[str, Any]:
        """
        Searching documents in vector store

        Args:
            query: query to search for
            n_results: number of results to return
            vector_store: name of vector store to use

        Returns:
            search results
        """

        store = self.vector_store[vector_store]
        if store is None:
            raise ValueError(f"Vector store {vector_store} not initialized")

        try:
            if vector_store == "chromadb":
                return store.search(query=query, n_results=n_results)
            else:
                query_embedding = self.embeddings_manager.encode_single(query)
                return store.search(n_results=n_results, query_embedding=query_embedding)

        except Exception as e:
            logger.error(f"Error searching results: {e}")
            raise


class PipelineFactory:
    """
    Factory class for creating pipelines
    """

    @staticmethod
    def create_fast_pipeline() -> DocumentPipeline:
        """
        Create fast pipeline
        """
        config = {
            "processing": {
                "chunk_size": 300,
                "chunk_overlap": 30,
                "chunking_strategy": "fixed"
            },
            "embeddings": {
                "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "device": "cpu",
                "batch_size": 64
            },
            "vector_stores": {
                "chromadb": {"enabled": True}
            }
        }
        return DocumentPipeline(config)

    @staticmethod
    def create_quality_pipeline() -> DocumentPipeline:
        """
        Create quality pipeline
        """
        config = {
            "processing": {
                "chunk_size": 500,
                "chunk_overlap": 100,
                "chunking_strategy": "recursive"
            },
            "embeddings": {
                "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "batch_size": 16,
                "normalize": True
            },
            "vector_stores": {
                "chromadb": {"enabled": True}
            }
        }
        return DocumentPipeline(config)

    @staticmethod
    def create_production_pipeline() -> DocumentPipeline:
        """
        Create production pipeline
        """
        config = {
            "processing": {
                "chunk_size": 400,
                "chunk_overlap": 80,
                "chunking_strategy": "sentences",
            },
            "embeddings": {
                "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "batch_size": 32,
                "normalize": True
            },
            "vector_stores": {
                "chromadb": {
                    "enabled": True,
                    "collection_name": "production_documents",
                    "persist_directory": "data/vector_store/production/chroma"
                }
            },
            "output": {
                "save_intermediate": True,
                "intermediate_dir": "data/processed/production",
                "export_formats": ["json", "csv"]
            }
        }
        return DocumentPipeline(config)