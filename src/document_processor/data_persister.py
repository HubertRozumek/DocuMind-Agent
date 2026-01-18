import json
import pickle
import csv
from typing import List, Dict, Any
import os
from datetime import datetime
from .text_splitter import Chunk

class DataPersister:
    """
    A class for saving processed data to various formats
    """

    def __init__(self,output_dir="data/processed"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_chunks_to_json(self,
                            chunks: List[Chunk],
                            filename: str = "chunks.json",
                            include_embeddings: bool = False) -> str:
        """
        Saves chunks to a json file

        Args:
            chunks: List of Chunks to save
            filename: Filename to save chunks to
            include_embeddings: Whether to include embeddings
        """

        filepath = os.path.join(self.output_dir, filename)

        chunks_data = []
        for chunk in chunks:
            chunk_dict = chunk.to_dict()

            if not include_embeddings and "embeddings" in chunk_dict:
                del chunk_dict["embeddings"]

            chunks_data.append(chunk_dict)

        metadata = {
            "generated_at": datetime.now().isoformat(),
            "total_chunks": len(chunks),
            "total_documents": len(set(ch.metadata("doc_id", "") for ch in chunks)),
            "chunk_schema": {
                "text": "string",
                "metadata": "dict",
                "chunk_id": "string",
                "embeddings": "list[float] (optional)",
            }
        }

        data = {
            "metadata": metadata,
            "chunks": chunks_data,
        }

        with open(filepath, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f'Saved {len(chunks)} chunks to {filepath}')
        return filepath

    def save_chunks_to_csv(self,
                           chunks: List[Chunk],
                           filename: str = "chunks.csv") -> str:
        """
        Saves chunks to a csv file

        Args:
            chunks: List of Chunks to save
            filename: Filename to save chunks to
        """
        filepath = os.path.join(self.output_dir, filename)

        rows = []
        for chunk in chunks:
            row = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "text_length": len(chunk.text),
                "document": chunk.metadata("doc_id", "unknown"),
                "strategy": chunk.metadata("strategy", "unknown"),
                "page": chunk.metadata("page", ""),
                "timestamp": chunk.metadata("timestamp", ""),
            }
            rows.append(row)

        with open(filepath, "w",newline='', encoding='utf-8') as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        print(f'Saved {len(chunks)} chunks to {filepath}')
        return filepath

    def save_chunks_to_pickle(self,
                              chunks: List[Chunk],
                              filename: str = "chunks.pkl") -> str:
        """
        Saves chunks to a pickle file

        Args:
            chunks: List of Chunks to save
            filename: Filename to save chunks to
        """
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "wb") as f:
            pickle.dump(chunks, f)

        print(f'Saved {len(chunks)} chunks to {filepath}')
        return filepath

    def save_statistics(self,
                        chunks: List[Chunk],
                        filename: str = "chunk_statistics.json") -> str:
        """
        Saves chunks statistics to a json file

        Args:
            chunks: List of Chunks to save
            filename: Filename to save chunks to
        """

        from .text_splitter import ChunkAnalyzer

        filepath = os.path.join(self.output_dir, filename)

        stats = ChunkAnalyzer.analyze_chunks(chunks)

        doc_distribution = {}
        for chunk in chunks:
            doc_id = chunk.metadata.get("doc_id", "unknown")
            doc_distribution[doc_id] = doc_distribution.get(doc_id, 0) + 1

        stats["doc_distribution"] = doc_distribution

        strategy_distribution = {}
        for chunk in chunks:
            strategy = chunk.metadata.get("strategy", "unknown")
            strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1

        stats["strategy_distribution"] = strategy_distribution

        with open(filepath, "w", encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"Saved statistics to {filepath}")
        return filepath

    def save_full_pipeline_output(self,
                                  documents: Dict[str, Any],
                                  chunks: List[Chunk],
                                  pipeline_config: Dict[str, Any]) -> Dict[str, str]:
        """
        Saves full pipeline output

        Args:
            documents: Dictionary of documents
            chunks: List of Chunks
            pipeline_config: Dictionary of pipeline configuration
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_filename = f"pipeline_output_{timestamp}"
        saved_files = {}

        docs_file = os.path.join(self.output_dir, f"{base_filename}_documents.json")
        with open(docs_file, "w", encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        saved_files["documents"] = docs_file

        chunks_json_file = self.save_chunks_to_json(chunks, f"{base_filename}_chunks.json")
        saved_files["chunks_json"] = chunks_json_file

        chunks_csv_file = self.save_chunks_to_csv(chunks, f"{base_filename}_chunks.csv")
        saved_files["chunks_csv"] = chunks_csv_file

        stats_file = self.save_statistics(chunks, f"{base_filename}_statistics.json")
        saved_files["statistics"] = stats_file

        config_file = os.path.join(self.output_dir, f"{base_filename}_pipeline_config.json")
        config_data = {
            "pipeline_config": pipeline_config,
            "timestamp": timestamp,
            "chunk_count": len(chunks),
            "document_count": len(documents),
        }
        with open(config_file, "w", encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        saved_files["config"] = config_file

        return saved_files

