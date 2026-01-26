import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Union
import uuid
import numpy as np
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ChromaDBVectorStore:
    """
    Class to manage vector store in ChromaDB
    """
    def __init__(self,
                 collection_name: str = "documents",
                 persist_directory: str = "data/vector_store/chroma",
                 embedding_function = None,
                 reset_on_start: bool = False):
        """
        Args:
            collection_name: name of the collection in ChromaDB
            persist_directory: directory to store vector store in ChromaDB
            embedding_function: function to create embeddings (if None, using default)
            reset_on_start: whether to reset vector store on start
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.reset_on_start = reset_on_start

        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = None
        self.collection = None

        self._initialize_client()

    def _initialize_client(self):
        """
        Initialize the client to connect to ChromaDB
        """
        try:

            self.client = chromadb.PersistentClient(path=self.persist_directory)

            if self.reset_on_start:
                try:
                    self.client.delete_collection(self.collection_name)
                    logger.info(f"Deleted {self.collection_name} collection")
                except Exception:
                    pass

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={
                    "hnsw:space": "cosine",
                    "created": datetime.now().isoformat()
                }
            )
            self._log_collection_stats()

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _log_collection_stats(self):
        """
        Loging collection stats
        """
        if self.collection is None:
            return

        try:
            count = self.collection.count()
            logger.info(f"Collection contains {count} documents")
        except:
            logger.info(f"Collection is Empty")

    def add_documents(self,
                      documents: List[Dict[str, Any]],
                      batch_size: int = 100) -> int:
        """
        Add documents to vector store

        Args:
            documents: list of documents
            embeddings optional embeddings(if None, will be calculated)
            batch_size: number of documents to add

        Returns:
            number of added documents
        """
        if not documents:
            logger.warning("No documents to add")
            return 0

        total_added = 0

        ids = []
        texts = []
        metadatas = []

        for i, doc in enumerate(documents):
            if "text" not in doc:
                logger.warning(f"Document {i} has no text")
                continue

            doc_id = doc.get("id", str(uuid.uuid4()))

            ids.append(doc_id)
            texts.append(doc["text"])

            metadata = doc.get("metadata", {}).copy()
            metadata["source"] = doc.get("source","unknown")
            metadata["timestamp"] = datetime.now().isoformat()
            metadata["text_length"] = len(doc["text"])

            metadatas.append(metadata)

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]

            try:
                self.collection.add(
                    ids = batch_ids,
                    metadatas = batch_metadatas,
                    documents = batch_texts
                )

                total_added += len(batch_ids)
                logger.info(f"Added {i//batch_size + 1}: {len(batch_ids)} documents")

            except Exception as e:
                logger.error(f"Failed to add {i//batch_size + 1}: {e}")

                for j in range(len(batch_ids)):
                    try:
                        self.collection.add(
                            ids=[batch_ids[j]],
                            documents=[batch_texts[j]],
                            metadatas=[batch_metadatas[j]],
                        )
                        total_added += 1
                    except Exception:
                        logger.error(f"Failed to add document {batch_ids[j]}")

        logger.info(f"Added {total_added}/{len(documents)} documents")
        self._log_collection_stats()
        return total_added

    def search(self,
               query: str,
               n_results: int = 5,
               where: Optional[Dict] = None,
               where_document: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Search for similar documents

        Returns:
            Dictionary with search results
        """
        if self.collection is None:
            raise ValueError("Collection must be initialized")

        try:
            results = self.collection.query(
                query_texts = [query],
                n_results = n_results,
                where = where,
                where_document = where_document,
            )

            processed_results = {
                "query": query,
                "ids": results["ids"][0] if results["ids"] else [],
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "similarities": [1 - d for d in results["distances"][0]] if results["distances"] else [],
            }
            logger.info(f"Found {len(processed_results['ids'])} results for {query}")
            return processed_results

        except Exception as e:
            logger.error(f"Failed to search results: {e}")
            raise

    def search_with_embeddings(self,
                               query_embedding: np.ndarray,
                               n_results: int = 5,
                               where: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Searches using the query embedding directly
        """
        if self.collection is None:
            raise ValueError("Collection must be initialized")

        try:
            results = self.collection.query(
                query_embeddings = [query_embedding.tolist()],
                n_results = n_results,
                where = where,
            )

            processed_results = {
                "ids": results["ids"][0] if results["ids"] else [],
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "similarities": [1 - d for d in results["distances"][0]] if results["distances"] else [],
            }
            return processed_results

        except Exception as e:
            logger.error(f"Error searched with embeddings: {e}")
            raise

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document by id
        """
        try:
            results = self.collection.get(ids=[document_id])

            if results["ids"]:
                return {
                    "id":results["ids"][0],
                    "document":results["documents"][0],
                    "metadata":results["metadatas"][0],
                }
            return None

        except Exception as e:
            logger.error(f"Failed to get document {document_id}")
            return None

    def update_document(self,
                        document_id: str,
                        text: Optional[str] = None,
                        metadata: Optional[Dict] = None):
        """
        Update document
        """
        try:
            update_data = {"ids": [document_id]}

            if text is not None:
                update_data["documents"] = [text]

            if metadata is not None:
                update_data["metadatas"] = [metadata]

            if len(update_data) > 1:
                self.collection.update(**update_data)
                logger.info(f"Updated document {document_id}")

        except Exception as e:
            logger.error(f"Failed to update document {document_id}")
            raise

    def delete_document(self, document_id: str):
        """
        Delete document
        """
        try:
            self.collection.delete(ids=[document_id])
            logger.info(f"Deleted document {document_id}")
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Return collection stats
        """
        try:
            count = self.collection.count()

            sample = self.collection.get(limit=min(10, count))

            stats = {
                "total_documents": count,
                "sample_size": len(sample["ids"]) if sample["ids"] else 0,
                "metadata_fields": set(),
                "timestamp": datetime.now().isoformat(),
            }

            if sample["metadatas"]:
                for metadata in sample["metadatas"]:
                    if metadata:
                        stats["metadata_fields"].update(metadata.keys())
                stats["metadata_fields"] = list(stats["metadata_fields"])

            return stats

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e), "total_documents": 0}

    def export_collection(self, export_path: str):
        """
        Export collection to JSON file
        """
        try:
            all_documents = self.collection.get()

            export_data = {
                "metadata": {
                    "collection_name": self.collection.name,
                    "export_date": datetime.now().isoformat(),
                    "total_documents": len(all_documents["ids"])
                },
                "documents": []
            }

            for i in range(len(all_documents["ids"])):
                document = {
                    "id": all_documents["ids"][i],
                    "text": all_documents["documents"][i],
                    "metadata": all_documents["metadatas"][i] if all_documents["metadatas"] else {},
                }
                export_data["documents"].append(document)

            with open(export_path, "w", encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Exported {len(export_data['documents'])} documents to {export_path}")

        except Exception as e:
            logger.error(f"Failed to export collection: {e}")
            raise

    def backup(self, backup_dir: str = "backups"):
        """
        Creates a backup of the entire database
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            backup_path = os.path.join(backup_dir, f"chroma_backup_{timestamp}")

            import shutil
            shutil.copytree(self.persist_directory, backup_path)

            logger.info(f"Created backup {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"Failed to backup backup: {e}")
            raise
