import os
from src.document_processor.data_persister import DataPersister
from src.document_processor.text_splitter import Chunk

def test_chunk_metadata_access():
    chunk = Chunk(
        text="test",
        metadata={"doc_id": "123", "page": 1},
        chunk_id="chunk_1"
    )

    persister = DataPersister(output_dir="./test_output")
    chunks = [chunk]

    filepath = persister.save_chunks_to_csv(chunks, "test.csv")
    assert os.path.exists(filepath)