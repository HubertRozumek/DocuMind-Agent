import pytest
from src.document_processor.text_splitter import TextSplitter, Chunk

@pytest.fixture
def sample_text():
    return """Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
    Sed non risus. Suspendisse lectus tortor, dignissim sit amet, 
    adipiscing nec, ultricies sed, dolor.

    Cras elementum ultrices diam. Maecenas ligula massa, varius a, 
    semper congue, euismod non, mi.

    Proin porttitor, orci nec nonummy molestie, enim est eleifend mi, 
    non fermentum diam nisl sit amet erat.
    """

@pytest.mark.parametrize("strategy", ["recursive","sentences","fixed"])
def test_text_splitter(strategy, sample_text):
    splitter = TextSplitter(chunk_size=150,chunk_overlap=20,strategy=strategy)

    chunks = splitter.split_text(sample_text,metadata={"doc_id": "lorem_test"})

    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)

    for i, chunk in enumerate(chunks):
        assert chunk.text.strip() != ""
        assert len(chunk.text) <= splitter.chunk_size
        assert "strategy" in chunk.metadata
        assert chunk.metadata["strategy"] == strategy
        assert chunk.chunk_id.startswith("lorem_test")

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))

def test_chunk_overlap_exists(sample_text):
    splitter = TextSplitter(
        chunk_size=120,
        chunk_overlap=40,
        strategy="recursive"
    )

    chunks = splitter.split_text(sample_text)

    overlaps = []
    for c1, c2 in zip(chunks, chunks[1:]):
        max_check = min(len(c1.text), len(c2.text), 40)
        overlap_found = False

        for size in range(max_check, 5, -1):
            if c1.text.endswith(c2.text[:size]):
                overlaps.append(size)
                overlap_found = True
                break

        assert overlap_found is True

    assert sum(overlaps) > 0


def test_no_empty_chunks(sample_text):
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_text(sample_text)

    assert all(len(chunk.text.strip()) > 0 for chunk in chunks)


def test_chunk_size_respected(sample_text):
    splitter = TextSplitter(chunk_size=80, chunk_overlap=10)
    chunks = splitter.split_text(sample_text)

    for chunk in chunks:
        assert len(chunk.text) <= 80


def test_single_chunk_when_text_is_short():
    text = "This is a short sentence."
    splitter = TextSplitter(chunk_size=500, chunk_overlap=50)

    chunks = splitter.split_text(text)

    assert len(chunks) == 1
    assert chunks[0].text == text.strip()