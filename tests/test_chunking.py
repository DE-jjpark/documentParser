import pytest

from document_parser import Chunk, ChunkingConfig, ChunkingEngine, ChunkingFailedError, Segment


@pytest.fixture(scope="module")
def engine() -> ChunkingEngine:
    return ChunkingEngine()


def test_chunk_plain_string(engine):
    chunks = engine.chunk("hello world")
    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].text == "hello world"
    assert chunks[0].index == 0


def test_chunks_respect_size(engine):
    text = " ".join(f"word{i}" for i in range(500))
    config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
    chunks = engine.chunk(text, config)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 100 for chunk in chunks)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_all_content_is_preserved(engine):
    text = " ".join(f"word{i}" for i in range(300))
    chunks = engine.chunk(text, ChunkingConfig(chunk_size=120, chunk_overlap=0))
    reassembled = " ".join(chunk.text for chunk in chunks)
    assert reassembled.split() == text.split()


def test_segment_metadata_propagates(engine):
    segments = [Segment(text="some text", metadata={"source": "a.txt", "page": 3})]
    chunks = engine.chunk(segments)
    assert chunks[0].metadata == {"source": "a.txt", "page": 3}


@pytest.mark.asyncio
async def test_achunk(engine):
    chunks = await engine.achunk("async text")
    assert chunks[0].text == "async text"


def test_unknown_strategy_raises(engine):
    with pytest.raises(ChunkingFailedError, match="unknown strategy"):
        engine.chunk("text", ChunkingConfig(strategy="does-not-exist"))


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=100, chunk_overlap=100)


def test_empty_input(engine):
    assert engine.chunk([]) == []
    assert engine.chunk("   ") == []
