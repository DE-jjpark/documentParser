import pytest

from document_parser import ChunkingConfig, IngestPipeline
from document_parser.pipeline import document_to_segments


@pytest.fixture(scope="module")
def pipeline() -> IngestPipeline:
    return IngestPipeline()


def test_ingest_roundtrip(pipeline, tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# Title\n\n" + " ".join(f"word{i}" for i in range(200)))
    chunks = pipeline.ingest(path, config=ChunkingConfig(chunk_size=150, chunk_overlap=30))
    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == str(path) for chunk in chunks)
    assert chunks[0].metadata["element_type"] == "heading"


@pytest.mark.asyncio
async def test_aingest(pipeline):
    chunks = await pipeline.aingest("doc.txt", data=b"paragraph one\n\nparagraph two")
    assert [chunk.text for chunk in chunks] == ["paragraph one", "paragraph two"]


def test_document_to_segments_carries_metadata(pipeline):
    document = pipeline.parsing.parse("doc.md", data=b"# Head\n\nbody")
    segments = document_to_segments(document)
    assert [seg.text for seg in segments] == ["Head", "body"]
    assert segments[0].metadata["element_type"] == "heading"
    assert segments[0].metadata["source"] == "doc.md"
