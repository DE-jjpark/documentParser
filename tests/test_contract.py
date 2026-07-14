"""Guards on the public API surface that backend consumers depend on."""

import document_parser
from document_parser import Chunk, ParsedDocument


def test_public_api_surface():
    expected = {
        "Chunk",
        "ChunkingConfig",
        "ChunkingEngine",
        "ChunkingFailedError",
        "DocumentElement",
        "DocumentParserError",
        "ElementType",
        "IngestPipeline",
        "MissingDependencyError",
        "ParsedDocument",
        "ParsingEngine",
        "ParsingFailedError",
        "Segment",
        "UnsupportedFormatError",
        "__version__",
    }
    assert set(document_parser.__all__) == expected


def test_models_roundtrip_through_json():
    document = ParsedDocument(
        source="a.txt",
        format="txt",
        elements=[{"type": "text", "text": "hello", "page": 1}],
        metadata={"element_count": 1},
    )
    assert ParsedDocument.model_validate_json(document.model_dump_json()) == document

    chunk = Chunk(id="abc", index=0, text="hello", metadata={"source": "a.txt"})
    assert Chunk.model_validate_json(chunk.model_dump_json()) == chunk


def test_importing_library_does_not_require_api_extra():
    # The base install must not pull in fastapi/uvicorn or the api package.
    import subprocess
    import sys

    code = (
        "import document_parser, sys; "
        "assert 'fastapi' not in sys.modules; "
        "assert 'document_parser.api' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
