import pytest

from document_parser import (
    ElementType,
    MissingDependencyError,
    ParsingEngine,
    ParsingFailedError,
    UnsupportedFormatError,
)


@pytest.fixture(scope="module")
def engine() -> ParsingEngine:
    return ParsingEngine()


def test_parse_txt_file(engine, tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("first paragraph\n\nsecond paragraph\n")
    document = engine.parse(path)
    assert document.format == "txt"
    assert [el.text for el in document.elements] == ["first paragraph", "second paragraph"]
    assert document.metadata["element_count"] == 2


def test_parse_markdown_headings(engine):
    data = b"# Title\n\nbody text\n\n## Section\n\nmore text"
    document = engine.parse("doc.md", data=data)
    types = [el.type for el in document.elements]
    assert types == [ElementType.HEADING, ElementType.TEXT, ElementType.HEADING, ElementType.TEXT]
    assert document.elements[0].metadata["level"] == 1


def test_elements_get_sequential_keys_per_page_and_type(engine):
    """assemble()이 붙이는 key는 "{page}-{type}-{n}" 형태 — 페이지 개념이
    없는 포맷(md)은 page를 0으로 취급하고, elements 순서(=읽기 순서) 그대로
    타입별 일련번호를 매긴다."""
    data = b"# Title\n\nfirst paragraph\n\n## Section\n\nsecond paragraph"
    document = engine.parse("doc.md", data=data)
    keys = [el.key for el in document.elements]
    assert keys == ["0-heading-1", "0-text-1", "0-heading-2", "0-text-2"]


def test_parse_bytes_with_explicit_format(engine):
    document = engine.parse("no-extension", data=b"hello", format="txt")
    assert document.format == "txt"
    assert document.text == "hello"


@pytest.mark.asyncio
async def test_aparse(engine):
    document = await engine.aparse("doc.txt", data=b"async hello")
    assert document.text == "async hello"


def test_unsupported_format_raises(engine):
    with pytest.raises(UnsupportedFormatError):
        engine.parse("doc.xyz", data=b"data")


def test_missing_file_raises(engine, tmp_path):
    with pytest.raises(ParsingFailedError):
        engine.parse(tmp_path / "missing.txt")


def test_pdf_without_extra_raises_helpful_error(engine):
    try:
        import pymupdf  # noqa: F401
    except ImportError:
        with pytest.raises(MissingDependencyError, match="pdf"):
            engine.parse("doc.pdf", data=b"%PDF-1.4")
    else:
        pytest.skip("pymupdf installed; missing-dependency path not reachable")
