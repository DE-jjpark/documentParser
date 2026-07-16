"""docx/pptx/ppt/doc 로더 테스트 — 실제 LibreOffice(soffice) 변환을 거친다.

python-docx 같은 외부 라이브러리 없이, zipfile로 최소한의 유효한 .docx를
직접 만들어서 쓴다(Office Open XML은 그냥 zip + XML이라 가능) — 이 프로젝트에
새 의존성을 추가하지 않기 위해서다.
"""

from __future__ import annotations

import io
import shutil
import zipfile

import pytest

from document_parser import ElementType, MissingDependencyError, ParsingEngine

_HAS_PYMUPDF = True
try:
    import pymupdf  # noqa: F401
except ImportError:
    _HAS_PYMUPDF = False

# office.py는 변환 후 내부적으로 pdf.load()를 재사용하므로(pymupdf 필요),
# LibreOffice가 있어도 'pdf' extra가 없으면(예: CI의 api-only 재현 환경) 이
# 테스트는 성립하지 않는다 — 둘 다 있어야 실제로 끝까지 돈다.
requires_libreoffice = pytest.mark.skipif(
    not (_HAS_PYMUPDF and (shutil.which("soffice") or shutil.which("libreoffice"))),
    reason="LibreOffice(soffice) 및/또는 'pdf' extra(pymupdf)가 없음 "
    "-- docx/pptx/ppt 변환 테스트 불가",
)

_CONTENT_TYPES = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels"
  ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml"
  ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

_RELS = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1"
  Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
  Target="word/document.xml"/>
</Relationships>"""


def _minimal_docx(paragraphs: list[str]) -> bytes:
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>{body}</w:body>
</w:document>""".encode()

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _CONTENT_TYPES)
        zf.writestr("_rels/.rels", _RELS)
        zf.writestr("word/document.xml", document_xml)
    return buffer.getvalue()


@pytest.fixture(scope="module")
def engine() -> ParsingEngine:
    return ParsingEngine()


@requires_libreoffice
def test_docx_converts_via_libreoffice_and_reuses_pdf_pipeline(engine):
    data = _minimal_docx(["Hello from docx", "Second paragraph."])
    document = engine.parse("report.docx", data=data)

    assert document.format == "docx"
    texts = [el.text for el in document.elements if el.type == ElementType.TEXT]
    assert any("Hello from docx" in t for t in texts)
    assert any("Second paragraph" in t for t in texts)


def test_missing_libreoffice_raises_helpful_error(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(MissingDependencyError, match="LibreOffice"):
        ParsingEngine().parse("report.docx", data=_minimal_docx(["x"]))
