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

# 이미지 위치 매칭 테스트는 pymupdf로 PDF를 직접 만들어 쓰므로(LibreOffice 변환
# 없이) 'pdf' extra만 있으면 된다.
requires_pdf_extra = pytest.mark.skipif(
    not _HAS_PYMUPDF, reason="'pdf' extra(pymupdf)가 없음 -- 이미지 위치 매칭 테스트 불가"
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


_DOCX_CONTENT_TYPES = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels"
  ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml"
  ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/comments.xml"
  ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml"/>
</Types>"""

_DOC_RELS = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1"
  Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments"
  Target="comments.xml"/>
</Relationships>"""


def _minimal_docx_with_comment(
    commented_text: str, comment_text: str, intro_text: str = "Intro paragraph, not commented."
) -> bytes:
    """댓글 하나가 두 번째 문단 전체에 앵커된 최소 docx. commentRangeStart/End
    로 감싸고 그 뒤 run에 commentReference를 붙이는 게 Word의 실제 구조."""
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>
<w:p><w:r><w:t>{intro_text}</w:t></w:r></w:p>
<w:p><w:commentRangeStart w:id="0"/><w:r><w:t>{commented_text}</w:t></w:r>
<w:commentRangeEnd w:id="0"/><w:r><w:commentReference w:id="0"/></w:r></w:p>
</w:body>
</w:document>""".encode()
    comments_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:comments xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:comment w:id="0" w:author="Tester"><w:p><w:r><w:t>{comment_text}</w:t></w:r></w:p></w:comment>
</w:comments>""".encode()

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _DOCX_CONTENT_TYPES)
        zf.writestr("_rels/.rels", _RELS)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/_rels/document.xml.rels", _DOC_RELS)
        zf.writestr("word/comments.xml", comments_xml)
    return buffer.getvalue()


_DOCX_IMAGE_CONTENT_TYPES = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels"
  ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/word/document.xml"
  ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/comments.xml"
  ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml"/>
</Types>"""

_DOC_RELS_WITH_IMAGE = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1"
  Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments"
  Target="comments.xml"/>
<Relationship Id="rId2"
  Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
  Target="media/image1.png"/>
</Relationships>"""

_DOCX_DRAWING_NAMESPACES = (
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
)


def _minimal_docx_with_image_comment(image_bytes: bytes, comment_text: str) -> bytes:
    """이미지 하나(인라인 drawing)에 댓글이 달린 최소 docx. commentRangeStart/
    End가 텍스트 run 대신 drawing run을 감싼다 -- Word가 실제로 이미지에 댓글을
    달 때 만드는 구조와 동일(앵커 텍스트가 비게 됨)."""
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document {_DOCX_DRAWING_NAMESPACES}>
<w:body>
<w:p><w:commentRangeStart w:id="0"/><w:r><w:drawing>
<wp:inline><wp:extent cx="914400" cy="914400"/><wp:docPr id="1" name="Picture 1"/>
<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
<pic:pic><pic:blipFill><a:blip r:embed="rId2"/></pic:blipFill></pic:pic>
</a:graphicData></a:graphic></wp:inline>
</w:drawing></w:r>
<w:commentRangeEnd w:id="0"/><w:r><w:commentReference w:id="0"/></w:r></w:p>
</w:body>
</w:document>""".encode()
    comments_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:comments xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:comment w:id="0" w:author="Tester"><w:p><w:r><w:t>{comment_text}</w:t></w:r></w:p></w:comment>
</w:comments>""".encode()

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _DOCX_IMAGE_CONTENT_TYPES)
        zf.writestr("_rels/.rels", _RELS)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/_rels/document.xml.rels", _DOC_RELS_WITH_IMAGE)
        zf.writestr("word/comments.xml", comments_xml)
        zf.writestr("word/media/image1.png", image_bytes)
    return buffer.getvalue()


_PPTX_CONTENT_TYPES = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels"
  ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/ppt/presentation.xml"
  ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
<Override PartName="/ppt/slides/slide1.xml"
  ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
<Override PartName="/ppt/notesSlides/notesSlide1.xml"
  ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/>
</Types>"""

_PPTX_ROOT_RELS = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1"
  Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
  Target="ppt/presentation.xml"/>
</Relationships>"""

_PRESENTATION_XML = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
  xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:sldIdLst><p:sldId id="256" r:id="rId2"/></p:sldIdLst>
<p:sldSz cx="9144000" cy="6858000"/>
</p:presentation>"""

_PRESENTATION_RELS = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId2"
  Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide"
  Target="slides/slide1.xml"/>
</Relationships>"""

_SLIDE1_RELS = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1"
  Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide"
  Target="../notesSlides/notesSlide1.xml"/>
</Relationships>"""


def _minimal_pptx_with_note(slide_text: str, note_text: str) -> bytes:
    slide_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
  xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:cSld><p:spTree>
<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
<p:grpSpPr/>
<p:sp><p:nvSpPr><p:cNvPr id="2" name="Title"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr/>
<p:txBody><a:bodyPr/><a:p><a:r><a:t>{slide_text}</a:t></a:r></a:p></p:txBody></p:sp>
</p:spTree></p:cSld>
</p:sld>""".encode()
    notes_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:notes xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
  xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:cSld><p:spTree>
<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
<p:grpSpPr/>
<p:sp><p:nvSpPr><p:cNvPr id="2" name="Notes Placeholder"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr/>
<p:txBody><a:bodyPr/><a:p><a:r><a:t>{note_text}</a:t></a:r></a:p></p:txBody></p:sp>
</p:spTree></p:cSld>
</p:notes>""".encode()

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _PPTX_CONTENT_TYPES)
        zf.writestr("_rels/.rels", _PPTX_ROOT_RELS)
        zf.writestr("ppt/presentation.xml", _PRESENTATION_XML)
        zf.writestr("ppt/_rels/presentation.xml.rels", _PRESENTATION_RELS)
        zf.writestr("ppt/slides/slide1.xml", slide_xml)
        zf.writestr("ppt/slides/_rels/slide1.xml.rels", _SLIDE1_RELS)
        zf.writestr("ppt/notesSlides/notesSlide1.xml", notes_xml)
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


# ---------------------------------------------------------------------------
# 슬라이드 노트(pptx/ppt) / 문서 댓글(docx) — LibreOffice의 기본 PDF 변환은
# 실측 확인 결과 둘 다 export하지 않아서(변환 시점에 유실됨), office.py가
# PDF 변환과 별도로 원본 zip에서 직접 뽑는다. 아래는 그 순수 추출 함수를
# LibreOffice 없이(가짜 zip만으로) 검증하고, 실제 전체 파이프라인 통합은
# @requires_libreoffice로 한 번 더 확인한다.
# ---------------------------------------------------------------------------


def test_extract_pptx_notes_follows_relationship_chain():
    """슬라이드 번호와 notesSlide 파일 번호가 항상 같지는 않다(실측 확인) —
    그래서 파일명 숫자가 아니라 presentation.xml -> slide -> notesSlide
    관계를 실제로 따라가는지 검증."""
    from document_parser.parsing.loaders.office import _extract_pptx_notes

    data = _minimal_pptx_with_note("Slide One", "Speaker note for slide one")

    notes = _extract_pptx_notes(data)

    assert len(notes) == 1
    assert notes[0].type == ElementType.NOTE
    assert notes[0].text == "Speaker note for slide one"
    assert notes[0].page == 1
    assert notes[0].metadata["source"] == "pptx_notes"


def test_extract_pptx_notes_returns_empty_when_no_notes_slide():
    from document_parser.parsing.loaders.office import _extract_pptx_notes

    # notesSlide 관계 자체가 없는 경우 -- _minimal_pptx_with_note 없이 그냥
    # 노트 없는 상태를 흉내내려면 rels에서 notesSlide 항목을 빼야 하므로,
    # 여기선 완전히 빈 zip으로 "필수 파트가 없음" 경로만 확인한다.
    empty_pptx = io.BytesIO()
    with zipfile.ZipFile(empty_pptx, "w") as zf:
        zf.writestr("dummy.txt", "not a real pptx")

    assert _extract_pptx_notes(empty_pptx.getvalue()) == []


def test_extract_docx_comments_raw_captures_anchor_spanning_full_run():
    from document_parser.parsing.loaders.office import _extract_docx_comments_raw

    data = _minimal_docx_with_comment(
        commented_text="This is the commented sentence.",
        comment_text="Reviewer note text.",
    )

    comments = _extract_docx_comments_raw(data)

    assert comments == [
        {
            "text": "Reviewer note text.",
            "anchor_text": "This is the commented sentence.",
            "image_rids": [],
        }
    ]


def test_extract_docx_comments_raw_returns_empty_without_comments_part():
    from document_parser.parsing.loaders.office import _extract_docx_comments_raw

    assert _extract_docx_comments_raw(_minimal_docx(["No comments here."])) == []


def test_docx_comment_matches_best_overlapping_element_and_borrows_its_position():
    """앵커 텍스트와 겹치는 elements 중 가장 잘 겹치는 것의 page/bbox를
    빌려 쓰는지 확인 -- 무관한 elements가 섞여 있어도 정확한 것 하나만
    골라야 한다."""
    from document_parser.core.models import BBox, DocumentElement
    from document_parser.parsing.loaders.office import _extract_docx_comments

    data = _minimal_docx_with_comment(
        commented_text="This is the commented sentence.",
        comment_text="Reviewer note text.",
    )
    unrelated = DocumentElement(
        type=ElementType.TEXT, text="Completely unrelated paragraph.", page=1, bboxes=[]
    )
    matching = DocumentElement(
        type=ElementType.TEXT,
        text="This is the commented sentence.",
        page=3,
        bboxes=[BBox(x0=1, y0=2, x1=3, y1=4)],
    )

    result = _extract_docx_comments(data, [unrelated, matching])

    assert len(result) == 1
    assert result[0].type == ElementType.NOTE
    assert result[0].text == "Reviewer note text."
    assert result[0].page == 3
    assert result[0].bboxes == [BBox(x0=1, y0=2, x1=3, y1=4)]
    assert result[0].metadata["source"] == "docx_comment"


def test_docx_comment_without_matching_element_has_no_position():
    from document_parser.core.models import DocumentElement
    from document_parser.parsing.loaders.office import _extract_docx_comments

    data = _minimal_docx_with_comment(
        commented_text="This is the commented sentence.",
        comment_text="Reviewer note text.",
    )
    unrelated = DocumentElement(type=ElementType.TEXT, text="Nothing like it.", page=1, bboxes=[])

    result = _extract_docx_comments(data, [unrelated])

    assert result[0].page is None
    assert result[0].bboxes == []
    assert result[0].metadata["anchor_text"] == "This is the commented sentence."


@requires_libreoffice
def test_pptx_notes_appear_as_note_elements_after_full_conversion(engine):
    data = _minimal_pptx_with_note("Slide One", "Speaker note for slide one")

    document = engine.parse("deck.pptx", data=data)

    notes = [el for el in document.elements if el.type == ElementType.NOTE]
    assert len(notes) == 1
    assert notes[0].text == "Speaker note for slide one"
    assert notes[0].page == 1


@requires_libreoffice
def test_docx_comments_appear_as_note_elements_after_full_conversion(engine):
    data = _minimal_docx_with_comment(
        commented_text="This is the commented sentence.",
        comment_text="Reviewer note text.",
    )

    document = engine.parse("report_with_comment.docx", data=data)

    notes = [el for el in document.elements if el.type == ElementType.NOTE]
    assert len(notes) == 1
    assert notes[0].text == "Reviewer note text."
    # 앵커 텍스트("This is the commented sentence.")와 겹치는 실제 TEXT
    # element가 변환된 PDF에도 있어야 하고, 댓글이 그 위치를 빌려써야 한다.
    assert notes[0].page is not None
    assert len(notes[0].bboxes) >= 1


def test_docx_comment_spans_captures_image_rid_when_anchor_wraps_a_drawing():
    """텍스트 대신(또는 텍스트 없이) 이미지에 댓글이 달리면 anchor_text는
    비지만 image_rids에 r:embed 값이 남아야 한다."""
    from document_parser.parsing.loaders.office import _docx_comment_spans

    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document {_DOCX_DRAWING_NAMESPACES}>
<w:body>
<w:p><w:commentRangeStart w:id="0"/><w:r><w:drawing>
<wp:inline><a:graphic><a:graphicData>
<pic:pic><pic:blipFill><a:blip r:embed="rId2"/></pic:blipFill></pic:pic>
</a:graphicData></a:graphic></wp:inline>
</w:drawing></w:r>
<w:commentRangeEnd w:id="0"/></w:p>
</w:body>
</w:document>""".encode()

    spans = _docx_comment_spans(document_xml)

    assert spans["0"]["text"] == ""
    assert spans["0"]["image_rids"] == ["rId2"]


def test_extract_docx_comments_raw_reports_image_rids():
    from document_parser.parsing.loaders.office import _extract_docx_comments_raw

    def make_png(seed: int) -> bytes:
        import pymupdf

        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 4, 4), False)
        pix.set_rect(pix.irect, (seed, seed, seed))
        return pix.tobytes("png")

    data = _minimal_docx_with_image_comment(make_png(10), comment_text="Comment on a picture.")

    comments = _extract_docx_comments_raw(data)

    assert comments == [
        {"text": "Comment on a picture.", "anchor_text": "", "image_rids": ["rId2"]}
    ]


@requires_pdf_extra
def test_bbox_iou_overlapping_vs_disjoint():
    from document_parser.parsing.loaders.office import _bbox_iou

    assert _bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)
    assert _bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


@requires_pdf_extra
def test_perceptual_hash_matches_reencoded_image_but_not_a_different_one():
    """LibreOffice가 원본 이미지를 재인코딩해도(포맷 변경 등) 같은 그림으로
    인식돼야 하고, 실제로 다른 그림과는 구분돼야 한다."""
    import pymupdf

    from document_parser.parsing.loaders.office import _hamming_distance, _perceptual_hash

    def make_pattern_png(seed: int) -> bytes:
        import random

        rng = random.Random(seed)
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 32, 32), False)
        for qy in range(4):
            for qx in range(4):
                shade = rng.randint(0, 255)
                rect = pymupdf.IRect(qx * 8, qy * 8, (qx + 1) * 8, (qy + 1) * 8)
                pix.set_rect(rect, (shade, shade, shade))
        return pix.tobytes("png")

    original = make_pattern_png(1)
    reencoded_as_jpg = pymupdf.Pixmap(original).tobytes("jpg")
    different = make_pattern_png(2)

    h_original = _perceptual_hash(original)
    h_reencoded = _perceptual_hash(reencoded_as_jpg)
    h_different = _perceptual_hash(different)

    assert _hamming_distance(h_original, h_reencoded) <= 2
    assert _hamming_distance(h_original, h_different) > 10


@requires_pdf_extra
def test_find_image_location_in_pdf_locates_matching_page_and_bbox():
    import pymupdf

    from document_parser.parsing.loaders.office import _find_image_location_in_pdf

    def make_pattern_png(seed: int) -> bytes:
        import random

        rng = random.Random(seed)
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 32, 32), False)
        for qy in range(4):
            for qx in range(4):
                shade = rng.randint(0, 255)
                rect = pymupdf.IRect(qx * 8, qy * 8, (qx + 1) * 8, (qy + 1) * 8)
                pix.set_rect(rect, (shade, shade, shade))
        return pix.tobytes("png")

    target_image = make_pattern_png(1)
    other_image = make_pattern_png(2)

    doc = pymupdf.open()
    page1 = doc.new_page(width=300, height=300)
    page1.insert_image(pymupdf.Rect(10, 10, 60, 60), stream=other_image)
    page2 = doc.new_page(width=300, height=300)
    page2.insert_image(pymupdf.Rect(100, 120, 150, 170), stream=target_image)
    pdf_bytes = doc.tobytes()
    doc.close()

    location = _find_image_location_in_pdf(pdf_bytes, target_image)

    assert location is not None
    page_number, bbox = location
    assert page_number == 2
    assert bbox[0] == pytest.approx(100, abs=1) and bbox[1] == pytest.approx(120, abs=1)
    assert bbox[2] == pytest.approx(150, abs=1) and bbox[3] == pytest.approx(170, abs=1)


@requires_pdf_extra
def test_docx_image_comment_matches_overlapping_image_element_by_position():
    """이미지에 달린 댓글은 텍스트 겹침이 아니라 위치(bbox)로 매칭돼야 한다
    -- 원본 docx 이미지가 변환된 PDF의 어디에 있는지 찾고, 그 bbox와 겹치는
    IMAGE element의 page/bbox를 빌려 쓴다."""
    import pymupdf

    from document_parser.core.models import BBox, DocumentElement
    from document_parser.parsing.loaders.office import _extract_docx_comments

    def make_pattern_png(seed: int) -> bytes:
        import random

        rng = random.Random(seed)
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 32, 32), False)
        for qy in range(4):
            for qx in range(4):
                shade = rng.randint(0, 255)
                rect = pymupdf.IRect(qx * 8, qy * 8, (qx + 1) * 8, (qy + 1) * 8)
                pix.set_rect(rect, (shade, shade, shade))
        return pix.tobytes("png")

    image_bytes = make_pattern_png(7)
    data = _minimal_docx_with_image_comment(image_bytes, comment_text="What is this chart?")

    doc = pymupdf.open()
    page = doc.new_page(width=300, height=300)
    page.insert_image(pymupdf.Rect(50, 50, 120, 120), stream=image_bytes)
    pdf_bytes = doc.tobytes()
    doc.close()

    unrelated_text = DocumentElement(
        type=ElementType.TEXT, text="Some unrelated paragraph.", page=1, bboxes=[]
    )
    matching_image = DocumentElement(
        type=ElementType.IMAGE,
        text="A caption made up by the VLM, unrelated to the doc wording.",
        page=1,
        bboxes=[BBox(x0=48, y0=48, x1=122, y1=122)],
    )

    result = _extract_docx_comments(data, [unrelated_text, matching_image], pdf_bytes)

    assert len(result) == 1
    assert result[0].type == ElementType.NOTE
    assert result[0].text == "What is this chart?"
    assert result[0].page == 1
    assert result[0].bboxes == [BBox(x0=48, y0=48, x1=122, y1=122)]
