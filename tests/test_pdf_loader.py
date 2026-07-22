"""pdf 로더의 페이지별 라우팅(native / native+vlm / vlm_only) 테스트.

fixture 파일 대신 pymupdf로 그 자리에서 작은 합성 PDF를 만든다(기존
text 로더 테스트의 "hello world" 스타일과 동일).

layout.analyze_page()는 실제로(설치된 것 기준: 'layout' extra + 가중치가
있으면 진짜 PP-DocLayoutV2, 없으면 pymupdf 휴리스틱 폴백) 실행해서 검증한다
— 둘 다 이 합성 페이지들을 올바르게 분류한다. vlm은 graph.py가 호출하는
지점에서 mock한다 — 실제 자격증명이 이 환경에 없어서다.

AzureDI는 더 이상 쓰지 않는다(팀 결정) — 표 구조 추출도 스캔 페이지 본문
추출도 전부 VLM이 담당한다.

라우팅 규칙(layout.route_page 참고):
  - 텍스트 레이어 있음 + 그림·표 없음  -> native만
  - 텍스트 레이어 있음 + 그림·표 있음  -> native(순수 텍스트) + vlm(그림
    캡션/표 마크다운) 병렬
  - 텍스트 레이어 없음(스캔 문서)      -> vlm(text_boxes 전사) + vlm(crop_boxes
    그림·표 캡션) 병렬
"""

import os
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

pymupdf = pytest.importorskip("pymupdf", reason="pdf extra not installed")

from document_parser import ElementType, ParsingEngine  # noqa: E402
from document_parser.core.models import BBox, DocumentElement  # noqa: E402
from document_parser.parsing.loaders.pdf.graph import build_page_graph  # noqa: E402
from document_parser.parsing.loaders.pdf.layout import PageLayout, route_page  # noqa: E402
from document_parser.parsing.weights import layout_model_dir  # noqa: E402

_HAS_REAL_LAYOUT_MODEL = False
try:
    import paddleocr  # noqa: F401

    _HAS_REAL_LAYOUT_MODEL = any(layout_model_dir().glob("*"))
except ImportError:
    pass

requires_real_layout_model = pytest.mark.skipif(
    not _HAS_REAL_LAYOUT_MODEL,
    reason="'layout' extra + 가중치 필요 (25개 카테고리는 실제 PP-DocLayoutV2가 있어야 나옴)",
)

requires_real_vlm = pytest.mark.skipif(
    not (os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN")),
    reason="DATABRICKS_HOST / DATABRICKS_TOKEN 환경변수 필요 (실제 in4u AI Gateway 호출)",
)


def _text_only_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "hello world")
    data = doc.tobytes()
    doc.close()
    return data


def _pdf_with_image() -> bytes:
    """텍스트 레이어 + 그림 — 텍스트 레이어가 있으므로 native+vlm 경로를 탄다."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "a caption above a figure")
    pix = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 8, 8), False)
    pix.set_rect(pix.irect, (255, 0, 0))
    page.insert_image(pymupdf.Rect(72, 200, 172, 300), pixmap=pix)
    data = doc.tobytes()
    doc.close()
    return data


def _scanned_pdf_with_text() -> bytes:
    """텍스트가 있는 페이지를 렌더링해 이미지로 만든 뒤 그 이미지만 새
    페이지에 붙여넣는다 — 텍스트 레이어는 없지만(스캔 문서처럼) 화면상으로는
    실제로 읽을 수 있는 텍스트가 있어서 VLM 전사가 진짜로 인식할 게 있다."""
    text_doc = pymupdf.open()
    text_page = text_doc.new_page(width=300, height=100)
    text_page.insert_text((10, 50), "Scanned document text for OCR test.", fontsize=14)
    pix = text_page.get_pixmap(dpi=200)
    text_doc.close()

    doc = pymupdf.open()
    page = doc.new_page(width=300, height=100)
    page.insert_image(page.rect, pixmap=pix)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture(scope="module")
def engine() -> ParsingEngine:
    return ParsingEngine()


def test_text_only_page_takes_native_route(engine):
    document = engine.parse("doc.pdf", data=_text_only_pdf())

    assert len(document.elements) == 1
    element = document.elements[0]
    assert element.type == ElementType.TEXT
    assert element.text == "hello world"
    assert element.page == 1
    assert len(element.bboxes) == 1
    assert element.metadata["source"] == "native"


def test_page_with_image_and_text_layer_takes_native_and_vlm(engine):
    fake_vlm_element = DocumentElement(
        type=ElementType.IMAGE,
        text="from vlm",
        page=1,
        bboxes=[BBox(x0=0, y0=0, x1=1, y1=1)],
        metadata={"source": "vlm"},
    )

    with patch(
        "document_parser.parsing.loaders.pdf.graph.caption_figures",
        return_value=[fake_vlm_element],
    ) as mock_vlm:
        document = engine.parse("doc.pdf", data=_pdf_with_image())

    mock_vlm.assert_called_once()

    sources = {el.metadata.get("source") for el in document.elements}
    assert sources == {"native", "vlm"}
    native_elements = [el for el in document.elements if el.metadata.get("source") == "native"]
    assert native_elements[0].text == "a caption above a figure"


def test_route_page_routing_rule():
    assert route_page(PageLayout(has_figures=True, has_text_layer=True)) == "native_and_vlm"
    assert route_page(PageLayout(has_figures=False, has_text_layer=True)) == "native"
    assert route_page(PageLayout(has_figures=False, has_text_layer=False)) == "vlm_only"
    assert route_page(PageLayout(has_figures=True, has_text_layer=False)) == "vlm_only"


def test_route_page_fast_tier_forces_native_regardless_of_content():
    """tier="fast"면 표/그림이 있든, 텍스트 레이어가 아예 없든(스캔) 무조건
    native만 -- VLM 호출 자체를 하지 않는다는 게 핵심."""
    for layout in (
        PageLayout(has_figures=True, has_text_layer=True),
        PageLayout(has_figures=True, has_text_layer=False),
        PageLayout(has_figures=False, has_text_layer=False),
    ):
        assert route_page(layout, tier="fast") == "native"


def test_fast_tier_skips_vlm_even_with_image(engine):
    """엔진 레벨에서: 그림이 있는 페이지도 tier="fast"면 VLM이 아예 호출되지
    않아야 한다."""
    with patch("document_parser.parsing.loaders.pdf.graph.caption_figures") as mock_vlm:
        document = engine.parse("doc.pdf", data=_pdf_with_image(), tier="fast")

    mock_vlm.assert_not_called()
    sources = {el.metadata.get("source") for el in document.elements}
    assert sources == {"native"}


def test_invalid_tier_raises_value_error(engine):
    with pytest.raises(ValueError):
        engine.parse("doc.pdf", data=_text_only_pdf(), tier="ultra")


def test_page_graph_runs_native_and_vlm_in_parallel():
    """텍스트 레이어 있음 + 그림 있음 케이스에서 native/vlm 둘 다 실제로 같은
    super-step에서 병렬 실행되는지 확인 — layout을 직접 만들어 강제한다."""
    graph = build_page_graph().compile()

    fake_layout = PageLayout(has_figures=True, has_text_layer=True, boxes=[])

    with (
        patch("document_parser.parsing.loaders.pdf.graph.analyze_page", return_value=fake_layout),
        patch(
            "document_parser.parsing.loaders.pdf.graph.extract_native",
            return_value=[DocumentElement(text="n", metadata={"source": "native"})],
        ),
        patch(
            "document_parser.parsing.loaders.pdf.graph.caption_figures",
            return_value=[DocumentElement(text="v", metadata={"source": "vlm"})],
        ),
    ):
        result = graph.invoke(
            {"page": None, "plumber_page": None, "page_number": 1, "raw_elements": []}
        )

    sources = {el.metadata["source"] for el in result["elements"]}
    assert sources == {"native", "vlm"}


def test_page_graph_runs_vlm_and_vlm_text_in_parallel():
    """텍스트 레이어 없음(스캔 문서) 케이스에서 vlm(그림·표 캡션)과
    vlm_text(본문 전사)가 실제로 같은 super-step에서 병렬 실행되는지 확인."""
    graph = build_page_graph().compile()

    fake_layout = PageLayout(has_figures=True, has_text_layer=False, boxes=[])

    with (
        patch("document_parser.parsing.loaders.pdf.graph.analyze_page", return_value=fake_layout),
        patch(
            "document_parser.parsing.loaders.pdf.graph.caption_figures",
            return_value=[DocumentElement(text="v", metadata={"source": "vlm_figure"})],
        ),
        patch(
            "document_parser.parsing.loaders.pdf.graph.transcribe_text_boxes",
            return_value=[DocumentElement(text="t", metadata={"source": "vlm_text"})],
        ),
    ):
        result = graph.invoke({"page": None, "page_number": 1, "raw_elements": []})

    sources = {el.metadata["source"] for el in result["elements"]}
    assert sources == {"vlm_figure", "vlm_text"}


def test_merge_key_prefers_layout_order_over_bbox_position():
    """layout.py의 _reading_order_key와 같은 패턴: metadata["layout_order"]가
    있으면 bbox 위치보다 그걸 먼저 본다 — 다단(컬럼) 레이아웃처럼 위치만으론
    읽기 순서가 애매한 경우, PP-DocLayoutV2가 실제로 추론한 순서를 살리기
    위함."""
    from document_parser.parsing.loaders.pdf.graph import _merge_key

    # 위치상으론 B가 A보다 위(y0 더 작음)라 좌표만 보면 B가 먼저지만,
    # layout_order는 A=1, B=2로 A가 먼저라고 말한다 — order를 신뢰해야 한다.
    a = DocumentElement(
        text="A", bboxes=[BBox(x0=0, y0=100, x1=10, y1=110)], metadata={"layout_order": 1}
    )
    b = DocumentElement(
        text="B", bboxes=[BBox(x0=0, y0=10, x1=10, y1=20)], metadata={"layout_order": 2}
    )

    assert _merge_key(a) < _merge_key(b)


def test_merge_key_falls_back_to_bbox_position_without_layout_order():
    from document_parser.parsing.loaders.pdf.graph import _merge_key

    top = DocumentElement(text="top", bboxes=[BBox(x0=0, y0=10, x1=10, y1=20)], metadata={})
    bottom = DocumentElement(text="bottom", bboxes=[BBox(x0=0, y0=100, x1=10, y1=110)], metadata={})

    assert _merge_key(top) < _merge_key(bottom)


def test_merge_key_order_present_elements_always_sort_before_order_missing_ones():
    """실제 모델도 일부 박스는 order를 못 정해서(None) 위치 기준 폴백과 섞일
    수 있다 — order 있는 요소가 항상 먼저 오게 튜플을 묶어야 뒤섞이지 않는다
    (layout.py의 _reading_order_key와 동일한 근거)."""
    from document_parser.parsing.loaders.pdf.graph import _merge_key

    # 위치상 훨씬 위에 있어도(y0=0) order가 없으면, order=99인 요소보다 뒤로
    # 가야 한다.
    no_order = DocumentElement(
        text="no-order", bboxes=[BBox(x0=0, y0=0, x1=10, y1=10)], metadata={}
    )
    has_order = DocumentElement(
        text="has-order",
        bboxes=[BBox(x0=0, y0=500, x1=10, y1=510)],
        metadata={"layout_order": 99},
    )

    assert _merge_key(has_order) < _merge_key(no_order)


def test_page_graph_merge_respects_layout_order_across_native_and_vlm_paths():
    """실측 회귀 테스트: native가 뽑은 text_box와 vlm이 캡션한 crop_box가
    같은 페이지에서 병렬로 처리된 뒤 하나로 합쳐질 때, 좌표만 보면 순서가
    뒤바뀔 상황(그림이 텍스트보다 위에 있음)이라도 PP-DocLayoutV2가 준
    layout_order를 따라야 한다."""
    graph = build_page_graph().compile()

    fake_layout = PageLayout(has_figures=True, has_text_layer=True, boxes=[])

    # 그림(order=2)이 텍스트(order=1)보다 화면상 위에 있다 — 좌표 정렬만
    # 쓰면 그림이 먼저 나오겠지만, order를 따르면 텍스트가 먼저다.
    native_element = DocumentElement(
        type=ElementType.TEXT,
        text="본문 텍스트",
        bboxes=[BBox(x0=0, y0=500, x1=100, y1=520)],
        metadata={"source": "native", "layout_order": 1},
    )
    vlm_element = DocumentElement(
        type=ElementType.IMAGE,
        text="그림 캡션",
        bboxes=[BBox(x0=0, y0=10, x1=100, y1=200)],
        metadata={"source": "vlm", "layout_order": 2},
    )

    with (
        patch("document_parser.parsing.loaders.pdf.graph.analyze_page", return_value=fake_layout),
        patch(
            "document_parser.parsing.loaders.pdf.graph.extract_native",
            return_value=[native_element],
        ),
        patch(
            "document_parser.parsing.loaders.pdf.graph.caption_figures",
            return_value=[vlm_element],
        ),
    ):
        result = graph.invoke(
            {"page": None, "plumber_page": None, "page_number": 1, "raw_elements": []}
        )

    ordered_texts = [el.text for el in result["elements"]]
    assert ordered_texts == ["본문 텍스트", "그림 캡션"]


def _looks_tabular_text() -> str:
    return "| A | B |\n| --- | --- |\n| 1 | 2 |"


def test_merge_recovers_misclassified_table_using_native_text():
    """PP-DocLayoutV2가 "table"이라고 했는데 VLM이 돌려준 결과가 실제 표
    구조(마크다운 | 행)로 안 보이고, 그 영역에 native로 뽑을 수 있는 텍스트가
    꽤 있으면(표가 아니라 일반 텍스트를 표로 오분류한 것으로 보고) type을
    TABLE에서 TEXT로 되돌리고 원문을 text로 써야 한다 — 실측으로 발견한
    문제(페이지 전체가 표로 오분류돼서 본문이 VLM 요약으로 통째로 대체되며
    유실된 것)의 회귀 테스트. DI 없이, "VLM 결과가 표처럼 안 보인다"는 신호로
    판단한다(graph.py의 _looks_like_markdown_table)."""
    pdfplumber = pytest.importorskip("pdfplumber", reason="pdf extra not installed")

    doc = pymupdf.open()
    page = doc.new_page()
    # pymupdf 기본 폰트가 한글 글리프를 지원 안 해서(찍히면 전부 '·'로 나옴)
    # 영어로 씀 — 여기서 검증하려는 건 어차피 언어가 아니라 "native로 뽑을
    # 수 있는 텍스트 분량"이라 무관하다.
    long_text = "This clause is actually plain numbered text, not a table. " * 3
    page.insert_text((72, 72), long_text, fontsize=10)
    # 실제 복구 로직은 pdfplumber로 텍스트를 뽑으므로, 같은 PDF 바이트를
    # pdfplumber로도 열어서 진짜 plumber_page를 넘겨야 회귀 테스트가 유효하다.
    pdoc = pdfplumber.open(BytesIO(doc.tobytes()))
    plumber_page = pdoc.pages[0]

    graph = build_page_graph().compile()
    fake_layout = PageLayout(has_figures=True, has_text_layer=True, boxes=[])
    misclassified_element = DocumentElement(
        type=ElementType.TABLE,
        text="[VLM이 이 영역을 표로 착각하고 만든 요약/설명 시도 — 표처럼 안 보임]",
        summary="표로 착각한 요약",
        bboxes=[BBox(x0=0, y0=0, x1=595, y1=200)],
        metadata={"source": "vlm", "block_type": "table"},
    )

    with (
        patch("document_parser.parsing.loaders.pdf.graph.analyze_page", return_value=fake_layout),
        patch("document_parser.parsing.loaders.pdf.graph.extract_native", return_value=[]),
        patch(
            "document_parser.parsing.loaders.pdf.graph.caption_figures",
            return_value=[misclassified_element],
        ),
    ):
        result = graph.invoke(
            {"page": page, "plumber_page": plumber_page, "page_number": 1, "raw_elements": []}
        )

    elements = result["elements"]
    assert len(elements) == 1
    assert elements[0].type == ElementType.TEXT
    assert "plain numbered text" in elements[0].text
    assert elements[0].metadata.get("misclassified_as_table") is True
    pdoc.close()
    doc.close()


def test_merge_keeps_table_when_vlm_output_looks_tabular():
    """VLM이 돌려준 텍스트가 실제 마크다운 표 구조로 보이면(| 구분 행이 2줄
    이상), native 텍스트가 있어도 오분류 복구를 타지 않고 TABLE 그대로
    유지돼야 한다."""
    pdfplumber = pytest.importorskip("pdfplumber", reason="pdf extra not installed")

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "some native text near the table region", fontsize=10)
    pdoc = pdfplumber.open(BytesIO(doc.tobytes()))
    plumber_page = pdoc.pages[0]

    graph = build_page_graph().compile()
    fake_layout = PageLayout(has_figures=True, has_text_layer=True, boxes=[])
    real_table_element = DocumentElement(
        type=ElementType.TABLE,
        text=_looks_tabular_text(),
        summary="a real table",
        bboxes=[BBox(x0=0, y0=0, x1=595, y1=200)],
        metadata={"source": "vlm", "block_type": "table"},
    )

    with (
        patch("document_parser.parsing.loaders.pdf.graph.analyze_page", return_value=fake_layout),
        patch("document_parser.parsing.loaders.pdf.graph.extract_native", return_value=[]),
        patch(
            "document_parser.parsing.loaders.pdf.graph.caption_figures",
            return_value=[real_table_element],
        ),
    ):
        result = graph.invoke(
            {"page": page, "plumber_page": plumber_page, "page_number": 1, "raw_elements": []}
        )

    elements = result["elements"]
    assert len(elements) == 1
    assert elements[0].type == ElementType.TABLE
    assert elements[0].text == _looks_tabular_text()
    assert "misclassified_as_table" not in elements[0].metadata
    pdoc.close()
    doc.close()


def test_merge_attaches_nearby_paragraphs_to_table_element():
    """표 근처(수직 거리 60pt 이내, 최대 2개) TEXT/HEADING element의 텍스트가
    metadata["nearby_paragraphs"]에 붙어야 한다 — 청킹할 때 표만 뚝 떼서
    주는 것보다 문맥이 있는 게 낫다는 요청. AzureDI 없이, 같은 페이지에서
    native/vlm_text가 이미 만든 TEXT/HEADING element를 그대로 재사용해서
    판단한다(별도 문단 추출 없음)."""
    graph = build_page_graph().compile()

    fake_layout = PageLayout(has_figures=True, has_text_layer=True, boxes=[])
    vlm_table_element = DocumentElement(
        type=ElementType.TABLE,
        text=_looks_tabular_text(),
        bboxes=[BBox(x0=10, y0=100, x1=110, y1=200)],
        metadata={"source": "vlm", "block_type": "table"},
    )
    near_above = DocumentElement(
        type=ElementType.TEXT, text="바로 위 문단", bboxes=[BBox(x0=10, y0=50, x1=110, y1=90)]
    )
    near_below = DocumentElement(
        type=ElementType.HEADING,
        text="바로 아래 문단",
        bboxes=[BBox(x0=10, y0=210, x1=110, y1=250)],
    )
    far_away = DocumentElement(
        type=ElementType.TEXT, text="아주 먼 문단", bboxes=[BBox(x0=10, y0=800, x1=110, y1=850)]
    )

    with (
        patch("document_parser.parsing.loaders.pdf.graph.analyze_page", return_value=fake_layout),
        patch(
            "document_parser.parsing.loaders.pdf.graph.extract_native",
            return_value=[near_above, near_below, far_away],
        ),
        patch(
            "document_parser.parsing.loaders.pdf.graph.caption_figures",
            return_value=[vlm_table_element],
        ),
    ):
        result = graph.invoke(
            {"page": None, "plumber_page": None, "page_number": 1, "raw_elements": []}
        )

    table_elements = [el for el in result["elements"] if el.type == ElementType.TABLE]
    assert len(table_elements) == 1
    nearby = table_elements[0].metadata["nearby_paragraphs"]
    assert nearby == ["바로 위 문단", "바로 아래 문단"]
    assert "아주 먼 문단" not in nearby


def _mixed_content_pdf() -> bytes:
    """제목 + 본문 문단 + (그리드로 그린) 표 — 그림은 없음."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 60), "Document Title", fontsize=24)
    page.insert_text(
        (72, 110), "This is a body paragraph describing the page content.", fontsize=11
    )

    x0, y0, x1, y1, rows, cols = 72, 160, 500, 260, 4, 3
    for r in range(rows + 1):
        y = y0 + r * (y1 - y0) / rows
        page.draw_line((x0, y), (x1, y))
    for c in range(cols + 1):
        x = x0 + c * (x1 - x0) / cols
        page.draw_line((x, y0), (x, y1))
    for r in range(rows):
        for c in range(cols):
            page.insert_text(
                (x0 + c * (x1 - x0) / cols + 5, y0 + r * (y1 - y0) / rows + 20),
                f"R{r}C{c}",
                fontsize=9,
            )

    data = doc.tobytes()
    doc.close()
    return data


def _scanned_no_text_layer_pdf() -> bytes:
    """텍스트 레이어 없이 페이지 전체를 이미지로 채운 '스캔본' 흉내."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    pix = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 595, 842), False)
    pix.set_rect(pix.irect, (120, 140, 160))
    page.insert_image(page.rect, pixmap=pix)
    data = doc.tobytes()
    doc.close()
    return data


def _multi_figure_pdf() -> bytes:
    """텍스트 레이어(제목) + 그림 2개를 일부러 '아래쪽 먼저, 위쪽 나중'
    순서로 삽입 — 삽입 순서가 아니라 실제 위치로 병합되는지 확인하기 위함.
    텍스트 레이어가 있으므로 native+vlm 경로."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 60), "Report With Multiple Figures", fontsize=18)

    def colored_pix(rgb):
        p = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 300, 200), False)
        p.set_rect(p.irect, rgb)
        return p

    page.insert_image(pymupdf.Rect(72, 500, 372, 700), pixmap=colored_pix((10, 200, 10)))
    page.insert_image(pymupdf.Rect(72, 120, 372, 320), pixmap=colored_pix((200, 10, 10)))

    data = doc.tobytes()
    doc.close()
    return data


@requires_real_layout_model
def test_mixed_content_classified_with_real_block_types(engine):
    """제목/본문/표가 섞인 페이지에서 25개 카테고리 중 실제 라벨(paragraph_title/
    text/table)이 최종 출력까지 그대로 이어지는지, 그리고 위→아래 읽기
    순서로 나오는지 확인한다. 표는 vlm 경로를 타므로(native가 아니라) vlm은
    mock — 이 테스트의 목적은 레이아웃 모델의 실제 라벨/순서 분류이지 vlm
    라이브 호출 자체가 아니다. 표로 인식된 박스에는 실제 마크다운 표처럼
    보이는 텍스트를 돌려줘야(오분류 복구 로직이 타지 않게) "표 라벨이 끝까지
    유지되는지"를 제대로 볼 수 있다."""

    def fake_caption_figures(page, page_number, boxes):
        return [
            DocumentElement(
                type=ElementType.TABLE if b.label == "table" else ElementType.IMAGE,
                text=_looks_tabular_text() if b.label == "table" else "fake vlm output",
                page=page_number,
                bboxes=[BBox(x0=b.bbox[0], y0=b.bbox[1], x1=b.bbox[2], y1=b.bbox[3])],
                metadata={
                    "source": "vlm",
                    "block_type": b.label,
                    "layout_cls_id": b.cls_id,
                    "layout_box_index": b.box_index,
                },
            )
            for b in boxes
        ]

    with patch(
        "document_parser.parsing.loaders.pdf.graph.caption_figures",
        side_effect=fake_caption_figures,
    ):
        document = engine.parse("mixed.pdf", data=_mixed_content_pdf())

    labels = [el.metadata.get("block_type") for el in document.elements]
    types = [el.type for el in document.elements]

    assert "paragraph_title" in labels
    assert "table" in labels
    assert ElementType.HEADING in types
    assert ElementType.TABLE in types

    # 읽기 순서(제목 → 본문 → 표) 확인
    y_positions = [el.bboxes[0].y0 for el in document.elements if el.bboxes]
    assert y_positions == sorted(y_positions)

    # cls_id/box_index가 원본 PP-DocLayoutV2 출력과 대조 가능하게 남아있는지
    cls_ids = [el.metadata.get("layout_cls_id") for el in document.elements]
    box_indices = [el.metadata.get("layout_box_index") for el in document.elements]
    assert all(isinstance(v, int) for v in cls_ids)
    assert all(isinstance(v, int) for v in box_indices)
    assert len(box_indices) == len(set(box_indices))  # 페이지 안에서 유일


def test_scanned_page_without_text_layer_routes_to_vlm_only(engine):
    """텍스트 레이어가 없는 페이지(스캔본 흉내)도 에러 없이 vlm_only 경로를
    타는지 확인 — 25개 카테고리 모델 없이도(pymupdf 휴리스틱만으로도)
    성립해야 한다(휴리스틱 폴백은 감지된 영역을 전부 "image" 라벨로 주므로
    crop_boxes만 채워지고 text_boxes는 빈다). VLM은 실제 호출로 바뀌어서
    (자격증명 필요) 여기선 routing/병합 로직만 보는 게 목적이라 mock한다."""
    fake_vlm_element = DocumentElement(
        type=ElementType.IMAGE, text="from vlm", page=1, metadata={"source": "vlm"}
    )
    with patch(
        "document_parser.parsing.loaders.pdf.graph.caption_figures",
        return_value=[fake_vlm_element],
    ):
        document = engine.parse("scanned.pdf", data=_scanned_no_text_layer_pdf())

    assert len(document.elements) >= 1
    assert all(el.metadata.get("source") == "vlm" for el in document.elements)


@requires_real_layout_model
def test_multiple_figures_merge_in_reading_order_not_insertion_order(engine):
    """그림이 여러 개일 때, PDF에 삽입된 순서가 아니라 실제 페이지상 위치
    (위→아래) 기준으로 병합되는지 확인한다. 텍스트 레이어 + 그림이 있으므로
    native+vlm 경로를 타는데, vlm은 mock하고 native는 실제로 동작하게 둔다."""

    def fake_caption_figures(page, page_number, boxes):
        return [
            DocumentElement(
                type=ElementType.IMAGE,
                text="a caption",
                page=page_number,
                bboxes=[BBox(x0=b.bbox[0], y0=b.bbox[1], x1=b.bbox[2], y1=b.bbox[3])],
                metadata={"source": "vlm", "block_type": b.label},
            )
            for b in boxes
        ]

    with patch(
        "document_parser.parsing.loaders.pdf.graph.caption_figures",
        side_effect=fake_caption_figures,
    ):
        document = engine.parse("multi_fig.pdf", data=_multi_figure_pdf())

    vlm_elements = [el for el in document.elements if el.metadata.get("source") == "vlm"]
    assert len(vlm_elements) == 2

    # 전체 요소(제목 포함)가 위→아래 순서로 병합됐는지 확인
    y_positions = [el.bboxes[0].y0 for el in document.elements if el.bboxes]
    assert y_positions == sorted(y_positions)


@requires_real_vlm
def test_vlm_real_call(engine):
    """DATABRICKS_HOST/DATABRICKS_TOKEN이 실제로 설정돼 있을 때만 도는 라이브
    검증 — in4u Databricks AI Gateway(Claude Sonnet 4.6)에 진짜 크롭 이미지를
    보내 캡션이 정상적으로 돌아오는지 확인한다."""
    document = engine.parse("doc.pdf", data=_pdf_with_image())

    vlm_elements = [el for el in document.elements if el.metadata.get("source") == "vlm"]
    assert len(vlm_elements) >= 1
    assert vlm_elements[0].text  # 실제 Claude Sonnet 4.6이 생성한 캡션
    assert len(vlm_elements[0].bboxes) >= 1


@requires_real_vlm
def test_vlm_text_transcription_real_call(engine):
    """스캔 페이지(텍스트 레이어 없음) 본문이 실제 VLM 전사로 뽑히는지 —
    AzureDI가 하던 역할을 VLM이 대신하는 경로의 라이브 검증."""
    document = engine.parse("scanned_ocr.pdf", data=_scanned_pdf_with_text())

    vlm_elements = [el for el in document.elements if el.metadata.get("source") == "vlm"]
    assert len(vlm_elements) >= 1
    assert any("scanned document text" in el.text.lower() for el in vlm_elements)


def test_extract_mermaid_pulls_out_fenced_code_block():
    """content 자체가 mermaid 소스가 돼야 한다(요청: "text: ... mermaid 반환") —
    두 반환값(text로 쓸 것, metadata용) 다 mermaid 소스 그 자체."""
    from document_parser.parsing.loaders.vlm_caption import extract_mermaid as _extract_mermaid

    response = "```mermaid\ngraph TD;\nA-->B;\n```"
    text, mermaid = _extract_mermaid(response)

    assert mermaid == "graph TD;\nA-->B;"
    assert text == "graph TD;\nA-->B;"


def test_extract_mermaid_returns_none_for_plain_caption():
    from document_parser.parsing.loaders.vlm_caption import extract_mermaid as _extract_mermaid

    text, mermaid = _extract_mermaid("그냥 평범한 사진 설명입니다.")

    assert mermaid is None
    assert text == "그냥 평범한 사진 설명입니다."


def test_caption_figures_extracts_latex_for_formula_labels(monkeypatch):
    """display_formula/inline_formula/formula_number 라벨이면 표/일반
    이미지와 다른 프롬프트(LaTeX 요청)를 타고, text가 LaTeX 소스 그 자체가
    되면서 metadata["latex"]에도 같은 값이 남는지(mermaid와 동일 패턴)."""
    from document_parser.parsing.clients.vlm import VLMCaptionResult
    from document_parser.parsing.loaders.pdf.layout import LayoutBox
    from document_parser.parsing.loaders.pdf.vlm import caption_figures

    fake_client = MagicMock()
    fake_client.caption_image.return_value = VLMCaptionResult(
        text="[CONTENT]\nE = mc^2\n[SUMMARY]\n질량-에너지 등가 공식.",
        usage=None,
    )
    monkeypatch.setattr("document_parser.parsing.loaders.pdf.vlm.get_client", lambda: fake_client)

    fake_pix = MagicMock()
    fake_pix.tobytes.return_value = b"fake-png-bytes"
    fake_page = MagicMock()
    fake_page.get_pixmap.return_value = fake_pix

    box = LayoutBox(label="display_formula", bbox=(0.0, 0.0, 10.0, 10.0))

    elements = caption_figures(fake_page, 1, [box])

    assert len(elements) == 1
    element = elements[0]
    assert element.type == ElementType.IMAGE
    assert element.text == "E = mc^2"
    assert element.summary == "질량-에너지 등가 공식."
    assert element.metadata["latex"] == "E = mc^2"
    assert element.metadata["block_type"] == "display_formula"


def test_transcribe_text_boxes_produces_heading_and_text_without_summary(monkeypatch):
    """스캔 페이지 본문 전사 — 라벨에 따라 HEADING/TEXT로 분류되고(native.py의
    _label_to_element_type 재사용), summary 개념이 없으므로 항상 None이어야
    한다."""
    from document_parser.parsing.clients.vlm import VLMCaptionResult
    from document_parser.parsing.loaders.pdf.layout import LayoutBox
    from document_parser.parsing.loaders.pdf.vlm import transcribe_text_boxes

    fake_client = MagicMock()
    fake_client.caption_image.return_value = VLMCaptionResult(text="Transcribed line", usage=None)
    monkeypatch.setattr("document_parser.parsing.loaders.pdf.vlm.get_client", lambda: fake_client)

    fake_pix = MagicMock()
    fake_pix.tobytes.return_value = b"fake-png-bytes"
    fake_page = MagicMock()
    fake_page.get_pixmap.return_value = fake_pix

    boxes = [
        LayoutBox(label="paragraph_title", bbox=(0.0, 0.0, 10.0, 10.0)),
        LayoutBox(label="text", bbox=(0.0, 20.0, 10.0, 30.0)),
    ]

    elements = transcribe_text_boxes(fake_page, 1, boxes)

    assert len(elements) == 2
    assert elements[0].type == ElementType.HEADING
    assert elements[1].type == ElementType.TEXT
    assert all(el.text == "Transcribed line" for el in elements)
    assert all(el.summary is None for el in elements)


def test_looks_like_markdown_table():
    from document_parser.parsing.loaders.pdf.graph import _looks_like_markdown_table

    assert _looks_like_markdown_table("| A | B |\n| --- | --- |\n| 1 | 2 |") is True
    assert _looks_like_markdown_table("just a plain paragraph, no table here.") is False
    assert _looks_like_markdown_table("| only one pipe line |") is False


def test_flag_continued_tables_marks_table_immediately_following_table_on_next_page():
    """직전 요소도 표이고 페이지 번호가 정확히 하나 차이면 이어지는 표로 본다 —
    헤더 유무가 아니라 위치+페이지 인접만으로 판단한다(헤더 신호는 VLM 프롬프트가
    항상 헤더 형태를 강제해서 신뢰할 수 없음)."""
    from document_parser.parsing.loaders.pdf import _flag_continued_tables

    table_page1 = DocumentElement(type=ElementType.TABLE, text="| a |\n| --- |\n| 1 |", page=1)
    table_page2 = DocumentElement(type=ElementType.TABLE, text="| b |\n| --- |\n| 2 |", page=2)

    result = _flag_continued_tables([table_page1, table_page2])

    assert "continued_from_previous_page" not in result[0].metadata
    assert result[1].metadata["continued_from_previous_page"] is True


def test_flag_continued_tables_ignores_non_adjacent_or_non_table_predecessor():
    from document_parser.parsing.loaders.pdf import _flag_continued_tables

    text_page1 = DocumentElement(type=ElementType.TEXT, text="intro", page=1)
    table_page2 = DocumentElement(type=ElementType.TABLE, text="| a |\n| --- |\n| 1 |", page=2)
    table_page4 = DocumentElement(type=ElementType.TABLE, text="| b |\n| --- |\n| 2 |", page=4)

    # 직전 요소가 표가 아님 -> 안 붙음
    result = _flag_continued_tables([text_page1, table_page2])
    assert "continued_from_previous_page" not in result[1].metadata

    # 표는 맞지만 페이지가 한 장 넘게 떨어짐 -> 안 붙음
    result = _flag_continued_tables([table_page2, table_page4])
    assert "continued_from_previous_page" not in result[1].metadata


def test_flag_continued_tables_skips_repeated_running_header_and_page_number():
    """매 페이지 반복되는 러닝헤더/대제목(pptx 데크 제목처럼) 때문에 "다음
    페이지의 진짜 첫 요소가 표"라는 게 가려지면 안 된다 — 상투 문구와 페이지
    번호("number" 라벨)는 건너뛰고 판단해야 한다."""
    from document_parser.parsing.loaders.pdf import _flag_continued_tables

    def page(n: int) -> list[DocumentElement]:
        return [
            DocumentElement(type=ElementType.HEADING, text="표준파서 스펙 정의서", page=n),
            DocumentElement(type=ElementType.TEXT, text="v1 · Draft", page=n),
        ]

    table1 = DocumentElement(type=ElementType.TABLE, text="| a |\n| --- |\n| 1 |", page=1)
    footer1 = DocumentElement(
        type=ElementType.TEXT, text="1 / 5", page=1, metadata={"block_type": "number"}
    )
    table2 = DocumentElement(type=ElementType.TABLE, text="| b |\n| --- |\n| 2 |", page=2)

    elements = page(1) + [table1, footer1] + page(2) + [table2] + page(3)
    result = _flag_continued_tables(elements)

    table2_result = next(el for el in result if el.text == table2.text)
    assert table2_result.metadata["continued_from_previous_page"] is True


def test_extract_native_captures_font_size_for_headings():
    """제목(HEADING) 카테고리 박스는 metadata["font_size"]를 같이 남겨야 한다
    — 번호 매김 없는 제목(PPT 등)의 레벨을 상대 크기로 역산하는 데 쓴다
    (pdf/__init__.py의 _assign_heading_levels 참고). TEXT 카테고리는 안 남김
    (헤더 레벨 추정에만 필요, 불필요한 계산 안 함)."""
    pdfplumber = pytest.importorskip("pdfplumber", reason="pdf extra not installed")
    from document_parser.parsing.loaders.pdf.layout import LayoutBox
    from document_parser.parsing.loaders.pdf.native import extract_native

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "A Big Title", fontsize=24)
    page.insert_text((72, 120), "some body text", fontsize=10)
    pdoc = pdfplumber.open(BytesIO(doc.tobytes()))
    plumber_page = pdoc.pages[0]

    layout = PageLayout(
        has_figures=False,
        has_text_layer=True,
        boxes=[
            LayoutBox(label="paragraph_title", bbox=(65, 60, 250, 90)),
            LayoutBox(label="text", bbox=(65, 108, 250, 135)),
        ],
    )

    elements = extract_native(plumber_page, 1, layout)

    heading = next(el for el in elements if el.type == ElementType.HEADING)
    text = next(el for el in elements if el.type == ElementType.TEXT)
    assert heading.metadata["font_size"] == pytest.approx(24, abs=1)
    assert "font_size" not in text.metadata
    pdoc.close()
    doc.close()


def test_numbering_level_parses_arabic_and_korean_patterns():
    from document_parser.parsing.loaders.pdf import _numbering_level

    assert _numbering_level("1. 개요") == 1
    assert _numbering_level("1.1 핵심 가치") == 2
    assert _numbering_level("4.2.1 세부 항목") == 3
    # "03"이 부모, "03.2"가 하위 항목(실측: test1.pptx)
    assert _numbering_level("03.2 개발 비용") == 2
    assert _numbering_level("가. 세부사항") == 2
    assert _numbering_level("나) 다른 항목") == 2
    assert _numbering_level("CHAPTER") is None
    assert _numbering_level("Overview") is None


def test_assign_heading_levels_prefers_numbering_over_font_size():
    from document_parser.parsing.loaders.pdf import _assign_heading_levels

    h1 = DocumentElement(type=ElementType.HEADING, text="1. 개요", metadata={"font_size": 18.0})
    h2 = DocumentElement(
        type=ElementType.HEADING, text="1.1 핵심 가치", metadata={"font_size": 18.0}
    )

    result = _assign_heading_levels([h1, h2])

    assert result[0].metadata["level"] == 1
    assert result[1].metadata["level"] == 2


def test_assign_heading_levels_falls_back_to_font_size_rank_when_no_numbering():
    """번호 매김이 없으면(PPT 슬라이드 제목 등) 문서 전체 제목 폰트 크기의
    상대 순위로 레벨을 매긴다 — 큰 글씨일수록 상위 레벨."""
    from document_parser.parsing.loaders.pdf import _assign_heading_levels

    big = DocumentElement(type=ElementType.HEADING, text="Overview", metadata={"font_size": 28.0})
    medium = DocumentElement(type=ElementType.HEADING, text="Details", metadata={"font_size": 20.0})
    small = DocumentElement(
        type=ElementType.HEADING, text="Sub-details", metadata={"font_size": 14.0}
    )

    result = _assign_heading_levels([big, medium, small])

    assert result[0].metadata["level"] == 1
    assert result[1].metadata["level"] == 2
    assert result[2].metadata["level"] == 3


def test_assign_heading_levels_doc_title_without_numbering_is_level_one():
    """doc_title 카테고리인데 번호 매김이 없으면 레벨 1 — 문서 전체 제목이라는
    걸 PP-DocLayoutV2가 이미 알려주기 때문."""
    from document_parser.parsing.loaders.pdf import _assign_heading_levels

    doc_title = DocumentElement(
        type=ElementType.HEADING,
        text="어쩌다 작은 폰트로 찍힌 대제목",
        metadata={"font_size": 10.0, "block_type": "doc_title"},
    )
    bigger_heading = DocumentElement(
        type=ElementType.HEADING, text="Section", metadata={"font_size": 30.0}
    )

    result = _assign_heading_levels([doc_title, bigger_heading])

    assert result[0].metadata["level"] == 1


def test_assign_heading_levels_numbering_overrides_misclassified_doc_title():
    """실측(test1.pptx) 회귀 테스트: PP-DocLayoutV2가 그냥 평범한 번호 매겨진
    슬라이드 제목("02.2 ...")을 doc_title로 잘못 분류한 사례가 있었다 — 번호
    매김이 명시적으로 있으면 doc_title 라벨보다 그걸 더 신뢰해야 한다."""
    from document_parser.parsing.loaders.pdf import _assign_heading_levels

    misclassified = DocumentElement(
        type=ElementType.HEADING,
        text="02.2 (3.2.5) 문서 파싱·구조화_Excel 인식·구조화",
        metadata={"font_size": 17.0, "block_type": "doc_title"},
    )

    result = _assign_heading_levels([misclassified])

    assert result[0].metadata["level"] == 2


def test_assign_heading_levels_font_size_fallback_clusters_and_caps():
    """실측(test1.pptx) 회귀 테스트: 30장짜리 덱에 서로 다른 폰트 크기가
    20개 넘게 나오면, 그냥 순위를 매기던 예전 로직은 레벨이 13~17까지
    치솟았다 — 가까운 크기는 같은 레벨로 묶고(군집), 최종 레벨도 상한을
    둬야 한다."""
    from document_parser.parsing.loaders.pdf import _assign_heading_levels

    def heading(text: str, font_size: float) -> DocumentElement:
        return DocumentElement(
            type=ElementType.HEADING, text=text, metadata={"font_size": font_size}
        )

    # 55.1/48.0는 서로 멀리 떨어져 있어 별개 레벨, 17.0도 별개 레벨이지만
    # 12.0/11.0/10.0/8.0은 서로 3pt 이내로 다닥다닥 붙어 있어 같은 레벨로
    # 묶여야 한다(실측 값 그대로).
    elements = [
        heading("A", 55.1),
        heading("B", 48.0),
        heading("C", 17.0),
        heading("D", 12.0),
        heading("E", 11.0),
        heading("F", 10.0),
        heading("G", 8.0),
    ]

    result = _assign_heading_levels(elements)
    levels = {el.text: el.metadata["level"] for el in result}

    assert levels["A"] == 1
    assert levels["B"] == 2
    assert levels["C"] == 3
    assert levels["D"] == levels["E"] == levels["F"] == levels["G"] == 4
    assert max(levels.values()) <= 5
