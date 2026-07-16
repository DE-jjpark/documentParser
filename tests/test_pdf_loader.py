"""pdf 로더의 페이지별 라우팅(native / native+vlm / azure_di+vlm) 테스트.

fixture 파일 대신 pymupdf로 그 자리에서 작은 합성 PDF를 만든다(기존
text 로더 테스트의 "hello world" 스타일과 동일).

layout.analyze_page()는 실제로(설치된 것 기준: 'layout' extra + 가중치가
있으면 진짜 PP-DocLayoutV2, 없으면 pymupdf 휴리스틱 폴백) 실행해서 검증한다
— 둘 다 이 합성 페이지들을 올바르게 분류한다. azure_di/vlm은 graph.py가
호출하는 지점에서 mock한다 — 실제 자격증명이 이 환경에 없어서다.

라우팅 규칙(리뷰 피드백으로 수정, layout.route_page 참고):
  - 텍스트 레이어 있음 + 그림 없음  -> native만
  - 텍스트 레이어 있음 + 그림 있음  -> native(텍스트) + vlm(그림 캡션) — AzureDI는 안 탐
  - 텍스트 레이어 없음(스캔 문서)   -> azure_di(페이지 전체) + vlm(그림 있으면)
"""

import os
from unittest.mock import patch

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

requires_real_azure_di = pytest.mark.skipif(
    not (
        os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        and os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    ),
    reason="AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT / _KEY 환경변수 필요 (실제 in4u 리소스 호출)",
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
    """텍스트 레이어 + 그림 — 텍스트 레이어가 있으므로 native+vlm 경로를
    타야 한다(azure_di는 안 탐)."""
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
    실제로 읽을 수 있는 텍스트가 있어서 AzureDI가 진짜로 인식할 게 있다."""
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


def test_page_with_figure_and_text_layer_takes_native_and_vlm_not_azure_di(engine):
    """텍스트 레이어가 있는 페이지에 그림이 있어도 AzureDI는 타면 안 된다
    (native가 이미 텍스트를 정확히 뽑을 수 있어서) — 리뷰 피드백으로 고친
    라우팅 규칙의 핵심 케이스."""
    fake_vlm_element = DocumentElement(
        type=ElementType.IMAGE,
        text="from vlm",
        page=1,
        bboxes=[BBox(x0=0, y0=0, x1=1, y1=1)],
        metadata={"source": "vlm"},
    )

    with (
        patch("document_parser.parsing.loaders.pdf.graph.extract_with_azure_di") as mock_azure,
        patch(
            "document_parser.parsing.loaders.pdf.graph.caption_figures",
            return_value=[fake_vlm_element],
        ) as mock_vlm,
    ):
        document = engine.parse("doc.pdf", data=_pdf_with_image())

    mock_azure.assert_not_called()
    mock_vlm.assert_called_once()

    sources = {el.metadata.get("source") for el in document.elements}
    assert sources == {"native", "vlm"}
    native_elements = [el for el in document.elements if el.metadata.get("source") == "native"]
    assert native_elements[0].text == "a caption above a figure"


def test_route_page_routing_rule():
    assert route_page(PageLayout(has_figures=True, has_text_layer=True)) == "native_and_vlm"
    assert route_page(PageLayout(has_figures=False, has_text_layer=True)) == "native"
    assert route_page(PageLayout(has_figures=False, has_text_layer=False)) == "azure_di_and_vlm"
    assert route_page(PageLayout(has_figures=True, has_text_layer=False)) == "azure_di_and_vlm"


def test_page_graph_runs_native_and_vlm_in_parallel():
    """텍스트 레이어 있음 + 그림 있음 케이스에서 native와 vlm이 실제로 같은
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
        result = graph.invoke({"page": None, "page_number": 1, "raw_elements": []})

    sources = {el.metadata["source"] for el in result["elements"]}
    assert sources == {"native", "vlm"}


def test_page_graph_runs_azure_di_and_vlm_in_parallel():
    """텍스트 레이어 없음(스캔 문서) 케이스에서 azure_di와 vlm이 실제로 같은
    super-step에서 병렬 실행되는지 확인."""
    graph = build_page_graph().compile()

    fake_layout = PageLayout(has_figures=True, has_text_layer=False, boxes=[])

    with (
        patch("document_parser.parsing.loaders.pdf.graph.analyze_page", return_value=fake_layout),
        patch(
            "document_parser.parsing.loaders.pdf.graph.extract_with_azure_di",
            return_value=[DocumentElement(text="a", metadata={"source": "azure_di"})],
        ),
        patch(
            "document_parser.parsing.loaders.pdf.graph.caption_figures",
            return_value=[DocumentElement(text="v", metadata={"source": "vlm"})],
        ),
    ):
        result = graph.invoke({"page": None, "page_number": 1, "raw_elements": []})

    sources = {el.metadata["source"] for el in result["elements"]}
    assert sources == {"azure_di", "vlm"}


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
    텍스트 레이어가 있으므로 native+vlm 경로(azure_di는 안 탐)."""
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
def test_mixed_content_classified_with_real_layout_labels(engine):
    """제목/본문/표가 섞인 페이지에서 25개 카테고리 중 실제 라벨(paragraph_title/
    text/table)이 최종 출력까지 그대로 이어지는지, 그리고 위→아래 읽기
    순서로 나오는지 확인한다."""
    document = engine.parse("mixed.pdf", data=_mixed_content_pdf())

    labels = [el.metadata.get("layout_label") for el in document.elements]
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


def test_scanned_page_without_text_layer_routes_to_azure_di_and_vlm(engine):
    """텍스트 레이어가 없는 페이지(스캔본 흉내)도 에러 없이 AzureDI+VLM
    경로를 타는지 확인 — 25개 카테고리 모델 없이도(pymupdf 휴리스틱만으로도)
    성립해야 한다. AzureDI는 실제 호출로 바뀌어서(자격증명 필요) 여기선
    routing/병합 로직만 보는 게 목적이라 mock한다."""
    fake_azure_element = DocumentElement(
        type=ElementType.TEXT, text="from azure", page=1, metadata={"source": "azure_di"}
    )
    with patch(
        "document_parser.parsing.loaders.pdf.graph.extract_with_azure_di",
        return_value=[fake_azure_element],
    ):
        document = engine.parse("scanned.pdf", data=_scanned_no_text_layer_pdf())

    assert len(document.elements) >= 1
    assert all(el.metadata.get("source") in {"azure_di", "vlm"} for el in document.elements)


@requires_real_layout_model
def test_multiple_figures_merge_in_reading_order_not_insertion_order(engine):
    """그림이 여러 개일 때, PDF에 삽입된 순서가 아니라 실제 페이지상 위치
    (위→아래) 기준으로 병합되는지 확인한다. 텍스트 레이어가 있으므로
    native+vlm 경로(azure_di는 안 탐) — vlm만 mock, native는 실제 동작."""

    def fake_caption_figures(page, page_number, boxes):
        return [
            DocumentElement(
                type=ElementType.IMAGE,
                text="a caption",
                page=page_number,
                bboxes=[BBox(x0=b.bbox[0], y0=b.bbox[1], x1=b.bbox[2], y1=b.bbox[3])],
                metadata={"source": "vlm", "layout_label": b.label},
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


@requires_real_azure_di
def test_azure_di_real_call(engine):
    """AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT/_KEY가 실제로 설정돼 있을 때만
    도는 라이브 검증 — in4u Document Intelligence 리소스(rnd-skep-commpf-di)에
    진짜 호출을 보내 텍스트/bbox가 정상적으로 돌아오는지 확인한다. 자격증명이
    없는 환경(CI 등)에서는 그냥 skip된다.

    텍스트 레이어가 없는 페이지(_scanned_pdf_with_text)를 써야 한다 —
    텍스트 레이어가 있으면 라우팅 규칙상 AzureDI를 아예 안 타기 때문이다."""
    with patch(
        "document_parser.parsing.loaders.pdf.graph.caption_figures",
        return_value=[],
    ):
        document = engine.parse("scanned_ocr.pdf", data=_scanned_pdf_with_text())

    azure_elements = [el for el in document.elements if el.metadata.get("source") == "azure_di"]
    assert len(azure_elements) >= 1
    assert azure_elements[0].text  # "Scanned document text for OCR test."에 해당하는 실제 인식 결과
    assert len(azure_elements[0].bboxes) >= 1


@requires_real_vlm
def test_vlm_real_call(engine):
    """DATABRICKS_HOST/DATABRICKS_TOKEN이 실제로 설정돼 있을 때만 도는 라이브
    검증 — in4u Databricks AI Gateway(Claude Sonnet 4.6)에 진짜 크롭 이미지를
    보내 캡션이 정상적으로 돌아오는지 확인한다. _pdf_with_image()는 텍스트
    레이어가 있어 native+vlm 경로를 타므로(azure_di는 아예 안 불림) 별도
    mock이 필요 없다."""
    document = engine.parse("doc.pdf", data=_pdf_with_image())

    vlm_elements = [el for el in document.elements if el.metadata.get("source") == "vlm"]
    assert len(vlm_elements) >= 1
    assert vlm_elements[0].text  # 실제 Claude Sonnet 4.6이 생성한 캡션
    assert len(vlm_elements[0].bboxes) >= 1
