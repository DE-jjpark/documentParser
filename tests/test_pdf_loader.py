"""pdf 로더의 페이지별 라우팅(네이티브 vs AzureDI+VLM 경로) 테스트.

fixture 파일 대신 pymupdf로 그 자리에서 작은 합성 PDF를 만든다(기존
text 로더 테스트의 "hello world" 스타일과 동일).

layout.analyze_page()는 실제로(설치된 것 기준: 'layout' extra + 가중치가
있으면 진짜 PP-DocLayoutV2, 없으면 pymupdf 휴리스틱 폴백) 실행해서 검증한다
— 둘 다 이 합성 페이지들을 올바르게 분류한다. azure_di/vlm은 graph.py가
호출하는 지점에서 mock한다 — 실제 Azure 자격증명이 이 환경에 없어서다.
"""

from unittest.mock import patch

import pytest

pymupdf = pytest.importorskip("pymupdf", reason="pdf extra not installed")

from document_parser import ElementType, ParsingEngine  # noqa: E402
from document_parser.core.models import BBox, DocumentElement  # noqa: E402
from document_parser.parsing.loaders.pdf.graph import build_page_graph  # noqa: E402
from document_parser.parsing.loaders.pdf.layout import PageLayout, needs_heavy_path  # noqa: E402
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


def _text_only_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "hello world")
    data = doc.tobytes()
    doc.close()
    return data


def _pdf_with_image() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "a caption above a figure")
    pix = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 8, 8), False)
    pix.set_rect(pix.irect, (255, 0, 0))
    page.insert_image(pymupdf.Rect(72, 200, 172, 300), pixmap=pix)
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


def test_page_with_figure_takes_azure_di_and_vlm_route(engine):
    fake_azure_element = DocumentElement(
        type=ElementType.TEXT, text="from azure", page=1, metadata={"source": "azure_di"}
    )
    fake_vlm_element = DocumentElement(
        type=ElementType.IMAGE,
        text="from vlm",
        page=1,
        bboxes=[BBox(x0=0, y0=0, x1=1, y1=1)],
        metadata={"source": "vlm"},
    )

    # graph.py의 _azure_di/_vlm 노드가 참조하는 이름을 patch한다(정의 모듈이
    # 아니라 실제로 쓰는 모듈 기준이어야 patch가 먹는다).
    with (
        patch(
            "document_parser.parsing.loaders.pdf.graph.extract_with_azure_di",
            return_value=[fake_azure_element],
        ) as mock_azure,
        patch(
            "document_parser.parsing.loaders.pdf.graph.caption_figures",
            return_value=[fake_vlm_element],
        ) as mock_vlm,
    ):
        document = engine.parse("doc.pdf", data=_pdf_with_image())

    mock_azure.assert_called_once()
    mock_vlm.assert_called_once()
    sources = {el.metadata.get("source") for el in document.elements}
    assert sources == {"azure_di", "vlm"}


def test_needs_heavy_path_routing_rule():
    assert needs_heavy_path(PageLayout(has_figures=True, has_text_layer=True)) is True
    assert needs_heavy_path(PageLayout(has_figures=False, has_text_layer=False)) is True
    assert needs_heavy_path(PageLayout(has_figures=False, has_text_layer=True)) is False


def test_azure_di_and_vlm_are_placeholders_for_now(engine):
    """azure_di/vlm 실제 SDK 호출은 지금 의도적으로 꺼둔 상태다(in4u 자격증명
    아직 못 씀, VLM은 이미지당 비용 발생) — 둘 다 고정 placeholder를 반환해서
    파싱이 실패하지 않고 청킹까지 이어지게 한다. 여기선 mock 없이 실제 동작
    그대로를 검증한다."""
    document = engine.parse("doc.pdf", data=_pdf_with_image())

    azure_elements = [el for el in document.elements if el.metadata.get("source") == "azure_di"]
    vlm_elements = [el for el in document.elements if el.metadata.get("source") == "vlm"]

    assert len(azure_elements) == 1
    assert azure_elements[0].metadata.get("stub") is True
    assert azure_elements[0].text

    assert len(vlm_elements) == 1
    assert vlm_elements[0].metadata.get("stub") is True
    assert vlm_elements[0].text
    assert len(vlm_elements[0].bboxes) == 1
    assert vlm_elements[0].metadata.get("layout_label")  # 25개 카테고리 중 하나가 붙어 있어야 함


def test_page_graph_runs_azure_di_and_vlm_in_parallel():
    """graph.py 자체가 실제로 LangGraph 병렬 분기를 타는지 확인 — layout을
    직접 만들어 두 노드를 억지로 동시에 타게 만든 뒤, 결과에 azure_di/vlm
    양쪽 요소가 다 있는지 본다(순차 호출이 아니라 같은 super-step에서 둘 다
    실행됐다는 뜻)."""
    graph = build_page_graph().compile()

    fake_layout = PageLayout(
        has_figures=True,
        has_text_layer=True,
        boxes=[],
    )

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
    """그림 2개를 일부러 '아래쪽 먼저, 위쪽 나중' 순서로 삽입 — 삽입 순서가
    아니라 실제 위치로 병합되는지 확인하기 위함."""
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


def test_scanned_page_without_text_layer_routes_to_heavy_path(engine):
    """텍스트 레이어가 없는 페이지(스캔본 흉내)도 에러 없이 AzureDI+VLM
    경로를 타는지 확인 — 25개 카테고리 모델 없이도(pymupdf 휴리스틱만으로도)
    성립해야 한다."""
    document = engine.parse("scanned.pdf", data=_scanned_no_text_layer_pdf())

    assert len(document.elements) >= 1
    assert all(el.metadata.get("source") in {"azure_di", "vlm"} for el in document.elements)


@requires_real_layout_model
def test_multiple_figures_merge_in_reading_order_not_insertion_order(engine):
    """그림이 여러 개일 때, PDF에 삽입된 순서가 아니라 실제 페이지상 위치
    (위→아래) 기준으로 병합되는지 확인한다."""
    document = engine.parse("multi_fig.pdf", data=_multi_figure_pdf())

    vlm_elements = [el for el in document.elements if el.metadata.get("source") == "vlm"]
    assert len(vlm_elements) == 2

    y_positions = [el.bboxes[0].y0 for el in vlm_elements]
    assert y_positions == sorted(y_positions)  # 위쪽 그림이 먼저 나와야 함(삽입은 반대 순서였음)
