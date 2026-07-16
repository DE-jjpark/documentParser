"""pdf 로더의 페이지별 라우팅(native / native+azure_di+vlm / azure_di+vlm) 테스트.

fixture 파일 대신 pymupdf로 그 자리에서 작은 합성 PDF를 만든다(기존
text 로더 테스트의 "hello world" 스타일과 동일).

layout.analyze_page()는 실제로(설치된 것 기준: 'layout' extra + 가중치가
있으면 진짜 PP-DocLayoutV2, 없으면 pymupdf 휴리스틱 폴백) 실행해서 검증한다
— 둘 다 이 합성 페이지들을 올바르게 분류한다. azure_di/vlm은 graph.py가
호출하는 지점에서 mock한다 — 실제 자격증명이 이 환경에 없어서다.

라우팅 규칙(layout.route_page 참고):
  - 텍스트 레이어 있음 + 그림·표 없음  -> native만
  - 텍스트 레이어 있음 + 그림·표 있음  -> native(순수 텍스트) + azure_di(표 구조만,
    include_text=False) + vlm(그림 캡션/표 요약) 셋 다 병렬
  - 텍스트 레이어 없음(스캔 문서)      -> azure_di(페이지 전체: 문단+표 구조) +
    vlm(그림 캡션/표 요약)
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


def test_page_with_image_and_text_layer_takes_native_and_vlm_not_azure_di(engine):
    """텍스트 레이어가 있는 페이지에 표 없이 이미지만 있으면 AzureDI는 아예
    스킵돼야 한다 — DI 역할이 표 구조 추출뿐이라, 표가 없으면 태워봤자
    매칭될 게 없어 완전히 헛수고(불필요한 호출/비용)라서."""
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
    assert (
        route_page(PageLayout(has_figures=True, has_text_layer=True, has_table=True))
        == "native_and_azure_di_and_vlm"
    )
    assert (
        route_page(PageLayout(has_figures=True, has_text_layer=True, has_table=False))
        == "native_and_vlm"
    )
    assert (
        route_page(PageLayout(has_figures=False, has_text_layer=True, has_table=False)) == "native"
    )
    assert (
        route_page(PageLayout(has_figures=False, has_text_layer=False, has_table=False))
        == "azure_di_and_vlm"
    )
    assert (
        route_page(PageLayout(has_figures=True, has_text_layer=False, has_table=True))
        == "azure_di_and_vlm"
    )


def test_page_graph_runs_native_and_azure_di_and_vlm_in_parallel():
    """텍스트 레이어 있음 + 표 있음 케이스에서 native/azure_di/vlm 셋 다
    실제로 같은 super-step에서 병렬 실행되는지 확인 — layout을 직접 만들어
    강제한다."""
    graph = build_page_graph().compile()

    fake_layout = PageLayout(has_figures=True, has_text_layer=True, has_table=True, boxes=[])

    with (
        patch("document_parser.parsing.loaders.pdf.graph.analyze_page", return_value=fake_layout),
        patch(
            "document_parser.parsing.loaders.pdf.graph.extract_native",
            return_value=[DocumentElement(text="n", metadata={"source": "native"})],
        ),
        patch(
            "document_parser.parsing.loaders.pdf.graph.extract_with_azure_di",
            return_value=([DocumentElement(text="a", metadata={"source": "azure_di"})], [], []),
        ),
        patch(
            "document_parser.parsing.loaders.pdf.graph.caption_figures",
            return_value=[DocumentElement(text="v", metadata={"source": "vlm"})],
        ),
    ):
        result = graph.invoke({"page": None, "page_number": 1, "raw_elements": []})

    sources = {el.metadata["source"] for el in result["elements"]}
    assert sources == {"native", "azure_di", "vlm"}


def test_page_graph_runs_azure_di_and_vlm_in_parallel():
    """텍스트 레이어 없음(스캔 문서) 케이스에서 azure_di와 vlm이 실제로 같은
    super-step에서 병렬 실행되는지 확인."""
    graph = build_page_graph().compile()

    fake_layout = PageLayout(has_figures=True, has_text_layer=False, boxes=[])

    with (
        patch("document_parser.parsing.loaders.pdf.graph.analyze_page", return_value=fake_layout),
        patch(
            "document_parser.parsing.loaders.pdf.graph.extract_with_azure_di",
            return_value=([DocumentElement(text="a", metadata={"source": "azure_di"})], [], []),
        ),
        patch(
            "document_parser.parsing.loaders.pdf.graph.caption_figures",
            return_value=[DocumentElement(text="v", metadata={"source": "vlm"})],
        ),
    ):
        result = graph.invoke({"page": None, "page_number": 1, "raw_elements": []})

    sources = {el.metadata["source"] for el in result["elements"]}
    assert sources == {"azure_di", "vlm"}


def test_merge_attaches_azure_di_table_html_to_matching_vlm_table_element():
    """DI가 찾은 표(bbox+html)와 VLM이 캡션한 표 박스(같은 위치)가 겹치면,
    최종 표 요소의 metadata["html"]에 DI 구조가 채워져야 한다 — PaddleX
    표 박스와 DI 표 검출은 서로 다른 결과라 id가 없으므로 bbox 겹침으로
    매칭한다(graph.py의 _best_matching_table)."""
    graph = build_page_graph().compile()

    fake_layout = PageLayout(has_figures=True, has_text_layer=True, has_table=True, boxes=[])
    vlm_table_element = DocumentElement(
        type=ElementType.TABLE,
        text="a table summary",
        bboxes=[BBox(x0=10, y0=10, x1=110, y1=60)],
        metadata={"source": "vlm", "layout_label": "table"},
    )
    from document_parser.parsing.loaders.pdf.azure_di import DetectedTable

    matching_table = DetectedTable(
        html="<table><tr><td>1</td></tr></table>",
        bboxes=[BBox(x0=12, y0=12, x1=108, y1=58)],  # vlm 박스와 거의 겹침
    )

    with (
        patch("document_parser.parsing.loaders.pdf.graph.analyze_page", return_value=fake_layout),
        patch(
            "document_parser.parsing.loaders.pdf.graph.extract_native",
            return_value=[],
        ),
        patch(
            "document_parser.parsing.loaders.pdf.graph.extract_with_azure_di",
            return_value=([], [matching_table], []),
        ),
        patch(
            "document_parser.parsing.loaders.pdf.graph.caption_figures",
            return_value=[vlm_table_element],
        ),
    ):
        result = graph.invoke({"page": None, "page_number": 1, "raw_elements": []})

    table_elements = [el for el in result["elements"] if el.type == ElementType.TABLE]
    assert len(table_elements) == 1
    assert table_elements[0].metadata["html"] == "<table><tr><td>1</td></tr></table>"
    assert table_elements[0].text == "a table summary"  # VLM 요약은 그대로 유지


def test_merge_leaves_table_without_matching_di_table_unchanged():
    """DI가 그 위치에서 표를 못 찾았으면(bbox가 하나도 안 겹치면), VLM 요약만
    남고 html은 안 붙는다."""
    graph = build_page_graph().compile()

    fake_layout = PageLayout(has_figures=True, has_text_layer=True, has_table=True, boxes=[])
    vlm_table_element = DocumentElement(
        type=ElementType.TABLE,
        text="a table summary",
        bboxes=[BBox(x0=10, y0=10, x1=110, y1=60)],
        metadata={"source": "vlm", "layout_label": "table"},
    )
    from document_parser.parsing.loaders.pdf.azure_di import DetectedTable

    unrelated_table = DetectedTable(
        html="<table><tr><td>unrelated</td></tr></table>",
        bboxes=[BBox(x0=500, y0=500, x1=600, y1=550)],  # 전혀 다른 위치
    )

    with (
        patch("document_parser.parsing.loaders.pdf.graph.analyze_page", return_value=fake_layout),
        patch(
            "document_parser.parsing.loaders.pdf.graph.extract_native",
            return_value=[],
        ),
        patch(
            "document_parser.parsing.loaders.pdf.graph.extract_with_azure_di",
            return_value=([], [unrelated_table], []),
        ),
        patch(
            "document_parser.parsing.loaders.pdf.graph.caption_figures",
            return_value=[vlm_table_element],
        ),
    ):
        result = graph.invoke({"page": None, "page_number": 1, "raw_elements": []})

    table_elements = [el for el in result["elements"] if el.type == ElementType.TABLE]
    assert len(table_elements) == 1
    assert "html" not in table_elements[0].metadata
    assert table_elements[0].text == "a table summary"


def test_merge_attaches_nearby_paragraphs_to_table_element():
    """DI가 include_text=False라 TEXT 요소로는 안 만든 문단이라도, 표 바로
    근처에 있으면 그 표 요소의 metadata["nearby_paragraphs"]에 붙어야 한다
    (청킹할 때 표만 뚝 떼서 주는 것보다 문맥이 있는 게 낫다는 요청)."""
    graph = build_page_graph().compile()

    fake_layout = PageLayout(has_figures=True, has_text_layer=True, has_table=True, boxes=[])
    vlm_table_element = DocumentElement(
        type=ElementType.TABLE,
        text="a table summary",
        bboxes=[BBox(x0=10, y0=100, x1=110, y1=200)],
        metadata={"source": "vlm", "layout_label": "table"},
    )
    from document_parser.parsing.loaders.pdf.azure_di import ContextParagraph

    near_above = ContextParagraph(text="바로 위 문단", bboxes=[BBox(x0=10, y0=50, x1=110, y1=90)])
    near_below = ContextParagraph(
        text="바로 아래 문단", bboxes=[BBox(x0=10, y0=210, x1=110, y1=250)]
    )
    far_away = ContextParagraph(text="아주 먼 문단", bboxes=[BBox(x0=10, y0=800, x1=110, y1=850)])

    with (
        patch("document_parser.parsing.loaders.pdf.graph.analyze_page", return_value=fake_layout),
        patch("document_parser.parsing.loaders.pdf.graph.extract_native", return_value=[]),
        patch(
            "document_parser.parsing.loaders.pdf.graph.extract_with_azure_di",
            return_value=([], [], [near_above, near_below, far_away]),
        ),
        patch(
            "document_parser.parsing.loaders.pdf.graph.caption_figures",
            return_value=[vlm_table_element],
        ),
    ):
        result = graph.invoke({"page": None, "page_number": 1, "raw_elements": []})

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
    순서로 나오는지 확인한다. 표는 이제 azure_di+vlm 경로를 타므로(native가
    아니라) 이 둘은 mock — 이 테스트의 목적은 레이아웃 모델의 실제 라벨/순서
    분류이지 azure_di/vlm 라이브 호출 자체가 아니다."""

    def fake_caption_figures(page, page_number, boxes):
        return [
            DocumentElement(
                type=ElementType.TABLE if b.label == "table" else ElementType.IMAGE,
                text="fake vlm output",
                page=page_number,
                bboxes=[BBox(x0=b.bbox[0], y0=b.bbox[1], x1=b.bbox[2], y1=b.bbox[3])],
                metadata={
                    "source": "vlm",
                    "layout_label": b.label,
                    "layout_cls_id": b.cls_id,
                    "layout_box_index": b.box_index,
                },
            )
            for b in boxes
        ]

    with (
        patch(
            "document_parser.parsing.loaders.pdf.graph.extract_with_azure_di",
            return_value=([], [], []),
        ),
        patch(
            "document_parser.parsing.loaders.pdf.graph.caption_figures",
            side_effect=fake_caption_figures,
        ),
    ):
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
        return_value=([fake_azure_element], [], []),
    ):
        document = engine.parse("scanned.pdf", data=_scanned_no_text_layer_pdf())

    assert len(document.elements) >= 1
    assert all(el.metadata.get("source") in {"azure_di", "vlm"} for el in document.elements)


@requires_real_layout_model
def test_multiple_figures_merge_in_reading_order_not_insertion_order(engine):
    """그림이 여러 개일 때, PDF에 삽입된 순서가 아니라 실제 페이지상 위치
    (위→아래) 기준으로 병합되는지 확인한다. 텍스트 레이어 + 그림이 있으므로
    native+azure_di+vlm 경로를 타는데(azure_di는 표 구조만 찾으니 그림만
    있는 이 문서에선 실제로는 아무것도 못 찾음), vlm/azure_di 둘 다 mock하고
    native는 실제로 동작하게 둔다."""

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

    with (
        patch(
            "document_parser.parsing.loaders.pdf.graph.extract_with_azure_di",
            return_value=([], [], []),
        ),
        patch(
            "document_parser.parsing.loaders.pdf.graph.caption_figures",
            side_effect=fake_caption_figures,
        ),
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
    include_text=True(텍스트 레이어 없을 때만)로 문단 텍스트를 실제로
    검증하려는 목적이라, 텍스트 레이어가 있으면 include_text=False가 돼서
    이 검증 자체가 성립하지 않는다."""
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
    레이어 + 그림이 있어 native+azure_di+vlm 경로를 타므로, 이 테스트 목적
    (VLM 캡션 검증)과 무관한 azure_di는 mock한다(azure 자격증명 없이도 이
    테스트만 따로 돌아가야 하므로)."""
    with patch(
        "document_parser.parsing.loaders.pdf.graph.extract_with_azure_di",
        return_value=([], [], []),
    ):
        document = engine.parse("doc.pdf", data=_pdf_with_image())

    vlm_elements = [el for el in document.elements if el.metadata.get("source") == "vlm"]
    assert len(vlm_elements) >= 1
    assert vlm_elements[0].text  # 실제 Claude Sonnet 4.6이 생성한 캡션
    assert len(vlm_elements[0].bboxes) >= 1


def test_extract_mermaid_pulls_out_fenced_code_block():
    from document_parser.parsing.loaders.pdf.vlm import _extract_mermaid

    response = "```mermaid\ngraph TD;\nA-->B;\n```"
    remainder, mermaid = _extract_mermaid(response)

    assert mermaid == "graph TD;\nA-->B;"
    assert remainder == "[다이어그램을 Mermaid로 추출함]"


def test_extract_mermaid_returns_none_for_plain_caption():
    from document_parser.parsing.loaders.pdf.vlm import _extract_mermaid

    text, mermaid = _extract_mermaid("그냥 평범한 사진 설명입니다.")

    assert mermaid is None
    assert text == "그냥 평범한 사진 설명입니다."
