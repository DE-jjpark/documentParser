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
        result = graph.invoke({"page": None, "page_number": 1, "elements": []})

    sources = {el.metadata["source"] for el in result["elements"]}
    assert sources == {"azure_di", "vlm"}
