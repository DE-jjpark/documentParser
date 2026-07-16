"""페이지 단위 LangGraph 그래프 조립. pdf 로더 내부 구현이다.

레이아웃 분석 → 3-way 라우팅(layout.route_page 참고) → native | native+azure_di+vlm |
azure_di+vlm → 병합.

이 그래프는 페이지 1장을 처리하는 단위다 — ``pdf/__init__.py``의 ``load()``가
문서의 페이지마다 한 번씩 invoke한다. 엔진 자체의 그래프(parsing/graph.py:
detect_format → extract → assemble)와 구조는 같은 패턴(StateGraph, TypedDict
상태, add_conditional_edges)을 따른다.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from document_parser.core.models import BBox, DocumentElement, ElementType
from document_parser.parsing.loaders.pdf.azure_di import DetectedTable, extract_with_azure_di
from document_parser.parsing.loaders.pdf.layout import PageLayout, analyze_page, route_page
from document_parser.parsing.loaders.pdf.native import extract_native
from document_parser.parsing.loaders.pdf.vlm import caption_figures

# route_page()가 반환하는 논리적 경로 이름 -> 실제로 실행할 노드 이름(들).
_ROUTE_TARGETS: dict[str, str | list[str]] = {
    "native": "native",
    "native_and_azure_di_and_vlm": ["native", "azure_di", "vlm"],
    "azure_di_and_vlm": ["azure_di", "vlm"],
}


class PageState(TypedDict, total=False):
    page: Any  # pymupdf.Page — 그래프 상태 스키마 자체엔 pymupdf 타입을 안 써서
    #             'pdf' extra 없이도 이 모듈을 import할 수 있게 한다.
    page_number: int
    layout: PageLayout
    # azure_di/vlm(/native)이 같은 super-step에서 병렬 실행되며 둘 다 이
    # 키에 쓰므로 reducer(operator.add)가 없으면 LangGraph가 충돌로 처리한다.
    # merge 노드가 이걸 읽어 정렬한 결과를 elements(리듀서 없음, 그냥 덮어쓰기)
    # 에 담는다 — 여기에 다시 쓰면 reducer 때문에 중복으로 쌓이므로 분리해뒀다.
    raw_elements: Annotated[list[DocumentElement], operator.add]
    # azure_di 노드만 쓰는 키라 reducer 없이도 충돌이 안 난다. DI가 페이지
    # 전체에서 찾은 표(PP-DocLayoutV2 표 박스와는 독립적인 검출 결과)를
    # merge 노드가 bbox 겹침으로 매칭해서 표 요소에 붙인다.
    azure_di_tables: list[DetectedTable]
    elements: list[DocumentElement]


def _analyze(state: PageState) -> dict:
    return {"layout": analyze_page(state["page"])}


def _route(state: PageState) -> str | list[str]:
    return _ROUTE_TARGETS[route_page(state["layout"])]


def _native(state: PageState) -> dict:
    elements = extract_native(state["page"], state["page_number"], state["layout"])
    return {"raw_elements": elements}


def _azure_di(state: PageState) -> dict:
    # 텍스트 레이어가 있으면 native가 이미 순수 텍스트를 뽑고 있으니, DI는
    # 표 구조만 필요하다 — 문단(paragraphs) 기반 TEXT 요소는 만들지 않는다
    # (텍스트 레이어가 없는 스캔 페이지는 반대로 DI가 유일한 텍스트 출처라
    # include_text=True).
    include_text = not state["layout"].has_text_layer
    elements, tables = extract_with_azure_di(
        state["page"], state["page_number"], include_text=include_text
    )
    return {"raw_elements": elements, "azure_di_tables": tables}


def _vlm(state: PageState) -> dict:
    boxes = state["layout"].crop_boxes
    elements = caption_figures(state["page"], state["page_number"], boxes)
    return {"raw_elements": elements}


def _merge_key(element: DocumentElement) -> tuple[float, float]:
    """(native와/또는 azure_di) + vlm은 같은 super-step에서 병렬 실행되므로
    reducer가 합친 직후의 순서는 어느 쪽이 먼저 끝났느냐에 달려 있어 신뢰할
    수 없다 — bbox 위치(top→bottom, left→right) 기준으로 다시 정렬해야 실제
    읽기 순서가 보장된다. bbox가 없는 요소(지금의 azure_di 페이지 전체
    placeholder처럼 위치 정보가 아예 없는 경우)는 페이지 맨 위(0, 0)로
    취급한다."""
    if element.bboxes:
        return (element.bboxes[0].y0, element.bboxes[0].x0)
    return (0.0, 0.0)


def _bbox_overlap_ratio(a: BBox, b: BBox) -> float:
    """두 bbox의 IoU(intersection over union) — DI가 찾은 표와 PaddleX가
    찾은 표 박스가 서로 다른 검출 결과라 id가 없으니, 이 겹침 비율로 "같은
    표"인지 판단한다."""
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    area_a = (a.x1 - a.x0) * (a.y1 - a.y0)
    area_b = (b.x1 - b.x0) * (b.y1 - b.y0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


# 이 밑으로는 겹침 비율이 얼마가 나와도 "다른 표"로 본다 — 살짝이라도 겹치면
# 다 매칭시키면 표 여러 개가 한쪽으로 몰릴 수 있어서 최소 기준을 둔다.
_TABLE_MATCH_THRESHOLD = 0.1


def _best_matching_table(element_bbox: BBox, tables: list[DetectedTable]) -> DetectedTable | None:
    best: DetectedTable | None = None
    best_overlap = _TABLE_MATCH_THRESHOLD
    for table in tables:
        for table_bbox in table.bboxes:
            overlap = _bbox_overlap_ratio(element_bbox, table_bbox)
            if overlap > best_overlap:
                best_overlap = overlap
                best = table
    return best


def _attach_azure_di_table_html(
    element: DocumentElement, tables: list[DetectedTable]
) -> DocumentElement:
    if element.type != ElementType.TABLE or not element.bboxes or not tables:
        return element
    match = _best_matching_table(element.bboxes[0], tables)
    if match is None:
        return element
    return element.model_copy(update={"metadata": {**element.metadata, "html": match.html}})


def _merge(state: PageState) -> dict:
    elements = sorted(state["raw_elements"], key=_merge_key)
    tables = state.get("azure_di_tables", [])
    if tables:
        elements = [_attach_azure_di_table_html(el, tables) for el in elements]
    return {"elements": elements}


def build_page_graph() -> StateGraph:
    graph = StateGraph(PageState)

    graph.add_node("analyze", _analyze)
    graph.add_node("native", _native)
    graph.add_node("azure_di", _azure_di)
    graph.add_node("vlm", _vlm)
    graph.add_node("merge", _merge)

    graph.add_edge(START, "analyze")
    graph.add_conditional_edges(
        "analyze",
        _route,
        {"native": "native", "azure_di": "azure_di", "vlm": "vlm"},
    )
    graph.add_edge("native", "merge")
    graph.add_edge("azure_di", "merge")
    graph.add_edge("vlm", "merge")
    graph.add_edge("merge", END)

    return graph
