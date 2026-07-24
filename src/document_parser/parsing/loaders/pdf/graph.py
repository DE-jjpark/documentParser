"""페이지 단위 LangGraph 그래프 조립. pdf 로더 내부 구현이다.

레이아웃 분석 → 3-way 라우팅(layout.route_page 참고) → native | native+vlm |
vlm_only → 병합.

AzureDI는 더 이상 쓰지 않는다(팀 결정) — 표 구조 추출도 스캔 페이지 본문
추출도 전부 VLM이 담당한다(vlm.py의 caption_figures/transcribe_text_boxes
참고).

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
from document_parser.parsing.loaders.pdf.layout import PageLayout, analyze_page, route_page
from document_parser.parsing.loaders.pdf.native import extract_native
from document_parser.parsing.loaders.pdf.vlm import caption_figures, transcribe_text_boxes

# route_page()가 반환하는 논리적 경로 이름 -> 실제로 실행할 노드 이름(들).
_ROUTE_TARGETS: dict[str, str | list[str]] = {
    "native": "native",
    "native_and_vlm": ["native", "vlm"],
    # 텍스트 레이어 없는(스캔) 페이지 — text_boxes(본문)도 crop_boxes(그림·표)도
    # 전부 VLM 크롭 경로를 탄다(vlm 노드는 crop_boxes, vlm_text 노드는
    # text_boxes를 처리 — vlm.py 참고).
    "vlm_only": ["vlm", "vlm_text"],
}


class PageState(TypedDict, total=False):
    page: Any  # pymupdf.Page — 그래프 상태 스키마 자체엔 pymupdf 타입을 안 써서
    #             'pdf' extra 없이도 이 모듈을 import할 수 있게 한다.
    tier: str  # "fast" | "balanced" — pdf/__init__.py의 load()가 페이지마다 그대로 넘긴다.
    # pdfplumber.Page — native(순수 텍스트) 추출은 이걸로 한다(요청:
    # "plumber 사용할거고 native일 때 사용하도록"). pymupdf page와 좌표계가
    # 같아서(포인트, 좌상단 원점) 같은 bbox를 그대로 재사용한다.
    plumber_page: Any
    page_number: int
    layout: PageLayout
    # native/vlm/vlm_text가 같은 super-step에서 병렬 실행되며 같이 이 키에
    # 쓰므로 reducer(operator.add)가 없으면 LangGraph가 충돌로 처리한다.
    # merge 노드가 이걸 읽어 정렬한 결과를 elements(리듀서 없음, 그냥 덮어쓰기)
    # 에 담는다 — 여기에 다시 쓰면 reducer 때문에 중복으로 쌓이므로 분리해뒀다.
    raw_elements: Annotated[list[DocumentElement], operator.add]
    elements: list[DocumentElement]


def _analyze(state: PageState) -> dict:
    return {"layout": analyze_page(state["page"])}


def _route(state: PageState) -> str | list[str]:
    return _ROUTE_TARGETS[route_page(state["layout"], state.get("tier", "balanced"))]


def _native(state: PageState) -> dict:
    elements = extract_native(state["plumber_page"], state["page_number"], state["layout"])
    return {"raw_elements": elements}


def _vlm(state: PageState) -> dict:
    boxes = state["layout"].crop_boxes
    elements = caption_figures(state["page"], state["page_number"], boxes)
    return {"raw_elements": elements}


def _vlm_text(state: PageState) -> dict:
    boxes = state["layout"].text_boxes
    elements = transcribe_text_boxes(state["page"], state["page_number"], boxes)
    return {"raw_elements": elements}


def _merge_key(element: DocumentElement) -> tuple:
    """(native와/또는 vlm/vlm_text는) 같은 super-step에서 병렬 실행되므로
    reducer가 합친 직후의 순서는 어느 쪽이 먼저 끝났느냐에 달려 있어 신뢰할
    수 없다 — 다시 정렬해야 실제 읽기 순서가 보장된다. layout.py의
    _reading_order_key와 같은 패턴: native.py/vlm.py가 남겨둔
    metadata["layout_order"](PP-DocLayoutV2 원본 읽기 순서)가 있으면 그걸
    최우선으로 쓰고, 없으면 bbox 위치(top→bottom, left→right)로 대체한다 —
    다단(컬럼) 레이아웃처럼 위치만으론 순서가 애매한 경우 실제 모델이
    추론한 순서를 버리지 않기 위함. order 있는 요소와 없는 요소가 섞여도
    order 있는 쪽이 항상 먼저 오도록 (0, order) vs (1, y0, x0) 튜플로
    묶는다."""
    order = element.metadata.get("layout_order")
    if order is not None:
        return (0, order, 0.0, 0.0)
    if element.bboxes:
        return (1, 0, element.bboxes[0].y0, element.bboxes[0].x0)
    return (1, 0, 0.0, 0.0)


def _extract_plumber_text(plumber_page: Any, bbox: BBox) -> str:
    """pdfplumber는 bbox가 페이지 경계를 살짝이라도 벗어나면 예외를 던진다
    (pymupdf는 알아서 잘라주는 것과 다름) — 레이아웃 모델이 준 bbox가 반올림
    등으로 아주 조금 넘어갈 수 있어서, 크롭 전에 페이지 크기 안으로 클램프
    해준다."""
    x0 = max(0.0, min(bbox.x0, plumber_page.width))
    y0 = max(0.0, min(bbox.y0, plumber_page.height))
    x1 = max(0.0, min(bbox.x1, plumber_page.width))
    y1 = max(0.0, min(bbox.y1, plumber_page.height))
    if x1 <= x0 or y1 <= y0:
        return ""
    return (plumber_page.crop((x0, y0, x1, y1)).extract_text() or "").strip()


def _looks_like_markdown_table(text: str) -> bool:
    """VLM이 돌려준 text가 실제로 마크다운 표 구조(``| ... |`` 형태의 행이
    최소 2줄, 보통 헤더+구분줄 또는 헤더+데이터 1줄)를 갖췄는지 본다 —
    AzureDI 없이 "이게 진짜 표 맞나"를 판단할 유일한 신호라서, DI가 표 구조를
    찾았는지 여부 대신 이걸로 오분류를 가른다."""
    pipe_lines = sum(1 for line in text.splitlines() if line.strip().startswith("|"))
    return pipe_lines >= 2


# PP-DocLayoutV2가 "table"이라고 했는데 VLM이 돌려준 결과가 실제 표 구조로
# 안 보이고, 그 영역에 native로 뽑을 수 있는 텍스트가 이 정도 이상 있으면
# "표가 아니라 일반 텍스트를 표로 오분류한 것"으로 본다 — 실측으로 발견한
# 문제: 페이지 전체(조항 여러 개짜리 본문)가 표 하나로 오분류돼서, 원문이
# 통째로 VLM의 추상적 요약으로 대체되며 유실됐었다. 표 구조는 잃어도 원문
# 보존이 우선이라, 이 경우 type도 TABLE→TEXT로 바로잡고 원문을 text로 쓴다.
_MISCLASSIFIED_TABLE_MIN_CHARS = 40


def _vertical_distance(a: BBox, b: BBox) -> float:
    """두 bbox의 수직 거리 — 겹치면(같은 높이대에 있으면, 예: 옆 컬럼) 0."""
    if a.y1 <= b.y0:
        return b.y0 - a.y1
    if b.y1 <= a.y0:
        return a.y0 - b.y1
    return 0.0


_NEARBY_PARAGRAPH_MAX_COUNT = 2
# 이 거리(포인트) 안에 있는 문단만 "근처"로 본다 — 너무 멀리 있는 문단까지
# 끌어오면 표랑 상관없는 내용이 섞여서 청킹 문맥으로 오히려 방해가 된다.
_NEARBY_PARAGRAPH_MAX_DISTANCE = 60.0


def _nearby_paragraph_texts(table_bbox: BBox, elements: list[DocumentElement]) -> list[str]:
    """AzureDI 없이 표 근처 문단을 찾는다 — DI가 include_text=False로 따로
    뽑아주던 것과 달리, 같은 페이지에서 native(또는 스캔 페이지면 vlm_text)가
    이미 만들어둔 TEXT/HEADING element를 그대로 재사용한다(문단 텍스트를
    별도로 다시 뽑을 필요가 없다)."""
    candidates: list[tuple[float, str]] = []
    for el in elements:
        if el.type not in (ElementType.TEXT, ElementType.HEADING) or not el.bboxes:
            continue
        distance = min(_vertical_distance(table_bbox, b) for b in el.bboxes)
        if distance <= _NEARBY_PARAGRAPH_MAX_DISTANCE:
            candidates.append((distance, el.text))
    candidates.sort(key=lambda c: c[0])
    return [text for _, text in candidates[:_NEARBY_PARAGRAPH_MAX_COUNT]]


def _enrich_table_element(
    element: DocumentElement, plumber_page: Any, elements: list[DocumentElement]
) -> DocumentElement:
    if element.type != ElementType.TABLE or not element.bboxes:
        return element

    if not _looks_like_markdown_table(element.text) and plumber_page is not None:
        native_text = _extract_plumber_text(plumber_page, element.bboxes[0])
        if len(native_text) >= _MISCLASSIFIED_TABLE_MIN_CHARS:
            metadata = dict(element.metadata)
            metadata["misclassified_as_table"] = True
            return element.model_copy(
                update={"text": native_text, "type": ElementType.TEXT, "metadata": metadata}
            )

    nearby = _nearby_paragraph_texts(element.bboxes[0], elements)
    if not nearby:
        return element
    metadata = dict(element.metadata)
    metadata["nearby_paragraphs"] = nearby
    return element.model_copy(update={"metadata": metadata})


def _merge(state: PageState) -> dict:
    elements = sorted(state["raw_elements"], key=_merge_key)
    plumber_page = state.get("plumber_page")
    enriched = [_enrich_table_element(el, plumber_page, elements) for el in elements]
    return {"elements": enriched}


def build_page_graph() -> StateGraph:
    graph = StateGraph(PageState)

    graph.add_node("analyze", _analyze)
    graph.add_node("native", _native)
    graph.add_node("vlm", _vlm)
    graph.add_node("vlm_text", _vlm_text)
    graph.add_node("merge", _merge)

    graph.add_edge(START, "analyze")
    graph.add_conditional_edges(
        "analyze",
        _route,
        {"native": "native", "vlm": "vlm", "vlm_text": "vlm_text"},
    )
    graph.add_edge("native", "merge")
    graph.add_edge("vlm", "merge")
    graph.add_edge("vlm_text", "merge")
    graph.add_edge("merge", END)

    return graph
