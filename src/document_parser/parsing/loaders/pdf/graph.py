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
from html.parser import HTMLParser
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from document_parser.core.models import BBox, DocumentElement, ElementType
from document_parser.parsing.loaders.pdf.azure_di import (
    ContextParagraph,
    DetectedTable,
    extract_with_azure_di,
)
from document_parser.parsing.loaders.pdf.layout import PageLayout, analyze_page, route_page
from document_parser.parsing.loaders.pdf.native import extract_native
from document_parser.parsing.loaders.pdf.vlm import caption_figures

# route_page()가 반환하는 논리적 경로 이름 -> 실제로 실행할 노드 이름(들).
_ROUTE_TARGETS: dict[str, str | list[str]] = {
    "native": "native",
    "native_and_azure_di_and_vlm": ["native", "azure_di", "vlm"],
    "native_and_vlm": ["native", "vlm"],
    "azure_di_and_vlm": ["azure_di", "vlm"],
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
    # azure_di/vlm(/native)이 같은 super-step에서 병렬 실행되며 둘 다 이
    # 키에 쓰므로 reducer(operator.add)가 없으면 LangGraph가 충돌로 처리한다.
    # merge 노드가 이걸 읽어 정렬한 결과를 elements(리듀서 없음, 그냥 덮어쓰기)
    # 에 담는다 — 여기에 다시 쓰면 reducer 때문에 중복으로 쌓이므로 분리해뒀다.
    raw_elements: Annotated[list[DocumentElement], operator.add]
    # azure_di 노드만 쓰는 키라 reducer 없이도 충돌이 안 난다. DI가 페이지
    # 전체에서 찾은 표(PP-DocLayoutV2 표 박스와는 독립적인 검출 결과)를
    # merge 노드가 bbox 겹침으로 매칭해서 표 요소에 붙인다.
    azure_di_tables: list[DetectedTable]
    # include_text=False라 TEXT 요소로는 안 만든 DI 문단들 — merge 노드가 표
    # 근처에 있는 것만 골라 그 표 요소의 metadata["nearby_paragraphs"]에 붙인다
    # (청킹할 때 표만 뚝 떼서 주는 것보다 문맥이 있는 게 낫다는 요청).
    azure_di_paragraphs: list[ContextParagraph]
    elements: list[DocumentElement]


def _analyze(state: PageState) -> dict:
    return {"layout": analyze_page(state["page"])}


def _route(state: PageState) -> str | list[str]:
    return _ROUTE_TARGETS[route_page(state["layout"], state.get("tier", "balanced"))]


def _native(state: PageState) -> dict:
    elements = extract_native(state["plumber_page"], state["page_number"], state["layout"])
    return {"raw_elements": elements}


def _azure_di(state: PageState) -> dict:
    # 텍스트 레이어가 있으면 native가 이미 순수 텍스트를 뽑고 있으니, DI는
    # 표 구조만 필요하다 — 문단(paragraphs) 기반 TEXT 요소는 만들지 않는다
    # (텍스트 레이어가 없는 스캔 페이지는 반대로 DI가 유일한 텍스트 출처라
    # include_text=True). 문단 자체(paragraphs)는 include_text와 무관하게
    # 항상 받아서 표 근처 문맥으로 쓴다(merge 노드 참고).
    include_text = not state["layout"].has_text_layer
    elements, tables, paragraphs = extract_with_azure_di(
        state["page"], state["page_number"], include_text=include_text
    )
    return {"raw_elements": elements, "azure_di_tables": tables, "azure_di_paragraphs": paragraphs}


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


_NEARBY_PARAGRAPH_MAX_COUNT = 2
# 이 거리(포인트) 안에 있는 문단만 "근처"로 본다 — 너무 멀리 있는 문단까지
# 끌어오면 표랑 상관없는 내용이 섞여서 청킹 문맥으로 오히려 방해가 된다.
_NEARBY_PARAGRAPH_MAX_DISTANCE = 60.0


def _vertical_distance(a: BBox, b: BBox) -> float:
    """두 bbox의 수직 거리 — 겹치면(같은 높이대에 있으면, 예: 옆 컬럼) 0."""
    if a.y1 <= b.y0:
        return b.y0 - a.y1
    if b.y1 <= a.y0:
        return a.y0 - b.y1
    return 0.0


def _nearby_paragraph_texts(table_bbox: BBox, paragraphs: list[ContextParagraph]) -> list[str]:
    candidates: list[tuple[float, str]] = []
    for paragraph in paragraphs:
        if not paragraph.bboxes:
            continue
        distance = min(_vertical_distance(table_bbox, b) for b in paragraph.bboxes)
        if distance <= _NEARBY_PARAGRAPH_MAX_DISTANCE:
            candidates.append((distance, paragraph.text))
    candidates.sort(key=lambda c: c[0])
    return [text for _, text in candidates[:_NEARBY_PARAGRAPH_MAX_COUNT]]


class _TableRowsParser(HTMLParser):
    """azure_document_intelligence.py의 _table_to_html()이 만든 <table><tr><td>
    구조에서 셀 텍스트만 뽑는다 — 마크다운 표로 다시 그리기 위해."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] = []
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._current_cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self._in_cell = False
            if self._current_row is not None:
                self._current_row.append("".join(self._current_cell).strip())
        elif tag == "tr":
            if self._current_row is not None:
                self.rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_cell.append(data)


def _html_table_to_markdown(html: str) -> str:
    """DI의 실제 표 구조(html)를 마크다운 표로 변환한다 — 병합 셀(colspan/
    rowspan)은 마크다운 표 자체가 표현 못 하는 개념이라 셀 하나로 뭉뚱그려
    지므로 구조 정보 손실이 있을 수 있다(그 손실 없는 원본은 metadata["html"]
    에 그대로 남아있음 — 마크다운은 어디까지나 .text/청킹용 표시)."""
    parser = _TableRowsParser()
    parser.feed(html)
    rows = parser.rows
    if not rows:
        return ""

    lines = [f"| {' | '.join(rows[0])} |", f"| {' | '.join(['---'] * len(rows[0]))} |"]
    lines.extend(f"| {' | '.join(row)} |" for row in rows[1:])
    return "\n".join(lines)


# PP-DocLayoutV2가 "table"이라고 했는데 AzureDI가 실제 표 구조를 그 자리에서
# 못 찾았고, 그 영역에 native로 뽑을 수 있는 텍스트가 이 정도 이상 있으면
# "표가 아니라 일반 텍스트를 표로 오분류한 것"으로 본다 — 실측으로 발견한
# 문제: 페이지 전체(조항 여러 개짜리 본문)가 표 하나로 오분류돼서, 원문이
# 통째로 VLM의 추상적 요약으로 대체되며 유실됐었다. 표 구조는 잃어도 원문
# 보존이 우선이라, 이 경우 type도 TABLE→TEXT로 바로잡고 원문을 text로 쓴다.
_MISCLASSIFIED_TABLE_MIN_CHARS = 40


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


def _enrich_table_element(
    element: DocumentElement,
    tables: list[DetectedTable],
    paragraphs: list[ContextParagraph],
    plumber_page: Any,
) -> DocumentElement:
    if element.type != ElementType.TABLE or not element.bboxes:
        return element

    updates: dict = {}
    metadata = dict(element.metadata)
    matched = False
    if tables:
        match = _best_matching_table(element.bboxes[0], tables)
        if match is not None:
            matched = True
            metadata["html"] = match.html
            # DI가 실제로 표를 찾았으면 그 구조에서 뽑은 마크다운이 VLM 자체
            # 마크다운(그냥 이미지 보고 추측한 것)보다 정확하니 text를 덮어쓴다
            # — VLM의 마크다운은 DI가 못 찾았을 때만 그대로 남는 폴백이 된다.
            markdown = _html_table_to_markdown(match.html)
            if markdown:
                updates["text"] = markdown

    if not matched and plumber_page is not None:
        native_text = _extract_plumber_text(plumber_page, element.bboxes[0])
        if len(native_text) >= _MISCLASSIFIED_TABLE_MIN_CHARS:
            updates["text"] = native_text
            updates["type"] = ElementType.TEXT
            metadata["misclassified_as_table"] = True

    if paragraphs:
        nearby = _nearby_paragraph_texts(element.bboxes[0], paragraphs)
        if nearby:
            metadata["nearby_paragraphs"] = nearby

    if metadata != element.metadata:
        updates["metadata"] = metadata
    if not updates:
        return element
    return element.model_copy(update=updates)


def _merge(state: PageState) -> dict:
    elements = sorted(state["raw_elements"], key=_merge_key)
    tables = state.get("azure_di_tables", [])
    paragraphs = state.get("azure_di_paragraphs", [])
    plumber_page = state.get("plumber_page")
    has_table_element = any(el.type == ElementType.TABLE for el in elements)
    # tables/paragraphs가 둘 다 비어 있어도(DI가 표를 하나도 못 찾은 경우
    # 포함) 표로 분류된 요소가 있으면 오분류 복구 로직(native 폴백)을 위해
    # 여전히 돌려야 한다 — 그래서 has_table_element도 조건에 넣는다.
    if tables or paragraphs or (plumber_page is not None and has_table_element):
        elements = [_enrich_table_element(el, tables, paragraphs, plumber_page) for el in elements]
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
