"""페이지 단위 LangGraph 그래프 조립. pdf 로더 내부 구현이다.

다이어그램 그대로: 레이아웃 분석 → (Only Text + 텍스트 레이어 있음) 네이티브
추출 → 아니면(그림 있음 / 텍스트 레이어 없음) AzureDI+VLM 병렬 실행 → 병합.

이 그래프는 페이지 1장을 처리하는 단위다 — ``pdf/__init__.py``의 ``load()``가
문서의 페이지마다 한 번씩 invoke한다. 엔진 자체의 그래프(parsing/graph.py:
detect_format → extract → assemble)와 구조는 같은 패턴(StateGraph, TypedDict
상태, add_conditional_edges)을 따른다.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from document_parser.core.models import DocumentElement
from document_parser.parsing.loaders.pdf.azure_di import extract_with_azure_di
from document_parser.parsing.loaders.pdf.layout import PageLayout, analyze_page, needs_heavy_path
from document_parser.parsing.loaders.pdf.native import extract_native
from document_parser.parsing.loaders.pdf.vlm import caption_figures


class PageState(TypedDict, total=False):
    page: Any  # pymupdf.Page — 그래프 상태 스키마 자체엔 pymupdf 타입을 안 써서
    #             'pdf' extra 없이도 이 모듈을 import할 수 있게 한다.
    page_number: int
    layout: PageLayout
    # azure_di/vlm이 같은 super-step에서 병렬 실행되며 둘 다 이 키에 쓰므로
    # reducer(operator.add)가 없으면 LangGraph가 충돌로 처리한다.
    elements: Annotated[list[DocumentElement], operator.add]


def _analyze(state: PageState) -> dict:
    return {"layout": analyze_page(state["page"])}


def _route(state: PageState) -> str | list[str]:
    if needs_heavy_path(state["layout"]):
        return ["azure_di", "vlm"]
    return "native"


def _native(state: PageState) -> dict:
    return {"elements": extract_native(state["page"], state["page_number"])}


def _azure_di(state: PageState) -> dict:
    return {"elements": extract_with_azure_di(state["page"], state["page_number"])}


def _vlm(state: PageState) -> dict:
    boxes = state["layout"].crop_boxes
    return {"elements": caption_figures(state["page"], state["page_number"], boxes)}


def build_page_graph() -> StateGraph:
    graph = StateGraph(PageState)

    graph.add_node("analyze", _analyze)
    graph.add_node("native", _native)
    graph.add_node("azure_di", _azure_di)
    graph.add_node("vlm", _vlm)
    graph.add_node("merge", lambda _state: {})

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
