"""Azure Document Intelligence 클라이언트 (prebuilt-layout 모델).

필요 환경변수:
  AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT   예) https://<리소스>.cognitiveservices.azure.com
  AZURE_DOCUMENT_INTELLIGENCE_KEY

아래 필드명(``paragraphs``, ``content``, ``bounding_regions``, ``polygon``)은
문서만 보고 짐작한 게 아니라 azure-ai-documentintelligence==1.0.2를 실제
설치해서 모델 클래스를 직접 열어보고 확인했다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from document_parser.core.exceptions import MissingDependencyError


@dataclass
class LayoutParagraph:
    text: str
    # 하나의 단락(paragraph)이 여러 bounding_regions을 가질 수 있어(예: 단락이
    # 컬럼을 넘어가며 이어지는 경우) 단일 bbox가 아니라 리스트로 담는다.
    bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)


@dataclass
class LayoutTable:
    """DI가 감지한 표 하나 — 실제 행/열/병합 셀 구조를 담은 HTML.

    PP-DocLayoutV2가 찾은 표 박스와는 별개의 검출 결과라 id가 서로 없다 —
    호출자(graph.py의 merge 노드)가 bbox 겹침으로 어느 PaddleX 표 박스와
    짝인지 판단한다."""

    html: str
    bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)


@dataclass
class AzureLayoutResult:
    paragraphs: list[LayoutParagraph]
    tables: list[LayoutTable] = field(default_factory=list)


class AzureDocumentIntelligenceClient:
    def __init__(self) -> None:
        try:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential
        except ImportError as exc:
            raise MissingDependencyError(
                "Azure Document Intelligence support requires the 'azure' extra: "
                "pip install 'document-parser[azure]'"
            ) from exc

        endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        if not endpoint or not key:
            raise MissingDependencyError(
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT / AZURE_DOCUMENT_INTELLIGENCE_KEY "
                "environment variables are required"
            )

        self._client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )

    def analyze_layout(self, image_bytes: bytes) -> AzureLayoutResult:
        """페이지(또는 표 크롭) 이미지 1장을 통째로 보낸다(다이어그램 메모:
        크롭 크기와 무관하게 페이지당 과금이라 크롭 없이 페이지 전체를 한
        번에 요청한다)."""
        poller = self._client.begin_analyze_document("prebuilt-layout", body=image_bytes)
        result = poller.result()
        paragraphs = [
            LayoutParagraph(text=p.content, bboxes=_polygon_to_bboxes(p.bounding_regions))
            for p in (result.paragraphs or [])
        ]
        tables = [
            LayoutTable(html=_table_to_html(t), bboxes=_polygon_to_bboxes(t.bounding_regions))
            for t in (result.tables or [])
        ]
        return AzureLayoutResult(paragraphs=paragraphs, tables=tables)


def _polygon_to_bboxes(bounding_regions) -> list[tuple[float, float, float, float]]:
    """각 bounding region의 폴리곤(윗왼쪽부터 시계방향 [x0,y0,x1,y1,...] 평탄화
    좌표) 하나하나를 축정렬 bbox로 변환 — region이 여러 개면 리스트로 그대로 반환.

    TODO: 실제 in4u Azure 엔드포인트 기준 좌표 단위 확인 필요 — 이미지 입력이면
    보낸 이미지 픽셀 좌표와 같아야 하는데, 아직 실제 호출로 검증한 적은 없다.
    """
    boxes = []
    for region in bounding_regions or []:
        xs = region.polygon[0::2]
        ys = region.polygon[1::2]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes


def _escape_html(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _table_to_html(table) -> str:
    """DocumentTable(row_count, column_count, cells) -> <table> HTML.

    cells는 (row_index, column_index)로 좌상단 위치만 갖고, row_span/column_span
    만큼 그 아래·오른쪽 칸까지 차지한다 — 병합으로 덮이는 칸은 <td>를 아예
    안 만들어야 브라우저/TEDS 파서가 표를 올바르게 그린다(그래서 grid에
    "이 칸은 이미 누가 차지함"만 표시하고, 셀의 좌상단 칸일 때만 실제 <td>를
    만든다)."""
    grid: dict[tuple[int, int], object] = {}
    for cell in table.cells:
        row_span = cell.row_span or 1
        col_span = cell.column_span or 1
        for r in range(cell.row_index, cell.row_index + row_span):
            for c in range(cell.column_index, cell.column_index + col_span):
                grid[(r, c)] = cell if (r == cell.row_index and c == cell.column_index) else None

    rows_html = []
    for r in range(table.row_count):
        cells_html = []
        for c in range(table.column_count):
            cell = grid.get((r, c))
            if cell is None:
                continue
            attrs = ""
            if cell.row_span and cell.row_span > 1:
                attrs += f' rowspan="{cell.row_span}"'
            if cell.column_span and cell.column_span > 1:
                attrs += f' colspan="{cell.column_span}"'
            cells_html.append(f"<td{attrs}>{_escape_html(cell.content or '')}</td>")
        rows_html.append(f"<tr>{''.join(cells_html)}</tr>")
    return f"<table>{''.join(rows_html)}</table>"
