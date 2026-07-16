"""텍스트 레이어가 있는 페이지의 네이티브 텍스트 추출 — pdfplumber로 뽑는다.

레이아웃 분석(layout.py)이 실제 PP-DocLayoutV2 결과(``layout.boxes``)를 줬으면
그 영역별로 텍스트를 뽑아 25개 카테고리 중 실제 감지된 라벨을 그대로
``metadata["layout_label"]``에 남긴다.

pymupdf가 아니라 pdfplumber를 쓰는 이유: 요청("plumber 사용할거고 native일
때 사용하도록")에 따른 것 — 페이지 렌더링(레이아웃 모델 입력, AzureDI/VLM
크롭)은 계속 pymupdf가 담당하고, 순수 텍스트 추출만 pdfplumber로 바꿨다.
둘의 좌표계가 같아서(포인트, 좌상단 원점) layout.py가 준 bbox를 그대로
pdfplumber에 넘길 수 있다(graph.py의 pdf/__init__.py가 같은 PDF를 pymupdf/
pdfplumber 둘 다로 열어서 페이지 객체를 같이 넘겨준다).

``layout.boxes``가 비어 있으면(휴리스틱 폴백이라 애초에 카테고리를 모르는
경우, 또는 극히 드물게 실제 모델이 영역을 하나도 못 찾은 경우) 줄 단위로
문단을 다시 묶는다 — 텍스트 자체를 놓치지 않기 위한 안전망.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from document_parser.core.models import BBox, DocumentElement, ElementType

if TYPE_CHECKING:
    from document_parser.parsing.loaders.pdf.layout import LayoutBox, PageLayout

# PP-DocLayoutV2 라벨 -> ElementType. "table"이 없는 이유: 표는 layout.py의
# _FIGURE_LABELS에 포함돼 있어서 text_boxes(이 모듈이 다루는 대상)에 아예
# 안 들어온다 — 표는 native가 아니라 VLM/AzureDI가 처리한다(vlm.py 참고).
# ElementType.LIST도 대응하는 라벨이 없어 의도적으로 비어 있다 — PP-DocLayoutV2
# 25개 카테고리에 글머리 기호 목록에 해당하는 라벨이 없다(skep_parser
# 프로젝트의 DP-Bench 비교에서도 같은 한계를 확인함). 매핑에 없는 라벨은
# ElementType.TEXT로 처리한다.
_LABEL_TO_TYPE = {
    "doc_title": ElementType.HEADING,
    "paragraph_title": ElementType.HEADING,
    "figure_title": ElementType.HEADING,
}


def _label_to_element_type(label: str) -> ElementType:
    return _LABEL_TO_TYPE.get(label, ElementType.TEXT)


def _clamp_bbox(
    plumber_page: Any, bbox: tuple[float, float, float, float]
) -> tuple[float, float, float, float] | None:
    """pdfplumber는 bbox가 페이지 경계를 살짝이라도 벗어나면 예외를 던진다
    (pymupdf의 clip=은 알아서 잘라주는 것과 다름) — 레이아웃 모델이 준 bbox가
    반올림 등으로 아주 조금 넘어갈 수 있어서 크롭 전에 클램프한다."""
    x0, y0, x1, y1 = bbox
    x0 = max(0.0, min(x0, plumber_page.width))
    y0 = max(0.0, min(y0, plumber_page.height))
    x1 = max(0.0, min(x1, plumber_page.width))
    y1 = max(0.0, min(y1, plumber_page.height))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def extract_native(
    plumber_page: Any, page_number: int, layout: PageLayout
) -> list[DocumentElement]:
    if layout.text_boxes:
        return _extract_from_boxes(plumber_page, page_number, layout.text_boxes)
    return _extract_from_lines(plumber_page, page_number)


def _extract_from_boxes(
    plumber_page: Any, page_number: int, boxes: list[LayoutBox]
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    for box in boxes:
        clamped = _clamp_bbox(plumber_page, box.bbox)
        if clamped is None:
            continue
        text = (plumber_page.crop(clamped).extract_text() or "").strip()
        if not text:
            continue
        x0, y0, x1, y1 = box.bbox
        elements.append(
            DocumentElement(
                type=_label_to_element_type(box.label),
                text=text,
                page=page_number,
                bboxes=[BBox(x0=x0, y0=y0, x1=x1, y1=y1)],
                metadata={
                    "source": "native",
                    "layout_label": box.label,
                    "layout_cls_id": box.cls_id,
                    "layout_box_index": box.box_index,
                },
            )
        )
    return elements


# 줄 사이 세로 간격이 (직전 줄 높이 * 이 배수 + 여유분)보다 크면 새 문단으로
# 본다 — pymupdf의 "blocks"가 하던 문단 묶기를 pdfplumber의 줄 단위 출력에서
# 직접 재현한다.
_PARAGRAPH_GAP_MULTIPLIER = 1.5
_PARAGRAPH_GAP_MARGIN = 2.0


def _extract_from_lines(plumber_page: Any, page_number: int) -> list[DocumentElement]:
    """레이아웃 분석이 카테고리 정보를 못 줬을 때 쓰는 대체 경로 — 줄 단위로
    뽑은 다음 세로 간격을 기준으로 문단을 다시 묶는다(카테고리 구분 없이
    전부 TEXT)."""
    lines = plumber_page.extract_text_lines()
    elements: list[DocumentElement] = []
    current: list[dict] = []

    def flush() -> None:
        if not current:
            return
        text = "\n".join(line["text"] for line in current).strip()
        if not text:
            return
        x0 = min(line["x0"] for line in current)
        y0 = min(line["top"] for line in current)
        x1 = max(line["x1"] for line in current)
        y1 = max(line["bottom"] for line in current)
        elements.append(
            DocumentElement(
                type=ElementType.TEXT,
                text=text,
                page=page_number,
                bboxes=[BBox(x0=x0, y0=y0, x1=x1, y1=y1)],
                metadata={"source": "native"},
            )
        )

    prev_bottom: float | None = None
    prev_height: float = 0.0
    for line in lines:
        if prev_bottom is not None:
            gap = line["top"] - prev_bottom
            if gap > prev_height * _PARAGRAPH_GAP_MULTIPLIER + _PARAGRAPH_GAP_MARGIN:
                flush()
                current = []
        current.append(line)
        prev_bottom = line["bottom"]
        prev_height = line["bottom"] - line["top"]
    flush()
    return elements
