"""텍스트 레이어가 있는 페이지의 네이티브 텍스트 추출 — pdfplumber로 뽑는다.

레이아웃 분석(layout.py)이 실제 PP-DocLayoutV2 결과(``layout.boxes``)를 줬으면
그 영역별로 텍스트를 뽑아 25개 카테고리 중 실제 감지된 라벨을 그대로
``metadata["block_type"]``에 남긴다.

pymupdf가 아니라 pdfplumber를 쓰는 이유: 요청("plumber 사용할거고 native일
때 사용하도록")에 따른 것 — 페이지 렌더링(레이아웃 모델 입력, VLM
크롭)은 계속 pymupdf가 담당하고, 순수 텍스트 추출만 pdfplumber로 바꿨다.
둘의 좌표계가 같아서(포인트, 좌상단 원점) layout.py가 준 bbox를 그대로
pdfplumber에 넘길 수 있다(graph.py의 pdf/__init__.py가 같은 PDF를 pymupdf/
pdfplumber 둘 다로 열어서 페이지 객체를 같이 넘겨준다).

``layout.boxes``가 비어 있으면(휴리스틱 폴백이라 애초에 카테고리를 모르는
경우, 또는 극히 드물게 실제 모델이 영역을 하나도 못 찾은 경우) 줄 단위로
문단을 다시 묶는다 — 텍스트 자체를 놓치지 않기 위한 안전망.

HEADING 타입 element는 대표 폰트 크기를 ``metadata["font_size"]``에도 남긴다
— PP-DocLayoutV2는 doc_title/paragraph_title/figure_title 3개 카테고리만
주고 숫자 계층(레벨)이 없어서, 문서 전체를 보는 후처리(pdf/__init__.py의
_assign_heading_levels)가 제목 텍스트의 번호 매김(우선) 또는 이 폰트 크기의
상대 순위(번호가 없을 때, PPT 슬라이드 제목 등)로 레벨을 역산한다.

``metadata["layout_order"]``에는 PP-DocLayoutV2가 매긴 원본 읽기 순서(box.order,
없으면 None)를 그대로 남긴다 — graph.py의 merge 노드가 native/vlm/vlm_text
결과를 하나로 합칠 때 이 값을 좌표보다 먼저 본다(layout.py의
_reading_order_key와 같은 패턴). 이걸 안 남기면 병합 시점에 좌표로만
재정렬하게 되는데, 다단(컬럼) 레이아웃처럼 위치만으론 읽기 순서가 애매한
경우 실제 모델이 추론한 순서를 버리게 된다.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from document_parser.core.models import BBox, DocumentElement, ElementType

if TYPE_CHECKING:
    from document_parser.parsing.loaders.pdf.layout import LayoutBox, PageLayout

# PP-DocLayoutV2 라벨 -> ElementType. "table"이 없는 이유: 표는 layout.py의
# _FIGURE_LABELS에 포함돼 있어서 text_boxes(이 모듈이 다루는 대상)에 아예
# 안 들어온다 — 표는 native가 아니라 VLM이 처리한다(vlm.py 참고).
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


def _representative_font_size(cropped: Any) -> float | None:
    """크롭된 영역 안 글자들의 대표(최빈) 폰트 크기 — 제목 레벨 추정에 쓴다.
    번호 매김이 없는 제목(PPT 슬라이드 제목 등)에서 문서 전체 제목들과 크기를
    비교해 상대적 레벨을 매기는 용도(pdf/__init__.py의 _assign_heading_levels
    참고) — 여기서는 크기만 남기고 레벨 계산은 문서 전체를 봐야 하는 후처리에
    맡긴다."""
    sizes = [round(c["size"], 1) for c in cropped.chars if c.get("text", "").strip()]
    if not sizes:
        return None
    return Counter(sizes).most_common(1)[0][0]


def _extract_from_boxes(
    plumber_page: Any, page_number: int, boxes: list[LayoutBox]
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    for box in boxes:
        clamped = _clamp_bbox(plumber_page, box.bbox)
        if clamped is None:
            continue
        cropped = plumber_page.crop(clamped)
        text = (cropped.extract_text() or "").strip()
        if not text:
            continue
        element_type = _label_to_element_type(box.label)
        metadata: dict[str, Any] = {
            "source": "native",
            "block_type": box.label,
            "layout_cls_id": box.cls_id,
            "layout_box_index": box.box_index,
            "layout_order": box.order,
        }
        if element_type == ElementType.HEADING:
            font_size = _representative_font_size(cropped)
            if font_size is not None:
                metadata["font_size"] = font_size
        x0, y0, x1, y1 = box.bbox
        elements.append(
            DocumentElement(
                type=element_type,
                text=text,
                page=page_number,
                bboxes=[BBox(x0=x0, y0=y0, x1=x1, y1=y1)],
                metadata=metadata,
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
