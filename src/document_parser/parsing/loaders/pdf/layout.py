"""페이지별 레이아웃 분석 — 어느 추출 경로를 탈지 결정한다.

'layout' extra(paddleocr)가 설치돼 있고 가중치가 받아져 있으면
(``document-parser download-models``, ``parsing/weights.py`` 참고) 실제
PP-DocLayoutV2 추론을 쓴다. paddleocr가 아예 설치 안 돼 있으면 pymupdf 전용
휴리스틱으로 폴백해서 'pdf' extra만으로도 일반 텍스트 PDF 파싱은 계속
동작하게 하고, paddleocr는 설치돼 있는데 가중치만 없으면 에러를 던진다
(가벼운 설치를 의도한 게 아니라 고쳐야 할 설정 누락이라서).

PP-DocLayoutV2를 실제로 돌리려면 paddlepaddle이 추가로 필요한데, 이건 PyPI
표준 인덱스가 아니라 전용 인덱스에서 받아야 한다(pyproject.toml의 'layout'
extra 주석 참고) — 공급망 검토는 보류, 팀 결정에 따라 가중치를 미리 받아
가는 방식으로 우선 진행.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

from document_parser.core.exceptions import MissingDependencyError
from document_parser.parsing.loaders.pdf.coords import RENDER_DPI, px_to_pt
from document_parser.parsing.weights import layout_model_dir

if TYPE_CHECKING:
    # 아래 타입 힌트에만 쓰인다 — 이 모듈은 pymupdf 객체를 직접 만들지 않고
    # 호출자가 이미 갖고 있는 `page`의 메서드만 호출한다. 런타임 경로에서
    # import를 빼둬야 'pdf' extra 없이도 document_parser가 import된다.
    import pymupdf

# PP-DocLayoutV2 전체 25개 카테고리 (~/.paddlex/official_models/PP-DocLayoutV2/
# inference.yml 기준). 실제 감지된 라벨은 LayoutBox.label에 그대로 담겨
# DocumentElement.metadata["layout_label"]까지 이어진다(vlm.py 참고) —
# has_figures 판정에만 쓰고 버리지 않도록 여기 전체 카탈로그를 보관해둔다.
ALL_LABELS = (
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote",
)

# 25개 중 "순수 텍스트 추출(pymupdf)만으로는 부족한" 카테고리 — has_figures
# 판정과 VLM/AzureDI 크롭 대상 결정에 쓴다. 원래는 그림류만 포함했는데(그림은
# pymupdf로 텍스트를 뽑을 수 없으니 당연), "표는 plumber(pymupdf) 텍스트
# 추출이 아니라 VLM/DI로 보내기로 했다"는 설계 결정에 따라 table도 여기
# 포함시켰다 — 표는 pymupdf로 글자는 뽑을 수 있어도 행/열/병합 구조를 못
# 뽑아서(TEDS/TEDS-S가 그 구조를 보는 지표라 점수가 안 나옴) 그림과 같은
# 취급으로 바꿨다.
_FIGURE_LABELS = {
    "chart",
    "image",
    "header_image",
    "footer_image",
    "seal",
    "display_formula",
    "inline_formula",
    "formula_number",
    "table",
}


@dataclass
class LayoutBox:
    """감지된 영역 하나 — 25개 카테고리 중 하나(label)와 좌표(bbox).

    order: PP-DocLayoutV2가 매긴 읽기 순서(1부터 시작). 못 정했으면 None —
    이 경우 bbox 위치(위→아래, 왼→오른) 기준으로 정렬한다(_reading_order_key
    참고). 휴리스틱 폴백(paddleocr 없을 때)은 애초에 순서를 모르므로 항상
    None이고, 감지된 이미지가 여러 개면 항상 위치 기준으로만 정렬된다.

    cls_id: PP-DocLayoutV2 원본 출력에 있는 카테고리 숫자 코드 — 원래는
    라우팅 판단(has_figures)과 label 문자열만 쓰고 버렸는데, 파싱 결과와
    원본 모델 출력을 비교할 때(예: DP-Bench 검증) 필요해서 metadata까지
    그대로 보존한다. 휴리스틱 폴백은 모른다(None).

    box_index: PP-DocLayoutV2가 반환한 원본 ``result["boxes"]`` 배열에서 이
    박스의 위치(0부터, 정렬 전 기준) — 페이지 안에서 유일하므로, 파싱 결과를
    원본 모델 출력 JSON과 대조해서 매핑할 때 이 값을 키로 쓰면 된다(order는
    많은 박스에서 null이라 이 용도로 못 씀). 휴리스틱 폴백은 모른다(None).
    """

    label: str
    bbox: tuple[float, float, float, float]
    order: int | None = None
    cls_id: int | None = None
    box_index: int | None = None


@dataclass
class PageLayout:
    has_figures: bool
    has_text_layer: bool
    boxes: list[LayoutBox] = field(default_factory=list)  # 항상 읽기 순서로 정렬돼 있음

    @property
    def crop_boxes(self) -> list[LayoutBox]:
        """그림류+표(_FIGURE_LABELS)만 골라낸 것 — VLM/AzureDI 크롭 대상.
        boxes가 이미 읽기 순서로 정렬돼 있어 여러 개여도 순서가 유지된다."""
        return [b for b in self.boxes if b.label in _FIGURE_LABELS]

    @property
    def text_boxes(self) -> list[LayoutBox]:
        """순수 텍스트류(제목·본문·목록 등, 표 제외)만 — 네이티브 텍스트
        추출 대상."""
        return [b for b in self.boxes if b.label not in _FIGURE_LABELS]


def _reading_order_key(box: LayoutBox) -> tuple:
    """order가 있으면 그걸 우선, 없으면 위치(top→bottom, left→right)로 대체.
    order 있는 박스와 없는 박스가 섞여도 순서가 뒤엉키지 않도록 (0, order)
    vs (1, y0, x0) 튜플로 묶어 order 있는 쪽이 항상 먼저 오게 한다."""
    if box.order is not None:
        return (0, box.order, 0.0, 0.0)
    return (1, 0, box.bbox[1], box.bbox[0])


def _sort_boxes(boxes: list[LayoutBox]) -> list[LayoutBox]:
    return sorted(boxes, key=_reading_order_key)


@lru_cache(maxsize=1)
def _get_model():
    from paddleocr import LayoutDetection

    model_dir = layout_model_dir()
    if not any(model_dir.glob("*")):
        raise MissingDependencyError(
            f"PP-DocLayoutV2 weights not found at {model_dir} -- run "
            "'document-parser download-models' first"
        )
    return LayoutDetection(model_name="PP-DocLayoutV2", model_dir=str(model_dir))


def analyze_page(page: pymupdf.Page) -> PageLayout:
    has_text_layer = bool(page.get_text().strip())

    try:
        model = _get_model()
    except ImportError:
        return _analyze_page_heuristic(page, has_text_layer)

    pix = page.get_pixmap(dpi=RENDER_DPI)
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        pix.save(f.name)
        (result,) = model.predict(f.name, batch_size=1, layout_nms=True)

    boxes = _sort_boxes(
        [
            LayoutBox(
                label=box["label"],
                bbox=px_to_pt(tuple(box["coordinate"])),
                order=box.get("order"),
                cls_id=box.get("cls_id"),
                box_index=i,
            )
            for i, box in enumerate(result["boxes"])
        ]
    )
    return PageLayout(
        has_figures=any(b.label in _FIGURE_LABELS for b in boxes),
        has_text_layer=has_text_layer,
        boxes=boxes,
    )


def _analyze_page_heuristic(page: pymupdf.Page, has_text_layer: bool) -> PageLayout:
    """'layout' extra가 없을 때 폴백: pymupdf 전용 휴리스틱.

    실제 25개 카테고리를 감지할 수 없으므로, 찾아낸 래스터 이미지는 그냥
    "image" 라벨(25개 카탈로그 중 하나)로 표시해둔다. 읽기 순서는 모르므로
    order=None으로 두고 위치 기준 정렬에 맡긴다."""
    images = page.get_images(full=True)
    boxes = _sort_boxes(
        [LayoutBox(label="image", bbox=tuple(page.get_image_bbox(img))) for img in images]
    )
    return PageLayout(
        has_figures=bool(boxes),
        has_text_layer=has_text_layer,
        boxes=boxes,
    )


def route_page(layout: PageLayout) -> str:
    """페이지 라우팅 규칙 (리뷰 피드백으로 수정 — 원래는 "그림 있음 OR
    텍스트 레이어 없음"이면 페이지 전체를 AzureDI로 보냈는데, 그러면 텍스트
    레이어가 멀쩡히 있어도 그림 하나 때문에 DI가 페이지 전체 텍스트를 다시
    추출하게 돼서 native가 이미 더 정확히 할 수 있는 일을 중복으로 하고
    있었다):

    - 텍스트 레이어가 없음(스캔 문서) → 원본 텍스트를 읽을 방법이 없으니
      AzureDI(페이지 전체) + VLM(그림·표 있으면 캡션/HTML 추출)
    - 텍스트 레이어는 있는데 그림·표도 있음 → 순수 텍스트는 native로 정확하게
      뽑고 그림·표만 VLM으로 처리 (AzureDI 불필요) — has_figures는 표 포함
      기준이라(_FIGURE_LABELS 참고), 표만 있고 다른 그림이 없는 페이지도
      여기로 온다.
    - 텍스트만 있고 텍스트 레이어도 있음 → native만

    반환값: "native" | "native_and_vlm" | "azure_di_and_vlm"
    """
    if not layout.has_text_layer:
        return "azure_di_and_vlm"
    if layout.has_figures:
        return "native_and_vlm"
    return "native"
