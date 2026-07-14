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

# 25개 중 "텍스트로 바로 추출하기 어려운 그림류" — has_figures 판정과 VLM
# 크롭 대상 결정에 쓴다(skep_parser 프로젝트의 DP-Bench 비교에서 쓴 것과
# 동일한 분류).
_FIGURE_LABELS = {
    "chart",
    "image",
    "header_image",
    "footer_image",
    "seal",
    "display_formula",
    "inline_formula",
    "formula_number",
}


@dataclass
class LayoutBox:
    """감지된 영역 하나 — 25개 카테고리 중 하나(label)와 좌표(bbox)."""

    label: str
    bbox: tuple[float, float, float, float]


@dataclass
class PageLayout:
    has_figures: bool
    has_text_layer: bool
    boxes: list[LayoutBox] = field(default_factory=list)

    @property
    def crop_boxes(self) -> list[LayoutBox]:
        """그림류(_FIGURE_LABELS)만 골라낸 것 — VLM 크롭 대상."""
        return [b for b in self.boxes if b.label in _FIGURE_LABELS]


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

    pix = page.get_pixmap(dpi=200)
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        pix.save(f.name)
        (result,) = model.predict(f.name, batch_size=1, layout_nms=True)

    boxes = [
        LayoutBox(label=box["label"], bbox=tuple(box["coordinate"])) for box in result["boxes"]
    ]
    return PageLayout(
        has_figures=any(b.label in _FIGURE_LABELS for b in boxes),
        has_text_layer=has_text_layer,
        boxes=boxes,
    )


def _analyze_page_heuristic(page: pymupdf.Page, has_text_layer: bool) -> PageLayout:
    """'layout' extra가 없을 때 폴백: pymupdf 전용 휴리스틱.

    실제 25개 카테고리를 감지할 수 없으므로, 찾아낸 래스터 이미지는 그냥
    "image" 라벨(25개 카탈로그 중 하나)로 표시해둔다."""
    images = page.get_images(full=True)
    boxes = [LayoutBox(label="image", bbox=tuple(page.get_image_bbox(img))) for img in images]
    return PageLayout(
        has_figures=bool(boxes),
        has_text_layer=has_text_layer,
        boxes=boxes,
    )


def needs_heavy_path(layout: PageLayout) -> bool:
    """다이어그램의 분기 규칙: 그림이 있거나 텍스트 레이어가 없으면(스캔 문서)
    AzureDI+VLM, 텍스트만 있고 텍스트 레이어가 있으면 네이티브 추출."""
    return layout.has_figures or not layout.has_text_layer
