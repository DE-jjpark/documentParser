"""페이지 레이아웃 영역(25개 카테고리)을 실제 페이지 위에 카테고리별 색
사각형+라벨로 표시해서 육안으로 확인할 수 있는 중간 산출물을 만든다. 세
소스를 나란히 비교할 수 있게 파일을 따로 만든다:
  - <파일명>.pdf      -- PP-DocLayoutV2(전용 레이아웃 모델) 감지 결과
  - <파일명>_v3.pdf   -- PP-DocLayoutV3 감지 결과. paddleocr의 PaddleOCRVL
    파이프라인(레이아웃 감지 + 별도 0.9B VL 텍스트 인식 모델 ~2GB, 로컬
    CPU 추론)은 안 쓰고, paddlex.create_model("PP-DocLayoutV3")로 레이아웃
    감지 모델만 직접 불러온다 -- V2와 같은 비교축(레이아웃 감지 자체)을
    보려는 거라 무거운 VL 인식 단계는 필요 없다. 실측 확인: 이 모델은
    coordinate(4점 bbox, 픽셀 좌표)를 그대로 주므로 다각형 처리가 따로
    필요 없었다.
  - <파일명>_vlm.pdf  -- VLM(페이지 이미지 + 25개 카테고리 프롬프트)이 직접
    감지한 결과 -- 전용 모델 없이 VLM만으로 레이아웃을 얼마나 잘 뽑는지
    비교하기 위한 것.

페이지를 이미지로 따로 안 뽑고, 원본 PDF 위에 벡터 도형/텍스트를 직접
그려서 파일 하나(PDF)로 저장한다 -- 여러 장 PNG를 하나씩 여는 것보다 한
PDF를 스크롤하며 보는 게 낫다는 요청에 따른 것.

pdf/docx/pptx/doc/ppt 아무거나 받는다(office 포맷은 LibreOffice로 PDF 변환
후 같은 경로를 탄다 -- office.py가 실제로 하는 것과 동일).

VLM 버전은 페이지마다 실제 VLM 호출이 발생한다(자격증명 필요, 문서 페이지
수만큼 시간이 걸림) -- 필요 없으면 --skip-vlm으로 끈다. V3 버전은 첫 실행
때 가중치(~200MB)를 자동으로 받는다(~/.paddlex/official_models/PP-DocLayoutV3
-- 이 프로젝트의 weights.py가 관리하는 캐시와는 별개, 순수 비교용 실험이라
그대로 둠) -- 필요 없으면 --skip-v3로 끈다.

사용법:
  python scripts/render_layout_boxes.py test1.pptx test2.pdf test3.docx
  (기본 출력: layout_boxes/<파일명>.pdf + <파일명>_v3.pdf + <파일명>_vlm.pdf,
   각각 첫 페이지는 카테고리 색 범례)
"""

from __future__ import annotations

import argparse
import colorsys
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pymupdf  # noqa: E402

from document_parser.parsing.loaders.pdf.coords import RENDER_DPI, px_to_pt  # noqa: E402
from document_parser.parsing.loaders.pdf.layout import ALL_LABELS, analyze_page  # noqa: E402
from document_parser.parsing.loaders.vlm_caption import (  # noqa: E402
    caption_with_hard_timeout,
    get_client,
)

# doc_title/paragraph_title/figure_title은 이전(heading 전용) 버전과 같은
# 색을 유지해서 헷갈리지 않게 한다. 나머지 22개는 HSV 색상환을 고르게
# 나눠 자동 배정 -- 25개를 손으로 다 고르는 건 유지보수 부담만 크다.
_FIXED_COLORS: dict[str, tuple[float, float, float]] = {
    "doc_title": (1, 0, 0),
    "paragraph_title": (0, 0.3, 1),
    "figure_title": (0, 0.6, 0),
}


def _build_color_map(labels: tuple[str, ...]) -> dict[str, tuple[float, float, float]]:
    colors = dict(_FIXED_COLORS)
    remaining = sorted(label for label in labels if label not in colors)
    n = len(remaining)
    for i, label in enumerate(remaining):
        hue = i / n
        colors[label] = colorsys.hsv_to_rgb(hue, 0.7, 0.75)
    return colors


_COLORS = _build_color_map(ALL_LABELS)


def _load_pdf_bytes(path: Path) -> bytes:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return path.read_bytes()
    if suffix in (".pptx", ".ppt", ".docx", ".doc"):
        from document_parser.parsing.loaders.office import _convert

        return _convert(path.read_bytes(), suffix, "pdf")
    raise ValueError(f"unsupported extension: {suffix} ({path})")


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _containment_ratio(
    inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]
) -> float:
    """inner가 outer 안에 얼마나(비율로) 들어있는지 -- inner 면적 기준(IoU가
    아니라)이라 outer가 훨씬 커도 inner가 완전히 그 안에 있으면 1.0이 된다."""
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    inner_area = _bbox_area(inner)
    return intersection / inner_area if inner_area > 0 else 0.0


_BoxList = list[tuple[str, tuple[float, float, float, float]]]


def _suppress_nested(boxes: _BoxList, threshold: float) -> _BoxList:
    """더 큰 박스 안에 threshold 비율 이상 포함된 박스를 제거한다(라벨
    무관) -- "표/카드처럼 큰 박스 하나로 감지됐는데 그 안의 글자들이 잘게
    text로 또 감지돼서" 겹쳐 그려지는 걸 줄여달라는 요청에 따른 것. 자기보다
    "엄격히 더 큰" 박스 기준으로만 억제한다(같은 크기끼리는 서로 안 지움)."""
    result: _BoxList = []
    for i, (label, bbox) in enumerate(boxes):
        area = _bbox_area(bbox)
        suppressed = any(
            j != i
            and _bbox_area(other_bbox) > area
            and _containment_ratio(bbox, other_bbox) >= threshold
            for j, (_, other_bbox) in enumerate(boxes)
        )
        if not suppressed:
            result.append((label, bbox))
    return result


def _insert_legend_page(doc: pymupdf.Document, title: str) -> None:
    page = doc.new_page(0)
    page.insert_text((36, 36), title, fontsize=12)
    for i, label in enumerate(sorted(_COLORS)):
        y = 60 + i * 18
        color = _COLORS[label]
        page.draw_rect(pymupdf.Rect(36, y, 50, y + 12), color=color, fill=color)
        page.insert_text((56, y + 10), label, fontsize=10)


def render(
    path: Path, out_dir: Path, suppress_nested: bool = False, nested_threshold: float = 0.8
) -> Path:
    pdf_bytes = _load_pdf_bytes(path)
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    total_boxes = 0
    for page in doc:
        layout = analyze_page(page)
        boxes: _BoxList = [(box.label, box.bbox) for box in layout.boxes]
        if suppress_nested:
            boxes = _suppress_nested(boxes, nested_threshold)
        for label, bbox in boxes:
            color = _COLORS[label]
            rect = pymupdf.Rect(*bbox)
            page.draw_rect(rect, color=color, width=1.2)
            page.insert_text((rect.x0, max(rect.y0 - 3, 0)), label, color=color, fontsize=7)
            total_boxes += 1

    title = "layout box legend (PP-DocLayoutV2, 25 categories)"
    if suppress_nested:
        title += ", nested boxes suppressed"
    _insert_legend_page(doc, title=title)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}.pdf"
    doc.save(str(out_path))
    page_count = doc.page_count - 1  # 범례 페이지 제외
    doc.close()
    print(f"  wrote {out_path} ({total_boxes} boxes across {page_count} pages)", file=sys.stderr)
    return out_path


def _get_v3_model():
    """PaddleOCRVL 파이프라인(레이아웃 감지 + 별도 0.9B VL 텍스트 인식 모델
    ~2GB) 전체는 안 띄우고, paddlex의 저수준 팩토리로 PP-DocLayoutV3
    레이아웃 감지 모델만 직접 불러온다 -- V2(paddleocr.LayoutDetection)와
    같은 비교축만 필요해서. 첫 호출 때 가중치(~200MB)를 자동으로 받는다."""
    import paddlex

    return paddlex.create_model("PP-DocLayoutV3")


def _v3_boxes_for_page(model, page: pymupdf.Page) -> _BoxList:
    pix = page.get_pixmap(dpi=RENDER_DPI)
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        pix.save(f.name)
        (result,) = model.predict(f.name)

    boxes: _BoxList = []
    for box in result["boxes"]:
        label = box["label"]
        if label not in _COLORS:
            print(f"    skipping unknown V3 label: {label}", file=sys.stderr)
            continue
        px_bbox = tuple(float(v) for v in box["coordinate"])
        boxes.append((label, px_to_pt(px_bbox)))
    return boxes


def render_v3(
    path: Path, out_dir: Path, suppress_nested: bool = False, nested_threshold: float = 0.8
) -> Path:
    pdf_bytes = _load_pdf_bytes(path)
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    model = _get_v3_model()

    total_boxes = 0
    for page in doc:
        boxes = _v3_boxes_for_page(model, page)
        if suppress_nested:
            boxes = _suppress_nested(boxes, nested_threshold)
        for label, bbox in boxes:
            color = _COLORS[label]
            rect = pymupdf.Rect(*bbox)
            page.draw_rect(rect, color=color, width=1.2)
            page.insert_text((rect.x0, max(rect.y0 - 3, 0)), label, color=color, fontsize=7)
            total_boxes += 1

    title = "layout box legend (PP-DocLayoutV3)"
    if suppress_nested:
        title += ", nested boxes suppressed"
    _insert_legend_page(doc, title=title)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_v3.pdf"
    doc.save(str(out_path))
    page_count = doc.page_count - 1  # 범례 페이지 제외
    doc.close()
    print(f"  wrote {out_path} ({total_boxes} boxes across {page_count} pages)", file=sys.stderr)
    return out_path


# bbox 정확도 보정: VLM은 좌표를 맨눈 추측으로 답하면 부정확하다(실측
# 확인: PP-DocLayoutV2 대비 훨씬 성기고 어긋난 박스). 이미지에 0~1000
# 좌표 눈금 격자를 미리 그려서 보내면(grid-overlay grounding 기법) VLM이
# 실제로 이미지에 찍힌 숫자를 읽어서 답할 수 있어 훨씬 정확해진다 --
# _grid_overlay_png()가 이 격자 이미지를 만들고, 최종 출력 PDF(원본
# page 객체)에는 격자를 안 남긴다(별도 스크래치 문서에만 그림).
_GRID_STEP = 50

_VLM_LAYOUT_PROMPT = (
    "이 페이지 이미지 안의 레이아웃 영역을 전부 감지해줘. 각 영역마다 카테고리와 "
    "위치(bounding box)를 알려줘. 카테고리는 반드시 다음 중에서만 골라라: "
    f"{', '.join(sorted(ALL_LABELS))}.\n\n"
    "이미지 위에 회색 좌표 격자가 그려져 있다 -- 위쪽 가장자리 숫자가 x좌표, "
    f"왼쪽 가장자리 숫자가 y좌표다({_GRID_STEP} 단위 눈금, 왼쪽 위가 (0,0), "
    "오른쪽 아래가 (1000,1000)). 이 눈금을 실제로 읽어서 각 영역의 경계가 "
    "어느 눈금 근처인지 최대한 정확히 맞춰 답해라 -- 대충 짐작하지 말 것.\n\n"
    "답변은 반드시 JSON 배열 하나만 출력해라(다른 설명 문장 없이). 각 원소는 "
    '{"label": "<카테고리>", "bbox": [x0, y0, x1, y1]} 형식이다(x0,y0,x1,y1 '
    "모두 0~1000 사이 정규화 좌표)."
)

_JSON_ARRAY = re.compile(r"\[[\s\S]*\]")


def _parse_vlm_boxes(text: str) -> list[tuple[str, tuple[float, float, float, float]]]:
    """VLM 응답에서 (label, 정규화된 0~1000 bbox) 목록을 뽑는다. heading_llm.py
    의 _parse_levels와 같은 관대한 원칙: 형식이 안 맞는 개별 원소는 그냥
    건너뛰고(전체를 버리지 않음) -- 여기선 "일부만 그려지는 것"이 heading
    레벨과 달리 위험하지 않다(육안 확인용이라 일부 박스 유실이 큰 문제가
    아님)."""
    match = _JSON_ARRAY.search(text)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []

    boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        bbox = item.get("bbox")
        if (
            isinstance(label, str)
            and label in _COLORS
            and isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(v, int | float) and not isinstance(v, bool) for v in bbox)
        ):
            boxes.append((label, (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))))
    return boxes


def _grid_overlay_png(page: pymupdf.Page) -> tuple[bytes, int, int]:
    """page 자체는 안 건드리고(최종 출력 PDF에 격자가 남으면 안 되므로)
    별도 스크래치 문서에 페이지를 복사해 그 위에만 0~1000 좌표 격자를
    그린다."""
    scratch = pymupdf.open()
    scratch.insert_pdf(page.parent, from_page=page.number, to_page=page.number)
    grid_page = scratch[0]
    rect = grid_page.rect
    gray = (0.6, 0.6, 0.6)
    for v in range(0, 1001, _GRID_STEP):
        x = rect.width * v / 1000
        y = rect.height * v / 1000
        grid_page.draw_line((x, 0), (x, rect.height), color=gray, width=0.3)
        grid_page.draw_line((0, y), (rect.width, y), color=gray, width=0.3)
        grid_page.insert_text((x + 1, 8), str(v), color=(0.8, 0, 0), fontsize=5)
        grid_page.insert_text((1, y + 5), str(v), color=(0.8, 0, 0), fontsize=5)
    pix = grid_page.get_pixmap(dpi=RENDER_DPI)
    png_bytes = pix.tobytes("png")
    width, height = pix.width, pix.height
    scratch.close()
    return png_bytes, width, height


def _vlm_boxes_for_page(
    client, page: pymupdf.Page
) -> list[tuple[str, tuple[float, float, float, float]]]:
    grid_png, width, height = _grid_overlay_png(page)
    result = caption_with_hard_timeout(client, grid_png, _VLM_LAYOUT_PROMPT)
    normalized = _parse_vlm_boxes(result.text)

    boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    for label, (x0, y0, x1, y1) in normalized:
        px_bbox = (
            x0 / 1000 * width,
            y0 / 1000 * height,
            x1 / 1000 * width,
            y1 / 1000 * height,
        )
        boxes.append((label, px_to_pt(px_bbox)))
    return boxes


def render_vlm(
    path: Path, out_dir: Path, suppress_nested: bool = False, nested_threshold: float = 0.8
) -> Path:
    pdf_bytes = _load_pdf_bytes(path)
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    client = get_client()

    total_boxes = 0
    for page_number, page in enumerate(doc, start=1):
        boxes = _vlm_boxes_for_page(client, page)
        if suppress_nested:
            boxes = _suppress_nested(boxes, nested_threshold)
        for label, bbox in boxes:
            color = _COLORS[label]
            rect = pymupdf.Rect(*bbox)
            page.draw_rect(rect, color=color, width=1.2)
            page.insert_text((rect.x0, max(rect.y0 - 3, 0)), label, color=color, fontsize=7)
            total_boxes += 1
        print(f"    page {page_number}/{doc.page_count}: {len(boxes)} box(es)", file=sys.stderr)

    title = "layout box legend (VLM, 25-category prompt)"
    if suppress_nested:
        title += ", nested boxes suppressed"
    _insert_legend_page(doc, title=title)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_vlm.pdf"
    doc.save(str(out_path))
    page_count = doc.page_count - 1  # 범례 페이지 제외
    doc.close()
    print(f"  wrote {out_path} ({total_boxes} boxes across {page_count} pages)", file=sys.stderr)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="pdf/docx/pptx/doc/ppt 파일 경로들")
    parser.add_argument("--out-dir", default="layout_boxes")
    parser.add_argument("--skip-v2", action="store_true", help="PP-DocLayoutV2 버전은 안 만듦")
    parser.add_argument("--skip-v3", action="store_true", help="PP-DocLayoutV3 버전은 안 만듦")
    parser.add_argument("--skip-vlm", action="store_true", help="VLM 버전은 안 만듦")
    parser.add_argument(
        "--suppress-nested",
        action="store_true",
        help="더 큰 박스 안에 거의 포함된(기본 80%%) 박스는 안 그림 -- "
        "예: 표/카드 박스 하나 안에 text가 잘게 또 잡힌 경우 정리",
    )
    parser.add_argument(
        "--nested-threshold",
        type=float,
        default=0.8,
        help="--suppress-nested의 포함 비율 임계값(0~1, 기본 0.8)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for f in args.files:
        path = Path(f)
        if not args.skip_v2:
            print(f"processing {path} (PP-DocLayoutV2)...", file=sys.stderr)
            render(path, out_dir, args.suppress_nested, args.nested_threshold)
        if not args.skip_v3:
            print(f"processing {path} (PP-DocLayoutV3)...", file=sys.stderr)
            render_v3(path, out_dir, args.suppress_nested, args.nested_threshold)
        if not args.skip_vlm:
            print(f"processing {path} (VLM)...", file=sys.stderr)
            render_vlm(path, out_dir, args.suppress_nested, args.nested_threshold)


if __name__ == "__main__":
    main()
