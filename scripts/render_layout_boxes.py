"""페이지 레이아웃 영역(25개 카테고리)을 실제 페이지 위에 카테고리별 색
사각형+라벨로 표시해서 육안으로 확인할 수 있는 중간 산출물을 만든다. 세
소스를 나란히 비교할 수 있게 파일을 따로 만든다:
  - <파일명>.pdf      -- PP-DocLayoutV2(전용 레이아웃 모델) 감지 결과. 원본
    production 코드(layout.py의 analyze_page)와 달리 정사각형 레터박스
    패딩을 적용한다(모델이 800x800 고정 입력이라 원본을 그대로 넣으면
    비정사각형 문서가 찌그러짐 -- _square_pad_png 참고).
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
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pymupdf  # noqa: E402

from document_parser.parsing.loaders.pdf.coords import RENDER_DPI, px_to_pt  # noqa: E402
from document_parser.parsing.loaders.pdf.layout import ALL_LABELS  # noqa: E402
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


def _square_pad_png(pix: pymupdf.Pixmap) -> bytes:
    """PP-DocLayoutV2/V3는 입력 크기가 고정(800x800, STATIC_SHAPE_MODEL_LIST)
    이고 keep_ratio=False라서, 원본 그대로 넣으면 세로 PDF나 16:9 PPT
    슬라이드가 정사각형으로 눌려 찌그러진다(실측 확인: img_size 오버라이드
    자체가 이 모델들에서 아예 막혀 있음). 대신 우리가 먼저 원본 비율 그대로
    정사각형 캔버스(흰 여백)에 좌상단 기준으로 얹어서 넣는다 -- 모델
    내부의 800x800 리사이즈가 "이미 정사각형인" 이미지에 적용되니 실제
    내용은 안 찌그러진다. 좌상단 기준(오프셋 0,0)이라 반환되는 bbox
    좌표가 원본 픽셀 좌표와 그대로 맞는다(별도 역변환 불필요)."""
    from PIL import Image

    img = Image.open(BytesIO(pix.tobytes("png")))
    side = max(pix.width, pix.height)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(img, (0, 0))
    out = BytesIO()
    canvas.save(out, format="PNG")
    return out.getvalue()


def _get_v2_model():
    """document_parser.parsing.loaders.pdf.layout.analyze_page()는 원본
    이미지를 그대로 모델에 넣어서(정사각형 패딩 없음) 쓴다 -- 이 스크립트는
    비교/실험 목적이라 여기서만 패딩된 버전을 별도로 돌린다. 가중치는
    이 프로젝트가 관리하는 캐시(weights.py)를 그대로 재사용."""
    from paddleocr import LayoutDetection

    from document_parser.parsing.weights import layout_model_dir

    return LayoutDetection(model_name="PP-DocLayoutV2", model_dir=str(layout_model_dir()))


_ScoredBoxList = list[tuple[str, tuple[float, float, float, float], float]]


def _v2_boxes_for_page(model, page: pymupdf.Page) -> _ScoredBoxList:
    pix = page.get_pixmap(dpi=RENDER_DPI)
    padded_png = _square_pad_png(pix)
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        f.write(padded_png)
        f.flush()
        (result,) = model.predict(f.name, batch_size=1, layout_nms=True)

    boxes: _ScoredBoxList = []
    for box in result["boxes"]:
        label = box["label"]
        if label not in _COLORS:
            print(f"    skipping unknown V2 label: {label}", file=sys.stderr)
            continue
        px_bbox = tuple(float(v) for v in box["coordinate"])
        boxes.append((label, px_to_pt(px_bbox), float(box.get("score", 0.0))))
    return boxes


def render(
    path: Path, out_dir: Path, suppress_nested: bool = False, nested_threshold: float = 0.8
) -> Path:
    pdf_bytes = _load_pdf_bytes(path)
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    model = _get_v2_model()

    total_boxes = 0
    for page in doc:
        boxes: _BoxList = [(label, bbox) for label, bbox, _score in _v2_boxes_for_page(model, page)]
        if suppress_nested:
            boxes = _suppress_nested(boxes, nested_threshold)
        for label, bbox in boxes:
            color = _COLORS[label]
            rect = pymupdf.Rect(*bbox)
            page.draw_rect(rect, color=color, width=1.2)
            page.insert_text((rect.x0, max(rect.y0 - 3, 0)), label, color=color, fontsize=7)
            total_boxes += 1

    title = "layout box legend (PP-DocLayoutV2, square-padded input)"
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


def _v3_boxes_for_page(model, page: pymupdf.Page) -> _ScoredBoxList:
    pix = page.get_pixmap(dpi=RENDER_DPI)
    padded_png = _square_pad_png(pix)
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        f.write(padded_png)
        f.flush()
        (result,) = model.predict(f.name)

    boxes: _ScoredBoxList = []
    for box in result["boxes"]:
        label = box["label"]
        if label not in _COLORS:
            print(f"    skipping unknown V3 label: {label}", file=sys.stderr)
            continue
        px_bbox = tuple(float(v) for v in box["coordinate"])
        boxes.append((label, px_to_pt(px_bbox), float(box.get("score", 0.0))))
    return boxes


def render_v3(
    path: Path, out_dir: Path, suppress_nested: bool = False, nested_threshold: float = 0.8
) -> Path:
    pdf_bytes = _load_pdf_bytes(path)
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    model = _get_v3_model()

    total_boxes = 0
    for page in doc:
        boxes: _BoxList = [(label, bbox) for label, bbox, _score in _v3_boxes_for_page(model, page)]
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
    "오른쪽 아래가 (1000,1000), 이미지 밖으로 나가는 값은 절대 없다). 이 눈금을 "
    "실제로 읽어서 각 영역의 경계가 어느 눈금 근처인지 최대한 정확히 맞춰 "
    "답해라 -- 대충 짐작하지 말 것. 모든 좌표는 반드시 0 이상 1000 이하여야 "
    "한다 -- 1000을 넘는 값은 절대 쓰지 마라(이미지 오른쪽/아래쪽 끝이 정확히 "
    "1000이다).\n\n"
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
    아님).

    실측 확인한 버그: 프롬프트에서 0~1000 범위를 명시해도 모델이 종종
    1000을 넘는 값(예: 1380)을 그냥 내놓는다 -- 이걸 검증 없이 그대로
    px_to_pt로 넘기면 박스가 페이지 밖으로 삐져나가게 그려진다. 여기서
    0~1000으로 클램프하고, 클램프 후 찌그러진(x1<=x0 또는 y1<=y0) 박스는
    버린다."""
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
        if not (
            isinstance(label, str)
            and label in _COLORS
            and isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(v, int | float) and not isinstance(v, bool) for v in bbox)
        ):
            continue
        x0 = min(max(float(bbox[0]), 0.0), 1000.0)
        y0 = min(max(float(bbox[1]), 0.0), 1000.0)
        x1 = min(max(float(bbox[2]), 0.0), 1000.0)
        y1 = min(max(float(bbox[3]), 0.0), 1000.0)
        if x1 <= x0 or y1 <= y0:
            continue
        boxes.append((label, (x0, y0, x1, y1)))
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


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    union = _bbox_area(a) + _bbox_area(b) - intersection
    return intersection / union if union > 0 else 0.0


# 이 IoU 이상이면 "같은 영역을 가리킨다"고 보고 매칭한다 -- 너무 낮으면
# 관련 없는 박스끼리 억지로 짝지어지고, 너무 높으면 실제로 같은 영역인데
# 두 모델이 살짝 다르게 잡은 것도 매칭 실패로 "둘 다 유니크한 박스"가 돼서
# 중복으로 그려진다.
_MATCH_IOU_THRESHOLD = 0.3


def _match_v2_v3(
    v2_boxes: _ScoredBoxList, v3_boxes: _ScoredBoxList
) -> tuple[list[tuple[int, int]], _ScoredBoxList, _ScoredBoxList]:
    """V2/V3 박스를 IoU로 매칭한다(greedy -- 각 V2 박스는 아직 안 잡힌 V3
    박스 중 IoU가 가장 높은 것과 짝짓는다). 반환: (매칭된 (v2_idx, v3_idx)
    쌍, V2에만 있는 박스, V3에만 있는 박스)."""
    matched: list[tuple[int, int]] = []
    matched_v3: set[int] = set()
    for i, (_, b2, _) in enumerate(v2_boxes):
        best_j, best_iou = None, _MATCH_IOU_THRESHOLD
        for j, (_, b3, _) in enumerate(v3_boxes):
            if j in matched_v3:
                continue
            iou = _iou(b2, b3)
            if iou > best_iou:
                best_j, best_iou = j, iou
        if best_j is not None:
            matched.append((i, best_j))
            matched_v3.add(best_j)
    matched_v2 = {i for i, _ in matched}
    v2_only = [v2_boxes[i] for i in range(len(v2_boxes)) if i not in matched_v2]
    v3_only = [v3_boxes[j] for j in range(len(v3_boxes)) if j not in matched_v3]
    return matched, v2_only, v3_only


_CONFLICT_PROMPT_HEADER = (
    "아래 페이지 이미지에 주황색 박스와 번호로 표시된 영역들은, 두 개의 서로 "
    "다른 레이아웃 감지 모델이 같은 위치에 서로 다른 카테고리를 붙인 곳이다. "
    "각 번호에 대해 실제 내용을 보고 더 정확한 카테고리를 골라라 -- 제시된 두 "
    "후보 중 하나를 골라도 되고, 둘 다 틀렸으면 올바른 카테고리를 직접 써도 "
    "된다. 카테고리는 반드시 다음 중에서만 골라라: {labels}.\n\n"
    "후보 목록(번호. v2 후보=..., v3 후보=...):\n{candidates}\n\n"
    "답변은 반드시 JSON 배열 하나만 출력해라(다른 설명 문장 없이). 각 원소는 "
    '{{"id": <번호>, "label": "<선택한 카테고리>"}} 형식이다.'
)


_ConflictSpan = tuple[int, tuple[float, float, float, float], tuple[float, float, float, float]]


def _conflict_overlay_png(page: pymupdf.Page, conflicts: list[_ConflictSpan]) -> bytes:
    """page 자체는 안 건드리고 별도 스크래치 문서에 충돌 영역(두 박스의
    합집합)만 번호와 함께 주황색으로 표시한다."""
    scratch = pymupdf.open()
    scratch.insert_pdf(page.parent, from_page=page.number, to_page=page.number)
    p = scratch[0]
    orange = (1, 0.5, 0)
    for cid, bbox_a, bbox_b in conflicts:
        union = (
            min(bbox_a[0], bbox_b[0]),
            min(bbox_a[1], bbox_b[1]),
            max(bbox_a[2], bbox_b[2]),
            max(bbox_a[3], bbox_b[3]),
        )
        rect = pymupdf.Rect(*union)
        p.draw_rect(rect, color=orange, width=2)
        p.insert_text((rect.x0, max(rect.y0 - 10, 0)), str(cid), color=orange, fontsize=10)
    pix = p.get_pixmap(dpi=RENDER_DPI)
    png_bytes = pix.tobytes("png")
    scratch.close()
    return png_bytes


def _parse_conflict_choices(text: str, expected_ids: set[int]) -> dict[int, str]:
    match = _JSON_ARRAY.search(text)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, list):
        return {}
    choices: dict[int, str] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        cid, label = item.get("id"), item.get("label")
        if (
            isinstance(cid, int)
            and not isinstance(cid, bool)
            and cid in expected_ids
            and isinstance(label, str)
            and label in _COLORS
        ):
            choices[cid] = label
    return choices


def _resolve_conflicts(
    client,
    page: pymupdf.Page,
    v2_boxes: _ScoredBoxList,
    v3_boxes: _ScoredBoxList,
    matched: list[tuple[int, int]],
) -> _ScoredBoxList:
    """매칭된 쌍 중 라벨이 같으면 LLM 호출 없이 score 높은 쪽 bbox를 채택하고,
    라벨이 다른(진짜 충돌인) 것만 모아서 페이지당 LLM 호출 한 번으로 판정한다
    -- 충돌마다 따로 호출하면 문서당 수십 번씩 불필요하게 부를 수 있어서."""
    resolved: _ScoredBoxList = []
    conflicts: list[tuple[int, str, tuple, float, str, tuple, float]] = []
    for v2i, v3i in matched:
        l2, b2, s2 = v2_boxes[v2i]
        l3, b3, s3 = v3_boxes[v3i]
        if l2 == l3:
            resolved.append((l2, b2, s2) if s2 >= s3 else (l3, b3, s3))
        else:
            conflicts.append((len(conflicts) + 1, l2, b2, s2, l3, b3, s3))

    if not conflicts:
        return resolved

    spans = [(cid, b2, b3) for cid, _, b2, _, _, b3, _ in conflicts]
    overlay_png = _conflict_overlay_png(page, spans)
    candidates_text = "\n".join(
        f"{cid}. v2 후보={l2}, v3 후보={l3}" for cid, l2, _, _, l3, _, _ in conflicts
    )
    prompt = _CONFLICT_PROMPT_HEADER.format(
        labels=", ".join(sorted(ALL_LABELS)), candidates=candidates_text
    )
    result = caption_with_hard_timeout(client, overlay_png, prompt)
    choices = _parse_conflict_choices(result.text, {c[0] for c in conflicts})

    for cid, l2, b2, s2, l3, b3, s3 in conflicts:
        chosen = choices.get(cid)
        if chosen == l3:
            resolved.append((l3, b3, s3))
        elif chosen is not None and chosen != l2:
            # LLM이 v2/v3 둘 다 아닌 라벨을 직접 줬다 -- bbox는 score 높은
            # 쪽을 쓰고 라벨만 교체.
            resolved.append((chosen, b2 if s2 >= s3 else b3, max(s2, s3)))
        else:
            # 응답이 없거나(파싱 실패) v2를 골랐으면 v2로 폴백.
            resolved.append((l2, b2, s2))
    return resolved


# VLM 전체 감지 결과 중 기존(V2/V3 병합) 박스와 이 비율 이상 겹치면
# "이미 다른 걸로 잡힌 영역"으로 보고 버린다 -- image/table로 잡힌 영역
# 안의 글자를 VLM이 별도 text로 또 잡거나, V2/V3가 이미 text로 잡은 걸
# VLM이 중복으로 또 잡는 걸 막기 위함. 0.3(기존 값)보다 낮춰서 살짝만
# 겹쳐도 보수적으로 버린다 -- "빠뜨린 텍스트 보충"이 목적이라 애매하면
# 안 넣는 쪽이 낫다(잘못 겹쳐 그려지는 것보다).
_GAP_FILL_OVERLAP_THRESHOLD = 0.15


def _gap_fill_boxes(vlm_boxes: _BoxList, existing: _ScoredBoxList) -> _ScoredBoxList:
    """V2/V3 둘 다 놓친 "text"만 VLM으로 보충한다 -- image/table/chart 등
    다른 카테고리는 V2/V3가 이미 전담하고 있어서 VLM 보간 대상에서 아예
    뺀다(카테고리 신뢰도가 V2/V3보다 낮기도 하고, 굳이 겹칠 이유가 없음).
    기존 박스(카테고리 무관 -- image/table 안이든 이미 잡힌 text든)와
    조금이라도 겹치면 그 영역은 이미 처리된 걸로 보고 버린다."""
    existing_bboxes = [b for _, b, _ in existing]
    gaps: _ScoredBoxList = []
    for label, bbox in vlm_boxes:
        if label != "text":
            continue
        max_overlap = max((_containment_ratio(bbox, eb) for eb in existing_bboxes), default=0.0)
        if max_overlap < _GAP_FILL_OVERLAP_THRESHOLD:
            # score=0: V2/V3 confidence가 아니라 VLM 보간이라는 표시
            gaps.append((label, bbox, 0.0))
    return gaps


def render_merged(
    path: Path, out_dir: Path, suppress_nested: bool = False, nested_threshold: float = 0.8
) -> Path:
    """V2 + V3 + VLM을 다 합친 버전 -- 겹치는 영역은 라벨이 같으면 score
    높은 쪽, 다르면 LLM이 페이지당 한 번에 판정하고, 어느 모델도 못 잡은
    영역은 VLM 전체 감지 결과에서 보간한다. 세 경로를 각각 다 돌리므로
    비용/시간이 제일 크다."""
    pdf_bytes = _load_pdf_bytes(path)
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    v2_model = _get_v2_model()
    v3_model = _get_v3_model()
    client = get_client()

    total_boxes = 0
    total_gap_filled = 0
    for page_number, page in enumerate(doc, start=1):
        v2_boxes = _v2_boxes_for_page(v2_model, page)
        v3_boxes = _v3_boxes_for_page(v3_model, page)
        matched, v2_only, v3_only = _match_v2_v3(v2_boxes, v3_boxes)
        merged = _resolve_conflicts(client, page, v2_boxes, v3_boxes, matched) + v2_only + v3_only

        vlm_boxes = _vlm_boxes_for_page(client, page)
        gaps = _gap_fill_boxes(vlm_boxes, merged)
        merged = merged + gaps

        boxes: _BoxList = [(label, bbox) for label, bbox, _score in merged]
        if suppress_nested:
            boxes = _suppress_nested(boxes, nested_threshold)
        for label, bbox in boxes:
            color = _COLORS[label]
            rect = pymupdf.Rect(*bbox)
            page.draw_rect(rect, color=color, width=1.2)
            page.insert_text((rect.x0, max(rect.y0 - 3, 0)), label, color=color, fontsize=7)
            total_boxes += 1
        total_gap_filled += len(gaps)
        print(
            f"    page {page_number}/{doc.page_count}: {len(boxes)} box(es) "
            f"({len(gaps)} gap-filled)",
            file=sys.stderr,
        )

    title = "layout box legend (V2+V3 merged, LLM-arbitrated + VLM gap-fill)"
    if suppress_nested:
        title += ", nested boxes suppressed"
    _insert_legend_page(doc, title=title)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_merged.pdf"
    doc.save(str(out_path))
    page_count = doc.page_count - 1  # 범례 페이지 제외
    doc.close()
    print(
        f"  wrote {out_path} ({total_boxes} boxes across {page_count} pages, "
        f"{total_gap_filled} gap-filled)",
        file=sys.stderr,
    )
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
    parser.add_argument(
        "--merge",
        action="store_true",
        help="V2+V3 병합 버전(<파일명>_merged.pdf)도 만듦 -- 겹치는 영역은 "
        "라벨 같으면 score 높은 쪽, 다르면 페이지당 LLM 호출 1회로 판정, "
        "어느 쪽도 못 잡은 영역은 VLM 전체 감지 결과에서 보간. V2/V3/VLM을 "
        "전부 내부적으로 다시 돌리므로(--skip-v2/v3/vlm과 무관) 비용/시간이"
        " 제일 큼",
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
        if args.merge:
            print(f"processing {path} (merged V2+V3+VLM)...", file=sys.stderr)
            render_merged(path, out_dir, args.suppress_nested, args.nested_threshold)


if __name__ == "__main__":
    main()
