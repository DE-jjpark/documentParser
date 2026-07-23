"""페이지 레이아웃 영역(25개 카테고리)을 실제 페이지 위에 카테고리별 색
사각형+라벨로 표시해서 육안으로 확인할 수 있는 중간 산출물을 만든다. 두
소스를 나란히 비교할 수 있게 파일을 따로 만든다:
  - <파일명>.pdf      -- PP-DocLayoutV2(전용 레이아웃 모델) 감지 결과
  - <파일명>_vlm.pdf  -- VLM(페이지 이미지 + 25개 카테고리 프롬프트)이 직접
    감지한 결과 -- 전용 모델 없이 VLM만으로 레이아웃을 얼마나 잘 뽑는지
    비교하기 위한 것.

페이지를 이미지로 따로 안 뽑고, 원본 PDF 위에 벡터 도형/텍스트를 직접
그려서 파일 하나(PDF)로 저장한다 -- 여러 장 PNG를 하나씩 여는 것보다 한
PDF를 스크롤하며 보는 게 낫다는 요청에 따른 것.

pdf/docx/pptx/doc/ppt 아무거나 받는다(office 포맷은 LibreOffice로 PDF 변환
후 같은 경로를 탄다 -- office.py가 실제로 하는 것과 동일).

VLM 버전은 페이지마다 실제 VLM 호출이 발생한다(자격증명 필요, 문서 페이지
수만큼 시간이 걸림) -- 필요 없으면 --skip-vlm으로 끈다.

사용법:
  python scripts/render_layout_boxes.py test1.pptx test2.pdf test3.docx
  (기본 출력: layout_boxes/<파일명>.pdf + <파일명>_vlm.pdf,
   각각 첫 페이지는 카테고리 색 범례)
"""

from __future__ import annotations

import argparse
import colorsys
import json
import re
import sys
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


def _insert_legend_page(doc: pymupdf.Document, title: str) -> None:
    page = doc.new_page(0)
    page.insert_text((36, 36), title, fontsize=12)
    for i, label in enumerate(sorted(_COLORS)):
        y = 60 + i * 18
        color = _COLORS[label]
        page.draw_rect(pymupdf.Rect(36, y, 50, y + 12), color=color, fill=color)
        page.insert_text((56, y + 10), label, fontsize=10)


def render(path: Path, out_dir: Path) -> Path:
    pdf_bytes = _load_pdf_bytes(path)
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    total_boxes = 0
    for page in doc:
        layout = analyze_page(page)
        for box in layout.boxes:
            color = _COLORS[box.label]
            rect = pymupdf.Rect(*box.bbox)
            page.draw_rect(rect, color=color, width=1.2)
            page.insert_text((rect.x0, max(rect.y0 - 3, 0)), box.label, color=color, fontsize=7)
            total_boxes += 1

    _insert_legend_page(doc, title="layout box legend (PP-DocLayoutV2, 25 categories)")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}.pdf"
    doc.save(str(out_path))
    page_count = doc.page_count - 1  # 범례 페이지 제외
    doc.close()
    print(f"  wrote {out_path} ({total_boxes} boxes across {page_count} pages)", file=sys.stderr)
    return out_path


_VLM_LAYOUT_PROMPT = (
    "이 페이지 이미지 안의 레이아웃 영역을 전부 감지해줘. 각 영역마다 카테고리와 "
    "위치(bounding box)를 알려줘. 카테고리는 반드시 다음 중에서만 골라라: "
    f"{', '.join(sorted(ALL_LABELS))}.\n\n"
    "좌표는 이미지 전체 가로/세로를 각각 0~1000으로 정규화한 값(왼쪽 위가 "
    "(0,0), 오른쪽 아래가 (1000,1000))으로 줘.\n\n"
    "답변은 반드시 JSON 배열 하나만 출력해라(다른 설명 문장 없이). 각 원소는 "
    '{"label": "<카테고리>", "bbox": [x0, y0, x1, y1]} 형식이다.'
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


def _vlm_boxes_for_page(
    client, page: pymupdf.Page
) -> list[tuple[str, tuple[float, float, float, float]]]:
    pix = page.get_pixmap(dpi=RENDER_DPI)
    result = caption_with_hard_timeout(client, pix.tobytes("png"), _VLM_LAYOUT_PROMPT)
    normalized = _parse_vlm_boxes(result.text)

    boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    for label, (x0, y0, x1, y1) in normalized:
        px_bbox = (
            x0 / 1000 * pix.width,
            y0 / 1000 * pix.height,
            x1 / 1000 * pix.width,
            y1 / 1000 * pix.height,
        )
        boxes.append((label, px_to_pt(px_bbox)))
    return boxes


def render_vlm(path: Path, out_dir: Path) -> Path:
    pdf_bytes = _load_pdf_bytes(path)
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    client = get_client()

    total_boxes = 0
    for page_number, page in enumerate(doc, start=1):
        boxes = _vlm_boxes_for_page(client, page)
        for label, bbox in boxes:
            color = _COLORS[label]
            rect = pymupdf.Rect(*bbox)
            page.draw_rect(rect, color=color, width=1.2)
            page.insert_text((rect.x0, max(rect.y0 - 3, 0)), label, color=color, fontsize=7)
            total_boxes += 1
        print(f"    page {page_number}/{doc.page_count}: {len(boxes)} box(es)", file=sys.stderr)

    _insert_legend_page(doc, title="layout box legend (VLM, 25-category prompt)")

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
    parser.add_argument(
        "--skip-vlm", action="store_true", help="VLM 버전은 안 만들고 PP-DocLayoutV2만"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for f in args.files:
        path = Path(f)
        print(f"processing {path} (PP-DocLayoutV2)...", file=sys.stderr)
        render(path, out_dir)
        if not args.skip_vlm:
            print(f"processing {path} (VLM)...", file=sys.stderr)
            render_vlm(path, out_dir)


if __name__ == "__main__":
    main()
