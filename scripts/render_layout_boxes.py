"""PP-DocLayoutV2가 감지한 영역 전체(25개 카테고리 다, heading 3개만이
아니라)를 실제 페이지 위에 카테고리별 색 사각형+라벨로 표시해서 육안으로
확인할 수 있는 중간 산출물을 만든다. 페이지를 이미지로 따로 안 뽑고,
원본 PDF 위에 벡터 도형/텍스트를 직접 그려서 파일 하나(PDF)로 저장한다
-- 여러 장 PNG를 하나씩 여는 것보다 한 PDF를 스크롤하며 보는 게 낫다는
요청에 따른 것.

pdf/docx/pptx/doc/ppt 아무거나 받는다(office 포맷은 LibreOffice로 PDF 변환
후 같은 경로를 탄다 -- office.py가 실제로 하는 것과 동일).

사용법:
  python scripts/render_layout_boxes.py test1.pptx test2.pdf test3.docx
  (기본 출력: layout_boxes/<파일명>.pdf, 첫 페이지는 카테고리 색 범례)
"""

from __future__ import annotations

import argparse
import colorsys
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pymupdf  # noqa: E402

from document_parser.parsing.loaders.pdf.layout import ALL_LABELS, analyze_page  # noqa: E402

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


def _insert_legend_page(doc: pymupdf.Document) -> None:
    page = doc.new_page(0)
    page.insert_text((36, 36), "layout box legend (PP-DocLayoutV2, 25 categories)", fontsize=12)
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

    _insert_legend_page(doc)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}.pdf"
    doc.save(str(out_path))
    page_count = doc.page_count - 1  # 범례 페이지 제외
    doc.close()
    print(f"  wrote {out_path} ({total_boxes} boxes across {page_count} pages)", file=sys.stderr)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="pdf/docx/pptx/doc/ppt 파일 경로들")
    parser.add_argument("--out-dir", default="layout_boxes")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for f in args.files:
        path = Path(f)
        print(f"processing {path}...", file=sys.stderr)
        render(path, out_dir)


if __name__ == "__main__":
    main()
