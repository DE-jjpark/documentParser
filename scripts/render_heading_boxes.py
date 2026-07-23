"""PP-DocLayoutV2가 heading으로 분류한 영역(doc_title/paragraph_title/
figure_title)을 실제 페이지 위에 사각형+카테고리로 표시한 이미지를 만든다 --
"실제로 뭘 제목으로 인식했는지"를 육안으로 확인하기 위한 중간 산출물.
헤딩이 하나도 없는 페이지는 건너뛴다(확인할 게 없어서).

pdf/docx/pptx/doc/ppt 아무거나 받는다(office 포맷은 LibreOffice로 PDF 변환
후 같은 경로를 탄다 -- office.py가 실제로 하는 것과 동일).

사용법:
  python scripts/render_heading_boxes.py test1.pptx test2.pdf test3.docx
  (기본 출력: heading_boxes/<파일명>/pXXX.png + <파일명>.json 요약)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pymupdf  # noqa: E402

from document_parser.parsing.loaders.pdf.layout import analyze_page  # noqa: E402

# doc_title/paragraph_title/figure_title만 표시한다 -- 나머지 22개 카테고리는
# 지금 확인 대상이 아니다(_LABEL_TO_TYPE, native.py 참고 -- 이 셋만 HEADING).
_COLORS: dict[str, tuple[float, float, float]] = {
    "doc_title": (1, 0, 0),
    "paragraph_title": (0, 0.3, 1),
    "figure_title": (0, 0.6, 0),
}


def _load_pdf_bytes(path: Path) -> bytes:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return path.read_bytes()
    if suffix in (".pptx", ".ppt", ".docx", ".doc"):
        from document_parser.parsing.loaders.office import _convert

        return _convert(path.read_bytes(), suffix, "pdf")
    raise ValueError(f"unsupported extension: {suffix} ({path})")


def render(path: Path, out_dir: Path) -> None:
    pdf_bytes = _load_pdf_bytes(path)
    target_dir = out_dir / path.stem
    target_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        for page_number, page in enumerate(doc, start=1):
            layout = analyze_page(page)
            headings = [b for b in layout.boxes if b.label in _COLORS]
            if not headings:
                continue

            for box in headings:
                color = _COLORS[box.label]
                rect = pymupdf.Rect(*box.bbox)
                page.draw_rect(rect, color=color, width=1.5)
                page.insert_text(
                    (rect.x0, max(rect.y0 - 4, 0)), box.label, color=color, fontsize=8
                )
                summary.append(
                    {
                        "page": page_number,
                        "block_type": box.label,
                        "bbox": list(box.bbox),
                        "order": box.order,
                    }
                )

            pix = page.get_pixmap(dpi=200)
            out_path = target_dir / f"p{page_number:03d}.png"
            pix.save(str(out_path))
            print(f"  wrote {out_path} ({len(headings)} heading box(es))", file=sys.stderr)
    finally:
        doc.close()

    summary_path = out_dir / f"{path.stem}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"  wrote {summary_path} ({len(summary)} heading box(es) total)", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="pdf/docx/pptx/doc/ppt 파일 경로들")
    parser.add_argument("--out-dir", default="heading_boxes")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for f in args.files:
        path = Path(f)
        print(f"processing {path}...", file=sys.stderr)
        render(path, out_dir)


if __name__ == "__main__":
    main()
