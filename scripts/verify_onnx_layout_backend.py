"""layout.py의 analyze_page()(paddle 백엔드)와 analyze_page_onnx()
(onnxruntime/HPI 백엔드)를 같은 문서에서 실행해서 정확도·속도를 비교한다.

**linux 전용** -- HPI 플러그인(ultra-infer-python)이 macOS wheel을 아예 안
내놔서 로컬(macOS) 개발 머신에서는 이 스크립트를 못 돌린다. Databricks
(linux) 클러스터에서 실행해서 실제로 검증해야 한다.

사전 준비(클러스터에서 한 번):
  paddlex --install hpi-cpu -y

사용법:
  python scripts/verify_onnx_layout_backend.py test1.pptx test2.pdf test3.docx
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pymupdf  # noqa: E402

from document_parser.parsing.loaders.pdf.layout import (  # noqa: E402
    analyze_page,
    analyze_page_onnx,
)


def _load_pdf_bytes(path: Path) -> bytes:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return path.read_bytes()
    if suffix in (".pptx", ".ppt", ".docx", ".doc"):
        from document_parser.parsing.loaders.office import _convert

        return _convert(path.read_bytes(), suffix, "pdf")
    raise ValueError(f"unsupported extension: {suffix} ({path})")


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compare(path: Path) -> None:
    pdf_bytes = _load_pdf_bytes(path)
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    total_paddle = 0
    total_onnx = 0
    total_matched = 0
    paddle_time = 0.0
    onnx_time = 0.0
    label_mismatches = 0

    for page_number, page in enumerate(doc, start=1):
        t0 = time.time()
        paddle_layout = analyze_page(page)
        paddle_time += time.time() - t0

        t0 = time.time()
        onnx_layout = analyze_page_onnx(page)
        onnx_time += time.time() - t0

        paddle_boxes = paddle_layout.boxes
        onnx_boxes = list(onnx_layout.boxes)
        matched_onnx: set[int] = set()
        page_matched = 0
        page_mismatch = 0
        for pb in paddle_boxes:
            best_j, best_iou = None, 0.5
            for j, ob in enumerate(onnx_boxes):
                if j in matched_onnx:
                    continue
                iou = _iou(pb.bbox, ob.bbox)
                if iou > best_iou:
                    best_j, best_iou = j, iou
            if best_j is not None:
                matched_onnx.add(best_j)
                page_matched += 1
                if pb.label != onnx_boxes[best_j].label:
                    page_mismatch += 1

        total_paddle += len(paddle_boxes)
        total_onnx += len(onnx_boxes)
        total_matched += page_matched
        label_mismatches += page_mismatch
        print(
            f"  page {page_number}/{doc.page_count}: paddle={len(paddle_boxes)} "
            f"onnx={len(onnx_boxes)} matched(iou>=0.5)={page_matched} "
            f"label_mismatch={page_mismatch}",
            file=sys.stderr,
        )

    doc.close()
    print(f"\n=== {path.name} ===")
    print(f"paddle total boxes: {total_paddle} ({paddle_time:.1f}s)")
    print(f"onnx   total boxes: {total_onnx} ({onnx_time:.1f}s)")
    print(f"matched (IoU>=0.5): {total_matched}")
    print(f"label mismatches among matched: {label_mismatches}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="pdf/docx/pptx/doc/ppt 파일 경로들")
    args = parser.parse_args()

    for f in args.files:
        path = Path(f)
        print(f"processing {path}...", file=sys.stderr)
        compare(path)


if __name__ == "__main__":
    main()
