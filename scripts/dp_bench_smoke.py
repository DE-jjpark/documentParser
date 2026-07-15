"""DP-Bench(Upstage) 데이터셋으로 ParsingEngine을 실제로 돌려보는 수동 검증
스크립트. pytest 스위트에 넣지 않은 이유: 문서 200개 x 실제 Azure DI/VLM
라이브 호출이면 너무 느리고 비용도 든다 — 이건 필요할 때 사람이 직접
돌리는 스모크 테스트용이다.

데이터셋은 이미 사이드 프로젝트(skep_parser 작업)에서 git-lfs로 받아둔 걸
그대로 가리킨다(중복 다운로드 안 함):
  /Users/doracoon/Claude/Projects/parser/experiments/dp-bench-comparison/
  dp-bench/dataset/pdfs/  (200개 PDF, ~36MB)

사용법:
  uv run python scripts/dp_bench_smoke.py                # 전체 200개
  uv run python scripts/dp_bench_smoke.py --limit 20      # 앞 20개만
  uv run python scripts/dp_bench_smoke.py --pdfs-dir /other/path
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from document_parser import ParsingEngine
from document_parser.core.exceptions import DocumentParserError

DEFAULT_PDFS_DIR = Path(
    "/Users/doracoon/Claude/Projects/parser/experiments/dp-bench-comparison/dp-bench/dataset/pdfs"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdfs-dir", type=Path, default=DEFAULT_PDFS_DIR)
    parser.add_argument("--limit", type=int, default=None, help="앞에서부터 N개만 처리")
    parser.add_argument(
        "--out", type=Path, default=Path("dp_bench_smoke_results.json"), help="요약 결과 저장 경로"
    )
    args = parser.parse_args()

    pdf_paths = sorted(args.pdfs_dir.glob("*.pdf"))
    if args.limit:
        pdf_paths = pdf_paths[: args.limit]

    print(f"대상 PDF: {len(pdf_paths)}개 (경로: {args.pdfs_dir})")

    engine = ParsingEngine()
    results = []
    ok = 0
    failed = 0
    t0 = time.time()

    for i, path in enumerate(pdf_paths, start=1):
        try:
            document = engine.parse(path)
            type_counts: dict[str, int] = {}
            for el in document.elements:
                type_counts[el.type.value] = type_counts.get(el.type.value, 0) + 1
            results.append(
                {
                    "file": path.name,
                    "status": "ok",
                    "element_count": len(document.elements),
                    "type_counts": type_counts,
                }
            )
            ok += 1
        except DocumentParserError as exc:
            results.append({"file": path.name, "status": "error", "error": str(exc)})
            failed += 1

        if i % 10 == 0 or i == len(pdf_paths):
            print(f"  [{i}/{len(pdf_paths)}] ok={ok} failed={failed}")

    elapsed = time.time() - t0
    print(f"완료: ok={ok} failed={failed} (소요 {elapsed:.1f}초)")

    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"결과 저장 -> {args.out}")


if __name__ == "__main__":
    main()
