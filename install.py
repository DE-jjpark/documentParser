"""오프라인 환경(pip이 PyPI에 못 붙는 곳, 예: 격리된 Databricks 클러스터)에서
document-parser를 설치한다. scripts/build_offline_bundle.py로 인터넷 되는
머신에서 미리 만든 번들(``wheels/`` + ``models/``)을 이 스크립트와 같이
대상 머신으로 옮긴 뒤 실행한다.

기본 가정: 이 스크립트가 번들 디렉터리 안(``wheels/``, ``models/``와 같은
위치)에 있다 -- 번들을 통째로 옮기면 그대로 동작. 다른 위치에 있으면
``--bundle-dir``로 지정한다.

동작:
  1. ``--find-links``로 로컬 wheels/만 보고 네트워크 없이 pip install
     (``pip install --no-index``) -- 도중에 PyPI를 찾으려는 시도가 있으면
     여기서 바로 실패한다(오프라인 보장이 깨졌다는 신호).
  2. models/PP-DocLayoutV2를 document_parser가 실제로 읽는 캐시 경로
     (parsing.weights.layout_model_dir(), 기본 ~/.cache/document-parser
     /models 또는 $DOCUMENT_PARSER_MODEL_DIR)로 복사.
  3. 가벼운 스모크 테스트: tier="fast"로 최소 PDF 하나를 실제로 파싱해서
     설치가 진짜 동작하는지 확인(가중치/VLM 자격증명은 여기선 안 건드림 --
     그건 별도로 DATABRICKS_HOST/TOKEN 설정하고 확인할 부분).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def install_wheels(wheels_dir: Path, extras: str, include_paddle: bool) -> None:
    packages = [f"document-parser[{extras}]"]
    if include_paddle:
        packages.append("paddlepaddle")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        str(wheels_dir),
        *packages,
    ]
    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)


def install_weights(bundle_models_dir: Path) -> Path:
    from document_parser.parsing.weights import layout_model_dir

    target = layout_model_dir()
    if target.exists() and any(target.iterdir()):
        print(f"weights already present at {target}, skipping copy", file=sys.stderr)
        return target
    target.mkdir(parents=True, exist_ok=True)
    for item in bundle_models_dir.iterdir():
        dest = target / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
    return target


def smoke_test() -> None:
    import importlib

    # 방금 이 프로세스 안에서 pip install을 했으므로, import 시스템의 경로
    # 캐시를 무효화해야 새로 설치된 패키지를 바로 찾는다(재시작 없이).
    importlib.invalidate_caches()
    from document_parser import ParsingEngine

    document = ParsingEngine().parse("smoke.txt", data=b"hello world", tier="fast")
    assert document.elements, "parsed document has no elements"
    print("smoke test ok: parsed a minimal document", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_bundle = Path(__file__).resolve().parent
    parser.add_argument(
        "--bundle-dir",
        default=str(default_bundle),
        help="wheels/ 와 models/ 를 담고 있는 디렉터리(기본: 이 스크립트가 있는 위치)",
    )
    parser.add_argument("--extras", default="pdf,layout,vlm")
    parser.add_argument("--skip-paddlepaddle", action="store_true")
    parser.add_argument("--skip-weights", action="store_true")
    parser.add_argument("--skip-smoke-test", action="store_true")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir).resolve()
    wheels_dir = bundle_dir / "wheels"
    models_dir = bundle_dir / "models" / "PP-DocLayoutV2"

    if not wheels_dir.is_dir():
        raise SystemExit(f"wheels directory not found: {wheels_dir}")

    include_paddle = not args.skip_paddlepaddle and "layout" in args.extras.split(",")
    install_wheels(wheels_dir, args.extras, include_paddle)

    if not args.skip_weights and "layout" in args.extras.split(","):
        if not models_dir.is_dir():
            raise SystemExit(f"weights directory not found: {models_dir}")
        target = install_weights(models_dir)
        print(f"weights installed at {target}", file=sys.stderr)

    if not args.skip_smoke_test:
        smoke_test()

    print("\ndocument-parser installed.", file=sys.stderr)


if __name__ == "__main__":
    main()
