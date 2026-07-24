"""오프라인(인터넷 접근 없는) 환경에 document-parser를 설치하기 위한 번들을
만든다 -- pip이 PyPI에 못 붙는 Databricks 클러스터 같은 곳이 대상이다.

이 스크립트는 인터넷이 되는 머신(지금 이 저장소를 개발하는 머신)에서 한 번
실행해서 ``offline_bundle/``을 만들고, 그 디렉터리를 통째로 대상 머신에
옮긴 뒤 거기서 ``install.py``를 돌리는 두 단계 흐름이다.

받는 것:
  1. document-parser 자체 wheel (``uv build --wheel``)
  2. pdf/layout/vlm extra의 모든 의존성 -- 대상 플랫폼(기본: linux x86_64)
     /파이썬 버전(기본: 3.11) 태그의 prebuilt wheel만 받는다
     (``--only-binary=:all:``) -- sdist만 있는 패키지가 있으면 여기서
     바로 실패하는 게 낫다(대상 머신에 컴파일러가 없을 가능성이 높음).
  3. paddlepaddle -- PyPI가 아니라 PaddlePaddle 자체 인덱스에서 받는다
     (pyproject.toml의 'layout' extra 주석 참고). 이 인덱스는 simple
     repository 프로토콜을 완전히 안 지켜서(버전/플랫폼 필터링이 pip
     플랫폼 태그 매칭과 안 맞음, 실측 확인) 직접 URL을 구성해서 받는다.
  4. PP-DocLayoutV2 가중치 (``document_parser.parsing.weights``가 이미
     아는 방법 그대로 재사용 -- HuggingFace에서 받아서 이 리비전이 그대로
     대상 머신에서도 재현되게 고정).

용량 참고(실측, 2026-07-23 기준): wheels ~450MB + 가중치 ~204MB ≈ 650MB.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# PaddlePaddle 자체 인덱스는 pip이 신뢰할 수 있게 필터링을 못 해서(실측:
# --only-binary=:all: + --platform linux_x86_64 + --python-version 311 조합으로
# 3.3.1을 요청해도 "no matching distribution"이 남 -- 3.0.0만 찾힘) 실제
# 파일이 있는 BOS 버킷 URL을 직접 구성한다. 이 인덱스 페이지
# (https://www.paddlepaddle.org.cn/packages/stable/cpu/paddlepaddle/)에서
# 실제 파일명 규칙을 확인했다.
_PADDLE_URL_TEMPLATE = (
    "https://paddle-whl.bj.bcebos.com/stable/cpu/paddlepaddle/"
    "paddlepaddle-{version}-cp{py_tag}-cp{py_tag}-{platform}.whl"
)
_PADDLE_VERSION = "3.3.1"
# paddlepaddle 자체의 추가 의존성(uv pip install로 실측 확인: paddlepaddle
# 외에 networkx/opt-einsum 두 개만 더 필요) -- 둘 다 순수 파이썬이라 플랫폼
# 무관하게 받는다.
_PADDLE_EXTRA_DEPS = ["networkx", "opt-einsum"]


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)


def build_own_wheel(dest: Path) -> Path:
    _run(["uv", "build", "--wheel", "--out-dir", str(dest)])
    wheels = sorted(dest.glob("document_parser-*.whl"))
    if not wheels:
        raise RuntimeError(f"document-parser wheel not found in {dest} after build")
    return wheels[-1]


def download_dependencies(
    own_wheel: Path,
    dest: Path,
    extras: str,
    platform_tags: list[str],
    python_version: str,
    abi: str,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--dest",
        str(dest),
        "--python-version",
        python_version,
        "--implementation",
        "cp",
        "--abi",
        abi,
        "--only-binary=:all:",
    ]
    for tag in platform_tags:
        cmd += ["--platform", tag]
    cmd.append(f"{own_wheel}[{extras}]")
    _run(cmd)


def download_paddlepaddle(dest: Path, platform_tag: str, py_tag: str) -> None:
    url = _PADDLE_URL_TEMPLATE.format(version=_PADDLE_VERSION, py_tag=py_tag, platform=platform_tag)
    target = dest / Path(url).name
    print(f"downloading {url}", file=sys.stderr)
    urllib.request.urlretrieve(url, target)  # noqa: S310 -- 고정된 신뢰 호스트, 사용자 입력 아님

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--dest",
        str(dest),
        "--python-version",
        py_tag,
        "--implementation",
        "cp",
        "--abi",
        f"cp{py_tag}",
        "--only-binary=:all:",
        "--no-deps",
        *_PADDLE_EXTRA_DEPS,
    ]
    _run(cmd)


def download_weights(dest: Path, revision: str | None) -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from document_parser.parsing.weights import download_layout_model

    kwargs = {"revision": revision} if revision else {}
    download_layout_model(dest=dest, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", default=str(REPO_ROOT / "offline_bundle"), help="번들 출력 디렉터리"
    )
    parser.add_argument(
        "--extras", default="pdf,layout,vlm", help="포함할 extras (콤마 구분, 기본: pdf,layout,vlm)"
    )
    parser.add_argument(
        "--platform-tag",
        default="manylinux_2_28_x86_64",
        help="pip download --platform 값(기본: manylinux_2_28_x86_64). "
        "필요하면 여러 번 줄 수 있게 콤마로 구분한 값도 받는다.",
    )
    parser.add_argument("--python-version", default="311", help="대상 파이썬 버전(기본: 311)")
    parser.add_argument("--skip-paddlepaddle", action="store_true")
    parser.add_argument("--skip-weights", action="store_true")
    parser.add_argument(
        "--weights-revision", default=None, help="가중치 HF revision override(기본: 코드에 핀된 값)"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    wheels_dir = out_dir / "wheels"
    models_dir = out_dir / "models"
    wheels_dir.mkdir(parents=True, exist_ok=True)

    own_wheel = build_own_wheel(wheels_dir)
    print(f"built {own_wheel.name}", file=sys.stderr)

    platform_tags = [
        # manylinux 규칙이 여러 세대라(2014/2_17/2_28) 패키지마다 어느 태그로
        # 배포됐는지 달라서, pip이 알아서 맞는 걸 고르게 세 개 다 준다(실측
        # 확인: numpy/pandas/onnxruntime는 2_28, pymupdf는 2_28, 대부분의
        # C 확장은 2014/2_17).
        "manylinux2014_x86_64",
        "manylinux_2_17_x86_64",
        args.platform_tag,
    ]
    download_dependencies(
        own_wheel,
        wheels_dir,
        args.extras,
        platform_tags=platform_tags,
        python_version=args.python_version,
        abi=f"cp{args.python_version}",
    )

    if not args.skip_paddlepaddle and "layout" in args.extras.split(","):
        download_paddlepaddle(wheels_dir, args.platform_tag, args.python_version)

    if not args.skip_weights and "layout" in args.extras.split(","):
        # 실제 캐시 레이아웃(weights.py의 layout_model_dir())을 그대로
        # 미러링한다 -- install.py가 이 하위 디렉터리를 통째로 대상 머신의
        # 캐시 경로에 복사하기만 하면 되게.
        layout_dir = models_dir / "PP-DocLayoutV2"
        layout_dir.mkdir(parents=True, exist_ok=True)
        download_weights(layout_dir, args.weights_revision)

    wheel_count = len(list(wheels_dir.glob("*.whl")))
    print(f"\ndone: {out_dir}", file=sys.stderr)
    print(f"  wheels: {wheel_count} files in {wheels_dir}", file=sys.stderr)
    if models_dir.exists():
        print(f"  weights: {models_dir}", file=sys.stderr)
    print(
        "\n대상 머신으로 이 디렉터리를 통째로 옮긴 뒤 "
        "`python install.py --bundle-dir <옮긴 경로>`를 실행하세요.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
