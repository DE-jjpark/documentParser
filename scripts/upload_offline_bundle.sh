#!/usr/bin/env bash
# offline_bundle/(scripts/build_offline_bundle.py 산출물)을 Databricks Unity
# Catalog Volume으로 올린다. 로컬에 databricks CLI가 설치·인증돼 있어야
# 한다(databricks auth login, 또는 DATABRICKS_HOST/DATABRICKS_TOKEN 환경변수).
#
# 사용법:
#   scripts/upload_offline_bundle.sh <catalog>.<schema>.<volume> [local_bundle_dir]
#
# 예:
#   scripts/upload_offline_bundle.sh main.default.skep_parser offline_bundle
#
# 참고: databricks CLI 버전에 따라 `fs cp`의 재귀/덮어쓰기 플래그 이름이
# 바뀔 수 있다(실행 전 `databricks fs cp --help`로 확인 권장) -- 아래는
# 최신 unified CLI(databricks CLI v0.2xx) 기준.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <catalog>.<schema>.<volume> [local_bundle_dir]" >&2
  exit 1
fi

VOLUME_PATH="$1"
BUNDLE_DIR="${2:-offline_bundle}"
CATALOG="$(echo "$VOLUME_PATH" | cut -d. -f1)"
SCHEMA="$(echo "$VOLUME_PATH" | cut -d. -f2)"
VOLUME="$(echo "$VOLUME_PATH" | cut -d. -f3)"

if [ -z "$CATALOG" ] || [ -z "$SCHEMA" ] || [ -z "$VOLUME" ]; then
  echo "error: expected <catalog>.<schema>.<volume>, got: $VOLUME_PATH" >&2
  exit 1
fi

if [ ! -d "$BUNDLE_DIR/wheels" ]; then
  echo "error: $BUNDLE_DIR/wheels not found -- scripts/build_offline_bundle.py 먼저 실행하세요" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# install.py는 저장소 루트에 있다 -- 대상 머신에서 바로 `python install.py`가
# 되려면(기본 --bundle-dir가 스크립트 자기 위치 기준) 번들 안에도 같이
# 있어야 한다.
cp "$SCRIPT_DIR/../install.py" "$BUNDLE_DIR/install.py"

REMOTE="dbfs:/Volumes/$CATALOG/$SCHEMA/$VOLUME/offline_bundle"
echo "uploading $BUNDLE_DIR -> $REMOTE" >&2
databricks fs cp "$BUNDLE_DIR" "$REMOTE" --recursive --overwrite

echo "" >&2
echo "done. 클러스터 노트북에서(%sh 셀 또는 웹터미널):" >&2
echo "  python /Volumes/$CATALOG/$SCHEMA/$VOLUME/offline_bundle/install.py" >&2
