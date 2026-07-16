"""JSON 로더 — 임의 구조를 key/value 텍스트로 평탄화한다(구조 보존 안 함).

중첩 dict는 점(.)으로, list는 인덱스([n])로 경로를 이어붙여서 "a.b[0].c: 값"
형태의 평문 한 줄로 만든다 — JSON 자체의 트리 구조를 다시 파싱해서 쓸 일이
없는 단순 텍스트 검색/청킹 용도라, 구조를 그대로 보존하는 것보다 평탄화가
더 쓸모 있다는 판단.
"""

from __future__ import annotations

import json as _json

from document_parser.core.exceptions import ParsingFailedError
from document_parser.core.models import DocumentElement, ElementType

FORMATS = ("json",)


def _flatten(value: object, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(value, dict):
        for key, val in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(_flatten(val, path))
    elif isinstance(value, list):
        for i, val in enumerate(value):
            lines.extend(_flatten(val, f"{prefix}[{i}]"))
    else:
        lines.append(f"{prefix}: {value}")
    return lines


def load(data: bytes, source: str) -> list[DocumentElement]:
    try:
        parsed = _json.loads(data.decode("utf-8", errors="replace"))
    except _json.JSONDecodeError as exc:
        raise ParsingFailedError(f"invalid JSON in {source}: {exc}") from exc

    lines = _flatten(parsed)
    if not lines:
        return []
    return [
        DocumentElement(
            type=ElementType.TEXT,
            text="\n".join(lines),
            metadata={"source": "json"},
        )
    ]
