"""단독 이미지 파일(png/jpg/jpeg/webp) 로더 — 크롭 없이 파일 전체를 그대로 VLM에 보낸다.

PDF 안에서 발견된 그림(pdf/vlm.py의 caption_figures)과 프롬프트·응답 파싱
로직은 동일하다(vlm_caption.py 공용 모듈) — 여기서는 페이지/레이아웃 개념이
없으므로 크롭 없이 파일 바이트를 통째로 보내고, page/bboxes는 채우지 않는다
(txt/md 로더와 같은 패턴 — "좌표계 없는 포맷"은 비워둠).
"""

from __future__ import annotations

from pathlib import PurePath

from document_parser.core.models import DocumentElement, ElementType
from document_parser.parsing.loaders.vlm_caption import (
    PROMPT,
    caption_with_hard_timeout,
    get_client,
)
from document_parser.parsing.loaders.vlm_caption import extract_mermaid as _extract_mermaid
from document_parser.parsing.loaders.vlm_caption import split_content_summary as _split

FORMATS = ("png", "jpg", "jpeg", "webp")

_MIME_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


def _mime_type(source: str) -> str:
    suffix = PurePath(source).suffix.lstrip(".").lower()
    return _MIME_TYPES.get(suffix, "image/png")


def load(data: bytes, source: str, tier: str = "balanced") -> list[DocumentElement]:
    if tier == "fast":
        # 이미지 로더는 애초에 VLM이 유일한 콘텐츠 출처라(native 텍스트가
        # 있을 수 없음) 대체 경로가 없다 — 그냥 스킵하지 않고, "감지는 했지만
        # 캡션을 안 만들었다"는 걸 명시적으로 남긴다.
        return [
            DocumentElement(
                type=ElementType.IMAGE,
                text="",
                metadata={"source": "skipped_fast_tier"},
            )
        ]

    client = get_client()
    result = caption_with_hard_timeout(client, data, PROMPT, mime_type=_mime_type(source))
    content, summary = _split(result.text)
    text, mermaid = _extract_mermaid(content)

    metadata = {"source": "vlm"}
    if mermaid is not None:
        metadata["mermaid"] = mermaid
    if result.usage is not None:
        metadata["vlm_usage"] = result.usage

    return [DocumentElement(type=ElementType.IMAGE, text=text, summary=summary, metadata=metadata)]
