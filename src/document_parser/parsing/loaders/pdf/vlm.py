"""VLM 그림/표 캡션·구조 추출 경로.

다이어그램 메모: AzureDI와 달리 먼저 크롭한 다음 VLM에 보낸다 — 페이지
전체보다 타이트하게 자른 영역이 정확도가 더 좋다.

실제 in4u Databricks AI Gateway(Claude Sonnet 4.6)로 연동 확인함 —
parsing.clients.vlm.VLMClient 참고. 크롭은 pymupdf의 get_pixmap(clip=...)로
하는데, clip은 포인트 좌표를 받으므로(픽셀 변환 불필요) box.bbox를 그대로
쓴다 — layout.py/azure_di.py에서 모델이 "돌려주는" 좌표를 변환해야 했던 것과는
반대 방향이라 혼동하지 말 것.

표(table) 처리: 표의 실제 행/열/병합 구조(HTML)는 AzureDI가 담당한다
(azure_di.py의 DetectedTable, graph.py의 merge 노드가 bbox로 매칭해서 채움).
여기 VLM은 표 구조를 다시 뽑지 않고, 대신 표 크롭 이미지를 보고 내용을
요약한 text만 만든다 — DI가 못 찾은 표(예: 선이 없는 표라 DI도 놓친 경우)는
merge에서 매칭이 안 돼 metadata에 html이 안 붙고 이 요약 text만 남는다.

이미지(그림/차트/순서도/서명 등) 처리: 종류에 따라 VLM한테 다르게 요청한다
(_PROMPT 참고) — 순서도·다이어그램이면 Mermaid 코드 블록으로 구조 자체를
받아서 metadata["mermaid"]에 담고(_extract_mermaid), 서명·손글씨면 캡션
대신 실제로 쓰인 글자를 그대로 옮겨 적게 하고, 그 외엔 평범한 설명. 전부
한국어로 요청한다(요약/설명은 한글로 보는 게 낫다는 결정).
"""

from __future__ import annotations

import concurrent.futures
import re
from functools import lru_cache
from typing import TYPE_CHECKING

from document_parser.core.models import BBox, DocumentElement, ElementType
from document_parser.parsing.clients.vlm import VLMCaptionResult, VLMClient

if TYPE_CHECKING:
    # 타입 힌트 전용 — layout.py 주석 참고.
    import pymupdf

    from document_parser.parsing.loaders.pdf.layout import LayoutBox

_PROMPT = (
    "이 이미지를 분석해서 아래 규칙에 맞게 한국어로만 답해줘:\n"
    "- 순서도(flowchart)나 다이어그램이면: 그 구조를 Mermaid 문법으로 표현해서 "
    "```mermaid 로 시작하는 코드 블록 하나만 출력해줘(다른 설명 문장 없이).\n"
    "- 서명이나 손글씨면: 설명하지 말고 실제로 쓰여 있는 글자만 그대로 옮겨 적어줘.\n"
    "- 그 외(사진·차트·일반 그림)면: 내용을 간결하게 설명해줘(차트면 구조와 "
    "핵심 값 포함)."
)

_TABLE_PROMPT = (
    "이 표의 내용과 구조를 한국어로 2~4문장으로 요약해줘. 주요 행/열과 특이한 "
    "값을 언급해줘. HTML이나 마크다운 표 문법은 쓰지 말고 문장으로만 설명해줘."
)

_MERMAID_BLOCK = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)


def _extract_mermaid(text: str) -> tuple[str, str | None]:
    """VLM 응답에 ```mermaid 코드 블록이 있으면 뽑아서 (나머지 텍스트, mermaid
    소스) 튜플로 반환한다. 없으면 (원본 텍스트, None) — 순서도가 아니라
    보통 이미지였다는 뜻."""
    match = _MERMAID_BLOCK.search(text)
    if not match:
        return text, None
    mermaid_src = match.group(1).strip()
    remainder = _MERMAID_BLOCK.sub("", text).strip()
    return (remainder or "[다이어그램을 Mermaid로 추출함]"), mermaid_src


# 200개 문서 배치 실행 중 실제로 발견: openai SDK의 timeout= 인자를 줘도 응답이
# 스트리밍성으로 찔끔찔끔 오면 read timeout이 계속 리셋돼서 30분 넘게 안 끝나는
# 호출이 있었다(httpx 레벨 timeout으로는 못 막음). 그래서 스레드로 감싸서 진짜
# 벽시계 상한을 강제한다 — 스레드가 안 끝나도 그냥 버리고 넘어간다(daemon처럼
# 새어나가지만, 어차피 client 자체에도 timeout=60이 있어 무한히 살아있진 않음).
_HARD_TIMEOUT_SEC = 45


def _caption_with_hard_timeout(
    client: VLMClient, image_bytes: bytes, prompt: str
) -> VLMCaptionResult:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(client.caption_image, image_bytes, prompt)
    try:
        return future.result(timeout=_HARD_TIMEOUT_SEC)
    except concurrent.futures.TimeoutError:
        return VLMCaptionResult(text="[VLM 응답 시간 초과로 캡션 생성 실패]", usage=None)
    finally:
        executor.shutdown(wait=False)


@lru_cache(maxsize=1)
def _get_client() -> VLMClient:
    return VLMClient()


def caption_figures(
    page: pymupdf.Page,
    page_number: int,
    boxes: list[LayoutBox],
) -> list[DocumentElement]:
    if not boxes:
        return []

    client = _get_client()
    elements: list[DocumentElement] = []
    for box in boxes:
        pix = page.get_pixmap(clip=box.bbox, dpi=200)
        image_bytes = pix.tobytes("png")
        x0, y0, x1, y1 = box.bbox
        metadata = {
            "source": "vlm",
            # layout_label: PP-DocLayoutV2의 25개 카테고리 중 실제 감지된 라벨
            # (예: "chart", "image", "table" 등) — 그림/표라는 것만 아는 게
            # 아니라 어떤 종류인지까지 남겨둔다.
            "layout_label": box.label,
            "layout_cls_id": box.cls_id,
            "layout_box_index": box.box_index,
        }

        if box.label == "table":
            element_type = ElementType.TABLE
            result = _caption_with_hard_timeout(client, image_bytes, _TABLE_PROMPT)
            text = result.text
        else:
            element_type = ElementType.IMAGE
            result = _caption_with_hard_timeout(client, image_bytes, _PROMPT)
            text, mermaid = _extract_mermaid(result.text)
            if mermaid is not None:
                metadata["mermaid"] = mermaid

        if result.usage is not None:
            metadata["vlm_usage"] = result.usage

        elements.append(
            DocumentElement(
                type=element_type,
                text=text,
                page=page_number,
                bboxes=[BBox(x0=x0, y0=y0, x1=x1, y1=y1)],
                metadata=metadata,
            )
        )
    return elements
