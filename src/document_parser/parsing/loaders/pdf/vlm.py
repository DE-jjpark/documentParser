"""VLM 그림/표 캡션·구조 추출 경로.

다이어그램 메모: AzureDI와 달리 먼저 크롭한 다음 VLM에 보낸다 — 페이지
전체보다 타이트하게 자른 영역이 정확도가 더 좋다.

실제 in4u Databricks AI Gateway(Claude Sonnet 4.6)로 연동 확인함 —
parsing.clients.vlm.VLMClient 참고. 크롭은 pymupdf의 get_pixmap(clip=...)로
하는데, clip은 포인트 좌표를 받으므로(픽셀 변환 불필요) box.bbox를 그대로
쓴다 — layout.py/azure_di.py에서 모델이 "돌려주는" 좌표를 변환해야 했던 것과는
반대 방향이라 혼동하지 말 것.

content/summary 분리: VLM한테 "[CONTENT]"/"[SUMMARY]" 두 구획으로 나눠서
답하게 하고(_split_content_summary), DocumentElement.text에는 content를,
.summary에는 summary를 담는다 — "내용 그 자체"와 "요약"을 별개로 원한다는
요청에 따른 것.
  - 표(table): content = 마크다운 표(문법 그대로), summary = 2~4문장 요약.
    다만 AzureDI가 그 표를 찾았으면(graph.py의 merge 노드) content는 DI의
    실제 구조에서 뽑은 마크다운으로 덮어써진다 — VLM의 마크다운은 DI가 못
    찾았을 때만 그대로 남는 폴백이다.
  - 이미지(image/chart/순서도/서명 등): content는 종류별로 다르다 —
    순서도·다이어그램이면 Mermaid 코드(그 자체가 text가 됨, metadata["mermaid"]
    에도 같은 걸 남겨 명시적으로 판별 가능하게 함), 서명·손글씨면 실제로
    쓰인 글자 그대로(OCR), 그 외엔 평범한 설명. summary는 그 내용이 뭘
    보여주는지의 요약.
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
    "이 이미지를 분석해서 아래 형식 그대로, 한국어로만 답해줘(다른 설명 문장 "
    "없이 이 형식만 출력):\n"
    "[CONTENT]\n"
    "- 순서도(flowchart)나 다이어그램이면: 그 구조를 Mermaid 문법으로 표현해서 "
    "```mermaid 코드 블록 하나만.\n"
    "- 서명이나 손글씨면: 실제로 쓰여 있는 글자만 그대로.\n"
    "- 그 외(사진·차트·일반 그림)면: 내용을 간결하게 설명(차트면 구조와 핵심 "
    "값 포함).\n"
    "[SUMMARY]\n"
    "위 내용이 무엇을 보여주는지 1~2문장으로 요약."
)

_TABLE_PROMPT = (
    "이 표를 분석해서 아래 형식 그대로, 한국어로만 답해줘(다른 설명 문장 없이 "
    "이 형식만 출력):\n"
    "[CONTENT]\n"
    "표 내용을 마크다운 표 문법으로 그대로 옮겨 적어줘(| 구분자, 헤더 구분줄 "
    "포함).\n"
    "[SUMMARY]\n"
    "표의 내용과 구조를 2~4문장으로 요약(주요 행/열과 특이한 값 언급)."
)

_MERMAID_BLOCK = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)
_CONTENT_SUMMARY = re.compile(r"\[CONTENT\]\s*(.*?)\s*\[SUMMARY\]\s*(.*)", re.DOTALL)


def _split_content_summary(text: str) -> tuple[str, str | None]:
    """VLM 응답에서 [CONTENT]/[SUMMARY] 구획을 나눈다. 형식을 안 지켰으면
    (드물게 있음) 전체를 content로 보고 summary는 None."""
    match = _CONTENT_SUMMARY.search(text)
    if not match:
        return text.strip(), None
    content = match.group(1).strip()
    summary = match.group(2).strip()
    return content, (summary or None)


def _extract_mermaid(content: str) -> tuple[str, str | None]:
    """content에 ```mermaid 코드 블록이 있으면 (mermaid 소스, mermaid 소스)를
    반환해서 text 자체가 mermaid가 되게 한다. 없으면 (원본 content, None)."""
    match = _MERMAID_BLOCK.search(content)
    if not match:
        return content, None
    mermaid_src = match.group(1).strip()
    return mermaid_src, mermaid_src


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
        return VLMCaptionResult(text="[CONTENT]\n[VLM 응답 시간 초과]\n[SUMMARY]\n", usage=None)
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
            text, summary = _split_content_summary(result.text)
        else:
            element_type = ElementType.IMAGE
            result = _caption_with_hard_timeout(client, image_bytes, _PROMPT)
            content, summary = _split_content_summary(result.text)
            text, mermaid = _extract_mermaid(content)
            if mermaid is not None:
                metadata["mermaid"] = mermaid

        if result.usage is not None:
            metadata["vlm_usage"] = result.usage

        elements.append(
            DocumentElement(
                type=element_type,
                text=text,
                summary=summary,
                page=page_number,
                bboxes=[BBox(x0=x0, y0=y0, x1=x1, y1=y1)],
                metadata=metadata,
            )
        )
    return elements
