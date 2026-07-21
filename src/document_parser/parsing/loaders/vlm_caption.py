"""이미지 한 장을 VLM에 보내 캡션(content/summary)을 받는 공용 로직.

pdf/vlm.py(페이지에서 크롭한 그림)와 image.py(파일 자체가 이미지인 경우) 둘 다
이 모듈을 쓴다 — 크롭 여부만 다르고 프롬프트·응답 파싱은 동일해서 여기로 뺐다.
표 전용 프롬프트(``_TABLE_PROMPT``)와 페이지 크롭 로직(``caption_figures``)은
PDF에서만 쓰이므로 pdf/vlm.py에 그대로 남겨둔다.
"""

from __future__ import annotations

import concurrent.futures
import re
from functools import lru_cache

from document_parser.parsing.clients.vlm import VLMCaptionResult, VLMClient

PROMPT = (
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

_MERMAID_BLOCK = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)
_CONTENT_SUMMARY = re.compile(r"\[CONTENT\]\s*(.*?)\s*\[SUMMARY\]\s*(.*)", re.DOTALL)


def split_content_summary(text: str) -> tuple[str, str | None]:
    """VLM 응답에서 [CONTENT]/[SUMMARY] 구획을 나눈다. 형식을 안 지켰으면
    (드물게 있음) 전체를 content로 보고 summary는 None."""
    match = _CONTENT_SUMMARY.search(text)
    if not match:
        return text.strip(), None
    content = match.group(1).strip()
    summary = match.group(2).strip()
    return content, (summary or None)


def extract_mermaid(content: str) -> tuple[str, str | None]:
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
HARD_TIMEOUT_SEC = 45


def caption_with_hard_timeout(
    client: VLMClient, image_bytes: bytes, prompt: str, mime_type: str = "image/png"
) -> VLMCaptionResult:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(client.caption_image, image_bytes, prompt, mime_type)
    try:
        return future.result(timeout=HARD_TIMEOUT_SEC)
    except concurrent.futures.TimeoutError:
        return VLMCaptionResult(text="[CONTENT]\n[VLM 응답 시간 초과]\n[SUMMARY]\n", usage=None)
    finally:
        executor.shutdown(wait=False)


@lru_cache(maxsize=1)
def get_client() -> VLMClient:
    return VLMClient()
