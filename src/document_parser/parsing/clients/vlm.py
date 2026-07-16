"""VLM client for figure captioning.

Databricks AI Gateway(OpenAI 호환 chat completions API)를 통해 Claude Sonnet
4.6(스코프: model-serving + ai-gateway 둘 다 필요한 토큰)를 호출한다. 실제
in4u 워크스페이스(adb-1017423463570685)에서 텍스트/비전 둘 다 라이브로
확인함.

애초 계획은 Gemini였지만(Databricks Foundation Model API pay-per-token으로
Gemini 3.5/3.1, GPT-5 mini까지 순서대로 시도) 이 워크스페이스(koreacentral)
에서 전부 "model not enabled/available" 에러가 나서, 팀이 이미 등록해둔
AI Gateway 경로(Unity Catalog 모델 `skep_parser.skep-parser-test.parser-
test-sn-4-6` → 실제로는 `global.anthropic.claude-sonnet-4-6`)로 전환했다.
OpenAI 호환 형식이라 openai SDK를 base_url만 바꿔서 그대로 쓴다 — 이 SDK는
api_key를 그대로 Authorization: Bearer 헤더로 써주므로 google-genai 때처럼
헤더를 직접 손볼 필요가 없다.

필요 환경변수:
  DATABRICKS_HOST   예) adb-1017423463570685.5.azuredatabricks.net
  DATABRICKS_TOKEN  model-serving + ai-gateway 스코프가 있는 토큰
  DATABRICKS_VLM_MODEL  Unity Catalog 모델 경로 (기본값: 지금 확인된
                        skep_parser.skep-parser-test.parser-test-sn-4-6 —
                        테스트용 이름이라 나중에 프로덕션 배포로 바뀔 수 있음)

LangSmith 트레이싱(선택): 200개 문서 배치를 돌릴 때 어느 VLM 호출이 실제로
느린지/실패하는지 직접 확인할 수 있게, LANGCHAIN_TRACING_V2=true일 때만
openai 클라이언트를 langsmith.wrappers.wrap_openai()로 감싼다 — 매 호출의
실제 소요시간·입력·출력·에러가 LangSmith 프로젝트에 남는다. 다음 환경변수가
필요하다:
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY=<LangSmith API 키>
  LANGCHAIN_PROJECT=<프로젝트 이름, 선택. 기본값 "default">
설정 안 하면(LANGCHAIN_TRACING_V2 미설정/false) 기존과 동일하게 동작한다 —
LangGraph 노드 자체(analyze/native/azure_di/vlm/merge)의 트레이싱도 같은
환경변수만으로 LangGraph가 자동으로 처리하므로 이쪽엔 별도 코드가 없다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from document_parser.core.exceptions import MissingDependencyError

DEFAULT_MODEL = "skep_parser.skep-parser-test.parser-test-sn-4-6"
_AI_GATEWAY_PATH = "ai-gateway/mlflow/v1"


@dataclass
class VLMCaptionResult:
    """caption_image()의 반환값 — 200개 문서 배치 실행 후 "토큰 얼마나 썼어?"
    질문에 답을 못 해서(usage를 그냥 버리고 있었음) 추가했다. usage는 응답에
    없거나(타임아웃 등) 게이트웨이가 안 준 모델이면 None."""

    text: str
    usage: dict[str, Any] | None = None


# 참고: koreacentral 워크스페이스에서 Foundation Model API(pay-per-token)로
# 네이티브 서빙 가능하다고 확인된 모델 후보군(2026-07-15 기준, 팀 확인).
# Gemini는 이 목록에 없어서(리전 미지원) Claude Sonnet 4.6으로 갔다 — 나중에
# 모델을 바꿀 일이 있으면 이 중에서 고르면 리전 문제로 또 막힐 가능성이
# 낮다. 단, 비전(이미지 입력) 지원 여부는 모델마다 별도 확인 필요 — 아래
# 목록은 "이 워크스페이스에서 서빙 가능하다"까지만 확인된 것이지 VLM 용도로
# 다 검증된 건 아니다.
#   Anthropic : Claude Sonnet 4.6/4.5/4, Claude Haiku 4.5,
#               Claude Opus 4.8/4.7/4.6/4.5/4.1, Claude Fable 5
#   Meta      : Llama 4 Maverick, Llama 3.3 70B Instruct,
#               Llama 3.1 405B Instruct, Llama 3.1 8B Instruct
#   Alibaba   : Qwen3.5 122B A10B, Qwen3-Next 80B A3B Instruct,
#               Qwen3-Embedding-0.6B(임베딩)
#   OpenAI    : GPT-OSS-120B, GPT-OSS-20B (오픈웨이트만)
#   Google    : Gemma 3 12B (오픈웨이트)
#   임베딩     : GTE Large (En)


class VLMClient:
    def __init__(self, model: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise MissingDependencyError(
                "VLM support requires the 'vlm' extra: pip install 'document-parser[vlm]'"
            ) from exc

        host = os.environ.get("DATABRICKS_HOST")
        token = os.environ.get("DATABRICKS_TOKEN")
        if not host or not token:
            raise MissingDependencyError(
                "DATABRICKS_HOST / DATABRICKS_TOKEN environment variables are required"
            )

        self._model = model or os.environ.get("DATABRICKS_VLM_MODEL", DEFAULT_MODEL)
        # openai SDK 기본값(10분 타임아웃 + 재시도 2회)은 게이트웨이가 느려질 때
        # 호출 하나가 30분 가까이 걸릴 수 있다는 걸 200개 문서 배치 실행 중
        # 실제로 발견해서(요청 하나가 39분 걸림), 훨씬 짧은 값으로 낮췄다 —
        # 응답이 느린 호출은 이 실패를 감수하고 넘어가는 게, 배치 전체가
        # 그 한 호출 때문에 몇십 분씩 막히는 것보다 낫다.
        client = OpenAI(
            api_key=token,
            base_url=f"https://{host}/{_AI_GATEWAY_PATH}",
            timeout=60.0,
            max_retries=1,
        )
        from langsmith.utils import tracing_is_enabled

        # LANGSMITH_TRACING(신규 명칭)과 LANGCHAIN_TRACING_V2(구 명칭) 둘 다
        # 알아서 확인해주는 langsmith 자체 판별 함수를 쓴다 — 직접 환경변수
        # 이름 하나만 확인하면 어느 한쪽 명칭을 놓칠 수 있다.
        if tracing_is_enabled():
            from langsmith.wrappers import wrap_openai

            client = wrap_openai(client)
        self._client = client

    def caption_image(
        self, image_bytes: bytes, prompt: str, mime_type: str = "image/png"
    ) -> VLMCaptionResult:
        import base64

        b64_image = base64.b64encode(image_bytes).decode()
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
                        },
                    ],
                }
            ],
        )
        usage = response.usage.model_dump() if response.usage is not None else None
        return VLMCaptionResult(text=response.choices[0].message.content, usage=usage)
