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
"""

from __future__ import annotations

import os

from document_parser.core.exceptions import MissingDependencyError

DEFAULT_MODEL = "skep_parser.skep-parser-test.parser-test-sn-4-6"
_AI_GATEWAY_PATH = "ai-gateway/mlflow/v1"

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
        self._client = OpenAI(api_key=token, base_url=f"https://{host}/{_AI_GATEWAY_PATH}")

    def caption_image(self, image_bytes: bytes, prompt: str, mime_type: str = "image/png") -> str:
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
        return response.choices[0].message.content
