"""VLM 클라이언트 단위 테스트.

실제 자격증명/네트워크 호출 없음: openai SDK 클라이언트 자체를 mock한다 —
env var 계약 + 요청/응답 배선만 확인. 실제 라이브 검증은
tests/test_pdf_loader.py의 requires_real_vlm 계열 테스트에서 한다.
"""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("openai", reason="vlm extra not installed")

from document_parser.core.exceptions import MissingDependencyError  # noqa: E402
from document_parser.parsing.clients.vlm import VLMClient  # noqa: E402


def test_missing_env_vars_raises(monkeypatch):
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.delenv("VLM_PROVIDER", raising=False)

    with pytest.raises(MissingDependencyError, match="DATABRICKS"):
        VLMClient()


def test_azure_provider_missing_env_vars_raises(monkeypatch):
    monkeypatch.setenv("VLM_PROVIDER", "azure")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    with pytest.raises(MissingDependencyError, match="AZURE_OPENAI"):
        VLMClient()


def test_azure_provider_missing_deployment_raises(monkeypatch):
    """Azure OpenAI는 모델명이 아니라 배포(deployment) 이름을 받으므로
    DATABRICKS_VLM_MODEL과 별도로 이게 없으면 못 넘어간다."""
    monkeypatch.setenv("VLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://luna.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-key")
    monkeypatch.delenv("AZURE_OPENAI_VLM_DEPLOYMENT", raising=False)

    with pytest.raises(MissingDependencyError, match="AZURE_OPENAI_VLM_DEPLOYMENT"):
        VLMClient()


def test_azure_provider_builds_azure_openai_client(monkeypatch):
    monkeypatch.setenv("VLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://luna.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("AZURE_OPENAI_VLM_DEPLOYMENT", "luna-vlm-deployment")

    mock_message = MagicMock()
    mock_message.content = "described via azure"
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_response.usage = None

    with (
        patch("openai.AzureOpenAI") as mock_azure_client,
        patch("langsmith.utils.tracing_is_enabled", return_value=False),
    ):
        mock_azure_client.return_value.chat.completions.create.return_value = mock_response
        client = VLMClient()
        result = client.caption_image(b"fake-png-bytes", "describe this")

    mock_azure_client.assert_called_once_with(
        azure_endpoint="https://luna.openai.azure.com",
        api_key="fake-key",
        timeout=60.0,
        max_retries=1,
    )
    assert result.text == "described via azure"
    call_kwargs = mock_azure_client.return_value.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "luna-vlm-deployment"


def test_caption_image_returns_response_text(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "adb-1017423463570685.5.azuredatabricks.net")
    monkeypatch.setenv("DATABRICKS_TOKEN", "fake-token")

    mock_message = MagicMock()
    mock_message.content = "a red square"
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_response.usage = None

    # tracing_is_enabled()는 실제 개발 환경의 LANGSMITH_TRACING 값에 영향받으면
    # 안 되므로(이 테스트는 캡션 배선 자체만 보는 테스트) 명시적으로 꺼둔다 —
    # 켜져 있으면 wrap_openai()가 이 MagicMock을 감싸면서 mock 동일성이 깨진다.
    with (
        patch("openai.OpenAI") as mock_openai_client,
        patch("langsmith.utils.tracing_is_enabled", return_value=False),
    ):
        mock_openai_client.return_value.chat.completions.create.return_value = mock_response
        client = VLMClient()
        result = client.caption_image(b"fake-png-bytes", "describe this")

    assert result.text == "a red square"
    assert result.usage is None
    mock_openai_client.return_value.chat.completions.create.assert_called_once()


def test_caption_image_captures_token_usage(monkeypatch):
    """200개 문서 배치 후 "토큰 얼마나 썼어?"에 답을 못 했던 문제 — 응답의
    usage를 이제 VLMCaptionResult.usage로 그대로 넘겨준다."""
    monkeypatch.setenv("DATABRICKS_HOST", "adb-1017423463570685.5.azuredatabricks.net")
    monkeypatch.setenv("DATABRICKS_TOKEN", "fake-token")

    mock_message = MagicMock()
    mock_message.content = "a red square"
    mock_usage = MagicMock()
    mock_usage.model_dump.return_value = {
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
    }
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_response.usage = mock_usage

    with (
        patch("openai.OpenAI") as mock_openai_client,
        patch("langsmith.utils.tracing_is_enabled", return_value=False),
    ):
        mock_openai_client.return_value.chat.completions.create.return_value = mock_response
        client = VLMClient()
        result = client.caption_image(b"fake-png-bytes", "describe this")

    assert result.usage == {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}


def test_complete_text_sends_plain_text_message_without_image(monkeypatch):
    """complete_text()는 caption_image()와 같은 클라이언트/모델을 쓰되 이미지
    블록 없이 순수 텍스트 content만 보내야 한다(heading_llm.py처럼 비전이
    필요 없는 호출용)."""
    monkeypatch.setenv("DATABRICKS_HOST", "adb-1017423463570685.5.azuredatabricks.net")
    monkeypatch.setenv("DATABRICKS_TOKEN", "fake-token")

    mock_message = MagicMock()
    mock_message.content = "[1, 2, 1]"
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_response.usage = None

    with (
        patch("openai.OpenAI") as mock_openai_client,
        patch("langsmith.utils.tracing_is_enabled", return_value=False),
    ):
        mock_openai_client.return_value.chat.completions.create.return_value = mock_response
        client = VLMClient()
        result = client.complete_text("classify these headings")

    assert result.text == "[1, 2, 1]"
    call_kwargs = mock_openai_client.return_value.chat.completions.create.call_args.kwargs
    assert call_kwargs["messages"] == [{"role": "user", "content": "classify these headings"}]


def test_langsmith_tracing_disabled_by_default(monkeypatch):
    """tracing_is_enabled()가 False면 wrap_openai를 안 거치고 원본 클라이언트를
    그대로 쓴다 — 평소엔 langsmith 관련 오버헤드가 전혀 없다. langsmith 자체
    판별 함수를 쓰므로(LANGSMITH_TRACING/LANGCHAIN_TRACING_V2 둘 다 인식) 여기선
    그 함수 자체를 mock해서 우리 쪽 분기 로직만 확인한다."""
    monkeypatch.setenv("DATABRICKS_HOST", "adb-1017423463570685.5.azuredatabricks.net")
    monkeypatch.setenv("DATABRICKS_TOKEN", "fake-token")

    with (
        patch("openai.OpenAI") as mock_openai_client,
        patch("langsmith.utils.tracing_is_enabled", return_value=False),
        patch("langsmith.wrappers.wrap_openai") as mock_wrap,
    ):
        client = VLMClient()

    mock_wrap.assert_not_called()
    assert client._client is mock_openai_client.return_value


def test_langsmith_tracing_wraps_client_when_enabled(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "adb-1017423463570685.5.azuredatabricks.net")
    monkeypatch.setenv("DATABRICKS_TOKEN", "fake-token")

    with (
        patch("openai.OpenAI") as mock_openai_client,
        patch("langsmith.utils.tracing_is_enabled", return_value=True),
        patch("langsmith.wrappers.wrap_openai") as mock_wrap,
    ):
        mock_wrap.return_value = "wrapped-client"
        client = VLMClient()

    mock_wrap.assert_called_once_with(mock_openai_client.return_value)
    assert client._client == "wrapped-client"
