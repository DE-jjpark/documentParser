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

    with pytest.raises(MissingDependencyError, match="DATABRICKS"):
        VLMClient()


def test_caption_image_returns_response_text(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "adb-1017423463570685.5.azuredatabricks.net")
    monkeypatch.setenv("DATABRICKS_TOKEN", "fake-token")

    mock_message = MagicMock()
    mock_message.content = "a red square"
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_message)]

    with patch("openai.OpenAI") as mock_openai_client:
        mock_openai_client.return_value.chat.completions.create.return_value = mock_response
        client = VLMClient()
        caption = client.caption_image(b"fake-png-bytes", "describe this")

    assert caption == "a red square"
    mock_openai_client.return_value.chat.completions.create.assert_called_once()


def test_langsmith_tracing_disabled_by_default(monkeypatch):
    """LANGCHAIN_TRACING_V2이 안 켜져 있으면 wrap_openai를 안 거치고 원본
    클라이언트를 그대로 쓴다 — 평소엔 langsmith 관련 오버헤드가 전혀 없다."""
    monkeypatch.setenv("DATABRICKS_HOST", "adb-1017423463570685.5.azuredatabricks.net")
    monkeypatch.setenv("DATABRICKS_TOKEN", "fake-token")
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

    with (
        patch("openai.OpenAI") as mock_openai_client,
        patch("langsmith.wrappers.wrap_openai") as mock_wrap,
    ):
        client = VLMClient()

    mock_wrap.assert_not_called()
    assert client._client is mock_openai_client.return_value


def test_langsmith_tracing_wraps_client_when_enabled(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "adb-1017423463570685.5.azuredatabricks.net")
    monkeypatch.setenv("DATABRICKS_TOKEN", "fake-token")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    with (
        patch("openai.OpenAI") as mock_openai_client,
        patch("langsmith.wrappers.wrap_openai") as mock_wrap,
    ):
        mock_wrap.return_value = "wrapped-client"
        client = VLMClient()

    mock_wrap.assert_called_once_with(mock_openai_client.return_value)
    assert client._client == "wrapped-client"
