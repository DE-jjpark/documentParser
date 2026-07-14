"""Unit tests for the VLM client.

No real credentials/network calls: the google-genai SDK client is mocked,
since we don't have a live in4u Azure-hosted Gemini endpoint to test against
yet. These tests only guard the env-var contract and the request/response
plumbing.
"""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("google.genai", reason="vlm extra not installed")

from document_parser.core.exceptions import MissingDependencyError  # noqa: E402
from document_parser.parsing.clients.vlm import VLMClient  # noqa: E402


def test_missing_env_vars_raises(monkeypatch):
    monkeypatch.delenv("AZURE_VLM_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_VLM_API_KEY", raising=False)

    with pytest.raises(MissingDependencyError, match="AZURE_VLM"):
        VLMClient()


def test_caption_image_returns_response_text(monkeypatch):
    monkeypatch.setenv("AZURE_VLM_ENDPOINT", "https://example.services.ai.azure.com/models")
    monkeypatch.setenv("AZURE_VLM_API_KEY", "fake-key")

    mock_response = MagicMock()
    mock_response.text = "a red square"

    with patch("google.genai.Client") as mock_genai_client:
        mock_genai_client.return_value.models.generate_content.return_value = mock_response
        client = VLMClient()
        caption = client.caption_image(b"fake-png-bytes", "describe this")

    assert caption == "a red square"
    mock_genai_client.return_value.models.generate_content.assert_called_once()
