"""Unit tests for the Azure Document Intelligence client.

No real credentials/network calls: the SDK client itself is mocked, since we
don't have a live in4u Azure endpoint to test against yet. These tests only
guard the env-var contract and the result -> LayoutParagraph mapping.
"""

from unittest.mock import MagicMock, patch

import pytest

from document_parser.core.exceptions import MissingDependencyError
from document_parser.parsing.clients.azure_document_intelligence import (
    AzureDocumentIntelligenceClient,
    _polygon_to_bbox,
)


def test_missing_env_vars_raises(monkeypatch):
    monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", raising=False)

    with pytest.raises(MissingDependencyError, match="AZURE_DOCUMENT_INTELLIGENCE"):
        AzureDocumentIntelligenceClient()


def test_analyze_layout_maps_paragraphs(monkeypatch):
    monkeypatch.setenv(
        "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "https://example.cognitiveservices.azure.com"
    )
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "fake-key")

    mock_paragraph = MagicMock()
    mock_paragraph.content = "hello world"
    mock_paragraph.bounding_regions = [MagicMock(polygon=[10, 20, 110, 20, 110, 40, 10, 40])]

    mock_result = MagicMock()
    mock_result.paragraphs = [mock_paragraph]

    mock_poller = MagicMock()
    mock_poller.result.return_value = mock_result

    with patch("azure.ai.documentintelligence.DocumentIntelligenceClient") as mock_sdk_client:
        mock_sdk_client.return_value.begin_analyze_document.return_value = mock_poller
        client = AzureDocumentIntelligenceClient()
        result = client.analyze_layout(b"fake-png-bytes")

    assert len(result.paragraphs) == 1
    assert result.paragraphs[0].text == "hello world"
    assert result.paragraphs[0].bbox == (10, 20, 110, 40)


def test_polygon_to_bbox_none_when_no_regions():
    assert _polygon_to_bbox([]) is None
    assert _polygon_to_bbox(None) is None
