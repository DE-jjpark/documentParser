"""Azure Document Intelligence 클라이언트 단위 테스트.

실제 자격증명/네트워크 호출 없음: SDK 클라이언트 자체를 mock — 아직 라이브
in4u Azure 엔드포인트가 없어서다. 환경변수 계약과 결과 -> LayoutParagraph
매핑만 검증한다.
"""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("azure.ai.documentintelligence", reason="azure extra not installed")

from document_parser.core.exceptions import MissingDependencyError  # noqa: E402
from document_parser.parsing.clients.azure_document_intelligence import (  # noqa: E402
    AzureDocumentIntelligenceClient,
    _polygon_to_bboxes,
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

    # 컬럼을 넘나드는 단락처럼 bounding_regions이 2개인 경우를 검증(issue:
    # 단락 하나가 영역 여러 개를 가질 수 있어 bbox 단수 필드로는 부족했음).
    mock_paragraph = MagicMock()
    mock_paragraph.content = "hello world"
    mock_paragraph.bounding_regions = [
        MagicMock(polygon=[10, 20, 110, 20, 110, 40, 10, 40]),
        MagicMock(polygon=[0, 100, 50, 100, 50, 120, 0, 120]),
    ]

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
    assert result.paragraphs[0].bboxes == [(10, 20, 110, 40), (0, 100, 50, 120)]


def test_polygon_to_bboxes_empty_when_no_regions():
    assert _polygon_to_bboxes([]) == []
    assert _polygon_to_bboxes(None) == []
