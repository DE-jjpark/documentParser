"""Azure Document Intelligence client (prebuilt-layout model).

Required env vars:
  AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT   e.g. https://<resource>.cognitiveservices.azure.com
  AZURE_DOCUMENT_INTELLIGENCE_KEY

Field names below (``paragraphs``, ``content``, ``bounding_regions``,
``polygon``) were confirmed against azure-ai-documentintelligence==1.0.2's
actual model classes, not guessed from docs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from document_parser.core.exceptions import MissingDependencyError


@dataclass
class LayoutParagraph:
    text: str
    bbox: tuple[float, float, float, float] | None


@dataclass
class AzureLayoutResult:
    paragraphs: list[LayoutParagraph]


class AzureDocumentIntelligenceClient:
    def __init__(self) -> None:
        try:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential
        except ImportError as exc:
            raise MissingDependencyError(
                "Azure Document Intelligence support requires the 'azure' extra: "
                "pip install 'document-parser[azure]'"
            ) from exc

        endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        if not endpoint or not key:
            raise MissingDependencyError(
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT / AZURE_DOCUMENT_INTELLIGENCE_KEY "
                "environment variables are required"
            )

        self._client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )

    def analyze_layout(self, image_bytes: bytes) -> AzureLayoutResult:
        """Send one page image (diagram: whole page, no cropping -- billing
        is per page regardless) to the prebuilt-layout model."""
        poller = self._client.begin_analyze_document("prebuilt-layout", body=image_bytes)
        result = poller.result()
        paragraphs = [
            LayoutParagraph(text=p.content, bbox=_polygon_to_bbox(p.bounding_regions))
            for p in (result.paragraphs or [])
        ]
        return AzureLayoutResult(paragraphs=paragraphs)


def _polygon_to_bbox(bounding_regions) -> tuple[float, float, float, float] | None:
    """First bounding region's polygon (flat [x0,y0,x1,y1,...] clockwise from
    top-left) -> axis-aligned bbox.

    TODO: confirm coordinate units against the real in4u Azure endpoint --
    for image input this should be pixels matching the submitted image, but
    that's not yet verified against a live call.
    """
    if not bounding_regions:
        return None
    polygon = bounding_regions[0].polygon
    xs = polygon[0::2]
    ys = polygon[1::2]
    return (min(xs), min(ys), max(xs), max(ys))
