"""Azure Document Intelligence extraction path.

Diagram note: cost is per-page regardless of crop size, so the whole page is
sent in one request (no cropping, unlike the VLM path).
"""

from functools import lru_cache

import pymupdf

from document_parser.core.models import BBox, DocumentElement, ElementType
from document_parser.parsing.clients.azure_document_intelligence import (
    AzureDocumentIntelligenceClient,
)


@lru_cache(maxsize=1)
def _get_client() -> AzureDocumentIntelligenceClient:
    return AzureDocumentIntelligenceClient()


def extract_with_azure_di(page: pymupdf.Page, page_number: int) -> list[DocumentElement]:
    client = _get_client()
    pix = page.get_pixmap(dpi=200)
    result = client.analyze_layout(pix.tobytes("png"))

    elements: list[DocumentElement] = []
    for paragraph in result.paragraphs:
        bbox = None
        if paragraph.bbox:
            x0, y0, x1, y1 = paragraph.bbox
            bbox = BBox(x0=x0, y0=y0, x1=x1, y1=y1)
        elements.append(
            DocumentElement(
                type=ElementType.TEXT,
                text=paragraph.text,
                page=page_number,
                bbox=bbox,
                metadata={"source": "azure_di"},
            )
        )
    return elements
