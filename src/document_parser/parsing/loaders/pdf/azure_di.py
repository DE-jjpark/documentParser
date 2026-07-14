"""Azure Document Intelligence extraction path.

Diagram note: cost is per-page regardless of crop size, so the whole page is
sent in one request (no cropping, unlike the VLM path).

Temporarily short-circuited to a fixed placeholder: in4u Azure DI
credentials aren't usable yet, so the real call is disabled for now (not
just untested) so parsing still produces chunkable output instead of
failing. The real client (parsing.clients.azure_document_intelligence,
commit ec8c30d) is untouched -- swap the placeholder below for a call to
`AzureDocumentIntelligenceClient` once credentials are ready.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from document_parser.core.models import DocumentElement, ElementType

if TYPE_CHECKING:
    # type hints only -- see layout.py's comment for why this stays lazy.
    import pymupdf

_PLACEHOLDER_TEXT = "[Azure Document Intelligence not connected - placeholder]"


def extract_with_azure_di(page: pymupdf.Page, page_number: int) -> list[DocumentElement]:
    return [
        DocumentElement(
            type=ElementType.TEXT,
            text=_PLACEHOLDER_TEXT,
            page=page_number,
            metadata={"source": "azure_di", "stub": True},
        )
    ]
