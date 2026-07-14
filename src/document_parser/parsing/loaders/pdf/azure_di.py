"""Azure Document Intelligence extraction path.

Diagram note: cost is per-page regardless of crop size, so the whole page is
sent in one request (no cropping, unlike the VLM path).

TODO(real implementation): call the azure-ai-documentintelligence SDK's
prebuilt-layout model on the rendered page image and convert its
text/table results into DocumentElements.
"""

import pymupdf

from document_parser.core.models import DocumentElement, ElementType


def extract_with_azure_di(page: pymupdf.Page, page_number: int) -> list[DocumentElement]:
    return [
        DocumentElement(
            type=ElementType.TEXT,
            text="[azure_di stub - not implemented]",
            page=page_number,
            metadata={"source": "azure_di", "stub": True},
        )
    ]
