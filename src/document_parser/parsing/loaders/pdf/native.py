"""Native text extraction for pages that already have a text layer.

Real implementation (not a stub): reads pymupdf's block-level text output,
which gives per-block bounding boxes for free, instead of the whole-page
``page.get_text()`` string the previous loader used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from document_parser.core.models import BBox, DocumentElement, ElementType

if TYPE_CHECKING:
    # type hints only -- see layout.py's comment for why this stays lazy.
    import pymupdf


def extract_native(page: pymupdf.Page, page_number: int) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    for x0, y0, x1, y1, text, _block_no, block_type in page.get_text("blocks"):
        if block_type != 0:  # 1 = image block; handled by the AzureDI+VLM path instead
            continue
        text = text.strip()
        if not text:
            continue
        elements.append(
            DocumentElement(
                type=ElementType.TEXT,
                text=text,
                page=page_number,
                bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
                metadata={"source": "native"},
            )
        )
    return elements
