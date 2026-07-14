"""VLM figure-captioning path.

Diagram note: crop first, then send to the VLM (unlike AzureDI) -- accuracy
is better on a tight crop than on the whole page.

Temporarily short-circuited to a fixed placeholder: VLM calls cost money per
image, so real calls are disabled for now regardless of whether credentials
exist, to avoid burning budget before this path is actually needed. The
real client (parsing.clients.vlm, commit ec8c30d) is untouched -- swap the
placeholder below for a call to `VLMClient` when ready to spend on it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from document_parser.core.models import BBox, DocumentElement, ElementType

if TYPE_CHECKING:
    # type hints only -- see layout.py's comment for why this stays lazy.
    import pymupdf

_PLACEHOLDER_CAPTION = "[image - VLM not connected, placeholder]"


def caption_figures(
    page: pymupdf.Page,
    page_number: int,
    crop_boxes: list[tuple[float, float, float, float]],
) -> list[DocumentElement]:
    return [
        DocumentElement(
            type=ElementType.IMAGE,
            text=_PLACEHOLDER_CAPTION,
            page=page_number,
            bbox=BBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3]),
            metadata={"source": "vlm", "stub": True},
        )
        for box in crop_boxes
    ]
