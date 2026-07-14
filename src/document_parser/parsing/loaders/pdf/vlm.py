"""VLM figure-captioning path.

Diagram note: crop first, then send to the VLM (unlike AzureDI) -- accuracy
is better on a tight crop than on the whole page.

TODO(real implementation): crop each of ``layout.crop_boxes`` out of the
rendered page image and send it to a VLM (e.g. Gemini, matching the
skep_parser.enrichers.vlm.VLMEnricher pattern from the sibling project) for a
caption.
"""

import pymupdf

from document_parser.core.models import BBox, DocumentElement, ElementType


def caption_figures(
    page: pymupdf.Page,
    page_number: int,
    crop_boxes: list[tuple[float, float, float, float]],
) -> list[DocumentElement]:
    return [
        DocumentElement(
            type=ElementType.IMAGE,
            text="[vlm caption stub - not implemented]",
            page=page_number,
            bbox=BBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3]),
            metadata={"source": "vlm", "stub": True},
        )
        for box in crop_boxes
    ]
