"""VLM figure-captioning path.

Diagram note: crop first, then send to the VLM (unlike AzureDI) -- accuracy
is better on a tight crop than on the whole page.
"""

from functools import lru_cache

import pymupdf

from document_parser.core.models import BBox, DocumentElement, ElementType
from document_parser.parsing.clients.vlm import VLMClient

_PROMPT = (
    "Describe the content of this figure concisely. If it's a chart, table-like "
    "image, or diagram, describe its structure and key values."
)


@lru_cache(maxsize=1)
def _get_client() -> VLMClient:
    return VLMClient()


def caption_figures(
    page: pymupdf.Page,
    page_number: int,
    crop_boxes: list[tuple[float, float, float, float]],
) -> list[DocumentElement]:
    if not crop_boxes:
        return []

    client = _get_client()
    elements: list[DocumentElement] = []
    for box in crop_boxes:
        pix = page.get_pixmap(clip=pymupdf.Rect(*box), dpi=200)
        caption = client.caption_image(pix.tobytes("png"), _PROMPT)
        elements.append(
            DocumentElement(
                type=ElementType.IMAGE,
                text=caption,
                page=page_number,
                bbox=BBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3]),
                metadata={"source": "vlm"},
            )
        )
    return elements
