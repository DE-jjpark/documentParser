"""Per-page layout analysis: decides which extraction path a page should take.

TODO(real implementation): run PP-DocLayoutV2 (see ``parsing/weights.py`` for
the pinned model weights, downloadable via the 'layout' extra +
``document-parser download-models``) over a rendered page image to get real
element boxes/labels, then set ``has_figures``/``has_text_layer`` from that
instead of the crude pymupdf-only heuristics below. Note: actually *running*
PP-DocLayoutV2 needs paddleocr+paddlepaddle, and paddlepaddle is not on a
normal PyPI index (custom index required) -- flagged in the supply-chain
review, not wired here yet.
"""

from dataclasses import dataclass, field

import pymupdf


@dataclass
class PageLayout:
    has_figures: bool
    has_text_layer: bool
    crop_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)


def analyze_page(page: pymupdf.Page) -> PageLayout:
    """stub: pymupdf-only heuristic in place of a real layout model.

    ``has_text_layer`` is genuinely determined (pymupdf can tell us this for
    free); ``has_figures``/``crop_boxes`` are stand-ins for what PP-DocLayoutV2
    would give us (image bboxes on the page, used as VLM crop regions).
    """
    has_text_layer = bool(page.get_text().strip())
    images = page.get_images(full=True)
    crop_boxes = [tuple(page.get_image_bbox(img)) for img in images]
    return PageLayout(
        has_figures=bool(crop_boxes),
        has_text_layer=has_text_layer,
        crop_boxes=crop_boxes,
    )


def needs_heavy_path(layout: PageLayout) -> bool:
    """Diagram routing rule: figures present, or no text layer (scanned page)
    -> AzureDI+VLM; plain text with a text layer -> native extraction."""
    return layout.has_figures or not layout.has_text_layer
