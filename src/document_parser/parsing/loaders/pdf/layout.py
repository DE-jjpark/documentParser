"""Per-page layout analysis: decides which extraction path a page should take.

Uses PP-DocLayoutV2 (via the 'layout' extra: paddleocr) when it's installed
and its weights have been downloaded (``document-parser download-models``,
see ``parsing/weights.py``). Falls back to a pymupdf-only heuristic when the
extra just isn't installed at all, so plain-text PDF parsing keeps working
with only the 'pdf' extra -- but raises if the extra *is* installed and the
weights are simply missing, since that's a fixable misconfiguration rather
than an intentional lighter install.

Running PP-DocLayoutV2 itself additionally needs paddlepaddle, which is not
on a normal PyPI index (see the comment on the 'layout' extra in
pyproject.toml) -- deferred supply-chain review, not blocking here per team
decision: pre-download the weights and bring them along.
"""

import tempfile
from dataclasses import dataclass, field
from functools import lru_cache

import pymupdf

from document_parser.core.exceptions import MissingDependencyError
from document_parser.parsing.weights import layout_model_dir

# PP-DocLayoutV2's 25 labels, bucketed the same way the sibling skep_parser
# project's DP-Bench comparison did: categories that are "a picture, not
# extractable text" count as a figure and get a VLM crop box.
_FIGURE_LABELS = {
    "chart",
    "image",
    "header_image",
    "footer_image",
    "seal",
    "display_formula",
    "inline_formula",
    "formula_number",
}


@dataclass
class PageLayout:
    has_figures: bool
    has_text_layer: bool
    crop_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)


@lru_cache(maxsize=1)
def _get_model():
    from paddleocr import LayoutDetection

    model_dir = layout_model_dir()
    if not any(model_dir.glob("*")):
        raise MissingDependencyError(
            f"PP-DocLayoutV2 weights not found at {model_dir} -- run "
            "'document-parser download-models' first"
        )
    return LayoutDetection(model_name="PP-DocLayoutV2", model_dir=str(model_dir))


def analyze_page(page: pymupdf.Page) -> PageLayout:
    has_text_layer = bool(page.get_text().strip())

    try:
        model = _get_model()
    except ImportError:
        return _analyze_page_heuristic(page, has_text_layer)

    pix = page.get_pixmap(dpi=200)
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        pix.save(f.name)
        (result,) = model.predict(f.name, batch_size=1, layout_nms=True)

    crop_boxes = [
        tuple(box["coordinate"]) for box in result["boxes"] if box["label"] in _FIGURE_LABELS
    ]
    return PageLayout(
        has_figures=bool(crop_boxes),
        has_text_layer=has_text_layer,
        crop_boxes=crop_boxes,
    )


def _analyze_page_heuristic(page: pymupdf.Page, has_text_layer: bool) -> PageLayout:
    """Fallback when the 'layout' extra isn't installed: pymupdf-only heuristic."""
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
