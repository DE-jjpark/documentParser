"""Loader for PDF documents. Requires the 'pdf' extra (pymupdf).

Per page: analyze_page() decides the route (diagram: layout analysis ->
Only Text + text layer present -> native extraction, otherwise (figures
present, or no text layer / scanned page) -> AzureDI + VLM). See
``layout.py`` for the routing rule and ``native.py``/``azure_di.py``/
``vlm.py`` for the three extraction paths themselves.
"""

from document_parser.core.exceptions import MissingDependencyError
from document_parser.core.models import DocumentElement
from document_parser.parsing.loaders.pdf.azure_di import extract_with_azure_di
from document_parser.parsing.loaders.pdf.layout import analyze_page, needs_heavy_path
from document_parser.parsing.loaders.pdf.native import extract_native
from document_parser.parsing.loaders.pdf.vlm import caption_figures


def load(data: bytes, source: str) -> list[DocumentElement]:
    try:
        import pymupdf
    except ImportError as exc:
        raise MissingDependencyError(
            "PDF support requires the 'pdf' extra: pip install 'document-parser[pdf]'"
        ) from exc

    elements: list[DocumentElement] = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for page_number, page in enumerate(doc, start=1):
            layout = analyze_page(page)
            if needs_heavy_path(layout):
                # independent I/O calls once real SDKs are wired in -- TODO:
                # parallelize (e.g. asyncio.gather / a thread pool) instead
                # of the sequential calls below.
                elements.extend(extract_with_azure_di(page, page_number))
                elements.extend(caption_figures(page, page_number, layout.crop_boxes))
            else:
                elements.extend(extract_native(page, page_number))
    return elements
