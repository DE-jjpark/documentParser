"""Loader for PDF documents. Requires the 'pdf' extra (pymupdf)."""

from document_parser.core.exceptions import MissingDependencyError
from document_parser.core.models import DocumentElement, ElementType


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
            page_text = page.get_text().strip()
            if page_text:
                elements.append(
                    DocumentElement(type=ElementType.TEXT, text=page_text, page=page_number)
                )
    return elements
