"""Format-specific loaders.

A loader is a callable ``(data: bytes, source: str) -> list[DocumentElement]``.
Register new formats with ``register()``.
"""

from collections.abc import Callable

from document_parser.core.models import DocumentElement
from document_parser.parsing.loaders import html_loader, json_loader, office, pdf, text

Loader = Callable[[bytes, str], list[DocumentElement]]

_REGISTRY: dict[str, Loader] = {}


def register(fmt: str, loader: Loader) -> None:
    _REGISTRY[fmt.lower()] = loader


def get_loader(fmt: str) -> Loader | None:
    return _REGISTRY.get(fmt.lower())


def supported_formats() -> list[str]:
    return sorted(_REGISTRY)


for _fmt in text.FORMATS:
    register(_fmt, text.load)
register("pdf", pdf.load)
for _fmt in html_loader.FORMATS:
    register(_fmt, html_loader.load)
for _fmt in json_loader.FORMATS:
    register(_fmt, json_loader.load)
for _fmt in office.FORMATS:
    register(_fmt, office.load)
