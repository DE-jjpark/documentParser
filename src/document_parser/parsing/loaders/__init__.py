"""Format-specific loaders.

A loader is a callable ``(data: bytes, source: str, tier: str) -> list[DocumentElement]``.
``tier`` is ``"fast"`` or ``"balanced"`` (``ParsingTier``) -- most loaders
ignore it (there's nothing to skip), only pdf/office/image care since they're
the ones that can call VLM. Register new formats with ``register()``.
"""

from collections.abc import Callable

from document_parser.core.models import DocumentElement
from document_parser.parsing.loaders import html_loader, image, json_loader, office, pdf, text

Loader = Callable[[bytes, str, str], list[DocumentElement]]

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
for _fmt in image.FORMATS:
    register(_fmt, image.load)
