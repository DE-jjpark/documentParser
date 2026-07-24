"""Loader for plain-text and markdown documents."""

import re

from document_parser.core.models import DocumentElement, ElementType

FORMATS = ("txt", "text", "md", "markdown")

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def load(data: bytes, source: str, tier: str = "balanced") -> list[DocumentElement]:
    # tier 무시 -- txt/md는 애초에 VLM을 안 쓴다.
    text = data.decode("utf-8", errors="replace")
    elements: list[DocumentElement] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        heading = _HEADING.match(block)
        if heading and "\n" not in block:
            elements.append(
                DocumentElement(
                    type=ElementType.HEADING,
                    text=heading.group(2).strip(),
                    metadata={"level": len(heading.group(1))},
                )
            )
        else:
            elements.append(DocumentElement(type=ElementType.TEXT, text=block))
    return elements
