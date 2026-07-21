"""HTML 로더 — 외부 의존성(BeautifulSoup 등) 없이 표준 라이브러리
html.parser.HTMLParser로 DOM을 직접 순회한다.

HTMLParser는 트리가 아니라 이벤트(시작태그/데이터/종료태그) 기반이라, 지금
보고 있는 블록 태그가 뭔지(_current_block)를 직접 추적해야 한다 — 안 그러면
인라인 태그(<b>, <span> 등)에 감싸인 텍스트를 놓친다(블록 태그 스택 top만
확인하면 인라인 태그가 top이 돼서 데이터를 못 잡음).
"""

from __future__ import annotations

from html.parser import HTMLParser

from document_parser.core.models import DocumentElement, ElementType

FORMATS = ("html", "htm")

_HEADING_LEVELS = {f"h{i}": i for i in range(1, 7)}
_BLOCK_TAGS = frozenset(_HEADING_LEVELS) | {"p", "li", "td", "th"}


class _DOMWalker(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: list[DocumentElement] = []
        self._current_block: str | None = None
        self._buffer: list[str] = []
        self._table_rows: list[list[str]] = []
        self._current_row: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._table_rows = []
        elif tag == "tr":
            self._current_row = []
        elif tag in _BLOCK_TAGS and self._current_block is None:
            self._current_block = tag
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._current_block is not None:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == self._current_block:
            text = "".join(self._buffer).strip()
            if tag in _HEADING_LEVELS:
                if text:
                    self.elements.append(
                        DocumentElement(
                            type=ElementType.HEADING,
                            text=text,
                            metadata={"level": _HEADING_LEVELS[tag]},
                        )
                    )
            elif tag == "li":
                if text:
                    self.elements.append(DocumentElement(type=ElementType.LIST, text=text))
            elif tag in ("td", "th"):
                self._current_row.append(text)
            elif tag == "p" and text:
                self.elements.append(DocumentElement(type=ElementType.TEXT, text=text))
            self._current_block = None
            self._buffer = []
        elif tag == "tr":
            if self._current_row:
                self._table_rows.append(self._current_row)
                self._current_row = []
        elif tag == "table":
            if self._table_rows:
                text = "\n".join("\t".join(row) for row in self._table_rows)
                self.elements.append(DocumentElement(type=ElementType.TABLE, text=text))
            self._table_rows = []


def load(data: bytes, source: str, tier: str = "balanced") -> list[DocumentElement]:
    # tier 무시 -- html은 애초에 AzureDI/VLM을 안 쓴다.
    html_text = data.decode("utf-8", errors="replace")
    walker = _DOMWalker()
    walker.feed(html_text)
    return walker.elements
