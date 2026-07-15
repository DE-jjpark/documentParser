"""Contract models shared between the engines and library consumers.

Everything here is a plain Pydantic model: serializable with ``.model_dump()``,
importable without pulling in LangGraph or any format-specific dependency.
"""

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator


class ElementType(StrEnum):
    TEXT = "text"
    HEADING = "heading"
    LIST = "list"
    TABLE = "table"
    IMAGE = "image"


class BBox(BaseModel):
    """Axis-aligned bounding box in the source page's native coordinate space."""

    x0: float
    y0: float
    x1: float
    y1: float


class DocumentElement(BaseModel):
    """A single structural unit extracted from a source document.

    ``bboxes`` is a list, not a single optional box: one element can
    legitimately span multiple regions (e.g. an Azure Document Intelligence
    paragraph that wraps across a column break has multiple
    ``bounding_regions``). Empty means "no region info" (e.g. txt/md, which
    have no coordinate space at all).
    """

    type: ElementType = ElementType.TEXT
    text: str
    page: int | None = None
    bboxes: list[BBox] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # "e{n}" (1부터) — assemble 노드가 elements 순서(=읽기 순서) 그대로 문서
    # 전체에 매기는 전역 일련번호. 페이지/타입 무관.
    elem_id: str | None = None


class ParsedDocument(BaseModel):
    """Output contract of the parsing engine."""

    source: str
    format: str
    elements: list[DocumentElement] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(el.text for el in self.elements if el.text)


class Segment(BaseModel):
    """Input unit of the chunking engine.

    Deliberately decoupled from ParsedDocument so the chunking engine can be
    fed from any source; the pipeline layer converts between the two.
    """

    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkingConfig(BaseModel):
    strategy: str = "recursive"
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class Chunk(BaseModel):
    """Output contract of the chunking engine."""

    id: str
    index: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
