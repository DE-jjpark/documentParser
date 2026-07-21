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
    # 슬라이드 노트(pptx/ppt)·문서 댓글(docx) 등, 본문이 아니라 본문에 "붙은" 메모.
    # office.py가 PDF 변환과 별도로 원본 zip에서 직접 뽑아 채운다 — PDF 변환
    # 과정에서 이런 메모는 사라지기 때문(LibreOffice 기본 변환 필터가 노트
    # 페이지/댓글을 export하지 않음, 실측으로 확인함).
    NOTE = "note"


class ParsingTier(StrEnum):
    """엔진 호출자가 고르는 속도/비용 대 품질 트레이드오프.

    - FAST: native(pdfplumber)만 쓴다 — AzureDI/VLM 호출 자체를 안 한다.
      텍스트 레이어 없는(스캔) 페이지는 뽑을 방법이 없어 그대로 유실된다 —
      "빠르고 무료"의 대가로 감수하는 것.
    - BALANCED: 지금까지의 기본 파이프라인 그대로(표/그림 있으면 AzureDI·
      VLM까지 태움).
    """

    FAST = "fast"
    BALANCED = "balanced"


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
    block_id: str | None = None
    # TABLE/IMAGE 전용 — 원본 "내용"(text: 표는 마크다운, 이미지는 설명/전사/
    # Mermaid)과 별도로, VLM이 만든 요약을 따로 둔다. 순수 텍스트류(TEXT/
    # HEADING/LIST)는 요약이라는 개념 자체가 없어서 항상 None.
    summary: str | None = None


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
