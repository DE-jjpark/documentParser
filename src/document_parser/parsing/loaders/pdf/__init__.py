"""PDF 문서 로더. 'pdf' extra(pymupdf) 필요.

페이지마다 ``graph.py``의 LangGraph를 한 번씩 invoke한다: 레이아웃 분석 →
(Only Text + 텍스트 레이어 있음) 네이티브 추출, 아니면(그림 있음 / 텍스트
레이어 없음) AzureDI+VLM 병렬 실행 → 병합. 그래프는 이 함수가 반환하는
바깥 형태(``(bytes, source) -> list[DocumentElement]``)에는 영향을 주지
않는다 — 로더 레지스트리(parsing/loaders/__init__.py) 입장에서는 여전히
평범한 동기 함수 하나일 뿐이다.
"""

from functools import lru_cache

from document_parser.core.exceptions import MissingDependencyError
from document_parser.core.models import DocumentElement


@lru_cache(maxsize=1)
def _get_page_graph():
    from document_parser.parsing.loaders.pdf.graph import build_page_graph

    return build_page_graph().compile()


def load(data: bytes, source: str) -> list[DocumentElement]:
    try:
        import pymupdf
    except ImportError as exc:
        raise MissingDependencyError(
            "PDF support requires the 'pdf' extra: pip install 'document-parser[pdf]'"
        ) from exc

    graph = _get_page_graph()
    elements: list[DocumentElement] = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for page_number, page in enumerate(doc, start=1):
            result = graph.invoke({"page": page, "page_number": page_number, "elements": []})
            elements.extend(result["elements"])
    return elements
