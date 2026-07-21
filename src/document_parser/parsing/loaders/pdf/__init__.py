"""PDF 문서 로더. 'pdf' extra(pymupdf + pdfplumber) 필요.

페이지마다 ``graph.py``의 LangGraph를 한 번씩 invoke한다: 레이아웃 분석 →
4-way 라우팅(``layout.route_page`` 참고) → native/azure_di/vlm 조합 → 병합.
그래프는 이 함수가 반환하는 바깥 형태(``(bytes, source) -> list[DocumentElement]``)
에는 영향을 주지 않는다 — 로더 레지스트리(parsing/loaders/__init__.py) 입장
에서는 여전히 평범한 동기 함수 하나일 뿐이다.

같은 PDF를 pymupdf와 pdfplumber 둘 다로 연다 — pymupdf는 페이지를 이미지로
렌더링하는 용도(레이아웃 모델 입력, AzureDI/VLM 크롭)로 계속 쓰고,
native(순수 텍스트) 추출만 pdfplumber로 바꿨다(요청: "plumber 사용할거고
native일 때 사용하도록"). 둘의 좌표계가 같아서(포인트 좌표, 좌상단 원점)
같은 bbox를 그대로 재사용할 수 있다(실측 확인함)."""

from functools import lru_cache
from io import BytesIO

from document_parser.core.exceptions import MissingDependencyError
from document_parser.core.models import DocumentElement


@lru_cache(maxsize=1)
def _get_page_graph():
    from document_parser.parsing.loaders.pdf.graph import build_page_graph

    return build_page_graph().compile()


def load(data: bytes, source: str, tier: str = "balanced") -> list[DocumentElement]:
    try:
        import pymupdf
    except ImportError as exc:
        raise MissingDependencyError(
            "PDF support requires the 'pdf' extra: pip install 'document-parser[pdf]'"
        ) from exc

    try:
        import pdfplumber
    except ImportError as exc:
        raise MissingDependencyError(
            "PDF support requires the 'pdf' extra: pip install 'document-parser[pdf]'"
        ) from exc

    graph = _get_page_graph()
    elements: list[DocumentElement] = []
    with pymupdf.open(stream=data, filetype="pdf") as doc, pdfplumber.open(BytesIO(data)) as pdoc:
        for page_number, page in enumerate(doc, start=1):
            plumber_page = pdoc.pages[page_number - 1]
            result = graph.invoke(
                {
                    "page": page,
                    "plumber_page": plumber_page,
                    "page_number": page_number,
                    "raw_elements": [],
                    "tier": tier,
                }
            )
            elements.extend(result["elements"])
    return elements
