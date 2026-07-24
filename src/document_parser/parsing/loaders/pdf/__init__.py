"""PDF 문서 로더. 'pdf' extra(pymupdf + pdfplumber) 필요.

페이지마다 ``graph.py``의 LangGraph를 한 번씩 invoke한다: 레이아웃 분석 →
3-way 라우팅(``layout.route_page`` 참고) → native/vlm 조합 → 병합. AzureDI는
더 이상 쓰지 않는다(팀 결정) — 표 구조 추출도 스캔 페이지 본문 추출도 전부
VLM이 담당한다. 그래프는 이 함수가 반환하는 바깥 형태(``(bytes, source) ->
list[DocumentElement]``)에는 영향을 주지 않는다 — 로더 레지스트리
(parsing/loaders/__init__.py) 입장에서는 여전히 평범한 동기 함수 하나일 뿐이다.

같은 PDF를 pymupdf와 pdfplumber 둘 다로 연다 — pymupdf는 페이지를 이미지로
렌더링하는 용도(레이아웃 모델 입력, VLM 크롭)로 계속 쓰고, native(순수 텍스트)
추출만 pdfplumber로 바꿨다(요청: "plumber 사용할거고 native일 때 사용하도록").
둘의 좌표계가 같아서(포인트 좌표, 좌상단 원점) 같은 bbox를 그대로 재사용할 수
있다(실측 확인함).

모든 페이지를 합친 뒤 _flag_continued_tables()로 표가 페이지를 넘어 이어지는
것 같은 경우를 표시하고(청킹이 이전 페이지 표도 같이 봐야 하는지 판단하는
힌트, metadata["continued_from_previous_page"]), _assign_heading_levels()로
제목들의 계층 레벨(metadata["level"])을 매긴다."""

import re
from functools import lru_cache
from io import BytesIO

from document_parser.core.exceptions import MissingDependencyError
from document_parser.core.models import DocumentElement, ElementType


@lru_cache(maxsize=1)
def _get_page_graph():
    from document_parser.parsing.loaders.pdf.graph import build_page_graph

    return build_page_graph().compile()


# 이 이상 페이지에서 글자 하나까지 똑같이 반복되는 TEXT/HEADING이면 러닝헤더/
# 대제목/로고 텍스트 같은 상투 문구로 본다(실측: pptx는 매 슬라이드 첫 요소가
# 항상 데크 제목, docx는 매 페이지가 "문서명 / 버전" 러닝헤더로 시작 — 이걸
# 안 걸러내면 "페이지의 첫 요소가 표"라는 조건이 진짜 연결에서도 항상 실패함).
_BOILERPLATE_MIN_PAGES = 3


def _is_boilerplate(el: DocumentElement, repeated_texts: set[str]) -> bool:
    """반복 상투 문구이거나(위 참고), PP-DocLayoutV2가 이미 "number" 카테고리로
    잡아준 페이지 번호(예: "12 / 28")면 실제 콘텐츠가 아니라고 본다 — 페이지
    번호는 페이지마다 글자가 달라서 반복 문구 탐지로는 못 잡으므로 라벨을
    따로 본다."""
    if el.metadata.get("block_type") == "number":
        return True
    return el.type in (ElementType.TEXT, ElementType.HEADING) and el.text.strip() in repeated_texts


def _repeated_texts(elements: list[DocumentElement], min_pages: int) -> set[str]:
    pages_by_text: dict[str, set[int]] = {}
    for el in elements:
        if el.type not in (ElementType.TEXT, ElementType.HEADING) or el.page is None:
            continue
        text = el.text.strip()
        if not text:
            continue
        pages_by_text.setdefault(text, set()).add(el.page)
    return {text for text, pages in pages_by_text.items() if len(pages) >= min_pages}


def _flag_continued_tables(elements: list[DocumentElement]) -> list[DocumentElement]:
    """표가 페이지를 넘어 이어지는 것 같은 경우 metadata["continued_from_previous_page"]
    에 힌트를 남긴다 — 청킹이 이 표를 볼 때 이전 페이지 표도 같이 봐야 문맥이
    맞을 수 있다는 신호. "헤더 유무"로 판단하려 했으나 VLM 표 프롬프트(vlm.py의
    _TABLE_PROMPT)가 원본에 헤더가 있든 없든 항상 마크다운 헤더 구분줄을 강제로
    만들게 하므로 신뢰할 수 없는 신호였다 — 대신 위치(바로 앞 요소)+페이지 인접만
    본다: 러닝헤더/대제목/페이지번호 같은 상투 문구를 걸러낸 "콘텐츠" 관점에서,
    직전 요소도 표이고 페이지 번호가 정확히 하나 차이면 이어지는 것으로 본다.
    완벽한 판단은 아니고(오탐/누락 둘 다 있을 수 있음) 청킹이 참고할 힌트일 뿐이다."""
    repeated_texts = _repeated_texts(elements, _BOILERPLATE_MIN_PAGES)
    content_indices = [
        i for i, el in enumerate(elements) if not _is_boilerplate(el, repeated_texts)
    ]

    continued_indices: set[int] = set()
    for pos, i in enumerate(content_indices):
        if pos == 0:
            continue
        el = elements[i]
        prev = elements[content_indices[pos - 1]]
        if (
            el.type == ElementType.TABLE
            and prev.type == ElementType.TABLE
            and el.page is not None
            and prev.page is not None
            and el.page == prev.page + 1
        ):
            continued_indices.add(i)

    result: list[DocumentElement] = []
    for i, el in enumerate(elements):
        if i in continued_indices:
            metadata = dict(el.metadata)
            metadata["continued_from_previous_page"] = True
            el = el.model_copy(update={"metadata": metadata})
        result.append(el)
    return result


# "1", "1.1", "4.2.1", "03.2" 같은 아라비아 숫자 점 표기 — 점 개수+1이 레벨.
_ARABIC_NUMBERING = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){0,5})\.?(?:\s|$)")
# "가.", "나)" 같은 한글 낱글자 번호 매김 — 한국어 공문서에서 숫자(1, 2, 3)
# 아래 두 번째 계층으로 쓰이는 게 흔한 관례라 레벨 2로 고정 가정한다(완벽하진
# 않지만 번호 정보가 아예 없는 것보다 낫다).
_KOREAN_NUMBERING = re.compile(r"^([가-힣])[.)]\s")


def _numbering_level(text: str) -> int | None:
    text = text.strip()
    match = _ARABIC_NUMBERING.match(text)
    if match:
        return match.group(1).count(".") + 1
    if _KOREAN_NUMBERING.match(text):
        return 2
    return None


# 폰트 크기 기반 레벨은 번호 매김이 없을 때만 쓰는 최후 수단이다 — 실측
# (30장짜리 PPT 덱)에서 디자인 요소마다 크기가 제각각이라(20개 넘는 서로
# 다른 크기) 크기별로 그냥 순위를 매기면 레벨이 13, 17까지 치솟는 문제가
# 있었다. 그래서 연속된 두 크기의 차이가 이 값(포인트) 이하면 같은 레벨로
# 묶고(미세한 디자인 차이로 보고 무시), 최종 레벨도 이 값에서 잘라낸다.
_FONT_SIZE_CLUSTER_GAP = 3.0
_MAX_FONT_SIZE_LEVEL = 5


def _font_size_levels(sizes_desc: list[float]) -> dict[float, int]:
    levels: dict[float, int] = {}
    level = 1
    prev: float | None = None
    for size in sizes_desc:
        if prev is not None and prev - size > _FONT_SIZE_CLUSTER_GAP:
            level = min(level + 1, _MAX_FONT_SIZE_LEVEL)
        levels[size] = level
        prev = size
    return levels


def _assign_heading_levels(elements: list[DocumentElement]) -> list[DocumentElement]:
    """제목(HEADING) element에 metadata["level"]을 매긴다 — 마크다운/HTML
    로더가 이미 쓰고 있는 것과 같은 필드(1이 최상위). PP-DocLayoutV2는
    doc_title/paragraph_title/figure_title 3개 카테고리만 주고 숫자 계층이
    없어서(예: "1. 개요"와 "1.1 핵심 가치"가 둘 다 paragraph_title) 직접
    역산해야 한다. 우선순위:
      1. 제목 텍스트 맨 앞의 번호 매김(1/1.1/가)이 있으면 그걸로 판단(워드
         문서에서 흔함, 실측 확인) — doc_title 카테고리보다 먼저 본다. 실측
         (PPT)에서 doc_title이 페이지 전체 제목이 아니라 그냥 일반 번호
         매겨진 슬라이드 제목("02.2 ...")에도 잘못 붙는 걸 확인해서, 번호가
         명시적으로 있으면 그쪽을 더 신뢰한다.
      2. 번호가 없고 doc_title 카테고리면 레벨 1(문서 전체 제목이라는 뜻으로
         잘 쓰인 경우).
      3. 그래도 없으면(PPT 슬라이드 제목처럼) native.py가 남겨둔
         metadata["font_size"]를 문서 전체 제목들과 비교한 군집 순위로
         대체한다 — 폰트 크기가 클수록 상위 레벨."""
    heading_indices = [i for i, el in enumerate(elements) if el.type == ElementType.HEADING]
    font_sizes_desc = sorted(
        {
            elements[i].metadata["font_size"]
            for i in heading_indices
            if "font_size" in elements[i].metadata
        },
        reverse=True,
    )
    font_size_levels = _font_size_levels(font_sizes_desc)

    result = list(elements)
    for i in heading_indices:
        el = elements[i]
        level = _numbering_level(el.text)
        if level is None and el.metadata.get("block_type") == "doc_title":
            level = 1
        if level is None and "font_size" in el.metadata:
            level = font_size_levels.get(el.metadata["font_size"])
        if level is None:
            continue
        metadata = dict(el.metadata)
        metadata["level"] = level
        result[i] = el.model_copy(update={"metadata": metadata})
    return result


_HEADING_STRATEGIES = ("font_size", "llm", "llm_categorized")


def _extract_raw_elements(data: bytes, tier: str) -> list[DocumentElement]:
    """레이아웃 분석 → 라우팅 → native/vlm 조합 → 병합 → 표 연결 힌트까지,
    heading 레벨 배정(_assign_heading_levels 등) 이전 단계. load()와
    scripts/compare_heading_strategies.py가 같이 쓴다 -- 세 heading_strategy
    를 비교할 때 이 비싼 단계(특히 VLM 그림/표 캡션 호출)를 매번 다시
    돌리지 않고 한 번만 실행해서 결과를 재사용하려는 목적."""
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
    return _flag_continued_tables(elements)


def assign_heading_levels(
    elements: list[DocumentElement], heading_strategy: str
) -> list[DocumentElement]:
    """_extract_raw_elements()가 만든 elements에 heading_strategy에 맞는
    레벨 배정 함수를 적용한다 -- load()와 비교 스크립트가 공유하는 분기점."""
    if heading_strategy not in _HEADING_STRATEGIES:
        raise ValueError(
            f"unknown heading_strategy {heading_strategy!r}; expected one of {_HEADING_STRATEGIES}"
        )
    if heading_strategy == "llm":
        from document_parser.parsing.loaders.pdf.heading_llm import assign_heading_levels_llm

        return assign_heading_levels_llm(elements)
    if heading_strategy == "llm_categorized":
        from document_parser.parsing.loaders.pdf.heading_llm import (
            assign_heading_levels_llm_categorized,
        )

        return assign_heading_levels_llm_categorized(elements)
    return _assign_heading_levels(elements)


def load(
    data: bytes,
    source: str,
    tier: str = "balanced",
    heading_strategy: str = "font_size",
) -> list[DocumentElement]:
    """``heading_strategy``: "font_size"(기본, _assign_heading_levels) /
    "llm"(heading_llm.assign_heading_levels_llm, block_type은 참고용 힌트) /
    "llm_categorized"(heading_llm.assign_heading_levels_llm_categorized,
    doc_title/paragraph_title/figure_title을 구조적으로 다른 역할로 프롬프트에
    못박음) -- 셋 중 뭐가 실제 문서에서 더 정확한 계층 구조를 뽑는지 비교
    평가하기 위한 병렬 경로들. loaders 레지스트리(parsing/loaders/__init__.py)
    의 공통 시그니처(tier까지)에는 아직 안 실었다 -- CLI/엔진까지 배선하는 건
    비교가 끝난 뒤에 결정."""
    if heading_strategy not in _HEADING_STRATEGIES:
        raise ValueError(
            f"unknown heading_strategy {heading_strategy!r}; expected one of {_HEADING_STRATEGIES}"
        )
    elements = _extract_raw_elements(data, tier)
    return assign_heading_levels(elements, heading_strategy)
