"""LLM 기반 heading 계층 구조 추정 — pdf/__init__.py의 _assign_heading_levels
(번호 매김 > doc_title > 폰트 크기 순위)와 나란히 두고 비교 평가하기 위한
병렬 경로. pdf.load()의 heading_strategy="llm"/"llm_categorized"로 켠다
(기본값은 여전히 "font_size" — 기존 함수는 그대로 둔다).

두 LLM 변형이 있다(둘 다 규칙 기반 우선순위 없이 LLM이 계층을 처음부터
통째로 판단한다는 점은 같음 -- _assign_heading_levels와 다른 지점):
  - "llm"(assign_heading_levels_llm): doc_title/paragraph_title/figure_title
    을 전부 똑같이 "제목"으로 뭉뚱그려 보여주고, block_type은 "참고만 하라"는
    가벼운 힌트로만 준다.
  - "llm_categorized"(assign_heading_levels_llm_categorized): 세 카테고리를
    구조적으로 다른 역할로 명시한다 -- doc_title은 최상위 제목 후보, figure_title
    은 문서 아웃라인이 아니라 그림/표에 딸린 캡션성 제목(직전 본문 제목보다
    한 단계 아래), paragraph_title만 실제 본문 섹션 계층을 이룬다는 전제를
    프롬프트에 못박는다. "다 heading으로 퉁치지 말고 구분지어서" 판단하면
    결과가 달라지는지 비교하려는 목적.

폰트 크기 접근의 알려진 약점(PPT처럼 디자인 요소마다 크기가 제각각이라
군집화가 틀어지는 경우)을 LLM이 의미 이해로 더 잘 잡아내는지, 그리고
카테고리 구분을 명시하는 게 실제로 더 나은 계층을 만드는지 실측으로
비교하려는 목적."""

from __future__ import annotations

import json
import re

from document_parser.core.models import DocumentElement, ElementType
from document_parser.parsing.loaders.vlm_caption import complete_text_with_hard_timeout, get_client

_MAX_LEVEL = 5

_PROMPT_HEADER = (
    "아래는 한 문서에서 등장 순서대로 뽑은 제목(heading) 목록이다. 각 제목이 "
    "문서 구조상 몇 단계 깊이에 있는지 판단해줘 -- 1은 최상위 제목(문서/장 "
    "전체 제목), 숫자가 클수록 더 하위 항목이다. 제목 텍스트의 의미와 문맥, "
    "페이지 번호를 참고해서 판단하되, block_type 힌트(doc_title=문서/슬라이드 "
    "전체 제목 가능성 높음, paragraph_title=본문 절 제목, figure_title=그림/표 "
    "캡션성 제목)는 참고만 하고 최종 판단은 실제 문서 구조로 해라.\n\n"
    "답변은 반드시 JSON 배열 하나만 출력해라(다른 설명 문장 없이). 배열 "
    f"길이는 입력 목록 개수와 정확히 같아야 하고, 각 원소는 1~{_MAX_LEVEL} "
    "사이의 정수다. 예: [1, 2, 2, 1, 3]"
)

# "llm"과 달리 block_type을 "참고용 힌트"가 아니라 구조적 전제로 못박는다 --
# 세 카테고리가 문서 구조에서 하는 역할 자체가 다르다고 보고, 그 역할에 맞는
# 판단 규칙을 명시적으로 준다.
_CATEGORIZED_PROMPT_HEADER = (
    "아래는 한 문서에서 등장 순서대로 뽑은 제목(heading) 목록이다. 각 제목이 "
    "문서 구조상 몇 단계 깊이에 있는지 판단해줘 -- 1은 최상위 제목, 숫자가 "
    "클수록 더 하위 항목이다.\n\n"
    "제목마다 PP-DocLayoutV2가 매긴 block_type이 붙어 있는데, 이 세 카테고리는 "
    "문서 구조에서 서로 다른 역할을 한다 -- 셋을 똑같은 '제목'으로 뭉뚱그리지 "
    "말고 아래 규칙에 따라 판단해라:\n"
    "- doc_title: 문서/슬라이드 전체를 대표하는 최상위 제목일 가능성이 높다. "
    "특별히 문맥상 아니라고 판단할 근거가 없으면 레벨 1로 봐라.\n"
    "- paragraph_title: 본문의 실제 절/하위절 제목이다. 문서의 진짜 아웃라인은 "
    "이 카테고리들의 의미·순서 관계로 구성된다고 보고 계층을 판단해라.\n"
    "- figure_title: 특정 그림/표/차트에 달린 캡션성 제목이다. 본문 아웃라인의 "
    "정식 항목이 아니라, 바로 앞(또는 가장 가까운) paragraph_title/doc_title의 "
    "하위 항목처럼 취급해라 -- 그 제목보다 한 단계 아래 레벨로 판단해라.\n\n"
    "제목 텍스트의 의미와 문맥, 페이지 번호도 같이 참고해서 최종 판단해라.\n\n"
    "답변은 반드시 JSON 배열 하나만 출력해라(다른 설명 문장 없이). 배열 "
    f"길이는 입력 목록 개수와 정확히 같아야 하고, 각 원소는 1~{_MAX_LEVEL} "
    "사이의 정수다. 예: [1, 2, 3, 2, 1]"
)

_JSON_ARRAY = re.compile(r"\[[\s\S]*\]")


def _build_prompt(headings: list[DocumentElement], header: str = _PROMPT_HEADER) -> str:
    lines = [header, ""]
    for i, el in enumerate(headings, start=1):
        block_type = el.metadata.get("block_type", "unknown")
        # 제목 텍스트에 줄바꿈이 섞여 있으면(실측: 여러 줄짜리 슬라이드 표지
        # 제목) "번호 하나 = 한 줄" 형식이 깨져서 LLM이 항목 개수를 잘못 세고,
        # 응답 배열 길이가 안 맞아 _parse_levels가 통째로 실패한다(실측 확인:
        # 117개 입력에 118개짜리 응답이 옴) -- 한 줄로 접어서 방지한다.
        text = " ".join(el.text.split())
        lines.append(f"{i}. (p.{el.page}, {block_type}) {text}")
    return "\n".join(lines)


def _parse_levels(response_text: str, expected_count: int) -> list[int | None] | None:
    """LLM 응답에서 레벨 배열을 뽑는다. 형식이 안 맞으면(길이 불일치, JSON
    파싱 실패, 범위 밖 값 등) None을 돌려줘서 호출부가 통째로 폴백(레벨 미설정)
    하게 한다 -- 일부만 신뢰할 수 없는 응답을 억지로 끼워맞추지 않는다."""
    match = _JSON_ARRAY.search(response_text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or len(parsed) != expected_count:
        return None

    levels: list[int | None] = []
    for value in parsed:
        if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= _MAX_LEVEL:
            levels.append(value)
        else:
            levels.append(None)
    return levels


def _assign_heading_levels_via_prompt(
    elements: list[DocumentElement], header: str, level_source: str
) -> list[DocumentElement]:
    """metadata["level"]을 LLM 판단으로 매긴다 -- _assign_heading_levels와
    같은 필드를 채우므로 호출부(pdf.load) 입장에서는 전략만 바꾸면 된다.
    자격증명 누락 등 실제 에러는 vlm.py의 다른 VLM 호출들과 동일하게 그대로
    전파한다(get_client()가 MissingDependencyError를 던짐) -- heading_strategy
    를 명시적으로 켠 상태에서 조용히 레벨 없이 넘어가면 원인을 알기 어렵다.
    타임아웃(complete_text_with_hard_timeout이 text=""로 흡수)이나 응답
    형식이 안 맞는 경우만 관대하게 넘어간다(level 미설정) --
    _assign_heading_levels가 끝내 못 정한 heading을 그냥 두는 것과 같은 처리."""
    heading_indices = [i for i, el in enumerate(elements) if el.type == ElementType.HEADING]
    if not heading_indices:
        return elements

    headings = [elements[i] for i in heading_indices]
    prompt = _build_prompt(headings, header=header)

    client = get_client()
    result = complete_text_with_hard_timeout(client, prompt)

    levels = _parse_levels(result.text, len(headings))
    if levels is None:
        return elements

    result_elements = list(elements)
    for i, level in zip(heading_indices, levels, strict=True):
        if level is None:
            continue
        el = elements[i]
        metadata = dict(el.metadata)
        metadata["level"] = level
        metadata["level_source"] = level_source
        result_elements[i] = el.model_copy(update={"metadata": metadata})
    return result_elements


def assign_heading_levels_llm(elements: list[DocumentElement]) -> list[DocumentElement]:
    """doc_title/paragraph_title/figure_title을 전부 동일한 "제목"으로 보고
    LLM이 계층을 판단한다(block_type은 참고용 힌트일 뿐)."""
    return _assign_heading_levels_via_prompt(elements, _PROMPT_HEADER, level_source="llm")


def assign_heading_levels_llm_categorized(elements: list[DocumentElement]) -> list[DocumentElement]:
    """세 block_type이 문서 구조에서 서로 다른 역할을 한다는 걸 프롬프트에
    구조적 전제로 못박는다(_CATEGORIZED_PROMPT_HEADER 참고) -- assign_heading_
    levels_llm과 결과가 어떻게 달라지는지 비교하기 위한 세 번째 병렬 경로."""
    return _assign_heading_levels_via_prompt(
        elements, _CATEGORIZED_PROMPT_HEADER, level_source="llm_categorized"
    )
