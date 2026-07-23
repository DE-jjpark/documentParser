"""LLM 기반 heading 계층 구조 추정 — pdf/__init__.py의 _assign_heading_levels
(번호 매김 > doc_title > 폰트 크기 순위)와 나란히 두고 비교 평가하기 위한
병렬 경로. pdf.load()의 heading_strategy="llm"으로 켠다(기본값은 여전히
"font_size" — 기존 함수는 그대로 둔다).

_assign_heading_levels와 달리 규칙 기반 우선순위를 섞지 않는다 — 문서 전체
heading 목록(텍스트 + 페이지 + PP-DocLayoutV2 block_type 힌트)을 한 번에
LLM에 보여주고 계층을 처음부터 통째로 판단하게 한다. 번호가 있는 제목도
LLM이 다르게 볼 수 있다 — 그게 두 파이프라인을 비교하는 포인트다.

폰트 크기 접근의 알려진 약점(PPT처럼 디자인 요소마다 크기가 제각각이라
군집화가 틀어지는 경우)을 LLM이 의미 이해로 더 잘 잡아내는지 실측으로
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

_JSON_ARRAY = re.compile(r"\[[\s\S]*\]")


def _build_prompt(headings: list[DocumentElement]) -> str:
    lines = [_PROMPT_HEADER, ""]
    for i, el in enumerate(headings, start=1):
        block_type = el.metadata.get("block_type", "unknown")
        lines.append(f"{i}. (p.{el.page}, {block_type}) {el.text}")
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


def assign_heading_levels_llm(elements: list[DocumentElement]) -> list[DocumentElement]:
    """metadata["level"]을 LLM 판단으로 매긴다 -- _assign_heading_levels와
    같은 필드를 채우므로 호출부(pdf.load) 입장에서는 전략만 바꾸면 된다.
    자격증명 누락 등 실제 에러는 vlm.py의 다른 VLM 호출들과 동일하게 그대로
    전파한다(get_client()가 MissingDependencyError를 던짐) -- heading_strategy
    ="llm"을 명시적으로 켠 상태에서 조용히 레벨 없이 넘어가면 원인을 알기
    어렵다. 타임아웃(complete_text_with_hard_timeout이 text=""로 흡수)이나
    응답 형식이 안 맞는 경우만 관대하게 넘어간다(level 미설정) --
    _assign_heading_levels가 끝내 못 정한 heading을 그냥 두는 것과 같은 처리."""
    heading_indices = [i for i, el in enumerate(elements) if el.type == ElementType.HEADING]
    if not heading_indices:
        return elements

    headings = [elements[i] for i in heading_indices]
    prompt = _build_prompt(headings)

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
        metadata["level_source"] = "llm"
        result_elements[i] = el.model_copy(update={"metadata": metadata})
    return result_elements
