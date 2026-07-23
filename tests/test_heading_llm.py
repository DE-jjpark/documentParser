"""heading_llm.py(LLM 기반 제목 계층 구조 추정) 단위 테스트.

실제 자격증명/네트워크 호출 없음: get_client()/complete_text_with_hard_timeout()
을 모듈 레벨에서 mock한다 -- 프롬프트 배선과 응답 파싱만 확인.
"""

import re
from unittest.mock import patch

from document_parser.core.models import DocumentElement, ElementType
from document_parser.parsing.clients.vlm import VLMCaptionResult
from document_parser.parsing.loaders.pdf.heading_llm import (
    _parse_levels,
    assign_heading_levels_llm,
    assign_heading_levels_llm_categorized,
)


def _heading(text: str, page: int = 1, block_type: str = "paragraph_title") -> DocumentElement:
    return DocumentElement(
        type=ElementType.HEADING, text=text, page=page, metadata={"block_type": block_type}
    )


def test_no_headings_returns_elements_unchanged():
    elements = [DocumentElement(type=ElementType.TEXT, text="just body text")]

    with patch("document_parser.parsing.loaders.pdf.heading_llm.get_client") as mock_get_client:
        result = assign_heading_levels_llm(elements)

    assert result == elements
    mock_get_client.assert_not_called()


def test_assigns_levels_from_valid_json_response():
    elements = [_heading("Overview"), _heading("Details"), _heading("Sub-details")]

    with patch(
        "document_parser.parsing.loaders.pdf.heading_llm.complete_text_with_hard_timeout",
        return_value=VLMCaptionResult(text="[1, 2, 3]"),
    ) as mock_complete:
        with patch("document_parser.parsing.loaders.pdf.heading_llm.get_client"):
            result = assign_heading_levels_llm(elements)

    assert [el.metadata["level"] for el in result] == [1, 2, 3]
    assert all(el.metadata["level_source"] == "llm" for el in result)
    mock_complete.assert_called_once()


def test_prompt_collapses_embedded_newlines_to_keep_one_line_per_item():
    """실측 회귀 테스트: 제목 텍스트에 줄바꿈이 있으면(여러 줄짜리 슬라이드
    표지 제목 등) "번호 하나 = 한 줄" 형식이 깨져서 LLM이 항목 개수를
    잘못 세고 응답 길이가 안 맞아 전체가 실패했다(117개 입력에 118개 응답)."""
    elements = [
        _heading("Line one\nLine two\n  Line three  "),
        _heading("Next item"),
    ]

    with patch(
        "document_parser.parsing.loaders.pdf.heading_llm.complete_text_with_hard_timeout",
        return_value=VLMCaptionResult(text="[1, 2]"),
    ) as mock_complete:
        with patch("document_parser.parsing.loaders.pdf.heading_llm.get_client"):
            assign_heading_levels_llm(elements)

    prompt = mock_complete.call_args.args[1]
    assert "Line one Line two Line three" in prompt
    item_lines = [line for line in prompt.splitlines() if re.match(r"^\d+\. \(p\.", line)]
    assert len(item_lines) == 2


def test_prompt_includes_heading_text_page_and_block_type():
    elements = [_heading("02.2 개발 비용", page=5, block_type="doc_title")]

    with patch(
        "document_parser.parsing.loaders.pdf.heading_llm.complete_text_with_hard_timeout",
        return_value=VLMCaptionResult(text="[1]"),
    ) as mock_complete:
        with patch("document_parser.parsing.loaders.pdf.heading_llm.get_client"):
            assign_heading_levels_llm(elements)

    prompt = mock_complete.call_args.args[1]
    assert "02.2 개발 비용" in prompt
    assert "p.5" in prompt
    assert "doc_title" in prompt


def test_only_heading_elements_are_sent_and_updated():
    """TEXT/TABLE 등 섞여 있어도 HEADING만 프롬프트에 들어가고 레벨도 그것만
    업데이트돼야 한다 -- 원래 순서·다른 element는 그대로 보존."""
    text_el = DocumentElement(type=ElementType.TEXT, text="body")
    heading_el = _heading("Title")
    elements = [text_el, heading_el]

    with patch(
        "document_parser.parsing.loaders.pdf.heading_llm.complete_text_with_hard_timeout",
        return_value=VLMCaptionResult(text="[2]"),
    ):
        with patch("document_parser.parsing.loaders.pdf.heading_llm.get_client"):
            result = assign_heading_levels_llm(elements)

    assert result[0] == text_el
    assert "level" not in result[0].metadata
    assert result[1].metadata["level"] == 2


def test_malformed_response_leaves_elements_unchanged():
    elements = [_heading("A"), _heading("B")]

    with patch(
        "document_parser.parsing.loaders.pdf.heading_llm.complete_text_with_hard_timeout",
        return_value=VLMCaptionResult(text="not json at all"),
    ):
        with patch("document_parser.parsing.loaders.pdf.heading_llm.get_client"):
            result = assign_heading_levels_llm(elements)

    assert "level" not in result[0].metadata
    assert "level" not in result[1].metadata


def test_length_mismatch_response_leaves_elements_unchanged():
    """heading 3개인데 배열이 2개만 오면 일부만 억지로 맞추지 않고 통째로
    폴백한다(_parse_levels의 계약)."""
    elements = [_heading("A"), _heading("B"), _heading("C")]

    with patch(
        "document_parser.parsing.loaders.pdf.heading_llm.complete_text_with_hard_timeout",
        return_value=VLMCaptionResult(text="[1, 2]"),
    ):
        with patch("document_parser.parsing.loaders.pdf.heading_llm.get_client"):
            result = assign_heading_levels_llm(elements)

    assert all("level" not in el.metadata for el in result)


def test_out_of_range_value_leaves_only_that_heading_unset():
    elements = [_heading("A"), _heading("B")]

    with patch(
        "document_parser.parsing.loaders.pdf.heading_llm.complete_text_with_hard_timeout",
        return_value=VLMCaptionResult(text="[1, 99]"),
    ):
        with patch("document_parser.parsing.loaders.pdf.heading_llm.get_client"):
            result = assign_heading_levels_llm(elements)

    assert result[0].metadata["level"] == 1
    assert "level" not in result[1].metadata


def test_response_wrapped_in_prose_or_code_fence_still_parses():
    elements = [_heading("A"), _heading("B")]

    with patch(
        "document_parser.parsing.loaders.pdf.heading_llm.complete_text_with_hard_timeout",
        return_value=VLMCaptionResult(text="Here you go:\n```json\n[2, 1]\n```"),
    ):
        with patch("document_parser.parsing.loaders.pdf.heading_llm.get_client"):
            result = assign_heading_levels_llm(elements)

    assert [el.metadata["level"] for el in result] == [2, 1]


def test_parse_levels_rejects_non_integer_and_boolean_values():
    assert _parse_levels(str([1, "two", 3]).replace("'", '"'), 3) == [1, None, 3]
    assert _parse_levels("[true, 2]", 2) == [None, 2]


def test_categorized_assigns_levels_and_tags_level_source():
    elements = [
        _heading("Deck Title", block_type="doc_title"),
        _heading("Chart caption", block_type="figure_title"),
    ]

    with patch(
        "document_parser.parsing.loaders.pdf.heading_llm.complete_text_with_hard_timeout",
        return_value=VLMCaptionResult(text="[1, 2]"),
    ) as mock_complete:
        with patch("document_parser.parsing.loaders.pdf.heading_llm.get_client"):
            result = assign_heading_levels_llm_categorized(elements)

    assert [el.metadata["level"] for el in result] == [1, 2]
    assert all(el.metadata["level_source"] == "llm_categorized" for el in result)
    mock_complete.assert_called_once()


def test_categorized_prompt_spells_out_role_per_block_type():
    """"llm"과 다르게 block_type을 참고용 힌트가 아니라 구조적 전제로
    못박아야 한다 -- 세 카테고리 각각에 대한 명시적 판단 규칙이 프롬프트에
    있는지 확인."""
    elements = [_heading("Title", block_type="doc_title")]

    with patch(
        "document_parser.parsing.loaders.pdf.heading_llm.complete_text_with_hard_timeout",
        return_value=VLMCaptionResult(text="[1]"),
    ) as mock_complete:
        with patch("document_parser.parsing.loaders.pdf.heading_llm.get_client"):
            assign_heading_levels_llm_categorized(elements)

    prompt = mock_complete.call_args.args[1]
    assert "doc_title" in prompt
    assert "paragraph_title" in prompt
    assert "figure_title" in prompt
    # "llm"(assign_heading_levels_llm)의 가벼운 힌트 문구와는 달라야 한다.
    assert "참고만 하고" not in prompt


def test_categorized_and_plain_llm_use_independent_prompts():
    """두 전략이 서로 다른 프롬프트를 쓰는지(하나가 다른 하나를 그냥 감싸는
    게 아닌지) 직접 비교."""
    elements = [_heading("Overview", block_type="paragraph_title")]

    with patch(
        "document_parser.parsing.loaders.pdf.heading_llm.complete_text_with_hard_timeout",
        return_value=VLMCaptionResult(text="[1]"),
    ) as mock_complete:
        with patch("document_parser.parsing.loaders.pdf.heading_llm.get_client"):
            assign_heading_levels_llm(elements)
            plain_prompt = mock_complete.call_args.args[1]

            assign_heading_levels_llm_categorized(elements)
            categorized_prompt = mock_complete.call_args.args[1]

    assert plain_prompt != categorized_prompt


def test_categorized_malformed_response_leaves_elements_unchanged():
    elements = [_heading("A"), _heading("B")]

    with patch(
        "document_parser.parsing.loaders.pdf.heading_llm.complete_text_with_hard_timeout",
        return_value=VLMCaptionResult(text="nope"),
    ):
        with patch("document_parser.parsing.loaders.pdf.heading_llm.get_client"):
            result = assign_heading_levels_llm_categorized(elements)

    assert all("level" not in el.metadata for el in result)
