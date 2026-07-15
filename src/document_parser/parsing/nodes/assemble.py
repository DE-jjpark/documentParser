from document_parser.core.models import ParsedDocument
from document_parser.parsing.state import ParsingState


def assemble(state: ParsingState) -> dict:
    """Build the ParsedDocument contract object from the extracted elements.

    포맷 무관하게 각 요소에 "{page}-{type}-{n}" 형태의 key를 붙인다(n은 같은
    페이지·같은 타입 안에서 1부터 매기는 일련번호) — elements 리스트 순서가
    이미 읽기 순서이므로, 이 카운터도 그 순서 그대로 채우면 된다. 페이지
    개념이 없는 포맷(txt/md)은 page가 None이라 0으로 취급한다.
    """
    elements = state.get("elements", [])
    counters: dict[tuple[int, str], int] = {}
    keyed_elements = []
    for el in elements:
        page = el.page if el.page is not None else 0
        counter_key = (page, el.type.value)
        counters[counter_key] = counters.get(counter_key, 0) + 1
        keyed_elements.append(
            el.model_copy(update={"key": f"{page}-{el.type.value}-{counters[counter_key]}"})
        )

    document = ParsedDocument(
        source=state["source"],
        format=state["format"],
        elements=keyed_elements,
        metadata={"element_count": len(keyed_elements)},
    )
    return {"document": document}
