from document_parser.core.models import ParsedDocument
from document_parser.parsing.state import ParsingState


def assemble(state: ParsingState) -> dict:
    """Build the ParsedDocument contract object from the extracted elements."""
    elements = state.get("elements", [])
    keyed_elements = [
        el.model_copy(update={"elem_id": f"e{i}"}) for i, el in enumerate(elements, start=1)
    ]
    document = ParsedDocument(
        source=state["source"],
        format=state["format"],
        elements=keyed_elements,
        metadata={"element_count": len(keyed_elements)},
    )
    return {"document": document}
