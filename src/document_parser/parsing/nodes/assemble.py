from document_parser.core.models import ParsedDocument
from document_parser.parsing.state import ParsingState


def assemble(state: ParsingState) -> dict:
    """Build the ParsedDocument contract object from the extracted elements."""
    elements = state.get("elements", [])
    document = ParsedDocument(
        source=state["source"],
        format=state["format"],
        elements=elements,
        metadata={"element_count": len(elements)},
    )
    return {"document": document}
