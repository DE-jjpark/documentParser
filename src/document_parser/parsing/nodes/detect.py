from pathlib import PurePath

from document_parser.core.exceptions import UnsupportedFormatError
from document_parser.parsing.state import ParsingState


def detect_format(state: ParsingState) -> dict:
    """Resolve the document format from an explicit hint or the file extension."""
    fmt = state.get("format") or PurePath(state["source"]).suffix.lstrip(".").lower()
    if not fmt:
        raise UnsupportedFormatError(
            f"cannot determine format of {state['source']!r}; pass format= explicitly"
        )
    return {"format": fmt.lower()}
