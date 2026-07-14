from document_parser.core.exceptions import UnsupportedFormatError
from document_parser.parsing.loaders import get_loader, supported_formats
from document_parser.parsing.state import ParsingState


def extract(state: ParsingState) -> dict:
    """Dispatch to the loader registered for the detected format."""
    fmt = state["format"]
    loader = get_loader(fmt)
    if loader is None:
        raise UnsupportedFormatError(
            f"unsupported format {fmt!r}; supported: {', '.join(supported_formats())}"
        )
    return {"elements": loader(state["data"], state["source"])}
