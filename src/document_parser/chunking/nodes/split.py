from document_parser.chunking.state import ChunkingState
from document_parser.chunking.strategies import available_strategies, get_strategy
from document_parser.core.exceptions import ChunkingFailedError
from document_parser.core.models import Segment


def split(state: ChunkingState) -> dict:
    """Apply the configured strategy to every input segment."""
    config = state["config"]
    strategy = get_strategy(config.strategy)
    if strategy is None:
        raise ChunkingFailedError(
            f"unknown strategy {config.strategy!r}; available: {', '.join(available_strategies())}"
        )
    pieces: list[Segment] = []
    for segment in state["segments"]:
        for piece_text in strategy(segment.text, config):
            pieces.append(Segment(text=piece_text, metadata=dict(segment.metadata)))
    return {"pieces": pieces}
