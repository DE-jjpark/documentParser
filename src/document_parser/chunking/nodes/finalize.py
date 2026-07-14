import hashlib

from document_parser.chunking.state import ChunkingState
from document_parser.core.models import Chunk


def finalize(state: ChunkingState) -> dict:
    """Assign stable ids and ordering to the split pieces."""
    chunks = [
        Chunk(
            id=hashlib.sha1(f"{index}:{piece.text}".encode()).hexdigest()[:12],
            index=index,
            text=piece.text,
            metadata=piece.metadata,
        )
        for index, piece in enumerate(state.get("pieces", []))
    ]
    return {"chunks": chunks}
