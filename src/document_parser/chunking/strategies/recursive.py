"""Greedy splitter that prefers paragraph, then line, then word boundaries."""

from document_parser.core.models import ChunkingConfig

_SEPARATORS = ("\n\n", "\n", " ")


def split(text: str, config: ChunkingConfig) -> list[str]:
    size = config.chunk_size
    overlap = config.chunk_overlap
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        if end < length:
            # Break at the latest natural boundary in the second half of the
            # window so chunks stay reasonably full.
            min_cut = start + size // 2
            for sep in _SEPARATORS:
                cut = text.rfind(sep, min_cut, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks
