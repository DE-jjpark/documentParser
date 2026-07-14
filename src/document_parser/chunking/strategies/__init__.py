"""Split strategies.

A strategy is a callable ``(text: str, config: ChunkingConfig) -> list[str]``.
Register new strategies with ``register()``.
"""

from collections.abc import Callable

from document_parser.chunking.strategies import recursive
from document_parser.core.models import ChunkingConfig

Strategy = Callable[[str, ChunkingConfig], list[str]]

_REGISTRY: dict[str, Strategy] = {}


def register(name: str, strategy: Strategy) -> None:
    _REGISTRY[name] = strategy


def get_strategy(name: str) -> Strategy | None:
    return _REGISTRY.get(name)


def available_strategies() -> list[str]:
    return sorted(_REGISTRY)


register("recursive", recursive.split)
