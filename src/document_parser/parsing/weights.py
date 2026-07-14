"""Model weight download helpers for the parsing engine.

Weights are fetched from HuggingFace and pinned to a known revision so that
deployments are reproducible; pass a different revision explicitly to upgrade.
"""

import os
from pathlib import Path

from document_parser.core.exceptions import MissingDependencyError

LAYOUT_MODEL_REPO = "PaddlePaddle/PP-DocLayoutV2"
# Pinned to the repo state as of 2026-01-29.
DEFAULT_REVISION = "b73668227b14316a38f8b345d6b474e4f1f0b84d"


def default_model_dir() -> Path:
    """Resolve the model cache root: $DOCUMENT_PARSER_MODEL_DIR, else XDG cache."""
    if env := os.environ.get("DOCUMENT_PARSER_MODEL_DIR"):
        return Path(env)
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_home) if cache_home else Path.home() / ".cache"
    return base / "document-parser" / "models"


def download_layout_model(
    dest: str | Path | None = None,
    revision: str = DEFAULT_REVISION,
) -> Path:
    """Download the PP-DocLayoutV2 weights and return the local directory.

    Idempotent: already-downloaded files are reused. Requires the 'layout'
    extra (huggingface_hub).
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise MissingDependencyError(
            "model download requires the 'layout' extra: pip install 'document-parser[layout]'"
        ) from exc

    target = Path(dest) if dest else default_model_dir() / LAYOUT_MODEL_REPO.rsplit("/", 1)[-1]
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=LAYOUT_MODEL_REPO, revision=revision, local_dir=str(target))
    return target
