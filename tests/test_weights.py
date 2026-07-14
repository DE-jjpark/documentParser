import sys
import types

from document_parser.parsing import weights


def test_default_model_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCUMENT_PARSER_MODEL_DIR", str(tmp_path / "custom"))
    assert weights.default_model_dir() == tmp_path / "custom"


def test_default_model_dir_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv("DOCUMENT_PARSER_MODEL_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert weights.default_model_dir() == tmp_path / "document-parser" / "models"


def test_download_layout_model(monkeypatch, tmp_path):
    calls: dict = {}
    stub = types.SimpleNamespace(snapshot_download=lambda **kwargs: calls.update(kwargs))
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)

    dest = weights.download_layout_model(dest=tmp_path / "models")

    assert dest == tmp_path / "models"
    assert dest.is_dir()
    assert calls["repo_id"] == "PaddlePaddle/PP-DocLayoutV2"
    assert calls["revision"] == weights.DEFAULT_REVISION
    assert calls["local_dir"] == str(dest)


def test_download_layout_model_default_dest(monkeypatch, tmp_path):
    calls: dict = {}
    stub = types.SimpleNamespace(snapshot_download=lambda **kwargs: calls.update(kwargs))
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)
    monkeypatch.setenv("DOCUMENT_PARSER_MODEL_DIR", str(tmp_path))

    dest = weights.download_layout_model(revision="my-rev")

    assert dest == tmp_path / "PP-DocLayoutV2"
    assert calls["revision"] == "my-rev"


def test_cli_accepts_download_models_subcommand():
    from document_parser.cli import build_parser

    args = build_parser().parse_args(["download-models", "--dest", "/tmp/x", "--revision", "abc"])
    assert args.command == "download-models"
    assert args.dest == "/tmp/x"
    assert args.revision == "abc"
