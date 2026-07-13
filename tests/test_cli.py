from document_parser import __version__
from document_parser.cli import build_parser


def test_version_is_set():
    assert __version__ == "0.1.0"


def test_parser_builds():
    parser = build_parser()
    args = parser.parse_args([])
    assert args is not None
