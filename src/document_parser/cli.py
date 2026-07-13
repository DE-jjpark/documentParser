"""Command-line entrypoint for document-parser."""

import argparse

from document_parser import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="document-parser",
        description="A document parsing toolkit.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main() -> None:
    parser = build_parser()
    parser.parse_args()
    print("Hello from document-parser!")


if __name__ == "__main__":
    main()
