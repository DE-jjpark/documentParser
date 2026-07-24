"""Command-line entrypoint for document-parser."""

import argparse
import json
import sys

from document_parser import (
    ChunkingConfig,
    DocumentParserError,
    IngestPipeline,
    ParsingTier,
    __version__,
)
from document_parser.parsing.weights import LAYOUT_MODEL_REPO, download_layout_model


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
    subparsers = parser.add_subparsers(dest="command")

    parse_cmd = subparsers.add_parser("parse", help="parse a document and print it as JSON")
    parse_cmd.add_argument("file", help="path to the document")
    parse_cmd.add_argument("--format", help="override format detection (e.g. txt, md, pdf)")
    parse_cmd.add_argument(
        "--tier",
        choices=[t.value for t in ParsingTier],
        default=ParsingTier.BALANCED.value,
        help="'fast' = native/plumber only, no VLM calls; 'balanced' = full pipeline (default)",
    )

    ingest_cmd = subparsers.add_parser(
        "ingest", help="parse and chunk a document, print chunks as JSON"
    )
    ingest_cmd.add_argument("file", help="path to the document")
    ingest_cmd.add_argument("--format", help="override format detection")
    ingest_cmd.add_argument(
        "--tier",
        choices=[t.value for t in ParsingTier],
        default=ParsingTier.BALANCED.value,
        help="'fast' = native/plumber only, no VLM calls; 'balanced' = full pipeline (default)",
    )
    ingest_cmd.add_argument("--strategy", default="recursive")
    ingest_cmd.add_argument("--chunk-size", type=int, default=1000)
    ingest_cmd.add_argument("--chunk-overlap", type=int, default=200)

    download_cmd = subparsers.add_parser(
        "download-models",
        help="download model weights for the parsing engine (requires the 'layout' extra)",
    )
    download_cmd.add_argument(
        "--dest",
        help="target directory (default: $DOCUMENT_PARSER_MODEL_DIR "
        "or ~/.cache/document-parser/models)",
    )
    download_cmd.add_argument(
        "--revision",
        default=None,
        help="HuggingFace revision to download (default: pinned known-good revision)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    try:
        if args.command == "parse":
            document = IngestPipeline().parsing.parse(args.file, format=args.format, tier=args.tier)
            print(document.model_dump_json(indent=2))
        elif args.command == "ingest":
            config = ChunkingConfig(
                strategy=args.strategy,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            chunks = IngestPipeline().ingest(
                args.file, format=args.format, tier=args.tier, config=config
            )
            print(json.dumps([chunk.model_dump() for chunk in chunks], indent=2))
        elif args.command == "download-models":
            kwargs = {"revision": args.revision} if args.revision else {}
            path = download_layout_model(dest=args.dest, **kwargs)
            print(f"downloaded {LAYOUT_MODEL_REPO} to {path}")
    except DocumentParserError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
