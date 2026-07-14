"""Composition layer: the only place that knows both engines."""

from document_parser.pipeline.ingest import IngestPipeline, document_to_segments

__all__ = ["IngestPipeline", "document_to_segments"]
