"""
Ingestion Pipeline — Multi-modal document processing.

Replaces the legacy text-extraction-only pipeline with layout-aware,
type-specific processing that preserves tables, formulas, callouts,
and diagrams as first-class elements.

Usage:
    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    result = await pipeline.ingest("path/to/doc.pdf", document_id="doc-1")
"""

from .layout_parser import (
    LayoutParser,
    DocumentElement,
    ElementType,
    BBox,
)
from .chunk_schema import ChunkRecord, StructuredStore, SCHEMA_VERSION
from .pipeline import IngestionPipeline, IngestionResult

__all__ = [
    "LayoutParser",
    "DocumentElement",
    "ElementType",
    "BBox",
    "ChunkRecord",
    "StructuredStore",
    "SCHEMA_VERSION",
    "IngestionPipeline",
    "IngestionResult",
]
