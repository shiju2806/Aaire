"""
Ingestion pipeline orchestrator.

Ties together layout parsing, type-specific processing, chunk schema
creation, embedding, and storage into Qdrant + structured store.

Usage:
    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    result = await pipeline.ingest("path/to/document.pdf", document_id="doc-123")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from .layout_parser import DocumentElement, ElementType, LayoutParser
from .processors.text_processor import TextProcessor, ProcessedChunk
from .processors.table_processor import TableProcessor
from .processors.formula_processor import FormulaProcessor
from .processors.callout_processor import CalloutProcessor
from .processors.image_processor import ImageProcessor
from .chunk_schema import ChunkRecord, StructuredStore, SCHEMA_VERSION

logger = structlog.get_logger()


@dataclass
class IngestionResult:
    """Result of ingesting a single document."""

    document_id: str
    file_path: str
    total_elements: int = 0
    total_chunks: int = 0
    element_counts: Dict[str, int] = field(default_factory=dict)
    chunk_counts: Dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)


class IngestionPipeline:
    """Orchestrate the full document ingestion flow.

    Steps:
    1. Parse document → List[DocumentElement]  (layout parser)
    2. Route elements → type-specific processors
    3. Produce ProcessedChunk objects
    4. Convert to ChunkRecord (versioned schema)
    5. Embed and store in Qdrant
    6. Store originals in structured store
    """

    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        embedding_provider: Optional[Any] = None,
        retrieval_provider: Optional[Any] = None,
        structured_store_dir: str = "data/structured_store",
    ) -> None:
        # Core components.
        self._parser = LayoutParser()
        self._structured_store = StructuredStore(structured_store_dir)

        # Type-specific processors.
        self._text_processor = TextProcessor()
        self._table_processor = TableProcessor(llm_provider=llm_provider)
        self._formula_processor = FormulaProcessor(llm_provider=llm_provider)
        self._callout_processor = CalloutProcessor()
        self._image_processor = ImageProcessor(llm_provider=llm_provider)

        # Provider references (may be None during testing).
        self._llm = llm_provider
        self._embedding = embedding_provider
        self._retrieval = retrieval_provider

    async def ingest(
        self,
        file_path: str | Path,
        document_id: str = "",
        document_title: str = "",
        jurisdiction: str = "unknown",
        product_type: str = "general",
        skip_embedding: bool = False,
    ) -> IngestionResult:
        """Ingest a document through the full pipeline.

        Args:
            file_path: Path to the document file.
            document_id: Unique identifier for the document.
            document_title: Human-readable title (used in context prefixes).
            jurisdiction: IFRS|US_GAAP|US_STAT|unknown.
            product_type: universal_life|whole_life|term|general.
            skip_embedding: If True, skip embedding + Qdrant storage
                           (useful for testing processors only).

        Returns:
            IngestionResult with stats and any errors.
        """
        start_time = time.time()
        path = Path(file_path)
        result = IngestionResult(
            document_id=document_id or path.stem,
            file_path=str(path),
        )

        if not document_title:
            document_title = path.stem.replace("_", " ").replace("-", " ").title()

        # Step 1: Parse document into typed elements.
        try:
            elements = self._parser.parse(path)
        except Exception as e:
            result.errors.append(f"Parsing failed: {e}")
            result.duration_seconds = time.time() - start_time
            logger.error("Ingestion parsing failed", path=str(path), error=str(e))
            return result

        result.total_elements = len(elements)
        result.element_counts = self._count_by_type(elements)
        logger.info(
            "Document parsed",
            path=str(path),
            elements=len(elements),
            types=result.element_counts,
        )

        # Step 2 & 3: Route to processors and collect chunks.
        all_chunks: List[ProcessedChunk] = []

        # Group elements by type.
        text_elements = [e for e in elements if e.element_type in (ElementType.TEXT, ElementType.LIST)]
        table_elements = [e for e in elements if e.element_type == ElementType.TABLE]
        formula_elements = [e for e in elements if e.element_type == ElementType.FORMULA]
        callout_elements = [e for e in elements if e.element_type == ElementType.CALLOUT]
        image_elements = [e for e in elements if e.element_type == ElementType.IMAGE]
        # Headers are included with text for section-aware chunking.
        header_elements = [e for e in elements if e.element_type == ElementType.HEADER]
        text_with_headers = sorted(
            text_elements + header_elements,
            key=lambda e: (e.page_number, getattr(e.bounding_box, "y0", 0) if e.bounding_box else 0),
        )

        # Process each type.
        try:
            text_chunks = self._text_processor.process(text_with_headers, document_title)
            all_chunks.extend(text_chunks)
        except Exception as e:
            result.errors.append(f"Text processing error: {e}")
            logger.error("Text processing failed", error=str(e))

        try:
            table_chunks = await self._table_processor.process(table_elements, document_title)
            all_chunks.extend(table_chunks)
        except Exception as e:
            result.errors.append(f"Table processing error: {e}")
            logger.error("Table processing failed", error=str(e))

        try:
            formula_chunks = await self._formula_processor.process(formula_elements, document_title)
            all_chunks.extend(formula_chunks)
        except Exception as e:
            result.errors.append(f"Formula processing error: {e}")
            logger.error("Formula processing failed", error=str(e))

        try:
            callout_chunks = self._callout_processor.process(callout_elements, document_title)
            all_chunks.extend(callout_chunks)
        except Exception as e:
            result.errors.append(f"Callout processing error: {e}")
            logger.error("Callout processing failed", error=str(e))

        try:
            image_chunks = await self._image_processor.process(image_elements, document_title)
            all_chunks.extend(image_chunks)
        except Exception as e:
            result.errors.append(f"Image processing error: {e}")
            logger.error("Image processing failed", error=str(e))

        # Step 4: Convert to ChunkRecords.
        embedding_model = ""
        if self._embedding and hasattr(self._embedding, "model_name"):
            embedding_model = self._embedding.model_name

        records: List[ChunkRecord] = []
        for chunk in all_chunks:
            record = ChunkRecord.from_processed_chunk(
                chunk,
                document_id=result.document_id,
                document_title=document_title,
                embedding_model=embedding_model,
                jurisdiction=jurisdiction,
                product_type=product_type,
            )
            records.append(record)

        result.total_chunks = len(records)
        result.chunk_counts = {}
        for r in records:
            result.chunk_counts[r.element_type] = result.chunk_counts.get(r.element_type, 0) + 1

        logger.info(
            "Chunks created",
            total=len(records),
            types=result.chunk_counts,
        )

        # Step 5: Store structured content (tables, formulas, images).
        self._store_structured_content(records, result.document_id)

        # Step 6: Embed and store in Qdrant.
        if not skip_embedding and self._embedding and self._retrieval:
            try:
                await self._embed_and_store(records)
            except Exception as e:
                result.errors.append(f"Embedding/storage error: {e}")
                logger.error("Embedding/storage failed", error=str(e))

        result.duration_seconds = time.time() - start_time
        logger.info(
            "Ingestion complete",
            document_id=result.document_id,
            chunks=result.total_chunks,
            duration=f"{result.duration_seconds:.1f}s",
            errors=len(result.errors),
        )
        return result

    # -- internal steps -----------------------------------------------------

    def _store_structured_content(
        self, records: List[ChunkRecord], document_id: str
    ) -> None:
        """Store original-fidelity content for non-text chunks."""
        for record in records:
            if record.element_type in ("table", "formula", "image"):
                content_type = record.element_type
                content = None

                if record.element_type == "table":
                    content = record.metadata.get("original_markdown", record.display_text)
                elif record.element_type == "formula":
                    content = record.metadata.get("latex", record.display_text)
                elif record.element_type == "image":
                    content = record.metadata.get("image_ref", record.display_text)

                if content:
                    self._structured_store.store(
                        chunk_id=record.chunk_id,
                        content_type=content_type,
                        content=content,
                        metadata={"document_id": document_id},
                    )

    async def _embed_and_store(self, records: List[ChunkRecord]) -> None:
        """Embed chunk texts and upsert into Qdrant."""
        if not records:
            return

        # Batch embed all chunks.
        texts = [r.embedding_text for r in records]
        vectors = await self._embedding.embed_batch(texts)

        # Prepare Qdrant points.
        points = []
        for record, vector in zip(records, vectors):
            points.append({
                "id": record.chunk_id,
                "vector": vector,
                "payload": record.to_qdrant_payload(),
            })

        # Upsert in batches.
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await self._retrieval.upsert_batch(
                collection_name="documents",
                points=batch,
            )

        logger.info("Stored chunks in Qdrant", count=len(points))

    @staticmethod
    def _count_by_type(elements: List[DocumentElement]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in elements:
            key = e.element_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts
