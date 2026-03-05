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

import hashlib
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
        search_engine: Optional[Any] = None,
        entity_extractor: Optional[Any] = None,
        relationship_extractor: Optional[Any] = None,
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
        self._search_engine = search_engine
        self._entity_extractor = entity_extractor
        self._relationship_extractor = relationship_extractor

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

        # Compute document-level content hash for deduplication.
        try:
            file_bytes = path.read_bytes()
            doc_content_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
        except Exception:
            doc_content_hash = ""

        # Step 0: Remove existing chunks for this document to prevent duplicates on re-upload.
        if self._retrieval and document_title:
            try:
                deleted = self._retrieval.delete_by_filter({"document_title": document_title})
                if deleted:
                    logger.info("Deleted existing chunks before re-ingestion",
                                document_title=document_title, deleted=deleted)
            except Exception as e:
                logger.warning("Failed to deduplicate before ingestion (non-fatal)", error=str(e))

        # Step 0b: Remove existing chunks by content hash (catches renamed re-uploads).
        if self._retrieval and doc_content_hash:
            try:
                deleted = self._retrieval.delete_by_filter({"doc_content_hash": doc_content_hash})
                if deleted:
                    logger.info("Deleted existing chunks by content hash",
                                doc_content_hash=doc_content_hash, deleted=deleted)
            except Exception as e:
                logger.warning("Content-hash dedup failed (non-fatal)", error=str(e))

        # Also remove from keyword search engine.
        if self._search_engine and document_title:
            try:
                if hasattr(self._search_engine, 'delete_by_metadata'):
                    self._search_engine.delete_by_metadata("document_title", document_title)
                    logger.info("Deleted existing search engine entries", document_title=document_title)
            except Exception as e:
                logger.warning("Failed to deduplicate search engine (non-fatal)", error=str(e))

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
        # Sync processors run first (no I/O).
        try:
            text_chunks = self._text_processor.process(text_with_headers, document_title)
            all_chunks.extend(text_chunks)
        except Exception as e:
            result.errors.append(f"Text processing error: {e}")
            logger.error("Text processing failed", error=str(e))

        try:
            callout_chunks = self._callout_processor.process(callout_elements, document_title)
            all_chunks.extend(callout_chunks)
        except Exception as e:
            result.errors.append(f"Callout processing error: {e}")
            logger.error("Callout processing failed", error=str(e))

        # Async processors run in parallel (independent I/O operations).
        import asyncio

        async_results = await asyncio.gather(
            self._table_processor.process(table_elements, document_title),
            self._formula_processor.process(formula_elements, document_title),
            self._image_processor.process(image_elements, document_title),
            return_exceptions=True,
        )

        for label, chunks_or_error in zip(
            ("Table", "Formula", "Image"), async_results
        ):
            if isinstance(chunks_or_error, BaseException):
                result.errors.append(f"{label} processing error: {chunks_or_error}")
                logger.error(f"{label} processing failed", error=str(chunks_or_error))
            else:
                all_chunks.extend(chunks_or_error)

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
            # Compute per-chunk content hash for dedup.
            record.content_hash = hashlib.sha256(
                record.display_text.encode("utf-8")
            ).hexdigest()[:16]
            # Store document-level hash for bulk dedup on re-upload.
            record.metadata["doc_content_hash"] = doc_content_hash
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

        # Step 5.5: Extract entities for each chunk.
        self._extract_entities(records)

        # Step 5.7: Extract entity relationships via LLM (async).
        await self._extract_relationships(records, result)

        # Step 5.8: Flush graph store writes to make entities searchable.
        if self._relationship_extractor:
            graph = getattr(self._relationship_extractor, "_graph", None)
            if graph and hasattr(graph, "flush"):
                graph.flush()

        # Step 6: Embed and store in Qdrant.
        if not skip_embedding and self._embedding and self._retrieval:
            try:
                await self._embed_and_store(records)
            except Exception as e:
                result.errors.append(f"Embedding/storage error: {e}")
                logger.error("Embedding/storage failed", error=str(e))

        # Step 7: Index in keyword search engine (Elasticsearch) for hybrid search.
        self._index_in_search_engine(records)

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

    def _index_in_search_engine(self, records: List[ChunkRecord]) -> None:
        """Push chunk records to keyword search engine (Elasticsearch/BM25).

        Graceful degradation: failure here must not block ingestion.
        """
        if not self._search_engine:
            return

        docs = []
        for rec in records:
            if rec.display_text and len(rec.display_text.strip()) > 10:
                docs.append({
                    "doc_id": rec.chunk_id,
                    "content": rec.display_text,
                    "title": rec.document_title,
                    "metadata": {
                        "node_id": rec.chunk_id,
                        "document_id": rec.document_id,
                        "document_title": rec.document_title,
                        "element_type": rec.element_type,
                        "section": rec.section,
                        "page": rec.page,
                        "primary_framework": rec.jurisdiction,
                        "product_type": rec.product_type,
                        "entities": rec.entities,
                        "entity_orgs": rec.entity_orgs,
                        "entity_persons": rec.entity_persons,
                    },
                })
        if docs:
            try:
                count = self._search_engine.add_documents(docs)
                logger.info("Indexed chunks in search engine", count=count)
            except Exception as e:
                logger.error("Search engine indexing failed (non-fatal)", error=str(e))

    def _extract_entities(self, records: List[ChunkRecord]) -> None:
        """Extract entities for each chunk and populate entity fields.

        Uses batch extraction via spaCy's nlp.pipe() for performance.
        Graceful degradation: if the extractor is unavailable or fails,
        chunks simply have empty entity lists.
        """
        if not self._entity_extractor:
            return

        start = time.time()

        # Use batch extraction if available (3-5x faster via nlp.pipe()).
        if hasattr(self._entity_extractor, "extract_batch"):
            try:
                texts = [r.display_text for r in records]
                all_entities = self._entity_extractor.extract_batch(texts)
                extracted_count = 0
                total_entity_count = 0
                for record, entities in zip(records, all_entities):
                    record.entities = entities.all_entities
                    record.entity_orgs = entities.organizations
                    record.entity_persons = entities.persons
                    if entities.entity_count > 0:
                        extracted_count += 1
                        total_entity_count += entities.entity_count

                duration = time.time() - start
                logger.info(
                    "Batch entity extraction complete",
                    total_chunks=len(records),
                    chunks_with_entities=extracted_count,
                    coverage_pct=round(extracted_count / max(len(records), 1) * 100, 1),
                    avg_entities_per_chunk=round(total_entity_count / max(extracted_count, 1), 1),
                    duration_seconds=round(duration, 2),
                    throughput_chunks_per_sec=round(len(records) / max(duration, 0.001), 1),
                )
                return
            except Exception as e:
                logger.warning(
                    "Batch entity extraction failed, falling back to per-chunk",
                    error=str(e),
                )

        # Fallback: per-chunk extraction.
        extracted_count = 0
        total_entity_count = 0
        for record in records:
            try:
                entities = self._entity_extractor.extract(record.display_text)
                record.entities = entities.all_entities
                record.entity_orgs = entities.organizations
                record.entity_persons = entities.persons
                if entities.entity_count > 0:
                    extracted_count += 1
                    total_entity_count += entities.entity_count
            except Exception as e:
                logger.warning(
                    "Entity extraction failed for chunk (non-fatal)",
                    chunk_id=record.chunk_id,
                    error=str(e),
                )

        duration = time.time() - start
        logger.info(
            "Entity extraction complete",
            total_chunks=len(records),
            chunks_with_entities=extracted_count,
            coverage_pct=round(extracted_count / max(len(records), 1) * 100, 1),
            avg_entities_per_chunk=round(total_entity_count / max(extracted_count, 1), 1),
            duration_seconds=round(duration, 2),
        )

    async def _extract_relationships(
        self, records: List[ChunkRecord], result: IngestionResult
    ) -> None:
        """Extract entity relationships via LLM and populate knowledge graph.

        Runs after _extract_entities so chunks have populated entity fields.
        Graceful degradation: if the extractor is unavailable or fails,
        ingestion continues without relationships.
        """
        if not self._relationship_extractor:
            return

        try:
            extraction_results = await self._relationship_extractor.extract_relationships(records)
            total_rels = sum(len(r.relationships) for r in extraction_results)
            if total_rels > 0:
                logger.info(
                    "Relationship extraction complete",
                    chunks_processed=len(extraction_results),
                    total_relationships=total_rels,
                )
        except Exception as e:
            result.errors.append(f"Relationship extraction error: {e}")
            logger.warning(
                "Relationship extraction failed (non-fatal)", error=str(e)
            )

    async def _embed_and_store(self, records: List[ChunkRecord]) -> None:
        """Embed chunk texts and upsert into Qdrant in batches."""
        if not records:
            return

        embed_batch_size = 100  # ~100 chunks keeps us well under OpenAI's 300K token limit
        total_stored = 0

        for i in range(0, len(records), embed_batch_size):
            batch_records = records[i : i + embed_batch_size]
            texts = [r.embedding_text for r in batch_records]
            vectors = await self._embedding.embed_batch(texts)

            points = [
                (rec.chunk_id, vec, rec.to_qdrant_payload())
                for rec, vec in zip(batch_records, vectors)
            ]
            self._retrieval.upsert_batch(points)
            total_stored += len(points)
            logger.info("Stored batch in Qdrant", batch=i // embed_batch_size + 1, count=len(points))

        logger.info("Stored chunks in Qdrant", count=total_stored)

    @staticmethod
    def _count_by_type(elements: List[DocumentElement]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in elements:
            key = e.element_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts
