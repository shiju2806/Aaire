"""
LLM-based entity-relationship extraction at ingestion time.

After spaCy/regex entity extraction, this module sends chunks (with ≥2 entities)
to the LLM for structured relationship extraction.  The LLM sees the chunk text
and pre-extracted entities, and returns triples: (source, target, relationship_type).

Cost controls:
1. Entity gate — skip chunks with <2 entities (~70% eliminated)
2. Content-hash cache — SHA256 of chunk text → skip if already processed
3. Batch prompts — group chunks per LLM call to reduce API overhead
4. Incremental extraction — only new/changed chunks on re-ingestion
5. Config kill switch — ``knowledge_graph.extraction.enabled: false``
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from ..providers.config_loader import get_config, get_nested

logger = structlog.get_logger()

# Prompt template path
_PROMPT_DIR = Path(__file__).resolve().parent.parent / "generation" / "prompts"
_PROMPT_FILE = _PROMPT_DIR / "entity_relationship_extraction.txt"


@dataclass
class ExtractedRelationship:
    """A single relationship triple extracted by the LLM."""

    source_name: str
    target_name: str
    rel_type: str
    confidence: float = 0.0


@dataclass
class ExtractionResult:
    """Result of relationship extraction for one chunk."""

    chunk_id: str
    relationships: List[ExtractedRelationship] = field(default_factory=list)
    new_entities: List[Dict[str, Any]] = field(default_factory=list)


class RelationshipExtractor:
    """Extract entity-relationship triples from chunks using LLM.

    Args:
        llm_provider: An ``LLMProvider`` instance (``generate_json`` method required).
        graph_store: Optional ``GraphStore`` for immediate upsert during extraction.
        config: Override config dict (default: loads ``config/knowledge_graph.yaml``).
    """

    def __init__(
        self,
        llm_provider: Any,
        graph_store: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._llm = llm_provider
        self._graph = graph_store
        self._config = config or get_config("knowledge_graph")
        self._prompt_template = self._load_prompt()

        # Content-hash cache
        cache_dir = get_nested(
            self._config, "extraction", "cache_dir",
            default="data/entity_extraction_cache",
        )
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract_relationships(
        self,
        records: List[Any],
    ) -> List[ExtractionResult]:
        """Extract entity-relationship triples from chunk records.

        Only processes chunks with ≥ ``min_entities_per_chunk`` entities.
        Uses ``generate_json()`` for structured output.

        Args:
            records: List of ``ChunkRecord`` objects with populated
                     ``entities``, ``entity_orgs``, ``entity_persons`` fields.

        Returns:
            List of ``ExtractionResult`` per processed chunk.
        """
        # Kill switch check
        if not get_nested(self._config, "extraction", "enabled", default=True):
            logger.info("Relationship extraction disabled by config")
            return []

        min_entities = get_nested(
            self._config, "extraction", "min_entities_per_chunk", default=2
        )

        # Filter to chunks with enough entities
        eligible = [
            r for r in records
            if len(getattr(r, "entities", []) or []) >= min_entities
        ]

        if not eligible:
            logger.info(
                "No chunks eligible for relationship extraction",
                total=len(records),
                min_entities=min_entities,
            )
            return []

        # Filter out already-cached chunks
        to_process = []
        for r in eligible:
            content_hash = self._content_hash(r.display_text)
            if not self._is_cached(content_hash):
                to_process.append((r, content_hash))

        if not to_process:
            logger.info(
                "All eligible chunks already in extraction cache",
                eligible=len(eligible),
            )
            return []

        logger.info(
            "Starting relationship extraction",
            total_chunks=len(records),
            eligible=len(eligible),
            to_process=len(to_process),
        )

        start = time.time()
        results: List[ExtractionResult] = []

        batch_size = get_nested(
            self._config, "extraction", "batch_size", default=5
        )

        # Process chunks in batches to reduce API overhead
        for batch_start in range(0, len(to_process), batch_size):
            batch = to_process[batch_start : batch_start + batch_size]

            if len(batch) > 1:
                # Batch extraction: group multiple chunks in one LLM call
                try:
                    batch_records = [r for r, _ in batch]
                    batch_results = await self._extract_from_batch(batch_records)
                    for (record, content_hash), result in zip(batch, batch_results):
                        results.append(result)
                        if self._graph and result.relationships:
                            self._upsert_to_graph(record, result)
                        self._mark_cached(content_hash)
                except Exception as e:
                    # Fallback: process individually on batch failure
                    logger.warning("Batch extraction failed, falling back to per-chunk", error=str(e))
                    for record, content_hash in batch:
                        try:
                            result = await self._extract_from_chunk(record)
                            results.append(result)
                            if self._graph and result.relationships:
                                self._upsert_to_graph(record, result)
                            self._mark_cached(content_hash)
                        except Exception as e2:
                            logger.warning(
                                "Relationship extraction failed for chunk (non-fatal)",
                                chunk_id=getattr(record, "chunk_id", "?"),
                                error=str(e2),
                            )
            else:
                # Single chunk — no batching needed
                record, content_hash = batch[0]
                try:
                    result = await self._extract_from_chunk(record)
                    results.append(result)
                    if self._graph and result.relationships:
                        self._upsert_to_graph(record, result)
                    self._mark_cached(content_hash)
                except Exception as e:
                    logger.warning(
                        "Relationship extraction failed for chunk (non-fatal)",
                        chunk_id=getattr(record, "chunk_id", "?"),
                        error=str(e),
                    )

        duration = time.time() - start
        total_rels = sum(len(r.relationships) for r in results)
        logger.info(
            "Relationship extraction complete",
            chunks_processed=len(results),
            total_relationships=total_rels,
            duration_seconds=round(duration, 2),
        )

        return results

    # ------------------------------------------------------------------
    # Per-chunk extraction
    # ------------------------------------------------------------------

    async def _extract_from_chunk(self, record: Any) -> ExtractionResult:
        """Extract relationships from a single chunk via LLM."""
        entities = getattr(record, "entities", []) or []
        chunk_text = getattr(record, "display_text", "")
        chunk_id = getattr(record, "chunk_id", "")

        prompt = self._prompt_template.format(
            entities="\n".join(f"- {e}" for e in entities),
            chunk_text=chunk_text[:2000],  # Cap text length for cost
        )

        raw = await self._llm.generate_json(
            prompt,
            task="entity_graph_extraction",
            temperature=0.0,
            max_tokens=1000,
        )

        # Parse relationships
        relationships: List[ExtractedRelationship] = []
        for rel in raw.get("relationships", []):
            relationships.append(
                ExtractedRelationship(
                    source_name=rel.get("source", ""),
                    target_name=rel.get("target", ""),
                    rel_type=rel.get("type", ""),
                    confidence=float(rel.get("confidence", 0.0)),
                )
            )

        # Parse new entities discovered by LLM
        new_entities = raw.get("new_entities", [])

        return ExtractionResult(
            chunk_id=chunk_id,
            relationships=relationships,
            new_entities=new_entities,
        )

    async def _extract_from_batch(
        self, records: List[Any]
    ) -> List[ExtractionResult]:
        """Extract relationships from multiple chunks in a single LLM call.

        Groups chunk texts and entities into one prompt, then parses
        per-chunk results from the batched response.
        """
        # Build batch prompt
        sections = []
        for i, record in enumerate(records):
            entities = getattr(record, "entities", []) or []
            chunk_text = getattr(record, "display_text", "")
            sections.append(
                f"--- CHUNK {i + 1} ---\n"
                f"Entities: {', '.join(entities)}\n"
                f"Text: {chunk_text[:1500]}\n"
            )

        batch_prompt = (
            "You are an expert at extracting structured information from insurance "
            "and actuarial documents.\n\n"
            "For EACH chunk below, identify RELATIONSHIPS between the pre-extracted entities.\n\n"
            "Relationship types: headed_by, part_of, governed_by, defined_as, "
            "reports_to, implements, contains\n\n"
            + "\n".join(sections)
            + '\nReturn JSON with a "chunks" array, one entry per chunk:\n'
            '{\n  "chunks": [\n'
            '    {\n'
            '      "chunk_index": 1,\n'
            '      "relationships": [{"source": "...", "target": "...", "type": "...", "confidence": 0.0}],\n'
            '      "new_entities": [{"name": "...", "type": "...", "confidence": 0.0}]\n'
            '    }\n'
            '  ]\n'
            '}'
        )

        raw = await self._llm.generate_json(
            batch_prompt,
            task="entity_graph_extraction",
            temperature=0.0,
            max_tokens=2000,
        )

        # Parse per-chunk results
        results: List[ExtractionResult] = []
        chunks_data = raw.get("chunks", [])

        for i, record in enumerate(records):
            chunk_id = getattr(record, "chunk_id", "")

            # Find matching chunk data (by index or position)
            chunk_data = {}
            if i < len(chunks_data):
                chunk_data = chunks_data[i]

            relationships: List[ExtractedRelationship] = []
            for rel in chunk_data.get("relationships", []):
                relationships.append(
                    ExtractedRelationship(
                        source_name=rel.get("source", ""),
                        target_name=rel.get("target", ""),
                        rel_type=rel.get("type", ""),
                        confidence=float(rel.get("confidence", 0.0)),
                    )
                )

            results.append(
                ExtractionResult(
                    chunk_id=chunk_id,
                    relationships=relationships,
                    new_entities=chunk_data.get("new_entities", []),
                )
            )

        return results

    # ------------------------------------------------------------------
    # Graph upsert
    # ------------------------------------------------------------------

    def _upsert_to_graph(self, record: Any, result: ExtractionResult) -> None:
        """Upsert extracted entities and relationships into the graph store."""
        from .graph_store import EntityNode

        chunk_id = getattr(record, "chunk_id", "")
        document_id = getattr(record, "document_id", "")
        all_entities = getattr(record, "entities", []) or []
        persons = set(getattr(record, "entity_persons", []) or [])
        orgs = set(getattr(record, "entity_orgs", []) or [])

        # Build entity name → node ID mapping
        name_to_id: Dict[str, str] = {}

        # First, upsert all known entities from the chunk
        for entity_name in all_entities:
            entity_type = "concept"
            if entity_name in persons:
                entity_type = "person"
            elif entity_name in orgs:
                entity_type = "department"

            node = self._graph.upsert_entity(
                EntityNode(
                    canonical_name=entity_name,
                    entity_type=entity_type,
                    source_chunk_ids=[chunk_id],
                    source_document_ids=[document_id] if document_id else [],
                )
            )
            if node.entity_id:
                name_to_id[entity_name] = node.entity_id

        # Upsert new entities discovered by LLM
        for new_ent in result.new_entities:
            ent_name = new_ent.get("name", "")
            ent_type = new_ent.get("type", "concept")
            if not ent_name:
                continue
            node = self._graph.upsert_entity(
                EntityNode(
                    canonical_name=ent_name,
                    entity_type=ent_type,
                    source_chunk_ids=[chunk_id],
                    source_document_ids=[document_id] if document_id else [],
                )
            )
            if node.entity_id:
                name_to_id[ent_name] = node.entity_id

        # Add relationships
        for rel in result.relationships:
            source_id = name_to_id.get(rel.source_name)
            target_id = name_to_id.get(rel.target_name)
            if source_id and target_id:
                self._graph.add_relationship(
                    source_id=source_id,
                    target_id=target_id,
                    rel_type=rel.rel_type,
                    confidence=rel.confidence,
                    source="llm",
                )

    # ------------------------------------------------------------------
    # Content-hash cache
    # ------------------------------------------------------------------

    @staticmethod
    def _content_hash(text: str) -> str:
        """SHA256 of chunk text for deduplication."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _is_cached(self, content_hash: str) -> bool:
        """Check if a chunk's content hash has been processed before."""
        return (self._cache_dir / content_hash).exists()

    def _mark_cached(self, content_hash: str) -> None:
        """Mark a content hash as processed."""
        try:
            (self._cache_dir / content_hash).touch()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    @staticmethod
    def _load_prompt() -> str:
        """Load the extraction prompt template from disk."""
        if _PROMPT_FILE.exists():
            return _PROMPT_FILE.read_text()

        # Fallback inline prompt
        return (
            "Extract relationships between the following entities found in the text.\n\n"
            "Entities:\n{entities}\n\n"
            "Text:\n{chunk_text}\n\n"
            "Return JSON with 'relationships' and 'new_entities' arrays."
        )
