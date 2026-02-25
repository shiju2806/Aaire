"""
Versioned chunk schema and structured content store.

Every chunk stored in Qdrant carries a schema version, enabling:
- Side-by-side comparison of old vs new chunking strategies
- Gradual migration (don't re-ingest everything at once)
- Filtering by schema version during retrieval

The structured store holds original-fidelity content (tables, formulas,
images) that gets passed to the LLM at generation time — not for
retrieval, but for answering.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger()

# Current schema version. Bump when chunking strategy changes.
SCHEMA_VERSION = "2.0"


@dataclass
class ChunkRecord:
    """A fully-formed chunk ready for Qdrant storage.

    This is the canonical schema for all chunks in the system.
    Every chunk — regardless of element type — must conform to this.
    """

    # Identity
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""

    # Content (what gets embedded and displayed)
    embedding_text: str = ""        # Context prefix + content → for embedding
    display_text: str = ""          # Original content → for user display
    context_prefix: str = ""        # The prepended context string

    # Classification
    element_type: str = "text"      # text|table|formula|callout|image|table_proposition
    schema_version: str = SCHEMA_VERSION
    chunking_strategy: str = "contextual"
    embedding_model: str = ""       # Resolved at embedding time

    # Location
    document_title: str = ""
    section: str = ""
    page: int = 1

    # Retrieval hints
    importance: float = 1.0
    jurisdiction: str = "unknown"   # IFRS|US_GAAP|US_STAT|unknown
    product_type: str = "general"   # universal_life|whole_life|term|general

    # Metadata (extensible)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert to a flat dict for Qdrant point payload."""
        payload = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "embedding_text": self.embedding_text,
            "display_text": self.display_text,
            "context_prefix": self.context_prefix,
            "element_type": self.element_type,
            "schema_version": self.schema_version,
            "chunking_strategy": self.chunking_strategy,
            "embedding_model": self.embedding_model,
            "document_title": self.document_title,
            "section": self.section,
            "page": self.page,
            "importance": self.importance,
            "jurisdiction": self.jurisdiction,
            "product_type": self.product_type,
        }
        # Flatten simple metadata into payload (Qdrant supports nested dicts).
        for k, v in self.metadata.items():
            if k not in payload:
                payload[k] = v
        return payload

    @classmethod
    def from_processed_chunk(
        cls,
        chunk: Any,  # ProcessedChunk from processors
        document_id: str = "",
        document_title: str = "",
        embedding_model: str = "",
        jurisdiction: str = "unknown",
        product_type: str = "general",
    ) -> "ChunkRecord":
        """Create a ChunkRecord from a ProcessedChunk."""
        return cls(
            document_id=document_id,
            embedding_text=chunk.embedding_text,
            display_text=chunk.display_text,
            context_prefix=chunk.context_prefix,
            element_type=chunk.element_type,
            embedding_model=embedding_model,
            document_title=document_title,
            section=chunk.parent_section,
            page=chunk.page_number,
            importance=chunk.importance,
            jurisdiction=jurisdiction,
            product_type=product_type,
            metadata=chunk.metadata,
        )


# ---------------------------------------------------------------------------
# Structured Content Store
# ---------------------------------------------------------------------------


class StructuredStore:
    """Store for original-fidelity content (tables, formulas, images).

    Retrieval flow:
    1. Search Qdrant → get chunk IDs for relevant text summaries
    2. Look up chunk IDs here → get original tables/formulas
    3. Pass originals (not summaries) to the LLM for generation

    This is a simple file-based JSON store. Can be swapped for
    PostgreSQL JSONB or S3 without changing the interface.
    """

    def __init__(self, store_dir: str | Path = "data/structured_store") -> None:
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "_index.json"
        self._index: Dict[str, Dict[str, Any]] = self._load_index()

    def store(
        self,
        chunk_id: str,
        content_type: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store structured content for a chunk.

        Args:
            chunk_id: ID of the chunk in Qdrant.
            content_type: Type of content (markdown_table, latex, image_path).
            content: The actual content (string, dict, or path).
            metadata: Optional metadata about the stored content.
        """
        record = {
            "chunk_id": chunk_id,
            "content_type": content_type,
            "content": content,
            "metadata": metadata or {},
        }

        # Write individual record.
        record_path = self._dir / f"{chunk_id}.json"
        record_path.write_text(json.dumps(record, default=str), encoding="utf-8")

        # Update index.
        self._index[chunk_id] = {
            "content_type": content_type,
            "path": str(record_path),
        }
        self._save_index()

    def retrieve(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve structured content by chunk ID.

        Returns None if the chunk has no structured content.
        """
        if chunk_id not in self._index:
            return None

        record_path = Path(self._index[chunk_id]["path"])
        if not record_path.exists():
            logger.warning("Structured store file missing", chunk_id=chunk_id)
            return None

        return json.loads(record_path.read_text(encoding="utf-8"))

    def retrieve_batch(self, chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Retrieve structured content for multiple chunks.

        Returns dict mapping chunk_id → content record (only for IDs found).
        """
        results: Dict[str, Dict[str, Any]] = {}
        for cid in chunk_ids:
            record = self.retrieve(cid)
            if record is not None:
                results[cid] = record
        return results

    def delete(self, chunk_id: str) -> bool:
        """Delete structured content for a chunk."""
        if chunk_id not in self._index:
            return False

        record_path = Path(self._index[chunk_id]["path"])
        if record_path.exists():
            record_path.unlink()

        del self._index[chunk_id]
        self._save_index()
        return True

    def list_by_document(self, document_id: str) -> List[str]:
        """List all chunk IDs associated with a document.

        Requires metadata to contain document_id.
        """
        results = []
        for chunk_id in self._index:
            record = self.retrieve(chunk_id)
            if record and record.get("metadata", {}).get("document_id") == document_id:
                results.append(chunk_id)
        return results

    # -- internals ----------------------------------------------------------

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Corrupted structured store index, starting fresh")
        return {}

    def _save_index(self) -> None:
        self._index_path.write_text(
            json.dumps(self._index, default=str), encoding="utf-8"
        )
