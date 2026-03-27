"""
ES-backed entity-relationship graph store.

Each entity is an ES document in the ``aaire-entities`` index.
Relationships are stored as nested objects on the source entity.
No Neo4j required — at our scale (<100K entities), ES nested queries
are fast enough (<50ms) and give us fuzzy entity resolution for free.

Usage:
    from src.knowledge_graph.graph_store import GraphStore

    graph = GraphStore(es_client)
    node = graph.upsert_entity(EntityNode(canonical_name="FP&A", ...))
    graph.add_relationship(node.entity_id, target_id, "part_of", 0.9)
    resolved = graph.resolve_entity("Financial Planning & Analysis")
    chunks = graph.get_connected_chunks(resolved.entity_id)
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import structlog

from ..providers.config_loader import get_config, get_nested

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EntityRelationship:
    """A directed relationship between two entity nodes."""

    target_id: str
    rel_type: str  # headed_by | part_of | governed_by | defined_as | reports_to | implements | contains
    confidence: float = 0.0
    source: str = ""  # "llm" | "regex" | "manual"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "type": self.rel_type,
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntityRelationship":
        return cls(
            target_id=d["target_id"],
            rel_type=d.get("type", d.get("rel_type", "")),
            confidence=d.get("confidence", 0.0),
            source=d.get("source", ""),
        )


@dataclass
class EntityNode:
    """A node in the knowledge graph representing a resolved entity."""

    entity_id: str = ""
    canonical_name: str = ""
    entity_type: str = ""  # person | department | regulation | concept
    aliases: List[str] = field(default_factory=list)
    embedding: List[float] = field(default_factory=list)  # Name embedding for Stage 3 resolution
    source_chunk_ids: List[str] = field(default_factory=list)
    source_document_ids: List[str] = field(default_factory=list)
    relationships: List[EntityRelationship] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_es_doc(self) -> Dict[str, Any]:
        """Serialize to ES document body."""
        doc: Dict[str, Any] = {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "aliases": self.aliases,
            "source_chunk_ids": self.source_chunk_ids,
            "source_document_ids": self.source_document_ids,
            "relationships": [r.to_dict() for r in self.relationships],
            "metadata": self.metadata,
        }
        if self.embedding:
            doc["embedding"] = self.embedding
        return doc

    @classmethod
    def from_es_doc(cls, doc: Dict[str, Any]) -> "EntityNode":
        """Deserialize from ES hit ``_source``."""
        source = doc.get("_source", doc)
        return cls(
            entity_id=source.get("entity_id", ""),
            canonical_name=source.get("canonical_name", ""),
            entity_type=source.get("entity_type", ""),
            aliases=source.get("aliases", []),
            embedding=source.get("embedding", []),
            source_chunk_ids=source.get("source_chunk_ids", []),
            source_document_ids=source.get("source_document_ids", []),
            relationships=[
                EntityRelationship.from_dict(r)
                for r in source.get("relationships", [])
            ],
            metadata=source.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Index settings
# ---------------------------------------------------------------------------


def _get_entity_index_settings() -> Dict[str, Any]:
    """ES index mappings for the entity graph."""
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "entity_id": {"type": "keyword"},
                "canonical_name": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "entity_type": {"type": "keyword"},
                "aliases": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": 384,  # all-MiniLM-L6-v2 dimension
                    "index": True,
                    "similarity": "cosine",
                },
                "source_chunk_ids": {"type": "keyword"},
                "source_document_ids": {"type": "keyword"},
                "relationships": {
                    "type": "nested",
                    "properties": {
                        "target_id": {"type": "keyword"},
                        "type": {"type": "keyword"},
                        "confidence": {"type": "float"},
                        "source": {"type": "keyword"},
                    },
                },
                "metadata": {"type": "object", "enabled": False},
            }
        },
    }


# ---------------------------------------------------------------------------
# GraphStore
# ---------------------------------------------------------------------------


class GraphStore:
    """ES-backed entity-relationship graph.

    Provides:
    - Entity CRUD with alias-based deduplication
    - 3-stage entity resolution (exact → fuzzy → embedding)
    - Graph traversal for connected chunk discovery
    - PII handling (plaintext / HMAC / exclude)
    """

    def __init__(
        self,
        es_client: Any,
        index_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        embedding_provider: Optional[Any] = None,
    ) -> None:
        self._es = es_client
        self._config = config or get_config("knowledge_graph")
        self._index = index_name or get_nested(
            self._config, "graph_store", "index_name", default="aaire-entities"
        )
        self._embedding_provider = embedding_provider  # For Stage 3 entity resolution
        self._ensure_index()

        # Batch write buffer — avoids per-entity refresh=wait_for overhead.
        # Writes use refresh=False; buffer tracks pending writes and flushes
        # (ES index refresh) every _buffer_size operations.
        self._write_count = 0
        self._buffer_size = get_nested(
            self._config, "graph_store", "bulk_buffer_size", default=50
        )

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        """Create the entity index if it doesn't already exist."""
        try:
            if not self._es.indices.exists(index=self._index):
                self._es.indices.create(
                    index=self._index,
                    body=_get_entity_index_settings(),
                )
                logger.info("Created entity graph index", index=self._index)
            else:
                logger.debug("Entity graph index already exists", index=self._index)
        except Exception as e:
            logger.error("Failed to create entity graph index", error=str(e))

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def upsert_entity(self, entity: EntityNode) -> EntityNode:
        """Insert or merge an entity. If an alias matches an existing entity, merge.

        Returns the (possibly merged) entity node.
        """
        # Apply PII handling before storing
        entity = self._apply_pii_policy(entity)

        # Try to resolve first — may already exist under a different name
        existing = self.resolve_entity(entity.canonical_name, entity.entity_type)
        if existing:
            return self._merge_entities(existing, entity)

        # New entity — assign ID if missing
        if not entity.entity_id:
            entity.entity_id = f"e_{uuid.uuid4().hex[:12]}"

        if not entity.metadata.get("first_seen"):
            entity.metadata["first_seen"] = time.strftime("%Y-%m-%d")
        entity.metadata.setdefault("extraction_source", "llm")
        entity.metadata.setdefault("review_status", "auto_approved")

        # Compute embedding for Stage 3 resolution if provider available
        if not entity.embedding and self._embedding_provider:
            vec = self._embed_entity_name(entity.canonical_name)
            if vec:
                entity.embedding = vec

        self._es.index(
            index=self._index,
            id=entity.entity_id,
            body=entity.to_es_doc(),
            refresh=False,
        )
        self._maybe_flush()
        logger.info(
            "Entity created",
            entity_id=entity.entity_id,
            name=entity.canonical_name,
            type=entity.entity_type,
        )
        return entity

    def _merge_entities(self, existing: EntityNode, incoming: EntityNode) -> EntityNode:
        """Merge incoming entity data into an existing node."""
        # Merge aliases (deduplicated)
        alias_set = set(existing.aliases)
        alias_set.add(incoming.canonical_name)
        for a in incoming.aliases:
            alias_set.add(a)
        # Remove canonical from aliases if present
        alias_set.discard(existing.canonical_name)
        existing.aliases = sorted(alias_set)

        # Merge chunk IDs
        chunk_set = set(existing.source_chunk_ids)
        chunk_set.update(incoming.source_chunk_ids)
        existing.source_chunk_ids = sorted(chunk_set)

        # Merge document IDs
        doc_set = set(existing.source_document_ids)
        doc_set.update(incoming.source_document_ids)
        existing.source_document_ids = sorted(doc_set)

        # Merge relationships (deduplicate by target_id + type)
        rel_keys = {(r.target_id, r.rel_type) for r in existing.relationships}
        for r in incoming.relationships:
            if (r.target_id, r.rel_type) not in rel_keys:
                existing.relationships.append(r)
                rel_keys.add((r.target_id, r.rel_type))

        # Update in ES
        self._es.index(
            index=self._index,
            id=existing.entity_id,
            body=existing.to_es_doc(),
            refresh=False,
        )
        self._maybe_flush()
        logger.info(
            "Entity merged",
            entity_id=existing.entity_id,
            name=existing.canonical_name,
            alias_count=len(existing.aliases),
        )
        return existing

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        confidence: float = 0.0,
        source: str = "llm",
    ) -> None:
        """Add a relationship between two existing entities."""
        min_conf = get_nested(
            self._config, "graph_store", "min_relationship_confidence", default=0.60
        )
        if confidence < min_conf:
            logger.debug(
                "Relationship below confidence threshold, skipping",
                source_id=source_id,
                target_id=target_id,
                confidence=confidence,
                threshold=min_conf,
            )
            return

        try:
            # Fetch current entity
            doc = self._es.get(index=self._index, id=source_id)
            node = EntityNode.from_es_doc(doc)

            # Check for duplicate
            for r in node.relationships:
                if r.target_id == target_id and r.rel_type == rel_type:
                    # Update confidence if higher
                    if confidence > r.confidence:
                        r.confidence = confidence
                        self._es.index(
                            index=self._index,
                            id=source_id,
                            body=node.to_es_doc(),
                            refresh=False,
                        )
                        self._maybe_flush()
                    return

            # Add new relationship
            node.relationships.append(
                EntityRelationship(
                    target_id=target_id,
                    rel_type=rel_type,
                    confidence=confidence,
                    source=source,
                )
            )
            self._es.index(
                index=self._index,
                id=source_id,
                body=node.to_es_doc(),
                refresh=False,
            )
            self._maybe_flush()
            logger.debug(
                "Relationship added",
                source_id=source_id,
                target_id=target_id,
                rel_type=rel_type,
            )
        except Exception as e:
            logger.warning("Failed to add relationship", error=str(e))

    def add_chunk_to_entity(self, entity_id: str, chunk_id: str) -> None:
        """Associate a chunk ID with an entity."""
        try:
            doc = self._es.get(index=self._index, id=entity_id)
            node = EntityNode.from_es_doc(doc)
            if chunk_id not in node.source_chunk_ids:
                node.source_chunk_ids.append(chunk_id)
                self._es.index(
                    index=self._index,
                    id=entity_id,
                    body=node.to_es_doc(),
                    refresh=False,
                )
                self._maybe_flush()
        except Exception as e:
            logger.warning("Failed to add chunk to entity", error=str(e))

    # ------------------------------------------------------------------
    # Batch flush
    # ------------------------------------------------------------------

    def _maybe_flush(self) -> None:
        """Increment write counter and refresh ES index when buffer is full."""
        self._write_count += 1
        if self._write_count >= self._buffer_size:
            self._flush()

    def _flush(self) -> None:
        """Force ES index refresh to make buffered writes searchable."""
        if self._write_count > 0:
            try:
                self._es.indices.refresh(index=self._index)
                logger.debug("ES index flushed", pending_writes=self._write_count)
                self._write_count = 0
            except Exception as e:
                logger.warning("ES flush failed", error=str(e))

    def flush(self) -> None:
        """Public flush — call after ingestion completes to ensure all
        writes are searchable."""
        self._flush()

    # ------------------------------------------------------------------
    # Entity Resolution — 3-stage
    # ------------------------------------------------------------------

    def resolve_entity(
        self, name: str, entity_type: Optional[str] = None
    ) -> Optional[EntityNode]:
        """Find an existing entity matching this name.

        Resolution order:
        1. Exact alias match (ES keyword query)
        2. Fuzzy string match (Jaro-Winkler on canonical_name + aliases)
        3. Embedding similarity (cosine on entity name embeddings)

        Returns None if no match → caller creates a new entity.
        """
        if not name or not name.strip():
            return None

        name_clean = name.strip()

        # Stage 1: Exact alias/canonical match
        node = self._exact_match(name_clean, entity_type)
        if node:
            return node

        # Stage 2: Fuzzy match
        fuzzy_threshold = get_nested(
            self._config, "entity_resolution", "fuzzy_match_threshold", default=0.85
        )
        node = self._fuzzy_match(name_clean, entity_type, fuzzy_threshold)
        if node:
            return node

        # Stage 3: Embedding similarity
        if self._embedding_provider:
            embed_threshold = get_nested(
                self._config, "entity_resolution", "embedding_threshold", default=0.80
            )
            node = self._embedding_match(name_clean, entity_type, embed_threshold)
            if node:
                return node

        return None

    def resolve_entities_batch(
        self, names: List[str], entity_type: Optional[str] = None
    ) -> Dict[str, Optional["EntityNode"]]:
        """Batch-resolve multiple entity names using ES msearch.

        Runs one msearch for exact matches, then a second msearch for fuzzy
        matches on any names that weren't resolved in the first pass.
        Skips embedding resolution (expensive, rarely needed at query time).

        Returns:
            Dict mapping each input name to its resolved EntityNode or None.
        """
        results: Dict[str, Optional[EntityNode]] = {n: None for n in names}
        clean_names = [(n, n.strip()) for n in names if n and n.strip()]

        if not clean_names:
            return results

        # --- Pass 1: Exact keyword match via msearch ---
        body_lines = []
        for _, name_clean in clean_names:
            body_lines.append({"index": self._index})
            query: Dict[str, Any] = {
                "bool": {
                    "should": [
                        {"term": {"canonical_name.keyword": name_clean}},
                        {"term": {"aliases.keyword": name_clean}},
                    ],
                    "minimum_should_match": 1,
                }
            }
            if entity_type:
                query["bool"]["filter"] = [{"term": {"entity_type": entity_type}}]
            body_lines.append({"query": query, "size": 1})

        try:
            resp = self._es.msearch(body=body_lines)
            for i, response in enumerate(resp.get("responses", [])):
                hits = response.get("hits", {}).get("hits", [])
                if hits:
                    original_name = clean_names[i][0]
                    results[original_name] = EntityNode.from_es_doc(hits[0])
        except Exception as e:
            logger.warning("Batch exact match failed", error=str(e))

        # --- Pass 2: Fuzzy match for unresolved names ---
        unresolved = [(orig, clean) for orig, clean in clean_names if results.get(orig) is None]
        if not unresolved:
            return results

        fuzzy_threshold = get_nested(
            self._config, "entity_resolution", "fuzzy_match_threshold", default=0.85
        )

        body_lines = []
        for _, name_clean in unresolved:
            body_lines.append({"index": self._index})
            query = {
                "bool": {
                    "should": [
                        {"match": {"canonical_name": {"query": name_clean, "fuzziness": "AUTO", "boost": 2.0}}},
                        {"match": {"aliases": {"query": name_clean, "fuzziness": "AUTO", "boost": 1.5}}},
                    ],
                    "minimum_should_match": 1,
                }
            }
            if entity_type:
                query["bool"]["filter"] = [{"term": {"entity_type": entity_type}}]
            body_lines.append({"query": query, "size": 3})

        try:
            resp = self._es.msearch(body=body_lines)
            for i, response in enumerate(resp.get("responses", [])):
                hits = response.get("hits", {}).get("hits", [])
                if not hits:
                    continue
                candidate = EntityNode.from_es_doc(hits[0])
                original_name, name_clean = unresolved[i]

                similarity = self._jaro_winkler(name_clean.lower(), candidate.canonical_name.lower())
                for alias in candidate.aliases:
                    similarity = max(similarity, self._jaro_winkler(name_clean.lower(), alias.lower()))

                if similarity >= fuzzy_threshold:
                    results[original_name] = candidate
        except Exception as e:
            logger.warning("Batch fuzzy match failed", error=str(e))

        resolved_count = sum(1 for v in results.values() if v is not None)
        logger.info("Batch entity resolution complete", total=len(names), resolved=resolved_count)
        return results

    def _exact_match(
        self, name: str, entity_type: Optional[str] = None
    ) -> Optional[EntityNode]:
        """Stage 1: Exact keyword match on canonical_name or aliases."""
        query: Dict[str, Any] = {
            "bool": {
                "should": [
                    {"term": {"canonical_name.keyword": name}},
                    {"term": {"aliases.keyword": name}},
                ],
                "minimum_should_match": 1,
            }
        }
        if entity_type:
            query["bool"]["filter"] = [{"term": {"entity_type": entity_type}}]

        try:
            resp = self._es.search(
                index=self._index, body={"query": query, "size": 1}
            )
            hits = resp.get("hits", {}).get("hits", [])
            if hits:
                return EntityNode.from_es_doc(hits[0])
        except Exception as e:
            logger.debug("Exact entity match failed", name=name, error=str(e))

        return None

    def _fuzzy_match(
        self,
        name: str,
        entity_type: Optional[str] = None,
        threshold: float = 0.85,
    ) -> Optional[EntityNode]:
        """Stage 2: Fuzzy text match with fuzziness on canonical_name and aliases."""
        query: Dict[str, Any] = {
            "bool": {
                "should": [
                    {
                        "match": {
                            "canonical_name": {
                                "query": name,
                                "fuzziness": "AUTO",
                                "boost": 2.0,
                            }
                        }
                    },
                    {
                        "match": {
                            "aliases": {
                                "query": name,
                                "fuzziness": "AUTO",
                                "boost": 1.5,
                            }
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        }
        if entity_type:
            query["bool"]["filter"] = [{"term": {"entity_type": entity_type}}]

        try:
            resp = self._es.search(
                index=self._index,
                body={"query": query, "size": 3},
            )
            hits = resp.get("hits", {}).get("hits", [])
            if not hits:
                return None

            # Check best hit score — ES scores are not normalized, so we use
            # Jaro-Winkler similarity between the query name and candidate names
            # as a secondary check.
            best = hits[0]
            candidate = EntityNode.from_es_doc(best)
            similarity = self._jaro_winkler(
                name.lower(), candidate.canonical_name.lower()
            )

            # Also check aliases
            for alias in candidate.aliases:
                alias_sim = self._jaro_winkler(name.lower(), alias.lower())
                similarity = max(similarity, alias_sim)

            if similarity >= threshold:
                logger.debug(
                    "Fuzzy entity match",
                    query=name,
                    matched=candidate.canonical_name,
                    similarity=round(similarity, 3),
                )
                return candidate

            # Check if in review zone — flag for human review
            review_low = get_nested(
                self._config, "entity_resolution", "review_zone", "low", default=0.65
            )
            review_high = get_nested(
                self._config, "entity_resolution", "review_zone", "high", default=0.85
            )
            if review_low <= similarity < review_high:
                self._add_to_review_queue(name, candidate.canonical_name, similarity)

        except Exception as e:
            logger.debug("Fuzzy entity match failed", name=name, error=str(e))

        return None

    def _embedding_match(
        self,
        name: str,
        entity_type: Optional[str] = None,
        threshold: float = 0.80,
    ) -> Optional[EntityNode]:
        """Stage 3: Embedding cosine similarity on entity name embeddings.

        Embeds the query name and searches the ES dense_vector field for
        the closest entity.  Only runs if ``embedding_provider`` is set.
        """
        try:
            # Embed the query name
            query_vec = self._embed_entity_name(name)
            if not query_vec:
                return None

            # ES knn query on the embedding field
            knn: Dict[str, Any] = {
                "field": "embedding",
                "query_vector": query_vec,
                "k": 3,
                "num_candidates": 50,
            }

            body: Dict[str, Any] = {"knn": knn, "size": 3}

            # Optional entity_type filter
            if entity_type:
                body["knn"]["filter"] = {"term": {"entity_type": entity_type}}

            resp = self._es.search(index=self._index, body=body)
            hits = resp.get("hits", {}).get("hits", [])
            if not hits:
                return None

            best = hits[0]
            score = best.get("_score", 0.0)

            # ES cosine similarity returns (1 + cosine) / 2, so 0.9 → cosine 0.8
            # For dense_vector with similarity=cosine, scores are already cosine.
            if score >= threshold:
                candidate = EntityNode.from_es_doc(best)
                logger.debug(
                    "Embedding entity match",
                    query=name,
                    matched=candidate.canonical_name,
                    cosine=round(score, 3),
                )
                return candidate

            # Check review zone
            review_low = get_nested(
                self._config, "entity_resolution", "review_zone", "low", default=0.65
            )
            if review_low <= score < threshold:
                candidate = EntityNode.from_es_doc(best)
                self._add_to_review_queue(name, candidate.canonical_name, score)

        except Exception as e:
            logger.debug("Embedding entity match failed", name=name, error=str(e))

        return None

    def _embed_entity_name(self, name: str) -> Optional[List[float]]:
        """Embed an entity name using the configured embedding provider.

        Uses the bi-encoder from SemanticSimilarityService (all-MiniLM-L6-v2,
        384 dims) rather than the expensive OpenAI embedding model.
        """
        if not self._embedding_provider:
            return None

        try:
            # SemanticSimilarityService exposes get_query_embedding()
            if hasattr(self._embedding_provider, "get_query_embedding"):
                vec = self._embedding_provider.get_query_embedding(name)
                if hasattr(vec, "tolist"):
                    return vec.tolist()
                return list(vec)

            # Generic fallback: encode() method
            if hasattr(self._embedding_provider, "encode"):
                vec = self._embedding_provider.encode([name])[0]
                if hasattr(vec, "tolist"):
                    return vec.tolist()
                return list(vec)
        except Exception as e:
            logger.debug("Entity name embedding failed", name=name, error=str(e))

        return None

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def get_connected_chunks(
        self, entity_id: str, max_hops: int = 2
    ) -> List[str]:
        """Traverse graph: entity → relationships → connected entities → chunk IDs.

        BFS traversal up to max_hops.
        """
        max_hops = min(
            max_hops,
            get_nested(self._config, "graph_store", "max_traversal_hops", default=2),
        )

        visited: Set[str] = set()
        chunk_ids: Set[str] = set()
        frontier: Set[str] = {entity_id}

        for _hop in range(max_hops):
            if not frontier:
                break

            next_frontier: Set[str] = set()
            for eid in frontier:
                if eid in visited:
                    continue
                visited.add(eid)

                try:
                    doc = self._es.get(index=self._index, id=eid)
                    node = EntityNode.from_es_doc(doc)
                    chunk_ids.update(node.source_chunk_ids)

                    for rel in node.relationships:
                        if rel.target_id not in visited:
                            next_frontier.add(rel.target_id)
                except Exception:
                    continue  # Entity may have been deleted

            frontier = next_frontier

        return sorted(chunk_ids)

    def get_entity_context(self, entity_ids: List[str]) -> str:
        """Build a human-readable context string from entity nodes and relationships.

        Injected into the LLM prompt to give structural knowledge.
        """
        lines: List[str] = []
        for eid in entity_ids:
            try:
                doc = self._es.get(index=self._index, id=eid)
                node = EntityNode.from_es_doc(doc)

                aliases_str = ""
                if node.aliases:
                    aliases_str = f" (also known as: {', '.join(node.aliases)})"

                line = f"- {node.canonical_name}{aliases_str} [{node.entity_type}]"
                for rel in node.relationships:
                    # Try to look up target name
                    target_name = rel.target_id
                    try:
                        target_doc = self._es.get(index=self._index, id=rel.target_id)
                        target_node = EntityNode.from_es_doc(target_doc)
                        target_name = target_node.canonical_name
                    except Exception:
                        pass
                    line += f"\n    {rel.rel_type} → {target_name}"

                lines.append(line)
            except Exception:
                continue

        if not lines:
            return ""

        return "Entity context:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Bulk operations (for ingestion)
    # ------------------------------------------------------------------

    def get_all_entities(self, entity_type: Optional[str] = None) -> List[EntityNode]:
        """Return all entities (optionally filtered by type). For small graphs."""
        query: Dict[str, Any] = {"match_all": {}}
        if entity_type:
            query = {"term": {"entity_type": entity_type}}

        try:
            resp = self._es.search(
                index=self._index,
                body={"query": query, "size": 10000},
            )
            return [EntityNode.from_es_doc(h) for h in resp["hits"]["hits"]]
        except Exception as e:
            logger.warning("Failed to list entities", error=str(e))
            return []

    # ------------------------------------------------------------------
    # PII handling
    # ------------------------------------------------------------------

    def _apply_pii_policy(self, entity: EntityNode) -> EntityNode:
        """Apply privacy policy to entity before storage."""
        pii_mode = get_nested(self._config, "privacy", "pii_mode", default="plaintext")
        pii_types = get_nested(
            self._config, "privacy", "pii_entity_types", default=["person"]
        )

        if entity.entity_type not in pii_types:
            return entity

        if pii_mode == "exclude":
            # Don't store PII entities in graph at all — return empty node
            # Caller should check entity_id == "" and skip
            logger.debug("PII entity excluded from graph", type=entity.entity_type)
            entity.entity_id = ""
            return entity

        if pii_mode == "hmac":
            key_env = get_nested(
                self._config, "privacy", "hmac_key_env", default="ENTITY_PII_KEY"
            )
            secret = os.environ.get(key_env, "default-dev-key")
            entity.canonical_name = self._hmac_name(entity.canonical_name, secret)
            entity.aliases = [self._hmac_name(a, secret) for a in entity.aliases]

        return entity

    @staticmethod
    def _hmac_name(name: str, secret: str) -> str:
        """HMAC-SHA256 a name for PII protection."""
        return hmac.new(
            secret.encode(), name.encode(), hashlib.sha256
        ).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Review queue
    # ------------------------------------------------------------------

    def _add_to_review_queue(
        self, entity_a: str, entity_b: str, similarity: float
    ) -> None:
        """Write low-confidence resolution to JSONL review queue."""
        queue_cfg = get_nested(
            self._config, "entity_resolution", "review_queue", default={}
        )
        if not queue_cfg.get("enabled", False):
            return

        import json
        from pathlib import Path

        queue_path = Path(queue_cfg.get("storage", "data/entity_review_queue.jsonl"))
        queue_path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "similarity": round(similarity, 4),
            "action": "pending",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        try:
            with open(queue_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            logger.info(
                "Entity resolution flagged for review",
                entity_a=entity_a,
                entity_b=entity_b,
                similarity=round(similarity, 3),
            )
        except Exception as e:
            logger.warning("Failed to write review queue", error=str(e))

    # ------------------------------------------------------------------
    # String similarity
    # ------------------------------------------------------------------

    @staticmethod
    def _jaro_winkler(s1: str, s2: str) -> float:
        """Jaro-Winkler string similarity (0.0–1.0)."""
        if s1 == s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        max_dist = max(len(s1), len(s2)) // 2 - 1
        if max_dist < 0:
            max_dist = 0

        s1_matches = [False] * len(s1)
        s2_matches = [False] * len(s2)

        matches = 0
        transpositions = 0

        for i in range(len(s1)):
            start = max(0, i - max_dist)
            end = min(i + max_dist + 1, len(s2))
            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        k = 0
        for i in range(len(s1)):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

        jaro = (
            matches / len(s1) + matches / len(s2) + (matches - transpositions / 2) / matches
        ) / 3.0

        # Winkler prefix bonus
        prefix = 0
        for i in range(min(4, len(s1), len(s2))):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break

        return jaro + prefix * 0.1 * (1 - jaro)
