"""
Retrieval Provider abstraction.

Wraps vector store operations (search, index, delete, scroll) behind an
interface so future backends (ColBERT, Pinecone, Weaviate) can be swapped
via config without touching application code.

Current implementation: QdrantProvider (delegates to qdrant_client).

Usage:
    provider = get_retrieval_provider()

    results = provider.search(query_vector, filters={"framework": "IFRS"}, limit=10)
    provider.upsert(point_id, vector, payload)
    provider.delete(point_ids)
    for batch in provider.scroll(filters={}, batch_size=50):
        process(batch)
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple
import structlog

logger = structlog.get_logger()


@dataclass
class SearchResult:
    """Unified search result across all retrieval backends."""
    point_id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None


@dataclass
class CollectionConfig:
    """Configuration for creating a vector collection."""
    name: str
    vector_size: int = 1536
    distance_metric: str = "cosine"
    on_disk: bool = True
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    payload_indexes: List[Dict[str, str]] = field(default_factory=list)


class RetrievalProvider(ABC):
    """Abstract interface for vector store operations."""

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        *,
        collection: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        with_vectors: bool = False,
    ) -> List[SearchResult]:
        """Search for similar vectors."""

    @abstractmethod
    def upsert(
        self,
        point_id: str,
        vector: List[float],
        payload: Dict[str, Any],
        *,
        collection: Optional[str] = None,
    ) -> None:
        """Insert or update a point."""

    @abstractmethod
    def upsert_batch(
        self,
        points: List[Tuple[str, List[float], Dict[str, Any]]],
        *,
        collection: Optional[str] = None,
    ) -> int:
        """Batch upsert. Returns count of upserted points."""

    @abstractmethod
    def delete(
        self,
        point_ids: List[str],
        *,
        collection: Optional[str] = None,
    ) -> int:
        """Delete points by ID. Returns count deleted."""

    @abstractmethod
    def delete_by_filter(
        self,
        filters: Dict[str, Any],
        *,
        collection: Optional[str] = None,
    ) -> int:
        """Delete points matching filter. Returns count deleted."""

    @abstractmethod
    def scroll(
        self,
        *,
        collection: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        batch_size: int = 50,
        with_vectors: bool = False,
    ) -> Generator[List[SearchResult], None, None]:
        """Iterate over all points matching filters, yielding batches."""

    @abstractmethod
    def count(
        self,
        *,
        collection: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Count points matching optional filter."""

    @abstractmethod
    def create_collection(self, config: CollectionConfig) -> None:
        """Create a collection if it doesn't exist."""

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """Delete an entire collection."""

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the backend is reachable."""

    @abstractmethod
    def get_client(self) -> Any:
        """Return the underlying client for LlamaIndex integration.

        This is a temporary escape hatch. Code should migrate to using
        provider methods directly over time. LlamaIndex's VectorStoreIndex
        still needs raw QdrantClient access.
        """


class QdrantProvider(RetrievalProvider):
    """Qdrant-backed retrieval provider."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_collection: str = "documents",
    ):
        self._url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self._api_key = api_key or os.getenv("QDRANT_API_KEY")
        self._default_collection = default_collection
        self._client = None

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=self._url, api_key=self._api_key)
            logger.info("Qdrant client initialized", url=self._url)
        return self._client

    def _resolve_collection(self, collection: Optional[str]) -> str:
        return collection or self._default_collection

    def _build_filter(self, filters: Optional[Dict[str, Any]]):
        """Convert a simple dict filter to Qdrant Filter object."""
        if not filters:
            return None
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        must = []
        for key, value in filters.items():
            if isinstance(value, list):
                from qdrant_client.models import MatchAny
                must.append(FieldCondition(key=key, match=MatchAny(any=value)))
            else:
                must.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=must) if must else None

    def get_client(self) -> Any:
        return self._get_client()

    def search(
        self,
        query_vector: List[float],
        *,
        collection: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        with_vectors: bool = False,
    ) -> List[SearchResult]:
        client = self._get_client()
        col = self._resolve_collection(collection)
        qdrant_filter = self._build_filter(filters)

        results = client.search(
            collection_name=col,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=limit,
            score_threshold=score_threshold if score_threshold > 0 else None,
            with_payload=True,
            with_vectors=with_vectors,
        )
        return [
            SearchResult(
                point_id=str(r.id),
                score=r.score,
                payload=r.payload or {},
                vector=r.vector if with_vectors else None,
            )
            for r in results
        ]

    def upsert(
        self,
        point_id: str,
        vector: List[float],
        payload: Dict[str, Any],
        *,
        collection: Optional[str] = None,
    ) -> None:
        from qdrant_client.models import PointStruct
        client = self._get_client()
        col = self._resolve_collection(collection)
        client.upsert(
            collection_name=col,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )

    def upsert_batch(
        self,
        points: List[Tuple[str, List[float], Dict[str, Any]]],
        *,
        collection: Optional[str] = None,
        max_retries: int = 2,
    ) -> int:
        import time as _time
        from qdrant_client.models import PointStruct
        client = self._get_client()
        col = self._resolve_collection(collection)
        structs = [
            PointStruct(id=pid, vector=vec, payload=pay) for pid, vec, pay in points
        ]
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                client.upsert(collection_name=col, points=structs)
                return len(structs)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    delay = 1.0 * (2 ** attempt)
                    logger.warning("Qdrant upsert failed, retrying",
                                   attempt=attempt + 1, delay=delay,
                                   error=str(e)[:120])
                    _time.sleep(delay)
        raise last_error

    def create_entity_indexes(self, collection: Optional[str] = None) -> None:
        """Create Qdrant payload indexes for entity fields.

        Idempotent — silently succeeds if indexes already exist.
        """
        from qdrant_client.models import PayloadSchemaType

        client = self._get_client()
        col = self._resolve_collection(collection)

        for field_name in ["entities", "entity_orgs", "entity_persons",
                          "doc_content_hash", "content_hash"]:
            try:
                client.create_payload_index(
                    collection_name=col,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                logger.info("Created entity payload index", field=field_name)
            except Exception as e:
                logger.debug(
                    "Entity index creation skipped (may already exist)",
                    field=field_name,
                    error=str(e),
                )

    def delete(
        self,
        point_ids: List[str],
        *,
        collection: Optional[str] = None,
    ) -> int:
        if not point_ids:
            return 0
        client = self._get_client()
        col = self._resolve_collection(collection)
        client.delete(collection_name=col, points_selector=point_ids)
        return len(point_ids)

    def delete_by_filter(
        self,
        filters: Dict[str, Any],
        *,
        collection: Optional[str] = None,
    ) -> int:
        client = self._get_client()
        col = self._resolve_collection(collection)
        qdrant_filter = self._build_filter(filters)
        if not qdrant_filter:
            return 0
        # Scroll to find matching points, then delete
        all_ids = []
        offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=col,
                scroll_filter=qdrant_filter,
                limit=100,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            all_ids.extend([str(p.id) for p in points])
            if next_offset is None or not points:
                break
            offset = next_offset
        if all_ids:
            client.delete(collection_name=col, points_selector=all_ids)
        return len(all_ids)

    def scroll(
        self,
        *,
        collection: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        batch_size: int = 50,
        with_vectors: bool = False,
    ) -> Generator[List[SearchResult], None, None]:
        client = self._get_client()
        col = self._resolve_collection(collection)
        qdrant_filter = self._build_filter(filters)
        offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=col,
                scroll_filter=qdrant_filter,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )
            if not points:
                break
            yield [
                SearchResult(
                    point_id=str(p.id),
                    score=0.0,
                    payload=p.payload or {},
                    vector=p.vector if with_vectors else None,
                )
                for p in points
            ]
            if next_offset is None:
                break
            offset = next_offset

    def count(
        self,
        *,
        collection: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        client = self._get_client()
        col = self._resolve_collection(collection)
        if filters:
            qdrant_filter = self._build_filter(filters)
            result = client.count(collection_name=col, count_filter=qdrant_filter)
        else:
            result = client.count(collection_name=col)
        return result.count

    def create_collection(self, config: CollectionConfig) -> None:
        from qdrant_client.models import Distance, VectorParams, HnswConfigDiff
        client = self._get_client()

        # Check if already exists
        try:
            collections = client.get_collections().collections
            if any(c.name == config.name for c in collections):
                logger.info("Collection already exists", collection=config.name)
                return
        except Exception:
            pass

        distance_map = {
            "cosine": Distance.COSINE,
            "euclid": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        client.create_collection(
            collection_name=config.name,
            vectors_config=VectorParams(
                size=config.vector_size,
                distance=distance_map.get(config.distance_metric, Distance.COSINE),
                on_disk=config.on_disk,
            ),
            hnsw_config=HnswConfigDiff(
                m=config.hnsw_m,
                ef_construct=config.hnsw_ef_construct,
            ),
        )

        # Create payload indexes
        for idx in config.payload_indexes:
            try:
                from qdrant_client.models import PayloadSchemaType
                schema_map = {
                    "keyword": PayloadSchemaType.KEYWORD,
                    "integer": PayloadSchemaType.INTEGER,
                    "float": PayloadSchemaType.FLOAT,
                    "text": PayloadSchemaType.TEXT,
                }
                client.create_payload_index(
                    collection_name=config.name,
                    field_name=idx["field"],
                    field_schema=schema_map.get(idx.get("type", "keyword"), PayloadSchemaType.KEYWORD),
                )
            except Exception as e:
                logger.warning("Failed to create payload index", field=idx["field"], error=str(e))

        logger.info("Collection created", collection=config.name, vector_size=config.vector_size)

    def delete_collection(self, name: str) -> None:
        client = self._get_client()
        client.delete_collection(name)
        logger.info("Collection deleted", collection=name)

    def health_check(self) -> bool:
        try:
            client = self._get_client()
            client.get_collections()
            return True
        except Exception as e:
            logger.error("Retrieval provider health check failed", error=str(e))
            return False


# --- Singleton ---

_provider_instance: Optional[RetrievalProvider] = None


def get_retrieval_provider(
    default_collection: str = "aaire-documents",
) -> RetrievalProvider:
    """Get or create the singleton retrieval provider."""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = QdrantProvider(default_collection=default_collection)
        logger.info("Retrieval provider initialized", backend="qdrant")
    return _provider_instance


def reset_retrieval_provider():
    """Reset the singleton (for testing)."""
    global _provider_instance
    _provider_instance = None
