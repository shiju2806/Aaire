"""
Elasticsearch Search Engine

Drop-in replacement for BM25SearchEngine using Elasticsearch for persistent,
scalable keyword search. Implements the same interface (add_documents, search,
clear, get_stats) so the rest of the pipeline is unchanged.

Advantages over in-memory BM25:
- Persistent: survives restarts without re-indexing from Qdrant
- Scalable: handles millions of documents without RAM pressure
- Incremental: no full rebuild on each document addition
- Domain-aware: custom analyzer with actuarial/insurance synonyms
"""

import os
import uuid
import structlog
from typing import List, Dict, Any, Optional

from .bm25_engine import SearchResult

from ...providers.config_loader import get_config, get_nested

logger = structlog.get_logger()

# Elasticsearch connection defaults (overridden by config/infrastructure.yaml)
_es_config = get_config("infrastructure")
_DEFAULT_HOSTS = get_nested(_es_config, "elasticsearch", "hosts", default=["http://localhost:9200"])
_DEFAULT_INDEX = get_nested(_es_config, "elasticsearch", "index_name", default="aaire-documents")
_REQUEST_TIMEOUT = get_nested(_es_config, "elasticsearch", "request_timeout", default=30)


def _get_index_settings() -> Dict[str, Any]:
    """Index settings with domain-aware analyzer for actuarial/insurance content."""
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "aaire_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "aaire_synonyms", "english_stemmer"],
                    }
                },
                "filter": {
                    "english_stemmer": {
                        "type": "stemmer",
                        "language": "english",
                    },
                    "aaire_synonyms": {
                        "type": "synonym",
                        "synonyms": [
                            "IFRS,international financial reporting standards",
                            "GAAP,generally accepted accounting principles",
                            "VM,valuation manual",
                            "CSO,commissioners standard ordinary",
                            "LICAT,life insurance capital adequacy test",
                            "PBR,principle based reserving",
                            "NAIC,national association of insurance commissioners",
                            "ASC,accounting standards codification",
                            "FP&A,financial planning and analysis",
                            "SCD,slowly changing dimension",
                        ],
                    },
                },
            },
        },
        "mappings": {
            "properties": {
                "content": {"type": "text", "analyzer": "aaire_analyzer"},
                "doc_id": {"type": "keyword"},
                "document_title": {"type": "keyword"},
                "doc_type": {"type": "keyword"},
                "section": {"type": "text", "analyzer": "aaire_analyzer"},
                "page": {"type": "integer"},
                "primary_framework": {"type": "keyword"},
                "content_domains": {"type": "keyword"},
                "document_type": {"type": "keyword"},
                "job_id": {"type": "keyword"},
                "node_id": {"type": "keyword"},
                "element_type": {"type": "keyword"},
                "context_tags": {"type": "keyword"},
                "entities": {"type": "keyword"},
                "entity_orgs": {"type": "keyword"},
                "entity_persons": {"type": "keyword"},
            }
        },
    }


class ElasticsearchEngine:
    """
    Elasticsearch-backed keyword search engine.

    Drop-in replacement for BM25SearchEngine — same interface, persistent storage.
    Uses Elasticsearch's built-in BM25 scoring with a domain-aware analyzer.
    """

    def __init__(
        self,
        hosts: Optional[List[str]] = None,
        index_name: Optional[str] = None,
    ):
        from elasticsearch import Elasticsearch

        self._hosts = hosts or os.environ.get(
            "ELASTICSEARCH_HOSTS", ",".join(_DEFAULT_HOSTS)
        ).split(",")
        self._index_name = index_name or os.environ.get(
            "ELASTICSEARCH_INDEX", _DEFAULT_INDEX
        )

        self.client = Elasticsearch(
            self._hosts,
            request_timeout=_REQUEST_TIMEOUT,
        )
        self.is_ready = False
        self._ensure_index()
        logger.info(
            "ElasticsearchEngine initialized",
            hosts=self._hosts,
            index=self._index_name,
        )

    def _ensure_index(self) -> None:
        """Create the index with custom mappings if it doesn't already exist."""
        try:
            if not self.client.indices.exists(index=self._index_name):
                self.client.indices.create(
                    index=self._index_name,
                    body=_get_index_settings(),
                )
                logger.info("Created Elasticsearch index", index=self._index_name)
            self.is_ready = True
        except Exception as e:
            logger.error(
                "Failed to create Elasticsearch index",
                index=self._index_name,
                error=str(e),
            )
            self.is_ready = False

    # ------------------------------------------------------------------
    # Public interface (matches BM25SearchEngine)
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Bulk-index documents into Elasticsearch.

        Args:
            documents: List of dicts with 'content' and 'metadata' keys.

        Returns:
            Number of documents successfully indexed.
        """
        if not documents:
            return 0

        from elasticsearch.helpers import bulk

        actions = []
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            if not content or not isinstance(content, str):
                continue

            doc_id = (
                metadata.get("node_id")
                or metadata.get("doc_id")
                or str(uuid.uuid4())
            )

            # Flatten metadata fields into the ES document.
            es_doc = {
                "_index": self._index_name,
                "_id": doc_id,
                "content": content,
                "doc_id": doc_id,
                "document_title": metadata.get("document_title", metadata.get("filename", "")),
                "doc_type": metadata.get("doc_type", ""),
                "section": metadata.get("section", ""),
                "page": metadata.get("page", 0),
                "primary_framework": metadata.get("primary_framework", ""),
                "content_domains": metadata.get("content_domains", []),
                "document_type": metadata.get("document_type", ""),
                "job_id": metadata.get("job_id", ""),
                "node_id": metadata.get("node_id", ""),
                "element_type": metadata.get("element_type", "text"),
                "context_tags": metadata.get("context_tags", []),
            }
            actions.append(es_doc)

        if not actions:
            return 0

        try:
            success, errors = bulk(
                self.client,
                actions,
                raise_on_error=False,
                refresh="wait_for",  # Make docs searchable immediately.
            )
            if errors:
                logger.warning(
                    "Some documents failed to index",
                    success=success,
                    errors=len(errors),
                )
            logger.info(
                "Indexed documents in Elasticsearch",
                count=success,
                index=self._index_name,
            )
            self.is_ready = True
            return success
        except Exception as e:
            logger.error("Elasticsearch bulk index failed", error=str(e))
            raise

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20,
        highlight: bool = False,
    ) -> List[SearchResult]:
        """
        Search documents using Elasticsearch BM25.

        Args:
            query: Search query string.
            filters: Optional metadata filters (doc_type, job_id, etc.).
            limit: Maximum results to return.
            highlight: Unused (interface compatibility).

        Returns:
            List of SearchResult objects, highest score first.
        """
        if not self.is_ready:
            logger.warning("Elasticsearch not ready for search")
            return []

        if not query or not isinstance(query, str):
            logger.warning("Invalid query provided to Elasticsearch search")
            return []

        body: Dict[str, Any] = {
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "content",
                                "section^1.5",
                                "document_title^2",
                            ],
                            "type": "best_fields",
                        }
                    },
                }
            },
            "size": limit,
        }

        # Build filter clauses from metadata filters.
        filter_clauses = self._build_filters(filters)
        if filter_clauses:
            body["query"]["bool"]["filter"] = filter_clauses

        try:
            response = self.client.search(
                index=self._index_name,
                body=body,
            )
            hits = response.get("hits", {}).get("hits", [])
            results = [self._hit_to_result(hit) for hit in hits]

            logger.info(
                "Elasticsearch search completed",
                query=query[:30],
                results=len(results),
            )
            for i, r in enumerate(results[:3]):
                preview = r.content[:80].replace("\n", " ")
                logger.debug(
                    f"ES result {i+1}: score={r.score:.3f}, preview='{preview}...'"
                )

            return results

        except Exception as e:
            logger.error("Elasticsearch search failed", error=str(e), query=query[:50])
            return []

    def clear(self) -> None:
        """Delete all documents from the index (keeps the index and mappings)."""
        try:
            if self.client.indices.exists(index=self._index_name):
                self.client.delete_by_query(
                    index=self._index_name,
                    body={"query": {"match_all": {}}},
                    refresh=True,
                )
            logger.info("Elasticsearch index cleared", index=self._index_name)
        except Exception as e:
            logger.error("Failed to clear Elasticsearch index", error=str(e))

    def delete_by_metadata(self, field: str, value: str) -> int:
        """Delete documents where a metadata field matches a value."""
        try:
            if not self.client.indices.exists(index=self._index_name):
                return 0
            resp = self.client.delete_by_query(
                index=self._index_name,
                body={"query": {"term": {f"metadata.{field}.keyword": value}}},
                refresh=True,
            )
            deleted = resp.get("deleted", 0)
            if deleted:
                logger.info("Deleted documents from Elasticsearch",
                            field=field, value=value, deleted=deleted)
            return deleted
        except Exception as e:
            logger.error("Failed to delete by metadata", field=field, value=value, error=str(e))
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            if not self.client.indices.exists(index=self._index_name):
                return {"total_documents": 0, "is_ready": False, "has_index": False}

            count = self.client.count(index=self._index_name)["count"]
            return {
                "total_documents": count,
                "is_ready": self.is_ready,
                "has_index": True,
            }
        except Exception as e:
            logger.error("Failed to get Elasticsearch stats", error=str(e))
            return {"total_documents": 0, "is_ready": self.is_ready, "has_index": False}

    def doc_count(self) -> int:
        """Return the number of indexed documents. Used for startup sync check."""
        try:
            return self.client.count(index=self._index_name)["count"]
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Corpus statistics (c-TF-IDF)
    # ------------------------------------------------------------------

    def get_significant_terms(
        self,
        text: str,
        *,
        top_k: int = 10,
        min_doc_count: int = 2,
        max_doc_percent: float = 30.0,
    ) -> List[Dict[str, Any]]:
        """
        Use ES significant_terms aggregation to find corpus-discriminative terms.

        Queries ES with the given text, then uses significant_terms on the
        matched subset to identify terms statistically overrepresented
        compared to the full corpus. ES uses chi-square scoring internally.

        Returns:
            List of {"term": str, "score": float, "doc_count": int}.
        """
        if not self.is_ready:
            return []

        total_docs = max(self.doc_count(), 1)
        max_doc_count = int(max_doc_percent / 100 * total_docs)

        body: Dict[str, Any] = {
            "query": {
                "match": {
                    "content": {
                        "query": text[:500],
                        "minimum_should_match": "30%",
                    }
                }
            },
            "aggs": {
                "discriminative": {
                    "significant_terms": {
                        "field": "content",
                        "size": top_k,
                        "min_doc_count": min_doc_count,
                    }
                }
            },
            "size": 0,
        }

        # Only add max_doc_count if positive
        if max_doc_count > 0:
            body["aggs"]["discriminative"]["significant_terms"][
                "max_doc_count"
            ] = max_doc_count

        try:
            response = self.client.search(index=self._index_name, body=body)
            buckets = (
                response.get("aggregations", {})
                .get("discriminative", {})
                .get("buckets", [])
            )
            return [
                {
                    "term": b["key"],
                    "score": b.get("score", 0),
                    "doc_count": b.get("doc_count", 0),
                }
                for b in buckets
            ]
        except Exception as e:
            logger.warning("ES significant_terms failed", error=str(e))
            return []

    def get_termvectors(
        self,
        doc_id: str,
        *,
        min_term_freq: int = 2,
        min_word_length: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Use ES termvectors API to get per-document TF-IDF stats.

        Returns terms sorted by c-TF-IDF score (tf * log(N/df)).
        """
        if not self.is_ready:
            return []

        try:
            import math as _math

            response = self.client.termvectors(
                index=self._index_name,
                id=doc_id,
                fields=["content"],
                term_statistics=True,
                field_statistics=True,
            )

            terms_data = (
                response.get("term_vectors", {})
                .get("content", {})
                .get("terms", {})
            )
            total_docs = max(self.doc_count(), 1)

            results = []
            for term, stats in terms_data.items():
                tf = stats.get("term_freq", 0)
                df = stats.get("doc_freq", 1)

                if tf >= min_term_freq and len(term) >= min_word_length:
                    tfidf_score = tf * _math.log(total_docs / max(df, 1))
                    results.append(
                        {"term": term, "tf": tf, "df": df, "tfidf": tfidf_score}
                    )

            results.sort(key=lambda x: x["tfidf"], reverse=True)
            return results

        except Exception as e:
            logger.warning("ES termvectors failed", error=str(e))
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filters(filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert metadata filter dict into Elasticsearch filter clauses."""
        if not filters:
            return []

        clauses: List[Dict[str, Any]] = []
        for key, value in filters.items():
            if key.startswith("_"):
                continue
            if isinstance(value, list):
                clauses.append({"terms": {key: value}})
            else:
                clauses.append({"term": {key: value}})
        return clauses

    @staticmethod
    def _hit_to_result(hit: Dict[str, Any]) -> SearchResult:
        """Convert an Elasticsearch hit into a SearchResult."""
        source = hit.get("_source", {})
        content = source.get("content", "")

        # Reconstruct metadata dict for downstream compatibility.
        metadata = {
            k: v
            for k, v in source.items()
            if k != "content" and v  # skip empty values
        }

        return SearchResult(
            doc_id=source.get("doc_id", hit.get("_id", "")),
            content=content,
            metadata=metadata,
            score=float(hit.get("_score", 0)),
        )


def create_elasticsearch_engine(
    hosts: Optional[List[str]] = None,
    index_name: Optional[str] = None,
) -> ElasticsearchEngine:
    """Factory function to create an ElasticsearchEngine instance."""
    return ElasticsearchEngine(hosts=hosts, index_name=index_name)
