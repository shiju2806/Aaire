"""
Element-type-aware retriever.

Extends basic hybrid search with knowledge of element types (text, table,
formula, callout, image). For numerical/quantitative queries, tables get
higher weight. For conceptual queries, text gets higher weight.

Also fetches original-fidelity content from the structured store so the
LLM receives actual tables and formulas, not just summaries.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import structlog

from ..ingestion.chunk_schema import StructuredStore
from ..providers.config_loader import get_config, get_nested

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Query classification for element-type routing
# ---------------------------------------------------------------------------

# Keywords that signal the query needs table/numerical data.
_NUMERICAL_SIGNALS = {
    "rate", "rates", "table", "tables", "factor", "factors", "mortality",
    "morbidity", "lapse", "premium", "reserve", "value", "amount",
    "percentage", "ratio", "how much", "how many", "calculate", "compute",
    "number", "figures", "data", "statistics", "average", "total", "sum",
}

# Keywords that signal the query needs formula content.
_FORMULA_SIGNALS = {
    "formula", "equation", "calculate", "computation", "derivation",
    "derive", "proof", "variable", "function", "integral", "summation",
    "actuarial present value", "apv", "net premium", "gross premium",
    "reserve formula", "npv", "irr",
}

# Keywords that signal conceptual/definitional queries.
_CONCEPTUAL_SIGNALS = {
    "what is", "what are", "define", "definition", "explain", "describe",
    "overview", "summary", "introduction", "concept", "principle",
    "meaning", "difference between", "compare", "contrast",
}


@dataclass
class ElementTypeWeights:
    """Scoring weights by element type for a given query."""

    text: float = 1.0
    table: float = 1.0
    formula: float = 1.0
    callout: float = 1.2  # Callouts always get a slight boost.
    image: float = 0.8
    table_proposition: float = 1.0


@dataclass
class EnrichedResult:
    """A retrieval result enriched with original-fidelity content."""

    content: str
    metadata: Dict[str, Any]
    score: float
    node_id: str
    search_type: str = "vector"
    element_type: str = "text"
    original_content: Optional[str] = None  # From structured store.
    original_content_type: Optional[str] = None
    rerank_score: Optional[float] = None  # Cross-encoder relevance score.


class ElementAwareRetriever:
    """Element-type-aware retrieval with structured content enrichment.

    Workflow:
    1. Extract entities from query (for disambiguation filtering)
    2. Resolve entities in knowledge graph (if available)
    3. Classify query → determine element type weights
    4. Run search (delegates to DocumentRetriever) with entity filters
    5. Run graph traversal in parallel (connected chunks)
    6. RRF merge: hybrid search + graph-connected chunks
    7. Apply element-type-specific score adjustments
    8. Apply entity overlap scoring
    9. Fetch original content from structured store for tables/formulas
    10. Return enriched results
    """

    def __init__(
        self,
        document_retriever: Any,
        structured_store: Optional[StructuredStore] = None,
        retrieval_provider: Optional[Any] = None,
        graph_store: Optional[Any] = None,
    ) -> None:
        self._retriever = document_retriever
        self._structured_store = structured_store or StructuredStore()
        self._retrieval_provider = retrieval_provider
        self._graph_store = graph_store  # GraphStore for knowledge graph traversal
        self._entity_extractor = None  # Set by rag_pipeline.py after init

    async def retrieve(
        self,
        query: str,
        doc_type_filter: Optional[List[str]] = None,
        similarity_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        fetch_originals: bool = True,
    ) -> List[EnrichedResult]:
        """Retrieve documents with element-type, entity, and graph awareness.

        Args:
            query: User query.
            doc_type_filter: Optional document type filter.
            similarity_threshold: Minimum similarity score.
            filters: Additional metadata filters.
            fetch_originals: Whether to fetch original content from
                           structured store for tables/formulas/images.

        Returns:
            List of EnrichedResult objects, ranked by adjusted score.
        """
        # 1. Extract entities from query for disambiguation.
        query_entities = self._extract_query_entities(query)

        # 2. Build entity-based filters (merged with existing filters).
        entity_filters = self._build_entity_filters(query_entities, filters)

        # 3. Classify query to determine element type weights.
        weights = self._classify_query(query)

        entity_cfg = get_nested(
            get_config("entity_extraction"),
            "retrieval_filter",
            default={},
        )

        # 4. Run THREE retrieval paths in parallel:
        #    a) Hybrid search (vector + BM25)
        #    b) Graph resolution + traversal (sync, offloaded to thread)
        #    c) Targeted element-type search
        hybrid_task = self._retriever.retrieve_documents(
            query, doc_type_filter, similarity_threshold, entity_filters
        )
        element_task = self._targeted_element_search(query, weights, filters)

        # Graph resolution is sync (ES calls) — wrap in asyncio.to_thread
        has_graph = (
            self._graph_store
            and query_entities is not None
            and query_entities.entity_count > 0
        )
        if has_graph:
            graph_task = asyncio.get_event_loop().run_in_executor(
                None, self._resolve_and_traverse, query_entities
            )
            raw_results, extra_results, graph_result = await asyncio.gather(
                hybrid_task, element_task, graph_task
            )
            resolved_nodes, graph_chunk_ids = graph_result
        else:
            raw_results, extra_results = await asyncio.gather(
                hybrid_task, element_task
            )
            resolved_nodes = []
            graph_chunk_ids: Set[str] = set()

        # Store resolved info for audit trail (consumed by rag_pipeline)
        self._last_resolved_nodes = resolved_nodes
        self._last_graph_chunk_ids = sorted(graph_chunk_ids)

        # 5. Fallback: if entity filter was too restrictive, retry unfiltered.
        fallback_threshold = entity_cfg.get("fallback_on_few_results", 3)

        if (
            len(raw_results) < fallback_threshold
            and query_entities is not None
            and query_entities.entity_count > 0
            and entity_filters != filters
        ):
            logger.info(
                "Entity-filtered retrieval too restrictive, falling back to unfiltered",
                entity_results=len(raw_results),
                threshold=fallback_threshold,
            )
            raw_results = await self._retriever.retrieve_documents(
                query, doc_type_filter, similarity_threshold, filters
            )

        # 6. Merge extra element-type results (dedup by node_id).
        seen_ids: Set[str] = {r['node_id'] for r in raw_results}
        for r in extra_results:
            if r['node_id'] not in seen_ids:
                raw_results.append(r)
                seen_ids.add(r['node_id'])

        # 7. Retrieve graph-connected chunks and RRF merge.
        if graph_chunk_ids:
            graph_results = self._retrieve_graph_chunks(graph_chunk_ids, seen_ids)
            raw_results = self._rrf_merge(raw_results, graph_results)
            seen_ids.update(r['node_id'] for r in graph_results)

        # 7.5. Section expansion — pull sibling chunks from high-scoring sections.
        section_cfg = get_nested(get_config("scoring"), "section_expansion", default={})
        if section_cfg.get("enabled", True):
            section_extras = self._expand_sections(
                raw_results,
                seen_ids,
                max_expansion=section_cfg.get("max_expansion_chunks", 10),
                score_threshold=section_cfg.get("score_threshold", 0.3),
                max_sections=section_cfg.get("max_sections", 3),
                expansion_discount=section_cfg.get("expansion_score_discount", 0.7),
            )
            if section_extras:
                raw_results.extend(section_extras)
                seen_ids.update(r['node_id'] for r in section_extras)
                logger.info(
                    "Section expansion: added chunks",
                    added=len(section_extras),
                )

        # 8. Apply element-type score adjustments.
        enriched: List[EnrichedResult] = []
        for result in raw_results:
            element_type = result.get('metadata', {}).get('element_type', 'text')
            weight = getattr(weights, element_type, 1.0)
            adjusted_score = (result.get('score', 0) or result.get('relevance_score', 0)) * weight

            enriched.append(
                EnrichedResult(
                    content=result.get('content', ''),
                    metadata=result.get('metadata', {}),
                    score=adjusted_score,
                    node_id=result.get('node_id', ''),
                    search_type=result.get('search_type', 'vector'),
                    element_type=element_type,
                )
            )

        # 9. Apply entity overlap scoring boost.
        if query_entities is not None and query_entities.entity_count > 0:
            boost_factor = entity_cfg.get("boost_factor", 1.5)
            self._apply_entity_scores(enriched, query_entities, boost_factor)

        # Sort by adjusted score.
        enriched.sort(key=lambda r: r.score, reverse=True)

        # 10. Fetch original content from structured store.
        if fetch_originals and self._structured_store:
            chunk_ids = [
                r.node_id for r in enriched
                if r.element_type in ('table', 'formula', 'image')
            ]
            if chunk_ids:
                originals = self._structured_store.retrieve_batch(chunk_ids)
                for result in enriched:
                    if result.node_id in originals:
                        record = originals[result.node_id]
                        result.original_content = record.get('content')
                        result.original_content_type = record.get('content_type')

        logger.info(
            "Element-aware retrieval complete",
            total=len(enriched),
            types={et: sum(1 for r in enriched if r.element_type == et)
                   for et in set(r.element_type for r in enriched)},
            originals_fetched=sum(1 for r in enriched if r.original_content),
            entity_filtered=entity_filters != filters,
            graph_chunks=len(graph_chunk_ids),
            resolved_entities=len(resolved_nodes),
        )

        return enriched

    # ------------------------------------------------------------------
    # Entity helpers
    # ------------------------------------------------------------------

    def _extract_query_entities(self, query: str):
        """Extract entities from query. Returns ExtractedEntities or None."""
        if not self._entity_extractor:
            return None
        try:
            entities = self._entity_extractor.extract_from_query(query)
            if entities.entity_count > 0:
                logger.info(
                    "Query entities extracted",
                    entities=entities.all_entities[:5],
                    count=entities.entity_count,
                    source=entities.extraction_source,
                )
            return entities
        except Exception as e:
            logger.warning("Query entity extraction failed (non-fatal)", error=str(e))
            return None

    def _build_entity_filters(
        self,
        query_entities,
        existing_filters: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Merge entity filters with existing filters based on config strategy.

        Returns the original filters if no entity filtering should be applied.

        IMPORTANT: The "should_match" and "must_match" strategies add entity
        filters to Qdrant, which means chunks WITHOUT entities in their payload
        will be excluded entirely. This is intentional for "must_match" but
        can cause entity-invisible chunks for "should_match". The recommended
        default is "boost_only" which scores entity matches without filtering.
        """
        if query_entities is None or query_entities.entity_count == 0:
            return existing_filters

        entity_cfg = get_nested(
            get_config("entity_extraction"),
            "retrieval_filter",
            default={},
        )

        strategy = entity_cfg.get("strategy", "boost_only")
        fallback_on_empty = entity_cfg.get("fallback_on_empty", True)
        max_filter = entity_cfg.get("max_filter_entities", 5)

        if fallback_on_empty and query_entities.entity_count == 0:
            return existing_filters

        if strategy == "boost_only":
            # Don't filter, only boost scores later in _apply_entity_scores.
            return existing_filters

        # "should_match" or "must_match" — add entity payload filter.
        # WARNING: This excludes chunks with empty entity lists.
        filter_entities = query_entities.all_entities[:max_filter]

        merged = dict(existing_filters) if existing_filters else {}
        merged["entities"] = filter_entities

        logger.info(
            "Entity filter applied",
            strategy=strategy,
            filter_entities=filter_entities,
            warning="chunks without entities will be excluded" if strategy == "should_match" else None,
        )
        return merged

    @staticmethod
    def _apply_entity_scores(
        enriched: List[EnrichedResult],
        query_entities,
        boost_factor: float = 1.5,
    ) -> None:
        """Boost scores for results whose entities overlap with query entities."""
        query_set = set(query_entities.all_entities)
        if not query_set:
            return

        for result in enriched:
            chunk_entities = set(result.metadata.get("entities", []))
            if not chunk_entities:
                continue

            # Jaccard-like overlap
            overlap = query_set & chunk_entities
            if overlap:
                overlap_ratio = len(overlap) / len(query_set)
                # Scale boost: full boost at 100% overlap, partial otherwise
                effective_boost = 1.0 + (boost_factor - 1.0) * overlap_ratio
                result.score *= effective_boost
                logger.debug(
                    "Entity overlap boost",
                    node_id=result.node_id[:12],
                    overlap=list(overlap),
                    boost=round(effective_boost, 2),
                )

    # ------------------------------------------------------------------
    # Knowledge graph helpers
    # ------------------------------------------------------------------

    def _resolve_and_traverse(self, query_entities) -> tuple:
        """Resolve query entities in the knowledge graph and get connected chunks.

        Returns:
            (resolved_nodes, graph_chunk_ids) tuple.
        """
        resolved_nodes = []
        graph_chunk_ids: Set[str] = set()

        kg_config = get_config("knowledge_graph")
        max_graph_chunks = get_nested(kg_config, "retrieval", "max_graph_chunks", default=20)

        for entity_name in query_entities.all_entities:
            try:
                node = self._graph_store.resolve_entity(entity_name)
                if node:
                    resolved_nodes.append(node)
                    connected = self._graph_store.get_connected_chunks(node.entity_id)
                    graph_chunk_ids.update(connected)
            except Exception as e:
                logger.debug("Graph entity resolution failed", entity=entity_name, error=str(e))

        # Cap to prevent overwhelming results
        if len(graph_chunk_ids) > max_graph_chunks:
            graph_chunk_ids = set(sorted(graph_chunk_ids)[:max_graph_chunks])

        if resolved_nodes:
            logger.info(
                "Graph entities resolved",
                resolved_count=len(resolved_nodes),
                graph_chunks=len(graph_chunk_ids),
                entities=[n.canonical_name for n in resolved_nodes[:5]],
            )

        return resolved_nodes, graph_chunk_ids

    def _retrieve_graph_chunks(
        self, graph_chunk_ids: Set[str], already_seen: Set[str]
    ) -> List[Dict]:
        """Fetch chunk content for graph-connected chunk IDs not already in results.

        Uses the search engine to look up chunks by their IDs.
        """
        new_ids = graph_chunk_ids - already_seen
        if not new_ids:
            return []

        results: List[Dict] = []

        # Try to fetch from structured store first (original content)
        if self._structured_store:
            originals = self._structured_store.retrieve_batch(list(new_ids))
            for chunk_id, record in originals.items():
                results.append({
                    'content': record.get('content', ''),
                    'metadata': record.get('metadata', {}),
                    'score': 0.0,  # Will be boosted by RRF
                    'node_id': chunk_id,
                    'search_type': 'graph',
                })
                new_ids.discard(chunk_id)

        # Remaining IDs: try to fetch from Qdrant via retrieval provider
        if new_ids and self._retrieval_provider:
            try:
                from qdrant_client.models import PointIdsList
                points = self._retrieval_provider.retrieve(
                    collection_name="documents",
                    ids=list(new_ids),
                )
                for point in points:
                    payload = point.payload or {}
                    results.append({
                        'content': payload.get('display_text', payload.get('content', '')),
                        'metadata': payload,
                        'score': 0.0,
                        'node_id': str(point.id),
                        'search_type': 'graph',
                    })
            except Exception as e:
                logger.debug("Failed to fetch graph chunks from Qdrant", error=str(e))

        return results

    @staticmethod
    def _rrf_merge(
        hybrid_results: List[Dict],
        graph_results: List[Dict],
        k: int = 60,
    ) -> List[Dict]:
        """Reciprocal Rank Fusion merge of hybrid search and graph-connected results.

        Graph-connected chunks get a configurable boost (via higher effective rank).

        Args:
            hybrid_results: Results from vector + BM25 search.
            graph_results: Results from graph traversal.
            k: RRF constant (default 60).

        Returns:
            Merged and re-scored list of result dicts.
        """
        kg_config = get_config("knowledge_graph")
        graph_boost = get_nested(kg_config, "retrieval", "graph_boost", default=1.3)

        scores: Dict[str, float] = {}
        result_map: Dict[str, Dict] = {}

        # Score hybrid results by rank
        for rank, result in enumerate(hybrid_results):
            node_id = result.get('node_id', '')
            if not node_id:
                continue
            scores[node_id] = scores.get(node_id, 0) + 1.0 / (k + rank + 1)
            result_map[node_id] = result

        # Score graph results with boost
        for rank, result in enumerate(graph_results):
            node_id = result.get('node_id', '')
            if not node_id:
                continue
            scores[node_id] = scores.get(node_id, 0) + graph_boost / (k + rank + 1)
            if node_id not in result_map:
                result_map[node_id] = result

        # Sort by RRF score and update result scores
        sorted_ids = sorted(scores.keys(), key=lambda nid: scores[nid], reverse=True)
        merged = []
        for node_id in sorted_ids:
            result = result_map[node_id]
            result['score'] = scores[node_id]
            merged.append(result)

        return merged

    def _expand_sections(
        self,
        results: List[Dict],
        seen_ids: Set[str],
        max_expansion: int = 10,
        score_threshold: float = 0.3,
        max_sections: int = 3,
        expansion_discount: float = 0.7,
    ) -> List[Dict]:
        """Expand top-scoring sections by pulling sibling chunks from Qdrant.

        For each high-scoring chunk, retrieve other chunks from the same
        (document_title, section) pair so the LLM gets full section context.

        Returns new chunks not already in ``seen_ids``.
        """
        if not self._retrieval_provider:
            return []

        # Collect unique (doc_title, section) pairs with max score.
        section_scores: Dict[tuple, float] = {}
        for r in results:
            score = r.get('score', 0) or r.get('relevance_score', 0)
            if score < score_threshold:
                continue
            meta = r.get('metadata', {})
            doc_title = meta.get('document_title', '')
            section = meta.get('section', '')
            if not doc_title or not section:
                continue
            key = (doc_title, section)
            if key not in section_scores or score > section_scores[key]:
                section_scores[key] = score

        if not section_scores:
            return []

        # Pick top N sections by score.
        top_sections = sorted(section_scores.items(), key=lambda x: x[1], reverse=True)[:max_sections]

        new_chunks: List[Dict] = []
        for (doc_title, section), parent_score in top_sections:
            if len(new_chunks) >= max_expansion:
                break
            try:
                for batch in self._retrieval_provider.scroll(
                    filters={"document_title": doc_title, "section": section},
                    batch_size=20,
                ):
                    for sr in batch:
                        if sr.point_id in seen_ids:
                            continue
                        if len(new_chunks) >= max_expansion:
                            break
                        new_chunks.append({
                            'content': sr.payload.get('display_text', sr.payload.get('content', '')),
                            'metadata': sr.payload,
                            'score': parent_score * expansion_discount,
                            'node_id': sr.point_id,
                            'search_type': 'section_expansion',
                        })
                    if len(new_chunks) >= max_expansion:
                        break
            except Exception as e:
                logger.debug("Section expansion scroll failed", section=section, error=str(e))

        return new_chunks

    def _classify_query(self, query: str) -> ElementTypeWeights:
        """Determine element type weights based on query content."""
        query_lower = query.lower()
        tokens = set(query_lower.split())

        numerical_score = len(tokens & _NUMERICAL_SIGNALS)
        formula_score = len(tokens & _FORMULA_SIGNALS)
        conceptual_score = sum(1 for phrase in _CONCEPTUAL_SIGNALS if phrase in query_lower)

        weights = ElementTypeWeights()

        if numerical_score >= 2:
            # Numerical query — boost tables.
            weights.table = 1.5
            weights.table_proposition = 1.4
            weights.text = 0.8
            logger.debug("Query classified as numerical", query=query[:50])

        if formula_score >= 1:
            # Formula query — boost formulas.
            weights.formula = 1.5
            weights.text = 0.9
            logger.debug("Query classified as formula-seeking", query=query[:50])

        if conceptual_score >= 1 and numerical_score == 0:
            # Conceptual query — text is primary.
            weights.text = 1.3
            weights.table = 0.7
            weights.formula = 0.8
            logger.debug("Query classified as conceptual", query=query[:50])

        return weights

    async def _targeted_element_search(
        self,
        query: str,
        weights: ElementTypeWeights,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict]:
        """Run additional targeted searches for high-weight element types.

        If the query strongly prefers tables or formulas, we run an
        additional search filtered to those element types to ensure
        they appear in results even if generic search missed them.
        """
        if not self._retrieval_provider:
            return []

        tasks = []

        if weights.table >= 1.3:
            tasks.append(self._search_by_element_type(query, "table", filters))
            tasks.append(self._search_by_element_type(query, "table_proposition", filters))

        if weights.formula >= 1.3:
            tasks.append(self._search_by_element_type(query, "formula", filters))

        if not tasks:
            return []

        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        all_results: List[Dict] = []
        for result in results_lists:
            if isinstance(result, list):
                all_results.extend(result)
            elif isinstance(result, Exception):
                logger.warning("Targeted element search failed", error=str(result))

        return all_results

    async def _search_by_element_type(
        self,
        query: str,
        element_type: str,
        filters: Optional[Dict[str, Any]],
        limit: int = 5,
    ) -> List[Dict]:
        """Search Qdrant filtered to a specific element type."""
        try:
            search_filters = dict(filters) if filters else {}
            search_filters['element_type'] = element_type

            results = await self._retrieval_provider.search(
                collection_name="documents",
                query_text=query,
                filters=search_filters,
                limit=limit,
            )

            return [
                {
                    'content': r.payload.get('display_text', r.payload.get('content', '')),
                    'metadata': r.payload,
                    'score': r.score,
                    'node_id': str(r.point_id),
                    'search_type': f'targeted_{element_type}',
                }
                for r in results
            ]
        except Exception as e:
            logger.debug("Element-type search unavailable", type=element_type, error=str(e))
            return []
