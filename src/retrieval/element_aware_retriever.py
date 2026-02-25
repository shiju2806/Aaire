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


class ElementAwareRetriever:
    """Element-type-aware retrieval with structured content enrichment.

    Workflow:
    1. Classify query → determine element type weights
    2. Run search (delegates to DocumentRetriever)
    3. Apply element-type-specific score adjustments
    4. Fetch original content from structured store for tables/formulas
    5. Return enriched results
    """

    def __init__(
        self,
        document_retriever: Any,
        structured_store: Optional[StructuredStore] = None,
        retrieval_provider: Optional[Any] = None,
    ) -> None:
        self._retriever = document_retriever
        self._structured_store = structured_store or StructuredStore()
        self._retrieval_provider = retrieval_provider

    async def retrieve(
        self,
        query: str,
        doc_type_filter: Optional[List[str]] = None,
        similarity_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        fetch_originals: bool = True,
    ) -> List[EnrichedResult]:
        """Retrieve documents with element-type awareness.

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
        # 1. Classify query to determine element type weights.
        weights = self._classify_query(query)

        # 2. Run hybrid search via existing retriever.
        raw_results = await self._retriever.retrieve_documents(
            query, doc_type_filter, similarity_threshold, filters
        )

        # 3. Also run element-type filtered searches in parallel for
        #    types that are especially relevant to this query.
        extra_results = await self._targeted_element_search(
            query, weights, filters
        )

        # Merge extra results (dedup by node_id).
        seen_ids: Set[str] = {r['node_id'] for r in raw_results}
        for r in extra_results:
            if r['node_id'] not in seen_ids:
                raw_results.append(r)
                seen_ids.add(r['node_id'])

        # 4. Apply element-type score adjustments.
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

        # Sort by adjusted score.
        enriched.sort(key=lambda r: r.score, reverse=True)

        # 5. Fetch original content from structured store.
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
        )

        return enriched

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
