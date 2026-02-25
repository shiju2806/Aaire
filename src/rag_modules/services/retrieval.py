"""
Document Retrieval Service Module

Handles all document retrieval operations including:
- Hybrid search (vector + keyword) with parallel execution
- Vector-based semantic search
- Keyword-based BM25 search
- Confidence-based early exit
- Search result combination and ranking
- Retrieval metrics instrumentation
"""

import asyncio
import re
import time
import structlog
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from ...providers.config_loader import get_config, get_nested

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Retrieval Metrics
# ---------------------------------------------------------------------------


@dataclass
class RetrievalMetrics:
    """Per-query retrieval metrics for monitoring and optimization."""

    retrieval_latency_ms: float = 0.0
    vector_latency_ms: float = 0.0
    keyword_latency_ms: float = 0.0
    vector_result_count: int = 0
    keyword_result_count: int = 0
    final_result_count: int = 0
    top_5_average_score: float = 0.0
    element_types_retrieved: List[str] = field(default_factory=list)
    early_exit_triggered: bool = False
    bm25_contributed: bool = False
    bm25_skipped: bool = False
    entropy_disambiguation_triggered: bool = False

    def log(self) -> None:
        """Emit structured log of metrics."""
        logger.info(
            "retrieval_metrics",
            retrieval_latency_ms=round(self.retrieval_latency_ms, 1),
            vector_latency_ms=round(self.vector_latency_ms, 1),
            keyword_latency_ms=round(self.keyword_latency_ms, 1),
            vector_results=self.vector_result_count,
            keyword_results=self.keyword_result_count,
            final_results=self.final_result_count,
            top_5_avg_score=round(self.top_5_average_score, 3),
            element_types=self.element_types_retrieved,
            early_exit=self.early_exit_triggered,
            bm25_contributed=self.bm25_contributed,
            bm25_skipped=self.bm25_skipped,
        )


class DocumentRetriever:
    """Handles document retrieval operations with hybrid search capabilities"""

    def __init__(self, vector_index=None, bm25_engine=None, relevance_engine=None,
                 metadata_analyzer=None, quality_metrics_manager=None, config=None):
        """Initialize retriever with required components"""
        self.index = vector_index
        self.bm25_engine = bm25_engine
        self.relevance_engine = relevance_engine
        self.metadata_analyzer = metadata_analyzer
        self.quality_metrics_manager = quality_metrics_manager
        self.config = config or {}

        # Derived attributes
        self.keyword_search_ready = bm25_engine is not None

    async def retrieve_documents(self, query: str, doc_type_filter: Optional[List[str]],
                               similarity_threshold: Optional[float] = None,
                               filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Hybrid retrieval: parallel vector + BM25 search with confidence-based early exit."""

        start_time = time.perf_counter()
        metrics = RetrievalMetrics()

        # Load confidence thresholds from config.
        scoring_config = get_config("scoring")
        early_exit_threshold = get_nested(
            scoring_config, "retrieval", "confidence_early_exit", default=0.90
        )
        low_confidence_threshold = get_nested(
            scoring_config, "retrieval", "default_similarity_threshold", default=0.5
        )

        # Analyze query intent for smart filtering
        try:
            query_intent = await self.metadata_analyzer.analyze_query_intent(query)
            logger.info("Query intent analyzed",
                       query=query[:50],
                       confidence=query_intent.confidence,
                       domains=query_intent.content_domains,
                       framework=query_intent.required_filters.get('framework'),
                       tags=query_intent.context_tags[:3])
        except Exception as e:
            logger.warning("Query intent analysis failed, proceeding without smart filtering",
                         error=str(e))

        # Get the target document limit
        doc_limit = self.quality_metrics_manager.get_document_limit(query)

        # --- Parallel search execution ---
        vector_start = time.perf_counter()
        vector_coro = self.vector_search(query, doc_type_filter, similarity_threshold, filters, buffer_limit=doc_limit)
        keyword_coro = self.keyword_search(query, doc_type_filter, filters, buffer_limit=doc_limit)

        vector_results, keyword_results = await asyncio.gather(vector_coro, keyword_coro)

        metrics.vector_latency_ms = (time.perf_counter() - vector_start) * 1000
        metrics.vector_result_count = len(vector_results)
        metrics.keyword_result_count = len(keyword_results)

        # --- Confidence-based early exit ---
        # If top 3 vector results all score above threshold, skip BM25 contribution
        if len(vector_results) >= 3:
            top_3_scores = [r['score'] for r in vector_results[:3]]
            if all(s >= early_exit_threshold for s in top_3_scores):
                logger.info(
                    "Early exit: top 3 vector scores all above threshold",
                    scores=top_3_scores,
                    threshold=early_exit_threshold,
                )
                metrics.early_exit_triggered = True
                metrics.bm25_skipped = True
                # Use only vector results — already high confidence
                combined_results = self.relevance_engine.rank_documents(query, vector_results)
                if len(combined_results) > doc_limit:
                    combined_results = combined_results[:doc_limit]
                metrics.final_result_count = len(combined_results)
                metrics.top_5_average_score = self._avg_top_n_score(combined_results, 5)
                metrics.element_types_retrieved = self._unique_element_types(combined_results)
                metrics.retrieval_latency_ms = (time.perf_counter() - start_time) * 1000
                metrics.log()
                return combined_results

        logger.info(f"Parallel retrieval: {len(vector_results)} vector + {len(keyword_results)} keyword results, target limit: {doc_limit}")

        # --- Combine results ---
        all_results = vector_results + keyword_results

        # Remove duplicates while preserving all scoring info
        unique_results = {}
        for result in all_results:
            node_id = result['node_id']
            if node_id not in unique_results:
                unique_results[node_id] = result
            else:
                existing = unique_results[node_id]
                existing['search_type'] = 'hybrid'
                if result['score'] > existing['score']:
                    existing['score'] = result['score']

        # Check if BM25 contributed unique results
        vector_ids = {r['node_id'] for r in vector_results}
        keyword_ids = {r['node_id'] for r in keyword_results}
        metrics.bm25_contributed = bool(keyword_ids - vector_ids)

        # Use advanced relevance engine for ranking
        combined_results = self.relevance_engine.rank_documents(query, list(unique_results.values()))

        if len(combined_results) > doc_limit:
            combined_results = combined_results[:doc_limit]

        # --- Metrics ---
        metrics.final_result_count = len(combined_results)
        metrics.top_5_average_score = self._avg_top_n_score(combined_results, 5)
        metrics.element_types_retrieved = self._unique_element_types(combined_results)
        metrics.retrieval_latency_ms = (time.perf_counter() - start_time) * 1000
        metrics.log()

        return combined_results

    @staticmethod
    def _avg_top_n_score(results: List[Dict], n: int) -> float:
        """Average score of top N results."""
        scores = [r.get('score', 0) or r.get('relevance_score', 0) for r in results[:n]]
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _unique_element_types(results: List[Dict]) -> List[str]:
        """Unique element types present in results."""
        types = set()
        for r in results:
            et = r.get('metadata', {}).get('element_type', 'text')
            types.add(et)
        return sorted(types)

    async def vector_search(self, query: str, doc_type_filter: Optional[List[str]],
                          similarity_threshold: Optional[float] = None,
                          filters: Optional[Dict[str, Any]] = None,
                          buffer_limit: Optional[int] = None) -> List[Dict]:
        """Original vector-based semantic search with filtering support"""
        all_results = []

        try:
            # Use buffer limit if provided (for combined ranking), otherwise use dynamic calculation
            if buffer_limit is not None:
                vector_limit = buffer_limit
            else:
                # Get dynamic document limit based on query
                doc_limit = self.quality_metrics_manager.get_document_limit(query)
                # For standalone search, allocate 70% to vector search
                vector_limit = int(doc_limit * 0.7)

            # Create retriever from single index with dynamic limit
            retriever = self.index.as_retriever(
                similarity_top_k=vector_limit
            )

            # Retrieve documents
            nodes = retriever.retrieve(query)

            # Use adaptive threshold if provided, otherwise fall back to config
            threshold = similarity_threshold if similarity_threshold is not None else self.config.get('retrieval_config', {}).get('similarity_threshold', 0.5)

            for node in nodes:
                if node.score >= threshold:
                    # Apply job_id filter if specified (highest priority)
                    if filters and 'job_id' in filters:
                        node_job_id = node.metadata.get('job_id') if node.metadata else None
                        if node_job_id != filters['job_id']:
                            continue  # Skip nodes from different documents

                    # Apply document type filter if specified
                    if doc_type_filter:
                        node_doc_type = node.metadata.get('doc_type') if node.metadata else None
                        if node_doc_type not in doc_type_filter:
                            continue  # Skip nodes that don't match filter

                    # Apply smart metadata filters if specified
                    if filters:
                        node_metadata = node.metadata or {}

                        # Apply required filters (must match exactly)
                        skip_node = False
                        for key, value in filters.items():
                            if key.startswith('_'):  # Skip special filter keys
                                continue
                            if key in ['job_id']:  # Skip already handled filters
                                continue

                            node_value = node_metadata.get(key)
                            if node_value != value:
                                # For context_tags, also check if the required tag is in the list
                                if key == 'context_tags' and isinstance(node_value, list):
                                    if value not in node_value:
                                        skip_node = True
                                        break
                                else:
                                    skip_node = True
                                    break

                        if skip_node:
                            continue

                        # Apply excluded filters (must NOT match)
                        if '_excluded' in filters:
                            excluded_filters = filters['_excluded']
                            skip_node = False
                            for key, value in excluded_filters.items():
                                node_value = node_metadata.get(key)
                                if node_value == value:
                                    # For context_tags, check if excluded tag is in the list
                                    if key == 'context_tags' and isinstance(node_value, list):
                                        if value in node_value:
                                            skip_node = True
                                            break
                                    else:
                                        skip_node = True
                                        break

                            if skip_node:
                                continue

                    all_results.append({
                        'content': node.text,
                        'metadata': node.metadata or {},
                        'score': node.score,
                        'source_type': node.metadata.get('doc_type', 'unknown') if node.metadata else 'unknown',
                        'node_id': node.id_,
                        'search_type': 'vector'
                    })

        except Exception as e:
            logger.warning("Failed to retrieve from index", error=str(e))

        # Sort by relevance score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        # Note: vector_limit is already applied in retriever, but trimming here for safety
        return all_results[:vector_limit] if 'vector_limit' in locals() else all_results

    async def keyword_search(self, query: str, doc_type_filter: Optional[List[str]],
                           filters: Optional[Dict[str, Any]] = None,
                           buffer_limit: Optional[int] = None) -> List[Dict]:
        """BM25-based keyword search"""
        results = []

        try:
            if not self.bm25_engine or not self.keyword_search_ready:
                logger.info("BM25 search engine not available, skipping keyword search")
                return results

            # Use buffer limit if provided (for combined ranking), otherwise use dynamic calculation
            if buffer_limit is not None:
                keyword_limit = buffer_limit
            else:
                # Get dynamic document limit based on query
                doc_limit = self.quality_metrics_manager.get_document_limit(query)
                # For standalone search, allocate 30% to keyword search
                keyword_limit = int(doc_limit * 0.3)

            # Convert filters to BM25 format
            bm25_filters = self.convert_filters_for_bm25(filters, doc_type_filter)

            # Perform BM25 search
            search_results = self.bm25_engine.search(
                query=query,
                filters=bm25_filters,
                limit=keyword_limit,
                highlight=False
            )

            # Convert BM25 SearchResults to our format
            for search_result in search_results:
                results.append({
                    'content': search_result.content,
                    'metadata': search_result.metadata,
                    'score': search_result.score,
                    'source_type': search_result.metadata.get('doc_type', 'unknown'),
                    'node_id': search_result.doc_id,
                    'search_type': 'keyword'
                })

            logger.info(f"BM25 keyword search found {len(results)} results")

            # Debug: Show first 5 results to understand what's being matched
            for i, result in enumerate(results[:5]):
                content_preview = result['content'][:100].replace('\n', ' ')
                logger.info(f"🔍 BM25 result {i+1}: doc_id={result['node_id']}, score={result['score']:.3f}, preview='{content_preview}...'")

        except Exception as e:
            logger.error("Failed to perform BM25 keyword search", error=str(e))

        return results

    def convert_filters_for_bm25(self, filters: Optional[Dict[str, Any]],
                               doc_type_filter: Optional[List[str]]) -> Optional[Dict[str, Any]]:
        """Convert RAG pipeline filters to BM25 filter format"""
        if not filters and not doc_type_filter:
            return None

        bm25_filters = {}

        if filters:
            # Handle primary_framework filter (most important for smart metadata)
            if 'primary_framework' in filters:
                bm25_filters['primary_framework'] = filters['primary_framework']

            # Handle content_domains filter
            if 'content_domains' in filters:
                bm25_filters['content_domains'] = filters['content_domains']

            # Handle document_type filter
            if 'document_type' in filters:
                bm25_filters['document_type'] = filters['document_type']

            # Handle job_id filter (highest priority)
            if 'job_id' in filters:
                bm25_filters['job_id'] = filters['job_id']

        # Convert doc_type_filter to BM25 format
        if doc_type_filter:
            # BM25 engine expects doc_type in the doc_type field
            bm25_filters['doc_type'] = doc_type_filter

        return bm25_filters if bm25_filters else None

    def combine_search_results(self, vector_results: List[Dict], keyword_results: List[Dict],
                             query: str) -> List[Dict]:
        """Combine and rerank results from vector and keyword search with exact match boosting"""

        try:
            # Create a combined results dictionary to avoid duplicates
            combined_dict = {}

            # Check for exact matches in query (like ASC codes)
            exact_match_patterns = re.findall(r'\b(ASC\s+\d{3}-\d{2}-\d{2}-\d{1,2})\b', query, re.IGNORECASE)
            has_exact_patterns = len(exact_match_patterns) > 0

            # Normalize scores and add vector results
            max_vector_score = max([r['score'] for r in vector_results], default=1.0)
            for result in vector_results:
                node_id = result['node_id']
                normalized_score = result['score'] / max_vector_score if max_vector_score > 0 else 0

                # Check for exact pattern matches in content
                exact_match_bonus = 0.0
                if has_exact_patterns:
                    content = result.get('content', '')
                    for pattern in exact_match_patterns:
                        if re.search(re.escape(pattern), content, re.IGNORECASE):
                            exact_match_bonus += 0.5  # Significant boost for exact matches
                            logger.info(f"Exact match bonus applied for '{pattern}' in {result['metadata'].get('filename', 'Unknown')}")

                combined_dict[node_id] = result.copy()
                combined_dict[node_id]['vector_score'] = normalized_score
                combined_dict[node_id]['keyword_score'] = 0.0
                combined_dict[node_id]['exact_match_bonus'] = exact_match_bonus
                combined_dict[node_id]['combined_score'] = (normalized_score * 0.6) + exact_match_bonus  # Vector + exact match bonus

            # Normalize scores and add/update keyword results
            max_keyword_score = max([r['score'] for r in keyword_results], default=1.0)
            for result in keyword_results:
                node_id = result['node_id']
                normalized_score = result['score'] / max_keyword_score if max_keyword_score > 0 else 0

                # Check for exact matches in keyword results too
                exact_match_bonus = 0.0
                if has_exact_patterns:
                    content = result.get('content', '')
                    for pattern in exact_match_patterns:
                        if re.search(re.escape(pattern), content, re.IGNORECASE):
                            exact_match_bonus += 0.5

                if node_id in combined_dict:
                    # Update existing result with keyword score
                    combined_dict[node_id]['keyword_score'] = normalized_score
                    # Recalculate with all components
                    combined_dict[node_id]['combined_score'] = (
                        combined_dict[node_id]['vector_score'] * 0.6 +
                        normalized_score * 0.4 +
                        combined_dict[node_id]['exact_match_bonus']  # Keep existing bonus
                    )
                    combined_dict[node_id]['search_type'] = 'hybrid'
                else:
                    # Add new keyword-only result
                    combined_dict[node_id] = result.copy()
                    combined_dict[node_id]['vector_score'] = 0.0
                    combined_dict[node_id]['keyword_score'] = normalized_score
                    combined_dict[node_id]['exact_match_bonus'] = exact_match_bonus
                    combined_dict[node_id]['combined_score'] = (normalized_score * 0.4) + exact_match_bonus

            # Convert back to list and sort by combined score
            final_results = list(combined_dict.values())
            final_results.sort(key=lambda x: x['combined_score'], reverse=True)

            # Take top results and clean up temporary scoring fields
            doc_limit = self.quality_metrics_manager.get_document_limit(query)
            final_results = final_results[:doc_limit]
            for result in final_results:
                result['score'] = result['combined_score']  # Set final score
                # Keep detailed scores for debugging but rename
                result.pop('combined_score', None)
                # Optionally remove detailed scores to clean up
                # result.pop('vector_score', None)
                # result.pop('keyword_score', None)

            logger.info(f"Hybrid search combined {len(vector_results)} vector + {len(keyword_results)} keyword results into {len(final_results)} final results")

            return final_results

        except Exception as e:
            logger.error("Failed to combine search results", error=str(e))
            # Fallback to vector results only with dynamic limit
            doc_limit = self.quality_metrics_manager.get_document_limit(query)
            return vector_results[:doc_limit]

    def get_diverse_context_documents(self, documents: List[Dict]) -> List[Dict]:
        """Select diverse documents for context to ensure comprehensive coverage"""
        # FIXED: Return all documents for maximum comprehensive coverage
        # The issue was complex deduplication logic that was too aggressive
        logger.info(f"🔍 DIVERSE SELECTION DEBUG: Input={len(documents)}, Output={len(documents)} (returning all)")
        return documents


def create_document_retriever(vector_index=None, bm25_engine=None, relevance_engine=None,
                            metadata_analyzer=None, quality_metrics_manager=None, config=None):
    """Factory function to create a DocumentRetriever instance"""
    return DocumentRetriever(
        vector_index=vector_index,
        bm25_engine=bm25_engine,
        relevance_engine=relevance_engine,
        metadata_analyzer=metadata_analyzer,
        quality_metrics_manager=quality_metrics_manager,
        config=config
    )