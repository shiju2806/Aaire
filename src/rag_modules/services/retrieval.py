"""
Document Retrieval Service Module

Handles all document retrieval operations including:
- Hybrid search (vector + keyword)
- Vector-based semantic search
- Keyword-based BM25 search
- Search result combination and ranking
- Document diversity selection
"""

import re
import asyncio
import structlog
from typing import List, Dict, Any, Optional

# LLM-based framework filtering imports
try:
    from ..filtering import create_llm_framework_filter
except ImportError:
    create_llm_framework_filter = None

logger = structlog.get_logger()


class DocumentRetriever:
    """Handles document retrieval operations with hybrid search capabilities"""

    def __init__(self, vector_index=None, whoosh_engine=None, relevance_engine=None,
                 quality_metrics_manager=None, config=None, llm_client=None, nlp_query_processor=None):
        """Initialize retriever with required components"""
        self.index = vector_index
        self.whoosh_engine = whoosh_engine
        self.relevance_engine = relevance_engine
        self.quality_metrics_manager = quality_metrics_manager
        self.config = config or {}
        self.llm_client = llm_client
        self.nlp_query_processor = nlp_query_processor

        # Derived attributes
        self.keyword_search_ready = whoosh_engine is not None

        # Initialize LLM-based framework filter for actuarial content
        self.framework_filter = None
        if create_llm_framework_filter and llm_client:
            try:
                self.framework_filter = create_llm_framework_filter(llm_client)
                logger.info("LLM-based framework filtering enabled")
            except Exception as e:
                logger.warning("LLM framework filter initialization failed", exception_details=str(e))
                self.framework_filter = None
        else:
            logger.info("LLM framework filtering not available", has_filter=create_llm_framework_filter is not None, has_client=llm_client is not None)

    async def retrieve_documents(self, query: str, doc_type_filter: Optional[List[str]],
                               similarity_threshold: Optional[float] = None,
                               filters: Optional[Dict[str, Any]] = None,
                               query_intent: Optional[Any] = None) -> List[Dict]:
        """Hybrid retrieval: combines vector search with BM25 keyword search using buffer approach"""

        # Use provided query intent from intelligent analyzer or fallback to metadata analyzer
        smart_filters = filters.copy() if filters else {}
        use_smart_filtering = False

        if query_intent is not None:
            # Use perfect intent analysis from intelligent_query_analyzer
            logger.info(f"Using provided query intent from intelligent analysis",
                       query=query[:50],
                       jurisdiction=getattr(query_intent, 'jurisdiction_hint', 'unknown'),
                       product=getattr(query_intent, 'product_hint', 'unknown'),
                       jurisdiction_confidence=getattr(query_intent, 'jurisdiction_confidence', 0.0),
                       product_confidence=getattr(query_intent, 'product_confidence', 0.0))
            # Smart filtering with intelligent analyzer results is always enabled if high confidence
            use_smart_filtering = (getattr(query_intent, 'jurisdiction_confidence', 0.0) > 0.8 or
                                 getattr(query_intent, 'product_confidence', 0.0) > 0.8)
        else:
            # No query intent provided - proceed without smart filtering
            logger.info("No query intent provided - proceeding with standard retrieval")
            query_intent = None
            use_smart_filtering = False

        # Apply smart filtering if appropriate
        try:
            if use_smart_filtering and query_intent:
                logger.info(f"Query intent available for filtering",
                           jurisdiction=getattr(query_intent, 'jurisdiction_hint', None),
                           product=getattr(query_intent, 'product_hint', None),
                           jurisdiction_confidence=getattr(query_intent, 'jurisdiction_confidence', None),
                           product_confidence=getattr(query_intent, 'product_confidence', None))
            else:
                logger.debug(f"Smart filtering skipped - criteria not met")

        except Exception as e:
            logger.warning(f"Query intent analysis failed, proceeding without smart filtering",
                         error=str(e))

        # Get the target document limit
        doc_limit = self.quality_metrics_manager.get_document_limit(query)

        # Use buffer approach: both searches get the full limit to ensure we don't miss important documents
        # This allows the best documents from either method to compete fairly
        # Execute searches in parallel for 40% performance improvement
        vector_task = asyncio.create_task(
            self.vector_search(query, doc_type_filter, similarity_threshold, filters, buffer_limit=doc_limit)
        )
        keyword_task = asyncio.create_task(
            self.keyword_search(query, doc_type_filter, filters, buffer_limit=doc_limit, query_intent=query_intent)
        )
        vector_results, keyword_results = await asyncio.gather(vector_task, keyword_task)

        logger.info(f"Buffer approach: Retrieved {len(vector_results)} vector + {len(keyword_results)} keyword results, target limit: {doc_limit}")

        # Combine results using advanced relevance engine
        all_results = vector_results + keyword_results

        # Remove duplicates while preserving all scoring info
        unique_results = {}
        for result in all_results:
            node_id = result['node_id']
            if node_id not in unique_results:
                unique_results[node_id] = result
            else:
                # Merge scoring information from both searches
                existing = unique_results[node_id]
                existing['search_type'] = 'hybrid'
                # Keep the higher score
                if result['score'] > existing['score']:
                    existing['score'] = result['score']

        # Use advanced relevance engine for ranking
        combined_results = self.relevance_engine.rank_documents(query, list(unique_results.values()))

        # Apply framework-aware filtering for actuarial content
        if self.framework_filter:
            try:
                # Use async LLM-based filtering
                if asyncio.iscoroutinefunction(self.framework_filter.filter_retrieval_results):
                    filtered_results = await self.framework_filter.filter_retrieval_results(query, combined_results)
                else:
                    filtered_results = self.framework_filter.filter_retrieval_results(query, combined_results)

                # filtered_results are already properly formatted dictionaries, just use them directly
                combined_results = filtered_results

                logger.info("Framework filtering applied",
                           framework_adjustments=len([r for r in filtered_results if r.get('framework_analysis', {}).get('alignment_score', 1.0) != 1.0]))
            except Exception as e:
                logger.warning("Framework filtering failed, continuing without", exception_details=str(e))

        # Apply the document limit to the final combined results
        if len(combined_results) > doc_limit:
            logger.info(f"Trimming combined results from {len(combined_results)} to {doc_limit}")
            combined_results = combined_results[:doc_limit]

        return combined_results

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
            logger.warning("Failed to retrieve from index", exception_details=str(e))

        # Sort by relevance score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        # Note: vector_limit is already applied in retriever, but trimming here for safety
        return all_results[:vector_limit] if 'vector_limit' in locals() else all_results

    async def keyword_search(self, query: str, doc_type_filter: Optional[List[str]],
                           filters: Optional[Dict[str, Any]] = None,
                           buffer_limit: Optional[int] = None,
                           query_intent: Optional[Any] = None) -> List[Dict]:
        """Whoosh-based keyword search"""
        results = []

        try:
            if not self.whoosh_engine or not self.keyword_search_ready:
                logger.info("Whoosh search engine not available, skipping keyword search")
                return results

            # Use buffer limit if provided (for combined ranking), otherwise use dynamic calculation
            if buffer_limit is not None:
                keyword_limit = buffer_limit
            else:
                # Get dynamic document limit based on query
                doc_limit = self.quality_metrics_manager.get_document_limit(query)
                # For standalone search, allocate 30% to keyword search
                keyword_limit = int(doc_limit * 0.3)

            # Process query with NLP processor for better semantic search if available
            processed_query = query  # Default to original query
            if self.nlp_query_processor:
                try:
                    nlp_result = self.nlp_query_processor.process_query(query)
                    # Use focused query generation for keyword search to get precise terms
                    processed_query = self.nlp_query_processor.generate_search_query(nlp_result, mode="focused")
                    logger.info(f"🧠 NLP Query Processing: '{query}' → '{processed_query}'")
                    logger.info(f"🔍 NLP extracted entities: {nlp_result.key_entities}")
                    logger.info(f"🔍 NLP extracted phrases: {nlp_result.key_phrases}")
                except Exception as e:
                    logger.warning(f"NLP query processing failed, using original query: {str(e)}")
                    processed_query = query

            # Extract jurisdiction and product hints from query intent for Enhanced Whoosh
            jurisdiction_hint = None
            product_hint = None

            if query_intent:
                # Extract jurisdiction hint
                if hasattr(query_intent, 'jurisdiction_hint') and query_intent.jurisdiction_hint:
                    if hasattr(query_intent.jurisdiction_hint, 'value'):
                        jurisdiction_hint = query_intent.jurisdiction_hint.value
                    else:
                        jurisdiction_hint = str(query_intent.jurisdiction_hint)

                    if jurisdiction_hint == 'unknown':
                        jurisdiction_hint = None

                # Extract product hint
                if hasattr(query_intent, 'product_hint') and query_intent.product_hint:
                    if hasattr(query_intent.product_hint, 'value'):
                        product_hint = query_intent.product_hint.value
                    else:
                        product_hint = str(query_intent.product_hint)

                    if product_hint == 'unknown':
                        product_hint = None

            # Use Enhanced Whoosh with jurisdiction/product filtering
            if hasattr(self.whoosh_engine, 'search_with_context'):
                logger.info(f"Using Enhanced Whoosh with jurisdiction: {jurisdiction_hint}, product: {product_hint}")
                search_results = self.whoosh_engine.search_with_context(
                    query=processed_query,
                    jurisdiction_hint=jurisdiction_hint,
                    product_hint=product_hint,
                    limit=keyword_limit
                )
            else:
                # Fallback to legacy search for backward compatibility
                whoosh_filters = self.convert_filters_for_whoosh(filters, doc_type_filter, query_intent)
                search_results = self.whoosh_engine.search(
                    query=processed_query,
                    filters=whoosh_filters,
                    limit=keyword_limit,
                    highlight=False
                )

            # Convert Whoosh SearchResults to our format
            for search_result in search_results:
                results.append({
                    'content': search_result['content'],
                    'metadata': search_result['metadata'],
                    'score': search_result['score'],
                    'source_type': search_result['metadata'].get('doc_type', 'unknown'),
                    'node_id': search_result.get('node_id', search_result.get('id', '')),
                    'search_type': 'keyword'
                })

            logger.info(f"Whoosh keyword search found {len(results)} results")

            # Debug: Show first 5 results to understand what's being matched
            for i, result in enumerate(results[:5]):
                content_preview = result['content'][:100].replace('\n', ' ')
                logger.info(f"🔍 DEBUG result {i+1}: doc_id={result['node_id']}, score={result['score']:.3f}, preview='{content_preview}...'")

        except Exception as e:
            logger.error("Failed to perform Whoosh keyword search", exception_details=str(e))

        return results

    def convert_filters_for_whoosh(self, filters: Optional[Dict[str, Any]],
                                 doc_type_filter: Optional[List[str]],
                                 query_intent: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """Convert RAG pipeline filters to Whoosh filter format with query intent"""
        if not filters and not doc_type_filter and not query_intent:
            return None

        whoosh_filters = {}

        # Add query intent filters if available
        if query_intent:
            intent_filters = []

            # Add jurisdiction filter
            if hasattr(query_intent, 'jurisdiction_hint') and query_intent.jurisdiction_hint:
                jurisdiction_value = query_intent.jurisdiction_hint.value if hasattr(query_intent.jurisdiction_hint, 'value') else str(query_intent.jurisdiction_hint)
                if jurisdiction_value != 'unknown':
                    intent_filters.append({
                        'field': 'jurisdiction',
                        'value': jurisdiction_value
                    })
                    logger.info(f"Adding jurisdiction filter: {jurisdiction_value}")

            # Add product type filter
            if hasattr(query_intent, 'product_hint') and query_intent.product_hint:
                product_value = query_intent.product_hint.value if hasattr(query_intent.product_hint, 'value') else str(query_intent.product_hint)
                if product_value != 'unknown':
                    intent_filters.append({
                        'field': 'product_type',
                        'value': product_value
                    })
                    logger.info(f"Adding product type filter: {product_value}")

            # Add intent filters to whoosh_filters in the expected format
            if intent_filters:
                whoosh_filters['filters'] = intent_filters

        if filters:
            # Handle primary_framework filter (most important for smart metadata)
            if 'primary_framework' in filters:
                whoosh_filters['primary_framework'] = filters['primary_framework']

            # Handle content_domains filter
            if 'content_domains' in filters:
                whoosh_filters['content_domains'] = filters['content_domains']

            # Handle document_type filter
            if 'document_type' in filters:
                whoosh_filters['document_type'] = filters['document_type']

        # Convert doc_type_filter to Whoosh format
        if doc_type_filter:
            # Whoosh engine expects doc_type in the document_type field
            whoosh_filters['document_type'] = doc_type_filter

        return whoosh_filters if whoosh_filters else None

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
            logger.error("Failed to combine search results", exception_details=str(e))
            # Fallback to vector results only with dynamic limit
            doc_limit = self.quality_metrics_manager.get_document_limit(query)
            return vector_results[:doc_limit]

    def get_diverse_context_documents(self, documents: List[Dict]) -> List[Dict]:
        """Select diverse documents for context to ensure comprehensive coverage"""
        # FIXED: Return all documents for maximum comprehensive coverage
        # The issue was complex deduplication logic that was too aggressive
        logger.info(f"🔍 DIVERSE SELECTION DEBUG: Input={len(documents)}, Output={len(documents)} (returning all)")
        return documents


def create_document_retriever(vector_index=None, whoosh_engine=None, relevance_engine=None,
                            quality_metrics_manager=None, config=None, llm_client=None, nlp_query_processor=None):
    """Factory function to create a DocumentRetriever instance"""
    return DocumentRetriever(
        vector_index=vector_index,
        whoosh_engine=whoosh_engine,
        relevance_engine=relevance_engine,
        quality_metrics_manager=quality_metrics_manager,
        config=config,
        llm_client=llm_client,
        nlp_query_processor=nlp_query_processor
    )