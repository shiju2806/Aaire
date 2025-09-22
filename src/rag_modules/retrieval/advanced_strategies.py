"""
Advanced Retrieval Strategies Module
Implements industry-standard retrieval techniques with zero hardcoding
"""

import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from openai import AsyncOpenAI
import structlog

logger = structlog.get_logger()

@dataclass
class QueryDecomposition:
    """Result of query decomposition"""
    original_query: str
    sub_queries: List[Dict[str, Any]]
    strategy: str
    confidence: float

@dataclass
class RetrievalStrategy:
    """Defines a retrieval strategy"""
    name: str
    description: str
    suitable_for: List[str]
    parameters: Dict[str, Any]

class QueryDecomposer:
    """Breaks complex queries into manageable sub-queries"""

    def __init__(self, llm_client: AsyncOpenAI, config: Dict[str, Any]):
        self.llm_client = llm_client
        self.config = config.get('query_decomposition', {})
        self.logger = logger.bind(component="query_decomposer")

    async def analyze_and_decompose(self, query: str) -> QueryDecomposition:
        """Analyze query complexity and decompose if needed"""

        # First, analyze if decomposition is needed
        needs_decomposition = await self._analyze_complexity(query)

        if not needs_decomposition['should_decompose']:
            return QueryDecomposition(
                original_query=query,
                sub_queries=[{"sub_query": query, "priority": 1, "type": "simple"}],
                strategy="single_query",
                confidence=needs_decomposition['confidence']
            )

        # Decompose the query
        sub_queries = await self._decompose_query(query, needs_decomposition)

        return QueryDecomposition(
            original_query=query,
            sub_queries=sub_queries,
            strategy="decomposed",
            confidence=needs_decomposition['confidence']
        )

    async def _analyze_complexity(self, query: str) -> Dict[str, Any]:
        """Determine if query needs decomposition"""

        complexity_prompt = self.config.get('complexity_analysis_prompt', """
Analyze this query to determine if it should be broken down into simpler parts.

Query: "{query}"

Consider:
- Does it ask multiple distinct questions?
- Does it involve multiple concepts that need separate analysis?
- Would breaking it down improve answer quality?

Return JSON:
{{
    "should_decompose": true/false,
    "complexity_score": 0.0-1.0,
    "reasoning": "explanation of decision",
    "confidence": 0.0-1.0,
    "detected_concepts": ["concept1", "concept2", ...],
    "question_count": number
}}
""").format(query=query)

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.config.get('model', 'gpt-4o-mini'),
                messages=[{"role": "user", "content": complexity_prompt}],
                temperature=self.config.get('temperature', 0.1),
                max_tokens=self.config.get('max_tokens', 300),
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            self.logger.info("Query complexity analyzed",
                           should_decompose=result.get('should_decompose'),
                           complexity=result.get('complexity_score'))
            return result

        except Exception as e:
            self.logger.error(f"Complexity analysis failed: {e}")
            return {"should_decompose": False, "confidence": 0.0}

    async def _decompose_query(self, query: str, analysis: Dict) -> List[Dict[str, Any]]:
        """Break query into sub-queries"""

        decomposition_prompt = self.config.get('decomposition_prompt', """
Break this complex query into simpler, independent sub-queries.

Original Query: "{query}"
Detected Concepts: {concepts}

Guidelines:
- Each sub-query should be answerable independently
- Maintain the intent of the original question
- Order by importance/dependency
- Maximum {max_subqueries} sub-queries

Return JSON array:
[
    {{
        "sub_query": "specific question",
        "priority": 1-5,
        "type": "factual|calculation|comparison|explanation",
        "dependency": null or "depends on query N",
        "rationale": "why this sub-query is needed"
    }}
]
""").format(
            query=query,
            concepts=analysis.get('detected_concepts', []),
            max_subqueries=self.config.get('max_subqueries', 3)
        )

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.config.get('model', 'gpt-4o-mini'),
                messages=[{"role": "user", "content": decomposition_prompt}],
                temperature=self.config.get('temperature', 0.1),
                max_tokens=self.config.get('max_tokens', 500),
                response_format={"type": "json_object"}
            )

            sub_queries = json.loads(response.choices[0].message.content)
            self.logger.info("Query decomposed", sub_query_count=len(sub_queries))
            return sub_queries

        except Exception as e:
            self.logger.error(f"Query decomposition failed: {e}")
            return [{"sub_query": query, "priority": 1, "type": "simple"}]

class ParentChildRetriever:
    """Expands retrieved chunks with surrounding context"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('parent_child', {})
        self.logger = logger.bind(component="parent_child_retriever")

    async def expand_chunks(self, retrieved_chunks: List[Dict], document_store) -> List[Dict]:
        """Expand chunks with parent/child context"""

        expanded_chunks = []
        expansion_size = self.config.get('expansion_size', 2)

        for chunk in retrieved_chunks:
            try:
                expanded = await self._expand_single_chunk(chunk, document_store, expansion_size)
                expanded_chunks.append(expanded)
            except Exception as e:
                self.logger.warning(f"Failed to expand chunk: {e}")
                expanded_chunks.append(chunk)  # Use original if expansion fails

        return expanded_chunks

    async def _expand_single_chunk(self, chunk: Dict, document_store, expansion_size: int) -> Dict:
        """Expand a single chunk with surrounding context"""

        # Get chunk metadata
        doc_id = chunk.get('metadata', {}).get('document_id')
        chunk_index = chunk.get('metadata', {}).get('chunk_index', 0)

        if not doc_id:
            return chunk  # Can't expand without document ID

        # Calculate expansion range
        start_chunk = max(0, chunk_index - expansion_size)
        end_chunk = chunk_index + expansion_size + 1

        # Retrieve surrounding chunks
        surrounding_chunks = await self._get_document_chunks(
            document_store, doc_id, start_chunk, end_chunk
        )

        if not surrounding_chunks:
            return chunk

        # Merge chunks with boundaries
        expanded_content = self._merge_chunks_with_boundaries(surrounding_chunks, chunk_index)

        # Create expanded chunk
        expanded_chunk = chunk.copy()
        expanded_chunk['content'] = expanded_content
        expanded_chunk['metadata']['expanded'] = True
        expanded_chunk['metadata']['expansion_range'] = f"{start_chunk}-{end_chunk}"

        self.logger.debug("Chunk expanded",
                         original_size=len(chunk['content']),
                         expanded_size=len(expanded_content))

        return expanded_chunk

    async def _get_document_chunks(self, document_store, doc_id: str, start: int, end: int) -> List[Dict]:
        """Retrieve chunks from document store"""
        # This would interface with your document storage system
        # Implementation depends on your storage backend
        try:
            # Placeholder - you'd implement this based on your storage
            chunks = await document_store.get_chunks_by_range(doc_id, start, end)
            return chunks
        except Exception as e:
            self.logger.error(f"Failed to retrieve chunks: {e}")
            return []

    def _merge_chunks_with_boundaries(self, chunks: List[Dict], target_chunk_index: int) -> str:
        """Merge chunks with clear boundaries"""

        if not chunks:
            return ""

        merged_content = []
        boundary_marker = self.config.get('boundary_marker', '\n--- CHUNK BOUNDARY ---\n')

        for i, chunk in enumerate(chunks):
            content = chunk.get('content', '')
            chunk_idx = chunk.get('metadata', {}).get('chunk_index', i)

            # Mark the target chunk
            if chunk_idx == target_chunk_index:
                content = f"[TARGET CHUNK]\n{content}\n[/TARGET CHUNK]"

            merged_content.append(content)

        return boundary_marker.join(merged_content)

class AdaptiveRetriever:
    """Routes queries to optimal retrieval strategies"""

    def __init__(self, llm_client: AsyncOpenAI, config: Dict[str, Any]):
        self.llm_client = llm_client
        self.config = config.get('adaptive_retrieval', {})
        self.logger = logger.bind(component="adaptive_retriever")
        self.strategies = self._load_strategies()

    def _load_strategies(self) -> List[RetrievalStrategy]:
        """Load retrieval strategies from configuration"""

        strategy_configs = self.config.get('strategies', [])
        strategies = []

        for strategy_config in strategy_configs:
            strategy = RetrievalStrategy(
                name=strategy_config['name'],
                description=strategy_config['description'],
                suitable_for=strategy_config['suitable_for'],
                parameters=strategy_config.get('parameters', {})
            )
            strategies.append(strategy)

        return strategies

    async def select_strategy(self, query: str, context: Dict = None) -> RetrievalStrategy:
        """Select optimal retrieval strategy for query"""

        if not self.strategies:
            # Return default strategy if none configured
            return RetrievalStrategy(
                name="default",
                description="Default hybrid retrieval",
                suitable_for=["general"],
                parameters={}
            )

        # Analyze query to determine best strategy
        analysis = await self._analyze_query_characteristics(query, context)

        # Select strategy based on analysis
        selected_strategy = self._match_strategy(analysis)

        self.logger.info("Retrieval strategy selected",
                        strategy=selected_strategy.name,
                        query_type=analysis.get('query_type'))

        return selected_strategy

    async def _analyze_query_characteristics(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """Analyze query to determine optimal retrieval approach"""

        # Build analysis prompt dynamically from config
        available_strategies = [s.name for s in self.strategies]
        strategy_descriptions = {s.name: s.description for s in self.strategies}

        analysis_prompt = self.config.get('analysis_prompt', """
Analyze this query to determine the optimal retrieval strategy.

Query: "{query}"
Context: {context}

Available Strategies:
{strategies}

Consider:
- Query complexity and type
- Information need (factual, procedural, comparative)
- Required precision vs. recall
- Domain specificity

Return JSON:
{{
    "query_type": "factual|calculation|comparison|explanation|complex",
    "complexity": "simple|moderate|complex",
    "precision_required": "high|medium|low",
    "domain_specificity": "high|medium|low",
    "recommended_strategy": "strategy_name",
    "reasoning": "explanation of choice",
    "confidence": 0.0-1.0
}}
""").format(
            query=query,
            context=context or {},
            strategies="\n".join([f"- {name}: {desc}" for name, desc in strategy_descriptions.items()])
        )

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.config.get('model', 'gpt-4o-mini'),
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=self.config.get('temperature', 0.1),
                max_tokens=self.config.get('max_tokens', 400),
                response_format={"type": "json_object"}
            )

            analysis = json.loads(response.choices[0].message.content)
            return analysis

        except Exception as e:
            self.logger.error(f"Query analysis failed: {e}")
            return {"query_type": "general", "recommended_strategy": self.strategies[0].name}

    def _match_strategy(self, analysis: Dict[str, Any]) -> RetrievalStrategy:
        """Match analysis results to best strategy"""

        recommended = analysis.get('recommended_strategy')

        # Try to find recommended strategy
        for strategy in self.strategies:
            if strategy.name == recommended:
                return strategy

        # Fallback: match by query type
        query_type = analysis.get('query_type', 'general')
        for strategy in self.strategies:
            if query_type in strategy.suitable_for:
                return strategy

        # Final fallback: return first strategy
        return self.strategies[0]

class AdvancedRetrievalManager:
    """Orchestrates all advanced retrieval strategies"""

    def __init__(self, llm_client: AsyncOpenAI, config: Dict[str, Any]):
        self.llm_client = llm_client
        self.config = config
        self.logger = logger.bind(component="advanced_retrieval_manager")

        # Initialize components
        self.query_decomposer = QueryDecomposer(llm_client, config)
        self.parent_child_retriever = ParentChildRetriever(config)
        self.adaptive_retriever = AdaptiveRetriever(llm_client, config)

        self.enabled = config.get('enabled', True)

    async def enhanced_retrieval(
        self,
        query: str,
        base_retrieval_func,
        document_store=None,
        context: Dict = None
    ) -> Dict[str, Any]:
        """Main entry point for advanced retrieval"""

        if not self.enabled:
            # Use base retrieval if advanced features disabled
            results = await base_retrieval_func(query)
            return {"documents": results, "strategy": "base"}

        try:
            # Step 1: Query decomposition
            decomposition = await self.query_decomposer.analyze_and_decompose(query)

            # Step 2: Strategy selection
            strategy = await self.adaptive_retriever.select_strategy(query, context)

            # Step 3: Execute retrieval based on strategy
            if decomposition.strategy == "decomposed":
                documents = await self._handle_decomposed_retrieval(
                    decomposition, base_retrieval_func, strategy
                )
            else:
                documents = await base_retrieval_func(query)

            # Step 4: Parent-child expansion if enabled
            if (self.config.get('parent_child', {}).get('enabled', False) and
                document_store is not None):
                documents = await self.parent_child_retriever.expand_chunks(
                    documents, document_store
                )

            return {
                "documents": documents,
                "strategy": strategy.name,
                "decomposition": decomposition,
                "metadata": {
                    "query_decomposed": decomposition.strategy == "decomposed",
                    "context_expanded": document_store is not None,
                    "strategy_confidence": strategy.parameters.get('confidence', 1.0)
                }
            }

        except Exception as e:
            self.logger.error(f"Advanced retrieval failed: {e}")
            # Fallback to base retrieval
            results = await base_retrieval_func(query)
            return {"documents": results, "strategy": "fallback"}

    async def _handle_decomposed_retrieval(
        self,
        decomposition: QueryDecomposition,
        base_retrieval_func,
        strategy: RetrievalStrategy
    ) -> List[Dict]:
        """Handle retrieval for decomposed queries"""

        all_documents = []

        # Retrieve for each sub-query
        for sub_query_info in decomposition.sub_queries:
            sub_query = sub_query_info['sub_query']

            try:
                sub_results = await base_retrieval_func(sub_query)

                # Add metadata about sub-query
                for doc in sub_results:
                    # Ensure doc is a dictionary and has metadata
                    if isinstance(doc, dict) and 'metadata' in doc and isinstance(doc['metadata'], dict):
                        doc['metadata']['sub_query'] = sub_query
                        doc['metadata']['sub_query_priority'] = sub_query_info.get('priority', 1)
                        doc['metadata']['sub_query_type'] = sub_query_info.get('type', 'general')
                    else:
                        self.logger.warning(f"Invalid document format in sub-query results: {type(doc)}")

                all_documents.extend(sub_results)

            except Exception as e:
                self.logger.warning(f"Sub-query retrieval failed for '{sub_query}': {e}")

        # Deduplicate and rerank
        deduplicated = self._deduplicate_documents(all_documents)
        return deduplicated[:self.config.get('max_results', 20)]

    def _deduplicate_documents(self, documents: List[Dict]) -> List[Dict]:
        """Remove duplicate documents from multiple sub-queries"""

        seen_ids = set()
        unique_docs = []

        for doc in documents:
            # Ensure doc is a dictionary before accessing attributes
            if not isinstance(doc, dict):
                self.logger.warning(f"Skipping non-dict document in deduplication: {type(doc)}")
                continue

            doc_id = doc.get('id') or doc.get('metadata', {}).get('doc_id')

            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)
            elif not doc_id:
                # Keep documents without IDs (might be different chunks)
                unique_docs.append(doc)

        return unique_docs