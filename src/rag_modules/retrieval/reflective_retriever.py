"""
Retrieval Reflection System
Implements LLM-based evaluation and iterative improvement of document retrieval quality
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from openai import AsyncOpenAI
import structlog

logger = structlog.get_logger()


@dataclass
class RetrievalReflection:
    """Result of retrieval quality reflection"""
    needs_improvement: bool
    quality_score: float
    confidence: float
    missing_information: List[str]
    suggested_queries: List[str]
    feedback: str


@dataclass
class ReflectiveRetrievalResult:
    """Enhanced retrieval result with reflection metadata"""
    documents: List[Dict[str, Any]]
    reflection_rounds: int
    quality_progression: List[float]
    final_quality_score: float
    improvement_achieved: bool


class RetrievalQualityEvaluator:
    """LLM-based evaluator for retrieval quality"""

    def __init__(self, llm_client: AsyncOpenAI, config: Dict[str, Any]):
        self.llm_client = llm_client
        self.config = config.get('retrieval_reflection', {})
        self.model = "gpt-4o-mini"
        self.logger = logger.bind(component="retrieval_evaluator")

    async def evaluate_retrieval_quality(self, query: str, documents: List[Dict[str, Any]]) -> RetrievalReflection:
        """Evaluate the quality of retrieved documents for a given query"""

        try:
            if not documents:
                return RetrievalReflection(
                    needs_improvement=True,
                    quality_score=0.0,
                    confidence=1.0,
                    missing_information=["No documents retrieved"],
                    suggested_queries=[query],
                    feedback="No documents found for the query"
                )

            # Prepare document summaries for evaluation
            doc_summaries = self._prepare_document_summaries(documents)

            evaluation_prompt = f"""
You are an expert information retrieval evaluator. Analyze whether the retrieved documents contain sufficient, relevant information to answer the user's query.

USER QUERY: {query}

RETRIEVED DOCUMENTS:
{doc_summaries}

Evaluate the retrieval quality and respond in JSON format:
{{
    "needs_improvement": boolean,
    "quality_score": float (0.0-1.0),
    "confidence": float (0.0-1.0),
    "missing_information": [list of specific information gaps],
    "suggested_queries": [list of 1-3 alternative/additional search queries],
    "feedback": "detailed explanation of the evaluation"
}}

Consider:
1. Relevance: Do documents directly address the query?
2. Completeness: Is sufficient information available to provide a comprehensive answer?
3. Specificity: Are documents specific enough or too general?
4. Coverage: Are all aspects of the query covered?

Be precise about what information is missing and suggest specific queries to find it.
"""

            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.1,
                max_tokens=1000
            )

            # Parse the JSON response
            evaluation_text = response.choices[0].message.content.strip()

            # Extract JSON from the response
            json_match = self._extract_json_from_response(evaluation_text)
            if not json_match:
                self.logger.warning("Failed to parse evaluation response", response=evaluation_text)
                return self._create_fallback_reflection(query, documents)

            evaluation_data = json.loads(json_match)

            reflection = RetrievalReflection(
                needs_improvement=evaluation_data.get('needs_improvement', False),
                quality_score=max(0.0, min(1.0, evaluation_data.get('quality_score', 0.5))),
                confidence=max(0.0, min(1.0, evaluation_data.get('confidence', 0.5))),
                missing_information=evaluation_data.get('missing_information', []),
                suggested_queries=evaluation_data.get('suggested_queries', [query]),
                feedback=evaluation_data.get('feedback', 'No detailed feedback available')
            )

            self.logger.info("Retrieval quality evaluated",
                           query=query[:50],
                           quality_score=reflection.quality_score,
                           needs_improvement=reflection.needs_improvement,
                           missing_info_count=len(reflection.missing_information))

            return reflection

        except Exception as e:
            self.logger.error("Retrieval evaluation failed", error=str(e))
            return self._create_fallback_reflection(query, documents)

    def _prepare_document_summaries(self, documents: List[Dict[str, Any]]) -> str:
        """Prepare concise summaries of documents for evaluation"""
        summaries = []

        for i, doc in enumerate(documents[:10]):  # Limit to top 10 for evaluation
            content = doc.get('content', '')
            title = doc.get('title', f'Document {i+1}')

            # Truncate content for evaluation
            summary_content = content[:300] + "..." if len(content) > 300 else content

            summaries.append(f"Document {i+1}: {title}\nContent: {summary_content}\n")

        return "\n".join(summaries)

    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """Extract JSON from LLM response"""
        try:
            # Try to find JSON content between braces
            start = response.find('{')
            end = response.rfind('}') + 1

            if start != -1 and end > start:
                return response[start:end]

            return None
        except Exception:
            return None

    def _create_fallback_reflection(self, query: str, documents: List[Dict[str, Any]]) -> RetrievalReflection:
        """Create fallback reflection when evaluation fails"""
        return RetrievalReflection(
            needs_improvement=len(documents) < self.config.get('min_docs_for_reflection', 3),
            quality_score=0.5,
            confidence=0.5,
            missing_information=["Unable to evaluate missing information"],
            suggested_queries=[query],
            feedback="Evaluation failed, using fallback assessment"
        )


class ReflectiveRetriever:
    """Enhanced retriever with reflection capabilities"""

    def __init__(self, base_retriever, llm_client: AsyncOpenAI, config: Dict[str, Any]):
        self.base_retriever = base_retriever
        self.evaluator = RetrievalQualityEvaluator(llm_client, config)
        self.config = config.get('retrieval_reflection', {})
        self.logger = logger.bind(component="reflective_retriever")

    async def retrieve_with_reflection(self, query: str, **kwargs) -> ReflectiveRetrievalResult:
        """Perform retrieval with reflection and iterative improvement"""

        if not self.config.get('enabled', True):
            # If reflection is disabled, use base retrieval
            documents = await self._perform_base_retrieval(query, **kwargs)
            return ReflectiveRetrievalResult(
                documents=documents,
                reflection_rounds=0,
                quality_progression=[1.0],
                final_quality_score=1.0,
                improvement_achieved=False
            )

        documents = []
        quality_progression = []
        max_rounds = self.config.get('max_reflection_rounds', 2)
        quality_threshold = self.config.get('quality_threshold', 0.7)

        for round_num in range(max_rounds + 1):  # +1 for initial retrieval

            if round_num == 0:
                # Initial retrieval
                current_docs = await self._perform_base_retrieval(query, **kwargs)
                search_query = query
            else:
                # Reflection-guided retrieval
                reflection = await self.evaluator.evaluate_retrieval_quality(query, documents)
                quality_progression.append(reflection.quality_score)

                if not reflection.needs_improvement or reflection.quality_score >= quality_threshold:
                    self.logger.info("Retrieval quality sufficient, stopping reflection",
                                   round=round_num, quality_score=reflection.quality_score)
                    break

                # Generate improved search queries
                improved_queries = self._generate_improved_queries(query, reflection)
                current_docs = await self._perform_targeted_retrieval(improved_queries, **kwargs)

                self.logger.info("Reflection round completed",
                               round=round_num,
                               quality_score=reflection.quality_score,
                               new_docs=len(current_docs),
                               missing_info=reflection.missing_information[:3])

            # Merge and deduplicate documents
            documents = self._merge_documents(documents, current_docs)

        # Final quality assessment
        if documents:
            final_reflection = await self.evaluator.evaluate_retrieval_quality(query, documents)
            final_quality = final_reflection.quality_score
        else:
            final_quality = 0.0

        if not quality_progression:
            quality_progression = [final_quality]

        improvement_achieved = len(quality_progression) > 1 and final_quality > quality_progression[0]

        result = ReflectiveRetrievalResult(
            documents=documents,
            reflection_rounds=len(quality_progression) - 1,
            quality_progression=quality_progression,
            final_quality_score=final_quality,
            improvement_achieved=improvement_achieved
        )

        self.logger.info("Reflective retrieval completed",
                       total_docs=len(documents),
                       reflection_rounds=result.reflection_rounds,
                       final_quality=final_quality,
                       improvement_achieved=improvement_achieved)

        return result

    async def _perform_base_retrieval(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Perform base retrieval using existing retriever"""
        try:
            # Use the existing retrieval method
            if hasattr(self.base_retriever, 'retrieve_documents'):
                return await self.base_retriever.retrieve_documents(query, **kwargs)
            elif hasattr(self.base_retriever, 'hybrid_search'):
                return await self.base_retriever.hybrid_search(query, **kwargs)
            else:
                self.logger.warning("No compatible retrieval method found")
                return []
        except Exception as e:
            self.logger.error("Base retrieval failed", error=str(e))
            return []

    async def _perform_targeted_retrieval(self, queries: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Perform targeted retrieval with multiple queries"""
        all_docs = []

        for query in queries:
            try:
                docs = await self._perform_base_retrieval(query, **kwargs)
                all_docs.extend(docs)
            except Exception as e:
                self.logger.error("Targeted retrieval failed", query=query, error=str(e))

        return all_docs

    def _generate_improved_queries(self, original_query: str, reflection: RetrievalReflection) -> List[str]:
        """Generate improved search queries based on reflection feedback"""
        improved_queries = []

        # Use suggested queries from reflection
        if reflection.suggested_queries:
            improved_queries.extend(reflection.suggested_queries[:2])  # Limit to 2 queries

        # Generate targeted queries for missing information
        if reflection.missing_information:
            for missing_info in reflection.missing_information[:2]:  # Limit to prevent over-expansion
                # Create focused query
                focused_query = f"{original_query} {missing_info}"
                improved_queries.append(focused_query)

        # Remove duplicates and ensure we have at least the original query
        improved_queries = list(dict.fromkeys(improved_queries))  # Remove duplicates preserving order
        if not improved_queries:
            improved_queries = [original_query]

        return improved_queries[:3]  # Limit to 3 queries max

    def _merge_documents(self, existing_docs: List[Dict[str, Any]],
                        new_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge and deduplicate documents"""
        merged_docs = existing_docs.copy()

        # Simple deduplication based on content similarity
        existing_contents = {doc.get('content', '')[:100] for doc in existing_docs}

        for doc in new_docs:
            doc_content_preview = doc.get('content', '')[:100]
            if doc_content_preview not in existing_contents:
                merged_docs.append(doc)
                existing_contents.add(doc_content_preview)

        return merged_docs


def create_reflective_retriever(base_retriever, llm_client: AsyncOpenAI, config: Dict[str, Any]) -> ReflectiveRetriever:
    """Factory function to create reflective retriever"""
    return ReflectiveRetriever(base_retriever, llm_client, config)