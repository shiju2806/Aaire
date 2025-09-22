"""
LLM-based Framework Detection for Actuarial Content
Uses AI to understand framework context semantically without hardcoded patterns
"""

import json
import structlog
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from openai import AsyncOpenAI

logger = structlog.get_logger()

@dataclass
class FrameworkDetection:
    """Result of LLM framework detection"""
    primary_framework: Optional[str]
    confidence: float
    frameworks_detected: List[str]
    reasoning: str
    technical_concepts: List[str]
    regulatory_context: Optional[str]

class LLMFrameworkDetector:
    """
    Uses LLM to semantically understand framework context
    No hardcoded patterns - relies on AI understanding
    """

    def __init__(self, llm_client: AsyncOpenAI, model: str = "gpt-4o-mini"):
        self.llm_client = llm_client
        self.model = model

    async def detect_framework(self, text: str) -> FrameworkDetection:
        """
        Use LLM to understand the actuarial/accounting framework context
        """
        try:
            # Create a prompt that asks the LLM to analyze framework context
            system_prompt = """You are an expert actuarial and accounting framework analyzer.
Analyze the given text and identify which regulatory/accounting framework it relates to.

Common frameworks include:
- US STAT (US Statutory/NAIC regulations)
- IFRS (International Financial Reporting Standards)
- US GAAP (Generally Accepted Accounting Principles)
- Solvency II (European insurance regulation)
- Other regional frameworks

Also identify:
1. Technical concepts mentioned (e.g., reserve types, valuation methods)
2. Regulatory context (e.g., specific standards, regulations)
3. Your confidence level (0.0 to 1.0)

Return your analysis as JSON with this structure:
{
    "primary_framework": "framework_name or null",
    "confidence": 0.0-1.0,
    "frameworks_detected": ["list", "of", "frameworks"],
    "reasoning": "Brief explanation of why you identified this framework",
    "technical_concepts": ["list", "of", "technical", "terms"],
    "regulatory_context": "specific regulation or null"
}

Be precise and base your analysis on the actual content, not assumptions."""

            user_prompt = f"Analyze this text for actuarial/accounting framework context:\n\n{text[:2000]}"  # Limit text length

            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistency
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            return FrameworkDetection(
                primary_framework=result.get("primary_framework"),
                confidence=float(result.get("confidence", 0.0)),
                frameworks_detected=result.get("frameworks_detected", []),
                reasoning=result.get("reasoning", ""),
                technical_concepts=result.get("technical_concepts", []),
                regulatory_context=result.get("regulatory_context")
            )

        except Exception as e:
            logger.warning(f"LLM framework detection failed: {e}")
            # Return neutral result on failure
            return FrameworkDetection(
                primary_framework=None,
                confidence=0.0,
                frameworks_detected=[],
                reasoning="Detection failed",
                technical_concepts=[],
                regulatory_context=None
            )

class LLMEnhancedFrameworkFilter:
    """
    Framework filtering using LLM understanding instead of hardcoded patterns
    """

    def __init__(self, llm_client: AsyncOpenAI, model: str = "gpt-4o-mini"):
        self.detector = LLMFrameworkDetector(llm_client, model)
        self.llm_client = llm_client
        self.model = model

    async def filter_retrieval_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply framework-aware filtering using batch LLM analysis - MUCH FASTER
        """
        if not results:
            return results

        # Quick framework detection on query only
        query_detection = await self.detector.detect_framework(query)

        if not query_detection.primary_framework:
            logger.info("No framework detected in query, skipping framework filtering")
            return results

        logger.info(f"Query framework: {query_detection.primary_framework}")

        # BATCH PROCESS all documents in one LLM call instead of 20+ individual calls
        batch_results = await self._batch_score_documents(
            query,
            query_detection.primary_framework,
            results
        )

        return batch_results

    async def _batch_score_documents(
        self,
        query: str,
        query_framework: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Score all documents in one batch call - 20x faster than individual calls
        """
        try:
            # Build batch analysis prompt with all documents
            docs_text = ""
            for i, result in enumerate(results[:10]):  # Limit to top 10 for speed
                content = result.get('content', '')[:200]  # First 200 chars only
                docs_text += f"\nDoc {i+1}: {content}"

            prompt = f"""Query needs {query_framework} framework information.
Query: {query[:150]}

Rate each document's relevance for this {query_framework} query.
Documents:{docs_text}

Return JSON array with scores 0.0-2.0:
[1.2, 0.8, 1.5, ...]

Consider framework match and content relevance only."""

            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
                response_format={"type": "json_object"}
            )

            scores_data = json.loads(response.choices[0].message.content)
            scores = scores_data.get('scores', [1.0] * len(results))

            # Apply scores to results
            for i, result in enumerate(results):
                score = scores[i] if i < len(scores) else 1.0
                original_score = result.get('score', 0.0)
                result['original_score'] = original_score
                result['score'] = original_score * score

            # Sort by adjusted score
            results.sort(key=lambda x: x['score'], reverse=True)
            return results

        except Exception as e:
            logger.warning(f"Batch scoring failed: {e}")
            return results  # Return unchanged on failure

def create_llm_framework_filter(llm_client: AsyncOpenAI, model: str = "gpt-4o-mini") -> LLMEnhancedFrameworkFilter:
    """Factory function to create LLM-based framework filter"""
    return LLMEnhancedFrameworkFilter(llm_client, model)