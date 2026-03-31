"""
Quality Metrics Manager for RAG Pipeline

This module provides quality assessment and metrics calculation for RAG responses,
including confidence scores, similarity thresholds, and document limits.
"""

import re
import math
from typing import List, Dict, Any, Optional
import structlog

from ...providers.config_loader import get_config, get_nested

logger = structlog.get_logger()


class QualityMetricsManager:
    """
    Manages quality metrics calculation and assessment for RAG pipeline responses.

    This class provides methods for:
    - Calculating comprehensive quality metrics for responses
    - Determining optimal similarity thresholds based on query type
    - Computing dynamic document limits based on query complexity
    - Calculating confidence scores for retrieved documents
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the QualityMetricsManager.

        Args:
            config: Configuration dictionary containing retrieval and quality settings
        """
        self.config = config or {}

    def calculate_quality_metrics(self, query: str, response: str, retrieved_docs: List[Dict], citations: List[Dict]) -> Dict[str, float]:
        """
        Calculate automated quality metrics for the response.

        This method evaluates multiple dimensions of response quality including:
        - Citation coverage: How well the response is supported by sources
        - Response length appropriateness: Not too short, not too long
        - Query-response relevance: Basic keyword overlap analysis
        - Source quality: Average similarity scores of retrieved documents
        - Response completeness: Structured content and specific details

        Args:
            query: The user's input query
            response: The generated response text
            retrieved_docs: List of retrieved documents with scores
            citations: List of citations used in the response

        Returns:
            Dict containing individual metric scores and overall quality score
        """
        try:
            # Initialize metrics
            metrics = {}

            # 1. Citation Coverage - How well the response is supported by sources
            if retrieved_docs:
                # Count how many retrieved docs are actually cited
                cited_docs = len(citations) if citations else 0
                total_docs = len(retrieved_docs)
                metrics['citation_coverage'] = cited_docs / total_docs if total_docs > 0 else 0.0
            else:
                metrics['citation_coverage'] = 0.0

            # 2. Response Length Appropriateness - Not too short, not too long
            response_words = len(response.split())
            if response_words < 20:
                metrics['length_score'] = 0.3  # Too short
            elif response_words > 500:
                metrics['length_score'] = 0.7  # Might be too long
            else:
                metrics['length_score'] = 1.0  # Appropriate length

            # 3. Query-Response Relevance - Basic keyword overlap
            query_words = set(query.lower().split())
            response_words_set = set(response.lower().split())

            # Remove common words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can'}
            query_keywords = query_words - stop_words
            response_keywords = response_words_set - stop_words

            if query_keywords:
                overlap = len(query_keywords & response_keywords)
                metrics['keyword_relevance'] = overlap / len(query_keywords)
            else:
                metrics['keyword_relevance'] = 0.5  # Neutral if no keywords

            # 4. Source Quality - Normalized similarity scores of retrieved docs
            #    Raw cosine scores from text-embedding-3-large are low (0.02-0.10 for relevant docs).
            #    Normalize against model-specific thresholds so source_quality reflects actual relevance.
            if retrieved_docs:
                scoring_cfg = get_config("scoring")
                llm_cfg = get_config("llm")
                model = get_nested(llm_cfg, "embedding", "model", default="unknown")
                all_thresholds = get_nested(scoring_cfg, "retrieval", "similarity_thresholds", default={})
                model_thresholds = all_thresholds.get(model) or all_thresholds.get("_default", {})
                floor = model_thresholds.get("general", 0.25)
                ceiling = max(floor + 0.30, 0.55)

                scores = [doc.get('score', 0.0) for doc in retrieved_docs]
                normalized = [min(max((s - floor) / (ceiling - floor), 0.0), 1.0) for s in scores]
                metrics['source_quality'] = sum(normalized) / len(normalized) if normalized else 0.0
            else:
                metrics['source_quality'] = 0.0

            # 5. Response Completeness - Basic heuristics
            has_structured_response = any(marker in response for marker in ['1.', '2.', '•', '-', 'Steps:', 'Requirements:'])
            has_specific_details = any(term in response.lower() for term in ['ratio', 'percentage', '%', '$', 'requirement', 'standard', 'regulation'])

            completeness_score = 0.5  # Base score
            if has_structured_response:
                completeness_score += 0.25
            if has_specific_details:
                completeness_score += 0.25

            metrics['completeness'] = min(completeness_score, 1.0)

            # 6. Overall Quality Score (weighted average)
            weights = {
                'citation_coverage': 0.25,
                'length_score': 0.15,
                'keyword_relevance': 0.25,
                'source_quality': 0.20,
                'completeness': 0.15
            }

            overall_score = sum(metrics[key] * weights[key] for key in weights if key in metrics)
            metrics['overall_quality'] = overall_score

            # Log quality metrics for monitoring
            logger.info("Response quality metrics calculated",
                       overall_quality=overall_score,
                       citation_coverage=metrics['citation_coverage'],
                       keyword_relevance=metrics['keyword_relevance'],
                       source_quality=metrics['source_quality'])

            return metrics

        except Exception as e:
            logger.error("Failed to calculate quality metrics", error=str(e))
            return {
                "citation_coverage": 0.0,
                "length_score": 0.5,
                "keyword_relevance": 0.5,
                "source_quality": 0.0,
                "completeness": 0.5,
                "overall_quality": 0.3
            }

    def get_similarity_threshold(self, query: str) -> float:
        """
        Determine optimal similarity threshold based on query type and embedding model.

        Classifies the query as specific/general/neutral using keyword indicators,
        then looks up the threshold from scoring.yaml keyed by the current embedding
        model name (from llm.yaml). If the model is not configured, falls back to
        _default with a warning. No threshold values are hardcoded in Python.

        Args:
            query: The user's input query

        Returns:
            Float similarity threshold value between 0.0 and 1.0
        """
        # Load thresholds from config, keyed by embedding model
        scoring_cfg = get_config("scoring")
        llm_cfg = get_config("llm")
        model = get_nested(llm_cfg, "embedding", "model", default="unknown")

        all_thresholds = get_nested(scoring_cfg, "retrieval", "similarity_thresholds", default={})
        model_thresholds = all_thresholds.get(model) or all_thresholds.get("_default", {})

        if model not in all_thresholds:
            logger.warning("No similarity thresholds configured for model, using _default",
                           model=model)

        query_lower = query.lower()

        # Use stricter threshold for specific/critical queries that need precision
        specific_indicators = [
            'specific', 'exact', 'precise', 'what is the', 'define',
            'calculation', 'formula', 'ratio', 'compliance requirement',
            'regulatory requirement', 'standard requires', 'rule states',
            'policy says', 'according to', 'as per', 'mandate'
        ]

        # Use relaxed threshold for general/exploratory queries that need comprehensiveness
        general_indicators = [
            'how to', 'what are ways', 'assess', 'evaluate', 'overview',
            'explain', 'understand', 'approach', 'methods', 'strategies',
            'best practices', 'considerations', 'factors', 'guidance',
            'help me', 'show me how'
        ]

        # Classify query and look up threshold from config
        if any(indicator in query_lower for indicator in specific_indicators):
            threshold = model_thresholds.get("specific", 0.35)
            reason = "specific query"
        elif any(indicator in query_lower for indicator in general_indicators):
            threshold = model_thresholds.get("general", 0.25)
            reason = "general query"
        else:
            threshold = model_thresholds.get("neutral", 0.30)
            reason = "neutral query"

        logger.info("Adaptive threshold selected",
                   threshold=threshold,
                   reason=reason,
                   model=model)

        return threshold

    def get_document_limit(self, query: str) -> int:
        """
        Dynamically determine document limit based on query complexity.

        Analyzes query characteristics to determine optimal number of documents:
        - Word count and length indicators
        - Question complexity patterns
        - Technical procedure requirements
        - Multi-part query detection
        - Regulatory/compliance complexity patterns

        Args:
            query: The user's input query

        Returns:
            Integer document limit based on query complexity
        """

        # Get config
        config = self.config.get('retrieval_config', {})
        base_limit = config.get('base_document_limit', 15)
        standard_limit = config.get('standard_document_limit', 20)
        complex_limit = config.get('complex_document_limit', 30)
        max_limit = config.get('max_document_limit', 40)

        # Analyze query complexity
        query_lower = query.lower()
        words = query_lower.split()
        word_count = len(words)

        # Initialize complexity score
        complexity_score = 0

        # Word count indicator
        if word_count > 15:
            complexity_score += 2  # Very long query
        elif word_count > 10:
            complexity_score += 1  # Long query

        # Question complexity indicators
        comprehensive_words = ['how', 'why', 'what', 'explain', 'describe', 'discuss']
        if any(word in words for word in comprehensive_words):
            complexity_score += 1

        # Technical procedure indicators
        technical_words = ['calculate', 'determine', 'implement', 'process', 'analyze', 'evaluate']
        if any(word in words for word in technical_words):
            complexity_score += 1

        # Multi-part query indicators
        multi_indicators = ['and', 'also', 'additionally', 'furthermore', 'moreover', 'including']
        if sum(1 for word in multi_indicators if word in words) >= 2:
            complexity_score += 1  # Multiple aspects to address

        # Regulatory/compliance complexity (generic patterns)
        if len(re.findall(r'\b[A-Z]{2,}\b', query)) > 2:  # Multiple acronyms
            complexity_score += 1

        # Choose limit based on complexity score
        if complexity_score >= 4:
            final_limit = min(complex_limit, max_limit)  # Complex: 45 docs
            complexity_name = "Complex"
        elif complexity_score >= 2:
            final_limit = min(standard_limit, max_limit)  # Standard: 35 docs
            complexity_name = "Standard"
        else:
            final_limit = min(base_limit, max_limit)      # Simple: 25 docs
            complexity_name = "Simple"

        logger.info(f"Dynamic document limit: {final_limit} ({complexity_name} query, complexity score: {complexity_score})")

        return final_limit

    def calculate_confidence(self, retrieved_docs: List[Dict], response: str) -> float:
        """
        Calculate confidence score for the response using multiple signals.

        Combines four signals:
        - Retrieval signal (35%): cosine scores normalized against model-specific thresholds
        - Coverage signal (15%): sufficient number of supporting documents
        - Citation signal (30%): response references sources via inline [N] markers
        - Substance signal (20%): non-trivial response with actual content

        Args:
            retrieved_docs: List of retrieved documents with similarity scores
            response: The generated response text

        Returns:
            Float confidence score between 0.0 and 1.0
        """
        if not retrieved_docs:
            return 0.0

        scoring_cfg = get_config("scoring")

        # 1. Retrieval signal: normalize cosine scores against model-specific thresholds
        #    text-embedding-3-large: relevant docs score 0.25-0.65 cosine
        #    Normalize to 0-1 using the configured threshold as floor
        llm_cfg = get_config("llm")
        model = get_nested(llm_cfg, "embedding", "model", default="unknown")
        all_thresholds = get_nested(scoring_cfg, "retrieval", "similarity_thresholds", default={})
        model_thresholds = all_thresholds.get(model) or all_thresholds.get("_default", {})
        floor = model_thresholds.get("general", 0.25)
        ceiling = max(floor + 0.30, 0.55)

        top_scores = [doc.get('score', 0) for doc in retrieved_docs[:5]]
        normalized = [min(max((s - floor) / (ceiling - floor), 0.0), 1.0) for s in top_scores]
        retrieval_signal = sum(normalized) / len(normalized) if normalized else 0.0

        # 2. Coverage signal: sufficient supporting documents (caps at 5)
        coverage_signal = min(len(retrieved_docs) / 5.0, 1.0)

        # 3. Citation signal: response references sources via inline [N] markers
        citation_count = len(set(re.findall(r'\[(\d+)\]', response)))
        citation_signal = min(citation_count / 3.0, 1.0)

        # 4. Response substance: non-trivial response with actual content
        response_len = len(response.strip())
        substance_signal = min(response_len / 500.0, 1.0)

        confidence = (
            0.35 * retrieval_signal +
            0.15 * coverage_signal +
            0.30 * citation_signal +
            0.20 * substance_signal
        )
        return round(min(confidence, 1.0), 3)


def create_quality_metrics_manager(config: Optional[Dict[str, Any]] = None) -> QualityMetricsManager:
    """
    Factory function to create a QualityMetricsManager instance.

    Args:
        config: Optional configuration dictionary for the metrics manager

    Returns:
        Configured QualityMetricsManager instance
    """
    return QualityMetricsManager(config)