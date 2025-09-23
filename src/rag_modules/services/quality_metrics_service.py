"""
Quality Metrics Service
Modern replacement for QualityMetricsManager using dependency injection and configuration
"""

import structlog
from typing import Dict, List, Any, Optional
from ..config.quality_config import QualityConfig
from ..core.dependency_injection import ServiceMixin

logger = structlog.get_logger()


class QualityMetricsService(ServiceMixin):
    """
    Modern quality metrics service that replaces the old QualityMetricsManager.
    Uses dependency injection and centralized configuration.
    """

    def __init__(self, config: Optional[QualityConfig] = None):
        """Initialize quality metrics service with configuration."""
        super().__init__()
        from ..config.quality_config import get_quality_config
        self._config = config or get_quality_config()

        # Learned thresholds (can be updated through adaptive learning)
        self.learned_similarity_threshold = 0.70
        self.performance_history = []

        logger.info("Quality metrics service initialized",
                   environment=self._config.environment)

    def get_similarity_threshold(self, query: str) -> float:
        """
        Determine optimal similarity threshold using configuration and adaptive learning.

        Args:
            query: The user's input query

        Returns:
            Float similarity threshold value between 0.0 and 1.0
        """
        # Get base threshold from configuration
        base_threshold = self._config.get_threshold('similarity_threshold_base')

        # Use learned threshold if available
        if hasattr(self, 'learned_similarity_threshold'):
            base_threshold = self.learned_similarity_threshold

        # Adaptive adjustment based on query characteristics
        query_length = len(query.split())
        query_complexity = len([word for word in query.split() if len(word) > 6])

        # Get adjustment factors from configuration
        long_query_adjustment = self._config.get_threshold('long_query_adjustment')
        complex_query_adjustment = self._config.get_threshold('complex_query_adjustment')

        if query_length > 15:  # Long queries may need lower threshold
            adjustment = long_query_adjustment
            reason = "long query - comprehensive search"
        elif query_complexity > 3:  # Complex queries may need precise matches
            adjustment = complex_query_adjustment
            reason = "complex query - precise search"
        else:
            adjustment = 0.0
            reason = "standard query"

        # Apply bounds from configuration
        min_threshold = self._config.get_threshold('similarity_threshold_min')
        max_threshold = self._config.get_threshold('similarity_threshold_max')
        threshold = max(min_threshold, min(max_threshold, base_threshold + adjustment))

        logger.debug("Adaptive threshold selected",
                    query=query[:50] + "..." if len(query) > 50 else query,
                    threshold=threshold,
                    base_threshold=base_threshold,
                    adjustment=adjustment,
                    reason=reason)

        return threshold

    def get_document_limit(self, query: str) -> int:
        """
        Dynamically determine document limit based on query complexity.

        Args:
            query: The user's input query

        Returns:
            Integer document limit
        """
        # Get base limits from configuration with proper defaults for document limits
        # These should be integers, not floats like similarity thresholds
        # Handle nested config structure: quality_validation.thresholds.document_limit_base
        quality_section = self._config.config.get("quality_validation", {})
        thresholds = quality_section.get("thresholds", {})
        base_limit = int(thresholds.get("document_limit_base", 10))
        max_limit = int(thresholds.get("document_limit_max", 20))

        query_length = len(query.split())

        if query_length > 20:  # Very long queries might need more documents
            return min(max_limit, base_limit + 5)
        elif query_length > 10:  # Medium queries get standard limit
            return base_limit
        else:  # Short queries need fewer documents
            return max(5, base_limit - 2)

    def calculate_confidence(self, retrieved_docs: List[Dict], response: str) -> float:
        """
        Calculate confidence score for the response using modern approach.

        Args:
            retrieved_docs: List of retrieved documents
            response: Generated response text

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not retrieved_docs or not response:
            return 0.1

        # Try to use semantic alignment validator for confidence calculation
        try:
            semantic_validator = self.get_service('semantic_alignment_validator')
            # Create a simple query simulation for validation
            mock_query = "confidence calculation"
            validation_result = semantic_validator.validate_alignment(mock_query, retrieved_docs)
            base_confidence = validation_result.confidence
        except Exception as e:
            logger.debug("Semantic validator not available, using fallback confidence", error=str(e))
            # Fallback confidence calculation
            base_confidence = min(0.8, len(retrieved_docs) / 5.0)  # More docs = higher confidence

        # Adjust confidence based on response length and document count
        response_length_factor = min(1.0, len(response.split()) / 100)  # Longer responses can be more confident
        doc_count_factor = min(1.0, len(retrieved_docs) / 10)  # More documents can increase confidence

        # Weight the factors
        final_confidence = (
            base_confidence * 0.6 +
            response_length_factor * 0.2 +
            doc_count_factor * 0.2
        )

        return max(0.1, min(1.0, final_confidence))

    def calculate_quality_metrics(self, query: str, response: str,
                                retrieved_docs: List[Dict], citations: List[Dict]) -> Dict[str, Any]:
        """
        Calculate comprehensive quality metrics for the response.

        Args:
            query: Original user query
            response: Generated response
            retrieved_docs: List of retrieved documents
            citations: List of citations in the response

        Returns:
            Dictionary containing quality metrics
        """
        try:
            # Get unified validator for comprehensive metrics
            unified_validator = self.get_service('unified_validator')

            # Perform quality validation
            validation_result = unified_validator.validate_response_quality(
                query, response, retrieved_docs
            )

            # Build comprehensive metrics
            metrics = {
                'overall_score': validation_result.overall_score,
                'confidence': validation_result.confidence,
                'is_valid': validation_result.is_valid,
                'component_scores': validation_result.component_scores,
                'citation_count': len(citations),
                'document_count': len(retrieved_docs),
                'response_length': len(response.split()),
                'query_length': len(query.split())
            }

            # Add detailed component metrics
            if validation_result.detailed_feedback:
                metrics['detailed_feedback'] = validation_result.detailed_feedback

            # Calculate additional metrics
            if retrieved_docs:
                metrics['avg_doc_relevance'] = sum(
                    doc.get('relevance_score', 0.5) for doc in retrieved_docs
                ) / len(retrieved_docs)

            if citations:
                metrics['citation_density'] = len(citations) / len(response.split()) * 100

            logger.debug("Quality metrics calculated",
                        overall_score=metrics['overall_score'],
                        confidence=metrics['confidence'],
                        component_count=len(metrics['component_scores']))

            return metrics

        except Exception as e:
            logger.error("Failed to calculate quality metrics", error=str(e))
            # Return basic fallback metrics
            return {
                'overall_score': 0.5,
                'confidence': self.calculate_confidence(retrieved_docs, response),
                'is_valid': True,  # Fail open
                'component_scores': {},
                'citation_count': len(citations),
                'document_count': len(retrieved_docs),
                'error': str(e)
            }

    def update_learned_threshold(self, new_threshold: float):
        """Update the learned similarity threshold."""
        old_threshold = self.learned_similarity_threshold
        self.learned_similarity_threshold = max(0.4, min(0.9, new_threshold))

        logger.info("Updated learned similarity threshold",
                   old_threshold=old_threshold,
                   new_threshold=self.learned_similarity_threshold)

    def record_performance_feedback(self, query: str, success: bool, metrics: Dict[str, Any]):
        """Record performance feedback for adaptive learning."""
        feedback_record = {
            'query': query,
            'success': success,
            'metrics': metrics,
            'timestamp': logger._context.get('timestamp', 'unknown')
        }

        self.performance_history.append(feedback_record)

        # Keep history bounded
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]

        # Trigger adaptive learning if we have enough data
        if len(self.performance_history) >= 50:
            self._update_adaptive_parameters()

    def _update_adaptive_parameters(self):
        """Update adaptive parameters based on performance history."""
        if not self.performance_history:
            return

        recent_history = self.performance_history[-50:]
        success_rate = sum(1 for record in recent_history if record['success']) / len(recent_history)

        # Adjust similarity threshold based on success rate
        if success_rate < 0.6:  # Too many failures, lower threshold
            adjustment = -0.05
        elif success_rate > 0.9:  # Too easy, raise threshold
            adjustment = 0.05
        else:
            adjustment = 0.0

        if adjustment != 0.0:
            new_threshold = self.learned_similarity_threshold + adjustment
            self.update_learned_threshold(new_threshold)


def create_quality_metrics_service(config: Optional[QualityConfig] = None) -> QualityMetricsService:
    """Factory function to create quality metrics service."""
    return QualityMetricsService(config)