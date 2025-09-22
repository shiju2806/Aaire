"""
Validation Service
Modern replacement for IntelligentValidationSystem using dependency injection
"""

import structlog
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from ..config.quality_config import QualityConfig
from ..core.dependency_injection import ServiceMixin

logger = structlog.get_logger()


@dataclass
class ValidationResult:
    """Result of response validation"""
    passed: bool
    overall_score: float
    confidence: float
    rejection_reason: Optional[str] = None
    components: Dict[str, Any] = None
    processing_time_ms: float = 0.0


class ValidationService(ServiceMixin):
    """
    Modern validation service that replaces the old IntelligentValidationSystem.
    Uses the new unified validator with dependency injection.
    """

    def __init__(self, config: Optional[QualityConfig] = None):
        """Initialize validation service with configuration."""
        super().__init__()
        from ..config.quality_config import get_quality_config
        self._config = config or get_quality_config()

        logger.info("Validation service initialized",
                   environment=self._config.environment)

    async def validate_response(self, query: str, response: str,
                              retrieved_docs: List[Dict[str, Any]],
                              citations: Optional[List[Dict]] = None,
                              confidence: Optional[float] = None) -> ValidationResult:
        """
        Validate response quality using the new unified validator.

        Args:
            query: Original user query
            response: Generated response text
            retrieved_docs: List of retrieved documents
            citations: Optional list of citations
            confidence: Optional confidence score (for compatibility)

        Returns:
            ValidationResult with validation outcome
        """
        import time
        start_time = time.time()

        try:
            # Get unified validator from service container
            unified_validator = self.get_service('unified_validator')

            # Perform comprehensive validation
            validation_result = unified_validator.validate_response_quality(
                query, response, retrieved_docs
            )

            # Convert to expected format for backward compatibility
            processing_time = (time.time() - start_time) * 1000

            result = ValidationResult(
                passed=validation_result.is_valid,
                overall_score=validation_result.overall_score,
                confidence=validation_result.confidence,
                rejection_reason=validation_result.rejection_reason,
                components=validation_result.component_scores,
                processing_time_ms=processing_time
            )

            logger.info("Response validation completed",
                       passed=result.passed,
                       overall_score=result.overall_score,
                       confidence=result.confidence,
                       processing_time_ms=result.processing_time_ms)

            return result

        except Exception as e:
            logger.error("Validation failed", error=str(e))
            processing_time = (time.time() - start_time) * 1000

            # Return permissive result on validation failure
            return ValidationResult(
                passed=True,  # Fail open
                overall_score=0.5,
                confidence=0.0,
                rejection_reason=f"Validation error: {str(e)}",
                components={},
                processing_time_ms=processing_time
            )

    def validate_response_sync(self, query: str, response: str,
                             retrieved_docs: List[Dict[str, Any]],
                             citations: Optional[List[Dict]] = None) -> ValidationResult:
        """
        Synchronous wrapper for response validation.

        Args:
            query: Original user query
            response: Generated response text
            retrieved_docs: List of retrieved documents
            citations: Optional list of citations

        Returns:
            ValidationResult with validation outcome
        """
        try:
            # Run async validation in event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.validate_response(query, response, retrieved_docs, citations)
                    )
                    return future.result()
            else:
                # Run in the current event loop
                return loop.run_until_complete(
                    self.validate_response(query, response, retrieved_docs, citations)
                )
        except Exception as e:
            logger.error("Sync validation failed", error=str(e))
            # Return permissive result
            return ValidationResult(
                passed=True,
                overall_score=0.5,
                confidence=0.0,
                rejection_reason=f"Sync validation error: {str(e)}",
                components={}
            )

    def get_validation_thresholds(self) -> Dict[str, float]:
        """Get current validation thresholds from configuration."""
        return {
            'semantic_alignment': self._config.get_semantic_alignment_threshold(),
            'grounding_score': self._config.get_grounding_threshold(),
            'evidence_coverage': self._config.get_evidence_coverage_threshold(),
            'hallucination_risk': self._config.get_hallucination_risk_threshold(),
            'overall_quality': self._config.get_threshold('overall_quality_minimum')
        }

    def update_threshold(self, metric: str, value: float):
        """Update a validation threshold dynamically."""
        self._config.update_threshold(metric, value)
        logger.info("Validation threshold updated",
                   metric=metric, value=value)

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation configuration."""
        unified_validator = self.get_service('unified_validator')
        return unified_validator.get_validation_summary()


def create_validation_service(config: Optional[QualityConfig] = None) -> ValidationService:
    """Factory function to create validation service."""
    return ValidationService(config)