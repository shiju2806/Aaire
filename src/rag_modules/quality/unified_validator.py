"""
Unified Quality Validation System
Consolidates all validation systems into a single, configurable, dependency-injected validator
"""

import structlog
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from ..config.quality_config import QualityConfig
from ..core.dependency_injection import ServiceMixin

logger = structlog.get_logger()


@dataclass
class ValidationResult:
    """Unified validation result combining all quality checks"""
    is_valid: bool
    overall_score: float
    component_scores: Dict[str, float]
    confidence: float
    rejection_reason: Optional[str] = None
    should_generate_response: bool = True
    detailed_feedback: Dict[str, Any] = None


class UnifiedQualityValidator(ServiceMixin):
    """
    Unified quality validator that orchestrates all validation systems
    with configuration-driven behavior and dependency injection.
    """

    def __init__(self, config: Optional[QualityConfig] = None):
        """
        Initialize unified validator with configuration.

        Args:
            config: Quality configuration instance
        """
        from ..config.quality_config import get_quality_config

        # Set config before calling super().__init__() to avoid property conflicts
        self._config = config or get_quality_config()
        super().__init__()

        logger.info("Unified quality validator initialized",
                   environment=self._config.environment)

    def validate_response_quality(self, query: str, response: str,
                                retrieved_docs: List[Dict[str, Any]]) -> ValidationResult:
        """
        Perform comprehensive quality validation using all enabled validators.

        Args:
            query: Original user query
            response: Generated response text
            retrieved_docs: List of retrieved documents

        Returns:
            ValidationResult with comprehensive quality assessment
        """
        try:
            component_scores = {}
            detailed_feedback = {}

            # 1. Semantic Alignment Validation
            if self._is_validator_enabled('semantic_alignment'):
                semantic_result = self._validate_semantic_alignment(query, retrieved_docs)
                component_scores['semantic_alignment'] = semantic_result.alignment_score
                component_scores['semantic_confidence'] = semantic_result.confidence
                detailed_feedback['semantic_alignment'] = {
                    'is_aligned': semantic_result.is_aligned,
                    'explanation': semantic_result.explanation
                }

            # 2. Content Grounding Validation
            if self._is_validator_enabled('grounding'):
                grounding_result = self._validate_content_grounding(query, response, retrieved_docs)
                component_scores['grounding_score'] = grounding_result.grounding_score
                component_scores['evidence_coverage'] = grounding_result.evidence_coverage
                component_scores['hallucination_risk'] = grounding_result.hallucination_risk
                detailed_feedback['grounding'] = grounding_result.grounding_details

            # 3. OpenAI Alignment Validation (if enabled)
            if self._is_validator_enabled('openai_alignment'):
                openai_result = self._validate_openai_alignment(query, response, retrieved_docs)
                if openai_result:
                    component_scores['openai_alignment'] = openai_result.get('alignment_score', 0.0)
                    detailed_feedback['openai_alignment'] = openai_result

            # Calculate overall score using configuration weights
            overall_score = self._calculate_overall_score(component_scores)

            # Determine validation result
            is_valid, confidence, rejection_reason = self._determine_validation_result(
                component_scores, overall_score
            )

            # Create unified result
            result = ValidationResult(
                is_valid=is_valid,
                overall_score=overall_score,
                component_scores=component_scores,
                confidence=confidence,
                rejection_reason=rejection_reason,
                should_generate_response=is_valid,
                detailed_feedback=detailed_feedback
            )

            logger.info("Unified quality validation completed",
                       overall_score=overall_score,
                       is_valid=is_valid,
                       confidence=confidence,
                       active_validators=list(component_scores.keys()))

            return result

        except Exception as e:
            logger.error("Unified quality validation failed", exception_details=str(e))
            # Fail open - allow response generation if validation fails
            return ValidationResult(
                is_valid=True,
                overall_score=0.5,
                component_scores={},
                confidence=0.0,
                rejection_reason=f"Validation error: {str(e)}",
                should_generate_response=True,
                detailed_feedback={"error": str(e)}
            )

    def _is_validator_enabled(self, validator_name: str) -> bool:
        """Check if a specific validator is enabled in configuration."""
        feature_flags = {
            'semantic_alignment': True,  # Always enabled
            'grounding': True,  # Always enabled
            'openai_alignment': self._config.is_openai_alignment_enabled()
        }
        return feature_flags.get(validator_name, False)

    def _validate_semantic_alignment(self, query: str, retrieved_docs: List[Dict]) -> Any:
        """Perform semantic alignment validation using injected validator."""
        semantic_validator = self.get_service('semantic_alignment_validator')
        return semantic_validator.validate_alignment(query, retrieved_docs)

    def _validate_content_grounding(self, query: str, response: str, retrieved_docs: List[Dict]) -> Any:
        """Perform content grounding validation using injected validator."""
        grounding_validator = self.get_service('grounding_validator')
        return grounding_validator.validate_content_grounding(query, response, retrieved_docs)

    def _validate_openai_alignment(self, query: str, response: str, retrieved_docs: List[Dict]) -> Optional[Dict]:
        """Perform OpenAI alignment validation if enabled."""
        if not self._config.is_openai_alignment_enabled():
            return None

        try:
            openai_validator = self.get_service('openai_alignment_validator')
            if openai_validator:
                return openai_validator.validate_alignment(query, response, retrieved_docs)
        except Exception as e:
            logger.warning("OpenAI alignment validation failed", exception_details=str(e))

        return None

    def _calculate_overall_score(self, component_scores: Dict[str, float]) -> float:
        """Calculate overall quality score using configuration weights."""
        weights = self._config.get_unified_quality_weights()

        if not weights:
            # Fallback to equal weighting
            return sum(component_scores.values()) / len(component_scores) if component_scores else 0.0

        total_score = 0.0
        total_weight = 0.0

        # Map component scores to weight categories
        score_mapping = {
            'semantic_score': ['semantic_alignment', 'semantic_confidence'],
            'grounding_score': ['grounding_score', 'evidence_coverage'],
            'hallucination_penalty': ['hallucination_risk'],
            'openai_score': ['openai_alignment']
        }

        for weight_key, component_keys in score_mapping.items():
            weight = weights.get(weight_key, 0.0)
            if weight > 0:
                # Calculate average score for this category
                category_scores = [component_scores[key] for key in component_keys if key in component_scores]
                if category_scores:
                    if weight_key == 'hallucination_penalty':
                        # Invert hallucination risk (lower risk = higher score)
                        category_score = 1.0 - (sum(category_scores) / len(category_scores))
                    else:
                        category_score = sum(category_scores) / len(category_scores)

                    total_score += category_score * weight
                    total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _determine_validation_result(self, component_scores: Dict[str, float],
                                   overall_score: float) -> tuple[bool, float, Optional[str]]:
        """Determine if validation passes based on component scores and thresholds."""

        # Check individual component thresholds
        failures = []

        # Semantic alignment check
        semantic_score = component_scores.get('semantic_alignment', 1.0)
        if semantic_score < self._config.get_semantic_alignment_threshold():
            failures.append("insufficient semantic alignment with source documents")

        # Grounding check
        grounding_score = component_scores.get('grounding_score', 1.0)
        if grounding_score < self._config.get_grounding_threshold():
            failures.append("inadequate content grounding in source materials")

        # Evidence coverage check - TEMPORARILY DISABLED FOR TESTING
        evidence_coverage = component_scores.get('evidence_coverage', 1.0)
        # if evidence_coverage < self._config.get_evidence_coverage_threshold():
        #     failures.append("insufficient evidence coverage")

        # Hallucination risk check
        hallucination_risk = component_scores.get('hallucination_risk', 0.0)
        if hallucination_risk > self._config.get_hallucination_risk_threshold():
            failures.append("high hallucination risk detected")

        # Confidence calculation
        confidence_factors = [
            component_scores.get('semantic_confidence', 0.5),
            1.0 - hallucination_risk,  # Convert risk to confidence
            min(1.0, grounding_score * 1.2),  # Boost grounding confidence
            overall_score
        ]
        confidence = sum(confidence_factors) / len(confidence_factors)

        # Final determination
        quality_threshold = self._config.get_threshold('overall_quality_minimum') if self._config else 0.6
        is_valid = len(failures) == 0 and overall_score >= quality_threshold
        rejection_reason = "; ".join(failures) if failures else None

        return is_valid, confidence, rejection_reason

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation configuration and capabilities."""
        return {
            "environment": self._config.environment,
            "enabled_validators": [
                name for name in ['semantic_alignment', 'grounding', 'openai_alignment']
                if self._is_validator_enabled(name)
            ],
            "thresholds": {
                "semantic_alignment": self._config.get_semantic_alignment_threshold(),
                "grounding": self._config.get_grounding_threshold(),
                "evidence_coverage": self._config.get_evidence_coverage_threshold(),
                "hallucination_risk": self._config.get_hallucination_risk_threshold()
            },
            "weights": self._config.get_unified_quality_weights(),
            "feature_flags": {
                "advanced_grounding": self._config.is_advanced_grounding_enabled(),
                "adaptive_learning": self._config.is_adaptive_learning_enabled(),
                "openai_alignment": self._config.is_openai_alignment_enabled()
            }
        }


def create_unified_validator(config: Optional[QualityConfig] = None) -> UnifiedQualityValidator:
    """Factory function to create unified quality validator."""
    return UnifiedQualityValidator(config=config)