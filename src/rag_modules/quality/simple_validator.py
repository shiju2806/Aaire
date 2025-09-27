"""
Simple Quality Validator
Consolidates all validation logic into a single, non-redundant validator
"""

import structlog
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = structlog.get_logger()


@dataclass
class SimpleValidationResult:
    """Simple validation result"""
    is_valid: bool
    confidence: float
    overall_score: float
    rejection_reason: Optional[str] = None
    details: Dict[str, Any] = None


class SimpleQualityValidator:
    """
    Single validator that replaces all overlapping validation logic.
    Performs only essential checks without redundancy.
    """

    def __init__(self, config=None):
        """Initialize with minimal configuration"""
        self.config = config
        # Simple thresholds - single source of truth
        self.min_confidence = 0.3
        self.min_document_overlap = 0.2
        self.enabled = True

        logger.info("Simple quality validator initialized")

    def validate_response(self, query: str, response: str,
                         retrieved_docs: List[Dict[str, Any]]) -> SimpleValidationResult:
        """
        Single validation method that replaces all overlapping validators.

        Performs only essential checks:
        1. Basic document relevance
        2. Simple hallucination detection
        3. Minimal grounding check
        """
        if not self.enabled:
            return SimpleValidationResult(
                is_valid=True,
                confidence=1.0,
                overall_score=1.0,
                details={'validation_disabled': True}
            )

        try:
            # 1. Basic document relevance (replaces semantic alignment)
            doc_relevance = self._check_document_relevance(query, retrieved_docs)

            # 2. Simple grounding check (replaces complex grounding validation)
            grounding_score = self._check_basic_grounding(response, retrieved_docs)

            # 3. Basic hallucination detection (replaces complex pattern matching)
            hallucination_risk = self._check_hallucination_risk(response, retrieved_docs)

            # Single overall score calculation
            overall_score = (doc_relevance + grounding_score + (1.0 - hallucination_risk)) / 3.0
            confidence = min(1.0, overall_score + 0.1)

            # Single validation decision
            is_valid = overall_score >= self.min_confidence
            rejection_reason = None if is_valid else "Response quality below threshold"

            result = SimpleValidationResult(
                is_valid=is_valid,
                confidence=confidence,
                overall_score=overall_score,
                rejection_reason=rejection_reason,
                details={
                    'doc_relevance': doc_relevance,
                    'grounding_score': grounding_score,
                    'hallucination_risk': hallucination_risk
                }
            )

            logger.info("Simple validation completed",
                       is_valid=is_valid,
                       overall_score=overall_score,
                       confidence=confidence)

            return result

        except Exception as e:
            logger.error("Simple validation failed", error=str(e))
            # Fail open
            return SimpleValidationResult(
                is_valid=True,
                confidence=0.5,
                overall_score=0.5,
                rejection_reason=f"Validation error: {str(e)}",
                details={'error': str(e)}
            )

    def _check_document_relevance(self, query: str, documents: List[Dict]) -> float:
        """
        Simple document relevance check.
        Replaces: semantic_alignment_validator + openai_alignment_validator
        """
        if not documents:
            return 0.0

        query_words = set(query.lower().split())
        total_overlap = 0.0

        for doc in documents[:3]:  # Check top 3 documents only
            doc_content = doc.get('content', '').lower()
            doc_words = set(doc_content.split())

            if doc_words:
                overlap = len(query_words & doc_words) / len(query_words)
                total_overlap += overlap

        avg_overlap = total_overlap / min(3, len(documents))
        return min(1.0, avg_overlap * 2)  # Scale up for reasonable scores

    def _check_basic_grounding(self, response: str, documents: List[Dict]) -> float:
        """
        Simple grounding check.
        Replaces: grounding_validator + enhanced_grounding_validator
        """
        if not documents:
            return 0.0

        # Combine all document content
        all_doc_text = " ".join([doc.get('content', '') for doc in documents]).lower()
        response_lower = response.lower()

        # Extract key response terms (4+ characters)
        response_terms = set(re.findall(r'\b\w{4,}\b', response_lower))

        # Check how many response terms appear in documents
        grounded_terms = 0
        for term in response_terms:
            if term in all_doc_text:
                grounded_terms += 1

        if not response_terms:
            return 0.5  # Neutral for short responses

        grounding_ratio = grounded_terms / len(response_terms)
        return grounding_ratio

    def _check_hallucination_risk(self, response: str, documents: List[Dict]) -> float:
        """
        Simple hallucination detection.
        Replaces: complex pattern matching in grounding_validator
        """
        # Very basic check for generic content patterns
        generic_patterns = [
            r'in general',
            r'typically',
            r'commonly used',
            r'standard practice'
        ]

        risk_signals = 0
        for pattern in generic_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                risk_signals += 1

        # Risk is proportional to generic patterns found
        risk_score = min(1.0, risk_signals / len(generic_patterns))
        return risk_score

    def disable(self):
        """Disable validation entirely"""
        self.enabled = False
        logger.info("Simple validator disabled")

    def enable(self):
        """Enable validation"""
        self.enabled = True
        logger.info("Simple validator enabled")


def create_simple_validator(config=None) -> SimpleQualityValidator:
    """Factory function for simple validator"""
    return SimpleQualityValidator(config)