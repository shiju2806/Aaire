"""
Enhanced Grounding Validator with Configuration-Driven Validation
Validates that specific claims, formulas, and numbers in responses are grounded in documents
"""

import re
import structlog
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from ..config.quality_config import QualityConfig

logger = structlog.get_logger()


@dataclass
class GroundingValidationResult:
    """Result of enhanced grounding validation"""
    is_grounded: bool
    confidence: float
    grounding_score: float
    specific_checks: Dict[str, bool]
    ungrounded_claims: List[str]
    evidence_map: Dict[str, List[str]]
    details: Dict[str, Any]


class EnhancedGroundingValidator:
    """
    Configuration-driven grounding validator that checks specific claims against documents.
    No hardcoded logic - all behavior controlled by configuration.
    """

    def __init__(self, config: Optional[QualityConfig] = None):
        """Initialize with configuration"""
        from ..config.quality_config import get_quality_config
        self.config = config or get_quality_config()

        # Get grounding settings from config
        grounding_config = self.config.config.get('grounding', {})
        self.enabled = grounding_config.get('enabled', True)
        self.strict_mode = grounding_config.get('strict_mode', True)
        self.check_formulas = grounding_config.get('check_formulas', True)
        self.check_numbers = grounding_config.get('check_numbers', True)
        self.check_specific_claims = grounding_config.get('check_specific_claims', True)
        self.min_evidence_threshold = grounding_config.get('min_evidence_threshold', 0.70)

        logger.info("Enhanced grounding validator initialized",
                   strict_mode=self.strict_mode,
                   checks_enabled={
                       'formulas': self.check_formulas,
                       'numbers': self.check_numbers,
                       'claims': self.check_specific_claims
                   })

    def validate_response_grounding(
        self,
        response: str,
        documents: List[Dict[str, Any]]
    ) -> GroundingValidationResult:
        """
        Validate that response claims are grounded in documents.

        Args:
            response: Generated response text
            documents: Retrieved documents

        Returns:
            GroundingValidationResult with detailed validation info
        """
        if not self.enabled:
            return GroundingValidationResult(
                is_grounded=True,
                confidence=1.0,
                grounding_score=1.0,
                specific_checks={},
                ungrounded_claims=[],
                evidence_map={},
                details={'validation_skipped': True}
            )

        # Combine all document content for searching
        combined_docs = "\n".join([doc.get('content', '') for doc in documents])

        specific_checks = {}
        ungrounded_claims = []
        evidence_map = {}

        # Extract and validate formulas
        if self.check_formulas:
            formula_validation = self._validate_formulas(response, combined_docs)
            specific_checks['formulas'] = formula_validation['valid']
            if not formula_validation['valid']:
                ungrounded_claims.extend(formula_validation['ungrounded'])
            evidence_map['formulas'] = formula_validation['evidence']

        # Extract and validate numbers/percentages
        if self.check_numbers:
            number_validation = self._validate_numbers(response, combined_docs)
            specific_checks['numbers'] = number_validation['valid']
            if not number_validation['valid']:
                ungrounded_claims.extend(number_validation['ungrounded'])
            evidence_map['numbers'] = number_validation['evidence']

        # Extract and validate specific technical claims
        if self.check_specific_claims:
            claim_validation = self._validate_specific_claims(response, combined_docs)
            specific_checks['claims'] = claim_validation['valid']
            if not claim_validation['valid']:
                ungrounded_claims.extend(claim_validation['ungrounded'])
            evidence_map['claims'] = claim_validation['evidence']

        # Calculate overall grounding score
        total_checks = len(specific_checks)
        passed_checks = sum(1 for v in specific_checks.values() if v)
        grounding_score = passed_checks / total_checks if total_checks > 0 else 0.0

        # Determine if response is grounded
        is_grounded = grounding_score >= self.min_evidence_threshold
        if self.strict_mode:
            # In strict mode, ALL checks must pass
            is_grounded = all(specific_checks.values()) if specific_checks else True

        # Calculate confidence based on evidence coverage
        confidence = min(1.0, grounding_score + 0.1 * len(evidence_map))

        return GroundingValidationResult(
            is_grounded=is_grounded,
            confidence=confidence,
            grounding_score=grounding_score,
            specific_checks=specific_checks,
            ungrounded_claims=ungrounded_claims,
            evidence_map=evidence_map,
            details={
                'strict_mode': self.strict_mode,
                'threshold': self.min_evidence_threshold,
                'total_claims_checked': len(ungrounded_claims) + sum(len(v) for v in evidence_map.values())
            }
        )

    def _validate_formulas(self, response: str, documents: str) -> Dict[str, Any]:
        """Validate mathematical formulas and expressions"""
        # Pattern to match formulas (e.g., "W = 0.50", "I = 0.03 + W × (R1 - 0.03)")
        formula_pattern = r'([A-Z]\w*\s*=\s*[0-9\.\+\-\*\/\(\)\s\w×]+)'
        formulas = re.findall(formula_pattern, response)

        valid_formulas = []
        ungrounded_formulas = []
        evidence = []

        for formula in formulas:
            # Clean the formula for searching
            clean_formula = formula.replace('×', '*').replace(' ', '')

            # Check if formula exists in documents (with some flexibility)
            formula_parts = re.split(r'[=\+\-\*\/\(\)]', formula)
            formula_found = False

            for part in formula_parts:
                if len(part.strip()) > 2:  # Skip single chars and operators
                    if part.strip() in documents:
                        formula_found = True
                        evidence.append(f"Formula component '{part.strip()}' found in documents")
                        break

            # Also check for the complete formula
            if formula in documents or clean_formula in documents:
                formula_found = True
                evidence.append(f"Complete formula '{formula}' found in documents")

            if formula_found:
                valid_formulas.append(formula)
            else:
                ungrounded_formulas.append(formula)

        return {
            'valid': len(ungrounded_formulas) == 0,
            'ungrounded': ungrounded_formulas,
            'evidence': evidence
        }

    def _validate_numbers(self, response: str, documents: str) -> Dict[str, Any]:
        """Validate specific numbers and percentages"""
        # Pattern to match numbers with context (e.g., "90%", "$2.50", "0.03")
        number_pattern = r'(\d+\.?\d*%?|\$\d+\.?\d*)'
        numbers = re.findall(number_pattern, response)

        # Filter out common numbers that don't need validation
        significant_numbers = [n for n in numbers if not n in ['1', '2', '3'] and len(n) > 1]

        valid_numbers = []
        ungrounded_numbers = []
        evidence = []

        for number in significant_numbers:
            if number in documents:
                valid_numbers.append(number)
                evidence.append(f"Number '{number}' found in documents")
            else:
                # Check if it's a reasonable derivative (e.g., calculation result)
                base_number = number.rstrip('%').lstrip('$')
                if base_number in documents:
                    valid_numbers.append(number)
                    evidence.append(f"Base number for '{number}' found in documents")
                else:
                    ungrounded_numbers.append(number)

        return {
            'valid': len(ungrounded_numbers) == 0,
            'ungrounded': ungrounded_numbers,
            'evidence': evidence
        }

    def _validate_specific_claims(self, response: str, documents: str) -> Dict[str, Any]:
        """Validate specific technical claims and terminology using dynamic extraction"""
        # Extract technical terms dynamically based on patterns
        # Look for technical terms that are uppercase acronyms or specific terminology patterns
        acronym_pattern = r'\b[A-Z]{2,5}\b'  # 2-5 letter acronyms
        technical_term_pattern = r'\b[a-z]+\s+[a-z]+\s+[a-z]+\b'  # Multi-word technical terms

        claims = []

        # Find acronyms
        acronyms = re.findall(acronym_pattern, response)
        claims.extend(acronyms)

        # Find technical terms (multi-word phrases that might be technical)
        technical_terms = re.findall(technical_term_pattern, response, re.IGNORECASE)
        # Filter for potentially technical terms (containing specific indicators)
        for term in technical_terms:
            term_lower = term.lower()
            if any(indicator in term_lower for indicator in ['factor', 'rate', 'test', 'reserve', 'value', 'duration']):
                claims.append(term)

        valid_claims = []
        ungrounded_claims = []
        evidence = []

        for claim in claims:
            # Check if the technical term appears in documents
            if claim.lower() in documents.lower():
                valid_claims.append(claim)
                evidence.append(f"Technical term '{claim}' found in documents")
            else:
                # Check for variations
                claim_words = claim.split()
                if any(word.lower() in documents.lower() for word in claim_words if len(word) > 3):
                    valid_claims.append(claim)
                    evidence.append(f"Related term for '{claim}' found in documents")
                else:
                    ungrounded_claims.append(claim)

        return {
            'valid': len(ungrounded_claims) == 0,
            'ungrounded': ungrounded_claims,
            'evidence': evidence
        }


def create_enhanced_grounding_validator(config: Optional[QualityConfig] = None) -> EnhancedGroundingValidator:
    """Factory function to create enhanced grounding validator"""
    return EnhancedGroundingValidator(config)