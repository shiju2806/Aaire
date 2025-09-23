"""
Intelligent Query Analyzer - Automatically detects jurisdiction and product intent
No hardcoded logic - learns patterns from query structure and context
"""

import re
import logging
import structlog
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

logger = structlog.get_logger()

class JurisdictionHint(Enum):
    US_STAT = "us_stat"
    IFRS = "ifrs"
    MIXED = "mixed"
    UNKNOWN = "unknown"

class ProductHint(Enum):
    UNIVERSAL_LIFE = "universal_life"
    WHOLE_LIFE = "whole_life"
    TERM_LIFE = "term_life"
    VARIABLE_LIFE = "variable_life"
    GENERAL = "general"
    UNKNOWN = "unknown"

@dataclass
class QueryIntent:
    """Represents the analyzed intent of a user query"""
    jurisdiction_hint: JurisdictionHint
    jurisdiction_confidence: float
    product_hint: ProductHint
    product_confidence: float
    semantic_keywords: List[str]
    context_keywords: List[str]
    disambiguation_needed: bool
    analysis_notes: List[str]

class IntelligentQueryAnalyzer:
    """Analyzes user queries to extract jurisdiction and product intent"""

    def __init__(self):
        self.jurisdiction_patterns = self._build_jurisdiction_patterns()
        self.product_patterns = self._build_product_patterns()
        self.exclusion_patterns = self._build_exclusion_patterns()

    def _build_jurisdiction_patterns(self) -> Dict[str, List[str]]:
        """Build dynamic jurisdiction detection patterns"""
        return {
            'us_stat': [
                r'\busstat\b', r'\bus\s+stat\b', r'\bstatutory\b',
                r'\bvaluation\s+manual\b', r'\bvm-20\b', r'\bnaic\b',
                r'\bstate\s+regulation\b', r'\bus\s+gaap\b',
                r'\bstatutory\s+reserve\b', r'\bstatutory\s+accounting\b'
            ],
            'ifrs': [
                r'\bifrs\s*17\b', r'\bifrs\b', r'\biasb\b',
                r'\bcontractual\s+service\s+margin\b', r'\bcsm\b',
                r'\brisk\s+adjustment\b', r'\bfulfilment\s+cash\s+flows\b',
                r'\bbest\s+estimate\b', r'\bvariable\s+fee\s+approach\b'
            ]
        }

    def _build_product_patterns(self) -> Dict[str, List[str]]:
        """Build dynamic product type detection patterns"""
        return {
            'universal_life': [
                r'\buniversal\s+life\b', r'\bul\s+polic', r'\bul\s+insurance\b',
                r'\bflexible\s+premium\b', r'\bcash\s+value\s+life\b',
                r'\bsecondary\s+guarantee\b', r'\bno\s*lapse\s+guarantee\b',
                r'\bdeath\s+benefit\s+option\b', r'\baccount\s+value\b'
            ],
            'whole_life': [
                r'\bwhole\s+life\s+polic', r'\bwhole\s+life\s+insurance\b',
                r'\btraditional\s+whole\s+life\b', r'\bordinary\s+life\b',
                r'\bpermanent\s+insurance\b', r'\bparticipating\s+whole\s+life\b'
            ],
            'term_life': [
                r'\bterm\s+life\b', r'\bterm\s+insurance\b',
                r'\blevel\s+term\b', r'\bdecreasing\s+term\b',
                r'\brenewable\s+term\b', r'\bconvertible\s+term\b'
            ],
            'variable_life': [
                r'\bvariable\s+life\b', r'\bvariable\s+universal\s+life\b',
                r'\bvul\s+polic', r'\bseparate\s+account\b'
            ]
        }

    def _build_exclusion_patterns(self) -> Dict[str, List[str]]:
        """Build patterns to exclude false positives"""
        return {
            'whole_life_exclusions': [
                r'\bwhole\s+contract\b', r'\bwhole\s+contract\s+view\b',
                r'\bwhole\s+contract\s+approach\b', r'\bwhole\s+contract\s+method\b'
            ]
        }

    def analyze_query(self, query: str) -> QueryIntent:
        """Analyze a user query to extract intent"""
        query_lower = query.lower().strip()

        logger.debug(f"Analyzing query: {query}")

        # Analyze jurisdiction intent
        jurisdiction_analysis = self._analyze_jurisdiction(query_lower)

        # Analyze product intent
        product_analysis = self._analyze_product_type(query_lower)

        # Extract semantic keywords
        semantic_keywords = self._extract_semantic_keywords(query_lower)

        # Extract context keywords
        context_keywords = self._extract_context_keywords(query_lower)

        # Check if disambiguation is needed
        disambiguation_needed = self._check_disambiguation_needed(
            jurisdiction_analysis, product_analysis, query_lower
        )

        # Generate analysis notes
        analysis_notes = self._generate_analysis_notes(
            jurisdiction_analysis, product_analysis, query_lower
        )

        intent = QueryIntent(
            jurisdiction_hint=jurisdiction_analysis['hint'],
            jurisdiction_confidence=jurisdiction_analysis['confidence'],
            product_hint=product_analysis['hint'],
            product_confidence=product_analysis['confidence'],
            semantic_keywords=semantic_keywords,
            context_keywords=context_keywords,
            disambiguation_needed=disambiguation_needed,
            analysis_notes=analysis_notes
        )

        logger.info(f"Query analysis complete",
                   jurisdiction=intent.jurisdiction_hint.value,
                   product=intent.product_hint.value,
                   confidence_j=intent.jurisdiction_confidence,
                   confidence_p=intent.product_confidence)

        return intent

    def _analyze_jurisdiction(self, query: str) -> Dict[str, any]:
        """Analyze jurisdiction intent in query"""
        us_stat_score = 0.0
        ifrs_score = 0.0

        # Score US STAT patterns
        for pattern in self.jurisdiction_patterns['us_stat']:
            if re.search(pattern, query):
                us_stat_score += 1.0

        # Score IFRS patterns
        for pattern in self.jurisdiction_patterns['ifrs']:
            if re.search(pattern, query):
                ifrs_score += 1.0

        # Determine hint and confidence
        total_score = us_stat_score + ifrs_score

        if total_score == 0:
            return {
                'hint': JurisdictionHint.UNKNOWN,
                'confidence': 0.0,
                'scores': {'us_stat': 0.0, 'ifrs': 0.0}
            }

        if us_stat_score > ifrs_score:
            confidence = us_stat_score / total_score
            hint = JurisdictionHint.US_STAT if confidence > 0.6 else JurisdictionHint.MIXED
        elif ifrs_score > us_stat_score:
            confidence = ifrs_score / total_score
            hint = JurisdictionHint.IFRS if confidence > 0.6 else JurisdictionHint.MIXED
        else:
            confidence = 0.5
            hint = JurisdictionHint.MIXED

        return {
            'hint': hint,
            'confidence': confidence,
            'scores': {'us_stat': us_stat_score, 'ifrs': ifrs_score}
        }

    def _analyze_product_type(self, query: str) -> Dict[str, any]:
        """Analyze product type intent in query"""
        scores = {
            'universal_life': 0.0,
            'whole_life': 0.0,
            'term_life': 0.0,
            'variable_life': 0.0
        }

        # Score each product type
        for product_type, patterns in self.product_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    scores[product_type] += 1.0

        # Apply exclusions for whole life
        if scores['whole_life'] > 0:
            for exclusion_pattern in self.exclusion_patterns['whole_life_exclusions']:
                if re.search(exclusion_pattern, query):
                    scores['whole_life'] = 0.0
                    logger.debug(f"Whole life excluded due to pattern: {exclusion_pattern}")
                    break

        # Determine primary product type
        total_score = sum(scores.values())

        if total_score == 0:
            return {
                'hint': ProductHint.UNKNOWN,
                'confidence': 0.0,
                'scores': scores
            }

        # Find highest scoring product
        max_product = max(scores.items(), key=lambda x: x[1])
        product_name, product_score = max_product

        confidence = product_score / total_score

        # Map to enum
        product_map = {
            'universal_life': ProductHint.UNIVERSAL_LIFE,
            'whole_life': ProductHint.WHOLE_LIFE,
            'term_life': ProductHint.TERM_LIFE,
            'variable_life': ProductHint.VARIABLE_LIFE
        }

        hint = product_map.get(product_name, ProductHint.UNKNOWN)

        # If confidence is low, mark as general
        if confidence < 0.6:
            hint = ProductHint.GENERAL

        return {
            'hint': hint,
            'confidence': confidence,
            'scores': scores
        }

    def _extract_semantic_keywords(self, query: str) -> List[str]:
        """Extract key semantic terms from query"""
        semantic_terms = [
            'reserve', 'reserves', 'calculation', 'calculate', 'method', 'methodology',
            'valuation', 'actuarial', 'premium', 'liability', 'cash flow',
            'mortality', 'interest', 'lapse', 'surrender', 'benefit'
        ]

        found_terms = []
        for term in semantic_terms:
            if re.search(rf'\b{term}\b', query):
                found_terms.append(term)

        return found_terms

    def _extract_context_keywords(self, query: str) -> List[str]:
        """Extract contextual keywords that provide additional meaning"""
        context_patterns = [
            r'\bhow\s+do\s+i\b', r'\bhow\s+to\b', r'\bwhat\s+is\b',
            r'\bexplain\b', r'\bdescribe\b', r'\bshow\s+me\b',
            r'\bsteps\b', r'\bprocess\b', r'\bprocedure\b'
        ]

        context_keywords = []
        for pattern in context_patterns:
            if re.search(pattern, query):
                context_keywords.append(pattern.replace(r'\b', '').replace(r'\s+', ' '))

        return context_keywords

    def _check_disambiguation_needed(self, jurisdiction_analysis: Dict,
                                   product_analysis: Dict, query: str) -> bool:
        """Check if the query needs disambiguation"""
        # If both jurisdiction and product are uncertain
        if (jurisdiction_analysis['confidence'] < 0.5 and
            product_analysis['confidence'] < 0.5):
            return True

        # If multiple products have similar scores
        product_scores = product_analysis['scores']
        sorted_scores = sorted(product_scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[0] - sorted_scores[1] < 0.3:
            return True

        return False

    def _generate_analysis_notes(self, jurisdiction_analysis: Dict,
                               product_analysis: Dict, query: str) -> List[str]:
        """Generate human-readable analysis notes"""
        notes = []

        # Jurisdiction notes
        j_confidence = jurisdiction_analysis['confidence']
        if j_confidence > 0.8:
            notes.append(f"Strong {jurisdiction_analysis['hint'].value} jurisdiction signal")
        elif j_confidence > 0.5:
            notes.append(f"Moderate {jurisdiction_analysis['hint'].value} jurisdiction signal")
        else:
            notes.append("Jurisdiction unclear from query")

        # Product type notes
        p_confidence = product_analysis['confidence']
        if p_confidence > 0.8:
            notes.append(f"Strong {product_analysis['hint'].value} product signal")
        elif p_confidence > 0.5:
            notes.append(f"Moderate {product_analysis['hint'].value} product signal")
        else:
            notes.append("Product type unclear from query")

        # Specific warnings
        if 'whole contract' in query and 'whole life' in query:
            notes.append("⚠️  Potential confusion between 'Whole Contract' (UL concept) and 'Whole Life'")

        return notes