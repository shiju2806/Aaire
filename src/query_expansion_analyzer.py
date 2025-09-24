"""
Query Expansion Analyzer - Expands queries with related terms for better retrieval
Focus on intelligent query enrichment rather than over-engineered classification
"""

import re
import logging
import structlog
from typing import List, Dict, Optional, Set
from dataclasses import dataclass

logger = structlog.get_logger()

@dataclass
class ExpandedQuery:
    """Represents an expanded query with enrichment terms"""
    original_query: str
    expanded_terms: List[str]
    synonyms: List[str]
    context_terms: List[str]
    semantic_keywords: List[str]
    expansion_notes: List[str]

class QueryExpansionAnalyzer:
    """Analyzes user queries and expands them with related terms for better retrieval"""

    def __init__(self):
        self.domain_synonyms = self._build_domain_synonyms()
        self.technical_terms = self._build_technical_terms()
        self.context_patterns = self._build_context_patterns()

    def _build_domain_synonyms(self) -> Dict[str, List[str]]:
        """Build domain-specific synonym mappings"""
        return {
            # Life Insurance Products
            'universal life': ['UL', 'flexible premium life', 'adjustable life', 'variable universal life', 'VUL'],
            'whole life': ['traditional life', 'ordinary life', 'permanent life insurance', 'participating whole life'],
            'term life': ['level term', 'decreasing term', 'renewable term', 'convertible term'],

            # Actuarial Terms
            'reserves': ['reserve calculations', 'actuarial reserves', 'policy reserves', 'liability reserves'],
            'valuation': ['actuarial valuation', 'policy valuation', 'asset valuation', 'reserve valuation'],
            'mortality': ['mortality rates', 'death rates', 'survival rates', 'life expectancy'],
            'lapse': ['surrender', 'policy termination', 'discontinuance', 'withdrawal'],

            # Financial Terms
            'cash value': ['account value', 'surrender value', 'cash accumulation', 'policy value'],
            'premium': ['payment', 'contribution', 'deposit', 'premium payment'],
            'interest': ['interest rates', 'crediting rate', 'guaranteed rate', 'current rate'],

            # Regulatory/Standards
            'statutory': ['stat', 'regulatory', 'state requirements', 'insurance regulation'],
            'gaap': ['generally accepted accounting principles', 'financial reporting'],
            'vm-20': ['valuation manual', 'principle based reserves', 'pbr'],
        }

    def _build_technical_terms(self) -> Dict[str, List[str]]:
        """Build technical term expansions"""
        return {
            # Calculation Methods
            'calculate': ['computation', 'determine', 'estimate', 'derive', 'compute'],
            'method': ['methodology', 'approach', 'technique', 'process', 'procedure'],
            'formula': ['equation', 'calculation', 'mathematical model', 'algorithm'],

            # Financial Concepts
            'liability': ['obligation', 'debt', 'financial commitment', 'reserve requirement'],
            'asset': ['investment', 'holding', 'financial asset', 'portfolio'],
            'cash flow': ['cash flows', 'payment stream', 'income stream', 'financial flow'],
        }

    def _build_context_patterns(self) -> List[Dict[str, str]]:
        """Build context patterns for query understanding"""
        return [
            {'pattern': r'\bhow\s+(?:do\s+i|to)\s+calculate\b', 'context': 'calculation_request', 'expansion': 'step-by-step process methodology'},
            {'pattern': r'\bwhat\s+is\s+the\s+(?:formula|method)\b', 'context': 'formula_request', 'expansion': 'mathematical equation approach'},
            {'pattern': r'\bexplain\s+(?:the\s+)?(?:process|procedure)\b', 'context': 'explanation_request', 'expansion': 'detailed explanation methodology'},
            {'pattern': r'\brequirements?\s+for\b', 'context': 'requirement_query', 'expansion': 'regulatory standards compliance'},
            {'pattern': r'\bdifference\s+between\b', 'context': 'comparison_query', 'expansion': 'comparison contrast differences'},
            {'pattern': r'\bsteps?\s+(?:to|for)\b', 'context': 'procedure_request', 'expansion': 'step-by-step procedure process'},
        ]

    def expand_query(self, query: str) -> ExpandedQuery:
        """Expand a query with related terms and synonyms"""
        query_lower = query.lower().strip()

        logger.debug(f"Expanding query: {query}")

        # Find synonyms for key terms
        synonyms = self._find_synonyms(query_lower)

        # Add technical term expansions
        technical_expansions = self._find_technical_expansions(query_lower)

        # Detect context and add context-specific terms
        context_terms = self._detect_context_terms(query_lower)

        # Extract semantic keywords
        semantic_keywords = self._extract_semantic_keywords(query_lower)

        # Combine all expansions
        expanded_terms = []
        expanded_terms.extend(synonyms)
        expanded_terms.extend(technical_expansions)
        expanded_terms.extend(context_terms)

        # Remove duplicates while preserving order
        seen = set()
        unique_expanded_terms = []
        for term in expanded_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_expanded_terms.append(term)

        # Generate expansion notes
        expansion_notes = self._generate_expansion_notes(
            query_lower, synonyms, technical_expansions, context_terms
        )

        expanded_query = ExpandedQuery(
            original_query=query,
            expanded_terms=unique_expanded_terms,
            synonyms=synonyms,
            context_terms=context_terms,
            semantic_keywords=semantic_keywords,
            expansion_notes=expansion_notes
        )

        logger.info(f"Query expansion complete",
                   original_length=len(query.split()),
                   expanded_terms_count=len(unique_expanded_terms),
                   synonyms_count=len(synonyms))

        return expanded_query

    def _find_synonyms(self, query: str) -> List[str]:
        """Find synonyms for terms in the query"""
        synonyms = []

        for term, synonym_list in self.domain_synonyms.items():
            # Use word boundaries for more precise matching
            pattern = r'\b' + re.escape(term.lower()) + r'\b'
            if re.search(pattern, query):
                synonyms.extend(synonym_list)

        return synonyms

    def _find_technical_expansions(self, query: str) -> List[str]:
        """Find technical term expansions"""
        expansions = []

        for term, expansion_list in self.technical_terms.items():
            pattern = r'\b' + re.escape(term.lower()) + r'\b'
            if re.search(pattern, query):
                expansions.extend(expansion_list)

        return expansions

    def _detect_context_terms(self, query: str) -> List[str]:
        """Detect context patterns and add relevant terms"""
        context_terms = []

        for pattern_info in self.context_patterns:
            if re.search(pattern_info['pattern'], query):
                context_terms.extend(pattern_info['expansion'].split())

        return context_terms

    def _extract_semantic_keywords(self, query: str) -> List[str]:
        """Extract important semantic terms from query"""
        semantic_terms = [
            'reserve', 'calculation', 'method', 'valuation', 'actuarial',
            'premium', 'liability', 'cash flow', 'mortality', 'interest',
            'lapse', 'surrender', 'benefit', 'policy', 'insurance'
        ]

        found_terms = []
        for term in semantic_terms:
            if re.search(rf'\b{term}\b', query):
                found_terms.append(term)

        return found_terms

    def _generate_expansion_notes(self, query: str, synonyms: List[str],
                                technical_expansions: List[str], context_terms: List[str]) -> List[str]:
        """Generate human-readable expansion notes"""
        notes = []

        if synonyms:
            notes.append(f"Added domain synonyms: {', '.join(synonyms[:3])}{'...' if len(synonyms) > 3 else ''}")

        if technical_expansions:
            notes.append(f"Added technical terms: {', '.join(technical_expansions[:3])}{'...' if len(technical_expansions) > 3 else ''}")

        if context_terms:
            notes.append(f"Added context terms: {', '.join(context_terms[:3])}{'...' if len(context_terms) > 3 else ''}")

        # Specific product detection
        if 'universal life' in query:
            notes.append("🎯 Universal Life query - added UL-specific terms")
        elif 'whole life' in query:
            notes.append("🎯 Whole Life query - added whole life-specific terms")

        if not notes:
            notes.append("Query used as-is - no expansion needed")

        return notes

    def get_expanded_query_string(self, expanded_query: ExpandedQuery,
                                mode: str = 'comprehensive') -> str:
        """Get expanded query as a string for search"""

        if mode == 'original_only':
            return expanded_query.original_query

        elif mode == 'synonyms_only':
            # Original + just synonyms
            terms = [expanded_query.original_query]
            terms.extend(expanded_query.synonyms)
            return ' '.join(terms)

        elif mode == 'conservative':
            # Original + just key product terms
            terms = [expanded_query.original_query]
            # Add only product-specific synonyms
            product_terms = [term for term in expanded_query.synonyms
                           if any(prod in term.lower() for prod in ['ul', 'universal', 'whole', 'term', 'life'])]
            terms.extend(product_terms[:3])  # Limit to 3 most relevant
            return ' '.join(terms)

        elif mode == 'comprehensive':
            # Original + all expansions
            terms = [expanded_query.original_query]
            terms.extend(expanded_query.expanded_terms)
            return ' '.join(terms)

        elif mode == 'semantic_boost':
            # Boost semantic keywords
            terms = [expanded_query.original_query]
            # Add semantic keywords with higher weight (repeat them)
            for keyword in expanded_query.semantic_keywords:
                terms.extend([keyword] * 2)  # Boost semantic terms
            terms.extend(expanded_query.synonyms)
            return ' '.join(terms)

        else:
            return expanded_query.original_query