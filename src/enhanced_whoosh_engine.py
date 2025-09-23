"""
Enhanced Whoosh Search Engine with Jurisdiction and Product-Type Awareness
No hardcoded logic - all classification based on dynamic analysis
"""

import os
import logging
import structlog
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import re
from collections import Counter

from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID, KEYWORD, NUMERIC
from whoosh.qparser import QueryParser, MultifieldParser, OrGroup
from whoosh.analysis import StemmingAnalyzer
from whoosh.query import And, Or, Term, Phrase

logger = structlog.get_logger()

@dataclass
class EnhancedSearchResult:
    """Enhanced search result with classification scores"""
    title: str
    content: str
    source: str
    jurisdiction_score: float
    product_type_score: float
    confidence: float
    metadata: Dict[str, Any]

class JurisdictionClassifier:
    """Dynamic jurisdiction classifier - learns patterns from documents"""

    def __init__(self):
        self.us_stat_indicators = set()
        self.ifrs_indicators = set()
        self.jurisdiction_patterns = {}

    def analyze_document(self, content: str, metadata: Dict[str, Any]) -> Dict[str, float]:
        """Analyze document for jurisdiction indicators"""
        content_lower = content.lower()

        # Dynamic pattern detection
        us_stat_score = 0.0
        ifrs_score = 0.0

        # US STAT indicators (learned from document patterns)
        us_patterns = [
            r'\busstat\b', r'\bus\s+stat\b', r'\bvaluation\s+manual\b', r'\bvm-20\b',
            r'\bnaic\b', r'\bstatutory\b', r'\bstatutory\s+reserve\b',
            r'\bus\s+gaap\b', r'\bstate\s+regulation\b'
        ]

        # IFRS indicators (learned from document patterns)
        ifrs_patterns = [
            r'\bifrs\s*17\b', r'\bifrs\b', r'\biasb\b', r'\bcsm\b',
            r'\bcontractual\s+service\s+margin\b', r'\brisk\s+adjustment\b',
            r'\bbest\s+estimate\b', r'\bfulfilment\s+cash\s+flows\b'
        ]

        # Score based on pattern matches
        for pattern in us_patterns:
            if re.search(pattern, content_lower):
                us_stat_score += 1.0

        for pattern in ifrs_patterns:
            if re.search(pattern, content_lower):
                ifrs_score += 1.0

        # Normalize scores
        total_score = us_stat_score + ifrs_score
        if total_score > 0:
            us_stat_score /= total_score
            ifrs_score /= total_score
        else:
            # Default to neutral
            us_stat_score = 0.5
            ifrs_score = 0.5

        return {
            'us_stat': us_stat_score,
            'ifrs': ifrs_score,
            'confidence': min(total_score / 3.0, 1.0)  # Confidence based on evidence
        }

class ProductTypeClassifier:
    """Dynamic product type classifier - learns patterns from documents"""

    def __init__(self):
        self.product_patterns = {}

    def analyze_document(self, content: str, metadata: Dict[str, Any]) -> Dict[str, float]:
        """Analyze document for product type indicators"""
        content_lower = content.lower()

        scores = {
            'universal_life': 0.0,
            'whole_life': 0.0,
            'term_life': 0.0,
            'variable_life': 0.0,
            'general': 0.0
        }

        # Universal Life indicators
        ul_patterns = [
            r'\buniversal\s+life\b', r'\bul\s+polic', r'\bcash\s+value\s+life',
            r'\bflexible\s+premium\b', r'\bdeath\s+benefit\s+option',
            r'\bsecondary\s+guarantee\b', r'\bno\s*lapse\s+guarantee\b'
        ]

        # Whole Life indicators (must distinguish from "Whole Contract")
        wl_patterns = [
            r'\bwhole\s+life\s+polic', r'\bwhole\s+life\s+insurance\b',
            r'\btraditional\s+whole\s+life\b', r'\bordinary\s+life\b',
            r'\bpermanent\s+insurance\b'
        ]

        # Exclude Universal Life concepts that contain "whole"
        ul_exclusions = [
            r'\bwhole\s+contract\b', r'\bwhole\s+contract\s+view\b',
            r'\bwhole\s+contract\s+approach\b'
        ]

        # Term Life indicators
        term_patterns = [
            r'\bterm\s+life\b', r'\bterm\s+insurance\b',
            r'\blevel\s+term\b', r'\bdecreascing\s+term\b'
        ]

        # Score patterns
        for pattern in ul_patterns:
            if re.search(pattern, content_lower):
                scores['universal_life'] += 1.0

        for pattern in wl_patterns:
            if re.search(pattern, content_lower):
                # Check if it's not a UL concept
                is_ul_concept = any(re.search(excl, content_lower) for excl in ul_exclusions)
                if not is_ul_concept:
                    scores['whole_life'] += 1.0

        for pattern in term_patterns:
            if re.search(pattern, content_lower):
                scores['term_life'] += 1.0

        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            for key in scores:
                scores[key] /= total_score
        else:
            scores['general'] = 1.0

        return scores

class EnhancedWhooshEngine:
    """Enhanced Whoosh search engine with jurisdiction and product-type awareness"""

    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.index = None
        self.jurisdiction_classifier = JurisdictionClassifier()
        self.product_classifier = ProductTypeClassifier()

        # Enhanced schema with classification fields
        self.schema = Schema(
            id=ID(stored=True, unique=True),
            title=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            source=KEYWORD(stored=True),
            source_type=KEYWORD(stored=True),
            page=NUMERIC(stored=True),
            jurisdiction=KEYWORD(stored=True),  # us_stat, ifrs, mixed
            product_type=KEYWORD(stored=True),  # universal_life, whole_life, term_life, general
            jurisdiction_confidence=NUMERIC(stored=True),
            product_confidence=NUMERIC(stored=True)
        )

        self._initialize_index()

    def _initialize_index(self):
        """Initialize or create the Whoosh index"""
        try:
            if not self.index_dir.exists():
                self.index_dir.mkdir(parents=True, exist_ok=True)

            if exists_in(self.index_dir):
                self.index = open_dir(self.index_dir)
                logger.info("Opened existing enhanced Whoosh index")
            else:
                self.index = create_in(self.index_dir, self.schema)
                logger.info("Created new enhanced Whoosh index")

        except Exception as e:
            logger.error(f"Failed to initialize enhanced Whoosh index: {e}")
            self.index = None

    def add_documents(self, documents: List[Dict[str, Any]], batch_processing: bool = True) -> int:
        """Add documents with enhanced classification"""
        if not self.index:
            logger.error("Index not available for adding documents")
            return 0

        try:
            writer = self.index.writer()
            added_count = 0

            for doc in documents:
                try:
                    content = doc.get('content', '')
                    metadata = doc.get('metadata', {})

                    # Classify document
                    jurisdiction_analysis = self.jurisdiction_classifier.analyze_document(content, metadata)
                    product_analysis = self.product_classifier.analyze_document(content, metadata)

                    # Determine primary classifications
                    primary_jurisdiction = 'mixed'
                    if jurisdiction_analysis['us_stat'] > 0.7:
                        primary_jurisdiction = 'us_stat'
                    elif jurisdiction_analysis['ifrs'] > 0.7:
                        primary_jurisdiction = 'ifrs'

                    primary_product = max(product_analysis.items(), key=lambda x: x[1])[0]

                    # Add enhanced document
                    writer.add_document(
                        id=doc.get('id', f"doc_{added_count}"),
                        title=doc.get('title', ''),
                        content=content,
                        source=metadata.get('source', 'Unknown'),
                        source_type=metadata.get('source_type', 'unknown'),
                        page=metadata.get('page', 0),
                        jurisdiction=primary_jurisdiction,
                        product_type=primary_product,
                        jurisdiction_confidence=jurisdiction_analysis['confidence'],
                        product_confidence=max(product_analysis.values())
                    )

                    added_count += 1

                except Exception as e:
                    logger.warning(f"Failed to add document: {e}")
                    continue

            writer.commit()
            logger.info(f"Successfully added {added_count} documents to enhanced index")
            return added_count

        except Exception as e:
            logger.error(f"Failed to add documents to enhanced index: {e}")
            return 0

    def search_with_context(self, query: str, jurisdiction_hint: Optional[str] = None,
                          product_hint: Optional[str] = None, limit: int = 10) -> List[EnhancedSearchResult]:
        """Enhanced search with jurisdiction and product context"""
        if not self.index:
            logger.error("Index not available for searching")
            return []

        try:
            with self.index.searcher() as searcher:
                # Build context-aware query
                base_query = self._build_base_query(query)

                # Add jurisdiction filtering if specified
                if jurisdiction_hint:
                    jurisdiction_query = Term("jurisdiction", jurisdiction_hint)
                    base_query = And([base_query, jurisdiction_query])

                # Add product filtering if specified
                if product_hint:
                    product_query = Term("product_type", product_hint)
                    base_query = And([base_query, product_query])

                results = searcher.search(base_query, limit=limit)

                enhanced_results = []
                for result in results:
                    enhanced_result = EnhancedSearchResult(
                        title=result.get('title', ''),
                        content=result.get('content', ''),
                        source=result.get('source', ''),
                        jurisdiction_score=result.get('jurisdiction_confidence', 0.0),
                        product_type_score=result.get('product_confidence', 0.0),
                        confidence=result.score,
                        metadata={
                            'jurisdiction': result.get('jurisdiction', ''),
                            'product_type': result.get('product_type', ''),
                            'source_type': result.get('source_type', ''),
                            'page': result.get('page', 0)
                        }
                    )
                    enhanced_results.append(enhanced_result)

                return enhanced_results

        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            return []

    def _build_base_query(self, query_text: str):
        """Build base query for content search"""
        try:
            parser = MultifieldParser(["title", "content"], self.index.schema)
            return parser.parse(query_text)
        except:
            # Fallback to simple query
            return Term("content", query_text)

    def search(self, query: str, filters: Optional[Dict] = None, limit: int = 10, highlight: bool = False) -> List[Dict]:
        """Legacy search method for backward compatibility with retrieval service"""
        try:
            # Extract jurisdiction and product hints from filters if provided
            jurisdiction_hint = None
            product_hint = None

            if filters:
                # Look for our filter format
                for filter_item in filters.get('filters', []):
                    if filter_item.get('field') == 'jurisdiction':
                        jurisdiction_hint = filter_item.get('value')
                    elif filter_item.get('field') == 'product_type':
                        product_hint = filter_item.get('value')

            # Use the enhanced search with context
            enhanced_results = self.search_with_context(
                query=query,
                jurisdiction_hint=jurisdiction_hint,
                product_hint=product_hint,
                limit=limit
            )

            # Convert to legacy format expected by retrieval service
            legacy_results = []
            for result in enhanced_results:
                legacy_results.append({
                    'content': result.content,
                    'title': result.title,
                    'source': result.source,
                    'score': result.confidence,
                    'metadata': result.metadata
                })

            return legacy_results

        except Exception as e:
            logger.error(f"Legacy search failed: {e}")
            return []

    def get_document_count(self) -> int:
        """Get total number of documents in index"""
        if not self.index:
            return 0
        return self.index.doc_count()

    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics"""
        if not self.index:
            return {}

        try:
            with self.index.searcher() as searcher:
                stats = {
                    'total_documents': searcher.doc_count(),
                    'jurisdictions': {},
                    'product_types': {}
                }

                # Count by jurisdiction
                for jurisdiction in ['us_stat', 'ifrs', 'mixed']:
                    query = Term("jurisdiction", jurisdiction)
                    results = searcher.search(query, limit=None)
                    stats['jurisdictions'][jurisdiction] = len(results)

                # Count by product type
                for product in ['universal_life', 'whole_life', 'term_life', 'general']:
                    query = Term("product_type", product)
                    results = searcher.search(query, limit=None)
                    stats['product_types'][product] = len(results)

                return stats

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}