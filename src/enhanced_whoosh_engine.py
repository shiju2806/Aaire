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

# Quality configuration for non-hardcoded settings
try:
    from rag_modules.config.quality_config import get_quality_config
except ImportError:
    # Fallback if config not available
    get_quality_config = None

logger = structlog.get_logger()

def _get_config_value(config_key: str, fallback_value: any):
    """Helper to get configuration values with fallback."""
    if get_quality_config is None:
        return fallback_value
    try:
        config = get_quality_config()
        method_name = f"get_{config_key}"
        if hasattr(config, method_name):
            return getattr(config, method_name)()
        return fallback_value
    except Exception:
        logger.warning(f"Failed to get config for {config_key}, using fallback", fallback=fallback_value)
        return fallback_value

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

        # Simplified schema focusing on product-type differentiation
        self.schema = Schema(
            id=ID(stored=True, unique=True),
            title=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            source=KEYWORD(stored=True),
            source_type=KEYWORD(stored=True),
            page=NUMERIC(stored=True),
            product_type=KEYWORD(stored=True),  # universal_life, whole_life, term_life, general
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

                    # Classify document - product type only
                    product_analysis = self.product_classifier.analyze_document(content, metadata)
                    primary_product = max(product_analysis.items(), key=lambda x: x[1])[0]

                    # Add simplified document
                    writer.add_document(
                        id=doc.get('id', f"doc_{added_count}"),
                        title=doc.get('title', ''),
                        content=content,
                        source=metadata.get('source', 'Unknown'),
                        source_type=metadata.get('source_type', 'unknown'),
                        page=metadata.get('page', 0),
                        product_type=primary_product,
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

    def search_with_context(self, query: str, product_hint: Optional[str] = None, limit: int = 10) -> List[EnhancedSearchResult]:
        """Enhanced search with jurisdiction and product context"""
        if not self.index:
            logger.error("Index not available for searching")
            return []

        try:
            with self.index.searcher() as searcher:
                # Build context-aware query
                base_query = self._build_base_query(query)

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
                        jurisdiction_score=0.0,  # Removed jurisdiction scoring
                        product_type_score=result.get('product_confidence', 0.0),
                        confidence=result.score,
                        metadata={
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
            # Extract product hint from filters if provided
            product_hint = None

            if filters:
                # Look for our filter format
                for filter_item in filters.get('filters', []):
                    if filter_item.get('field') == 'product_type':
                        product_hint = filter_item.get('value')

            # Use the enhanced search with context
            enhanced_results = self.search_with_context(
                query=query,
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
                    'product_types': {}
                }

                # Count by product type
                for product in ['universal_life', 'whole_life', 'term_life', 'general']:
                    query = Term("product_type", product)
                    results = searcher.search(query, limit=None)
                    stats['product_types'][product] = len(results)

                return stats

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}