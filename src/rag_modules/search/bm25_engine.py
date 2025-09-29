"""
BM25 Search Engine

A robust BM25-based keyword search engine that replaces the Whoosh implementation.
This engine provides better ranking precision and fixes issues from the previous
broken BM25 implementation.
"""

import re
import uuid
import structlog
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from dataclasses import dataclass

logger = structlog.get_logger()


@dataclass
class SearchResult:
    """Search result with consistent interface matching Whoosh results"""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    score: float


class BM25SearchEngine:
    """
    High-performance BM25 search engine with robust tokenization and filtering.
    Designed to replace Whoosh with better precision and performance.
    """

    def __init__(self):
        """Initialize the BM25 search engine"""
        self.bm25_index = None
        self.documents = []  # Store document texts
        self.metadata = []   # Store document metadata with doc_ids
        self.doc_id_to_index = {}  # Map doc_id to document index
        self.is_ready = False
        logger.info("BM25SearchEngine initialized")

    def tokenize_text(self, text: str) -> List[str]:
        """
        Advanced tokenization for BM25 that handles technical/financial terms.
        Preserves important patterns like ASC codes, percentages, ratios, etc.
        """
        if not text or not isinstance(text, str):
            return []

        # Convert to lowercase for consistent matching
        text = text.lower()

        # Extract special patterns first (preserve as single tokens)
        special_patterns = []

        # ASC codes (e.g., "ASC 842-10-15-2")
        asc_pattern = r'\basc\s+\d{3}-\d{2}-\d{2}-\d{1,2}\b'
        special_patterns.extend(re.findall(asc_pattern, text))

        # Financial ratios and percentages
        ratio_pattern = r'\b\d+(?:\.\d+)?%?\b'
        special_patterns.extend(re.findall(ratio_pattern, text))

        # Technical terms with hyphens/underscores
        technical_pattern = r'\b[a-z]+[-_][a-z]+(?:[-_][a-z]+)*\b'
        special_patterns.extend(re.findall(technical_pattern, text))

        # Regular word tokenization (alphanumeric + some special chars)
        word_pattern = r'\b[a-z0-9]+(?:[\.\-][a-z0-9]+)*\b'
        regular_tokens = re.findall(word_pattern, text)

        # Combine special patterns and regular tokens
        all_tokens = special_patterns + regular_tokens

        # Remove duplicates while preserving order
        seen = set()
        unique_tokens = []
        for token in all_tokens:
            if token not in seen and len(token) > 1:  # Filter out single chars
                seen.add(token)
                unique_tokens.append(token)

        return unique_tokens

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the BM25 index.
        Expected document format: {'content': str, 'metadata': dict}
        """
        try:
            if not documents:
                logger.warning("No documents provided to BM25 engine")
                return

            new_docs = []
            new_metadata = []

            for doc in documents:
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})

                if not content or not isinstance(content, str):
                    continue

                # Generate or use existing doc_id
                doc_id = metadata.get('node_id') or metadata.get('doc_id') or str(uuid.uuid4())

                # Store document content and metadata
                new_docs.append(content)
                doc_metadata = metadata.copy()
                doc_metadata['doc_id'] = doc_id
                new_metadata.append(doc_metadata)

                # Map doc_id to index for quick lookup
                self.doc_id_to_index[doc_id] = len(self.documents) + len(new_docs) - 1

            # Add to existing collections
            self.documents.extend(new_docs)
            self.metadata.extend(new_metadata)

            # Rebuild BM25 index with all documents
            self._rebuild_index()

            logger.info(f"Added {len(new_docs)} documents to BM25 index. Total: {len(self.documents)}")

        except Exception as e:
            logger.error("Failed to add documents to BM25 index", error=str(e))
            raise

    def _rebuild_index(self) -> None:
        """Rebuild the BM25 index with all current documents"""
        try:
            if not self.documents:
                logger.warning("No documents available for BM25 indexing")
                self.is_ready = False
                return

            # Tokenize all documents
            tokenized_docs = []
            for i, doc in enumerate(self.documents):
                tokens = self.tokenize_text(doc)
                if not tokens:
                    logger.warning(f"Document {i} produced no tokens: {doc[:50]}...")
                tokenized_docs.append(tokens)

            # Build BM25 index
            self.bm25_index = BM25Okapi(tokenized_docs)
            self.is_ready = True

            logger.info(f"BM25 index rebuilt with {len(tokenized_docs)} documents")

        except Exception as e:
            logger.error("Failed to rebuild BM25 index", error=str(e))
            self.is_ready = False
            raise

    def search(self, query: str, filters: Optional[Dict[str, Any]] = None,
               limit: int = 20, highlight: bool = False) -> List[SearchResult]:
        """
        Search documents using BM25 ranking.

        Args:
            query: Search query string
            filters: Optional filters to apply (doc_type, job_id, etc.)
            limit: Maximum number of results to return
            highlight: Not used (for Whoosh compatibility)

        Returns:
            List of SearchResult objects
        """
        try:
            if not self.is_ready or not self.bm25_index:
                logger.warning("BM25 index not ready for search")
                return []

            if not query or not isinstance(query, str):
                logger.warning("Invalid query provided to BM25 search")
                return []

            # Tokenize the query
            query_tokens = self.tokenize_text(query)
            if not query_tokens:
                logger.warning(f"Query produced no tokens: {query}")
                return []

            # Get BM25 scores for all documents
            scores = self.bm25_index.get_scores(query_tokens)

            # Create results with scores and metadata
            results = []
            for i, score in enumerate(scores):
                if score > 0 and i < len(self.metadata):  # Only include docs with positive scores
                    doc_metadata = self.metadata[i]
                    doc_content = self.documents[i]

                    # Apply filters if specified
                    if filters and not self._passes_filters(doc_metadata, filters):
                        continue

                    # Create search result
                    result = SearchResult(
                        doc_id=doc_metadata['doc_id'],
                        content=doc_content,
                        metadata=doc_metadata,
                        score=float(score)
                    )
                    results.append(result)

            # Sort by BM25 score (descending) and apply limit
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:limit]

            logger.info(f"BM25 search found {len(results)} results for query: '{query[:30]}...'")

            # Debug: Show top 3 results
            for i, result in enumerate(results[:3]):
                content_preview = result.content[:80].replace('\n', ' ')
                logger.debug(f"BM25 result {i+1}: score={result.score:.3f}, preview='{content_preview}...'")

            return results

        except Exception as e:
            logger.error("BM25 search failed", error=str(e), query=query[:50])
            return []

    def _passes_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document metadata passes the specified filters"""
        try:
            for filter_key, filter_value in filters.items():
                if filter_key.startswith('_'):  # Skip special filter keys
                    continue

                meta_value = metadata.get(filter_key)

                # Handle list filters (e.g., document_type can be a list)
                if isinstance(filter_value, list):
                    if meta_value not in filter_value:
                        return False
                # Handle exact matches
                elif meta_value != filter_value:
                    # Special handling for context_tags (if it's a list in metadata)
                    if filter_key == 'context_tags' and isinstance(meta_value, list):
                        if filter_value not in meta_value:
                            return False
                    else:
                        return False

            return True

        except Exception as e:
            logger.warning(f"Filter check failed: {e}")
            return True  # Default to including document if filter check fails

    def clear(self) -> None:
        """Clear all documents and reset the index"""
        self.documents.clear()
        self.metadata.clear()
        self.doc_id_to_index.clear()
        self.bm25_index = None
        self.is_ready = False
        logger.info("BM25 index cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the search engine"""
        return {
            'total_documents': len(self.documents),
            'is_ready': self.is_ready,
            'has_index': self.bm25_index is not None
        }


def create_bm25_search_engine() -> BM25SearchEngine:
    """Factory function to create a BM25SearchEngine instance"""
    return BM25SearchEngine()