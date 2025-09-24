"""
Simple Whoosh Search Engine - Pure keyword search without over-engineered classification
Complements vector search with exact term matching
"""

import os
import logging
import structlog
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID, KEYWORD, NUMERIC
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.analysis import StemmingAnalyzer

logger = structlog.get_logger()

@dataclass
class SearchResult:
    """Simple search result without over-engineered classification"""
    title: str
    content: str
    source: str
    confidence: float
    metadata: Dict[str, Any]

class SimpleWhooshEngine:
    """Simple Whoosh search engine - pure keyword search without classification"""

    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.index = None

        # Basic schema without over-engineered classification
        self.schema = Schema(
            id=ID(stored=True, unique=True),
            title=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            source=KEYWORD(stored=True),
            source_type=KEYWORD(stored=True),
            page=NUMERIC(stored=True)
        )

        self._initialize_index()

    def _initialize_index(self):
        """Initialize or create the Whoosh index"""
        try:
            if not self.index_dir.exists():
                self.index_dir.mkdir(parents=True, exist_ok=True)

            if exists_in(self.index_dir):
                self.index = open_dir(self.index_dir)
                logger.info("Opened existing Whoosh index")
            else:
                self.index = create_in(self.index_dir, self.schema)
                logger.info("Created new Whoosh index")

        except Exception as e:
            logger.error(f"Failed to initialize Whoosh index: {e}")
            self.index = None

    def add_documents(self, documents: List[Dict[str, Any]], batch_processing: bool = True) -> int:
        """Add documents without classification"""
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

                    # Add document without classification
                    writer.add_document(
                        id=doc.get('id', f"doc_{added_count}"),
                        title=doc.get('title', ''),
                        content=content,
                        source=metadata.get('source', 'Unknown'),
                        source_type=metadata.get('source_type', 'unknown'),
                        page=metadata.get('page', 0)
                    )

                    added_count += 1

                except Exception as e:
                    logger.warning(f"Failed to add document: {e}")
                    continue

            writer.commit()
            logger.info(f"Successfully added {added_count} documents to index")
            return added_count

        except Exception as e:
            logger.error(f"Failed to add documents to index: {e}")
            return 0

    def search_simple(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Simple keyword search without over-engineered filtering"""
        if not self.index:
            logger.error("Index not available for searching")
            return []

        try:
            with self.index.searcher() as searcher:
                # Build simple query
                base_query = self._build_base_query(query)
                results = searcher.search(base_query, limit=limit)

                simple_results = []
                for result in results:
                    simple_result = SearchResult(
                        title=result.get('title', ''),
                        content=result.get('content', ''),
                        source=result.get('source', ''),
                        confidence=result.score,
                        metadata={
                            'source_type': result.get('source_type', ''),
                            'page': result.get('page', 0)
                        }
                    )
                    simple_results.append(simple_result)

                return simple_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _build_base_query(self, query_text: str):
        """Build base query for content search"""
        try:
            parser = MultifieldParser(["title", "content"], self.index.schema)
            return parser.parse(query_text)
        except:
            # Fallback to simple query
            from whoosh.query import Term
            return Term("content", query_text)

    def search(self, query: str, filters: Optional[Dict] = None, limit: int = 10, highlight: bool = False) -> List[Dict]:
        """Main search method for retrieval service compatibility"""
        try:
            # Use simple search without filtering
            simple_results = self.search_simple(
                query=query,
                limit=limit
            )

            # Convert to format expected by retrieval service
            results = []
            for result in simple_results:
                results.append({
                    'content': result.content,
                    'title': result.title,
                    'source': result.source,
                    'score': result.confidence,
                    'metadata': result.metadata
                })

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
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
                    'total_documents': searcher.doc_count()
                }

                return stats

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}