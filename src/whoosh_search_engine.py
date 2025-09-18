"""
Advanced Whoosh Search Engine
Replaces the problematic BM25 implementation with a robust, scalable search solution
"""

import os
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from contextlib import contextmanager
import structlog

# Whoosh imports
from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID, KEYWORD, NUMERIC, STORED
from whoosh.analysis import StemmingAnalyzer, StandardAnalyzer
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.query import Query, Term, And, Or
from whoosh.writing import AsyncWriter
import whoosh.scoring

logger = structlog.get_logger()

@dataclass
class SearchResult:
    """Search result with metadata and scoring"""
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    highlights: Optional[str] = None

@dataclass
class SearchStats:
    """Search performance and indexing statistics"""
    total_docs: int
    search_time_ms: float
    indexing_time_ms: float
    last_update: float
    query_count: int
    average_query_time: float

class WhooshSearchEngine:
    """
    Production-ready search engine using Whoosh

    Features:
    - Incremental indexing (no full rebuilds)
    - Thread-safe operations
    - Persistent storage
    - Advanced filtering
    - Metadata integration
    - Performance monitoring
    """

    def __init__(self,
                 index_dir: str = "search_index",
                 analyzer_type: str = "stemming",
                 max_memory_mb: int = 256):
        """
        Initialize Whoosh search engine

        Args:
            index_dir: Directory to store search index
            analyzer_type: 'stemming' or 'standard' text analyzer
            max_memory_mb: Maximum memory for index operations
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)

        # Choose analyzer
        self.analyzer = (StemmingAnalyzer() if analyzer_type == "stemming"
                        else StandardAnalyzer())

        # Configuration
        self.max_memory_mb = max_memory_mb
        self.batch_size = 100

        # Thread safety
        self.lock = threading.RLock()
        self.writer_lock = threading.Lock()

        # Performance tracking
        self.stats = SearchStats(
            total_docs=0,
            search_time_ms=0,
            indexing_time_ms=0,
            last_update=time.time(),
            query_count=0,
            average_query_time=0
        )

        # Initialize index
        self.schema = self._create_schema()
        self.index = self._get_or_create_index()

        logger.info("WhooshSearchEngine initialized",
                   index_dir=str(self.index_dir),
                   analyzer=analyzer_type,
                   total_docs=self.get_document_count())

    def _create_schema(self) -> Schema:
        """Create Whoosh schema with comprehensive fields"""
        return Schema(
            # Primary fields
            doc_id=ID(stored=True, unique=True),
            content=TEXT(analyzer=self.analyzer, stored=True),
            title=TEXT(analyzer=self.analyzer, stored=True),

            # Metadata fields for filtering
            primary_framework=KEYWORD(stored=True),
            content_domains=KEYWORD(stored=True),
            document_type=KEYWORD(stored=True),
            file_path=STORED(),

            # Numeric fields
            chunk_index=NUMERIC(stored=True),
            confidence_score=NUMERIC(stored=True),
            timestamp=NUMERIC(stored=True),

            # Full metadata as JSON
            metadata_json=STORED()
        )

    def _get_or_create_index(self):
        """Get existing index or create new one"""
        if exists_in(str(self.index_dir)):
            logger.info("Opening existing Whoosh index")
            return open_dir(str(self.index_dir))
        else:
            logger.info("Creating new Whoosh index")
            return create_in(str(self.index_dir), self.schema)

    def add_documents(self,
                     documents: List[Dict[str, Any]],
                     batch_processing: bool = True) -> int:
        """
        Add multiple documents with batch processing

        Args:
            documents: List of document dictionaries
            batch_processing: Use batch commits for performance

        Returns:
            Number of documents successfully added
        """
        start_time = time.time()
        added_count = 0

        try:
            with self.lock:
                # Process in batches for memory efficiency
                for i in range(0, len(documents), self.batch_size):
                    batch = documents[i:i + self.batch_size]

                    with self._get_writer() as writer:
                        for doc in batch:
                            try:
                                self._add_single_document(writer, doc)
                                added_count += 1
                            except Exception as e:
                                logger.warning("Failed to index document",
                                             doc_id=doc.get('doc_id', 'unknown'),
                                             error=str(e))

                    # Commit batch
                    if batch_processing:
                        logger.debug(f"Committed batch {i//self.batch_size + 1}, "
                                   f"documents: {len(batch)}")

            # Update statistics
            indexing_time = (time.time() - start_time) * 1000
            self.stats.indexing_time_ms = indexing_time
            self.stats.total_docs = self.get_document_count()
            self.stats.last_update = time.time()

            logger.info("Documents indexed successfully",
                       added_count=added_count,
                       total_docs=self.stats.total_docs,
                       indexing_time_ms=f"{indexing_time:.2f}")

            return added_count

        except Exception as e:
            logger.error("Batch indexing failed", error=str(e))
            raise

    def _add_single_document(self, writer, doc: Dict[str, Any]):
        """Add a single document to the writer"""

        # Extract required fields
        doc_id = doc.get('doc_id', doc.get('id', f"doc_{time.time()}"))
        content = doc.get('content', doc.get('text', ''))

        if not content.strip():
            return  # Skip empty documents

        # Extract metadata
        metadata = doc.get('metadata', {})

        # Prepare document for indexing
        doc_fields = {
            'doc_id': doc_id,
            'content': content,
            'title': doc.get('title', ''),
            'chunk_index': doc.get('chunk_index', 0),
            'timestamp': time.time(),
            'metadata_json': str(metadata)  # Store full metadata as JSON
        }

        # Add metadata fields for filtering
        if 'primary_framework' in metadata:
            doc_fields['primary_framework'] = metadata['primary_framework']

        if 'content_domains' in metadata:
            # Handle list of domains
            domains = metadata['content_domains']
            if isinstance(domains, list):
                doc_fields['content_domains'] = ' '.join(domains)
            else:
                doc_fields['content_domains'] = str(domains)

        if 'document_type' in metadata:
            doc_fields['document_type'] = metadata['document_type']

        if 'file_path' in metadata:
            doc_fields['file_path'] = metadata['file_path']

        if 'confidence_score' in metadata:
            doc_fields['confidence_score'] = float(metadata['confidence_score'])

        # Update or add document
        writer.update_document(**doc_fields)

    @contextmanager
    def _get_writer(self):
        """Thread-safe writer context manager"""
        with self.writer_lock:
            writer = AsyncWriter(self.index,
                               writerargs={'limitmb': self.max_memory_mb, 'multisegment': True})
            try:
                yield writer
            finally:
                writer.commit()

    def search(self,
               query: str,
               filters: Optional[Dict[str, Any]] = None,
               limit: int = 10,
               highlight: bool = False) -> List[SearchResult]:
        """
        Advanced search with filtering and highlighting

        Args:
            query: Search query string
            filters: Dictionary of metadata filters
            limit: Maximum number of results
            highlight: Enable text highlighting

        Returns:
            List of SearchResult objects
        """
        start_time = time.time()

        try:
            with self.index.searcher() as searcher:
                # Parse query
                whoosh_query = self._build_query(query, filters)

                # Execute search
                results = searcher.search(whoosh_query, limit=limit)

                # Convert to SearchResult objects
                search_results = []
                for hit in results:
                    result = SearchResult(
                        doc_id=hit['doc_id'],
                        content=hit['content'],
                        score=hit.score,
                        metadata=self._parse_metadata(hit),
                        highlights=self._get_highlights(hit, query) if highlight else None
                    )
                    search_results.append(result)

                # Update statistics
                search_time = (time.time() - start_time) * 1000
                self._update_search_stats(search_time)

                logger.info("Search completed",
                           query=query[:50],
                           results_count=len(search_results),
                           search_time_ms=f"{search_time:.2f}")

                return search_results

        except Exception as e:
            logger.error("Search failed", query=query, error=str(e))
            return []

    def _build_query(self, query_str: str, filters: Optional[Dict[str, Any]]) -> Query:
        """Build Whoosh query with filters"""

        # Parse main content query
        parser = MultifieldParser(['content', 'title'], self.index.schema)
        content_query = parser.parse(query_str)

        # Add filters
        if not filters:
            return content_query

        filter_queries = []

        # Framework filter
        if 'primary_framework' in filters:
            frameworks = filters['primary_framework']
            if isinstance(frameworks, list):
                framework_queries = [Term('primary_framework', fw) for fw in frameworks]
                if framework_queries:
                    filter_queries.append(Or(framework_queries))
            else:
                filter_queries.append(Term('primary_framework', frameworks))

        # Content domains filter
        if 'content_domains' in filters:
            domains = filters['content_domains']
            if isinstance(domains, list):
                domain_queries = [Term('content_domains', domain) for domain in domains]
                if domain_queries:
                    filter_queries.append(Or(domain_queries))
            else:
                filter_queries.append(Term('content_domains', domains))

        # Document type filter
        if 'document_type' in filters:
            filter_queries.append(Term('document_type', filters['document_type']))

        # Combine content query with filters
        if filter_queries:
            return And([content_query] + filter_queries)
        else:
            return content_query

    def _parse_metadata(self, hit) -> Dict[str, Any]:
        """Parse metadata from search hit"""
        metadata = {}

        # Add stored fields
        for field in ['primary_framework', 'content_domains', 'document_type',
                     'file_path', 'chunk_index', 'confidence_score', 'timestamp']:
            if field in hit:
                metadata[field] = hit[field]

        # Parse JSON metadata if available
        if 'metadata_json' in hit:
            try:
                import json
                json_metadata = json.loads(hit['metadata_json'])
                metadata.update(json_metadata)
            except:
                pass  # Ignore JSON parsing errors

        return metadata

    def _get_highlights(self, hit, query: str) -> Optional[str]:
        """Get highlighted text snippets"""
        try:
            return hit.highlights('content', top=3)
        except:
            return None

    def _update_search_stats(self, search_time_ms: float):
        """Update search performance statistics"""
        self.stats.query_count += 1
        total_time = self.stats.average_query_time * (self.stats.query_count - 1) + search_time_ms
        self.stats.average_query_time = total_time / self.stats.query_count
        self.stats.search_time_ms = search_time_ms

    def get_document_count(self) -> int:
        """Get total number of indexed documents"""
        try:
            with self.index.searcher() as searcher:
                return searcher.doc_count()
        except:
            return 0

    def delete_documents(self, doc_ids: List[str]) -> int:
        """Delete documents by ID"""
        deleted_count = 0

        try:
            with self._get_writer() as writer:
                for doc_id in doc_ids:
                    writer.delete_by_term('doc_id', doc_id)
                    deleted_count += 1

            self.stats.total_docs = self.get_document_count()
            logger.info("Documents deleted", count=deleted_count)
            return deleted_count

        except Exception as e:
            logger.error("Document deletion failed", error=str(e))
            return 0

    def optimize_index(self):
        """Optimize index for better performance"""
        try:
            with self.writer_lock:
                writer = self.index.writer()
                writer.commit(optimize=True)
            logger.info("Index optimization completed")
        except Exception as e:
            logger.error("Index optimization failed", error=str(e))

    def get_stats(self) -> SearchStats:
        """Get current search engine statistics"""
        self.stats.total_docs = self.get_document_count()
        return self.stats

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            # Test basic operations
            start_time = time.time()
            doc_count = self.get_document_count()
            index_access_time = (time.time() - start_time) * 1000

            # Test search
            start_time = time.time()
            test_results = self.search("test", limit=1)
            search_access_time = (time.time() - start_time) * 1000

            return {
                "status": "healthy",
                "total_documents": doc_count,
                "index_access_time_ms": index_access_time,
                "search_access_time_ms": search_access_time,
                "index_directory": str(self.index_dir),
                "index_exists": self.index_dir.exists(),
                "average_query_time_ms": self.stats.average_query_time,
                "total_queries": self.stats.query_count
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "index_directory": str(self.index_dir)
            }

# Factory function for easy integration
def create_search_engine(index_dir: str = "search_index", **kwargs) -> WhooshSearchEngine:
    """Create a new search engine instance"""
    return WhooshSearchEngine(index_dir=index_dir, **kwargs)