"""
Search Module

Provides search engines for the RAG pipeline including:
- Elasticsearch keyword search (persistent, scalable)
- BM25 keyword search (in-memory fallback)
- Search utilities and interfaces
"""

from .bm25_engine import SearchResult

# Try Elasticsearch first; fall back to in-memory BM25 if unavailable.
try:
    from .elasticsearch_engine import ElasticsearchEngine, create_elasticsearch_engine

    def create_search_engine(**kwargs):
        """Create the default search engine (Elasticsearch)."""
        return create_elasticsearch_engine(**kwargs)

    __all__ = [
        'SearchResult',
        'ElasticsearchEngine',
        'create_elasticsearch_engine',
        'create_search_engine',
    ]
except ImportError:
    from .bm25_engine import BM25SearchEngine, create_bm25_search_engine

    def create_search_engine(**kwargs):
        """Fallback: create in-memory BM25 search engine."""
        return create_bm25_search_engine()

    __all__ = [
        'SearchResult',
        'BM25SearchEngine',
        'create_bm25_search_engine',
        'create_search_engine',
    ]
