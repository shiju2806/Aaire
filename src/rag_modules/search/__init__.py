"""
Search Module

Provides search engines for the RAG pipeline including:
- BM25 keyword search (replaces Whoosh)
- Search utilities and interfaces
"""

from .bm25_engine import BM25SearchEngine, create_bm25_search_engine

__all__ = [
    'BM25SearchEngine',
    'create_bm25_search_engine'
]