"""
RAG Services Module

Provides core service components for the RAG pipeline including:
- Document retrieval (hybrid search, vector search, keyword search)
- Response generation (LLM-based answer generation)
- Follow-up question generation
"""

from .retrieval import DocumentRetriever, create_document_retriever
from .generation import ResponseGenerator, create_response_generator
from .semantic_similarity import SemanticSimilarityService, create_semantic_similarity_service

__all__ = [
    'DocumentRetriever',
    'create_document_retriever',
    'ResponseGenerator',
    'create_response_generator',
    'SemanticSimilarityService',
    'create_semantic_similarity_service'
]