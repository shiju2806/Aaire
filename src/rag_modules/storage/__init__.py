"""
Storage module for RAG pipeline components.

This module contains classes and utilities for managing document storage,
including vector stores, indexes, and document management operations.
"""

from .documents import DocumentManager, create_document_manager

__all__ = [
    "DocumentManager",
    "create_document_manager"
]