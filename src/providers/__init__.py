"""
Provider abstractions for LLM, Embedding, and Retrieval services.
Swap implementations via config without touching application code.
"""

from .llm_provider import LLMProvider, OpenAILLMProvider, get_llm_provider
from .retrieval_provider import (
    RetrievalProvider,
    QdrantProvider,
    get_retrieval_provider,
    SearchResult,
    CollectionConfig,
)
from .embedding_provider import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
)

__all__ = [
    "LLMProvider",
    "OpenAILLMProvider",
    "get_llm_provider",
    "RetrievalProvider",
    "QdrantProvider",
    "get_retrieval_provider",
    "SearchResult",
    "CollectionConfig",
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "get_embedding_provider",
]
