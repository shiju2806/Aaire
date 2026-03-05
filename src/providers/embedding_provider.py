"""
Embedding Provider abstraction.

Centralizes embedding model configuration so changing the model or
dimension is a single config edit. The primary consumer is LlamaIndex's
Settings.embed_model, but this provider can also be used directly for
custom embedding needs outside LlamaIndex.

Usage:
    provider = get_embedding_provider()

    # Get config for LlamaIndex integration
    model_name = provider.model_name       # "text-embedding-3-large"
    dimension = provider.dimension          # 1536

    # Direct embedding (bypasses LlamaIndex)
    vector = await provider.embed_text("some text")
    vectors = await provider.embed_batch(["text1", "text2"])
"""

import os
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
import structlog

logger = structlog.get_logger()

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
_LLM_CONFIG_PATH = _CONFIG_DIR / "llm.yaml"


def _load_embedding_config() -> Dict[str, Any]:
    """Load embedding config from config/llm.yaml (embedding section)."""
    if _LLM_CONFIG_PATH.exists():
        with open(_LLM_CONFIG_PATH) as f:
            config = yaml.safe_load(f) or {}
        return config.get("embedding", {})
    return {}


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the embedding model name."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""

    @abstractmethod
    def get_llama_index_embedding(self) -> Any:
        """Return a LlamaIndex-compatible embedding object.

        This is for integration with LlamaIndex's Settings.embed_model.
        Returns an OpenAIEmbedding (or equivalent) instance.
        """


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-backed embedding provider."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or _load_embedding_config()
        self._model_name = self._config.get("model", "text-embedding-3-large")
        self._dimension = self._config.get("dimension", 1536)
        self._async_client = None
        self._llama_embedding = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_async_client(self):
        if self._async_client is None:
            from openai import AsyncOpenAI
            self._async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._async_client

    async def embed_text(self, text: str) -> List[float]:
        client = self._get_async_client()
        response = await client.embeddings.create(
            model=self._model_name,
            input=text,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        client = self._get_async_client()
        response = await client.embeddings.create(
            model=self._model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def get_llama_index_embedding(self) -> Any:
        """Return a LlamaIndex OpenAIEmbedding configured with our model."""
        if self._llama_embedding is None:
            from llama_index.embeddings.openai import OpenAIEmbedding
            self._llama_embedding = OpenAIEmbedding(
                model=self._model_name,
                dimensions=self._dimension,
            )
        return self._llama_embedding


# --- Singleton ---

_provider_instance: Optional[EmbeddingProvider] = None


def get_embedding_provider(config: Optional[Dict[str, Any]] = None) -> EmbeddingProvider:
    """Get or create the singleton embedding provider."""
    global _provider_instance
    if _provider_instance is None:
        cfg = config or _load_embedding_config()
        _provider_instance = OpenAIEmbeddingProvider(cfg)
        logger.info(
            "Embedding provider initialized",
            model=_provider_instance.model_name,
            dimension=_provider_instance.dimension,
        )
    return _provider_instance


def reset_embedding_provider():
    """Reset the singleton (for testing)."""
    global _provider_instance
    _provider_instance = None
