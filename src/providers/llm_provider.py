"""
LLM Provider abstraction.

All LLM calls in the application go through this interface.
Swap providers (OpenAI, Anthropic, local models) by changing config/llm.yaml.

Usage:
    provider = get_llm_provider()  # singleton, reads config once

    # Simple text generation
    text = await provider.generate("Summarize this document", task="generation")

    # Classification (short response)
    label = await provider.classify("Is this about insurance?", ["yes", "no"], task="classification")

    # Structured JSON output
    data = await provider.generate_json("Extract entities", schema_hint="...", task="extraction")

    # Sync calls (for non-async contexts)
    text = provider.generate_sync("Summarize this", task="generation")
"""

import os
import json
import re
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
import structlog

logger = structlog.get_logger()

# Config path resolution
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
_LLM_CONFIG_PATH = _CONFIG_DIR / "llm.yaml"


def _load_llm_config() -> Dict[str, Any]:
    """Load LLM configuration from config/llm.yaml."""
    if _LLM_CONFIG_PATH.exists():
        with open(_LLM_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    logger.warning("llm.yaml not found, using defaults", path=str(_LLM_CONFIG_PATH))
    return {}


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        task: str = "generation",
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from a prompt. Task selects model/params from config."""

    @abstractmethod
    async def generate_json(
        self,
        prompt: str,
        *,
        task: str = "extraction",
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        schema_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate structured JSON output."""

    @abstractmethod
    async def classify(
        self,
        prompt: str,
        categories: List[str],
        *,
        task: str = "classification",
    ) -> str:
        """Classify text into one of the given categories."""

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        *,
        task: str = "generation",
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream text tokens from a prompt. Yields content strings as they arrive."""
        yield ""  # pragma: no cover — abstract

    @abstractmethod
    def generate_sync(
        self,
        prompt: str,
        *,
        task: str = "generation",
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Synchronous generation for non-async contexts."""

    @abstractmethod
    def get_model_name(self, task: str = "generation") -> str:
        """Return the model name configured for a given task."""

    @abstractmethod
    def get_task_params(self, task: str) -> Dict[str, Any]:
        """Return the full parameter dict for a task (model, temperature, max_tokens)."""


class OpenAILLMProvider(LLMProvider):
    """OpenAI-backed LLM provider."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or _load_llm_config()
        provider_config = self._config.get("providers", {}).get("openai", {})

        # Model mapping per task
        self._models = provider_config.get("models", {})
        self._params = provider_config.get("params", {})
        self._default_model = provider_config.get("default_model", "gpt-4o-mini")

        # Lazily initialized clients
        self._async_client = None
        self._sync_client = None

    def _get_async_client(self):
        if self._async_client is None:
            from openai import AsyncOpenAI
            self._async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._async_client

    def _get_sync_client(self):
        if self._sync_client is None:
            from openai import OpenAI
            self._sync_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._sync_client

    def get_model_name(self, task: str = "generation") -> str:
        return self._models.get(task, self._default_model)

    def get_task_params(self, task: str) -> Dict[str, Any]:
        params = self._params.get(task, {}).copy()
        params.setdefault("model", self.get_model_name(task))
        return params

    def _resolve_params(
        self,
        task: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> tuple:
        """Resolve model, temperature, max_tokens from config + overrides."""
        task_params = self._params.get(task, {})
        model = self.get_model_name(task)
        temp = temperature if temperature is not None else task_params.get("temperature", 0.1)
        tokens = max_tokens if max_tokens is not None else task_params.get("max_tokens", 1000)
        return model, temp, tokens

    async def generate(
        self,
        prompt: str,
        *,
        task: str = "generation",
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        model, temp, tokens = self._resolve_params(task, temperature, max_tokens)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        client = self._get_async_client()
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temp,
            max_tokens=tokens,
        )
        return response.choices[0].message.content.strip()

    async def generate_stream(
        self,
        prompt: str,
        *,
        task: str = "generation",
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        model, temp, tokens = self._resolve_params(task, temperature, max_tokens)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        client = self._get_async_client()
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temp,
            max_tokens=tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    async def generate_json(
        self,
        prompt: str,
        *,
        task: str = "extraction",
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        schema_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        model, temp, tokens = self._resolve_params(task, temperature, max_tokens)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        client = self._get_async_client()
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temp,
            max_tokens=tokens,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Fallback: try to extract JSON object from response text
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                return json.loads(match.group())
            raise e

    async def classify(
        self,
        prompt: str,
        categories: List[str],
        *,
        task: str = "classification",
    ) -> str:
        model, temp, _ = self._resolve_params(task, None, None)
        client = self._get_async_client()
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=20,
        )
        result = response.choices[0].message.content.strip().lower()
        # Return the closest matching category
        for cat in categories:
            if cat.lower() in result:
                return cat
        return result

    def generate_sync(
        self,
        prompt: str,
        *,
        task: str = "generation",
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        model, temp, tokens = self._resolve_params(task, temperature, max_tokens)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        client = self._get_sync_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temp,
            max_tokens=tokens,
        )
        return response.choices[0].message.content.strip()


# --- Singleton ---

_provider_instance: Optional[LLMProvider] = None


def get_llm_provider(config: Optional[Dict[str, Any]] = None) -> LLMProvider:
    """Get or create the singleton LLM provider.

    Reads config/llm.yaml on first call. Provider type is determined by
    config.providers.default (currently only 'openai' is implemented).
    """
    global _provider_instance
    if _provider_instance is None:
        cfg = config or _load_llm_config()
        default_provider = cfg.get("providers", {}).get("default", "openai")
        if default_provider == "openai":
            _provider_instance = OpenAILLMProvider(cfg)
        else:
            raise ValueError(f"Unknown LLM provider: {default_provider}")
        logger.info(
            "LLM provider initialized",
            provider=default_provider,
            default_model=_provider_instance.get_model_name(),
        )
    return _provider_instance


def reset_llm_provider():
    """Reset the singleton (for testing)."""
    global _provider_instance
    _provider_instance = None
