"""
Dependency Injection Container for AAIRE
Manages service instantiation and lifecycle with configuration-driven initialization
"""

import structlog
from typing import Dict, Any, Optional, TypeVar, Type, Callable, Union
from functools import lru_cache
from abc import ABC, abstractmethod
from ..config.quality_config import QualityConfig, get_quality_config

logger = structlog.get_logger()

T = TypeVar('T')


class ServiceContainer:
    """
    Dependency injection container that manages service instantiation and lifecycle.
    Provides configuration-driven initialization and singleton pattern support.
    """

    def __init__(self, config: Optional[QualityConfig] = None):
        """Initialize the service container with configuration."""
        self.config = config or get_quality_config()
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}

        self._register_default_factories()

        logger.info("Service container initialized",
                   environment=self.config.environment)

    def _register_default_factories(self):
        """Register default service factories."""
        # Model factories
        self._factories.update({
            'embedding_model': self._create_embedding_model,
            'llm_client': self._create_llm_client,
            'advanced_llm_client': self._create_advanced_llm_client,
            'async_llm_client': self._create_async_llm_client,

            # Validator factories
            'semantic_alignment_validator': self._create_semantic_alignment_validator,
            'grounding_validator': self._create_grounding_validator,
            'openai_alignment_validator': self._create_openai_alignment_validator,
            'unified_validator': self._create_unified_validator,

            # Core services
            'document_retriever': self._create_document_retriever,
            'response_generator': self._create_response_generator,
            'response_formatter': self._create_response_formatter,

            # Quality services (replacing old system)
            'quality_metrics_service': self._create_quality_metrics_service,
            'validation_service': self._create_validation_service,

            # Enhanced retrieval services
            'reflective_retriever': self._create_reflective_retriever,

            # Formatting services
            'formatting_manager': self._create_formatting_manager,

            # Enhanced grounding validation
            'enhanced_grounding_validator': self._create_enhanced_grounding_validator
        })

    def register_factory(self, service_name: str, factory: Callable[[], T]):
        """Register a custom factory for a service."""
        self._factories[service_name] = factory
        logger.debug("Factory registered", service=service_name)

    def register_singleton(self, service_name: str, instance: T):
        """Register a singleton instance."""
        self._singletons[service_name] = instance
        logger.debug("Singleton registered", service=service_name)

    def get(self, service_name: str, force_new: bool = False) -> Any:
        """
        Get a service instance.

        Args:
            service_name: Name of the service to retrieve
            force_new: Force creation of new instance (skip singleton cache)

        Returns:
            Service instance
        """
        # Check singleton cache first (unless force_new)
        if not force_new and service_name in self._singletons:
            return self._singletons[service_name]

        # Check if factory exists
        if service_name not in self._factories:
            raise ValueError(f"No factory registered for service: {service_name}")

        # Create instance using factory
        try:
            instance = self._factories[service_name]()
            logger.debug("Service created", service=service_name)
            return instance
        except Exception as e:
            logger.error("Failed to create service",
                        service=service_name, error=str(e))
            raise

    def get_singleton(self, service_name: str) -> Any:
        """Get or create a singleton instance."""
        if service_name not in self._singletons:
            self._singletons[service_name] = self.get(service_name)
        return self._singletons[service_name]

    # Model Factories
    def _create_embedding_model(self):
        """Create embedding model based on configuration."""
        from sentence_transformers import SentenceTransformer

        model_name = self.config.get_embedding_model()
        device = self.config.get_model_params('embedding').get('device', 'cpu')

        model = SentenceTransformer(model_name, device=device)

        logger.info("Embedding model created",
                   model=model_name, device=device)
        return model

    def _create_llm_client(self):
        """Create primary LLM client based on configuration."""
        from openai import OpenAI
        import os

        model_name = self.config.get_llm_model()
        model_params = self.config.get_model_params('llm')

        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # Store model configuration on client for easy access
        client.model_name = model_name
        client.model_params = model_params

        logger.info("LLM client created", model=model_name)
        return client

    def _create_advanced_llm_client(self):
        """Create advanced LLM client for complex tasks."""
        from openai import OpenAI
        import os

        model_name = self.config.get_advanced_llm_model()
        model_params = self.config.get_model_params('llm')

        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        client.model_name = model_name
        client.model_params = model_params

        logger.info("Advanced LLM client created", model=model_name)
        return client

    # Validator Factories
    def _create_semantic_alignment_validator(self):
        """Create semantic alignment validator with configuration."""
        from ..quality.semantic_alignment_validator import SemanticAlignmentValidator

        # Get embedding model name from configuration
        embedding_model_name = self.config.get_model_params('embedding').get('name', 'all-MiniLM-L6-v2')

        validator = SemanticAlignmentValidator(
            model_name=embedding_model_name,
            config=self.config
        )

        logger.info("Semantic alignment validator created")
        return validator

    def _create_grounding_validator(self):
        """Create grounding validator with configuration."""
        from ..quality.grounding_validator import ContentGroundingValidator

        validator = ContentGroundingValidator(
            learning_data_path="/tmp/rag_grounding_data.json",
            config=self.config
        )

        logger.info("Grounding validator created with configuration")
        return validator

    def _create_openai_alignment_validator(self):
        """Create OpenAI alignment validator if enabled."""
        if not self.config.is_openai_alignment_enabled():
            logger.info("OpenAI alignment validator disabled by feature flag")
            return None

        from ..quality.openai_alignment_validator import OpenAIAlignmentValidator

        # Get embedding model name from configuration
        embedding_model_name = self.config.get_model_params('embedding').get('name', 'text-embedding-ada-002')

        validator = OpenAIAlignmentValidator(
            model=embedding_model_name,
            config=self.config
        )

        logger.info("OpenAI alignment validator created")
        return validator


    def _create_unified_validator(self):
        """Create new unified quality validator (recommended approach)."""
        from ..quality.unified_validator import UnifiedQualityValidator

        validator = UnifiedQualityValidator(config=self.config)

        logger.info("Unified validator created")
        return validator

    def _create_quality_metrics_service(self):
        """Create quality metrics service (replacement for QualityMetricsManager)."""
        from ..services.quality_metrics_service import QualityMetricsService

        service = QualityMetricsService(config=self.config)

        logger.info("Quality metrics service created")
        return service

    def _create_validation_service(self):
        """Create validation service (replacement for IntelligentValidationSystem)."""
        from ..services.validation_service import ValidationService

        service = ValidationService(config=self.config)

        logger.info("Validation service created")
        return service

    # Core Service Factories
    def _create_document_retriever(self):
        """Create document retriever service."""
        # Implementation would depend on your document retrieval system
        logger.info("Document retriever placeholder created")
        return None

    def _create_response_generator(self):
        """Create response generator service with full dependency injection."""
        try:
            from ..services.generation import create_response_generator

            # Get required dependencies
            llm_client = self.get_singleton('llm_client')
            async_client = self.get_singleton('async_llm_client')
            formatting_manager = self.get_singleton('formatting_manager')
            grounding_validator = self.get_singleton('enhanced_grounding_validator')

            # Create response generator with all dependencies
            generator = create_response_generator(
                llm_client=llm_client,
                async_client=async_client,
                formatting_manager=formatting_manager,
                grounding_validator=grounding_validator,
                config=self.config.config
            )

            logger.info("Response generator created with grounding validation")
            return generator
        except Exception as e:
            logger.error("Failed to create response generator", exception_details=str(e))
            return None

    def _create_response_formatter(self):
        """Create response formatter service."""
        from ..formatting.response_formatter import ResponseFormatter

        formatter = ResponseFormatter(config=self.config)

        logger.info("Response formatter created")
        return formatter

    def _create_async_llm_client(self):
        """Create async LLM client for reflective retrieval."""
        from openai import AsyncOpenAI
        import os

        model_name = self.config.get_llm_model()
        model_params = self.config.get_model_params('llm')

        client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        client.model_name = model_name
        client.model_params = model_params

        logger.info("Async LLM client created", model=model_name)
        return client

    def _create_reflective_retriever(self):
        """Create reflective retriever that enhances document retrieval with LLM-based reflection."""
        try:
            from ..retrieval.reflective_retriever import create_reflective_retriever

            # Get base retriever (if available) - will be set later during pipeline initialization
            base_retriever = None

            # Get or create LLM client
            llm_client = self.get_singleton('async_llm_client')

            return create_reflective_retriever(base_retriever, llm_client, self.config.config)
        except Exception as e:
            logger.error("Failed to create reflective retriever", exception_details=str(e))
            return None

    def _create_formatting_manager(self):
        """Create formatting manager with dependency injection."""
        try:
            from ..formatting.manager import FormattingManager

            # Get async LLM client for better performance with formatting operations
            llm_client = self.get_singleton('async_llm_client')

            # Use configuration from quality config
            formatting_config = self.config.config.get('formatting', {})

            formatter = FormattingManager(
                llm_client=llm_client,
                llm_model=self.config.get_llm_model(),
                config=formatting_config
            )

            logger.info("Formatting manager created")
            return formatter
        except Exception as e:
            logger.error("Failed to create formatting manager", exception_details=str(e))
            return None

    def _create_enhanced_grounding_validator(self):
        """Create enhanced grounding validator for hallucination detection."""
        try:
            from ..quality.enhanced_grounding_validator import EnhancedGroundingValidator

            validator = EnhancedGroundingValidator(config=self.config)

            logger.info("Enhanced grounding validator created")
            return validator
        except Exception as e:
            logger.error("Failed to create enhanced grounding validator", exception_details=str(e))
            return None

    # Utility methods
    def reload_config(self):
        """Reload configuration and clear singleton cache."""
        from ..config.quality_config import reload_quality_config

        reload_quality_config()
        self.config = get_quality_config()
        self._singletons.clear()

        logger.info("Configuration reloaded, singleton cache cleared")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all registered services."""
        status = {
            "environment": self.config.environment,
            "registered_factories": len(self._factories),
            "active_singletons": len(self._singletons),
            "services": {}
        }

        # Test critical services
        critical_services = [
            'embedding_model', 'llm_client', 'semantic_alignment_validator'
        ]

        for service in critical_services:
            try:
                instance = self.get_singleton(service)
                status["services"][service] = {
                    "status": "healthy",
                    "type": type(instance).__name__
                }
            except Exception as e:
                status["services"][service] = {
                    "status": "unhealthy",
                    "error": str(e)
                }

        return status


# Global container instance
_global_container: Optional[ServiceContainer] = None


def get_container(config: Optional[QualityConfig] = None,
                 force_reload: bool = False) -> ServiceContainer:
    """
    Get global service container instance.

    Args:
        config: Optional configuration instance
        force_reload: Force reload container

    Returns:
        ServiceContainer instance
    """
    global _global_container

    if _global_container is None or force_reload:
        _global_container = ServiceContainer(config)

    return _global_container


def inject(service_name: str, singleton: bool = True):
    """
    Decorator for dependency injection.

    Args:
        service_name: Name of service to inject
        singleton: Whether to use singleton instance

    Returns:
        Decorated function with injected dependency
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            container = get_container()

            if singleton:
                service = container.get_singleton(service_name)
            else:
                service = container.get(service_name)

            # Inject as first argument
            return func(service, *args, **kwargs)

        return wrapper
    return decorator


class ServiceMixin:
    """Mixin class that provides easy access to services."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._container = get_container()

    def get_service(self, service_name: str, singleton: bool = True):
        """Get a service from the container."""
        if singleton:
            return self._container.get_singleton(service_name)
        return self._container.get(service_name)

    @property
    def config(self) -> QualityConfig:
        """Get quality configuration."""
        return self._container.config