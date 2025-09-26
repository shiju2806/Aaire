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


            # Smart validator optimized for GPT-4o-mini
            'smart_validator': self._create_smart_validator,

            # Domain knowledge service for authoritative terminology
            'domain_knowledge_service': self._create_domain_knowledge_service,

            # Entropy-based disambiguation for semantically similar concepts
            'entropy_disambiguation_service': self._create_entropy_disambiguation_service,

            # Performance optimization for tiered processing and async learning
            'performance_optimizer': self._create_performance_optimizer
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
            # Use smart_validator instead of enhanced_grounding_validator
            grounding_validator = self.get_singleton('smart_validator')

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


    def _create_smart_validator(self):
        """Create smart validator optimized for GPT-4o-mini hallucination detection."""
        try:
            from ..validation.smart_validator import SmartValidator

            # Get domain knowledge service for authoritative terminology
            domain_knowledge = self.get_singleton('domain_knowledge_service')
            validator = SmartValidator(config=self.config, domain_knowledge_service=domain_knowledge)

            logger.info("Smart validator created for GPT-4o-mini optimization")
            return validator
        except Exception as e:
            logger.error("Failed to create smart validator", exception_details=str(e))
            return None

    def _create_domain_knowledge_service(self):
        """Create domain knowledge service with authoritative terminology sources."""
        try:
            from ..services.domain_knowledge_service import DomainKnowledgeService

            service = DomainKnowledgeService(config=self.config)

            # Initialize domain knowledge asynchronously in background
            import asyncio
            try:
                # Try to get current event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create task for background initialization
                    asyncio.create_task(service.initialize_domain_knowledge())
                else:
                    # Run initialization synchronously
                    loop.run_until_complete(service.initialize_domain_knowledge())
            except RuntimeError:
                # No event loop running, create new one
                asyncio.run(service.initialize_domain_knowledge())

            logger.info("Domain knowledge service created with authoritative sources")
            return service
        except Exception as e:
            logger.error("Failed to create domain knowledge service", exception_details=str(e))
            return None

    def _create_entropy_disambiguation_service(self):
        """Create entropy-based disambiguation service for semantically similar concepts."""
        try:
            from ..services.entropy_disambiguation_service import EntropyDisambiguationService

            service = EntropyDisambiguationService(config=self.config)

            # Initialize models asynchronously in background
            import asyncio
            try:
                # Try to get current event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create task for background initialization
                    asyncio.create_task(service.initialize_models())
                else:
                    # Run initialization synchronously
                    loop.run_until_complete(service.initialize_models())
            except RuntimeError:
                # No event loop running, create new one
                asyncio.run(service.initialize_models())

            logger.info("Entropy disambiguation service created with KeyBERT integration")
            return service
        except Exception as e:
            logger.error("Failed to create entropy disambiguation service", exception_details=str(e))
            return None

    def _create_performance_optimizer(self):
        """Create performance optimizer for tiered processing and async learning."""
        try:
            from ..services.performance_optimizer import PerformanceOptimizer

            optimizer = PerformanceOptimizer(config=self.config)

            logger.info("Performance optimizer created with tiered processing and async learning")
            return optimizer
        except Exception as e:
            logger.error("Failed to create performance optimizer", exception_details=str(e))
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
            'embedding_model', 'llm_client', 'smart_validator', 'entropy_disambiguation_service'
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