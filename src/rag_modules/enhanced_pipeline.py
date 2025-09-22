"""
Enhanced RAG Pipeline Integration
Orchestrates Advanced Retrieval and Self-Correction modules
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from openai import AsyncOpenAI
import structlog

from .retrieval.advanced_strategies import AdvancedRetrievalManager
from .reasoning.self_correction import SelfCorrectionManager, CorrectedResponse

logger = structlog.get_logger()

class EnhancedRAGPipeline:
    """
    Enhanced RAG Pipeline that integrates advanced retrieval and self-correction
    Designed to seamlessly integrate with existing RAG systems
    """

    def __init__(
        self,
        llm_client: AsyncOpenAI,
        config_dir: str = "/Users/shijuprakash/AAIRE/config",
        retrieval_config_file: str = "advanced_retrieval.yaml",
        correction_config_file: str = "self_correction.yaml"
    ):
        self.llm_client = llm_client
        self.config_dir = Path(config_dir)
        self.logger = logger.bind(component="enhanced_rag_pipeline")

        # Load configurations
        self.retrieval_config = self._load_config(retrieval_config_file)
        self.correction_config = self._load_config(correction_config_file)

        # Initialize managers
        self.advanced_retrieval = AdvancedRetrievalManager(
            llm_client, self.retrieval_config
        )
        self.self_correction = SelfCorrectionManager(
            llm_client, self.correction_config
        )

        # Track if modules are enabled
        self.retrieval_enabled = self.retrieval_config.get('enabled', True)
        self.correction_enabled = self.correction_config.get('enabled', True)

        self.logger.info("Enhanced RAG Pipeline initialized",
                        retrieval_enabled=self.retrieval_enabled,
                        correction_enabled=self.correction_enabled)

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = self.config_dir / config_file

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info("Configuration loaded", file=config_file)
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config {config_file}: {e}")
            return {}

    async def enhanced_rag_query(
        self,
        query: str,
        base_retrieval_func,
        base_generation_func,
        document_store=None,
        context: Dict[str, Any] = None,
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for enhanced RAG processing

        Args:
            query: User query
            base_retrieval_func: Existing retrieval function to enhance
            base_generation_func: Existing generation function to enhance
            document_store: Document storage system for parent-child retrieval
            context: Additional context for processing
            options: Processing options and overrides

        Returns:
            Enhanced response with metadata
        """
        options = options or {}
        start_time = self._get_timestamp()

        try:
            # Phase 1: Enhanced Retrieval
            retrieval_result = await self._enhanced_retrieval_phase(
                query, base_retrieval_func, document_store, context
            )

            # Phase 2: Enhanced Generation with Self-Correction
            generation_result = await self._enhanced_generation_phase(
                query, retrieval_result, base_generation_func, options
            )

            # Phase 3: Compile Results
            final_result = self._compile_enhanced_result(
                query, retrieval_result, generation_result, start_time
            )

            self.logger.info("Enhanced RAG processing completed",
                           total_time_ms=final_result['metadata']['processing_time_ms'],
                           retrieval_strategy=final_result['metadata']['retrieval_strategy'],
                           correction_applied=final_result['metadata']['correction_applied'])

            return final_result

        except Exception as e:
            self.logger.error(f"Enhanced RAG processing failed: {e}")

            # Fallback to base processing
            return await self._fallback_processing(
                query, base_retrieval_func, base_generation_func, str(e)
            )

    async def _enhanced_retrieval_phase(
        self,
        query: str,
        base_retrieval_func,
        document_store=None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Enhanced retrieval with advanced strategies"""

        if not self.retrieval_enabled:
            # Use base retrieval if enhancement disabled
            documents = await base_retrieval_func(query)
            return {
                "documents": documents,
                "strategy": "base",
                "metadata": {"enhanced": False}
            }

        try:
            # Use advanced retrieval strategies
            retrieval_result = await self.advanced_retrieval.enhanced_retrieval(
                query=query,
                base_retrieval_func=base_retrieval_func,
                document_store=document_store,
                context=context
            )

            return retrieval_result

        except Exception as e:
            self.logger.error(f"Enhanced retrieval failed: {e}")

            # Fallback to base retrieval
            documents = await base_retrieval_func(query)
            return {
                "documents": documents,
                "strategy": "fallback",
                "metadata": {"enhanced": False, "error": str(e)}
            }

    async def _enhanced_generation_phase(
        self,
        query: str,
        retrieval_result: Dict[str, Any],
        base_generation_func,
        options: Dict[str, Any]
    ) -> CorrectedResponse:
        """Enhanced generation with self-correction"""

        documents = retrieval_result.get("documents", [])

        # Prepare context from retrieved documents
        context = self._prepare_context_from_documents(documents)

        if not self.correction_enabled:
            # Use base generation if enhancement disabled
            response = await base_generation_func(query, context)

            return CorrectedResponse(
                final_response=response,
                reasoning_chain=None,
                verification_result=None,
                correction_applied=False,
                iterations=1,
                metadata={"enhanced": False}
            )

        try:
            # Use self-correction with reasoning
            generation_result = await self.self_correction.enhanced_generation(
                query=query,
                context=context,
                base_generation_func=lambda q, c: base_generation_func(q, c),
                options=options
            )

            return generation_result

        except Exception as e:
            self.logger.error(f"Enhanced generation failed: {e}")

            # Fallback to base generation
            response = await base_generation_func(query, context)

            return CorrectedResponse(
                final_response=response,
                reasoning_chain=None,
                verification_result=None,
                correction_applied=False,
                iterations=1,
                metadata={"enhanced": False, "error": str(e)}
            )

    def _prepare_context_from_documents(self, documents: List[Dict]) -> str:
        """Prepare context string from retrieved documents"""

        if not documents:
            return ""

        context_parts = []

        for i, doc in enumerate(documents[:10], 1):  # Limit to top 10 documents
            content = doc.get('content', '')

            # Add document with clear boundaries
            doc_context = f"[Document {i}]\n{content}\n[/Document {i}]"
            context_parts.append(doc_context)

        return "\n\n".join(context_parts)

    def _compile_enhanced_result(
        self,
        query: str,
        retrieval_result: Dict[str, Any],
        generation_result: CorrectedResponse,
        start_time: float
    ) -> Dict[str, Any]:
        """Compile final enhanced result"""

        processing_time = (self._get_timestamp() - start_time) * 1000

        # Extract key information
        documents = retrieval_result.get("documents", [])
        retrieval_strategy = retrieval_result.get("strategy", "unknown")

        final_response = generation_result.final_response
        reasoning_chain = generation_result.reasoning_chain
        verification_result = generation_result.verification_result
        correction_applied = generation_result.correction_applied
        iterations = generation_result.iterations

        # Build comprehensive metadata
        metadata = {
            "processing_time_ms": processing_time,
            "retrieval_strategy": retrieval_strategy,
            "correction_applied": correction_applied,
            "reasoning_used": reasoning_chain is not None,
            "verification_confidence": verification_result.confidence if verification_result else None,
            "generation_iterations": iterations,
            "documents_count": len(documents),
            "enhanced_features": {
                "advanced_retrieval": self.retrieval_enabled,
                "self_correction": self.correction_enabled
            }
        }

        # Add retrieval metadata
        if "metadata" in retrieval_result:
            metadata["retrieval_metadata"] = retrieval_result["metadata"]

        # Add generation metadata
        if generation_result.metadata:
            metadata["generation_metadata"] = generation_result.metadata

        # Add verification details if available
        if verification_result:
            metadata["verification"] = {
                "confidence": verification_result.confidence,
                "issues_count": len(verification_result.issues),
                "quality_scores": verification_result.quality_scores
            }

        # Add reasoning details if available
        if reasoning_chain:
            metadata["reasoning"] = {
                "methodology": reasoning_chain.methodology,
                "steps_count": len(reasoning_chain.steps),
                "overall_confidence": reasoning_chain.overall_confidence
            }

        return {
            "response": final_response,
            "documents": documents,
            "reasoning_chain": reasoning_chain,
            "verification_result": verification_result,
            "metadata": metadata
        }

    async def _fallback_processing(
        self,
        query: str,
        base_retrieval_func,
        base_generation_func,
        error: str
    ) -> Dict[str, Any]:
        """Fallback to base processing when enhanced processing fails"""

        try:
            # Use base functions
            documents = await base_retrieval_func(query)
            context = self._prepare_context_from_documents(documents)
            response = await base_generation_func(query, context)

            return {
                "response": response,
                "documents": documents,
                "reasoning_chain": None,
                "verification_result": None,
                "metadata": {
                    "processing_mode": "fallback",
                    "error": error,
                    "enhanced_features": {
                        "advanced_retrieval": False,
                        "self_correction": False
                    }
                }
            }

        except Exception as fallback_error:
            self.logger.error(f"Fallback processing also failed: {fallback_error}")

            return {
                "response": f"Processing failed: {error}",
                "documents": [],
                "reasoning_chain": None,
                "verification_result": None,
                "metadata": {
                    "processing_mode": "error",
                    "error": error,
                    "fallback_error": str(fallback_error)
                }
            }

    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()

    # Utility methods for configuration management
    def update_configuration(
        self,
        module: str,
        config_updates: Dict[str, Any]
    ) -> bool:
        """Update configuration for a specific module"""

        try:
            if module == "retrieval":
                self.retrieval_config.update(config_updates)
                self.advanced_retrieval = AdvancedRetrievalManager(
                    self.llm_client, self.retrieval_config
                )
                self.retrieval_enabled = self.retrieval_config.get('enabled', True)

            elif module == "correction":
                self.correction_config.update(config_updates)
                self.self_correction = SelfCorrectionManager(
                    self.llm_client, self.correction_config
                )
                self.correction_enabled = self.correction_config.get('enabled', True)

            else:
                raise ValueError(f"Unknown module: {module}")

            self.logger.info("Configuration updated", module=module)
            return True

        except Exception as e:
            self.logger.error(f"Configuration update failed for {module}: {e}")
            return False

    def get_configuration(self, module: str = None) -> Dict[str, Any]:
        """Get current configuration"""

        if module == "retrieval":
            return self.retrieval_config.copy()
        elif module == "correction":
            return self.correction_config.copy()
        elif module is None:
            return {
                "retrieval": self.retrieval_config.copy(),
                "correction": self.correction_config.copy()
            }
        else:
            raise ValueError(f"Unknown module: {module}")

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""

        return {
            "retrieval_enabled": self.retrieval_enabled,
            "correction_enabled": self.correction_enabled,
            "modules_loaded": {
                "advanced_retrieval": self.advanced_retrieval is not None,
                "self_correction": self.self_correction is not None
            },
            "config_files": {
                "retrieval": str(self.config_dir / "advanced_retrieval.yaml"),
                "correction": str(self.config_dir / "self_correction.yaml")
            }
        }

class EnhancedRAGIntegrator:
    """
    Integration helper for existing RAG systems
    Provides simple methods to integrate enhanced features
    """

    @staticmethod
    def create_enhanced_pipeline(
        llm_client: AsyncOpenAI,
        config_dir: str = None
    ) -> EnhancedRAGPipeline:
        """Create enhanced pipeline with default configuration"""

        config_dir = config_dir or "/Users/shijuprakash/AAIRE/config"

        return EnhancedRAGPipeline(
            llm_client=llm_client,
            config_dir=config_dir
        )

    @staticmethod
    def wrap_existing_rag_function(
        existing_rag_func,
        enhanced_pipeline: EnhancedRAGPipeline
    ):
        """Wrap existing RAG function to use enhanced pipeline"""

        async def enhanced_wrapper(
            query: str,
            retrieval_func,
            generation_func,
            **kwargs
        ):
            return await enhanced_pipeline.enhanced_rag_query(
                query=query,
                base_retrieval_func=retrieval_func,
                base_generation_func=generation_func,
                **kwargs
            )

        return enhanced_wrapper

    @staticmethod
    def gradual_migration_wrapper(
        existing_rag_func,
        enhanced_pipeline: EnhancedRAGPipeline,
        migration_percentage: float = 0.1
    ):
        """Gradually migrate to enhanced pipeline for testing"""

        import random

        async def migration_wrapper(*args, **kwargs):
            if random.random() < migration_percentage:
                # Use enhanced pipeline
                try:
                    return await enhanced_pipeline.enhanced_rag_query(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Enhanced pipeline failed, falling back: {e}")
                    return await existing_rag_func(*args, **kwargs)
            else:
                # Use existing pipeline
                return await existing_rag_func(*args, **kwargs)

        return migration_wrapper