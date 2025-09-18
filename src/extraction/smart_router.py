"""
Smart Extraction Router with Circuit Breaker
Implements tiered processing: Cache -> Pattern -> Light LLM -> Full LLM -> Fallback
Industry-standard circuit breaker pattern for LLM reliability
"""

import time
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import structlog

from .models import (
    ExtractionResult,
    ExtractionMethod,
    DocumentFingerprint,
    MetadataContext,
    CircuitBreakerState
)
from .fingerprinting import FingerprintCache
from .registry import ExtractorRegistry

logger = structlog.get_logger()


@dataclass
class RoutingStrategy:
    """Configuration for routing strategy"""
    cache_enabled: bool = True
    pattern_threshold: float = 0.8
    light_llm_threshold: float = 0.6
    full_llm_threshold: float = 0.4
    fallback_enabled: bool = True


class LLMCircuitBreaker:
    """
    Circuit breaker for LLM calls to prevent cascading failures
    States: CLOSED (normal), OPEN (failing), HALF_OPEN (testing)
    """

    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.state_data = CircuitBreakerState()

    async def call_llm(self, llm_func, *args, **kwargs):
        """Execute LLM call with circuit breaker protection"""

        # Check circuit state
        if self.state_data.state == "OPEN":
            if self._should_attempt_reset():
                self.state_data.state = "HALF_OPEN"
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker OPEN - LLM calls blocked")

        try:
            # Increment request counter
            self.state_data.total_requests += 1

            # Execute LLM call
            result = await llm_func(*args, **kwargs)

            # Success - update state
            self._on_success()
            return result

        except Exception as e:
            # Failure - update state
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.state_data.last_failure_time is None:
            return True
        return time.time() - self.state_data.last_failure_time > self.timeout_seconds

    def _on_success(self):
        """Handle successful LLM call"""
        if self.state_data.state == "HALF_OPEN":
            # Successful call in half-open state - reset circuit
            self.state_data.state = "CLOSED"
            self.state_data.failure_count = 0
            self.state_data.success_count += 1
            logger.info("Circuit breaker reset to CLOSED")
        elif self.state_data.state == "CLOSED":
            self.state_data.success_count += 1

    def _on_failure(self):
        """Handle failed LLM call"""
        self.state_data.failure_count += 1
        self.state_data.last_failure_time = time.time()

        if self.state_data.failure_count >= self.failure_threshold:
            self.state_data.state = "OPEN"
            logger.warning(
                f"Circuit breaker OPEN - {self.state_data.failure_count} failures",
                failure_count=self.state_data.failure_count,
                threshold=self.failure_threshold
            )

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            'state': self.state_data.state,
            'failure_count': self.state_data.failure_count,
            'success_count': self.state_data.success_count,
            'total_requests': self.state_data.total_requests,
            'failure_rate': (
                self.state_data.failure_count / self.state_data.total_requests
                if self.state_data.total_requests > 0 else 0.0
            )
        }


class SmartExtractionRouter:
    """
    Smart router implementing tiered extraction strategy
    Optimizes for cost and reliability while maintaining quality
    """

    def __init__(
        self,
        extractor_registry: ExtractorRegistry,
        routing_config: Dict[str, Any]
    ):
        self.registry = extractor_registry
        self.routing_strategy = RoutingStrategy(
            cache_enabled=routing_config.get('cache_enabled', True),
            pattern_threshold=routing_config.get('pattern_matching_threshold', 0.8),
            light_llm_threshold=routing_config.get('light_llm_threshold', 0.6),
            full_llm_threshold=routing_config.get('full_llm_threshold', 0.4),
            fallback_enabled=routing_config.get('fallback_enabled', True)
        )

        # Initialize caching
        cache_ttl = routing_config.get('cache_ttl_hours', 24)
        self.fingerprint_cache = FingerprintCache(ttl_hours=cache_ttl)

        # Initialize circuit breaker
        circuit_config = routing_config.get('circuit_breaker', {})
        self.circuit_breaker = LLMCircuitBreaker(
            failure_threshold=circuit_config.get('failure_threshold', 5),
            timeout_seconds=circuit_config.get('timeout_seconds', 60)
        )

        # Performance tracking
        self.routing_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'pattern_successes': 0,
            'light_llm_successes': 0,
            'full_llm_successes': 0,
            'fallback_uses': 0,
            'total_requests': 0
        }

    async def route_extraction(
        self,
        content: str,
        document_fingerprint: DocumentFingerprint,
        context: MetadataContext,
        llm_client=None
    ) -> ExtractionResult:
        """
        Route extraction through tiered processing strategy

        Processing order:
        1. Cache check (if enabled)
        2. Pattern matching
        3. Light LLM (if confidence too low)
        4. Full LLM (if still too low)
        5. Fallback (if all else fails)
        """
        start_time = time.time()
        self.routing_stats['total_requests'] += 1

        try:
            # Stage 1: Cache Check
            if self.routing_strategy.cache_enabled:
                cached_result = await self._try_cache(document_fingerprint, context)
                if cached_result:
                    self.routing_stats['cache_hits'] += 1
                    logger.info(
                        "Cache hit",
                        document_id=document_fingerprint.document_id,
                        cache_fingerprint=document_fingerprint.composite_fingerprint[:12]
                    )
                    return cached_result

            self.routing_stats['cache_misses'] += 1

            # Stage 2: Pattern Matching
            pattern_result = await self._try_pattern_extraction(
                content, document_fingerprint, context
            )

            if pattern_result and pattern_result.confidence >= self.routing_strategy.pattern_threshold:
                self.routing_stats['pattern_successes'] += 1
                await self._cache_result(document_fingerprint, pattern_result)
                logger.info(
                    "Pattern extraction successful",
                    document_id=document_fingerprint.document_id,
                    confidence=pattern_result.confidence,
                    method=pattern_result.extraction_method.value
                )
                return pattern_result

            # Stage 3: Light LLM (with circuit breaker)
            if llm_client and self.circuit_breaker.state_data.state != "OPEN":
                try:
                    light_result = await self.circuit_breaker.call_llm(
                        self._try_light_llm_extraction,
                        content, document_fingerprint, context, llm_client
                    )

                    if light_result and light_result.confidence >= self.routing_strategy.light_llm_threshold:
                        self.routing_stats['light_llm_successes'] += 1
                        await self._cache_result(document_fingerprint, light_result)
                        logger.info(
                            "Light LLM extraction successful",
                            document_id=document_fingerprint.document_id,
                            confidence=light_result.confidence
                        )
                        return light_result

                except Exception as e:
                    logger.warning(f"Light LLM extraction failed: {e}")

            # Stage 4: Full LLM (with circuit breaker)
            if llm_client and self.circuit_breaker.state_data.state != "OPEN":
                try:
                    full_result = await self.circuit_breaker.call_llm(
                        self._try_full_llm_extraction,
                        content, document_fingerprint, context, llm_client
                    )

                    if full_result and full_result.confidence >= self.routing_strategy.full_llm_threshold:
                        self.routing_stats['full_llm_successes'] += 1
                        await self._cache_result(document_fingerprint, full_result)
                        logger.info(
                            "Full LLM extraction successful",
                            document_id=document_fingerprint.document_id,
                            confidence=full_result.confidence
                        )
                        return full_result

                except Exception as e:
                    logger.warning(f"Full LLM extraction failed: {e}")

            # Stage 5: Fallback
            if self.routing_strategy.fallback_enabled:
                fallback_result = await self._try_fallback_extraction(
                    content, document_fingerprint, context
                )
                self.routing_stats['fallback_uses'] += 1
                logger.info(
                    "Using fallback extraction",
                    document_id=document_fingerprint.document_id,
                    confidence=fallback_result.confidence if fallback_result else 0.0
                )
                return fallback_result

            # Complete failure
            return self._create_failure_result(document_fingerprint, "All extraction methods failed")

        except Exception as e:
            logger.error(
                "Routing failed completely",
                document_id=document_fingerprint.document_id,
                error=str(e)
            )
            return self._create_failure_result(document_fingerprint, str(e))

        finally:
            processing_time = (time.time() - start_time) * 1000
            logger.debug(
                "Routing completed",
                document_id=document_fingerprint.document_id,
                total_time_ms=f"{processing_time:.2f}"
            )

    async def _try_cache(
        self,
        document_fingerprint: DocumentFingerprint,
        context: MetadataContext
    ) -> Optional[ExtractionResult]:
        """Try to get result from cache"""
        try:
            cached_data = self.fingerprint_cache.get(document_fingerprint.composite_fingerprint)
            if cached_data:
                # Reconstruct result from cached data
                return ExtractionResult(**cached_data)
        except Exception as e:
            logger.debug(f"Cache lookup failed: {e}")
        return None

    async def _cache_result(self, document_fingerprint: DocumentFingerprint, result: ExtractionResult):
        """Cache successful result"""
        try:
            # Cache only successful results
            if result.success and result.confidence > 0.3:
                cache_data = {
                    'success': result.success,
                    'document_id': result.document_id,
                    'extraction_method': result.extraction_method,
                    'entities': [entity.__dict__ for entity in result.entities],
                    'document_type': result.document_type,
                    'confidence': result.confidence,
                    'processing_time_ms': result.processing_time_ms,
                    'metadata': result.metadata,
                    'warnings': result.warnings,
                    'extraction_timestamp': result.extraction_timestamp
                }
                self.fingerprint_cache.put(document_fingerprint.composite_fingerprint, cache_data)
        except Exception as e:
            logger.debug(f"Failed to cache result: {e}")

    async def _try_pattern_extraction(
        self,
        content: str,
        document_fingerprint: DocumentFingerprint,
        context: MetadataContext
    ) -> Optional[ExtractionResult]:
        """Try pattern-based extraction"""
        # Use registry with no LLM client to force pattern-only extraction
        return await self.registry.extract(
            content=content,
            document_fingerprint=document_fingerprint,
            context=context,
            llm_client=None  # This forces pattern-only extraction
        )

    async def _try_light_llm_extraction(
        self,
        content: str,
        document_fingerprint: DocumentFingerprint,
        context: MetadataContext,
        llm_client
    ) -> Optional[ExtractionResult]:
        """Try light LLM extraction with simplified prompt"""
        # Force general extractor for light processing
        return await self.registry.extract(
            content=content[:2000],  # Truncate for light processing
            document_fingerprint=document_fingerprint,
            context=context,
            llm_client=llm_client,
            force_extractor='general'
        )

    async def _try_full_llm_extraction(
        self,
        content: str,
        document_fingerprint: DocumentFingerprint,
        context: MetadataContext,
        llm_client
    ) -> Optional[ExtractionResult]:
        """Try full LLM extraction with best extractor"""
        return await self.registry.extract(
            content=content,
            document_fingerprint=document_fingerprint,
            context=context,
            llm_client=llm_client
        )

    async def _try_fallback_extraction(
        self,
        content: str,
        document_fingerprint: DocumentFingerprint,
        context: MetadataContext
    ) -> ExtractionResult:
        """Fallback extraction using basic pattern matching"""
        # Use general extractor without LLM
        result = await self.registry.extract(
            content=content,
            document_fingerprint=document_fingerprint,
            context=context,
            llm_client=None,
            force_extractor='general'
        )

        # Mark as fallback
        result.extraction_method = ExtractionMethod.FALLBACK
        result.metadata['fallback_used'] = True
        result.warnings.append("Using fallback extraction method")

        return result

    def _create_failure_result(
        self,
        document_fingerprint: DocumentFingerprint,
        error_message: str
    ) -> ExtractionResult:
        """Create result for complete failure"""
        return ExtractionResult(
            success=False,
            document_id=document_fingerprint.document_id,
            extraction_method=ExtractionMethod.FALLBACK,
            entities=[],
            document_type="error",
            confidence=0.0,
            processing_time_ms=0.0,
            metadata={'error': error_message},
            warnings=[f"Complete extraction failure: {error_message}"]
        )

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing performance statistics"""
        total = self.routing_stats['total_requests']
        if total == 0:
            return self.routing_stats

        stats = self.routing_stats.copy()
        stats['cache_hit_rate'] = stats['cache_hits'] / total
        stats['pattern_success_rate'] = stats['pattern_successes'] / total
        stats['light_llm_success_rate'] = stats['light_llm_successes'] / total
        stats['full_llm_success_rate'] = stats['full_llm_successes'] / total
        stats['fallback_rate'] = stats['fallback_uses'] / total
        stats['circuit_breaker_state'] = self.circuit_breaker.get_state()

        return stats

    def clear_cache(self):
        """Clear the fingerprint cache"""
        self.fingerprint_cache = FingerprintCache(ttl_hours=24)
        logger.info("Router cache cleared")

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on router"""
        try:
            stats = self.get_routing_stats()
            circuit_state = self.circuit_breaker.get_state()

            return {
                'status': 'healthy' if circuit_state['state'] != 'OPEN' else 'degraded',
                'routing_stats': stats,
                'circuit_breaker': circuit_state,
                'cache_size': self.fingerprint_cache.size()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }