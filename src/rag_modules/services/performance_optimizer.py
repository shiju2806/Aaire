"""
Performance Optimization Service for Enhanced Retrieval
Implements tiered processing and async pattern learning for production scalability
"""

import re
import asyncio
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
import structlog

from ..core.dependency_injection import ServiceMixin
from ..config.quality_config import QualityConfig

logger = structlog.get_logger()


@dataclass
class QueryPerformanceMetrics:
    """Performance metrics for query processing"""
    query_hash: str
    tier_used: str
    response_time: float
    cache_hit: bool
    disambiguation_applied: bool
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CachedPattern:
    """Cached disambiguation pattern"""
    pattern_hash: str
    query_signature: str
    disambiguation_result: Dict[str, Any]
    confidence: float
    usage_count: int = 0
    last_used: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for disambiguation service"""
    is_open: bool = False
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    half_open_attempt_count: int = 0


class PerformanceOptimizer(ServiceMixin):
    """
    Performance optimization service implementing:
    1. Tiered processing (fast/cached/full)
    2. Async pattern learning
    3. Performance monitoring
    4. Circuit breaker pattern
    """

    def __init__(self, config: Optional[QualityConfig] = None):
        """Initialize performance optimizer"""
        super().__init__()
        self._config = config or self.config

        # Get performance optimization config
        self.perf_config = self._config.config.get('performance_optimization', {})
        self.tiered_config = self.perf_config.get('tiered_processing', {})
        self.async_config = self.perf_config.get('async_learning', {})
        self.monitoring_config = self.perf_config.get('monitoring', {})
        self.circuit_config = self.perf_config.get('circuit_breaker', {})

        # Compile regex patterns for query classification
        self.simple_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.tiered_config.get('simple_query_patterns', [])
        ]
        self.complex_indicators = self.tiered_config.get('complex_query_indicators', [])

        # Pattern cache
        self.pattern_cache: Dict[str, CachedPattern] = {}
        self.max_cached_patterns = self.tiered_config.get('max_cached_patterns', 1000)
        self.cache_ttl = timedelta(seconds=self.tiered_config.get('cached_pattern_ttl', 3600))

        # Async learning queue
        self.learning_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.async_config.get('learning_queue_size', 100)
        )
        self.learning_task: Optional[asyncio.Task] = None

        # Performance metrics
        self.metrics: deque = deque(maxlen=1000)  # Keep last 1000 queries
        self.tier_usage_stats = defaultdict(int)
        self.cache_hit_stats = {'hits': 0, 'misses': 0}

        # Circuit breaker
        self.circuit_breaker = CircuitBreakerState()

        # Initialize async learning if enabled
        if (self.async_config.get('enabled', True) and
            self.async_config.get('background_learning_enabled', True)):
            self.start_background_learning()

        # Load persisted patterns
        self._load_persisted_patterns()

        logger.info("Performance optimizer initialized",
                   tiered_enabled=self.tiered_config.get('enabled', True),
                   async_enabled=self.async_config.get('enabled', True),
                   monitoring_enabled=self.monitoring_config.get('enabled', True))


    async def _learn_from_interaction(self, interaction: Dict[str, Any]):
        """Learn patterns from query interaction"""
        query = interaction.get('query', '')
        disambiguation_result = interaction.get('disambiguation_result')
        response_time = interaction.get('response_time', 0)

        if not query or not disambiguation_result:
            return

        # Create pattern signature
        query_signature = self._create_query_signature(query)
        pattern_hash = self._hash_pattern(query_signature, disambiguation_result)

        # Check if this pattern should be cached
        if self._should_cache_pattern(disambiguation_result, response_time):
            cached_pattern = CachedPattern(
                pattern_hash=pattern_hash,
                query_signature=query_signature,
                disambiguation_result=disambiguation_result,
                confidence=disambiguation_result.get('confidence', 0.0)
            )

            # Add to cache (with LRU eviction if needed)
            self._add_to_cache(pattern_hash, cached_pattern)

            logger.debug("Learned new pattern",
                        query_signature=query_signature,
                        confidence=cached_pattern.confidence)

    def determine_processing_tier(self, query: str) -> str:
        """
        Determine the appropriate processing tier for a query

        Returns:
            "fast" - Simple lookup, skip disambiguation
            "cached" - Use cached disambiguation pattern
            "full" - Full entropy + disambiguation processing
        """
        if not self.tiered_config.get('enabled', True):
            return "full"

        # Check circuit breaker
        if self._is_circuit_open():
            logger.debug("Circuit breaker open, using fast tier")
            return "fast"

        # Check for simple query patterns
        if self._is_simple_query(query):
            return "fast"

        # Check for cached patterns
        if self._has_cached_pattern(query):
            return "cached"

        # Check for complex query indicators
        if self._is_complex_query(query):
            return "full"

        # Default to cached if available, otherwise full
        return "cached" if self._has_similar_cached_pattern(query) else "full"

    def _is_simple_query(self, query: str) -> bool:
        """Check if query matches simple lookup patterns"""
        query_lower = query.lower().strip()

        # Check regex patterns
        for pattern in self.simple_patterns:
            if pattern.match(query_lower):
                return True

        # Check for very short queries (likely simple lookups)
        if len(query_lower.split()) <= 3:
            return True

        return False

    def _is_complex_query(self, query: str) -> bool:
        """Check if query has complexity indicators"""
        query_lower = query.lower()

        for indicator in self.complex_indicators:
            if indicator.lower() in query_lower:
                return True

        # Check for multiple concepts (longer queries)
        if len(query_lower.split()) > 10:
            return True

        return False

    def _has_cached_pattern(self, query: str) -> bool:
        """Check if we have a cached pattern for this query"""
        query_signature = self._create_query_signature(query)

        # Clean expired patterns
        self._clean_expired_patterns()

        # Look for exact match
        for pattern in self.pattern_cache.values():
            if pattern.query_signature == query_signature:
                return True

        return False

    def _has_similar_cached_pattern(self, query: str) -> bool:
        """Check if we have similar cached patterns"""
        query_signature = self._create_query_signature(query)

        # Simple similarity check based on key terms
        query_terms = set(query_signature.split())

        for pattern in self.pattern_cache.values():
            pattern_terms = set(pattern.query_signature.split())
            similarity = len(query_terms & pattern_terms) / max(len(query_terms), 1)

            if similarity > 0.7:  # 70% term overlap
                return True

        return False

    async def get_cached_pattern(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached disambiguation pattern for query"""
        query_signature = self._create_query_signature(query)

        # Look for exact match first
        for pattern_hash, pattern in self.pattern_cache.items():
            if pattern.query_signature == query_signature:
                # Update usage stats
                pattern.usage_count += 1
                pattern.last_used = datetime.now()

                self.cache_hit_stats['hits'] += 1
                logger.debug("Cache hit for query pattern", query_signature=query_signature)

                return pattern.disambiguation_result

        # Look for similar patterns
        best_match = None
        best_similarity = 0.0
        query_terms = set(query_signature.split())

        for pattern in self.pattern_cache.values():
            pattern_terms = set(pattern.query_signature.split())
            similarity = len(query_terms & pattern_terms) / max(len(query_terms), 1)

            if similarity > best_similarity and similarity > 0.8:  # High similarity threshold
                best_similarity = similarity
                best_match = pattern

        if best_match:
            best_match.usage_count += 1
            best_match.last_used = datetime.now()

            self.cache_hit_stats['hits'] += 1
            logger.debug("Cache hit for similar pattern",
                        query_signature=query_signature,
                        matched_signature=best_match.query_signature,
                        similarity=best_similarity)

            return best_match.disambiguation_result

        self.cache_hit_stats['misses'] += 1
        return None

    async def cache_pattern(self, query: str, disambiguation_result: Dict[str, Any], response_time: float):
        """Manually cache a pattern for testing and optimization"""
        if not self._should_cache_pattern(disambiguation_result, response_time):
            return False

        query_signature = self._create_query_signature(query)
        pattern_hash = self._hash_pattern(query_signature, disambiguation_result)

        pattern = CachedPattern(
            pattern_hash=pattern_hash,
            query_signature=query_signature,
            disambiguation_result=disambiguation_result,
            confidence=disambiguation_result.get('confidence', 0.0)
        )

        self._add_to_cache(pattern_hash, pattern)
        logger.debug("Pattern manually cached",
                    query_preview=query[:50] + "..." if len(query) > 50 else query,
                    confidence=pattern.confidence)
        return True

    async def record_interaction(self, query: str, tier_used: str, response_time: float,
                               disambiguation_result: Optional[Dict[str, Any]] = None,
                               cache_hit: bool = False):
        """Record query interaction for learning and monitoring"""

        # Record metrics
        if self.monitoring_config.get('enabled', True):
            query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]

            metrics = QueryPerformanceMetrics(
                query_hash=query_hash,
                tier_used=tier_used,
                response_time=response_time,
                cache_hit=cache_hit,
                disambiguation_applied=disambiguation_result is not None
            )

            self.metrics.append(metrics)
            self.tier_usage_stats[tier_used] += 1

            # Log slow queries
            slow_threshold = self.monitoring_config.get('slow_query_threshold', 2.0)
            if response_time > slow_threshold:
                logger.warning("Slow query detected",
                             query_preview=query[:50] + "..." if len(query) > 50 else query,
                             tier=tier_used,
                             response_time=response_time)

        # Queue for async learning
        if (self.async_config.get('enabled', True) and
            disambiguation_result and
            not cache_hit):  # Don't learn from cached results

            learning_item = {
                'query': query,
                'tier_used': tier_used,
                'response_time': response_time,
                'disambiguation_result': disambiguation_result,
                'timestamp': datetime.now().isoformat()
            }

            try:
                self.learning_queue.put_nowait(learning_item)
            except asyncio.QueueFull:
                logger.warning("Learning queue full, dropping interaction")

    def record_circuit_breaker_event(self, success: bool):
        """Record circuit breaker success/failure"""
        if not self.circuit_config.get('enabled', True):
            return

        if success:
            # Reset failure count on success
            if self.circuit_breaker.failure_count > 0:
                logger.info("Circuit breaker reset after successful operation")
                self.circuit_breaker.failure_count = 0
                self.circuit_breaker.half_open_attempt_count = 0
        else:
            # Increment failure count
            self.circuit_breaker.failure_count += 1
            self.circuit_breaker.last_failure_time = datetime.now()

            failure_threshold = self.circuit_config.get('failure_threshold', 5)
            if self.circuit_breaker.failure_count >= failure_threshold:
                self.circuit_breaker.is_open = True
                logger.warning("Circuit breaker opened due to failures",
                             failure_count=self.circuit_breaker.failure_count)

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open"""
        if not self.circuit_config.get('enabled', True):
            return False

        if not self.circuit_breaker.is_open:
            return False

        # Check if timeout has elapsed
        timeout_seconds = self.circuit_config.get('timeout_seconds', 60)
        if (self.circuit_breaker.last_failure_time and
            datetime.now() - self.circuit_breaker.last_failure_time > timedelta(seconds=timeout_seconds)):

            # Try half-open state
            half_open_requests = self.circuit_config.get('half_open_requests', 3)
            if self.circuit_breaker.half_open_attempt_count < half_open_requests:
                self.circuit_breaker.half_open_attempt_count += 1
                logger.info("Circuit breaker attempting half-open state")
                return False
            else:
                # Close circuit after successful half-open attempts
                self.circuit_breaker.is_open = False
                self.circuit_breaker.failure_count = 0
                self.circuit_breaker.half_open_attempt_count = 0
                logger.info("Circuit breaker closed after successful recovery")
                return False

        return True

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        if not self.monitoring_config.get('enabled', True):
            return {}

        # Calculate metrics from recent queries
        recent_metrics = [m for m in self.metrics
                         if datetime.now() - m.timestamp < timedelta(hours=1)]

        if not recent_metrics:
            return {
                'total_queries': 0,
                'cache_hit_rate': 0.0,
                'tier_usage': dict(self.tier_usage_stats),
                'avg_response_time': 0.0
            }

        total_queries = len(recent_metrics)
        cache_hits = sum(1 for m in recent_metrics if m.cache_hit)
        cache_hit_rate = cache_hits / total_queries if total_queries > 0 else 0.0
        avg_response_time = sum(m.response_time for m in recent_metrics) / total_queries

        return {
            'total_queries': total_queries,
            'cache_hit_rate': cache_hit_rate,
            'tier_usage': dict(self.tier_usage_stats),
            'avg_response_time': avg_response_time,
            'cached_patterns': len(self.pattern_cache),
            'learning_queue_size': self.learning_queue.qsize() if hasattr(self.learning_queue, 'qsize') else 0,
            'circuit_breaker_open': self.circuit_breaker.is_open,
            'circuit_failure_count': self.circuit_breaker.failure_count
        }

    # Helper methods

    def _create_query_signature(self, query: str) -> str:
        """Create a normalized signature for query matching"""
        # Normalize query for pattern matching
        signature = re.sub(r'[^\w\s]', '', query.lower())
        signature = ' '.join(signature.split())  # Normalize whitespace
        return signature

    def _hash_pattern(self, query_signature: str, disambiguation_result: Dict[str, Any]) -> str:
        """Create hash for disambiguation pattern"""
        pattern_data = f"{query_signature}:{disambiguation_result.get('confidence', 0)}"
        return hashlib.sha256(pattern_data.encode()).hexdigest()[:16]

    def _should_cache_pattern(self, disambiguation_result: Dict[str, Any], response_time: float) -> bool:
        """Determine if a pattern should be cached"""
        confidence = disambiguation_result.get('confidence', 0.0)
        conflicting_pairs = len(disambiguation_result.get('conflicting_pairs', []))

        # Cache high-confidence patterns or patterns with conflicts
        return confidence > 0.7 or conflicting_pairs > 0

    def _add_to_cache(self, pattern_hash: str, pattern: CachedPattern):
        """Add pattern to cache with LRU eviction"""
        # Remove if already exists
        if pattern_hash in self.pattern_cache:
            del self.pattern_cache[pattern_hash]

        # Evict least recently used if cache is full
        if len(self.pattern_cache) >= self.max_cached_patterns:
            lru_hash = min(self.pattern_cache.keys(),
                          key=lambda k: self.pattern_cache[k].last_used)
            del self.pattern_cache[lru_hash]

        self.pattern_cache[pattern_hash] = pattern

    def _clean_expired_patterns(self):
        """Remove expired patterns from cache"""
        current_time = datetime.now()
        expired_hashes = [
            hash_key for hash_key, pattern in self.pattern_cache.items()
            if current_time - pattern.created_at > self.cache_ttl
        ]

        for hash_key in expired_hashes:
            del self.pattern_cache[hash_key]

        if expired_hashes:
            logger.debug("Cleaned expired patterns", count=len(expired_hashes))

    def _load_persisted_patterns(self):
        """Load cached patterns from disk"""
        if not self.async_config.get('pattern_persistence_enabled', True):
            return

        pattern_file = self.async_config.get('pattern_file_path', 'cache/learned_patterns.json')
        pattern_path = Path(pattern_file)

        if not pattern_path.exists():
            return

        try:
            with open(pattern_path, 'r') as f:
                patterns_data = json.load(f)

            for pattern_data in patterns_data:
                pattern = CachedPattern(
                    pattern_hash=pattern_data['pattern_hash'],
                    query_signature=pattern_data['query_signature'],
                    disambiguation_result=pattern_data['disambiguation_result'],
                    confidence=pattern_data['confidence'],
                    usage_count=pattern_data.get('usage_count', 0),
                    last_used=datetime.fromisoformat(pattern_data.get('last_used', datetime.now().isoformat())),
                    created_at=datetime.fromisoformat(pattern_data.get('created_at', datetime.now().isoformat()))
                )

                self.pattern_cache[pattern.pattern_hash] = pattern

            logger.info("Loaded persisted patterns", count=len(patterns_data))

        except Exception as e:
            logger.warning("Failed to load persisted patterns", error=str(e))

    async def persist_patterns(self):
        """Save cached patterns to disk"""
        if not self.async_config.get('pattern_persistence_enabled', True):
            return

        pattern_file = self.async_config.get('pattern_file_path', 'cache/learned_patterns.json')
        pattern_path = Path(pattern_file)
        pattern_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            patterns_data = []
            for pattern in self.pattern_cache.values():
                patterns_data.append({
                    'pattern_hash': pattern.pattern_hash,
                    'query_signature': pattern.query_signature,
                    'disambiguation_result': pattern.disambiguation_result,
                    'confidence': pattern.confidence,
                    'usage_count': pattern.usage_count,
                    'last_used': pattern.last_used.isoformat(),
                    'created_at': pattern.created_at.isoformat()
                })

            with open(pattern_path, 'w') as f:
                json.dump(patterns_data, f, indent=2)

            logger.info("Persisted patterns to disk", count=len(patterns_data))

        except Exception as e:
            logger.error("Failed to persist patterns", error=str(e))

    async def shutdown(self):
        """Gracefully shutdown the optimizer"""
        # Stop background learning
        if self.learning_task and not self.learning_task.done():
            self.learning_task.cancel()
            try:
                await self.learning_task
            except asyncio.CancelledError:
                pass

        # Persist patterns
        await self.persist_patterns()

        logger.info("Performance optimizer shutdown complete")

    async def _background_learning_loop(self):
        """Background learning loop to process queued interactions"""
        logger.info("Background learning loop started")

        learning_interval = self.async_config.get('learning_interval_seconds', 30)
        batch_size = self.async_config.get('learning_batch_size', 10)

        while True:
            try:
                # Wait for learning interval
                await asyncio.sleep(learning_interval)

                # Process queued learning items in batches
                if self.learning_queue.qsize() > 0:
                    batch = []

                    # Collect batch items
                    for _ in range(min(batch_size, self.learning_queue.qsize())):
                        try:
                            item = self.learning_queue.get_nowait()
                            batch.append(item)
                        except asyncio.QueueEmpty:
                            break

                    if batch:
                        await self._process_learning_batch(batch)

                # Clean expired patterns periodically
                self._clean_expired_patterns()

            except asyncio.CancelledError:
                logger.info("Background learning loop cancelled")
                break
            except Exception as e:
                logger.error("Error in background learning loop", error=str(e))
                # Continue the loop even if there's an error

    async def _process_learning_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of learning interactions"""
        logger.debug("Processing learning batch", batch_size=len(batch))

        patterns_learned = 0

        for item in batch:
            try:
                query = item['query']
                disambiguation_result = item['disambiguation_result']
                response_time = item['response_time']

                # Only learn from successful disambiguation results
                if (disambiguation_result and
                    self._should_cache_pattern(disambiguation_result, response_time)):

                    query_signature = self._create_query_signature(query)
                    pattern_hash = self._hash_pattern(query_signature, disambiguation_result)

                    # Check if pattern already exists
                    if pattern_hash not in self.pattern_cache:
                        pattern = CachedPattern(
                            pattern_hash=pattern_hash,
                            query_signature=query_signature,
                            disambiguation_result=disambiguation_result,
                            confidence=disambiguation_result.get('confidence', 0.0)
                        )

                        self._add_to_cache(pattern_hash, pattern)
                        patterns_learned += 1

                        logger.debug("Learned new pattern",
                                   query_preview=query[:50] + "..." if len(query) > 50 else query,
                                   confidence=pattern.confidence)

            except Exception as e:
                logger.warning("Failed to process learning item", error=str(e))

        if patterns_learned > 0:
            logger.info("Background learning completed", patterns_learned=patterns_learned)

            # Persist patterns if enabled
            if self.async_config.get('pattern_persistence_enabled', True):
                try:
                    await self.persist_patterns()
                except Exception as e:
                    logger.warning("Failed to persist learned patterns", error=str(e))

    def start_background_learning(self):
        """Start the background learning task"""
        if (self.async_config.get('background_learning_enabled', True) and
            (not self.learning_task or self.learning_task.done())):

            self.learning_task = asyncio.create_task(self._background_learning_loop())
            logger.info("Background learning task started")


def create_performance_optimizer(config: Optional[QualityConfig] = None) -> PerformanceOptimizer:
    """Factory function to create performance optimizer"""
    return PerformanceOptimizer(config)