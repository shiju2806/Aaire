"""
Intent analysis caching with pattern-based fast path.

Caches query intent results to avoid redundant LLM calls.
Provides pattern-based shortcuts for obvious query types:
  - "What is X?" → definition/conceptual
  - "How to calculate X?" → procedural
  - "Compare X vs Y" → comparison
  - "ASC 842-10-15-2" → specific_reference

Cache key: hash of normalized query.
TTL: configurable (default 24 hours).
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import structlog

from ..providers.config_loader import get_config, get_nested

logger = structlog.get_logger()


@dataclass
class CachedIntent:
    """Cached query intent analysis result."""

    query_type: str = "unknown"          # specific_reference, conceptual, comparison, procedural, contextual
    jurisdiction: str = "unknown"        # IFRS, US_GAAP, US_STAT, unknown
    product_type: str = "general"        # universal_life, whole_life, term, general
    domain: str = "unknown"              # insurance, actuarial, accounting, general
    entities: List[str] = field(default_factory=list)
    specificity_score: float = 0.5
    source: str = "pattern"              # pattern | llm | cache
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedIntent":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Pattern-based fast path rules
# ---------------------------------------------------------------------------

_PATTERNS: List[tuple[re.Pattern, Dict[str, str]]] = [
    # "What is X?" / "What are X?" → conceptual
    (re.compile(r"^what\s+(is|are)\s+", re.IGNORECASE),
     {"query_type": "conceptual", "specificity_score": "0.3"}),

    # "Define X" / "Definition of X"
    (re.compile(r"^defin(e|ition\s+of)\s+", re.IGNORECASE),
     {"query_type": "conceptual", "specificity_score": "0.4"}),

    # "How to calculate X?" / "How do you compute X?"
    (re.compile(r"^how\s+(to|do\s+(?:you|we|i))\s+(calculate|compute|determine|derive)", re.IGNORECASE),
     {"query_type": "procedural", "specificity_score": "0.6"}),

    # "Compare X vs Y" / "Difference between X and Y"
    (re.compile(r"(compare|comparison|difference\s+between|versus|vs\.?)\s+", re.IGNORECASE),
     {"query_type": "comparison", "specificity_score": "0.5"}),

    # Specific ASC/IFRS/IAS references → specific_reference
    (re.compile(r"\b(ASC|IFRS|IAS|GAAP|VM-\d+|SSAP)\s*\d", re.IGNORECASE),
     {"query_type": "specific_reference", "specificity_score": "0.9"}),

    # "Explain X" / "Describe X"
    (re.compile(r"^(explain|describe|overview\s+of|summarize)\s+", re.IGNORECASE),
     {"query_type": "conceptual", "specificity_score": "0.4"}),

    # "List X" / "What are the types of X"
    (re.compile(r"^(list|enumerate|what\s+types\s+of|what\s+kinds\s+of)", re.IGNORECASE),
     {"query_type": "conceptual", "specificity_score": "0.4"}),
]

# Jurisdiction detection patterns.
_JURISDICTION_PATTERNS: List[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(IFRS|IAS)\b", re.IGNORECASE), "IFRS"),
    (re.compile(r"\b(US.?GAAP|ASC\s+\d|FASB)\b", re.IGNORECASE), "US_GAAP"),
    (re.compile(r"\b(US.?STAT|SSAP|VM-\d+|NAIC)\b", re.IGNORECASE), "US_STAT"),
]

# Domain detection patterns.
_DOMAIN_PATTERNS: List[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(mortality|morbidity|lapse|surrender|annuit|life\s+insurance|whole\s+life|term\s+life|universal\s+life|endowment|death\s+benefit|cash\s+value)\b", re.IGNORECASE), "insurance"),
    (re.compile(r"\b(actuarial|actuary|reserve|valuation|pricing|experience\s+study|CSO|decrement)\b", re.IGNORECASE), "actuarial"),
    (re.compile(r"\b(revenue\s+recognition|lease|impairment|goodwill|depreciation|amortization|consolidat|fair\s+value|hedge|derivative)\b", re.IGNORECASE), "accounting"),
]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class IntentCache:
    """In-memory intent cache with TTL and pattern-based fast path.

    For production, swap the in-memory dict for Redis. The interface
    is intentionally simple to make that easy.
    """

    def __init__(self, ttl_hours: float = 24.0) -> None:
        config = get_config("infrastructure")
        self._ttl_seconds = get_nested(
            config, "timeouts", "cache_ttl_hours", default=ttl_hours
        ) * 3600
        self._cache: Dict[str, CachedIntent] = {}

    def get(self, query: str) -> Optional[CachedIntent]:
        """Look up cached intent. Returns None if miss or expired."""
        key = self._cache_key(query)
        intent = self._cache.get(key)
        if intent is None:
            return None
        if time.time() - intent.timestamp > self._ttl_seconds:
            del self._cache[key]
            return None
        intent.source = "cache"
        return intent

    def put(self, query: str, intent: CachedIntent) -> None:
        """Store intent in cache."""
        intent.timestamp = time.time()
        self._cache[self._cache_key(query)] = intent

    def classify_fast(self, query: str) -> Optional[CachedIntent]:
        """Try pattern-based fast path. Returns None if no pattern matches."""
        intent = CachedIntent(timestamp=time.time(), source="pattern")

        # Match query type.
        matched = False
        for pattern, attrs in _PATTERNS:
            if pattern.search(query):
                intent.query_type = attrs.get("query_type", intent.query_type)
                intent.specificity_score = float(attrs.get("specificity_score", "0.5"))
                matched = True
                break

        if not matched:
            return None

        # Detect jurisdiction.
        for pattern, jurisdiction in _JURISDICTION_PATTERNS:
            if pattern.search(query):
                intent.jurisdiction = jurisdiction
                break

        # Detect domain.
        for pattern, domain in _DOMAIN_PATTERNS:
            if pattern.search(query):
                intent.domain = domain
                break

        # Extract entity-like tokens (ASC codes, standard references).
        entities = re.findall(
            r"\b(ASC\s+\d[\d\-\.]+|IFRS\s+\d[\d\.]*|IAS\s+\d[\d\.]*|VM-\d+|SSAP\s+\d+)\b",
            query, re.IGNORECASE
        )
        intent.entities = entities

        logger.debug(
            "Pattern fast path matched",
            query=query[:50],
            query_type=intent.query_type,
            jurisdiction=intent.jurisdiction,
        )
        return intent

    def invalidate(self, query: Optional[str] = None) -> None:
        """Clear cache — all or specific query."""
        if query:
            self._cache.pop(self._cache_key(query), None)
        else:
            self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)

    @staticmethod
    def _cache_key(query: str) -> str:
        """Normalize and hash query for cache key."""
        normalized = " ".join(query.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]
