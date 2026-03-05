"""
Retrieval audit trail for compliance.

Every retrieval decision is logged as a structured event:
- Which entities were resolved
- Which graph paths were traversed
- Which chunks were selected and why
- Final ranking scores

Stored as structured log via structlog — queryable for compliance audits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog

from ..providers.config_loader import get_config, get_nested

logger = structlog.get_logger()
_audit_logger = structlog.get_logger("aaire.retrieval_audit")


@dataclass
class RetrievalAudit:
    """A single retrieval decision record for compliance audit."""

    query: str = ""
    query_entities: List[str] = field(default_factory=list)
    resolved_entities: List[str] = field(default_factory=list)  # Graph node IDs
    graph_connected_chunks: List[str] = field(default_factory=list)
    hybrid_search_chunks: List[str] = field(default_factory=list)
    final_ranked_chunks: List[str] = field(default_factory=list)
    entity_boosts_applied: Dict[str, float] = field(default_factory=dict)
    rerank_scores: Dict[str, float] = field(default_factory=dict)
    graph_context_injected: bool = False
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "query_entities": self.query_entities,
            "resolved_entities": self.resolved_entities,
            "graph_connected_chunks": self.graph_connected_chunks,
            "hybrid_search_chunks": self.hybrid_search_chunks,
            "final_ranked_chunks": self.final_ranked_chunks,
            "entity_boosts_applied": self.entity_boosts_applied,
            "rerank_scores": self.rerank_scores,
            "graph_context_injected": self.graph_context_injected,
            "timestamp": self.timestamp,
        }

    def log(self) -> None:
        """Emit as structured log event for compliance."""
        config = get_config("knowledge_graph")
        if not get_nested(config, "audit", "enabled", default=True):
            return

        _audit_logger.info(
            "retrieval_audit",
            **self.to_dict(),
        )
