"""
Query Decomposition Module

Decomposes complex queries into sub-questions for multi-pass retrieval.
Uses a lightweight heuristic gate (no LLM cost) to detect complexity,
then decomposes via a single LLM call only when needed.
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

import structlog

from ...providers.config_loader import get_config, get_nested

logger = structlog.get_logger()


@dataclass
class DecompositionResult:
    """Result of query decomposition."""

    is_decomposed: bool
    sub_queries: List[str]
    original_query: str = ""


class QueryDecomposer:
    """Decomposes complex queries into sub-questions for multi-pass retrieval.

    Complexity gating is heuristic-based (no LLM call). Only queries that
    pass the gate are decomposed via a single LLM call.
    """

    def __init__(self, llm) -> None:
        self._llm = llm
        scoring_cfg = get_config("scoring")
        decomp_cfg = get_nested(scoring_cfg, "query_decomposition", default={})

        self._enabled = decomp_cfg.get("enabled", True)
        self._max_sub_queries = decomp_cfg.get("max_sub_queries", 4)
        self._comparison_words = set(
            decomp_cfg.get(
                "comparison_words",
                ["compare", "contrast", "difference", "vs", "versus", "similarities"],
            )
        )
        self._aggregation_words = decomp_cfg.get(
            "aggregation_words",
            ["all", "every", "across", "throughout", "summarize all", "list all"],
        )
        self._min_word_count = decomp_cfg.get("min_word_count_for_complex", 20)

    async def maybe_decompose(self, query: str) -> DecompositionResult:
        """Gate + decompose. Returns original query unchanged if simple."""
        if not self._enabled:
            return DecompositionResult(
                is_decomposed=False, sub_queries=[query], original_query=query
            )

        if not self._is_complex(query):
            return DecompositionResult(
                is_decomposed=False, sub_queries=[query], original_query=query
            )

        # Complex query detected — decompose with LLM.
        try:
            sub_queries = await self._decompose(query)
            if sub_queries and len(sub_queries) > 1:
                logger.info(
                    "Query decomposed",
                    original=query[:60],
                    sub_queries=sub_queries,
                )
                return DecompositionResult(
                    is_decomposed=True,
                    sub_queries=sub_queries[: self._max_sub_queries],
                    original_query=query,
                )
        except Exception as e:
            logger.warning("Query decomposition failed, using original", error=str(e))

        return DecompositionResult(
            is_decomposed=False, sub_queries=[query], original_query=query
        )

    def _is_complex(self, query: str) -> bool:
        """Lightweight heuristic gate -- no LLM call.

        A query is complex if ANY of:
        - Contains comparison words
        - Contains aggregation phrases
        - Word count > threshold AND contains question words
        """
        query_lower = query.lower()
        tokens = set(query_lower.split())

        # Check comparison words.
        if tokens & self._comparison_words:
            return True

        # Check aggregation phrases (multi-word, so use substring match).
        for phrase in self._aggregation_words:
            if phrase in query_lower:
                return True

        # Long query with question structure.
        question_words = {"what", "how", "why", "which", "when", "where", "who"}
        if len(query_lower.split()) > self._min_word_count and tokens & question_words:
            return True

        return False

    async def _decompose(self, query: str) -> List[str]:
        """Decompose a complex query into sub-questions via LLM."""
        prompt = (
            "Break this query into 2-4 independent sub-questions that together "
            "fully answer the original question. Each sub-question should be "
            "answerable from a single document section.\n\n"
            f"Query: {query}\n\n"
            'Return JSON only: {{"sub_queries": ["question 1", "question 2", ...]}}'
        )

        response = self._llm.complete(prompt)
        text = response.text.strip()

        # Parse JSON from response (handle markdown code fences).
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        parsed = json.loads(text)
        sub_queries = parsed.get("sub_queries", [])

        # Validate.
        if not isinstance(sub_queries, list):
            return [query]
        return [q for q in sub_queries if isinstance(q, str) and q.strip()]
