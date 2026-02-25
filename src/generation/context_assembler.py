"""
Context assembly layer.

Takes enriched retrieval results and builds an optimal LLM prompt.
Element-type-aware: each type is rendered differently for the LLM.

Principle: the LLM receives *original* tables and formulas, not
summaries. Summaries are for search; originals are for answering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog

from ..providers.config_loader import get_config, get_nested

logger = structlog.get_logger()


@dataclass
class AssembledContext:
    """The assembled context ready for LLM generation.

    Attributes:
        text: The full context string to include in the prompt.
        element_count: Number of elements included.
        element_types: Breakdown of element types included.
        truncated: Whether truncation was applied.
        total_chars: Total character count of the context.
        source_documents: List of source document titles referenced.
    """

    text: str = ""
    element_count: int = 0
    element_types: Dict[str, int] = field(default_factory=dict)
    truncated: bool = False
    total_chars: int = 0
    source_documents: List[str] = field(default_factory=list)


class ContextAssembler:
    """Assemble retrieval results into an optimal LLM prompt context.

    Rendering rules by element type:
    - text: Display text only (context prefix stripped — LLM doesn't
            need "This chunk is from..." preamble).
    - table: Original markdown table from structured store, with a
             header line identifying the table.
    - formula: LaTeX + variable definitions from structured store.
    - callout: Wrapped in [IMPORTANT] tags.
    - image: Text description (vision model output).
    - table_proposition: Inline as a factual statement.

    Ordering: relevance score descending, with tables and formulas
    grouped near related text chunks (same section).
    """

    def __init__(self, max_context_chars: int = 0) -> None:
        config = get_config("infrastructure")
        target_tokens = get_nested(config, "token_limits", "target_tokens", default=30000)
        chars_per_token = get_nested(config, "token_limits", "chars_per_token", default=4)
        self._max_chars = max_context_chars or (target_tokens * chars_per_token)

    def assemble(
        self,
        results: List[Any],
        query: str = "",
    ) -> AssembledContext:
        """Assemble retrieval results into a context string.

        Args:
            results: List of EnrichedResult objects (or dicts with
                    compatible keys) from the retrieval layer.
            query: The user query (for logging).

        Returns:
            AssembledContext with the rendered text and metadata.
        """
        if not results:
            return AssembledContext()

        # Group results by section for locality.
        grouped = self._group_by_section(results)

        # Render each element.
        rendered_blocks: List[tuple[float, str, str]] = []  # (score, element_type, text)
        for section, items in grouped:
            if section:
                rendered_blocks.append((999.0, "section_header", f"\n### {section}\n"))

            for item in items:
                score = self._get_score(item)
                element_type = self._get_element_type(item)
                rendered = self._render_element(item, element_type)
                if rendered:
                    rendered_blocks.append((score, element_type, rendered))

        # Build the context string, respecting token limit.
        context_parts: List[str] = []
        total_chars = 0
        truncated = False
        type_counts: Dict[str, int] = {}
        doc_titles: set = set()

        for _, element_type, text in rendered_blocks:
            if total_chars + len(text) > self._max_chars:
                truncated = True
                # Try to include a truncated version of this block.
                remaining = self._max_chars - total_chars
                if remaining > 200:
                    context_parts.append(text[:remaining] + "\n[...truncated]")
                    total_chars += remaining
                break

            context_parts.append(text)
            total_chars += len(text)
            if element_type != "section_header":
                type_counts[element_type] = type_counts.get(element_type, 0) + 1

            # Track source documents.
            doc_title = self._get_doc_title(rendered_blocks)

        context_text = "\n\n".join(context_parts)

        result = AssembledContext(
            text=context_text,
            element_count=sum(type_counts.values()),
            element_types=type_counts,
            truncated=truncated,
            total_chars=total_chars,
            source_documents=sorted(doc_titles),
        )

        logger.info(
            "Context assembled",
            query=query[:50] if query else "",
            elements=result.element_count,
            types=result.element_types,
            chars=result.total_chars,
            truncated=result.truncated,
        )
        return result

    # -- element rendering --------------------------------------------------

    def _render_element(self, item: Any, element_type: str) -> str:
        """Render a single retrieval result based on its element type."""
        renderer = self._renderers.get(element_type, self._render_text)
        return renderer(self, item)

    def _render_text(self, item: Any) -> str:
        """Render a text element — display text only, no context prefix."""
        display_text = self._get_display_text(item)
        return display_text

    def _render_table(self, item: Any) -> str:
        """Render a table — use original markdown from structured store."""
        # Prefer original content from structured store.
        original = self._get_original_content(item)
        if original:
            markdown = original
        else:
            # Fall back to display text (which may already be markdown).
            markdown = self._get_display_text(item)

        metadata = self._get_metadata(item)
        columns = metadata.get("columns", [])
        row_count = metadata.get("row_count", 0)

        header = "[TABLE"
        if columns:
            header += f" | Columns: {', '.join(columns[:6])}"
            if len(columns) > 6:
                header += f" (+{len(columns) - 6} more)"
        if row_count:
            header += f" | {row_count} rows"
        header += "]"

        return f"{header}\n{markdown}"

    def _render_formula(self, item: Any) -> str:
        """Render a formula — LaTeX + variable definitions."""
        metadata = self._get_metadata(item)
        latex = metadata.get("latex") or self._get_original_content(item)
        variables = metadata.get("variables", [])

        parts = ["[FORMULA]"]
        if latex:
            parts.append(latex)
        else:
            parts.append(self._get_display_text(item))

        if variables:
            parts.append(f"Variables: {', '.join(variables)}")

        return "\n".join(parts)

    def _render_callout(self, item: Any) -> str:
        """Render a callout — wrapped in [IMPORTANT] tags."""
        text = self._get_display_text(item)
        return f"[IMPORTANT]\n{text}\n[/IMPORTANT]"

    def _render_image(self, item: Any) -> str:
        """Render an image — text description."""
        text = self._get_display_text(item)
        return f"[DIAGRAM/IMAGE]\n{text}"

    def _render_proposition(self, item: Any) -> str:
        """Render a table proposition — inline factual statement."""
        return self._get_display_text(item)

    _renderers: Dict[str, Any] = {
        "text": _render_text,
        "table": _render_table,
        "formula": _render_formula,
        "callout": _render_callout,
        "image": _render_image,
        "table_proposition": _render_proposition,
    }

    # -- grouping -----------------------------------------------------------

    @staticmethod
    def _group_by_section(
        results: List[Any],
    ) -> List[tuple[str, List[Any]]]:
        """Group results by section, maintaining score order within groups."""
        section_order: List[str] = []
        section_items: Dict[str, List[Any]] = {}

        for item in results:
            section = ""
            if hasattr(item, "metadata"):
                section = getattr(item, "metadata", {}).get("section", "") or \
                          getattr(item, "parent_section", "")
            elif isinstance(item, dict):
                section = item.get("metadata", {}).get("section", "")

            if section not in section_items:
                section_order.append(section)
                section_items[section] = []
            section_items[section].append(item)

        return [(s, section_items[s]) for s in section_order]

    # -- accessors (work with both EnrichedResult and dict) -----------------

    @staticmethod
    def _get_score(item: Any) -> float:
        if hasattr(item, "score"):
            return item.score
        if isinstance(item, dict):
            return item.get("score", 0) or item.get("relevance_score", 0)
        return 0.0

    @staticmethod
    def _get_element_type(item: Any) -> str:
        if hasattr(item, "element_type"):
            return item.element_type
        if isinstance(item, dict):
            return item.get("metadata", {}).get("element_type", "text")
        return "text"

    @staticmethod
    def _get_display_text(item: Any) -> str:
        if hasattr(item, "content"):
            return item.content
        if isinstance(item, dict):
            return item.get("content", item.get("display_text", ""))
        return str(item)

    @staticmethod
    def _get_original_content(item: Any) -> Optional[str]:
        if hasattr(item, "original_content"):
            return item.original_content
        if isinstance(item, dict):
            return item.get("original_content")
        return None

    @staticmethod
    def _get_metadata(item: Any) -> Dict[str, Any]:
        if hasattr(item, "metadata") and isinstance(item.metadata, dict):
            return item.metadata
        if isinstance(item, dict):
            return item.get("metadata", {})
        return {}

    @staticmethod
    def _get_doc_title(item: Any) -> str:
        meta = ContextAssembler._get_metadata(item)
        return meta.get("document_title", meta.get("filename", ""))
