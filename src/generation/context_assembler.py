"""
Context assembly layer.

Takes enriched retrieval results and builds an optimal LLM prompt.
Element-type-aware: each type is rendered differently for the LLM.

Principle: the LLM receives *original* tables and formulas, not
summaries. Summaries are for search; originals are for answering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
        source_map: Maps source index [N] to the EnrichedResult or metadata
                    that produced it, enabling inline citation extraction.
    """

    text: str = ""
    element_count: int = 0
    element_types: Dict[str, int] = field(default_factory=dict)
    truncated: bool = False
    total_chars: int = 0
    source_documents: List[str] = field(default_factory=list)
    source_map: Dict[int, Any] = field(default_factory=dict)
    coverage_score: float = 1.0
    uncovered_topics: List[str] = field(default_factory=list)


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

        # Render each element with a source index [N].
        # rendered_blocks: (score, element_type, text, source_item_or_None)
        rendered_blocks: List[tuple[float, str, str, Any]] = []
        for section, items in grouped:
            if section:
                rendered_blocks.append((999.0, "section_header", f"\n### {section}\n", None))

            for item in items:
                score = self._get_score(item)
                element_type = self._get_element_type(item)
                rendered = self._render_element(item, element_type)
                if rendered:
                    rendered_blocks.append((score, element_type, rendered, item))

        # Priority packing: maximize coverage within budget.
        packed_blocks = self._priority_pack(rendered_blocks, self._max_chars)

        # Build context string with [N] source indices.
        context_parts: List[str] = []
        total_chars = 0
        truncated = False
        type_counts: Dict[str, int] = {}
        doc_titles: set = set()
        source_map: Dict[int, Any] = {}
        source_index = 0

        for _, element_type, text, source_item in packed_blocks:
            if element_type == "section_header":
                block_text = text
            else:
                source_index += 1
                block_text = f"[{source_index}] {text}"
                if source_item is not None:
                    source_map[source_index] = source_item

            block_len = len(block_text)
            if total_chars + block_len > self._max_chars:
                truncated = True
                break

            context_parts.append(block_text)
            total_chars += block_len
            if element_type != "section_header":
                type_counts[element_type] = type_counts.get(element_type, 0) + 1

            if source_item is not None:
                title = self._get_doc_title(source_item)
                if title:
                    doc_titles.add(title)

        context_text = "\n\n".join(context_parts)

        # Assess coverage: check if all query topics are present in context
        coverage_score, uncovered = self._assess_coverage(query, context_text)
        if uncovered:
            gap_note = (
                f"\n\n[NOTE: The user also asked about {', '.join(uncovered)} "
                f"but no retrieved documents address these specifically. "
                f"State this gap clearly in your response.]"
            )
            context_text += gap_note
            total_chars += len(gap_note)

        result = AssembledContext(
            text=context_text,
            element_count=sum(type_counts.values()),
            element_types=type_counts,
            truncated=truncated,
            total_chars=total_chars,
            source_documents=sorted(doc_titles),
            source_map=source_map,
            coverage_score=coverage_score,
            uncovered_topics=uncovered,
        )

        logger.info(
            "Context assembled",
            query=query[:50] if query else "",
            elements=result.element_count,
            types=result.element_types,
            chars=result.total_chars,
            truncated=result.truncated,
            coverage=round(coverage_score, 2),
            uncovered=uncovered,
        )
        return result

    # -- priority packing ---------------------------------------------------

    _STOP_WORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "of", "in", "to",
        "for", "with", "on", "at", "by", "from", "as", "into", "through",
        "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
        "neither", "each", "every", "all", "any", "few", "more", "most",
        "other", "some", "such", "no", "only", "same", "than", "too", "very",
        "this", "that", "these", "those", "it", "its",
    })

    def _priority_pack(
        self,
        rendered_blocks: List[tuple],
        max_chars: int,
    ) -> List[tuple]:
        """Select blocks to maximize information coverage within budget.

        Strategy:
        1. Always include: first chunk from each unique section (coverage).
        2. Always include: tables and formulas (high-value structured content).
        3. Dedup: skip chunks with >80% word overlap with already-selected.
        4. Fill remaining budget with highest-scoring text chunks.
        5. Whole-element: drop entire element if it doesn't fit.
        """
        HIGH_VALUE_TYPES = {"table", "formula", "callout", "table_proposition"}
        OVERLAP_THRESHOLD = 0.80

        # Separate section headers from content blocks.
        headers: List[tuple] = []
        content: List[tuple] = []
        for block in rendered_blocks:
            if block[1] == "section_header":
                headers.append(block)
            else:
                content.append(block)

        # Track which sections we've covered.
        covered_sections: set = set()
        selected: List[tuple] = []
        selected_texts: List[str] = []
        budget_used = 0

        def _try_add(block: tuple) -> bool:
            nonlocal budget_used
            text = block[2]
            block_len = len(text) + 10  # account for [N] prefix + newlines
            if budget_used + block_len > max_chars:
                return False
            # Dedup check
            for existing_text in selected_texts:
                if self._text_overlap(text, existing_text) > OVERLAP_THRESHOLD:
                    return False
            selected.append(block)
            selected_texts.append(text)
            budget_used += block_len
            return True

        # Pass 1: Coverage — one chunk per section (highest-scoring).
        section_best: Dict[str, tuple] = {}
        for block in content:
            source_item = block[3]
            section = ""
            if source_item is not None:
                meta = self._get_metadata(source_item)
                section = meta.get("section", "")
            if section and (section not in section_best or block[0] > section_best[section][0]):
                section_best[section] = block

        for section, block in sorted(section_best.items(), key=lambda x: x[1][0], reverse=True):
            if _try_add(block):
                covered_sections.add(section)

        # Pass 2: High-value elements not yet selected.
        already_selected = set(id(b) for b in selected)
        for block in sorted(content, key=lambda b: b[0], reverse=True):
            if id(block) in already_selected:
                continue
            if block[1] in HIGH_VALUE_TYPES:
                _try_add(block)
                already_selected.add(id(block))

        # Pass 3: Fill with remaining by score.
        for block in sorted(content, key=lambda b: b[0], reverse=True):
            if id(block) in already_selected:
                continue
            _try_add(block)
            already_selected.add(id(block))

        # Reconstruct with section headers.
        # Build a set of sections that have at least one selected block.
        used_sections: set = set()
        for block in selected:
            if block[3] is not None:
                meta = self._get_metadata(block[3])
                sec = meta.get("section", "")
                if sec:
                    used_sections.add(sec)

        # Interleave headers before their section's blocks.
        # Group selected blocks by section, preserving score order.
        section_blocks: Dict[str, List[tuple]] = {}
        no_section: List[tuple] = []
        for block in selected:
            sec = ""
            if block[3] is not None:
                meta = self._get_metadata(block[3])
                sec = meta.get("section", "")
            if sec:
                section_blocks.setdefault(sec, []).append(block)
            else:
                no_section.append(block)

        final: List[tuple] = []
        # Emit sections in the order they appeared in original headers.
        emitted_sections: set = set()
        for header in headers:
            # Extract section name from header text "### SectionName"
            header_section = header[2].strip().lstrip('#').strip()
            if header_section in used_sections and header_section not in emitted_sections:
                final.append(header)
                final.extend(section_blocks.get(header_section, []))
                emitted_sections.add(header_section)

        # Append blocks from sections without headers or unsectioned.
        for sec, blocks in section_blocks.items():
            if sec not in emitted_sections:
                final.extend(blocks)
        final.extend(no_section)

        return final

    @staticmethod
    def _text_overlap(text_a: str, text_b: str) -> float:
        """Word-level Jaccard overlap, ignoring stop words."""
        words_a = set(text_a.lower().split()) - ContextAssembler._STOP_WORDS
        words_b = set(text_b.lower().split()) - ContextAssembler._STOP_WORDS
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    # -- coverage gap detection ----------------------------------------------

    _NOUN_PHRASE_PATTERN = re.compile(
        r'\b(?:(?:capital|solvency|loss|combined|expense|claims?|risk|best estimate|contractual service|insurance|underwriting|premium)'
        r'\s+(?:ratio|margin|capital|reserve|adjustment|liabilit(?:y|ies)|revenue|contract|profit|income|result|deficiency|sufficiency|adequacy)s?)\b',
        re.IGNORECASE,
    )
    _CONCEPT_SPLIT = re.compile(r"[,;]|\band\b|\bor\b|\bto\b|\bin\b|\bfor\b|\bof\b|\babout\b|\bwith\b")

    def _assess_coverage(
        self, query: str, context_text: str
    ) -> Tuple[float, List[str]]:
        """Check if assembled context covers all key topics from the query.

        Uses lightweight regex noun-phrase extraction (no LLM call).
        Returns (coverage_score 0-1, list_of_uncovered_topic_strings).
        """
        if not query:
            return 1.0, []

        # Extract domain noun phrases from query
        topics: List[str] = []
        for m in self._NOUN_PHRASE_PATTERN.finditer(query):
            topics.append(m.group().lower().strip())

        # Also extract simple noun segments (2+ word chunks after splitting on conjunctions)
        segments = self._CONCEPT_SPLIT.split(query.lower())
        for seg in segments:
            seg = seg.strip()
            words = [w for w in seg.split() if w not in self._STOP_WORDS and len(w) > 2]
            if 2 <= len(words) <= 4:
                phrase = " ".join(words)
                if phrase not in topics:
                    topics.append(phrase)

        if not topics:
            return 1.0, []

        context_lower = context_text.lower()
        uncovered = []
        for topic in topics:
            # Check if any significant word from the topic appears in context
            topic_words = set(topic.split()) - self._STOP_WORDS
            if not topic_words:
                continue
            matches = sum(1 for w in topic_words if w in context_lower)
            if matches < len(topic_words) * 0.5:
                uncovered.append(topic)

        covered = len(topics) - len(uncovered)
        score = covered / len(topics) if topics else 1.0
        return score, uncovered

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
