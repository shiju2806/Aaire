"""
Contextual text chunking processor.

Splits text elements into semantically coherent chunks, then prepends
a context prefix to each chunk so the embedding carries document-level
and section-level information.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog

from ..layout_parser import DocumentElement, ElementType
from ...providers.config_loader import get_config, get_nested

logger = structlog.get_logger()


@dataclass
class ProcessedChunk:
    """A chunk ready for embedding and storage.

    Attributes:
        embedding_text: Context prefix + chunk text (for embedding).
        display_text: Original chunk text (for display to user).
        context_prefix: The prepended context string.
        element_type: Source element type.
        page_number: Source page.
        parent_section: Section heading.
        metadata: Additional metadata.
        importance: Retrieval importance multiplier (default 1.0).
    """

    embedding_text: str
    display_text: str
    context_prefix: str = ""
    element_type: str = "text"
    page_number: int = 1
    parent_section: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 1.0


# ---------------------------------------------------------------------------
# Sentence boundary detection (simple, no NLTK dependency)
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using simple regex heuristics."""
    parts = _SENTENCE_END.split(text)
    return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class TextProcessor:
    """Chunk text elements with semantic boundaries and context prefixes.

    Chunking strategy:
    1. Split text into sentences.
    2. Accumulate sentences into chunks up to target_tokens.
    3. Allow overlap_sentences to bleed into the next chunk.
    4. Prepend context prefix: "From [document], section [heading]: "
    """

    def __init__(
        self,
        target_tokens: int = 512,
        overlap_sentences: int = 2,
        chars_per_token: int = 4,
    ) -> None:
        config = get_config("infrastructure")
        self.chars_per_token = get_nested(
            config, "token_limits", "chars_per_token", default=chars_per_token
        )
        self.target_chars = target_tokens * self.chars_per_token
        self.overlap_sentences = overlap_sentences

    def process(
        self,
        elements: List[DocumentElement],
        document_title: str = "",
    ) -> List[ProcessedChunk]:
        """Process a list of text elements into contextual chunks.

        Adjacent text elements under the same section heading are merged
        before chunking to avoid artificial splits.
        """
        if not elements:
            return []

        # Group consecutive text elements by section.
        groups = self._group_by_section(elements)

        chunks: List[ProcessedChunk] = []
        for section, group_elements in groups:
            merged_text = "\n\n".join(e.content for e in group_elements)
            page_num = group_elements[0].page_number

            raw_chunks = self._chunk_text(merged_text)
            for chunk_text in raw_chunks:
                prefix = self._build_context_prefix(document_title, section)
                chunks.append(
                    ProcessedChunk(
                        embedding_text=f"{prefix}{chunk_text}" if prefix else chunk_text,
                        display_text=chunk_text,
                        context_prefix=prefix,
                        element_type="text",
                        page_number=page_num,
                        parent_section=section,
                        metadata={
                            "processor": "text",
                            "chunking_strategy": "contextual",
                            "char_count": len(chunk_text),
                        },
                    )
                )

        logger.debug(
            "Text processing complete",
            input_elements=len(elements),
            output_chunks=len(chunks),
        )
        return chunks

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _group_by_section(
        elements: List[DocumentElement],
    ) -> List[tuple[str, List[DocumentElement]]]:
        """Group consecutive elements sharing the same parent_section."""
        if not elements:
            return []

        groups: List[tuple[str, List[DocumentElement]]] = []
        current_section = elements[0].parent_section
        current_group: List[DocumentElement] = [elements[0]]

        for elem in elements[1:]:
            if elem.parent_section == current_section:
                current_group.append(elem)
            else:
                groups.append((current_section, current_group))
                current_section = elem.parent_section
                current_group = [elem]
        groups.append((current_section, current_group))
        return groups

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks at sentence boundaries with overlap."""
        sentences = _split_sentences(text)
        if not sentences:
            return [text] if text.strip() else []

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current and (current_len + sentence_len) > self.target_chars:
                chunks.append(" ".join(current))
                # Keep overlap sentences.
                overlap = current[-self.overlap_sentences :] if self.overlap_sentences else []
                current = list(overlap)
                current_len = sum(len(s) for s in current)

            current.append(sentence)
            current_len += sentence_len

        if current:
            chunks.append(" ".join(current))

        return chunks

    @staticmethod
    def _build_context_prefix(document_title: str, section: str) -> str:
        """Build context prefix for embedding."""
        parts: List[str] = []
        if document_title:
            parts.append(f"From '{document_title}'")
        if section:
            parts.append(f"section '{section}'")
        if parts:
            return ", ".join(parts) + ": "
        return ""
