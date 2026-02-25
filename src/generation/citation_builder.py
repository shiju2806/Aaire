"""
Precise citation builder using structured metadata.

Replaces vague citations ("Based on retrieved documents...") with
specific references using element-type metadata:

  Before: "Based on the retrieved documents..."
  After:  "Table 3.2 from VM-20 Section 3 (2017 CSO mortality rates,
           male nonsmoker, page 45)"

Works with EnrichedResult objects from the retrieval layer, which
carry element_type, section, page, document_title, and other metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import structlog

logger = structlog.get_logger()


@dataclass
class Citation:
    """A precise citation for a source used in the response.

    Attributes:
        id: Citation index (1-based).
        source: Formatted source string (e.g., "VM-20 Section 3, Page 45").
        element_type: Type of source element (text, table, formula, etc.).
        document_title: Source document name.
        section: Section within the document.
        page: Page number (if available).
        content_preview: First ~150 chars of the source content.
        confidence: Relevance score of this source.
        metadata: Additional metadata for downstream use.
    """

    id: int = 0
    source: str = ""
    element_type: str = "text"
    document_title: str = ""
    section: str = ""
    page: int = 0
    content_preview: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "element_type": self.element_type,
            "document_title": self.document_title,
            "section": self.section,
            "page": self.page,
            "content_preview": self.content_preview,
            "confidence": self.confidence,
        }


class CitationBuilder:
    """Build precise citations from enriched retrieval results.

    Usage:
        builder = CitationBuilder()
        citations = builder.build(enriched_results, response_text)
    """

    def __init__(self, min_confidence: float = 0.1) -> None:
        self._min_confidence = min_confidence

    def build(
        self,
        results: List[Any],
        response_text: str = "",
    ) -> List[Citation]:
        """Build citations from retrieval results.

        Args:
            results: List of EnrichedResult objects (or compatible dicts).
            response_text: The generated response. If provided, citations
                          are filtered to only include sources that likely
                          contributed to the response (content overlap).

        Returns:
            Ordered list of Citation objects.
        """
        if not results:
            return []

        citations: List[Citation] = []
        seen_sources: Set[str] = set()

        for idx, item in enumerate(results):
            score = self._get_score(item)
            if score < self._min_confidence:
                continue

            metadata = self._get_metadata(item)
            element_type = self._get_element_type(item)
            doc_title = metadata.get("document_title", metadata.get("filename", ""))
            section = metadata.get("section", "")
            page = metadata.get("page", 0)
            content = self._get_content(item)

            # Build the source string.
            source = self._format_source(
                element_type, doc_title, section, page, metadata
            )

            # Deduplicate by source string.
            if source in seen_sources:
                continue
            seen_sources.add(source)

            # If response text provided, check for content overlap.
            if response_text and not self._content_overlaps(content, response_text):
                continue

            citations.append(
                Citation(
                    id=len(citations) + 1,
                    source=source,
                    element_type=element_type,
                    document_title=doc_title,
                    section=section,
                    page=page,
                    content_preview=content[:150] + "..." if len(content) > 150 else content,
                    confidence=score,
                    metadata=metadata,
                )
            )

        logger.debug(
            "Citations built",
            total=len(citations),
            types={c.element_type: 0 for c in citations},
        )
        return citations

    def format_citation_block(self, citations: List[Citation]) -> str:
        """Format citations as a markdown reference block.

        Returns a string like:
            **Sources:**
            1. Table from VM-20 Section 3 (page 45) — 2017 CSO mortality rates
            2. ASC 944-40, Section 2.1 (page 12) — Long-duration contract reserves
        """
        if not citations:
            return ""

        lines = ["**Sources:**"]
        for c in citations:
            line = f"{c.id}. {c.source}"
            if c.content_preview:
                # Add a brief description from content.
                brief = c.content_preview.split(".")[0]
                if brief and len(brief) < 100:
                    line += f" — {brief}"
            lines.append(line)

        return "\n".join(lines)

    # -- source formatting --------------------------------------------------

    @staticmethod
    def _format_source(
        element_type: str,
        doc_title: str,
        section: str,
        page: int,
        metadata: Dict[str, Any],
    ) -> str:
        """Format a human-readable source string based on element type."""
        parts: List[str] = []

        # Element type prefix.
        type_labels = {
            "table": "Table",
            "table_proposition": "Table data",
            "formula": "Formula",
            "callout": "Important note",
            "image": "Diagram",
        }
        label = type_labels.get(element_type, "")

        # Document title.
        if doc_title:
            if label:
                parts.append(f"{label} from {doc_title}")
            else:
                parts.append(doc_title)
        elif label:
            parts.append(label)

        # Section.
        if section:
            parts.append(f"Section '{section}'")

        # Page.
        if page and page > 0:
            parts.append(f"page {page}")

        # Columns for tables.
        columns = metadata.get("columns", [])
        if columns and element_type in ("table", "table_proposition"):
            col_str = ", ".join(columns[:4])
            if len(columns) > 4:
                col_str += f" +{len(columns) - 4} more"
            parts.append(f"columns: {col_str}")

        # Jurisdiction for regulatory content.
        jurisdiction = metadata.get("jurisdiction", "")
        if jurisdiction and jurisdiction != "unknown":
            parts.append(jurisdiction)

        if not parts:
            return "Retrieved document"

        # Join: first part as-is, rest in parentheses.
        main = parts[0]
        if len(parts) > 1:
            main += " (" + ", ".join(parts[1:]) + ")"

        return main

    # -- content overlap detection ------------------------------------------

    @staticmethod
    def _content_overlaps(source_content: str, response_text: str, threshold: int = 3) -> bool:
        """Check if source content likely contributed to the response.

        Uses simple word overlap heuristic — if N or more significant
        words from the source appear in the response, consider it used.
        """
        if not source_content or not response_text:
            return True  # If we can't check, assume it contributed.

        # Extract significant words (length > 4, not common).
        stop_words = {"about", "above", "after", "again", "based", "being",
                      "between", "could", "would", "should", "which", "their",
                      "there", "these", "those", "under", "where", "while"}
        source_words = set(
            w.lower() for w in source_content.split()
            if len(w) > 4 and w.lower() not in stop_words
        )
        response_lower = response_text.lower()

        overlap = sum(1 for w in source_words if w in response_lower)
        return overlap >= threshold

    # -- accessors ----------------------------------------------------------

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
    def _get_metadata(item: Any) -> Dict[str, Any]:
        if hasattr(item, "metadata") and isinstance(item.metadata, dict):
            return item.metadata
        if isinstance(item, dict):
            return item.get("metadata", {})
        return {}

    @staticmethod
    def _get_content(item: Any) -> str:
        if hasattr(item, "content"):
            return item.content
        if isinstance(item, dict):
            return item.get("content", "")
        return ""
