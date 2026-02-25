"""
Multi-representation table processor.

For each table element, produces up to three representations:
1. Text summary → embedded in Qdrant (for semantic search)
2. Structured metadata → Qdrant payload (for filtered search)
3. Original markdown → structured store (for LLM generation)

Large tables (>10 rows) also get row-level propositions for
fine-grained retrieval.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog

from ..layout_parser import DocumentElement
from .text_processor import ProcessedChunk

logger = structlog.get_logger()

# Row threshold for generating per-row propositions.
_LARGE_TABLE_ROW_THRESHOLD = 10


class TableProcessor:
    """Process table elements into multi-representation chunks."""

    def __init__(self, llm_provider: Optional[Any] = None) -> None:
        """Initialize with optional LLM provider for summary generation.

        If no provider is given, summaries are built heuristically from
        column headers and row counts.
        """
        self._llm = llm_provider

    async def process(
        self,
        elements: List[DocumentElement],
        document_title: str = "",
    ) -> List[ProcessedChunk]:
        """Process table elements into searchable chunks.

        Each table produces:
        - 1 summary chunk (always)
        - N proposition chunks for large tables (optional)
        """
        chunks: List[ProcessedChunk] = []

        for elem in elements:
            structured = elem.structured_content or {}
            markdown = structured.get("markdown", elem.content)
            columns = structured.get("columns", [])
            row_count = structured.get("row_count", 0)
            data = structured.get("data", [])

            # 1. Generate text summary for embedding.
            summary = await self._generate_summary(
                markdown, columns, row_count, document_title, elem.parent_section
            )

            chunks.append(
                ProcessedChunk(
                    embedding_text=summary,
                    display_text=markdown,
                    context_prefix=f"Table from '{document_title}', section '{elem.parent_section}': " if document_title else "",
                    element_type="table",
                    page_number=elem.page_number,
                    parent_section=elem.parent_section,
                    metadata={
                        "processor": "table",
                        "columns": columns,
                        "row_count": row_count,
                        "has_structured_data": bool(data),
                        "original_markdown": markdown,
                    },
                    importance=1.1,  # Slight boost for tables.
                )
            )

            # 2. Generate row-level propositions for large tables.
            if row_count > _LARGE_TABLE_ROW_THRESHOLD and data and columns:
                propositions = self._generate_propositions(
                    data, columns, document_title, elem.parent_section
                )
                for prop in propositions:
                    chunks.append(
                        ProcessedChunk(
                            embedding_text=prop,
                            display_text=prop,
                            element_type="table_proposition",
                            page_number=elem.page_number,
                            parent_section=elem.parent_section,
                            metadata={
                                "processor": "table",
                                "proposition": True,
                                "source_table_columns": columns,
                            },
                        )
                    )

        logger.debug(
            "Table processing complete",
            input_elements=len(elements),
            output_chunks=len(chunks),
        )
        return chunks

    async def _generate_summary(
        self,
        markdown: str,
        columns: List[str],
        row_count: int,
        document_title: str,
        section: str,
    ) -> str:
        """Generate a text summary of the table for embedding."""
        # Try LLM-based summary first.
        if self._llm is not None:
            try:
                prompt = (
                    f"Summarize this table in 1-2 sentences. "
                    f"Include what it shows, key columns, and data context.\n\n"
                    f"Document: {document_title}\n"
                    f"Section: {section}\n"
                    f"Table:\n{markdown[:3000]}"
                )
                return await self._llm.generate(prompt, task="extraction")
            except Exception as e:
                logger.warning("LLM table summary failed, using heuristic", error=str(e))

        # Heuristic fallback.
        col_str = ", ".join(columns[:8])
        if len(columns) > 8:
            col_str += f", ... ({len(columns)} columns total)"

        parts = []
        if document_title:
            parts.append(f"Table from '{document_title}'")
        if section:
            parts.append(f"in section '{section}'")
        parts.append(f"with columns: {col_str}")
        parts.append(f"containing {row_count} rows")

        return " ".join(parts) + "."

    @staticmethod
    def _generate_propositions(
        data: List[Dict[str, Any]],
        columns: List[str],
        document_title: str,
        section: str,
    ) -> List[str]:
        """Generate natural-language propositions from table rows.

        Example: "The mortality rate for age 45 male nonsmoker is 0.00298 (2017 CSO)"
        """
        propositions: List[str] = []

        # Use first column as the key identifier.
        key_col = columns[0] if columns else None
        value_cols = columns[1:] if len(columns) > 1 else []

        for row in data:
            if not key_col:
                break

            key_val = row.get(key_col, "")
            if not key_val:
                continue

            # Build proposition from non-empty value columns.
            parts = []
            for col in value_cols:
                val = row.get(col, "")
                if val is not None and str(val).strip():
                    parts.append(f"{col}: {val}")

            if parts:
                prop = f"{key_col} '{key_val}' has {', '.join(parts)}"
                if section:
                    prop += f" (from {section})"
                propositions.append(prop)

        return propositions
