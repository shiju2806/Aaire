"""
Callout/sidebar processor.

Callout boxes, important notes, and sidebars are chunked as standalone
units with elevated retrieval importance. They are never merged with
surrounding text.
"""

from __future__ import annotations

from typing import Any, Dict, List

import structlog

from ..layout_parser import DocumentElement
from .text_processor import ProcessedChunk
from ...providers.config_loader import get_config, get_nested

logger = structlog.get_logger()


class CalloutProcessor:
    """Process callout/sidebar elements as high-priority standalone chunks."""

    def __init__(self) -> None:
        config = get_config("scoring")
        self._boost = get_nested(config, "extraction", "callout_boost", default=1.3)

    def process(
        self,
        elements: List[DocumentElement],
        document_title: str = "",
    ) -> List[ProcessedChunk]:
        """Process callout elements.

        Each callout becomes a standalone chunk with importance boost.
        Callouts are never split or merged with other elements.
        """
        chunks: List[ProcessedChunk] = []

        for elem in elements:
            content = elem.content.strip()
            if not content:
                continue

            context_prefix = ""
            if document_title or elem.parent_section:
                parts = []
                if document_title:
                    parts.append(f"From '{document_title}'")
                if elem.parent_section:
                    parts.append(f"section '{elem.parent_section}'")
                context_prefix = ", ".join(parts) + " [Important Note]: "

            chunks.append(
                ProcessedChunk(
                    embedding_text=f"{context_prefix}{content}",
                    display_text=content,
                    context_prefix=context_prefix,
                    element_type="callout",
                    page_number=elem.page_number,
                    parent_section=elem.parent_section,
                    metadata={
                        "processor": "callout",
                        "is_callout": True,
                    },
                    importance=self._boost,
                )
            )

        logger.debug(
            "Callout processing complete",
            input_elements=len(elements),
            output_chunks=len(chunks),
        )
        return chunks
