"""
Image/diagram processor.

Sends images to a vision model for captioning, then stores:
1. Text description → embedded in Qdrant (for semantic search)
2. Original image reference → structured store (for LLM generation)

The principle: "search on descriptions, answer from originals."
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog

from ..layout_parser import DocumentElement
from .text_processor import ProcessedChunk

logger = structlog.get_logger()

_VISION_PROMPT = (
    "Describe this diagram from an actuarial, insurance, or accounting document. "
    "Include all labels, relationships, data values, flow direction, and any "
    "text visible in the image. Be specific about numbers and terminology."
)


class ImageProcessor:
    """Process image elements using vision model captioning."""

    def __init__(self, llm_provider: Optional[Any] = None) -> None:
        """Initialize with optional LLM provider that supports vision.

        If no provider is given, images get a placeholder description
        based on available metadata.
        """
        self._llm = llm_provider

    async def process(
        self,
        elements: List[DocumentElement],
        document_title: str = "",
    ) -> List[ProcessedChunk]:
        """Process image elements.

        Each image produces one chunk. The embedding text is the
        description; the image reference is stored in metadata.
        """
        chunks: List[ProcessedChunk] = []

        for elem in elements:
            description = await self._describe_image(
                elem, document_title
            )

            context_prefix = ""
            if document_title or elem.parent_section:
                parts = []
                if document_title:
                    parts.append(f"From '{document_title}'")
                if elem.parent_section:
                    parts.append(f"section '{elem.parent_section}'")
                context_prefix = ", ".join(parts) + " [Diagram/Image]: "

            # Store image reference for later retrieval.
            image_ref = None
            if elem.structured_content:
                image_ref = elem.structured_content.get("image_path") or \
                           elem.structured_content.get("base64")

            chunks.append(
                ProcessedChunk(
                    embedding_text=f"{context_prefix}{description}",
                    display_text=description,
                    context_prefix=context_prefix,
                    element_type="image",
                    page_number=elem.page_number,
                    parent_section=elem.parent_section,
                    metadata={
                        "processor": "image",
                        "image_ref": image_ref,
                        "has_vision_description": self._llm is not None,
                    },
                )
            )

        logger.debug(
            "Image processing complete",
            input_elements=len(elements),
            output_chunks=len(chunks),
        )
        return chunks

    async def _describe_image(
        self,
        elem: DocumentElement,
        document_title: str,
    ) -> str:
        """Generate text description of an image element.

        When Docling VLM is enabled, elem.content already contains the
        VLM-generated description from parsing — no extra API call needed.
        Falls back to a placeholder when content is empty.
        """
        if elem.content and elem.content.strip():
            return elem.content.strip()

        parts = [f"Image/diagram on page {elem.page_number}"]
        if elem.parent_section:
            parts.append(f"in section '{elem.parent_section}'")
        if document_title:
            parts.append(f"from '{document_title}'")
        return " ".join(parts)
