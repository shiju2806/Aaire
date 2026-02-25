"""
Formula/equation processor.

Produces multi-representation chunks for mathematical content:
1. Natural-language description → embedded in Qdrant (for search)
2. LaTeX/original formula → structured store (for LLM generation)
3. Variable definitions → metadata (for knowledge graph, future)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import structlog

from ..layout_parser import DocumentElement
from .text_processor import ProcessedChunk

logger = structlog.get_logger()

# Common variable patterns in actuarial/financial formulas.
_VARIABLE_PATTERN = re.compile(r"\b([A-Z][a-z]*(?:_\{[^}]+\})?)\b")


class FormulaProcessor:
    """Process formula elements into searchable chunks."""

    def __init__(self, llm_provider: Optional[Any] = None) -> None:
        self._llm = llm_provider

    async def process(
        self,
        elements: List[DocumentElement],
        document_title: str = "",
    ) -> List[ProcessedChunk]:
        """Process formula elements.

        Each formula produces one chunk. The embedding text is the
        natural-language description; the original LaTeX/formula is
        stored in metadata for retrieval-augmented generation.
        """
        chunks: List[ProcessedChunk] = []

        for elem in elements:
            structured = elem.structured_content or {}
            latex = structured.get("latex", elem.content)
            raw_content = elem.content

            # Generate natural-language description.
            description = await self._describe_formula(
                latex, raw_content, document_title, elem.parent_section
            )

            # Extract variable names.
            variables = self._extract_variables(latex or raw_content)

            context_prefix = ""
            if document_title or elem.parent_section:
                parts = []
                if document_title:
                    parts.append(f"From '{document_title}'")
                if elem.parent_section:
                    parts.append(f"section '{elem.parent_section}'")
                context_prefix = ", ".join(parts) + ": "

            chunks.append(
                ProcessedChunk(
                    embedding_text=f"{context_prefix}{description}",
                    display_text=raw_content,
                    context_prefix=context_prefix,
                    element_type="formula",
                    page_number=elem.page_number,
                    parent_section=elem.parent_section,
                    metadata={
                        "processor": "formula",
                        "latex": latex,
                        "variables": variables,
                        "original_content": raw_content,
                    },
                    importance=1.2,  # Formulas are high-value in actuarial docs.
                )
            )

        logger.debug(
            "Formula processing complete",
            input_elements=len(elements),
            output_chunks=len(chunks),
        )
        return chunks

    async def _describe_formula(
        self,
        latex: str,
        raw_content: str,
        document_title: str,
        section: str,
    ) -> str:
        """Generate natural-language description of a formula."""
        if self._llm is not None:
            try:
                prompt = (
                    f"Describe this mathematical formula in plain English. "
                    f"Include what it calculates, its variables, and domain context.\n\n"
                    f"Document: {document_title}\n"
                    f"Section: {section}\n"
                    f"Formula: {latex or raw_content}"
                )
                return await self._llm.generate(prompt, task="extraction")
            except Exception as e:
                logger.warning("LLM formula description failed", error=str(e))

        # Heuristic fallback.
        formula_text = latex or raw_content
        parts = [f"Mathematical formula: {formula_text[:300]}"]
        if section:
            parts.append(f"from section '{section}'")
        return " ".join(parts)

    @staticmethod
    def _extract_variables(formula_text: str) -> List[str]:
        """Extract variable names from formula text/LaTeX."""
        if not formula_text:
            return []

        variables = set()

        # LaTeX-style variables: \alpha, x_{i}, V^{net}
        latex_vars = re.findall(r"\\([a-zA-Z]+)", formula_text)
        # Skip LaTeX commands that aren't variables.
        latex_commands = {"frac", "sum", "prod", "int", "lim", "log", "ln",
                         "sin", "cos", "tan", "exp", "sqrt", "text", "mathrm",
                         "left", "right", "begin", "end", "cdot", "times",
                         "leq", "geq", "neq", "approx", "infty"}
        for var in latex_vars:
            if var.lower() not in latex_commands:
                variables.add(var)

        # Plain variable patterns: single uppercase or subscripted.
        plain_vars = re.findall(r"\b([A-Z][a-z_]*)\b", formula_text)
        for var in plain_vars:
            if len(var) <= 10:  # Avoid matching regular words
                variables.add(var)

        return sorted(variables)
