"""
Generation layer — context assembly, verification, compliance, and citations.

Replaces the multi-pass generation pipeline with a streamlined flow:
  1. Assemble rich context (original tables, formulas, callouts)
  2. Generate with grounding instructions
  3. Single verification pass (max 3 LLM calls total)
  4. Post-generation compliance check
  5. Precise citations from structured metadata
"""

from .context_assembler import ContextAssembler, AssembledContext
from .verification import VerificationPipeline, VerificationResult
from .compliance_check import ComplianceChecker, ComplianceResult
from .citation_builder import CitationBuilder, Citation

__all__ = [
    "ContextAssembler",
    "AssembledContext",
    "VerificationPipeline",
    "VerificationResult",
    "ComplianceChecker",
    "ComplianceResult",
    "CitationBuilder",
    "Citation",
]
