"""
Lightweight verification pipeline.

Replaces the previous multi-pass flow (up to 7 LLM calls) with a
streamlined 2-3 call approach:

  Stage 1: Generate with grounding instructions (1 LLM call)
  Stage 2: Verify hallucination + completeness + relevance (1 LLM call)
  Stage 3: If verification fails, one correction attempt (1 LLM call)

Max 3 LLM calls total. The key insight: better input context (original
tables, formulas) reduces the need for correction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional, Union

import structlog

logger = structlog.get_logger()

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    """Load a prompt template from the prompts/ directory."""
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


@dataclass
class VerificationResult:
    """Result of the verification pipeline.

    Attributes:
        response: The final (possibly corrected) response text.
        verified: Whether the response passed verification.
        llm_calls: Number of LLM calls made.
        hallucination_detected: Whether hallucination was found.
        completeness_score: 0.0-1.0 completeness rating.
        relevance_score: 0.0-1.0 relevance rating.
        correction_applied: Whether a correction was made.
        verification_details: Raw verification output.
    """

    response: str = ""
    verified: bool = False
    llm_calls: int = 0
    hallucination_detected: bool = False
    completeness_score: float = 0.0
    relevance_score: float = 0.0
    correction_applied: bool = False
    verification_details: Dict[str, Any] = field(default_factory=dict)


class VerificationPipeline:
    """Generate → Verify → (optionally Correct) pipeline.

    Usage:
        pipeline = VerificationPipeline(llm_provider)
        result = await pipeline.run(query, context_text)
        print(result.response)
        print(f"Verified: {result.verified}, LLM calls: {result.llm_calls}")
    """

    def __init__(
        self,
        llm_provider: Any,
        skip_verification: bool = False,
    ) -> None:
        """Initialize the pipeline.

        Args:
            llm_provider: LLM provider with generate() and generate_json().
            skip_verification: If True, skip Stage 2+3 (for low-latency mode).
        """
        self._llm = llm_provider
        self._skip_verification = skip_verification

        # Load prompt templates.
        self._generation_prompt = _load_prompt("generation")
        self._verification_prompt = _load_prompt("verification")
        self._correction_prompt = _load_prompt("correction")

    async def run(
        self,
        query: str,
        context_text: str,
        conversation_history: str = "",
    ) -> VerificationResult:
        """Run the full generate → verify → correct pipeline.

        Args:
            query: User query.
            context_text: Assembled context from ContextAssembler.
            conversation_history: Prior conversation for continuity.

        Returns:
            VerificationResult with the final response and metadata.
        """
        result = VerificationResult()

        # --- Stage 1: Generate with grounding instructions ---
        response = await self._generate(query, context_text, conversation_history)
        result.response = response
        result.llm_calls = 1

        if self._skip_verification or not response.strip():
            result.verified = True
            return result

        # --- Stage 2: Verify ---
        verification = await self._verify(query, context_text, response)
        result.llm_calls = 2
        result.verification_details = verification
        result.hallucination_detected = verification.get("hallucination_detected", False)
        result.completeness_score = verification.get("completeness_score", 0.0)
        result.relevance_score = verification.get("relevance_score", 0.0)
        result.verified = verification.get("overall_pass", False)

        if result.verified:
            logger.info(
                "Verification passed",
                completeness=result.completeness_score,
                relevance=result.relevance_score,
            )
            return result

        # --- Stage 3: Correct (one attempt) ---
        correction_instructions = verification.get("correction_needed", "")
        if not correction_instructions:
            # Verification failed but no specific instructions — return as-is.
            logger.warning("Verification failed but no correction instructions provided")
            return result

        corrected = await self._correct(
            query, context_text, response, correction_instructions
        )
        result.response = corrected
        result.correction_applied = True
        result.llm_calls = 3
        result.verified = True  # Accept after one correction.

        logger.info(
            "Correction applied",
            hallucination=result.hallucination_detected,
            completeness=result.completeness_score,
        )
        return result

    async def run_streaming(
        self,
        query: str,
        context_text: str,
        conversation_history: str = "",
    ) -> AsyncGenerator[Union[str, VerificationResult], None]:
        """Stream Stage 1 tokens, then verify/correct.

        Yields:
            str tokens during generation (Stage 1).
            If correction needed, yields more str tokens (Stage 3).
            Final yield is always a VerificationResult with metadata.
        """
        result = VerificationResult()

        # --- Stage 1: Stream the generation ---
        prompt = self._generation_prompt.format(query=query, context=context_text)
        if conversation_history:
            prompt = f"CONVERSATION HISTORY:\n{conversation_history}\n\n{prompt}"

        accumulated = []
        async for token in self._llm.generate_stream(prompt, task="generation"):
            accumulated.append(token)
            yield token

        response = "".join(accumulated)
        result.response = response
        result.llm_calls = 1

        if self._skip_verification or not response.strip():
            result.verified = True
            yield result
            return

        # --- Stage 2: Verify (blocking, fast) ---
        verification = await self._verify(query, context_text, response)
        result.llm_calls = 2
        result.verification_details = verification
        result.hallucination_detected = verification.get("hallucination_detected", False)
        result.completeness_score = verification.get("completeness_score", 0.0)
        result.relevance_score = verification.get("relevance_score", 0.0)
        result.verified = verification.get("overall_pass", False)

        if result.verified:
            logger.info(
                "Verification passed (streaming)",
                completeness=result.completeness_score,
                relevance=result.relevance_score,
            )
            yield result
            return

        # --- Stage 3: Stream correction ---
        correction_instructions = verification.get("correction_needed", "")
        if not correction_instructions:
            logger.warning("Verification failed but no correction instructions (streaming)")
            yield result
            return

        correction_prompt = self._correction_prompt.format(
            query=query,
            context=context_text,
            response=response,
            correction_instructions=correction_instructions,
        )

        # Signal that we're replacing the response
        yield "\n\n---\n\n"

        corrected_parts = []
        async for token in self._llm.generate_stream(correction_prompt, task="generation"):
            corrected_parts.append(token)
            yield token

        result.response = "".join(corrected_parts)
        result.correction_applied = True
        result.llm_calls = 3
        result.verified = True

        logger.info(
            "Correction applied (streaming)",
            hallucination=result.hallucination_detected,
            completeness=result.completeness_score,
        )
        yield result

    # -- stages -------------------------------------------------------------

    async def _generate(
        self,
        query: str,
        context: str,
        conversation_history: str,
    ) -> str:
        """Stage 1: Generate response with grounding instructions."""
        prompt = self._generation_prompt.format(
            query=query,
            context=context,
        )

        # Prepend conversation history if available.
        if conversation_history:
            prompt = f"CONVERSATION HISTORY:\n{conversation_history}\n\n{prompt}"

        return await self._llm.generate(prompt, task="generation")

    async def _verify(
        self,
        query: str,
        context: str,
        response: str,
    ) -> Dict[str, Any]:
        """Stage 2: Verify response against context."""
        prompt = self._verification_prompt.format(
            query=query,
            context=context,
            response=response,
        )

        last_error = None
        for attempt in range(2):
            try:
                verification = await self._llm.generate_json(
                    prompt, task="scoring"
                )
                return verification
            except Exception as e:
                last_error = e
                if attempt == 0:
                    logger.warning("Verification parsing failed, retrying", error=str(e))

        # Both attempts failed — mark as unverified to trigger correction
        logger.warning("Verification parsing failed after retry", error=str(last_error))
        return {
            "overall_pass": False,
            "completeness_score": 0.0,
            "relevance_score": 0.0,
            "parse_failure": True,
            "correction_needed": "Verification could not parse the response. Please regenerate a clear, well-structured answer grounded in the provided context.",
        }

    async def _correct(
        self,
        query: str,
        context: str,
        response: str,
        correction_instructions: str,
    ) -> str:
        """Stage 3: Apply one correction based on verification feedback."""
        prompt = self._correction_prompt.format(
            query=query,
            context=context,
            response=response,
            correction_instructions=correction_instructions,
        )

        return await self._llm.generate(prompt, task="generation")
