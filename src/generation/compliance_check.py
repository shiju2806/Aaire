"""
Post-generation compliance check.

Defense-in-depth: catches subtle compliance issues that the pre-retrieval
regex-based compliance gate might miss. Uses a lightweight LLM call to
detect specific tax advice, legal opinions, reserve adequacy
recommendations, and other boundary-crossing content.

If detected, appends a professional judgment disclaimer rather than
blocking the response entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger()

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_DISCLAIMER = (
    "\n\n---\n"
    "**Disclaimer:** This information is provided for educational and "
    "reference purposes only. It does not constitute professional advice. "
    "Consult a qualified actuary, accountant, or legal professional before "
    "making decisions based on this information. Specific situations require "
    "professional judgment that accounts for all relevant facts and "
    "circumstances."
)


@dataclass
class ComplianceResult:
    """Result of the post-generation compliance check.

    Attributes:
        response: The response (possibly with disclaimer appended).
        has_issue: Whether a compliance issue was detected.
        issues: List of specific issues found.
        severity: none | low | high.
        disclaimer_added: Whether a disclaimer was appended.
    """

    response: str = ""
    has_issue: bool = False
    issues: List[str] = field(default_factory=list)
    severity: str = "none"
    disclaimer_added: bool = False


class ComplianceChecker:
    """Post-generation compliance checker.

    Usage:
        checker = ComplianceChecker(llm_provider)
        result = await checker.check(response_text)
        final_response = result.response  # May have disclaimer appended
    """

    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        always_add_disclaimer: bool = False,
    ) -> None:
        """Initialize the compliance checker.

        Args:
            llm_provider: LLM provider with generate_json(). If None,
                         falls back to regex-only checks.
            always_add_disclaimer: If True, always append disclaimer
                                  regardless of check result.
        """
        self._llm = llm_provider
        self._always_disclaimer = always_add_disclaimer

        prompt_path = _PROMPTS_DIR / "compliance_check.txt"
        self._prompt_template = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""

    async def check(self, response: str) -> ComplianceResult:
        """Check a response for compliance issues.

        Args:
            response: The generated response text.

        Returns:
            ComplianceResult with the (possibly modified) response.
        """
        result = ComplianceResult(response=response)

        if self._always_disclaimer:
            result.response = response + _DISCLAIMER
            result.disclaimer_added = True
            return result

        # Quick regex pre-check for obvious patterns.
        regex_issues = self._regex_check(response)
        if regex_issues:
            result.has_issue = True
            result.issues = regex_issues
            result.severity = "low"

        # LLM-based check for subtle issues.
        if self._llm and self._prompt_template:
            try:
                llm_result = await self._llm_check(response)
                if llm_result.get("has_compliance_issue", False):
                    result.has_issue = True
                    result.issues.extend(llm_result.get("issues", []))
                    result.severity = llm_result.get("severity", "low")
            except Exception as e:
                logger.warning("LLM compliance check failed", error=str(e))

        # Append disclaimer if any issues detected.
        if result.has_issue:
            result.response = response + _DISCLAIMER
            result.disclaimer_added = True
            logger.info(
                "Compliance issue detected, disclaimer added",
                issues=result.issues,
                severity=result.severity,
            )

        return result

    async def _llm_check(self, response: str) -> Dict[str, Any]:
        """Run LLM-based compliance check."""
        prompt = self._prompt_template.format(response=response)
        return await self._llm.generate_json(prompt, task="scoring")

    @staticmethod
    def _regex_check(response: str) -> List[str]:
        """Quick regex-based check for obvious compliance patterns."""
        import re

        issues: List[str] = []
        response_lower = response.lower()

        # Specific advice patterns.
        advice_patterns = [
            (r"you should (file|claim|deduct|report)", "Specific filing/reporting advice"),
            (r"i recommend (that you|you)", "Direct recommendation"),
            (r"your (reserves?|valuation|filing) (is|are) (adequate|inadequate|sufficient|insufficient)",
             "Reserve adequacy opinion"),
            (r"you (must|need to|are required to) (change|update|revise|adjust) your",
             "Directive professional advice"),
            (r"based on your (situation|data|numbers|figures), (you|i|we)",
             "Situation-specific advice"),
        ]

        for pattern, description in advice_patterns:
            if re.search(pattern, response_lower):
                issues.append(description)

        return issues
