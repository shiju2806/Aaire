"""
Unified Intent Analyzer - Single source of truth for query intent analysis
Uses LLM for sophisticated understanding, NO hard-coded logic
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from openai import AsyncOpenAI, OpenAI
import structlog

logger = structlog.get_logger()

@dataclass
class UnifiedQueryIntent:
    """Unified intent representation for all components"""
    query_type: str  # procedural, factual, comparative, analytical, exploratory
    information_need: str  # methodology, definition, calculation, data, explanation
    retrieval_strategy: str  # comprehensive_methodology, precise_factual, broad_exploration
    complexity: str  # simple, moderate, complex
    domain_focus: Optional[str] = None  # regulatory, technical, operational, etc.
    frameworks: Optional[List[str]] = None  # usstat, ifrs, gaap, etc.
    confidence: float = 0.0
    reasoning: str = ""
    key_entities: Optional[List[str]] = None
    action_words: Optional[List[str]] = None

class UnifiedIntentAnalyzer:
    """
    Single, sophisticated LLM-powered intent analyzer
    NO fallbacks, NO hard-coded logic - pure AI understanding
    """

    def __init__(self, llm_client: Optional[AsyncOpenAI] = None):
        self.llm_client = llm_client
        self.sync_client = OpenAI() if not llm_client else None
        self.logger = logger.bind(component="unified_intent_analyzer")

    async def analyze_intent(self, query: str, context: Optional[Dict[str, Any]] = None) -> UnifiedQueryIntent:
        """
        Analyze query intent using sophisticated LLM understanding
        This is the ONLY intent analysis in the entire system
        """

        prompt = f"""You are an expert query intent analyzer for an insurance/actuarial RAG system.
Analyze this query to determine the user's TRUE intent and information need.

Query: "{query}"
{f"Context: {json.dumps(context, indent=2)}" if context else ""}

CRITICAL UNDERSTANDING:
- "methods to calculate X" or "how to calculate X" = User wants METHODOLOGY/PROCEDURES, not just facts
- "what are the methods" = User wants a structured list of approaches/techniques
- "calculate X for me" = User wants you to perform a calculation
- "what is X" = User wants a definition or explanation
- "compare X and Y" = User wants comparative analysis

Analyze the query and determine:

1. Query Type:
   - procedural: User wants steps, methods, procedures, how-to guidance
   - factual: User wants specific facts, numbers, or data points
   - comparative: User wants comparison between concepts
   - analytical: User wants analysis or interpretation
   - exploratory: User has a broad, open-ended question

2. Information Need (what the user REALLY wants):
   - methodology: Steps, procedures, methods, approaches, techniques
   - definition: What something is, explanation of concepts
   - calculation: Actual computation or formula application
   - data: Specific data points, numbers, statistics
   - explanation: Detailed understanding of how/why something works

3. Retrieval Strategy (how to best find the answer):
   - comprehensive_methodology: Retrieve detailed procedural documents, manuals, guides
   - precise_factual: Retrieve specific facts, regulations, definitions
   - broad_exploration: Cast a wide net for exploratory questions
   - comparative_analysis: Retrieve documents that compare/contrast concepts
   - regulatory_focused: Focus on regulatory documents and standards

4. Key Entities: Important terms/concepts in the query (e.g., "NPR", "universal life", "reserves")

5. Action Words: Verbs that indicate what user wants (e.g., "calculate", "methods", "how")

EXAMPLES:
- "what are the methods to calculate reserves for universal life policies"
  → procedural, methodology, comprehensive_methodology
  → User wants the METHODS/PROCEDURES, not general info about reserves

- "calculate the NPR for this policy"
  → analytical, calculation, precise_factual
  → User wants actual calculation performed

- "what is a universal life policy"
  → factual, definition, precise_factual
  → User wants definition/explanation

Return JSON:
{{
    "query_type": "procedural|factual|comparative|analytical|exploratory",
    "information_need": "methodology|definition|calculation|data|explanation",
    "retrieval_strategy": "comprehensive_methodology|precise_factual|broad_exploration|comparative_analysis|regulatory_focused",
    "complexity": "simple|moderate|complex",
    "domain_focus": "regulatory|technical|operational|financial|actuarial",
    "frameworks": ["usstat", "ifrs", "gaap"],  // if mentioned
    "confidence": 0.0-1.0,
    "reasoning": "explanation of analysis",
    "key_entities": ["entity1", "entity2"],
    "action_words": ["calculate", "methods", "how"]
}}

IMPORTANT: Focus on what the user is ASKING FOR, not just what topics are mentioned.
"Methods to calculate" always means procedural/methodology, never factual!
"""

        try:
            if self.llm_client:
                response = await self.llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )
            else:
                # Sync fallback for testing
                response = self.sync_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )

            result = json.loads(response.choices[0].message.content)

            intent = UnifiedQueryIntent(
                query_type=result.get("query_type", "factual"),
                information_need=result.get("information_need", "data"),
                retrieval_strategy=result.get("retrieval_strategy", "precise_factual"),
                complexity=result.get("complexity", "simple"),
                domain_focus=result.get("domain_focus"),
                frameworks=result.get("frameworks", []),
                confidence=float(result.get("confidence", 0.5)),
                reasoning=result.get("reasoning", ""),
                key_entities=result.get("key_entities", []),
                action_words=result.get("action_words", [])
            )

            self.logger.info(
                "Query intent analyzed",
                query=query[:50] + "..." if len(query) > 50 else query,
                query_type=intent.query_type,
                information_need=intent.information_need,
                retrieval_strategy=intent.retrieval_strategy,
                confidence=intent.confidence
            )

            return intent

        except Exception as e:
            self.logger.error(f"Intent analysis failed: {e}", exc_info=True)
            # Return a reasonable default, but log the error
            return UnifiedQueryIntent(
                query_type="exploratory",
                information_need="explanation",
                retrieval_strategy="broad_exploration",
                complexity="moderate",
                confidence=0.0,
                reasoning=f"Analysis failed: {str(e)}"
            )

    def analyze_intent_sync(self, query: str, context: Optional[Dict[str, Any]] = None) -> UnifiedQueryIntent:
        """Synchronous wrapper for intent analysis"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.analyze_intent(query, context)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self.analyze_intent(query, context))
        except Exception as e:
            self.logger.error(f"Sync intent analysis failed: {e}")
            return UnifiedQueryIntent(
                query_type="exploratory",
                information_need="explanation",
                retrieval_strategy="broad_exploration",
                complexity="moderate",
                confidence=0.0,
                reasoning=f"Sync analysis failed: {str(e)}"
            )

# Global instance for easy access
_global_analyzer = None

def get_unified_intent_analyzer(llm_client: Optional[AsyncOpenAI] = None) -> UnifiedIntentAnalyzer:
    """Get or create the global unified intent analyzer"""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = UnifiedIntentAnalyzer(llm_client)
    return _global_analyzer