"""
Self-Correction & Reasoning Module
Implements multi-pass reasoning and self-verification with zero hardcoding
"""

import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from openai import AsyncOpenAI
import structlog

logger = structlog.get_logger()

@dataclass
class VerificationResult:
    """Result of response verification"""
    needs_correction: bool
    confidence: float
    issues: List[str]
    reasoning: str
    missing_info: List[str]
    quality_scores: Dict[str, float]

@dataclass
class ReasoningStep:
    """A single step in chain of thought"""
    step_number: int
    description: str
    content: str
    confidence: float

@dataclass
class ReasoningChain:
    """Complete chain of thought reasoning"""
    steps: List[ReasoningStep]
    conclusion: str
    overall_confidence: float
    methodology: str

@dataclass
class CorrectedResponse:
    """Result of self-correction process"""
    final_response: str
    reasoning_chain: Optional[ReasoningChain]
    verification_result: Optional[VerificationResult]
    correction_applied: bool
    iterations: int
    metadata: Dict[str, Any]

class SelfVerificationModule:
    """Verifies response quality and identifies issues"""

    def __init__(self, llm_client: AsyncOpenAI, config: Dict[str, Any]):
        self.llm_client = llm_client
        self.config = config.get('self_verification', {})
        self.logger = logger.bind(component="self_verification")

    async def verify_response(
        self,
        query: str,
        response: str,
        context: str,
        metadata: Dict = None
    ) -> VerificationResult:
        """Comprehensive response verification"""

        verification_criteria = self.config.get('verification_criteria', [])

        if not verification_criteria:
            # Default verification if no criteria configured
            verification_criteria = [
                "factual_accuracy",
                "completeness",
                "logical_consistency",
                "hallucination_check"
            ]

        # Run verification checks
        verification_results = await self._run_verification_checks(
            query, response, context, verification_criteria, metadata
        )

        # Aggregate results
        overall_result = self._aggregate_verification_results(verification_results)

        self.logger.info("Response verification completed",
                        needs_correction=overall_result.needs_correction,
                        confidence=overall_result.confidence,
                        issues_count=len(overall_result.issues))

        return overall_result

    async def _run_verification_checks(
        self,
        query: str,
        response: str,
        context: str,
        criteria: List[str],
        metadata: Dict = None
    ) -> Dict[str, Any]:
        """Run individual verification checks"""

        results = {}

        # Build verification prompt dynamically
        verification_prompt = self._build_verification_prompt(
            query, response, context, criteria, metadata
        )

        try:
            llm_response = await self.llm_client.chat.completions.create(
                model=self.config.get('model', 'gpt-4o-mini'),
                messages=[{"role": "user", "content": verification_prompt}],
                temperature=self.config.get('temperature', 0.1),
                max_tokens=self.config.get('max_tokens', 800),
                response_format={"type": "json_object"}
            )

            results = json.loads(llm_response.choices[0].message.content)

        except Exception as e:
            self.logger.error(f"Verification check failed: {e}")
            # Return conservative verification result
            results = {
                "overall_assessment": {
                    "needs_correction": True,
                    "confidence": 0.3,
                    "reasoning": f"Verification failed: {e}"
                }
            }

        return results

    def _build_verification_prompt(
        self,
        query: str,
        response: str,
        context: str,
        criteria: List[str],
        metadata: Dict = None
    ) -> str:
        """Build verification prompt from configuration"""

        base_prompt = self.config.get('verification_prompt_template', """
You are a quality assurance expert. Verify this response against the given criteria.

QUERY: "{query}"

CONTEXT: {context}

RESPONSE TO VERIFY: {response}

VERIFICATION CRITERIA: {criteria}

METADATA: {metadata}

For each criterion, provide detailed analysis. Return JSON:
{{
    "criterion_results": {{
        "{criterion}": {{
            "score": 0.0-1.0,
            "passed": true/false,
            "issues": ["specific issue 1", "specific issue 2"],
            "reasoning": "detailed explanation"
        }}
    }},
    "overall_assessment": {{
        "needs_correction": true/false,
        "confidence": 0.0-1.0,
        "reasoning": "overall assessment explanation",
        "priority_issues": ["most critical issues"],
        "missing_information": ["what's missing if incomplete"],
        "quality_scores": {{
            "accuracy": 0.0-1.0,
            "completeness": 0.0-1.0,
            "clarity": 0.0-1.0,
            "relevance": 0.0-1.0
        }}
    }}
}}
""")

        # Get criteria descriptions from config
        criteria_descriptions = {}
        for criterion in criteria:
            criteria_descriptions[criterion] = self.config.get(
                'criteria_descriptions', {}
            ).get(criterion, f"Verify {criterion}")

        return base_prompt.format(
            query=query,
            context=context[:2000] + "..." if len(context) > 2000 else context,
            response=response[:1500] + "..." if len(response) > 1500 else response,
            criteria=criteria_descriptions,
            metadata=metadata or {}
        )

    def _aggregate_verification_results(self, results: Dict[str, Any]) -> VerificationResult:
        """Aggregate individual verification results"""

        overall = results.get('overall_assessment', {})

        return VerificationResult(
            needs_correction=overall.get('needs_correction', True),
            confidence=overall.get('confidence', 0.5),
            issues=overall.get('priority_issues', []),
            reasoning=overall.get('reasoning', ''),
            missing_info=overall.get('missing_information', []),
            quality_scores=overall.get('quality_scores', {})
        )

class ChainOfThoughtGenerator:
    """Generates responses with explicit reasoning chains"""

    def __init__(self, llm_client: AsyncOpenAI, config: Dict[str, Any]):
        self.llm_client = llm_client
        self.config = config.get('chain_of_thought', {})
        self.logger = logger.bind(component="chain_of_thought")

    async def generate_with_reasoning(
        self,
        query: str,
        context: str,
        methodology: str = None
    ) -> Tuple[str, ReasoningChain]:
        """Generate response with explicit reasoning chain"""

        # Determine reasoning methodology
        if not methodology:
            methodology = await self._select_reasoning_methodology(query)

        # Generate reasoning chain
        reasoning_chain = await self._generate_reasoning_chain(query, context, methodology)

        # Generate final response based on reasoning
        final_response = await self._generate_from_reasoning(query, reasoning_chain, context)

        self.logger.info("Chain of thought generation completed",
                        methodology=methodology,
                        steps=len(reasoning_chain.steps),
                        confidence=reasoning_chain.overall_confidence)

        return final_response, reasoning_chain

    async def _select_reasoning_methodology(self, query: str) -> str:
        """Select appropriate reasoning methodology for query"""

        methodologies = self.config.get('methodologies', {})

        if not methodologies:
            return "general"

        methodology_prompt = self.config.get('methodology_selection_prompt', """
Analyze this query to select the best reasoning methodology.

Query: "{query}"

Available Methodologies:
{methodologies}

Return JSON:
{{
    "selected_methodology": "methodology_name",
    "reasoning": "why this methodology is best",
    "confidence": 0.0-1.0
}}
""").format(
            query=query,
            methodologies="\n".join([f"- {name}: {desc}" for name, desc in methodologies.items()])
        )

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.config.get('model', 'gpt-4o-mini'),
                messages=[{"role": "user", "content": methodology_prompt}],
                temperature=self.config.get('temperature', 0.1),
                max_tokens=self.config.get('max_tokens', 300),
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            return result.get('selected_methodology', 'general')

        except Exception as e:
            self.logger.error(f"Methodology selection failed: {e}")
            return "general"

    async def _generate_reasoning_chain(
        self,
        query: str,
        context: str,
        methodology: str
    ) -> ReasoningChain:
        """Generate step-by-step reasoning chain"""

        methodology_config = self.config.get('methodologies', {}).get(methodology, {})

        reasoning_prompt = methodology_config.get('prompt_template',
            self.config.get('default_reasoning_prompt', """
Think step-by-step to answer this query. Show your complete reasoning process.

QUERY: {query}
CONTEXT: {context}

Methodology: {methodology}

Structure your reasoning as follows:
1. Understanding the question: [what exactly is being asked]
2. Relevant information: [key facts from context that apply]
3. Analysis: [how the facts relate to the question]
4. Reasoning: [logical steps to reach conclusion]
5. Conclusion: [direct answer with confidence level]

Return JSON:
{{
    "reasoning_steps": [
        {{
            "step_number": 1,
            "description": "Understanding the question",
            "content": "detailed explanation",
            "confidence": 0.0-1.0
        }}
    ],
    "conclusion": "final answer",
    "overall_confidence": 0.0-1.0,
    "methodology_used": "{methodology}"
}}
""")).format(query=query, context=context, methodology=methodology)

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.config.get('model', 'gpt-4o-mini'),
                messages=[{"role": "user", "content": reasoning_prompt}],
                temperature=self.config.get('temperature', 0.2),
                max_tokens=self.config.get('max_tokens', 1200),
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # Convert to ReasoningChain object
            steps = []
            for step_data in result.get('reasoning_steps', []):
                step = ReasoningStep(
                    step_number=step_data.get('step_number', 0),
                    description=step_data.get('description', ''),
                    content=step_data.get('content', ''),
                    confidence=step_data.get('confidence', 0.5)
                )
                steps.append(step)

            return ReasoningChain(
                steps=steps,
                conclusion=result.get('conclusion', ''),
                overall_confidence=result.get('overall_confidence', 0.5),
                methodology=result.get('methodology_used', methodology)
            )

        except Exception as e:
            self.logger.error(f"Reasoning chain generation failed: {e}")
            # Return minimal reasoning chain
            return ReasoningChain(
                steps=[],
                conclusion="",
                overall_confidence=0.0,
                methodology=methodology
            )

    async def _generate_from_reasoning(
        self,
        query: str,
        reasoning_chain: ReasoningChain,
        context: str
    ) -> str:
        """Generate final response based on reasoning chain"""

        if not reasoning_chain.steps:
            # Fallback to direct generation
            return await self._direct_generation_fallback(query, context)

        response_prompt = self.config.get('response_from_reasoning_prompt', """
Based on this step-by-step reasoning, provide a clear, comprehensive answer.

ORIGINAL QUERY: {query}

REASONING PROCESS:
{reasoning_steps}

CONCLUSION: {conclusion}

Provide a well-structured response that:
1. Directly answers the query
2. Is clear and comprehensive
3. Maintains logical flow from the reasoning
4. Includes relevant details from the reasoning process

Response:
""").format(
            query=query,
            reasoning_steps="\n".join([f"{step.step_number}. {step.description}: {step.content}"
                                     for step in reasoning_chain.steps]),
            conclusion=reasoning_chain.conclusion
        )

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.config.get('model', 'gpt-4o-mini'),
                messages=[{"role": "user", "content": response_prompt}],
                temperature=self.config.get('temperature', 0.1),
                max_tokens=self.config.get('max_tokens', 1000)
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            self.logger.error(f"Response generation from reasoning failed: {e}")
            return reasoning_chain.conclusion or "Unable to generate response"

    async def _direct_generation_fallback(self, query: str, context: str) -> str:
        """Fallback direct generation if reasoning fails"""

        prompt = f"Answer this query based on the provided context:\n\nQuery: {query}\nContext: {context}\n\nAnswer:"

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.config.get('model', 'gpt-4o-mini'),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            self.logger.error(f"Fallback generation failed: {e}")
            return "Unable to generate response"

class MultiPassGenerator:
    """Orchestrates multi-pass generation with self-correction"""

    def __init__(self, llm_client: AsyncOpenAI, config: Dict[str, Any]):
        self.llm_client = llm_client
        self.config = config.get('multi_pass', {})
        self.logger = logger.bind(component="multi_pass_generator")

        # Initialize components
        self.verifier = SelfVerificationModule(llm_client, config)
        self.reasoning_generator = ChainOfThoughtGenerator(llm_client, config)

    async def generate_with_correction(
        self,
        query: str,
        context: str,
        max_iterations: int = None,
        use_reasoning: bool = None
    ) -> CorrectedResponse:
        """Generate response with self-correction"""

        max_iterations = max_iterations or self.config.get('max_iterations', 3)
        use_reasoning = use_reasoning if use_reasoning is not None else self.config.get('use_reasoning', True)

        iteration = 0
        current_response = ""
        reasoning_chain = None
        all_verifications = []

        while iteration < max_iterations:
            iteration += 1

            # Generate response (with or without reasoning)
            if use_reasoning and iteration == 1:  # Use reasoning for first pass
                current_response, reasoning_chain = await self.reasoning_generator.generate_with_reasoning(
                    query, context
                )
            else:
                # Subsequent iterations or non-reasoning mode
                current_response = await self._generate_correction(
                    query, context, current_response, all_verifications
                )

            # Verify response
            verification = await self.verifier.verify_response(
                query, current_response, context, {"iteration": iteration}
            )
            all_verifications.append(verification)

            self.logger.info(f"Iteration {iteration} completed",
                           needs_correction=verification.needs_correction,
                           confidence=verification.confidence)

            # Check if response is good enough
            if not verification.needs_correction or verification.confidence >= self.config.get('confidence_threshold', 0.8):
                break

            # Check if we've reached max iterations
            if iteration >= max_iterations:
                self.logger.warning("Max iterations reached without satisfactory response")
                break

        final_verification = all_verifications[-1] if all_verifications else None

        return CorrectedResponse(
            final_response=current_response,
            reasoning_chain=reasoning_chain,
            verification_result=final_verification,
            correction_applied=iteration > 1,
            iterations=iteration,
            metadata={
                "all_verifications": all_verifications,
                "reasoning_used": use_reasoning and reasoning_chain is not None,
                "max_iterations_reached": iteration >= max_iterations
            }
        )

    async def _generate_correction(
        self,
        query: str,
        context: str,
        previous_response: str,
        previous_verifications: List[VerificationResult]
    ) -> str:
        """Generate corrected response based on previous attempts"""

        if not previous_verifications:
            # First iteration - direct generation
            return await self._direct_generation(query, context)

        # Build correction prompt
        latest_verification = previous_verifications[-1]

        correction_prompt = self.config.get('correction_prompt_template', """
Improve this response based on the identified issues.

QUERY: {query}
CONTEXT: {context}

PREVIOUS RESPONSE:
{previous_response}

IDENTIFIED ISSUES:
{issues}

SPECIFIC FEEDBACK:
{feedback}

Generate an improved response that addresses all the issues while maintaining accuracy and completeness.

Improved Response:
""").format(
            query=query,
            context=context[:2000] + "..." if len(context) > 2000 else context,
            previous_response=previous_response,
            issues="\n".join([f"- {issue}" for issue in latest_verification.issues]),
            feedback=latest_verification.reasoning
        )

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.config.get('model', 'gpt-4o-mini'),
                messages=[{"role": "user", "content": correction_prompt}],
                temperature=self.config.get('temperature', 0.1),
                max_tokens=self.config.get('max_tokens', 1200)
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            self.logger.error(f"Correction generation failed: {e}")
            return previous_response  # Return previous if correction fails

    async def _direct_generation(self, query: str, context: str) -> str:
        """Direct response generation for first iteration"""

        prompt = self.config.get('initial_generation_prompt', """
Answer this query comprehensively based on the provided context.

QUERY: {query}
CONTEXT: {context}

Provide a complete, accurate response:
""").format(query=query, context=context)

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.config.get('model', 'gpt-4o-mini'),
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.get('temperature', 0.1),
                max_tokens=self.config.get('max_tokens', 1000)
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            self.logger.error(f"Direct generation failed: {e}")
            return "Unable to generate response"

class SelfCorrectionManager:
    """Main orchestrator for self-correction and reasoning"""

    def __init__(self, llm_client: AsyncOpenAI, config: Dict[str, Any]):
        self.llm_client = llm_client
        self.config = config
        self.logger = logger.bind(component="self_correction_manager")

        # Initialize components
        self.multi_pass_generator = MultiPassGenerator(llm_client, config)
        self.enabled = config.get('enabled', True)

    async def enhanced_generation(
        self,
        query: str,
        context: str,
        base_generation_func=None,
        options: Dict[str, Any] = None
    ) -> CorrectedResponse:
        """Main entry point for enhanced generation with self-correction"""

        if not self.enabled:
            # Use base generation if self-correction disabled
            if base_generation_func:
                response = await base_generation_func(query, context)
            else:
                response = "Self-correction disabled and no base generation provided"

            return CorrectedResponse(
                final_response=response,
                reasoning_chain=None,
                verification_result=None,
                correction_applied=False,
                iterations=1,
                metadata={"mode": "base_generation"}
            )

        options = options or {}

        try:
            # Use multi-pass generation with self-correction
            result = await self.multi_pass_generator.generate_with_correction(
                query=query,
                context=context,
                max_iterations=options.get('max_iterations'),
                use_reasoning=options.get('use_reasoning')
            )

            self.logger.info("Enhanced generation completed",
                           iterations=result.iterations,
                           correction_applied=result.correction_applied,
                           final_confidence=result.verification_result.confidence if result.verification_result else None)

            return result

        except Exception as e:
            self.logger.error(f"Enhanced generation failed: {e}")

            # Fallback to base generation
            if base_generation_func:
                try:
                    fallback_response = await base_generation_func(query, context)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback generation also failed: {fallback_error}")
                    fallback_response = "Generation failed"
            else:
                fallback_response = f"Enhanced generation failed: {e}"

            return CorrectedResponse(
                final_response=fallback_response,
                reasoning_chain=None,
                verification_result=None,
                correction_applied=False,
                iterations=1,
                metadata={"mode": "fallback", "error": str(e)}
            )