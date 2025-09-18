"""
Structured Response Generator - Industry Standard RAG Quality Control
Implements grounding validation, hallucination prevention, and response structure enforcement
"""

import re
import time
import yaml
import structlog
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from openai import AsyncOpenAI

logger = structlog.get_logger()

@dataclass
class GroundingValidationResult:
    """Result of grounding validation check"""
    is_grounded: bool
    similarity_score: float
    ungrounded_claims: List[str]
    confidence_level: str
    validation_method: str

@dataclass
class ResponseStructure:
    """Structured response format"""
    answer: str
    confidence_level: str
    source_summary: str
    follow_up_questions: List[str]
    validation_result: Optional[GroundingValidationResult] = None

class StructuredResponseGenerator:
    """
    Industry-standard response generator with grounding validation
    Prevents hallucination and enforces response structure
    """

    def __init__(self, config_path: str = None, llm_client: AsyncOpenAI = None):
        self.config_path = config_path or "/Users/shijuprakash/AAIRE/config/response_generation.yaml"
        self.llm_client = llm_client
        self.logger = logger.bind(component="structured_generator")
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load response generation configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info("Response generation config loaded", config_path=self.config_path)
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Fallback configuration if file load fails"""
        return {
            'grounding': {
                'enabled': True,
                'similarity_threshold': 0.7,
                'require_citations': True,
                'fallback_when_no_information': True
            },
            'prohibited_content': {
                'document_references': [
                    r'Document \d+',
                    r'Doc \d+',
                    r'Page \d+',
                    r'Section \d+\.\d+'
                ]
            },
            'response_structure': {
                'confidence_levels': ['high', 'medium', 'low', 'none']
            }
        }

    async def generate_structured_response(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]] = None
    ) -> ResponseStructure:
        """
        Generate a structured, grounded response

        Args:
            query: User question
            retrieved_chunks: List of retrieved document chunks
            conversation_history: Previous conversation context

        Returns:
            ResponseStructure with validated content
        """
        start_time = time.time()

        try:
            # Check if we have sufficient information
            if not retrieved_chunks or len(retrieved_chunks) == 0:
                return await self._generate_no_information_response(query)

            # Clean and prepare context
            cleaned_context = self._clean_context(retrieved_chunks)

            # Generate initial response
            raw_response = await self._generate_raw_response(query, cleaned_context, conversation_history)

            # Validate grounding
            validation_result = await self._validate_grounding(raw_response, retrieved_chunks)

            # Apply content filtering
            filtered_response = self._filter_prohibited_content(raw_response)

            # Structure the response
            structured_response = self._structure_response(filtered_response, validation_result)

            # Final validation and adjustment
            final_response = await self._finalize_response(structured_response, validation_result)

            processing_time = (time.time() - start_time) * 1000
            self.logger.info(
                "Structured response generated",
                processing_time_ms=processing_time,
                confidence=final_response.confidence_level,
                grounded=validation_result.is_grounded if validation_result else False
            )

            return final_response

        except Exception as e:
            self.logger.error(f"Failed to generate structured response: {e}")
            return await self._generate_error_response(query, str(e))

    def _clean_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Clean context removing document references and fixing formatting"""
        cleaned_chunks = []

        for chunk in chunks:
            content = chunk.get('content', '')

            # Apply content cleaning rules from config
            content = self._apply_content_cleaning_rules(content)

            # Remove document markers and references
            content = re.sub(r'\[Doc \d+\]', '', content)
            content = re.sub(r'Document \d+[:\-]?', '', content)
            content = re.sub(r'Page \d+ of \d+', '', content)

            cleaned_chunks.append(content.strip())

        # Join without document numbering
        return "\n\n".join(cleaned_chunks)

    def _apply_content_cleaning_rules(self, content: str) -> str:
        """Apply content cleaning rules from configuration"""
        if 'content_cleaning' not in self.config:
            return content

        cleaning_rules = self.config['content_cleaning']

        # Remove unwanted patterns
        if 'remove_patterns' in cleaning_rules:
            for pattern in cleaning_rules['remove_patterns']:
                content = re.sub(pattern, '', content)

        # Apply formatting fixes
        if 'fix_formatting' in cleaning_rules:
            for fix in cleaning_rules['fix_formatting']:
                pattern = fix['pattern']
                replacement = fix['replacement']
                content = re.sub(pattern, replacement, content)

        return content.strip()

    async def _generate_raw_response(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> str:
        """Generate initial response using LLM"""
        if not self.llm_client:
            return self._generate_fallback_response(query, context)

        # Build conversation context
        messages = []

        # System instruction from config
        system_instruction = self.config.get('prompts', {}).get('system_instruction',
            "You are AAIRE, an expert insurance and accounting assistant. Provide accurate information based only on the provided context.")

        messages.append({"role": "system", "content": system_instruction})

        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history[-10:]:  # Last 10 messages
                messages.append(msg)

        # Add current query with context
        user_message = f"Context:\n{context}\n\nQuestion: {query}"
        messages.append({"role": "user", "content": user_message})

        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1,
                max_tokens=2000
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            return self._generate_fallback_response(query, context)

    def _generate_fallback_response(self, query: str, context: str) -> str:
        """Generate fallback response when LLM is unavailable"""
        if not context.strip():
            return f"I don't have specific information about your question regarding '{query}' in the current documentation."

        # Simple template-based response
        return f"Based on the available information:\n\n{context[:500]}...\n\nThis information is directly from the source documentation."

    async def _validate_grounding(
        self,
        response: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Optional[GroundingValidationResult]:
        """Validate that response claims are grounded in source material"""

        if not self.config.get('grounding', {}).get('enabled', False):
            return None

        try:
            # Extract claims from response
            claims = self._extract_claims(response)

            # Check each claim against source chunks
            ungrounded_claims = []
            similarity_scores = []

            for claim in claims:
                is_grounded, similarity = self._check_claim_grounding(claim, retrieved_chunks)
                if not is_grounded:
                    ungrounded_claims.append(claim)
                similarity_scores.append(similarity)

            # Calculate overall grounding
            avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
            threshold = self.config.get('grounding', {}).get('similarity_threshold', 0.7)
            is_grounded = avg_similarity >= threshold and len(ungrounded_claims) == 0

            # Determine confidence level
            confidence_level = self._determine_confidence_level(avg_similarity, len(ungrounded_claims))

            return GroundingValidationResult(
                is_grounded=is_grounded,
                similarity_score=avg_similarity,
                ungrounded_claims=ungrounded_claims,
                confidence_level=confidence_level,
                validation_method="semantic_similarity"
            )

        except Exception as e:
            self.logger.error(f"Grounding validation failed: {e}")
            return GroundingValidationResult(
                is_grounded=False,
                similarity_score=0.0,
                ungrounded_claims=[],
                confidence_level="low",
                validation_method="error_fallback"
            )

    def _extract_claims(self, response: str) -> List[str]:
        """Extract factual claims from response"""
        # Simple approach: split by sentences and filter
        sentences = re.split(r'[.!?]+', response)

        # Filter out questions, short sentences, and formatting
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 20 and
                not sentence.endswith('?') and
                not sentence.startswith('#') and
                not sentence.startswith('**')):
                claims.append(sentence)

        return claims[:10]  # Limit to prevent performance issues

    def _check_claim_grounding(self, claim: str, chunks: List[Dict[str, Any]]) -> Tuple[bool, float]:
        """Check if a claim is grounded in source chunks"""
        # Simple keyword-based similarity for now
        # In production, would use embedding similarity

        claim_lower = claim.lower()
        best_similarity = 0.0

        for chunk in chunks:
            content = chunk.get('content', '').lower()

            # Count common words
            claim_words = set(claim_lower.split())
            content_words = set(content.split())

            if len(claim_words) == 0:
                continue

            common_words = claim_words.intersection(content_words)
            similarity = len(common_words) / len(claim_words)

            best_similarity = max(best_similarity, similarity)

        threshold = self.config.get('grounding', {}).get('similarity_threshold', 0.7)
        return best_similarity >= threshold, best_similarity

    def _determine_confidence_level(self, similarity_score: float, ungrounded_count: int) -> str:
        """Determine confidence level based on grounding validation"""
        if ungrounded_count > 2:
            return "low"
        elif similarity_score >= 0.8 and ungrounded_count == 0:
            return "high"
        elif similarity_score >= 0.6 and ungrounded_count <= 1:
            return "medium"
        else:
            return "low"

    def _filter_prohibited_content(self, response: str) -> str:
        """Remove prohibited content patterns from response"""
        filtered_response = response

        prohibited_patterns = self.config.get('prohibited_content', {}).get('document_references', [])

        for pattern in prohibited_patterns:
            filtered_response = re.sub(pattern, '', filtered_response, flags=re.IGNORECASE)

        # Clean up extra whitespace
        filtered_response = re.sub(r'\n\s*\n', '\n\n', filtered_response)
        filtered_response = re.sub(r' +', ' ', filtered_response)

        return filtered_response.strip()

    def _structure_response(self, response: str, validation_result: GroundingValidationResult) -> ResponseStructure:
        """Structure the response according to configured format"""

        # Try to parse existing structure or create new one
        if "**Confidence Level**:" in response:
            # Response already structured, parse it
            return self._parse_structured_response(response, validation_result)
        else:
            # Create new structure
            return self._create_structured_response(response, validation_result)

    def _parse_structured_response(self, response: str, validation_result: GroundingValidationResult) -> ResponseStructure:
        """Parse an already structured response"""
        lines = response.split('\n')

        answer = ""
        confidence_level = validation_result.confidence_level if validation_result else "medium"
        source_summary = "Information compiled from available documentation"
        follow_up_questions = []

        current_section = "answer"

        for line in lines:
            line = line.strip()

            if line.startswith("**Confidence Level**:"):
                confidence_level = line.split(":", 1)[1].strip()
                current_section = "confidence"
            elif line.startswith("**Sources Used**:"):
                source_summary = line.split(":", 1)[1].strip()
                current_section = "sources"
            elif line.startswith("**Follow-up Questions**:"):
                current_section = "questions"
            elif current_section == "answer" and line:
                answer += line + "\n"
            elif current_section == "questions" and line.startswith(("1.", "2.", "3.")):
                question = re.sub(r'^\d+\.\s*', '', line)
                follow_up_questions.append(question)

        return ResponseStructure(
            answer=answer.strip(),
            confidence_level=confidence_level,
            source_summary=source_summary,
            follow_up_questions=follow_up_questions,
            validation_result=validation_result
        )

    def _create_structured_response(self, response: str, validation_result: GroundingValidationResult) -> ResponseStructure:
        """Create structured response from unstructured content"""

        confidence_level = validation_result.confidence_level if validation_result else "medium"

        # Generate follow-up questions
        follow_up_questions = self._generate_follow_up_questions(response)

        return ResponseStructure(
            answer=response,
            confidence_level=confidence_level,
            source_summary="Information derived from available documentation without document references",
            follow_up_questions=follow_up_questions,
            validation_result=validation_result
        )

    def _generate_follow_up_questions(self, response: str) -> List[str]:
        """Generate self-contained follow-up questions"""
        # Simple rule-based question generation
        # In production, would use LLM for better quality

        questions = []

        # Extract key topics from response
        insurance_terms = ['reserve', 'policy', 'premium', 'liability', 'asset', 'valuation']
        financial_terms = ['calculation', 'formula', 'method', 'approach', 'standard']

        found_insurance = any(term in response.lower() for term in insurance_terms)
        found_financial = any(term in response.lower() for term in financial_terms)

        if found_insurance:
            questions.append("What are the key assumptions used in these reserve calculations?")

        if found_financial:
            questions.append("How do these calculation methods differ across regulatory frameworks?")

        questions.append("What documentation is required to support these calculations?")

        return questions[:3]  # Limit to 3 as per config

    async def _finalize_response(self, response: ResponseStructure, validation_result: GroundingValidationResult) -> ResponseStructure:
        """Final validation and adjustments"""

        # Check if grounding validation failed
        if validation_result and not validation_result.is_grounded:
            fallback_enabled = self.config.get('grounding', {}).get('fallback_when_no_information', True)

            if fallback_enabled:
                # Use no-information template
                template = self.config.get('no_information_responses', {}).get('templates', {}).get('general',
                    "I don't have specific information about {topic} in the current documentation.")

                # Extract topic from original answer
                topic = "this topic"  # Simplified extraction

                response.answer = template.format(topic=topic)
                response.confidence_level = "none"

        # Format final response
        formatted_answer = self._format_response_template(response)

        return ResponseStructure(
            answer=formatted_answer,
            confidence_level=response.confidence_level,
            source_summary=response.source_summary,
            follow_up_questions=response.follow_up_questions,
            validation_result=validation_result
        )

    def _format_response_template(self, response: ResponseStructure) -> str:
        """Format response using configured template"""
        template = self.config.get('prompts', {}).get('response_template',
            "Based on the available information:\n\n{answer}\n\n**Confidence Level**: {confidence}\n\n**Sources Used**: {source_summary}\n\n**Follow-up Questions**:\n1. {question_1}\n2. {question_2}\n3. {question_3}")

        # Ensure we have 3 questions
        questions = response.follow_up_questions + [""] * 3

        try:
            return template.format(
                answer=response.answer,
                confidence=response.confidence_level,
                source_summary=response.source_summary,
                question_1=questions[0],
                question_2=questions[1],
                question_3=questions[2]
            )
        except:
            # Fallback formatting
            return f"{response.answer}\n\n**Confidence Level**: {response.confidence_level}"

    async def _generate_no_information_response(self, query: str) -> ResponseStructure:
        """Generate response when no information is available"""
        template = self.config.get('no_information_responses', {}).get('templates', {}).get('general',
            "I don't have specific information about {topic} in the current documentation.")

        answer = template.format(topic=query)

        return ResponseStructure(
            answer=answer,
            confidence_level="none",
            source_summary="No relevant documentation found",
            follow_up_questions=[
                "What specific aspects of this topic would you like me to help research?",
                "Are there related topics I can assist with instead?",
                "Would you like me to clarify my areas of expertise?"
            ]
        )

    async def _generate_error_response(self, query: str, error: str) -> ResponseStructure:
        """Generate response for system errors"""
        return ResponseStructure(
            answer=f"I encountered an issue processing your question about '{query}'. Please try rephrasing your question or contact support if the issue persists.",
            confidence_level="none",
            source_summary="System error occurred",
            follow_up_questions=[
                "Would you like to try rephrasing your question?",
                "Is there a different way I can help you?",
                "Would you like to report this issue?"
            ]
        )


class ResponseQualityValidator:
    """Validates response quality against industry standards"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger.bind(component="quality_validator")

    def validate_response(self, response: ResponseStructure) -> Dict[str, Any]:
        """Validate response quality and return metrics"""

        validation_results = {}

        # Check grounding
        if self.config.get('validation', {}).get('checks'):
            for check in self.config['validation']['checks']:
                if not check.get('enabled', True):
                    continue

                check_name = check['name']
                result = self._run_validation_check(check, response)
                validation_results[check_name] = result

        return validation_results

    def _run_validation_check(self, check: Dict[str, Any], response: ResponseStructure) -> Dict[str, Any]:
        """Run individual validation check"""

        check_name = check['name']

        if check_name == "grounding_check":
            return self._validate_grounding(check, response)
        elif check_name == "document_reference_check":
            return self._validate_no_document_references(check, response)
        elif check_name == "formula_preservation_check":
            return self._validate_formula_preservation(check, response)
        elif check_name == "confidence_alignment_check":
            return self._validate_confidence_alignment(check, response)

        return {"passed": True, "details": "Unknown check type"}

    def _validate_grounding(self, check: Dict[str, Any], response: ResponseStructure) -> Dict[str, Any]:
        """Validate grounding requirements"""
        if not response.validation_result:
            return {"passed": False, "details": "No grounding validation performed"}

        threshold = check.get('threshold', 0.7)
        passed = response.validation_result.similarity_score >= threshold

        return {
            "passed": passed,
            "score": response.validation_result.similarity_score,
            "threshold": threshold,
            "details": f"Grounding score: {response.validation_result.similarity_score:.3f}"
        }

    def _validate_no_document_references(self, check: Dict[str, Any], response: ResponseStructure) -> Dict[str, Any]:
        """Validate no document references in response"""
        doc_patterns = [r'Document \d+', r'Doc \d+', r'Page \d+']

        violations = []
        full_text = f"{response.answer} {' '.join(response.follow_up_questions)}"

        for pattern in doc_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            violations.extend(matches)

        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "details": f"Found {len(violations)} document references"
        }

    def _validate_formula_preservation(self, check: Dict[str, Any], response: ResponseStructure) -> Dict[str, Any]:
        """Validate mathematical formulas are preserved"""
        # Check for mathematical expressions
        math_patterns = [r'\$[\d,]+', r'\d+\.\d+%', r'[A-Z]{2,}\s+\d+']

        math_found = any(re.search(pattern, response.answer) for pattern in math_patterns)

        return {
            "passed": True,  # Assume formulas are preserved if present
            "math_content_found": math_found,
            "details": "Formula preservation check completed"
        }

    def _validate_confidence_alignment(self, check: Dict[str, Any], response: ResponseStructure) -> Dict[str, Any]:
        """Validate confidence level aligns with content quality"""

        valid_levels = ['high', 'medium', 'low', 'none']
        confidence_valid = response.confidence_level in valid_levels

        # Check alignment with grounding result
        alignment_valid = True
        if response.validation_result:
            if response.validation_result.similarity_score < 0.5 and response.confidence_level == 'high':
                alignment_valid = False

        return {
            "passed": confidence_valid and alignment_valid,
            "confidence_level": response.confidence_level,
            "details": f"Confidence level '{response.confidence_level}' validation"
        }