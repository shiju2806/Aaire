"""
Structured Response Generator - Industry Standard RAG Quality Control
Implements grounding validation, hallucination prevention, and response structure enforcement
"""

import re
import time
import yaml
import json
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
class SemanticAlignmentResult:
    """Result of semantic alignment validation"""
    is_aligned: bool
    intent_match_score: float
    content_specificity: str  # "specific", "general", "unrelated"
    query_intent: str
    content_intent: str
    confidence_level: str
    explanation: str

@dataclass
class ResponseStructure:
    """Structured response format"""
    answer: str
    confidence_level: str
    source_summary: str
    follow_up_questions: List[str]
    validation_result: Optional[GroundingValidationResult] = None
    semantic_alignment: Optional[SemanticAlignmentResult] = None

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
        retrieved_chunks: List[Dict[str, Any]] = None,
        processed_context: str = None,
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
            # Use processed context if provided, otherwise process chunks
            if processed_context:
                cleaned_context = processed_context
                # Create dummy chunks for validation (since we have processed context)
                retrieved_chunks = [{'content': processed_context, 'metadata': {}, 'score': 1.0}]
            else:
                # Check if we have sufficient information
                if not retrieved_chunks or len(retrieved_chunks) == 0:
                    return await self._generate_no_information_response(query)
                # Clean and prepare context from raw chunks
                cleaned_context = self._clean_context(retrieved_chunks)

            # Generate initial response
            raw_response = await self._generate_raw_response(query, cleaned_context, conversation_history)
            self.logger.info(f"🚨 PIPELINE DEBUG 1: raw_response length={len(raw_response)}, preview='{raw_response[:200]}...'")

            # Semantic alignment validation - replaces old grounding validation
            semantic_alignment_enabled = self.config.get('semantic_alignment', {}).get('enabled', True)
            if semantic_alignment_enabled:
                semantic_alignment = await self._validate_semantic_alignment(query, raw_response, retrieved_chunks)
                self.logger.info(f"🚨 SEMANTIC ALIGNMENT: is_aligned={semantic_alignment.is_aligned}, intent_match={semantic_alignment.intent_match_score}, specificity={semantic_alignment.content_specificity}")
            else:
                # Skip semantic alignment validation when disabled
                self.logger.info("🚨 SEMANTIC ALIGNMENT: DISABLED - using default alignment")
                semantic_alignment = SemanticAlignmentResult(
                    is_aligned=True,
                    intent_match_score=1.0,
                    content_specificity="specific",
                    query_intent="user_query",
                    content_intent="matching_content",
                    confidence_level="high",
                    explanation="Semantic alignment validation disabled in configuration"
                )

            # Create dummy validation result for compatibility
            validation_result = GroundingValidationResult(
                is_grounded=semantic_alignment.is_aligned,
                similarity_score=semantic_alignment.intent_match_score,
                ungrounded_claims=[],
                confidence_level=semantic_alignment.confidence_level,
                validation_method="semantic_alignment"
            )

            # Apply content filtering
            filtered_response = self._filter_prohibited_content(raw_response)
            self.logger.info(f"🚨 PIPELINE DEBUG 3: filtered_response length={len(filtered_response)}, preview='{filtered_response[:200]}...'")

            # Structure the response with semantic alignment
            structured_response = self._structure_response(filtered_response, validation_result, semantic_alignment)
            self.logger.info(f"🚨 PIPELINE DEBUG 4: structured_response.answer length={len(structured_response.answer)}, preview='{structured_response.answer[:200]}...'")

            # Final validation and adjustment
            final_response = await self._finalize_response(structured_response, validation_result)
            self.logger.info(f"🚨 PIPELINE DEBUG 5: final_response.answer length={len(final_response.answer)}, preview='{final_response.answer[:200]}...'")

            processing_time = (time.time() - start_time) * 1000
            self.logger.info(
                "Structured response generated",
                processing_time_ms=processing_time,
                confidence=final_response.confidence_level,
                grounded=validation_result.is_grounded if validation_result else False
            )

            return final_response

        except Exception as e:
            self.logger.error(f"Failed to generate structured response: {e}", exc_info=True)
            # More detailed error logging
            if "retrieved_chunks" in locals():
                self.logger.info(f"Retrieved chunks count: {len(retrieved_chunks) if retrieved_chunks else 0}")
            if "cleaned_context" in locals():
                self.logger.info(f"Cleaned context length: {len(cleaned_context) if cleaned_context else 0}")

            # Instead of error response, try to generate a basic response from the chunks
            if retrieved_chunks and len(retrieved_chunks) > 0:
                try:
                    # Create a basic response directly from the context without LLM
                    basic_response = self._generate_basic_response_from_chunks(query, retrieved_chunks)
                    return basic_response
                except Exception as fallback_error:
                    self.logger.error(f"Fallback basic response also failed: {fallback_error}")

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
        conversation_history: List[Dict[str, str]] = None,
        force_no_information: bool = False
    ) -> str:
        """Generate initial response using LLM"""
        if not self.llm_client:
            return self._generate_fallback_response(query, context)

        # Build conversation context
        messages = []

        # System instruction from config or special instruction for insufficient info
        if force_no_information:
            system_instruction = """You are AAIRE, an expert insurance and accounting assistant.
            The provided context does not contain specific enough information to answer the query properly.
            You should acknowledge what general information is available but clearly state that you don't have
            the specific methodologies, calculations, or detailed procedures requested.
            Be helpful by suggesting what type of information would be needed."""
        else:
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
            self.logger.info(f"🚨 STRUCTURED DEBUG: Making OpenAI call with context length={len(context)}")
            response = await self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1,
                max_tokens=2000
            )

            result = response.choices[0].message.content.strip()
            self.logger.info(f"🚨 STRUCTURED DEBUG: OpenAI call succeeded, response length={len(result)}")
            return result

        except Exception as e:
            self.logger.error(f"🚨 STRUCTURED DEBUG: LLM generation failed: {e}")
            return self._generate_fallback_response(query, context)

    def _generate_fallback_response(self, query: str, context: str) -> str:
        """Generate fallback response when LLM is unavailable"""
        self.logger.error(f"🚨 FALLBACK DEBUG: context length={len(context)}, context_stripped_length={len(context.strip())}")
        self.logger.error(f"🚨 FALLBACK DEBUG: context preview='{context[:200]}...'")

        if not context.strip():
            return f"I don't have specific information about your question regarding '{query}' in the current documentation."

        # Simple template-based response
        return f"Based on the available information:\n\n{context[:500]}...\n\nThis information is directly from the source documentation."






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

    def _structure_response(self, response: str, validation_result: GroundingValidationResult, semantic_alignment: SemanticAlignmentResult = None) -> ResponseStructure:
        """Structure the response according to configured format"""

        # Try to parse existing structure or create new one
        if "**Confidence Level**:" in response:
            # Response already structured, parse it
            return self._parse_structured_response(response, validation_result, semantic_alignment)
        else:
            # Create new structure
            return self._create_structured_response(response, validation_result, semantic_alignment)

    def _parse_structured_response(self, response: str, validation_result: GroundingValidationResult, semantic_alignment: SemanticAlignmentResult = None) -> ResponseStructure:
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
            validation_result=validation_result,
            semantic_alignment=semantic_alignment
        )

    def _create_structured_response(self, response: str, validation_result: GroundingValidationResult, semantic_alignment: SemanticAlignmentResult = None) -> ResponseStructure:
        """Create structured response from unstructured content"""

        confidence_level = validation_result.confidence_level if validation_result else "medium"

        # Generate follow-up questions
        follow_up_questions = self._generate_follow_up_questions(response)

        return ResponseStructure(
            answer=response,
            confidence_level=confidence_level,
            source_summary="Information derived from available documentation without document references",
            follow_up_questions=follow_up_questions,
            validation_result=validation_result,
            semantic_alignment=semantic_alignment
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

        # Don't format template here - WebSocket already handles this
        # Just return the response as-is

        return ResponseStructure(
            answer=response.answer,
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

    def _generate_basic_response_from_chunks(self, query: str, chunks: List[Dict[str, Any]]) -> ResponseStructure:
        """Generate basic response directly from chunks when LLM fails"""
        try:
            # Clean and combine chunk content
            cleaned_context = self._clean_context(chunks)

            # Create a simple response based on the available content
            if cleaned_context.strip():
                answer = f"Based on the available documentation:\n\n{cleaned_context[:1500]}..."
                if len(cleaned_context) > 1500:
                    answer += "\n\n(Additional information available in source documents)"

                confidence_level = "medium"
                source_summary = "Information extracted from uploaded documentation"
            else:
                answer = f"I found relevant documents for '{query}' but encountered issues processing the content."
                confidence_level = "low"
                source_summary = "Relevant documents found but content processing failed"

            return ResponseStructure(
                answer=answer,
                confidence_level=confidence_level,
                source_summary=source_summary,
                follow_up_questions=[
                    "Would you like me to search for more specific information?",
                    "Are there particular aspects of this topic you'd like me to focus on?",
                    "Would a different phrasing of your question be helpful?"
                ]
            )
        except Exception as e:
            self.logger.error(f"Basic response generation failed: {e}")
            # Final fallback - return minimal response
            return ResponseStructure(
                answer=f"I found relevant information about '{query}' but encountered technical difficulties processing it.",
                confidence_level="low",
                source_summary="Technical processing issue occurred",
                follow_up_questions=[
                    "Would you like to try rephrasing your question?",
                    "Are there related topics I can help with?",
                    "Would you like to report this technical issue?"
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

    def _extract_critical_phrases(self, query: str) -> List[str]:
        """Dynamic extraction of critical phrases without hardcoding"""
        import re

        phrases = []

        # Method 1: Quoted phrases (explicit user emphasis)
        quoted = re.findall(r'"([^"]+)"', query)
        phrases.extend([q.lower() for q in quoted if len(q.split()) > 1])

        # Method 2: Capitalized multi-word terms (likely proper nouns/products)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', query)
        phrases.extend([p.lower() for p in proper_nouns])

        # Method 3: Adjacent words that form meaningful units
        # Look for patterns like "adjective + noun" or "noun + noun"
        words = query.split()
        for i in range(len(words) - 1):
            # Two consecutive words that might form a concept
            bigram = f"{words[i].lower()} {words[i+1].lower()}"
            # Filter: must contain at least one meaningful word (3+ chars)
            if any(len(word) >= 3 for word in [words[i], words[i+1]]):
                phrases.append(bigram)

        # Method 4: Trigrams for compound concepts
        for i in range(len(words) - 2):
            trigram = f"{words[i].lower()} {words[i+1].lower()} {words[i+2].lower()}"
            # Only keep if it looks like a meaningful phrase
            if len(trigram) > 10:  # Reasonable length filter
                phrases.append(trigram)

        return list(set(phrases))  # Remove duplicates

    async def _validate_semantic_alignment(
        self,
        query: str,
        response: str,
        chunks: List[Dict[str, Any]]
    ) -> SemanticAlignmentResult:
        """
        Validate semantic alignment between query intent and content using LLM
        Replaces word-based grounding with intent-aware validation
        """
        try:
            # Extract critical phrases from query
            critical_phrases = self._extract_critical_phrases(query)
            self.logger.info(f"🔍 ENHANCED VALIDATION: Extracted critical phrases: {critical_phrases}")

            # Summarize available content for analysis
            content_summary = self._summarize_chunks(chunks)

            # Check which critical phrases appear in content
            phrases_in_content = []
            phrases_missing = []

            for phrase in critical_phrases:
                found_in_content = any(phrase in chunk.get('content', '').lower() for chunk in chunks)
                if found_in_content:
                    phrases_in_content.append(phrase)
                else:
                    phrases_missing.append(phrase)

            self.logger.info(f"🔍 PHRASE ANALYSIS: Found={phrases_in_content}, Missing={phrases_missing}")

            # Enhanced LLM-based semantic alignment validation
            alignment_prompt = f"""
You are validating whether retrieved content can properly address a user query.

USER QUERY: "{query}"

CRITICAL PHRASES DETECTED: {critical_phrases}
- Found in content: {phrases_in_content}
- Missing from content: {phrases_missing}

AVAILABLE CONTENT SUMMARY:
{content_summary}

PROPOSED RESPONSE (first 500 chars):
{response[:500]}...

🚨 CRITICAL VALIDATION: If the query contains specific phrases (like "whole life") but the content discusses different concepts (like "universal life"), this is a CONCEPT CONFUSION error.

Analyze the semantic alignment between the query intent and available content. Return ONLY a JSON response:

{{
    "query_intent": "brief description of what user is asking for",
    "content_intent": "brief description of what content actually covers",
    "intent_match_score": 0.0-1.0,
    "content_specificity": "specific|general|unrelated",
    "is_aligned": true/false,
    "confidence_level": "high|medium|low|none",
    "explanation": "brief explanation of alignment assessment"
}}

SCORING GUIDE:
- intent_match_score: 0.9+ = excellent match, 0.7-0.9 = good match, 0.5-0.7 = partial match, <0.5 = poor match
- content_specificity: "specific" = detailed methodologies/procedures, "general" = principles only, "unrelated" = different topic
- is_aligned: true if intent_match_score >= 0.4 OR (query asks for specific calculation/procedure AND content_specificity != "unrelated")

ENHANCED VALIDATION:
- Calculation queries should be marked as "specific" if content contains relevant formulas or procedures
- If critical phrases are missing from content, reduce intent_match_score significantly
- If query asks about one concept but content discusses a different (though related) concept, mark as "unrelated"
- Example: Query about "whole life" but content only has "universal life" = CONCEPT CONFUSION = unrelated
- Check if response generates information not present in the content summary
"""

            response_text = await self._call_llm(alignment_prompt, max_tokens=300)

            # Parse JSON response
            try:
                alignment_data = json.loads(response_text.strip())

                # Apply business logic overrides for insurance calculation queries
                intent_match_score = float(alignment_data.get("intent_match_score", 0.0))
                content_specificity = alignment_data.get("content_specificity", "unrelated")
                is_aligned = alignment_data.get("is_aligned", False)

                # Override for specific calculation queries
                if any(phrase in query.lower() for phrase in ["how to calculate", "calculate the reserves", "reserve calculation", "how do i calculate"]):
                    if content_specificity != "unrelated" and intent_match_score >= 0.3:
                        is_aligned = True
                        self.logger.info(f"Calculation query override: Aligned due to specific calculation request")

                return SemanticAlignmentResult(
                    is_aligned=is_aligned,
                    intent_match_score=intent_match_score,
                    content_specificity=content_specificity,
                    query_intent=alignment_data.get("query_intent", ""),
                    content_intent=alignment_data.get("content_intent", ""),
                    confidence_level=alignment_data.get("confidence_level", "none"),
                    explanation=alignment_data.get("explanation", "")
                )

            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse semantic alignment JSON: {e}")
                # Return conservative fallback
                return SemanticAlignmentResult(
                    is_aligned=False,
                    intent_match_score=0.0,
                    content_specificity="unrelated",
                    query_intent=query,
                    content_intent="Unable to analyze",
                    confidence_level="none",
                    explanation="JSON parsing failed"
                )

        except Exception as e:
            self.logger.error(f"Semantic alignment validation failed: {e}")
            # Return conservative fallback
            return SemanticAlignmentResult(
                is_aligned=False,
                intent_match_score=0.0,
                content_specificity="unrelated",
                query_intent=query,
                content_intent="Validation error",
                confidence_level="none",
                explanation=f"Validation error: {str(e)}"
            )

    def _summarize_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Create a concise summary of available content for semantic analysis"""
        if not chunks:
            return "No content available"

        # Get first few chunks to understand content scope
        sample_chunks = chunks[:5]  # Analyze first 5 chunks

        content_snippets = []
        for chunk in sample_chunks:
            content = chunk.get('content', '').strip()
            if content:
                # Get first 200 chars of each chunk
                snippet = content[:200].replace('\n', ' ')
                content_snippets.append(snippet)

        combined_content = " ... ".join(content_snippets)

        # If content is too long, truncate
        if len(combined_content) > 1000:
            combined_content = combined_content[:1000] + "..."

        return combined_content

    async def _call_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        """Helper method to call LLM for validation tasks"""
        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return ""

