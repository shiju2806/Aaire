"""
Response Generation Service Module

Handles all response generation operations including:
- Main response generation with document context
- Organizational structure responses
- Enhanced single-pass responses
- Chunked response processing for large document sets
- Follow-up question generation
"""

import re
import asyncio
import structlog
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

logger = structlog.get_logger()


class ResponseGenerator:
    """Handles response generation with various processing modes"""

    def __init__(self, llm_client=None, async_client=None, memory_manager=None,
                 formatting_manager=None, query_analyzer=None, config=None):
        """Initialize generator with required components"""
        self.llm = llm_client
        self.async_client = async_client
        self.memory_manager = memory_manager
        self.formatting_manager = formatting_manager
        self.query_analyzer = query_analyzer
        self.config = config or {}

        # Extract model name from config
        self.actual_model = self.config.get('llm_config', {}).get('model', 'gpt-4o-mini')

        # Extract response configuration
        response_config = self.config.get('response_config', {})
        self.base_max_tokens = response_config.get('max_tokens', 4000)  # Keep as base for dynamic calculation
        self.simple_response_threshold = response_config.get('simple_response_threshold', 8)
        self.relevance_threshold = response_config.get('relevance_threshold', 0.15)
        self.min_content_length = response_config.get('min_content_length', 100)
        self.min_response_part_length = response_config.get('min_response_part_length', 50)

    def calculate_dynamic_max_tokens(self, query: str, doc_count: int) -> int:
        """Calculate dynamic max_tokens based on query complexity and document count"""

        # Base calculation factors
        query_length = len(query)
        query_words = len(query.split())

        # Query complexity scoring
        complexity_score = 0

        # Length factor (longer queries often need longer responses)
        if query_length > 100:
            complexity_score += 2
        elif query_length > 50:
            complexity_score += 1

        # Word count factor
        if query_words > 15:
            complexity_score += 2
        elif query_words > 8:
            complexity_score += 1

        # Question type analysis
        complex_indicators = [
            'how', 'what', 'why', 'explain', 'describe', 'analyze', 'compare',
            'detail', 'breakdown', 'difference', 'relationship', 'process'
        ]
        simple_indicators = ['is', 'are', 'can', 'does', 'will', 'define']

        query_lower = query.lower()
        complex_matches = sum(1 for indicator in complex_indicators if indicator in query_lower)
        simple_matches = sum(1 for indicator in simple_indicators if indicator in query_lower)

        if complex_matches > simple_matches:
            complexity_score += complex_matches
        elif simple_matches > 0:
            complexity_score -= 1

        # Document count factor (more docs = potentially longer response)
        doc_factor = min(doc_count // 3, 3)  # Cap at 3 additional points
        complexity_score += doc_factor

        # Calculate dynamic max_tokens - more generous for complete responses
        if complexity_score <= 0:
            # Very simple queries
            dynamic_tokens = 2000
        elif complexity_score <= 2:
            # Simple queries
            dynamic_tokens = 2500
        elif complexity_score <= 4:
            # Medium complexity
            dynamic_tokens = 3000
        elif complexity_score <= 6:
            # High complexity
            dynamic_tokens = 3500
        else:
            # Very complex queries
            dynamic_tokens = 4000

        # Ensure within reasonable bounds
        dynamic_tokens = max(600, min(dynamic_tokens, self.base_max_tokens))

        logger.info(f"🎯 Dynamic token calculation: query_len={query_length}, words={query_words}, "
                   f"complexity_score={complexity_score}, docs={doc_count} → max_tokens={dynamic_tokens}")

        return dynamic_tokens

    async def generate_response(
        self,
        query: str,
        retrieved_docs: List[Dict],
        user_context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict]] = None,
        session_id: Optional[str] = None
    ) -> str:
        """Generate response using retrieved documents and conversation context"""

        # Build conversation context using memory manager
        conversation_context = ""
        if self.memory_manager and session_id:
            conversation_context = await self.memory_manager.get_conversation_context(session_id)
            if conversation_context:
                conversation_context = f"\n\n{conversation_context}\n"

        # Check if we have relevant documents
        if not retrieved_docs:
            # No relevant documents found - return message indicating lack of information
            logger.info(f"❌ No documents retrieved for query: '{query[:50]}...'")
            return "I don't have any relevant information in the uploaded documents to answer this question. Please ensure you've uploaded the necessary documents or rephrase your query."

        # Trust vector search results - if documents were retrieved with high similarity scores, use them
        # Use dynamic chunked processing for all document-based queries
        return await self.process_with_chunked_enhancement(query, retrieved_docs, conversation_context)

    async def process_with_chunked_enhancement(self, query: str, retrieved_docs: List[Dict], conversation_context: str) -> str:
        """Process documents with enhanced chunked approach"""
        logger.info(f"📄 Processing {len(retrieved_docs)} documents with enhanced chunked approach")

        # Determine processing approach based on document count
        if len(retrieved_docs) <= self.simple_response_threshold:
            # Enhanced single-pass for smaller document sets
            logger.info("📋 Using enhanced single-pass approach")
            return self.generate_enhanced_single_pass(query, retrieved_docs, conversation_context)
        else:
            # Chunked processing for larger document sets
            logger.info("📋 Using chunked processing approach")
            return await self.generate_chunked_response(query, retrieved_docs, conversation_context)

    def generate_organizational_response(self, query: str, documents: List[Dict], conversation_context: str) -> str:
        """Generate response for organizational structure queries"""
        context = "\n\n".join([doc['content'] for doc in documents])

        prompt = f"""You are AAIRE, an expert in insurance accounting and actuarial matters.
{conversation_context}
Question: {query}

Organizational data:
{context}

🚨 CRITICAL CONSTRAINT: ONLY use information explicitly stated in the provided documents above.
If the documents do not contain sufficient information to answer the query, respond with:
"I don't have sufficient information about [specific topic] in the uploaded documents to provide a complete answer."

DO NOT use your general knowledge, training data, or external information beyond what is explicitly stated in the documents.

Provide a clear organizational breakdown based ONLY on the spatial extraction data found in the documents.
Use appropriate headings and structure the information clearly."""

        response = self.llm.complete(prompt)
        return response.text.strip()

    def generate_enhanced_single_pass(self, query: str, documents: List[Dict], conversation_context: str) -> str:
        """Enhanced single-pass response for smaller document sets"""
        context = "\n\n".join([doc['content'] for doc in documents])

        prompt = f"""You are AAIRE, an expert in insurance accounting.
{conversation_context}
Question: {query}

Documents:
{context}

🚨 CRITICAL CONSTRAINT: ONLY use information explicitly stated in the provided documents above.
If the documents do not contain sufficient information to answer the query, respond with:
"I don't have sufficient information about [specific topic] in the uploaded documents to provide a complete answer."

DO NOT use your general knowledge, training data, or external information beyond what is explicitly stated in the documents.

CONTENT REQUIREMENTS:
- ONLY include information directly found in the provided documents
- Quote exact formulas, calculations, and technical details from the documents
- Reference specific document sections when citing information
- If documents contain partial information, clearly state what is missing
- Preserve ALL technical details exactly as stated in documents
- Convert complex mathematical notation to readable text format only when present in documents

FORMATTING REQUIREMENTS:
- Use proper markdown formatting with headers: # for main sections, ## for subsections
- Use numbered lists (1., 2., 3.) with proper line breaks
- Use bullet points (-) for sub-items
- Keep mathematical formulas and expressions clear and readable
- Use **bold** only for emphasis within text, not for headers
- Ensure proper spacing between sections and lists

Structure your response to systematically cover only the topics found in the source material using clear markdown formatting."""

        response = self.llm.complete(prompt)
        return response.text.strip()

    async def generate_chunked_response(self, query: str, documents: List[Dict], conversation_context: str) -> str:
        """Generate response using semantic chunking for large document sets"""
        # Create semantic groups
        document_groups = self.create_semantic_document_groups(documents)
        response_parts = []

        # Process all groups in parallel using async for faster response
        async def process_group_async(group_index: int, doc_group: List[Dict]) -> str:
            group_context = "\n\n".join([doc['content'] for doc in doc_group])

            group_prompt = f"""You are answering: {query}

This is document group {group_index} of {len(document_groups)}. Focus on these documents:

{group_context}

🚨 CRITICAL CONSTRAINT: ONLY use information explicitly stated in the provided documents above.

IMPORTANT: If these documents do not contain ANY information relevant to the query, simply return "SKIP" - do not create empty sections or explanatory text.

DO NOT use your general knowledge, training data, or external information beyond what is explicitly stated in these documents.

CONTENT REQUIREMENTS:
- ONLY include information directly found in the provided documents
- Copy EXACT formulas, calculations, and mathematical expressions from documents
- Preserve specific numerical values exactly as stated (90%, $2.50 per $1,000, etc.)
- Include calculation methods and procedures ONLY if present in documents
- If documents contain partial information, clearly state what is present vs. missing

FORMATTING REQUIREMENTS:
- Follow this EXACT structure pattern:

**1. Main Section Title**

Content paragraph with details.

**1.1 Subsection Title**

- Bullet point item
- Another bullet point
- Third bullet point

**2. Next Main Section**

More content here.

CRITICAL FORMATTING RULES:
- Main headings: **1. Title**, **2. Title**, **3. Title** (consistent numbering)
- Sub-headings: **1.1 Title**, **1.2 Title** (consistent sub-numbering)
- NEVER use random ** mid-sentence or inconsistent bold patterns
- NEVER create orphaned dashes like ":\n-\n" - always use complete bullet points
- NEVER end with random asterisks or incomplete formatting
- Always double line break between sections
- Write formulas clearly: use simple notation like (A + B)/C
- End every section with blank line for readability

Provide a detailed response covering ONLY information that relates to the question AND is present in these documents using proper markdown formatting."""

            # Calculate dynamic max_tokens for this specific group/query
            dynamic_max_tokens = self.calculate_dynamic_max_tokens(query, len(doc_group))

            # Use AsyncOpenAI for true parallel processing
            response = await self.async_client.chat.completions.create(
                model=self.actual_model,
                messages=[{"role": "user", "content": group_prompt}],
                temperature=0,
                max_tokens=dynamic_max_tokens
            )

            logger.info(f"⚡ Processed group {group_index}/{len(document_groups)} (async)")
            return response.choices[0].message.content.strip()

        # Process all groups concurrently using asyncio.gather for true parallelism
        logger.info(f"⚡ Starting parallel processing of {len(document_groups)} groups with AsyncOpenAI")
        response_parts = await asyncio.gather(*[
            process_group_async(i+1, doc_group)
            for i, doc_group in enumerate(document_groups)
        ])

        # Temporarily disable structured JSON approach - has parsing issues
        # TODO: Fix JSON parsing and markdown conversion in structured approach
        logger.info("📋 Using enhanced chunked response approach")

        # Fallback to existing chunked approach
        # Merge all parts
        merged_response = self.merge_response_parts(query, response_parts)

        # Apply basic normalization only (no heavy post-processing since structured failed)
        return self.formatting_manager.normalize_spacing(merged_response)

    def filter_relevant_documents(self, documents: List[Dict]) -> List[Dict]:
        """Pre-filter documents to only include those with substantial relevant content"""
        relevant_docs = []

        for doc in documents:
            content = doc.get('content', '')

            # Skip obviously empty or generic content
            if len(content.strip()) < self.min_content_length:
                continue

            # Skip generic assistant responses
            content_lower = content.lower()
            if any(phrase in content_lower for phrase in [
                'how can i assist you',
                'feel free to share',
                'what can i help you with',
                'i apologize but i don\'t have',
                'i don\'t have sufficient information'
            ]):
                continue

            # Include documents with good relevance scores
            relevance_score = doc.get('relevance_score', doc.get('score', 0.0))
            if relevance_score > self.relevance_threshold:
                relevant_docs.append(doc)

        logger.info(f"📋 Filtered to {len(relevant_docs)} relevant documents from {len(documents)} total")
        return relevant_docs

    def create_semantic_document_groups(self, documents: List[Dict]) -> List[List[Dict]]:
        """Smart grouping that only creates multiple groups when necessary and filters for relevant content"""
        try:
            # Pre-filter documents for relevance to avoid empty groups
            relevant_docs = self.filter_relevant_documents(documents)

            if len(relevant_docs) <= 3:
                # If we have very few relevant documents, just use single group
                logger.info(f"📚 Using single group for {len(relevant_docs)} relevant documents")
                return [relevant_docs] if relevant_docs else [documents[:5]]  # Fallback to top 5 if no relevant found

            # First check if documents are homogeneous
            doc_summaries = []
            docs_to_analyze = relevant_docs[:15]  # Limit for analysis efficiency
            for i, doc in enumerate(docs_to_analyze, 1):
                content_sample = doc['content'][:300].replace('\n', ' ')
                doc_summaries.append(f"Doc {i}: {content_sample}...")

            grouping_prompt = f"""Analyze these document excerpts and determine their diversity.

Documents:
{chr(10).join(doc_summaries)}

IMPORTANT: Only create multiple groups if documents have DISTINCTLY different themes.
If all documents are about the same topic or from the same source, respond with:
"Group 1: All documents (unified theme: [describe])"

Otherwise, create 2-3 groups ONLY if truly needed:
Group 1: Doc X, Doc Y (theme: [describe])
Group 2: Doc Z, Doc W (theme: [describe])

Be conservative - prefer fewer groups unless documents are clearly different.
Group documents:"""

            response = self.llm.complete(grouping_prompt)
            grouping_result = response.text.strip()

            logger.info(f"📊 SEMANTIC GROUPING RESULT:\n{grouping_result}")

            # Parse the grouping response to create actual groups
            groups = self.parse_document_groupings(grouping_result, docs_to_analyze)

            # Add remaining relevant documents to smallest groups
            if len(relevant_docs) > 15:
                remaining_docs = relevant_docs[15:]
                for doc in remaining_docs:
                    smallest_group = min(groups, key=len)
                    smallest_group.append(doc)

            logger.info(f"📚 Created {len(groups)} semantic document groups:")
            for i, group in enumerate(groups, 1):
                logger.info(f"  Group {i}: {len(group)} documents")
            return groups

        except Exception as e:
            logger.error(f"Failed to create semantic groups: {e}")
            # Safe fallback - single group (no hardcoded multiple groups)
            logger.info(f"📚 Fallback: using single group for {len(documents)} documents")
            return [documents]

    def parse_document_groupings(self, grouping_result: str, documents: List[Dict]) -> List[List[Dict]]:
        """Parse LLM grouping result and create actual document groups"""
        try:
            groups = []
            lines = grouping_result.split('\n')

            for line in lines:
                if 'Group' in line and ':' in line:
                    # Extract document numbers from the line
                    doc_numbers = re.findall(r'Doc (\d+)', line)
                    if doc_numbers:
                        group = []
                        for num_str in doc_numbers:
                            doc_index = int(num_str) - 1  # Convert to 0-based index
                            if 0 <= doc_index < len(documents):
                                group.append(documents[doc_index])
                        if group:
                            groups.append(group)

            # If parsing failed or we got too many small groups, just use one group
            if not groups or sum(len(g) for g in groups) < len(documents) * 0.7:
                logger.warning("Grouping parsing failed or insufficient coverage, using single group")
                # Just return all documents as a single group
                groups = [documents]

            return groups

        except Exception as e:
            logger.error(f"Failed to parse document groupings: {e}")
            # Fallback to single group for simplicity
            return [documents]

    def merge_response_parts(self, query: str, response_parts: List[str]) -> str:
        """Merge multiple response parts into a coherent final response"""
        try:
            # Filter out empty, very short parts, and SKIP responses
            valid_parts = []
            for part in response_parts:
                if part and len(part.strip()) > 10:
                    part_content = part.strip()
                    # Skip responses that are just "SKIP" or very short
                    if part_content.upper() != "SKIP" and len(part_content) > self.min_response_part_length:
                        valid_parts.append(part_content)

            if not valid_parts:
                return "I apologize, but I couldn't generate a comprehensive response based on the available documents."

            # If only one valid part, return it directly
            if len(valid_parts) == 1:
                return valid_parts[0].strip()

            # Combine multiple parts with proper sectioning
            merged_sections = []
            for i, part in enumerate(valid_parts, 1):
                # Clean up the part
                cleaned_part = part.strip()

                # Add section header if multiple parts
                if len(valid_parts) > 1:
                    section_header = f"## Section {i}\n\n"
                    merged_sections.append(section_header + cleaned_part)
                else:
                    merged_sections.append(cleaned_part)

            return "\n\n".join(merged_sections)

        except Exception as e:
            logger.error(f"Failed to merge response parts: {e}")
            # Fallback: just join the parts
            return "\n\n".join(response_parts)

    def assess_response_quality(self, response: str, query: str, documents: List[Dict]) -> Dict[str, float]:
        """Score response quality on multiple dimensions"""
        scores = {}

        # Completeness: Does response address all aspects of the query?
        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())
        coverage = len(query_terms & response_terms) / len(query_terms) if query_terms else 0
        scores['completeness'] = min(coverage * 1.5, 1.0)  # Scale up but cap at 1.0

        # Accuracy: Are facts from documents preserved?
        doc_numbers = set()
        response_numbers = set()
        for doc in documents[:3]:  # Check top 3 docs
            doc_numbers.update(re.findall(r'\d+(?:\.\d+)?%?', doc['content']))
        response_numbers.update(re.findall(r'\d+(?:\.\d+)?%?', response))

        if doc_numbers:
            accuracy = len(doc_numbers & response_numbers) / len(doc_numbers)
            scores['accuracy'] = min(accuracy * 2, 1.0)  # Numbers are important indicators
        else:
            scores['accuracy'] = 0.8  # Default if no numbers to compare

        # Formatting: Check structure quality
        has_sections = bool(re.search(r'^#{1,3}\s+\w+', response, re.MULTILINE))
        has_bullets = '•' in response or '-' in response
        has_proper_spacing = '\n\n' in response
        formatting_score = sum([has_sections * 0.4, has_bullets * 0.3, has_proper_spacing * 0.3])
        scores['formatting'] = formatting_score

        # Relevance: How well does response match query intent?
        # Simple keyword-based relevance
        important_terms = ['calculate', 'formula', 'reserve', 'premium', 'liability', 'gaap', 'ifrs']
        query_important = sum(1 for term in important_terms if term in query.lower())
        response_important = sum(1 for term in important_terms if term in response.lower())

        if query_important > 0:
            scores['relevance'] = min(response_important / query_important, 1.0)
        else:
            scores['relevance'] = 0.8  # Default relevance

        # Overall quality score
        scores['overall'] = (
            scores['completeness'] * 0.3 +
            scores['accuracy'] * 0.3 +
            scores['formatting'] * 0.2 +
            scores['relevance'] * 0.2
        )

        return scores

    async def regenerate_with_focus(self, response: str, quality_scores: Dict[str, float],
                                   query: str, documents: List[Dict]) -> str:
        """Regenerate response focusing on low-scoring aspects"""
        focus_areas = []

        if quality_scores['completeness'] < 0.7:
            focus_areas.append("Ensure all aspects of the query are addressed")
        if quality_scores['accuracy'] < 0.7:
            focus_areas.append("Include specific numbers and formulas from the documents")
        if quality_scores['formatting'] < 0.7:
            focus_areas.append("Use clear sections, bullet points, and proper spacing")
        if quality_scores['relevance'] < 0.7:
            focus_areas.append("Focus on the specific topic asked about")

        if not focus_areas:
            return response  # Quality is acceptable

        # Regenerate with specific focus
        context = "\n\n".join([doc['content'] for doc in documents[:5]])
        regeneration_prompt = f"""Improve this response to better answer the query.

Query: {query}

Current Response:
{response}

IMPROVEMENT FOCUS:
{chr(10).join(f"- {area}" for area in focus_areas)}

Documents for reference:
{context[:3000]}

Provide an improved response that addresses the focus areas:"""

        # Calculate dynamic max_tokens for regeneration
        dynamic_max_tokens = self.calculate_dynamic_max_tokens(query, len(documents))

        improved = await self.async_client.chat.completions.create(
            model=self.actual_model,
            messages=[{"role": "user", "content": regeneration_prompt}],
            temperature=0.3,
            max_tokens=dynamic_max_tokens
        )

        logger.info(f"✨ Response regenerated with focus on: {', '.join(focus_areas)}")
        return improved.choices[0].message.content.strip()

    async def generate_follow_up_questions(self, query: str, response: str, retrieved_docs: List[Dict]) -> List[str]:
        """Generate contextual follow-up questions based on the query and response"""

        # Determine appropriate question categories
        categories = self.query_analyzer.determine_question_categories(query, response, retrieved_docs)
        category_examples = self.get_category_examples(categories)

        # Analyze actual document content for specific follow-up opportunities
        content_insights = self.analyze_document_content_for_followups(retrieved_docs[:2])  # Analyze top 2 docs

        # Build rich context from document content analysis
        topic_context = ""
        if content_insights:
            context_parts = []
            for insight_type, items in content_insights.items():
                if items and insight_type != 'source_docs':
                    context_parts.append(f"{insight_type}: {', '.join(items[:3])}")  # Top 3 items per type

            if context_parts:
                topic_context = f"Document content analysis:\n" + "\n".join(context_parts)

            # Add source document info
            source_docs = content_insights.get('source_docs', [])
            if source_docs:
                topic_context += f"\nSource documents: {', '.join(source_docs[:2])}"

        # Build category guidance
        category_guidance = ""
        for cat, examples in category_examples.items():
            category_guidance += f"\n{cat.title()}: {', '.join(examples[:2])}"

        # Create content-specific guidance for better follow-ups
        content_guidance = ""
        if content_insights:
            guidance_parts = []

            if content_insights.get('standards_mentioned'):
                guidance_parts.append(f"Consider asking about other standards: {', '.join(content_insights['standards_mentioned'][:3])}")

            if content_insights.get('examples_found'):
                guidance_parts.append("Ask about specific examples or scenarios mentioned in the documents")

            if content_insights.get('tables_data'):
                guidance_parts.append(f"Reference specific data: {', '.join(content_insights['tables_data'][:2])}")

            if content_insights.get('key_concepts'):
                guidance_parts.append(f"Explore concepts like: {', '.join(content_insights['key_concepts'][:3])}")

            if content_insights.get('implementation_terms'):
                guidance_parts.append("Ask about implementation, transition, or adoption aspects")

            if guidance_parts:
                content_guidance = f"\nContent-specific opportunities:\n" + "\n".join([f"- {part}" for part in guidance_parts])

        # Extract specific elements from the actual response to create targeted follow-ups
        response_elements = self.extract_response_elements(response)
        document_specifics = self.get_document_specifics(retrieved_docs[:2])

        prompt = f"""Generate 2-3 highly specific follow-up questions based on this EXACT conversation and documents.

USER ASKED: "{query}"

MY RESPONSE: "{response[:500]}..."

SPECIFIC DOCUMENT CONTENT USED:
{document_specifics}

RESPONSE ANALYSIS:
{response_elements}

CRITICAL INSTRUCTIONS:
- Questions must be DIRECTLY related to what I just explained to the user
- Reference SPECIFIC information from the documents that were actually cited
- Build upon the EXACT conversation context and dig deeper into specifics
- NO generic business/insurance questions
- Focus on specific metrics, segments, time periods, or data points I mentioned
- If discussing financial results, ask about specific components or related metrics
- If discussing business segments, ask about other segments or comparative performance
- If discussing time periods, ask about trends or comparisons to other periods

Examples of GOOD contextual questions for financial discussions:
- "What drove the unfavorable claims experience in U.S. Traditional that you mentioned?"
- "How did the other business segments perform compared to the 14.3% ROE you cited?"
- "What specific factors contributed to the $276 million capital deployment figure?"
- "Can you break down the components of the variable investment income mentioned?"

Examples of GOOD contextual questions for technical documents:
- "What does the PWC document say about the implementation timeline for this standard?"
- "How does the calculation method in section 3.2 apply to different scenarios?"
- "What are the disclosure requirements mentioned alongside this guidance?"

Examples of BAD generic questions to COMPLETELY AVOID:
- "How do claims impact profitability?"
- "What strategies do insurers use for capital management?"
- "Can you explain adjusted operating income?"
- "What are the benefits of this approach?"

Generate exactly 2-3 contextual follow-up questions that dig deeper into the specific information I just provided:"""

        try:
            response_obj = self.llm.complete(prompt)
            questions_text = response_obj.text.strip()

            # Parse the response into individual questions
            questions = []
            logger.info(f"🎯 AI generated follow-up questions: {questions_text}")
            for line in questions_text.split('\n'):
                line = line.strip()
                if line and len(line) > 10:  # Filter out empty or very short lines
                    # Clean up any unwanted formatting - remove numbers, bullets, quotes
                    clean_question = line.strip('- •').strip()
                    # Remove numbering like "1. " or "2. "
                    clean_question = re.sub(r'^\d+\.\s*', '', clean_question)
                    # Remove surrounding quotes
                    clean_question = clean_question.strip('"\'').strip()

                    if clean_question.endswith('?') and len(clean_question) > 10:
                        # Validate question is contextual (not generic)
                        is_contextual = self.query_analyzer.is_contextual_question(clean_question, query, response)
                        logger.info(f"🔍 Question validation: '{clean_question}' -> {'✅' if is_contextual else '❌'}")
                        if is_contextual:
                            questions.append(clean_question)

            # Return max 3 questions, fallback if none are contextual
            if questions:
                return questions[:3]
            else:
                logger.warning("No contextual follow-up questions generated, using response-based fallback")
                return self.generate_response_based_fallback(query, response, retrieved_docs)

        except Exception as e:
            logger.error("Failed to generate follow-up questions", error=str(e))
            # Return fallback questions if generation fails
            return [
                "Can you explain this in more detail?",
                "What are the practical implications?",
                "How does this apply in practice?"
            ]

    def analyze_document_content_for_followups(self, retrieved_docs: List[Dict]) -> Dict[str, List[str]]:
        """Analyze document content to extract specific elements for targeted follow-up questions"""
        insights = {
            'standards_mentioned': [],
            'examples_found': [],
            'tables_data': [],
            'implementation_terms': [],
            'key_concepts': [],
            'cross_references': [],
            'source_docs': []
        }

        try:
            for doc in retrieved_docs:
                content = doc.get('content', '').lower()
                filename = doc.get('metadata', {}).get('title', doc.get('metadata', {}).get('filename', 'Unknown'))
                insights['source_docs'].append(filename)

                # Extract accounting standards (ASC, IFRS, etc.)
                standards = re.findall(r'\b(?:asc|ifrs|ias|fas)\s+\d+(?:[-\.\s]\d+)*\b', content)
                insights['standards_mentioned'].extend([std.upper() for std in standards])

                # Find examples in the document
                examples = re.findall(r'example\s+\d+[:\s][^\.]*\.', content)
                insights['examples_found'].extend([ex.strip()[:50] + "..." for ex in examples[:2]])

                # Detect tables and data references
                table_refs = re.findall(r'table\s+\d+|schedule\s+\d+|appendix\s+[a-z]', content)
                insights['tables_data'].extend([ref.title() for ref in table_refs])

                # Find implementation-related terms
                impl_patterns = [
                    r'transition\s+requirements?',
                    r'implementation\s+guidance',
                    r'effective\s+date',
                    r'adoption\s+process',
                    r'system\s+changes?'
                ]
                for pattern in impl_patterns:
                    matches = re.findall(pattern, content)
                    insights['implementation_terms'].extend([match.title() for match in matches])

                # Extract key financial/actuarial concepts
                concept_patterns = [
                    r'present\s+value',
                    r'discount\s+rate',
                    r'fair\s+value',
                    r'amortization',
                    r'reserve\s+adequacy',
                    r'capital\s+ratio',
                    r'risk\s+adjustment'
                ]
                for pattern in concept_patterns:
                    matches = re.findall(pattern, content)
                    insights['key_concepts'].extend([match.title() for match in matches])

                # Find cross-references to other standards/sections
                cross_refs = re.findall(r'see\s+(?:also\s+)?(?:asc|ifrs|section|paragraph)\s+\d+(?:[-\.\s]\d+)*', content)
                insights['cross_references'].extend([ref.upper() for ref in cross_refs])

            # Clean up and deduplicate
            for key in insights:
                if key != 'source_docs':
                    insights[key] = list(set(insights[key]))[:5]  # Max 5 unique items per category

            logger.info(f"📊 Content analysis extracted: {sum(len(v) if isinstance(v, list) else 0 for v in insights.values())} content elements")
            return insights

        except Exception as e:
            logger.error(f"Failed to analyze document content: {e}")
            return {'source_docs': [doc.get('metadata', {}).get('title', doc.get('metadata', {}).get('filename', 'Unknown')) for doc in retrieved_docs]}

    def extract_response_elements(self, response: str) -> str:
        """Extract specific elements mentioned in the response for targeted follow-ups"""
        elements = []
        response_lower = response.lower()

        try:
            # Extract specific financial metrics and dollar amounts
            financial_metrics = []
            dollar_amounts = re.findall(r'\$[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand)?', response_lower)
            percentages = re.findall(r'\d+\.\d+%', response_lower)

            if dollar_amounts:
                elements.append(f"Financial amounts: {', '.join(dollar_amounts[:3])}")
            if percentages:
                elements.append(f"Performance metrics: {', '.join(percentages[:3])}")

            # Extract specific business segments mentioned
            segments = re.findall(r'(?:u\.s\.|us|canada|emea|latin america)\s+(?:traditional|group|individual life|financial solutions)', response_lower)
            if segments:
                elements.append(f"Business segments: {', '.join(set([s.title() for s in segments[:3]]))}")

            # Extract time periods and quarters
            periods = re.findall(r'q[1-4]\s+(?:20\d{2}|results?)|(?:second|first|third|fourth)\s+quarter', response_lower)
            if periods:
                elements.append(f"Time periods discussed: {', '.join(set([p.upper() for p in periods]))}")

            # Extract company/entity names
            companies = re.findall(r'\b(?:rga|reinsurance group|equitable holdings?)\b', response_lower)
            if companies:
                elements.append(f"Companies mentioned: {', '.join(set([c.upper() for c in companies]))}")

            # Extract performance trends and outcomes
            performance_terms = []
            if 'favorable' in response_lower:
                favorable_items = re.findall(r'favorable\s+(?:\w+\s+){0,2}(?:experience|performance|results?|investment)', response_lower)
                performance_terms.extend(favorable_items)
            if 'unfavorable' in response_lower:
                unfavorable_items = re.findall(r'unfavorable\s+(?:\w+\s+){0,2}(?:experience|claims?|results?)', response_lower)
                performance_terms.extend(unfavorable_items)

            if performance_terms:
                elements.append(f"Performance trends: {', '.join(set(performance_terms[:3]))}")

            # Extract specific standards mentioned
            standards_mentioned = re.findall(r'\b(?:asc|ifrs|ias|fas)\s+\d+(?:[-\.\s]\d+)*\b', response_lower)
            if standards_mentioned:
                elements.append(f"Standards referenced: {', '.join(set([s.upper() for s in standards_mentioned]))}")

            # Extract key financial concepts
            financial_concepts = []
            concept_patterns = [
                r'adjusted\s+operating\s+income',
                r'return\s+on\s+equity',
                r'excess\s+capital',
                r'variable\s+investment\s+income',
                r'claims?\s+experience',
                r'premium\s+growth',
                r'reserve\s+adequacy',
                r'capital\s+deployment'
            ]
            for pattern in concept_patterns:
                matches = re.findall(pattern, response_lower)
                financial_concepts.extend([match.title() for match in matches])

            if financial_concepts:
                elements.append(f"Key concepts: {', '.join(set(financial_concepts[:3]))}")

            return "\n".join([f"- {element}" for element in elements]) if elements else "- General financial/business information provided"

        except Exception as e:
            logger.error(f"Failed to extract response elements: {e}")
            return "- Unable to analyze response elements"

    def get_document_specifics(self, retrieved_docs: List[Dict]) -> str:
        """Extract specific information from the documents that were actually used"""
        specifics = []

        try:
            for i, doc in enumerate(retrieved_docs[:2], 1):
                filename = doc.get('metadata', {}).get('title', doc.get('metadata', {}).get('filename', 'Unknown'))
                content = doc.get('content', '')[:400]  # First 400 chars

                specifics.append(f"Source ({filename}):")
                specifics.append(f"Content excerpt: \"{content}...\"")

                # Extract specific elements from this document
                content_lower = content.lower()
                doc_elements = []

                # Standards in this specific document
                doc_standards = re.findall(r'\b(?:asc|ifrs|ias|fas)\s+\d+(?:[-\.\s]\d+)*\b', content_lower)
                if doc_standards:
                    doc_elements.append(f"Standards: {', '.join(set([s.upper() for s in doc_standards]))}")

                # Tables or data references
                tables = re.findall(r'table\s+\d+|schedule\s+\d+|appendix\s+[a-z]', content_lower)
                if tables:
                    doc_elements.append(f"Data references: {', '.join(set(tables))}")

                if doc_elements:
                    specifics.append(f"Specific elements: {'; '.join(doc_elements)}")

                specifics.append("")  # Add blank line between documents

            return "\n".join(specifics) if specifics else "No specific document content available"

        except Exception as e:
            logger.error(f"Failed to get document specifics: {e}")
            return "Unable to analyze document specifics"

    def get_category_examples(self, categories: List[str]) -> Dict[str, List[str]]:
        """Get example questions for each category"""
        category_questions = {
            'clarification': [
                "Can you explain this in simpler terms?",
                "What does this mean in practice?",
                "Could you break this down further?"
            ],
            'examples': [
                "Can you provide a real-world example?",
                "How would this work for a life insurance company?",
                "What would this look like in financial statements?"
            ],
            'comparison': [
                "How does this differ under IFRS vs US GAAP?",
                "What are the key differences from previous standards?",
                "How does this compare to industry practice?"
            ],
            'technical': [
                "What are the detailed calculation steps?",
                "What assumptions are typically used?",
                "How do you handle edge cases?"
            ],
            'application': [
                "How do companies typically implement this?",
                "What systems support this process?",
                "How often should this be performed?"
            ],
            'compliance': [
                "What are the audit requirements?",
                "How do regulators typically examine this?",
                "What documentation is needed?"
            ]
        }

        return {cat: category_questions.get(cat, []) for cat in categories}

    def generate_response_based_fallback(self, query: str, response: str, retrieved_docs: List[Dict]) -> List[str]:
        """Generate simple, contextual follow-ups based on the actual response content"""
        fallback_questions = []

        try:
            # Extract key terms from the response to create specific follow-ups
            response_lower = response.lower()

            # If response mentions specific standards, ask about related ones
            standards = re.findall(r'\b(?:asc|ifrs|ias|fas)\s+\d+(?:[-\.\s]\d+)*\b', response_lower)
            if standards:
                fallback_questions.append(f"How does {standards[0].upper()} relate to other accounting standards?")

            # If response mentions calculations, ask for details
            if any(word in response_lower for word in ['calculate', 'computation', 'formula']):
                fallback_questions.append("Can you walk through the calculation steps in more detail?")

            # If response mentions examples, ask for more
            if 'example' in response_lower:
                fallback_questions.append("Can you provide another example of this concept?")

            # If response mentions implementation, ask about challenges
            if any(word in response_lower for word in ['implement', 'apply', 'adopt']):
                fallback_questions.append("What are the main challenges in implementing this?")

            # Document-specific fallback
            if retrieved_docs:
                filename = retrieved_docs[0].get('metadata', {}).get('filename', '')
                if filename:
                    fallback_questions.append(f"What else does the {filename} document cover on this topic?")

            # If no specific fallbacks, use minimal contextual ones
            if not fallback_questions:
                fallback_questions = [
                    "Can you clarify any part of this explanation?",
                    "Are there related concepts I should understand?",
                    "How would this apply in practice?"
                ]

            return fallback_questions[:3]

        except Exception as e:
            logger.error(f"Failed to generate fallback questions: {e}")
            return [
                "Can you elaborate on this topic?",
                "What are the key takeaways?",
                "How does this relate to our previous discussion?"
            ]


def create_response_generator(llm_client=None, async_client=None, memory_manager=None,
                            formatting_manager=None, query_analyzer=None, config=None):
    """Factory function to create a ResponseGenerator instance"""
    return ResponseGenerator(
        llm_client=llm_client,
        async_client=async_client,
        memory_manager=memory_manager,
        formatting_manager=formatting_manager,
        query_analyzer=query_analyzer,
        config=config
    )