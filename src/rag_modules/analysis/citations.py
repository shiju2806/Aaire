"""
Citation Analysis Module
Handles document citation extraction and analysis for RAG responses
"""

from typing import List, Dict, Any, Tuple, Optional
import re
import structlog
from difflib import SequenceMatcher
import asyncio
from llama_index.llms.openai import OpenAI

logger = structlog.get_logger()


class CitationAnalyzer:
    """
    Analyzes retrieved documents to determine citation information
    and assess document usage in generated responses.
    """

    async def analyze_document_usage_with_llm(self, retrieved_docs: List[Dict], response: str, query: str) -> List[Dict]:
        """
        Use LLM to analyze which retrieved documents were actually used to generate the response.
        Returns list of documents identified as actually used by the LLM.
        """
        if not retrieved_docs or not response:
            return []

        logger.info(f"🤖 LLM-BASED CITATION ANALYSIS: Analyzing {len(retrieved_docs)} documents against response")

        # Prepare document summaries for LLM analysis
        doc_summaries = []
        for i, doc in enumerate(retrieved_docs[:10]):  # Limit to top 10 for LLM analysis
            metadata = doc.get('metadata', {})
            filename = (
                # Standard metadata fields
                metadata.get('title') or
                metadata.get('filename') or
                metadata.get('source_document') or
                metadata.get('file_name') or
                metadata.get('source') or
                metadata.get('document_name') or
                metadata.get('name') or
                # Check document root level for source info
                doc.get('filename') or
                doc.get('title') or
                doc.get('source') or
                doc.get('document_name') or
                # Fallback with index
                f'Document_{i+1}'
            )

            content_preview = doc.get('content', '')[:500]  # First 500 chars
            doc_summaries.append({
                'index': i + 1,
                'filename': filename,
                'content_preview': content_preview,
                'full_doc': doc
            })

        # Create LLM prompt for usage analysis
        prompt = f"""
You are analyzing which documents were actually used to generate a response to a user query.

USER QUERY: {query}

GENERATED RESPONSE:
{response}

AVAILABLE DOCUMENTS:
"""

        for doc_summary in doc_summaries:
            prompt += f"""
Document {doc_summary['index']}: {doc_summary['filename']}
Content preview: {doc_summary['content_preview']}

"""

        prompt += """
ANALYSIS TASK:
Please analyze the generated response and determine which documents were actually used to create the content. Look for:
1. Direct content matches or paraphrases
2. Specific information that could only come from certain documents
3. Concepts, terms, or data points mentioned in the response

IMPORTANT: Only identify documents that were clearly used. If the response contains general information that doesn't clearly come from the documents, or if the response states insufficient information, return fewer or no documents.

RESPOND WITH ONLY THE DOCUMENT NUMBERS (comma-separated), or 'NONE' if no documents were clearly used.
Example responses:
- "1,3,5" (if documents 1, 3, and 5 were used)
- "2" (if only document 2 was used)
- "NONE" (if no documents were clearly used)

Your response:"""

        try:
            # Use LLM to analyze document usage
            llm = OpenAI(model="gpt-4o-mini", temperature=0)
            llm_response = await llm.acomplete(prompt)
            usage_analysis = llm_response.text.strip().upper()

            logger.info(f"🤖 LLM Usage Analysis Result: '{usage_analysis}'")

            # Parse LLM response
            used_docs = []
            if usage_analysis != 'NONE' and usage_analysis:
                try:
                    # Extract document indices
                    if ',' in usage_analysis:
                        doc_indices = [int(idx.strip()) for idx in usage_analysis.split(',') if idx.strip().isdigit()]
                    else:
                        doc_indices = [int(usage_analysis)] if usage_analysis.isdigit() else []

                    # Get corresponding documents
                    for idx in doc_indices:
                        if 1 <= idx <= len(doc_summaries):
                            used_docs.append(doc_summaries[idx - 1]['full_doc'])
                            logger.info(f"✅ LLM identified document {idx} ({doc_summaries[idx - 1]['filename']}) as used")

                except (ValueError, IndexError) as e:
                    logger.warning(f"⚠️ Error parsing LLM response '{usage_analysis}': {e}")

            if not used_docs:
                logger.info(f"🤖 LLM Analysis: No documents identified as clearly used")
            else:
                logger.info(f"🤖 LLM Analysis: {len(used_docs)} documents identified as used")

            return used_docs

        except Exception as e:
            logger.error(f"❌ Error in LLM usage analysis: {e}")
            # Fallback to top document if LLM analysis fails
            if retrieved_docs:
                logger.info(f"🔄 Falling back to top retrieved document")
                return [retrieved_docs[0]]
            return []

    def extract_citations(self, retrieved_docs: List[Dict], query: str = "", response: str = "") -> List[Dict[str, Any]]:
        """Extract citation information - analyze response to determine which documents were actually used"""
        citations = []

        if not retrieved_docs:
            logger.warning("❌ NO CITATIONS GENERATED - no retrieved documents")
            return citations

        if not response:
            logger.warning("❌ NO RESPONSE PROVIDED - falling back to top document citation")
            # Fallback: cite the most relevant document if no response analysis possible
            top_doc = retrieved_docs[0]
            # Enhanced filename extraction with better fallback handling
            metadata = top_doc.get('metadata', {})
            filename = (
                # Standard metadata fields
                metadata.get('title') or
                metadata.get('filename') or
                metadata.get('source_document') or
                metadata.get('file_name') or
                metadata.get('source') or
                metadata.get('document_name') or
                metadata.get('name') or
                # Check document root level for source info
                top_doc.get('filename') or
                top_doc.get('title') or
                top_doc.get('source') or
                top_doc.get('document_name') or
                # Check if there's a source_type that gives hints
                (f"Document ({metadata.get('source_type', 'unknown')} source)" if metadata.get('source_type') and metadata.get('source_type') != 'unknown' else None) or
                # Last resort
                'Unknown'
            )

            # Clean up filename if it contains file extension or path
            if filename != 'Unknown':
                # Remove file extension if present
                if '.' in filename:
                    name_parts = filename.rsplit('.', 1)
                    if len(name_parts[1]) <= 4:  # Likely a file extension
                        filename = name_parts[0]
                # Remove path if present
                if '/' in filename:
                    filename = filename.split('/')[-1]
                if '\\' in filename:
                    filename = filename.split('\\')[-1]

            logger.info(f"📄 Fallback citation using document: {filename}")

            citations.append({
                "id": 1,
                "text": top_doc['content'][:200] + "..." if len(top_doc['content']) > 200 else top_doc['content'],
                "source": filename,
                "source_type": top_doc.get('source_type', 'unknown'),
                "confidence": round(top_doc.get('relevance_score', top_doc.get('score', 0.0)), 3)
            })
            return citations

        logger.info(f"🎯 LLM-BASED CITATION ANALYSIS: Analyzing response against {len(retrieved_docs)} documents")

        # Use LLM-based document usage analysis
        try:
            # Run async LLM analysis - handle event loop properly
            import asyncio
            try:
                # Check if we're in an existing event loop
                loop = asyncio.get_running_loop()
                # If we're in an event loop, we need to use a different approach
                # Convert async to sync by running in thread pool
                import concurrent.futures
                import threading

                def run_async_in_thread():
                    # Create new event loop in thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            self.analyze_document_usage_with_llm(retrieved_docs, response, query)
                        )
                    finally:
                        new_loop.close()

                # Run in thread pool to avoid event loop conflict
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async_in_thread)
                    used_doc_list = future.result(timeout=30)  # 30 second timeout

            except RuntimeError:
                # No running loop, safe to use asyncio.run()
                used_doc_list = asyncio.run(self.analyze_document_usage_with_llm(retrieved_docs, response, query))

            # Convert to (doc, score) format for compatibility
            used_docs = []
            for doc in used_doc_list:
                relevance_score = doc.get('relevance_score', doc.get('score', 0.8))  # Default decent score for LLM-identified docs
                used_docs.append((doc, relevance_score))

            logger.info(f"📋 LLM-based citations: {len(used_docs)} documents identified as actually used")

        except Exception as e:
            logger.error(f"❌ Error in LLM citation analysis: {e}")
            # No fallback system - if LLM analysis fails, return no citations
            logger.warning(f"🚫 No fallback system - returning 0 citations due to LLM analysis failure")
            used_docs = []

        for i, (doc, relevance_score) in enumerate(used_docs):
            relevance_score = doc.get('relevance_score', doc.get('score', 0.0))

            # Enhanced filename extraction with better fallback handling
            metadata = doc.get('metadata', {})

            # CRITICAL FIX: Filter out documents with corrupted metadata that would create generic citations
            # Skip documents that would result in "Document (X source)" citations
            title = metadata.get('title')
            filename = metadata.get('filename')

            # Check if document has corrupted metadata (literal string 'None' or None values)
            if (title is None or title == 'None' or title == '') and \
               (filename is None or filename == 'None' or filename == ''):
                logger.info(f"❌ SKIPPING document with corrupted metadata - would create generic citation")
                continue

            # Try multiple metadata extraction approaches
            extracted_filename = (
                # Standard metadata fields
                title or
                filename or
                metadata.get('source_document') or
                metadata.get('file_name') or
                metadata.get('source') or
                metadata.get('document_name') or
                metadata.get('name') or
                # Check document root level for source info
                doc.get('filename') or
                doc.get('title') or
                doc.get('source') or
                doc.get('document_name') or
                # Last resort - but this should now be rare due to filtering above
                'Unknown'
            )

            # Clean up filename if it contains file extension or path
            if filename != 'Unknown':
                # Remove file extension if present
                if '.' in filename:
                    name_parts = filename.rsplit('.', 1)
                    if len(name_parts[1]) <= 4:  # Likely a file extension
                        filename = name_parts[0]
                # Remove path if present
                if '/' in filename:
                    filename = filename.split('/')[-1]
                if '\\' in filename:
                    filename = filename.split('\\')[-1]

            # Additional debugging for first citation
            if i == 0:
                logger.info(f"🔍 CITATION DEBUG - Main extraction first doc metadata: {list(metadata.keys())}")
                logger.info(f"🔍 CITATION DEBUG - Main extraction metadata values: title='{metadata.get('title')}', filename='{metadata.get('filename')}'")

            logger.info(f"📄 Processing citation {i+1}: {filename}, relevance_score={relevance_score:.3f}")

            # Simple quality filter - only skip obviously bad documents
            if relevance_score < 0.1:  # Very permissive threshold
                logger.info(f"❌ SKIPPING - Extremely low relevance: {relevance_score:.3f}")
                continue

            # Skip obvious generic responses only
            content_lower = doc.get('content', '').lower()
            if any(phrase in content_lower for phrase in [
                'how can i assist you today',
                'feel free to share',
                'what can i help you with'
            ]):
                logger.info(f"❌ SKIPPING - Generic assistant response")
                continue

            # Extract page information if available (skip Page 0)
            page_info = ""
            page_num = None

            if 'page' in doc['metadata']:
                page_num = doc['metadata']['page']
            elif 'page_label' in doc['metadata']:
                page_num = doc['metadata']['page_label']
            elif hasattr(doc, 'node_id') and 'page_' in str(doc.get('node_id', '')):
                # Extract page from node_id like "page_1_chunk_2"
                try:
                    page_num = str(doc.get('node_id', '')).split('page_')[1].split('_')[0]
                except:
                    pass

            # Only add page info if it's not 0 or None
            if page_num is not None and str(page_num) != '0':
                page_info = f", Page {page_num}"

            # Check if content contains page references from shape-aware extraction
            content = doc.get('content', '')
            if 'Source: Page' in content and page_num is None:
                # Extract page number from content like "Source: Page 2, cluster_1_page_2"
                page_match = re.search(r'Source: Page (\d+)', content)
                if page_match:
                    extracted_page = page_match.group(1)
                    # Only add if not page 0
                    if extracted_page != '0':
                        page_info = f", Page {extracted_page}"

            citation = {
                "id": len(citations) + 1,  # Use actual citation count, not doc index
                "text": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                "source": f"{filename}{page_info}",
                "source_type": doc['source_type'],
                "confidence": round(relevance_score, 3)  # Use relevance score instead of original score
            }

            # Add additional metadata if available
            if 'page' in doc['metadata']:
                citation['page'] = doc['metadata']['page']
            if 'section' in doc['metadata']:
                citation['section'] = doc['metadata']['section']
            if 'standard' in doc['metadata']:
                citation['standard'] = doc['metadata']['standard']

            citations.append(citation)
            logger.info(f"✅ ADDED citation from: {filename} (relevance: {relevance_score:.3f})")

        logger.info(f"🎯 FINAL RESULT: Generated {len(citations)} citations from {len(retrieved_docs)} retrieved documents")

        # DEBUG: Print citation details for troubleshooting
        if citations:
            for i, citation in enumerate(citations):
                logger.info(f"Citation {i+1}: source={citation.get('source')}, confidence={citation.get('confidence')}")
        else:
            logger.warning("❌ NO CITATIONS GENERATED - this explains missing citation display")

        return citations

    def analyze_document_usage_in_response(self, retrieved_docs: List[Dict], response: str, query: str) -> List[Tuple[Dict, float]]:
        """
        Analyze which documents were actually used in generating the response.
        Returns list of (document, usage_score) tuples, sorted by usage likelihood.
        """
        doc_usage_scores = []
        response_lower = response.lower()
        query_lower = query.lower()

        logger.info(f"🔍 Analyzing response usage for {len(retrieved_docs)} documents")

        for i, doc in enumerate(retrieved_docs):
            content = doc.get('content', '')
            content_lower = content.lower()

            # Enhanced filename extraction with logging
            metadata = doc.get('metadata', {})
            filename = (
                # Standard metadata fields
                metadata.get('title') or
                metadata.get('filename') or
                metadata.get('source_document') or
                metadata.get('file_name') or
                metadata.get('source') or
                metadata.get('document_name') or
                metadata.get('name') or
                # Check document root level for source info
                doc.get('filename') or
                doc.get('title') or
                doc.get('source') or
                doc.get('document_name') or
                # Check if there's a source_type that gives hints
                (f"Document ({metadata.get('source_type', 'unknown')} source)" if metadata.get('source_type') and metadata.get('source_type') != 'unknown' else None) or
                # Last resort
                'Unknown'
            )

            # Log metadata for first document to debug (use info level to ensure visibility)
            if i == 0:
                logger.info(f"🔍 CITATION DEBUG - First document metadata keys: {list(metadata.keys())}")
                logger.info(f"🔍 CITATION DEBUG - Metadata values: title='{metadata.get('title')}', filename='{metadata.get('filename')}', source_document='{metadata.get('source_document')}'")

            usage_score = 0.0
            evidence = []

            # 1. Direct content overlap - look for exact phrase matches
            # Split into meaningful phrases (3+ words)
            content_phrases = re.findall(r'\b\w+(?:\s+\w+){2,}\b', content_lower)
            response_phrases = re.findall(r'\b\w+(?:\s+\w+){2,}\b', response_lower)

            phrase_matches = 0
            for phrase in content_phrases:
                if len(phrase) > 15 and phrase in response_lower:  # Only significant phrases
                    phrase_matches += 1
                    usage_score += 0.3
                    evidence.append(f"phrase_match: {phrase[:50]}...")

            # 2. Key terminology overlap - insurance/accounting specific terms
            key_terms_doc = re.findall(r'\b(?:reserve|liability|premium|policy|actuarial|gaap|ifrs|asc|fas|ssap|vnpr|dac|ual)\b', content_lower)
            key_terms_response = re.findall(r'\b(?:reserve|liability|premium|policy|actuarial|gaap|ifrs|asc|fas|ssap|vnpr|dac|ual)\b', response_lower)

            term_overlap = len(set(key_terms_doc) & set(key_terms_response))
            if term_overlap > 0:
                usage_score += term_overlap * 0.2
                evidence.append(f"key_terms: {term_overlap} matches")

            # 3. Numerical values overlap - formulas, percentages, amounts
            doc_numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', content_lower)
            response_numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', response_lower)

            number_overlap = len(set(doc_numbers) & set(response_numbers))
            if number_overlap > 0:
                usage_score += number_overlap * 0.15
                evidence.append(f"numbers: {number_overlap} matches")

            # 4. Structural similarity - similar section headers or organization
            doc_headers = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}:', content)
            response_headers = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}:', response)

            header_overlap = len(set(doc_headers) & set(response_headers))
            if header_overlap > 0:
                usage_score += header_overlap * 0.1
                evidence.append(f"headers: {header_overlap} matches")

            # 5. Query-specific content relevance
            # If the response addresses the specific query, and this doc contains query-relevant content
            query_words = set(query_lower.split())
            doc_query_relevance = len([word for word in query_words if word in content_lower and len(word) > 3])
            response_query_relevance = len([word for word in query_words if word in response_lower and len(word) > 3])

            if doc_query_relevance > 0 and response_query_relevance > 0:
                relevance_score = min(doc_query_relevance, response_query_relevance) * 0.1
                usage_score += relevance_score
                evidence.append(f"query_relevance: {relevance_score:.2f}")

            # 6. Overall content similarity using sequence matching
            # Use smaller chunks to avoid memory issues with large documents
            doc_chunk = content_lower[:1000]  # First 1000 chars
            response_chunk = response_lower[:1000]

            if doc_chunk and response_chunk:
                similarity = SequenceMatcher(None, doc_chunk, response_chunk).ratio()
                if similarity > 0.1:  # Only if some meaningful similarity
                    usage_score += similarity * 0.2
                    evidence.append(f"similarity: {similarity:.3f}")

            logger.info(f"📊 Document {i+1} ({filename}): usage_score={usage_score:.3f}, evidence={evidence}")

            # Only include documents with meaningful usage evidence
            if usage_score > 0.1:  # Threshold for inclusion
                doc_usage_scores.append((doc, usage_score))

        # Sort by usage score (highest first) and return top document(s)
        doc_usage_scores.sort(key=lambda x: x[1], reverse=True)

        # Use configurable threshold for citation inclusion
        # Make threshold configurable - default 0.1 for permissive citation inclusion
        citation_threshold = 0.1
        max_citations = 5

        # Return top documents based on retrieval relevance and usage scores
        significant_docs = [doc for doc in doc_usage_scores if doc[1] > citation_threshold][:max_citations]

        # If no significant documents found, use retrieval-based fallback
        if not significant_docs and retrieved_docs:
            # Trust the vector search - if documents were retrieved, they have some relevance
            top_doc = retrieved_docs[0]
            retrieval_score = top_doc.get('relevance_score', top_doc.get('score', 0.0))
            if retrieval_score > 0.1:  # Basic retrieval relevance check
                return [(top_doc, retrieval_score)]
            else:
                logger.info(f"❌ No sufficiently relevant documents for citation")
                return []

        return significant_docs


    def infer_document_domain(self, filename: str, content: str) -> str:
        """Dynamically infer document domain from filename and content"""
        filename_lower = filename.lower()
        content_lower = content.lower()

        # Insurance/Regulatory domain
        if any(term in filename_lower for term in ['licat', 'insurance', 'regulatory', 'capital']):
            return 'insurance'

        # Accounting standards domain
        if any(term in filename_lower for term in ['pwc', 'asc', 'ifrs', 'gaap']) or \
           any(term in content_lower for term in ['asc ', 'ifrs', 'accounting standard']):
            return 'accounting_standards'

        # Foreign currency domain
        if any(term in filename_lower for term in ['foreign', 'currency', 'fx']) or \
           any(term in content_lower for term in ['foreign currency', 'exchange rate']):
            return 'foreign_currency'

        # Actuarial domain
        if any(term in filename_lower for term in ['actuarial', 'valuation', 'reserves']) or \
           any(term in content_lower for term in ['actuarial', 'present value', 'discount rate']):
            return 'actuarial'

        return 'general'

    def check_domain_compatibility(self, query_domain: str, doc_domain: str, doc_filename: str) -> Dict[str, Any]:
        """Check if query domain is compatible with document domain"""

        # Define domain compatibility matrix - make more permissive for debugging
        compatibility_matrix = {
            'accounting_standards': ['accounting_standards', 'foreign_currency', 'general', 'accounting'],
            'foreign_currency': ['foreign_currency', 'accounting_standards', 'general', 'accounting'],
            'insurance': ['insurance', 'actuarial', 'general'],
            'actuarial': ['actuarial', 'insurance', 'general'],
            'accounting': ['accounting', 'accounting_standards', 'foreign_currency', 'general'],
            'general': ['general', 'accounting_standards', 'foreign_currency', 'insurance', 'actuarial', 'accounting'],
            None: ['general', 'accounting_standards', 'foreign_currency', 'insurance', 'actuarial', 'accounting']  # Handle None domain
        }

        # Get compatible domains for query
        compatible_domains = compatibility_matrix.get(query_domain, ['general'])

        # Check compatibility
        if doc_domain in compatible_domains:
            return {
                'compatible': True,
                'reason': f"Query domain '{query_domain}' compatible with document domain '{doc_domain}'"
            }
        else:
            return {
                'compatible': False,
                'reason': f"Query domain '{query_domain}' incompatible with document domain '{doc_domain}' (file: {doc_filename})"
            }

    def is_insufficient_information_response(self, response: str) -> bool:
        """Check if response is an 'I don't know' type response that shouldn't have citations"""
        if not response:
            return False

        response_lower = response.lower()

        # If the response contains substantial content (sections with actual information),
        # it's not an insufficient response even if it mentions some sections don't have info
        if "## section 1 ##" in response_lower and len(response) > 500:
            # Response has structured sections with content
            return False

        # Patterns that indicate insufficient information responses
        # These should only trigger for truly insufficient responses, not mixed content
        insufficient_patterns = [
            "i don't have sufficient information",
            "i don't have any relevant information",
            "insufficient information to answer",
            "i don't have information",
            "no relevant information available",
            "i cannot provide",
            "i am unable to provide",
            "i cannot answer",
            "i am unable to answer"
        ]

        return any(pattern in response_lower for pattern in insufficient_patterns)

    def remove_citations_from_response(self, response: str) -> str:
        """Remove any citation numbers [1], [2], etc. from response text"""
        # Remove citation patterns like [1], [2], [1][2][3], etc.
        cleaned_response = re.sub(r'\[[\d\s,]+\]', '', response)
        # Clean up any double spaces left behind
        cleaned_response = re.sub(r'\s+', ' ', cleaned_response)
        return cleaned_response.strip()


# Factory function for easy creation
def create_citation_analyzer() -> CitationAnalyzer:
    """Create a new citation analyzer instance"""
    return CitationAnalyzer()