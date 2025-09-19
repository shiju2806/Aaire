"""
Citation Analysis Module
Handles document citation extraction and analysis for RAG responses
"""

from typing import List, Dict, Any, Tuple, Optional
import re
import structlog
from difflib import SequenceMatcher

logger = structlog.get_logger()


class CitationAnalyzer:
    """
    Analyzes retrieved documents to determine citation information
    and assess document usage in generated responses.
    """

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
            filename = top_doc['metadata'].get('filename', 'Unknown')
            citations.append({
                "id": 1,
                "text": top_doc['content'][:200] + "..." if len(top_doc['content']) > 200 else top_doc['content'],
                "source": filename,
                "source_type": top_doc['source_type'],
                "confidence": round(top_doc.get('relevance_score', top_doc.get('score', 0.0)), 3)
            })
            return citations

        logger.info(f"🎯 INTELLIGENT CITATION ANALYSIS: Analyzing response against {len(retrieved_docs)} documents")

        # Analyze which documents were actually used in the response
        used_docs = self.analyze_document_usage_in_response(retrieved_docs, response, query)

        logger.info(f"📋 Analysis result: {len(used_docs)} documents determined to be used in response")

        for i, (doc, usage_score) in enumerate(used_docs):
            relevance_score = doc.get('relevance_score', doc.get('score', 0.0))
            filename = doc['metadata'].get('filename', 'Unknown')

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

            # Get filename for source
            filename = doc['metadata'].get('filename', 'Unknown')

            # Extract page information if available
            page_info = ""
            if 'page' in doc['metadata']:
                page_info = f", Page {doc['metadata']['page']}"
            elif 'page_label' in doc['metadata']:
                page_info = f", Page {doc['metadata']['page_label']}"
            elif hasattr(doc, 'node_id') and 'page_' in str(doc.get('node_id', '')):
                # Extract page from node_id like "page_1_chunk_2"
                try:
                    page_num = str(doc.get('node_id', '')).split('page_')[1].split('_')[0]
                    page_info = f", Page {page_num}"
                except:
                    pass

            # Check if content contains page references from shape-aware extraction
            content = doc.get('content', '')
            if 'Source: Page' in content:
                # Extract page number from content like "Source: Page 2, cluster_1_page_2"
                page_match = re.search(r'Source: Page (\d+)', content)
                if page_match:
                    page_info = f", Page {page_match.group(1)}"

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
            filename = doc['metadata'].get('filename', 'Unknown')

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

        # Return only the top 1 document that was most likely used
        return doc_usage_scores[:1] if doc_usage_scores else [(retrieved_docs[0], 0.5)]  # Fallback to top doc

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