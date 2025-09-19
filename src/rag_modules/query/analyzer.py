"""
Query Analysis Module - Extracted from RAG Pipeline
Handles all query analysis and classification functionality
"""

import re
import random
import asyncio
from typing import List, Dict, Any, Optional
import structlog
from llama_index.llms.openai import OpenAI

logger = structlog.get_logger()


class QueryAnalyzer:
    """
    Handles query analysis, classification, and expansion for the RAG pipeline.

    This class provides methods to:
    - Classify query topics and determine relevance
    - Analyze organizational structure queries
    - Determine question categories for follow-ups
    - Check if questions are contextual
    - Expand queries with domain-specific terms
    - Identify general knowledge vs document-specific queries
    """

    def __init__(self, llm: OpenAI):
        """
        Initialize the QueryAnalyzer.

        Args:
            llm: OpenAI LLM instance for query classification and response generation
        """
        self.llm = llm

    def is_organizational_query(self, query: str, documents: List[Dict]) -> bool:
        """
        Check if this is an organizational structure query.

        Args:
            query: The user query to analyze
            documents: List of retrieved documents

        Returns:
            bool: True if this is an organizational structure query
        """
        org_terms = ['breakdown by job', 'organizational structure', 'job titles', 'hierarchy']
        has_org_query = any(term in query.lower() for term in org_terms)

        if has_org_query:
            # Check if documents contain spatial extraction data
            sample_content = " ".join([doc['content'][:300] for doc in documents[:3]])
            return '[SHAPE-AWARE ORGANIZATIONAL EXTRACTION]' in sample_content

        return False

    def generate_organizational_response(self, query: str, documents: List[Dict], conversation_context: str) -> str:
        """
        Generate response for organizational structure queries.

        Args:
            query: The user query
            documents: List of retrieved documents
            conversation_context: Previous conversation context

        Returns:
            str: Generated response for organizational queries
        """
        context = "\n\n".join([doc['content'] for doc in documents])

        prompt = f"""You are AAIRE, an expert in insurance accounting and actuarial matters.
{conversation_context}
Question: {query}

Organizational data:
{context}

Provide a clear organizational breakdown based on the spatial extraction data found in the documents.
Use appropriate headings and structure the information clearly."""

        response = self.llm.complete(prompt)
        return response.text.strip()

    def determine_question_categories(self, query: str, response: str, retrieved_docs: List[Dict]) -> List[str]:
        """
        Determine appropriate question categories based on context.

        Args:
            query: The user query
            response: The generated response
            retrieved_docs: List of retrieved documents

        Returns:
            List[str]: List of relevant question categories
        """
        categories = []

        if not query or not response:
            return ['general']  # Default fallback categories

        query_lower = query.lower()
        response_lower = response.lower()

        # Check for specific topics and suggest relevant categories
        if any(term in query_lower for term in ['gaap', 'ifrs', 'standard', 'compliance']):
            categories.extend(['comparison', 'compliance'])

        if any(term in query_lower for term in ['reserve', 'calculation', 'premium', 'claim']):
            categories.extend(['examples', 'technical'])

        if any(term in query_lower for term in ['audit', 'test', 'review']):
            categories.extend(['application', 'compliance'])

        if any(term in response_lower for term in ['require', 'must', 'shall']):
            categories.append('clarification')

        # Default categories if none detected
        if not categories:
            categories = ['clarification', 'examples', 'application']

        return list(set(categories))  # Remove duplicates

    def is_contextual_question(self, question: str, original_query: str, response: str) -> bool:
        """
        Validate that a follow-up question is contextual to the conversation.

        Args:
            question: The follow-up question to validate
            original_query: The original query
            response: The generated response

        Returns:
            bool: True if the question is contextual
        """
        question_lower = question.lower()
        response_lower = response.lower()
        query_lower = original_query.lower()

        # Generic phrases that indicate non-contextual questions
        generic_phrases = [
            'how do insurers typically',
            'what strategies do insurers',
            'how does claims impact profitability',
            'what are some strategies for',
            'how do companies typically handle',
            'what are the benefits of this approach',
            'how can organizations improve',
            'what factors influence profitability',
            'what are some common practices',
            'how can we improve our',
            'what does this mean for the industry',
            'what are the implications for',
            'how should companies approach',
            'what best practices should',
            'how does this compare to industry standards'
        ]

        # Check if question contains generic phrases
        if any(phrase in question_lower for phrase in generic_phrases):
            return False

        # Extract specific metrics, amounts, or data points from response
        response_specifics = []
        response_specifics.extend(re.findall(r'\$[\d,]+(?:\.\d+)?\s*(?:million|billion)', response_lower))
        response_specifics.extend(re.findall(r'\d+\.\d+%', response_lower))
        response_specifics.extend(re.findall(r'q[1-4]\s+(?:20\d{2}|results?)', response_lower))
        response_specifics.extend(re.findall(r'(?:u\.s\.|canada|emea)\s+(?:traditional|group)', response_lower))

        # Check if question references something mentioned in the response or query
        contextual_indicators = [
            # References specific financial amounts/metrics from response
            any(specific in question_lower for specific in response_specifics),
            # References specific elements from response/query
            any(word in question_lower for word in ['mentioned', 'explained', 'described', 'discussed', 'cited']),
            # References specific standards that appear in response
            bool(re.search(r'\b(?:asc|ifrs|ias|fas)\s+\d+', question_lower)) and bool(re.search(r'\b(?:asc|ifrs|ias|fas)\s+\d+', response_lower)),
            # References document-specific terms
            any(term in question_lower for term in ['document', 'section', 'example', 'table', 'schedule', 'presentation']),
            # References calculation or specific concept from response
            any(term in question_lower and term in response_lower for term in ['calculation', 'method', 'approach', 'guidance', 'component', 'breakdown']),
            # References specific company/business terms that appear in both
            any(term in question_lower and term in response_lower for term in ['rga', 'equitable', 'segment', 'traditional', 'group']),
            # References specific time periods or comparative language
            any(term in question_lower for term in ['compared to', 'other segments', 'different', 'breakdown', 'components']),
            # References specific technical terms from insurance/accounting domain
            any(term in question_lower and term in response_lower for term in ['reserve', 'reserves', 'liability', 'premium', 'valuation', 'actuarial', 'capital', 'ratio']),
            # References specific regulatory or compliance terms
            any(term in question_lower and term in response_lower for term in ['regulatory', 'compliance', 'standard', 'requirement', 'disclosure']),
            # References specific business concepts that appear in both
            any(term in question_lower and term in response_lower for term in ['analysis', 'assessment', 'evaluation', 'review', 'implementation']),
            # Contains words from original query (indicates building on the conversation)
            len([word for word in query_lower.split() if len(word) > 3 and word in question_lower]) >= 2,
            # Contains words from response (indicates referencing the answer)
            len([word for word in response_lower.split() if len(word) > 4 and word in question_lower]) >= 3,
        ]

        # More lenient: if it has at least one contextual indicator OR is clearly building on the conversation
        has_contextual_indicator = any(contextual_indicators)

        # Additional check: if question length is reasonable and contains domain-specific terms
        has_domain_terms = any(term in question_lower for term in [
            'reserve', 'liability', 'premium', 'actuarial', 'capital', 'ratio', 'valuation',
            'standard', 'compliance', 'regulatory', 'gaap', 'ifrs', 'calculation', 'method'
        ])

        return has_contextual_indicator or (len(question_lower) > 20 and has_domain_terms)

    async def classify_query_topic(self, query: str) -> Dict[str, Any]:
        """
        Classify whether the query is within AAIRE's domain expertise.

        Args:
            query: The query to classify

        Returns:
            Dict[str, Any]: Classification result with relevance and confidence
        """
        logger.info(f"📋 Starting topic classification for: '{query[:30]}...'")

        # Define relevant financial/insurance/accounting domains
        relevant_keywords = {
            'financial': [
                'financial', 'finance', 'revenue', 'profit', 'loss', 'earnings', 'income', 'expense',
                'assets', 'liabilities', 'equity', 'balance sheet', 'cash flow', 'statement',
                'budget', 'forecast', 'valuation', 'investment', 'portfolio', 'returns',
                'capital', 'funding', 'financing', 'debt', 'credit', 'loan', 'mortgage'
            ],
            'accounting': [
                'accounting', 'gaap', 'ifrs', 'asc', 'fas', 'ias', 'standard', 'compliance',
                'audit', 'auditing', 'journal', 'ledger', 'depreciation', 'amortization',
                'accrual', 'recognition', 'measurement', 'disclosure', 'reporting',
                'consolidation', 'segment', 'fair value', 'impairment', 'tax'
            ],
            'insurance': [
                'insurance', 'insurer', 'policy', 'premium', 'claim', 'coverage', 'underwriting',
                'reinsurance', 'actuarial', 'risk', 'reserve', 'liability', 'benefit',
                'annuity', 'life insurance', 'health insurance', 'property', 'casualty',
                'solvency', 'capital adequacy', 'licat', 'regulatory'
            ],
            'banking': [
                'bank', 'banking', 'deposit', 'withdrawal', 'account', 'lending', 'borrowing',
                'interest rate', 'mortgage', 'loan', 'credit', 'debit', 'payment',
                'financial institution', 'federal reserve', 'monetary policy', 'currency'
            ],
            'investment': [
                'investment', 'investing', 'stock', 'bond', 'security', 'portfolio',
                'mutual fund', 'etf', 'dividend', 'yield', 'return', 'risk', 'volatility',
                'market', 'trading', 'hedge fund', 'private equity', 'venture capital'
            ],
            'mathematical': [
                'calculation', 'formula', 'equation', 'mathematical', 'statistics', 'probability',
                'model', 'modeling', 'quantitative', 'analysis', 'ratio', 'percentage',
                'present value', 'future value', 'discount rate', 'compound', 'regression'
            ],
            'economics': [
                'economic', 'economics', 'inflation', 'deflation', 'gdp', 'recession',
                'growth', 'unemployment', 'monetary', 'fiscal', 'policy', 'market',
                'supply', 'demand', 'price', 'cost', 'microeconomic', 'macroeconomic'
            ]
        }

        # Quick keyword-based check first
        query_lower = query.lower()
        has_relevant_keywords = False

        for domain, keywords in relevant_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                has_relevant_keywords = True
                break

        # If obvious keywords found, likely relevant
        if has_relevant_keywords:
            return {'is_relevant': True, 'confidence': 0.9}

        # Use AI classification for ambiguous cases
        classification_prompt = f"""Determine if this question is relevant to AAIRE, an AI assistant specialized in:
- Financial analysis and reporting
- Accounting standards (GAAP, IFRS, ASC, etc.)
- Insurance and actuarial topics
- Banking and investment concepts
- Mathematical and statistical analysis
- Economics and business finance

Question: "{query}"

Is this question within AAIRE's expertise domain?

Respond with either:
"RELEVANT" - if the question relates to finance, accounting, insurance, banking, investments, mathematics/statistics, or economics
"NOT_RELEVANT" - if the question is about other topics like sports, entertainment, cooking, travel, personal relationships, general knowledge, etc.

Answer:"""

        try:
            response = self.llm.complete(classification_prompt)
            classification = response.text.strip().upper()

            if "RELEVANT" in classification:
                return {'is_relevant': True, 'confidence': 0.8}
            else:
                polite_responses = [
                    "I'm AAIRE, a specialized AI assistant focused on financial, accounting, insurance, and actuarial topics. I'd be happy to help you with questions related to these areas instead!",
                    "I specialize in financial, accounting, insurance, banking, and related business topics. Could you ask me something within these domains? I'd love to help!",
                    "As an insurance and financial industry specialist, I'm designed to assist with accounting standards, financial analysis, insurance topics, and related mathematical concepts. How can I help you with these areas?",
                    "I'm focused on providing expert assistance with financial, accounting, actuarial, and insurance-related questions. Is there something in these areas I can help you with instead?"
                ]

                selected_response = random.choice(polite_responses)

                return {
                    'is_relevant': False,
                    'polite_response': selected_response,
                    'confidence': 0.8
                }

        except Exception as e:
            logger.error(f"Failed to classify query topic: {e}")
            # Default to allowing the query if classification fails
            return {'is_relevant': True, 'confidence': 0.3}

    def expand_query(self, query: str) -> str:
        """
        Expand general queries with specific domain terms for better retrieval.

        Args:
            query: The original query to expand

        Returns:
            str: Expanded query with domain-specific terms
        """
        query_lower = query.lower()

        # Domain-specific term mappings for insurance and accounting
        expansion_mappings = {
            # Capital and financial health terms
            'capital health': 'capital health LICAT ratio core ratio total ratio capital adequacy',
            'company capital': 'company capital LICAT ratio core ratio total ratio regulatory capital',
            'assess capital': 'assess capital LICAT ratio core ratio total ratio capital adequacy',
            'financial strength': 'financial strength LICAT ratio core ratio total ratio capital adequacy',
            'capital adequacy': 'capital adequacy LICAT ratio core ratio total ratio regulatory capital',

            # Insurance specific expansions
            'insurance': 'insurance LICAT OSFI regulatory capital solvency',
            'regulatory': 'regulatory OSFI LICAT compliance capital requirements',
            'solvency': 'solvency LICAT ratio capital adequacy regulatory capital',

            # Accounting standard expansions
            'accounting': 'accounting GAAP IFRS standards disclosure requirements',
            'financial reporting': 'financial reporting GAAP IFRS disclosure standards',
            'compliance': 'compliance regulatory requirements OSFI GAAP IFRS',

            # Risk management expansions
            'risk': 'risk management capital risk regulatory risk operational risk',
            'management': 'management risk management capital management regulatory management'
        }

        # Apply expansions
        expanded_query = query
        for general_term, expansion in expansion_mappings.items():
            if general_term in query_lower:
                # Add specific terms to the query
                specific_terms = expansion.replace(general_term, '').strip()
                if specific_terms:
                    expanded_query = f"{query} {specific_terms}"
                break

        # Log expansion for debugging
        if expanded_query != query:
            logger.info("Query expanded",
                       original=query,
                       expanded=expanded_query)

        return expanded_query

    def is_general_knowledge_query(self, query: str) -> bool:
        """
        Check if query is asking for general knowledge vs specific document content.

        Args:
            query: The query to analyze

        Returns:
            bool: True if this is a general knowledge query
        """
        query_lower = query.lower()

        # First check for document-specific indicators (these override general patterns)
        document_indicators = [
            r'\bour company\b',
            r'\bthe uploaded\b',
            r'\bthe document\b',
            r'\bin the document\b',
            r'\bshow me\b',
            r'\bfind\b.*\bin\b',
            r'\banalyze\b',
            r'\bspecific\b.*\bmentioned\b',
            r'\bpolicy\b',
            r'\bprocedure\b',
            r'\bin the.*image\b',  # "in the chatgpt image"
            r'\bfrom the.*image\b',  # "from the image"
            r'\bthe.*chart\b',  # "the revenue chart"
            r'\buploaded.*image\b',  # "uploaded image"
            r'\bASC\s+\d{3}-\d{2}-\d{2}-\d{2}\b',  # ASC codes like "ASC 255-10-50-51"
            r'\bFASB\b',  # FASB references
            r'\bGAAP\b',  # GAAP references
            r'\bIFRS\b'   # IFRS references
        ]

        for pattern in document_indicators:
            if re.search(pattern, query_lower):
                return False  # Document-specific query

        # Common general knowledge question patterns (only if no document indicators)
        general_patterns = [
            r'^\s*what is\s+[a-z\s]+\??$',  # Simple "what is X?" questions
            r'^\s*define\s+[a-z\s]+\??$',   # Simple "define X" questions
            r'^\s*what\s+does\s+[a-z\s]+\s+mean\??$',  # Simple "what does X mean" questions
            r'^\s*what\s+are\s+the\s+types\s+of\s+[a-z\s?]+\??$',  # "what are the types of X" questions
            r'^\s*how\s+does\s+[a-z\s]+\s+work\??$'  # Simple "how does X work" questions
        ]

        for pattern in general_patterns:
            if re.search(pattern, query_lower):
                return True

        return False


def create_query_analyzer(llm: OpenAI) -> QueryAnalyzer:
    """
    Factory function to create a QueryAnalyzer instance.

    Args:
        llm: OpenAI LLM instance for query classification

    Returns:
        QueryAnalyzer: Configured QueryAnalyzer instance
    """
    return QueryAnalyzer(llm)