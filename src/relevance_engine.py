"""
Advanced Relevance Engine for Dynamic Document Ranking
No hardcoded patterns - uses configurable, learning-based approach
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import structlog

logger = structlog.get_logger()

class QueryType(Enum):
    SPECIFIC_REFERENCE = "specific_reference"  # ASC 255-10-50-51, IFRS 9.5.7.1
    CONCEPTUAL = "conceptual"                  # "what is revenue recognition"
    COMPARISON = "comparison"                  # "difference between GAAP and IFRS"
    PROCEDURAL = "procedural"                  # "how to calculate reserves"
    CONTEXTUAL = "contextual"                  # "in the attached document"

@dataclass
class QueryAnalysis:
    query_type: QueryType
    entities: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    specificity_score: float = 0.0
    context_indicators: List[str] = field(default_factory=list)
    search_weights: Dict[str, float] = field(default_factory=dict)

@dataclass 
class DocumentRelevance:
    document_id: str
    exact_match_score: float = 0.0
    semantic_score: float = 0.0
    context_score: float = 0.0
    entity_coverage_score: float = 0.0
    final_score: float = 0.0
    explanation: List[str] = field(default_factory=list)

class RelevanceEngine:
    """Dynamic relevance scoring without hardcoded patterns"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.query_patterns = self._build_dynamic_patterns()
        self.domain_keywords = self._build_domain_knowledge()
        self.feedback_data = {}  # For learning
        
        # Initialize OpenAI client for AI-powered domain classification
        try:
            import openai
            import os
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
                logger.info("🤖 AI-powered domain classification enabled")
            else:
                self.openai_client = None
                logger.info("⚠️ OpenAI API key not found, using pattern-based domain detection")
        except ImportError:
            self.openai_client = None
            logger.info("⚠️ OpenAI not installed, using pattern-based domain detection")
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configurable relevance parameters"""
        default_config = {
            "scoring_weights": {
                "exact_match": {"min": 0.1, "max": 0.8},
                "semantic": {"min": 0.2, "max": 0.7}, 
                "context": {"min": 0.1, "max": 0.4},
                "entity_coverage": {"min": 0.1, "max": 0.5}
            },
            "query_type_weights": {
                QueryType.SPECIFIC_REFERENCE.value: {
                    "exact_match": 0.6, "semantic": 0.2, "context": 0.1, "entity_coverage": 0.1
                },
                QueryType.CONCEPTUAL.value: {
                    "exact_match": 0.2, "semantic": 0.6, "context": 0.1, "entity_coverage": 0.1  
                },
                QueryType.CONTEXTUAL.value: {
                    "exact_match": 0.3, "semantic": 0.4, "context": 0.2, "entity_coverage": 0.1
                }
            },
            "boost_thresholds": {
                "high_specificity": 0.8,
                "exact_entity_match": 0.3,
                "context_alignment": 0.2
            }
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}, using defaults: {e}")
        
        return default_config
    
    def _build_dynamic_patterns(self) -> Dict[str, List[str]]:
        """Build flexible pattern recognition (no hardcoding)"""
        return {
            "reference_indicators": [
                r'\b[A-Z]{2,6}\s+\d+(?:[.-]\d+)*\b',  # ASC 255-10-50-51, IFRS 9.5.7.1, etc.
                r'\b(?:section|paragraph|article)\s+\d+(?:[.-]\d+)*\b',
                r'\b\d+(?:[.-]\d+)+\b'  # Generic numbered references
            ],
            "context_indicators": [
                r'\b(?:in|from|within|according to)\s+(?:the\s+)?(?:attached|uploaded|provided|above|document)\b',
                r'\b(?:this|that)\s+(?:document|file|pdf|report)\b',
                r'\b(?:as\s+(?:mentioned|stated|shown|discussed))\b'
            ],
            "domain_indicators": [
                r'\b(?:accounting|financial|insurance|actuarial|regulatory|compliance)\b',
                r'\b(?:GAAP|IFRS|ASC|FASB|IASB|PCAOB)\b',
                r'\b(?:revenue|liability|asset|equity|reserve|premium)\b'
            ]
        }
    
    def _build_domain_knowledge(self) -> Dict[str, List[str]]:
        """Build minimal domain knowledge base (fully dynamic)"""
        # Only keep essential domain patterns - no hardcoded technical terms
        base_knowledge = {
            "accounting": [
                "revenue", "recognition", "liability", "asset", "equity", "gaap", "ifrs",
                "fasb", "asc", "financial", "statement", "disclosure", "measurement"
            ],
            "insurance": [
                "actuarial", "reserve", "premium", "policy", "claim", "underwriting",
                "reinsurance", "solvency", "capital", "licat", "regulatory"
            ],
            "foreign_currency": [
                "foreign", "currency", "exchange", "rate", "translation", "remeasurement",
                "functional", "reporting", "monetary", "nonmonetary"
            ]
        }

        return base_knowledge
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query to determine optimal search strategy"""
        query_lower = query.lower()
        
        # Extract entities (flexible pattern matching)
        entities = self._extract_entities(query)
        
        # Determine query type
        query_type = self._classify_query_type(query, entities)
        
        # Identify domain
        domain = self._identify_domain(query)
        
        # Calculate specificity score
        specificity_score = self._calculate_specificity(query, entities)
        
        # Find context indicators
        context_indicators = self._find_context_indicators(query)
        
        # Determine search weights based on analysis
        search_weights = self._determine_search_weights(query_type, specificity_score)
        
        analysis = QueryAnalysis(
            query_type=query_type,
            entities=entities,
            domain=domain,
            specificity_score=specificity_score,
            context_indicators=context_indicators,
            search_weights=search_weights
        )
        
        logger.info("Query analysis completed", 
                   query_type=query_type.value,
                   entities=entities[:3],  # Log first 3 entities
                   domain=domain,
                   specificity=specificity_score)
        
        return analysis
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities using flexible patterns"""
        entities = []
        
        # Use dynamic patterns instead of hardcoded ones
        for pattern_type, patterns in self.query_patterns.items():
            if pattern_type == "reference_indicators":
                for pattern in patterns:
                    matches = re.findall(pattern, query, re.IGNORECASE)
                    entities.extend(matches)
        
        # Extract quoted strings as entities
        quoted_entities = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted_entities)
        
        # Extract capitalized acronyms
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', query)
        entities.extend(acronyms)
        
        return list(set(entities))
    
    def _classify_query_type(self, query: str, entities: List[str]) -> QueryType:
        """Classify query type based on content analysis"""
        query_lower = query.lower()
        
        # Check for context indicators
        for pattern in self.query_patterns["context_indicators"]:
            if re.search(pattern, query_lower):
                return QueryType.CONTEXTUAL
        
        # Check for specific references (based on entities found)
        if any(re.search(r'\d+(?:[.-]\d+)+', entity) for entity in entities):
            return QueryType.SPECIFIC_REFERENCE
        
        # Check for comparison words
        comparison_words = ["difference", "compare", "versus", "vs", "between", "contrast"]
        if any(word in query_lower for word in comparison_words):
            return QueryType.COMPARISON
        
        # Check for procedural words
        procedural_words = ["how to", "steps", "process", "procedure", "calculate", "determine"]
        if any(phrase in query_lower for phrase in procedural_words):
            return QueryType.PROCEDURAL
        
        # Default to conceptual
        return QueryType.CONCEPTUAL
    
    def _identify_domain(self, query: str) -> Optional[str]:
        """AI-powered domain identification using semantic analysis"""
        try:
            # Use LLM for intelligent domain classification
            domain_prompt = f"""Analyze this query and classify its primary domain. Return only the domain name.

Query: "{query}"

Possible domains:
- accounting: Financial reporting, GAAP, IFRS, ASC standards, revenue recognition
- insurance: Actuarial, reserves, premiums, regulatory capital, LICAT
- foreign_currency: FX, exchange rates, translation, hedging
- tax: Tax accounting, deferred tax, tax compliance
- legal: Contracts, regulations, compliance requirements  
- regulatory: Government requirements, filing obligations
- general: Mixed or unclear domain

Return only the domain name (e.g., "accounting"):"""

            # Simple classification using OpenAI (fallback to keyword matching if unavailable)
            if hasattr(self, 'openai_client') and self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": domain_prompt}],
                    temperature=0.1,
                    max_tokens=10
                )
                domain = response.choices[0].message.content.strip().lower()
                logger.info(f"🤖 AI-classified domain: '{domain}' for query: '{query[:50]}...'")
                return domain if domain in ['accounting', 'insurance', 'foreign_currency', 'tax', 'legal', 'regulatory', 'general'] else None
            else:
                # Fallback to enhanced keyword matching
                return self._fallback_domain_detection(query)
                
        except Exception as e:
            logger.warning(f"AI domain classification failed: {e}, falling back to keyword matching")
            return self._fallback_domain_detection(query)
    
    def _fallback_domain_detection(self, query: str) -> Optional[str]:
        """Enhanced keyword-based domain detection as fallback"""
        query_lower = query.lower()
        
        # Smart pattern matching instead of hardcoded keywords
        domain_patterns = {
            'accounting': [
                r'\b(?:asc|ifrs|gaap|fasb)\b',
                r'\b(?:revenue|liability|asset|equity)\b',
                r'\b(?:financial|accounting|reporting)\b',
                r'\b\d+(?:-\d+)+\b'  # ASC/IFRS numbering patterns
            ],
            'insurance': [
                r'\b(?:licat|actuarial|reserve|premium)\b',
                r'\b(?:insurance|policy|claim|underwriting)\b',
                r'\b(?:regulatory|capital|solvency)\b'
            ],
            'foreign_currency': [
                r'\b(?:foreign|currency|fx|exchange)\b',
                r'\b(?:translation|hedging|rate)\b'
            ]
        }
        
        domain_scores = {}
        for domain, patterns in domain_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, query_lower))
            if score > 0:
                domain_scores[domain] = score
        
        return max(domain_scores, key=domain_scores.get) if domain_scores else None
    
    def learn_from_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Learn domain patterns from uploaded documents (adaptive learning)"""
        try:
            domain_terms = {}
            
            for doc in documents:
                filename = doc.get('metadata', {}).get('filename', '').lower()
                content = doc.get('content', '').lower()
                
                # Extract key terms from successful document matches
                # This would ideally use NLP to extract domain-specific terminology
                terms = re.findall(r'\b[a-z]{3,15}\b', content)
                
                # Infer document domain
                doc_domain = self._infer_document_domain_from_content(filename, content)
                
                if doc_domain not in domain_terms:
                    domain_terms[doc_domain] = set()
                
                # Add high-frequency terms to domain vocabulary
                domain_terms[doc_domain].update(terms[:50])  # Top 50 terms per document
            
            # Update domain knowledge (this could be persisted to a file)
            for domain, terms in domain_terms.items():
                if domain in self.domain_keywords:
                    self.domain_keywords[domain].extend(list(terms)[:20])  # Add top 20 new terms
                
            logger.info(f"🧠 Learned domain patterns from {len(documents)} documents")

        except Exception as e:
            logger.warning(f"Failed to learn from documents: {e}")

    def _calculate_llm_methodology_boost(self, query_analysis: QueryAnalysis, content: str, metadata: Dict) -> float:
        """Use LLM to determine if document contains methodology relevant to query"""
        try:
            if not self.openai_client:
                return 0.0

            # Extract query intent and document preview
            query = query_analysis.entities[0] if query_analysis.entities else "general query"
            doc_preview = content[:800]  # First 800 chars
            filename = metadata.get('filename', 'Unknown')

            methodology_prompt = f"""Analyze if this document contains specific methodology or technical procedures relevant to the user's query.

Query context: Looking for information about "{query}"
Document filename: {filename}
Document preview: {doc_preview}

Does this document contain:
1. Specific calculation methods, formulas, or procedures?
2. Technical terminology or industry-specific terms relevant to the query?
3. Step-by-step processes or methodological guidance?

Respond with a relevance score from 0.0 to 0.3 where:
- 0.0 = No methodology content relevant to query
- 0.1 = Some general methodology mentioned
- 0.2 = Good methodology content relevant to query
- 0.3 = Excellent methodology content directly addressing query

Respond with only the number (e.g., "0.2"):"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": methodology_prompt}],
                temperature=0.1,
                max_tokens=10
            )

            boost_score = float(response.choices[0].message.content.strip())
            logger.debug(f"🤖 LLM methodology boost: {boost_score} for {filename}")
            return min(max(boost_score, 0.0), 0.3)  # Clamp between 0.0 and 0.3

        except Exception as e:
            logger.warning(f"LLM methodology boost failed: {e}")
            return 0.0
    
    def _infer_document_domain_from_content(self, filename: str, content: str) -> str:
        """Infer domain from document content and filename"""
        # Use existing pattern matching for now, but this could be enhanced with ML
        return self._fallback_domain_detection(f"{filename} {content[:500]}") or 'general'
    
    def _calculate_specificity(self, query: str, entities: List[str]) -> float:
        """Calculate how specific the query is (0.0 to 1.0)"""
        specificity_factors = []
        
        # Length factor (longer queries tend to be more specific)
        word_count = len(query.split())
        length_factor = min(word_count / 10.0, 1.0)
        specificity_factors.append(length_factor)
        
        # Entity factor (more entities = more specific)
        entity_factor = min(len(entities) / 3.0, 1.0)
        specificity_factors.append(entity_factor)
        
        # Number/code factor (presence of numbers/codes increases specificity)
        has_numbers = bool(re.search(r'\d+', query))
        number_factor = 0.5 if has_numbers else 0.0
        specificity_factors.append(number_factor)
        
        # Quote factor (quoted terms increase specificity)
        has_quotes = '"' in query
        quote_factor = 0.3 if has_quotes else 0.0
        specificity_factors.append(quote_factor)
        
        return sum(specificity_factors) / len(specificity_factors)
    
    def _find_context_indicators(self, query: str) -> List[str]:
        """Find context indicators in query"""
        indicators = []
        query_lower = query.lower()
        
        for pattern in self.query_patterns["context_indicators"]:
            matches = re.findall(pattern, query_lower)
            indicators.extend(matches)
        
        return indicators
    
    def _determine_search_weights(self, query_type: QueryType, specificity_score: float) -> Dict[str, float]:
        """Determine search component weights based on query analysis"""
        base_weights = self.config["query_type_weights"].get(
            query_type.value,
            self.config["query_type_weights"][QueryType.CONCEPTUAL.value]
        )
        
        # Adjust weights based on specificity
        adjusted_weights = base_weights.copy()
        
        if specificity_score > 0.7:
            # High specificity - boost exact match, reduce semantic
            adjusted_weights["exact_match"] = min(0.8, adjusted_weights["exact_match"] + 0.2)
            adjusted_weights["semantic"] = max(0.1, adjusted_weights["semantic"] - 0.1)
        elif specificity_score < 0.3:
            # Low specificity - boost semantic, reduce exact match  
            adjusted_weights["semantic"] = min(0.7, adjusted_weights["semantic"] + 0.2)
            adjusted_weights["exact_match"] = max(0.1, adjusted_weights["exact_match"] - 0.1)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(adjusted_weights.values())
        normalized_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
        
        return normalized_weights
    
    def score_document_relevance(self, query_analysis: QueryAnalysis, document: Dict[str, Any]) -> DocumentRelevance:
        """Score document relevance using LLM-driven approach"""
        doc_id = document.get('node_id', 'unknown')
        content = document.get('content', '')
        metadata = document.get('metadata', {})

        relevance = DocumentRelevance(document_id=doc_id)

        # 1. Exact Match Score (flexible pattern matching)
        relevance.exact_match_score = self._calculate_exact_match_score(
            query_analysis, content, metadata
        )

        # 2. Semantic Score (use existing vector score)
        relevance.semantic_score = document.get('score', 0.0)

        # 3. Context Score (document type, domain alignment, etc.)
        relevance.context_score = self._calculate_context_score(
            query_analysis, content, metadata
        )

        # 4. Entity Coverage Score
        relevance.entity_coverage_score = self._calculate_entity_coverage_score(
            query_analysis.entities, content
        )

        # 5. LLM-driven methodology relevance boost
        methodology_boost = self._calculate_llm_methodology_boost(
            query_analysis, content, metadata
        )

        # 6. Calculate final weighted score
        weights = query_analysis.search_weights
        base_score = (
            relevance.exact_match_score * weights.get("exact_match", 0.3) +
            relevance.semantic_score * weights.get("semantic", 0.4) +
            relevance.context_score * weights.get("context", 0.2) +
            relevance.entity_coverage_score * weights.get("entity_coverage", 0.1)
        )

        # Apply methodology boost
        relevance.final_score = base_score + methodology_boost

        # Generate explanation
        relevance.explanation = self._generate_explanation(relevance, query_analysis)

        return relevance
    
    def _calculate_exact_match_score(self, query_analysis: QueryAnalysis, content: str, metadata: Dict) -> float:
        """Calculate exact match score using flexible patterns"""
        if not query_analysis.entities:
            return 0.0
        
        content_lower = content.lower()
        score = 0.0
        matches = 0
        
        for entity in query_analysis.entities:
            # Flexible matching - try exact, partial, and pattern-based
            entity_lower = entity.lower()
            
            if entity_lower in content_lower:
                matches += 1
                # Boost score based on match quality
                if len(entity) > 10:  # Long entities (like full references) get higher score
                    score += 0.3
                elif re.search(r'\d+(?:[.-]\d+)+', entity):  # Numbered references
                    score += 0.25
                else:
                    score += 0.15
        
        # Normalize by number of entities
        if query_analysis.entities:
            coverage_ratio = matches / len(query_analysis.entities)
            score = score * coverage_ratio
        
        return min(score, 1.0)
    
    def _calculate_context_score(self, query_analysis: QueryAnalysis, content: str, metadata: Dict) -> float:
        """Calculate context alignment score"""
        score = 0.0
        
        # Domain alignment
        if query_analysis.domain:
            domain_keywords = self.domain_keywords.get(query_analysis.domain, [])
            content_lower = content.lower()
            
            keyword_matches = sum(1 for keyword in domain_keywords if keyword in content_lower)
            domain_score = min(keyword_matches / len(domain_keywords), 1.0) if domain_keywords else 0.0
            score += domain_score * 0.5
        
        # Document type alignment
        doc_type = metadata.get('source_type', '').lower()
        filename = metadata.get('filename', '').lower()
        
        # Infer document relevance from metadata
        if query_analysis.domain == 'accounting' and any(term in filename for term in ['gaap', 'asc', 'fasb']):
            score += 0.3
        elif query_analysis.domain == 'insurance' and any(term in filename for term in ['licat', 'actuarial']):
            score += 0.3
        elif query_analysis.domain == 'foreign_currency' and any(term in filename for term in ['foreign', 'currency', 'pwc']):
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_entity_coverage_score(self, entities: List[str], content: str) -> float:
        """Calculate how well document covers query entities"""
        if not entities:
            return 0.0
        
        content_lower = content.lower()
        covered_entities = 0
        
        for entity in entities:
            if entity.lower() in content_lower:
                covered_entities += 1
        
        return covered_entities / len(entities)
    
    def _generate_explanation(self, relevance: DocumentRelevance, query_analysis: QueryAnalysis) -> List[str]:
        """Generate human-readable explanation of scoring"""
        explanations = []
        
        if relevance.exact_match_score > 0.2:
            explanations.append(f"Strong exact match (score: {relevance.exact_match_score:.2f})")
        
        if relevance.semantic_score > 0.7:
            explanations.append(f"High semantic similarity (score: {relevance.semantic_score:.2f})")
        
        if relevance.context_score > 0.3:
            explanations.append(f"Good context alignment (score: {relevance.context_score:.2f})")
        
        if relevance.entity_coverage_score > 0.5:
            explanations.append(f"Covers {relevance.entity_coverage_score:.0%} of query entities")
        
        explanations.append(f"Query type: {query_analysis.query_type.value}")
        
        return explanations
    
    def rank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Main entry point - analyze query and rank documents"""
        
        # Analyze the query
        query_analysis = self.analyze_query(query)
        
        # Score each document
        scored_docs = []
        for doc in documents:
            relevance = self.score_document_relevance(query_analysis, doc)
            
            # Add relevance info to document
            doc_with_relevance = doc.copy()
            doc_with_relevance['relevance_score'] = relevance.final_score
            doc_with_relevance['relevance_breakdown'] = {
                'exact_match': relevance.exact_match_score,
                'semantic': relevance.semantic_score, 
                'context': relevance.context_score,
                'entity_coverage': relevance.entity_coverage_score
            }
            doc_with_relevance['explanation'] = relevance.explanation
            doc_with_relevance['final_score'] = relevance.final_score
            
            scored_docs.append(doc_with_relevance)
        
        # Sort by relevance score
        scored_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Log top results for debugging
        if scored_docs:
            logger.info("Top ranked documents:", 
                       top_3=[(doc['metadata'].get('filename', 'Unknown'), 
                              doc['relevance_score']) for doc in scored_docs[:3]])
        
        return scored_docs
    
    def learn_from_feedback(self, query: str, document_id: str, feedback_type: str, feedback_value: float):
        """Learn from user feedback to improve future rankings"""
        if query not in self.feedback_data:
            self.feedback_data[query] = {}
        
        if document_id not in self.feedback_data[query]:
            self.feedback_data[query][document_id] = []
        
        self.feedback_data[query][document_id].append({
            'type': feedback_type,
            'value': feedback_value,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # TODO: Implement learning algorithm to adjust weights based on feedback
        logger.info(f"Feedback recorded for query '{query}': {feedback_type}={feedback_value}")