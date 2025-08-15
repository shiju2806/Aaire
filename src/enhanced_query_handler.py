"""
Enhanced Query Handler for Intelligent Document Processing
Determines when to use intelligent extraction vs standard RAG
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()

@dataclass
class QueryAnalysis:
    """Analysis result for a user query"""
    needs_intelligent_extraction: bool
    extraction_type: str
    confidence: float
    suggested_filters: Dict[str, Any]
    reasoning: str

class EnhancedQueryHandler:
    """
    Analyzes queries to determine the optimal processing method
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
        self.extraction_threshold = 0.3
        
        # Define extraction indicators
        self.extraction_indicators = {
            'job_titles': [
                'job title', 'position', 'role', 'who is', 'breakdown by',
                'list all', 'employees with title', 'staff members',
                'team members', 'organizational', 'people with'
            ],
            'organizational': [
                'department', 'division', 'team', 'structure', 'hierarchy',
                'reports to', 'organization', 'org chart', 'reporting'
            ],
            'financial_roles': [
                'cfo', 'treasurer', 'financial', 'accounting', 'finance team',
                'controller', 'analyst', 'financial analyst', 'accountant'
            ],
            'approval_authority': [
                'approval', 'authorization', 'authority', 'spending limit',
                'approval matrix', 'who can approve', 'signature authority'
            ]
        }
    
    def analyze_query(self, query: str, user_context: Dict[str, Any] = None) -> QueryAnalysis:
        """Analyze query to determine processing strategy"""
        
        query_lower = query.lower()
        
        # Check for extraction indicators
        needs_extraction, extraction_type, confidence = self.needs_intelligent_extraction(query)
        
        # Determine suggested filters
        suggested_filters = self._suggest_filters(query, user_context)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(query, extraction_type, confidence)
        
        return QueryAnalysis(
            needs_intelligent_extraction=needs_extraction,
            extraction_type=extraction_type,
            confidence=confidence,
            suggested_filters=suggested_filters,
            reasoning=reasoning
        )
    
    def needs_intelligent_extraction(self, query: str) -> Tuple[bool, str, float]:
        """
        Determine if query needs intelligent extraction
        Returns: (needs_extraction, extraction_type, confidence)
        """
        
        query_lower = query.lower()
        max_confidence = 0.0
        best_type = "standard"
        
        # Check each extraction type
        for extraction_type, indicators in self.extraction_indicators.items():
            confidence = 0.0
            
            for indicator in indicators:
                if indicator in query_lower:
                    # Weight by indicator strength
                    if indicator in ['job title', 'breakdown by', 'list all']:
                        confidence += 0.3
                    elif indicator in ['cfo', 'treasurer', 'controller']:
                        confidence += 0.25
                    elif indicator in ['who is', 'employees with']:
                        confidence += 0.2
                    else:
                        confidence += 0.1
            
            # Check for question patterns that suggest extraction
            extraction_patterns = [
                r'who are? the',
                r'list.*(?:employees|staff|people)',
                r'breakdown.*by.*(?:title|role|position)',
                r'what.*(?:titles|positions|roles)',
                r'how many.*(?:employees|staff)',
                r'who.*(?:cfo|treasurer|controller|analyst)'
            ]
            
            for pattern in extraction_patterns:
                if re.search(pattern, query_lower):
                    confidence += 0.2
            
            if confidence > max_confidence:
                max_confidence = confidence
                best_type = extraction_type
        
        needs_extraction = max_confidence >= self.extraction_threshold
        
        logger.info(f"Query analysis: '{query}' -> extraction_needed={needs_extraction}, type={best_type}, confidence={max_confidence}")
        
        return needs_extraction, best_type, max_confidence
    
    def _suggest_filters(self, query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Suggest filters based on query and user context"""
        
        filters = {}
        query_lower = query.lower()
        
        # Department-based filtering
        if user_context and 'department' in user_context:
            dept = user_context['department'].lower()
            if dept in ['accounting', 'actuarial', 'finance']:
                filters['department_focus'] = dept
        
        # Document type filtering
        if any(term in query_lower for term in ['financial structure', 'finance team']):
            filters['document_type'] = 'financial'
        elif any(term in query_lower for term in ['organizational', 'org chart']):
            filters['document_type'] = 'organizational'
        elif any(term in query_lower for term in ['approval', 'authorization']):
            filters['document_type'] = 'approval'
        
        return filters
    
    def _generate_reasoning(self, query: str, extraction_type: str, confidence: float) -> str:
        """Generate human-readable reasoning for the analysis"""
        
        if confidence >= self.extraction_threshold:
            return f"Query requires intelligent extraction (type: {extraction_type}, confidence: {confidence:.3f}). " \
                   f"The query contains specific indicators that suggest structured data extraction is needed."
        else:
            return f"Query can be handled with standard RAG retrieval (confidence: {confidence:.3f}). " \
                   f"No specific extraction indicators detected."
    
    async def route_query(self, query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Route query to appropriate processing method
        Returns routing decision with metadata
        """
        
        analysis = self.analyze_query(query, user_context)
        
        routing_decision = {
            'method': 'intelligent_extraction' if analysis.needs_intelligent_extraction else 'standard_rag',
            'extraction_type': analysis.extraction_type,
            'confidence': analysis.confidence,
            'filters': analysis.suggested_filters,
            'reasoning': analysis.reasoning,
            'query_analysis': analysis
        }
        
        logger.info(f"Query routing decision: {routing_decision['method']}", 
                   extraction_type=analysis.extraction_type,
                   confidence=analysis.confidence)
        
        return routing_decision