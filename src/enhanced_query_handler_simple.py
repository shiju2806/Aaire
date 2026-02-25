"""
Simplified Enhanced Query Handler 
"""

from typing import Dict, Any, List, Optional, Tuple
import structlog

logger = structlog.get_logger()

class EnhancedQueryHandler:
    """Detects queries that need intelligent extraction"""

    def __init__(self, openai_client=None):
        # Accept legacy openai_client param but prefer provider
        from .providers import get_llm_provider
        self._llm = get_llm_provider()
        
        self.extraction_indicators = {
            'job_titles': ['job title', 'position', 'role', 'who is', 'breakdown by', 'list of people', 'employees', 'staff', 'personnel'],
            'organizational': ['department', 'division', 'team', 'structure', 'hierarchy', 'organization', 'grouped by', 'categorize'],
            'financial_roles': ['cfo', 'treasurer', 'financial', 'accounting', 'finance team', 'approval', 'authority'],
            'extraction_verbs': ['extract', 'identify', 'find', 'list', 'show me', 'breakdown', 'analyze', 'categorize']
        }
    
    def needs_intelligent_extraction(self, query: str) -> Tuple[bool, str, float]:
        """Determine if query needs intelligent extraction"""
        query_lower = query.lower()
        
        scores = {}
        for extraction_type, indicators in self.extraction_indicators.items():
            score = sum(1 for indicator in indicators if indicator in query_lower)
            if score > 0:
                scores[extraction_type] = score / len(indicators)
        
        if not scores:
            return False, "none", 0.0
        
        best_type = max(scores.keys(), key=lambda k: scores[k])
        compound_score = sum(scores.values())
        
        if compound_score >= 0.3:
            return True, best_type, min(compound_score, 1.0)
        
        return False, "none", compound_score
    
    async def enhance_extraction_query(self, original_query: str, extraction_type: str) -> str:
        """Enhance query for better extraction"""
        try:
            enhancement_prompt = f"""Enhance this query for {extraction_type} extraction:
            
Original: "{original_query}"

Make it more specific and request:
1. Explicit information only
2. Structured format 
3. Confidence scores
4. Context for each finding

Return only the enhanced query."""
            
            enhanced = await self._llm.generate(
                enhancement_prompt,
                task="query_enhancement",
                system_prompt="Create precise extraction queries.",
            )
            return enhanced + " Please provide only explicitly stated information with confidence scores."
            
        except Exception as e:
            logger.error("Query enhancement failed", error=str(e))
            return original_query + " (Please provide only explicitly stated information.)"
