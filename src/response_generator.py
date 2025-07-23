"""
Response Generator - Legacy module for backward compatibility
Functionality moved to RAGPipeline
"""

import structlog
from typing import Dict, Any, List

logger = structlog.get_logger()

class ResponseGenerator:
    """Legacy response generator - functionality moved to RAGPipeline"""
    
    def __init__(self):
        logger.warning("ResponseGenerator is deprecated. Use RAGPipeline instead.")
    
    def generate(self, query: str, knowledge: List[Dict], framework: str = "US_GAAP") -> Dict[str, Any]:
        """Basic response generation for backward compatibility"""
        
        # Simple fallback response
        response = {
            "answer": "I'm currently initializing. Please use the RAGPipeline for full functionality.",
            "citations": [],
            "confidence_score": 0.5,
            "sources": []
        }
        
        return response
    
    async def generate_async(self, query: str, knowledge: List[Dict], framework: str = "US_GAAP") -> Dict[str, Any]:
        """Async version of generate"""
        return self.generate(query, knowledge, framework)