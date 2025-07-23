"""
Validation Layer - Legacy module for backward compatibility
Functionality moved to ComplianceEngine
"""

import structlog
from typing import Dict, Any, List

logger = structlog.get_logger()

class ValidationLayer:
    """Legacy validation layer - functionality moved to ComplianceEngine"""
    
    def __init__(self):
        logger.warning("ValidationLayer is deprecated. Use ComplianceEngine instead.")
    
    def validate(self, response: Dict[str, Any], knowledge: List[Dict]) -> Dict[str, Any]:
        """Basic validation for backward compatibility"""
        
        # Simple passthrough validation
        validated_response = {
            "answer": response.get("answer", ""),
            "citations": response.get("citations", []),
            "confidence_score": response.get("confidence_score", 0.5),
            "sources": response.get("sources", [])
        }
        
        return type('Response', (), validated_response)()
    
    async def validate_async(self, response: Dict[str, Any], knowledge: List[Dict]) -> Dict[str, Any]:
        """Async version of validate"""
        return self.validate(response, knowledge)