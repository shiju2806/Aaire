"""
Query Processor - Legacy module for backward compatibility
Functionality moved to RAGPipeline
"""

import structlog
from typing import Dict, Any, Optional

logger = structlog.get_logger()

class QueryProcessor:
    """Legacy query processor - functionality moved to RAGPipeline"""
    
    def __init__(self):
        logger.warning("QueryProcessor is deprecated. Use RAGPipeline instead.")
    
    def process(self, query: str, context: Optional[str] = None) -> str:
        """Basic query processing for backward compatibility"""
        # Simple passthrough - real processing is in RAGPipeline
        return query.strip()
    
    async def process_async(self, query: str, context: Optional[str] = None) -> str:
        """Async version of process"""
        return self.process(query, context)