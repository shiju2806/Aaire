"""
Knowledge Engine for AAIRE - Deprecated in favor of RAG Pipeline
This module is kept for compatibility but functionality moved to rag_pipeline.py
"""

from typing import List, Dict, Any
from datetime import datetime
import structlog

logger = structlog.get_logger()

class KnowledgeEngine:
    """
    Legacy knowledge engine - replaced by RAGPipeline in MVP
    Kept for backward compatibility
    """
    
    def __init__(self):
        """Initialize legacy knowledge engine"""
        logger.warning("KnowledgeEngine is deprecated. Use RAGPipeline instead.")
        
        # Basic knowledge data for backward compatibility
        self.knowledge_data = {
            "us_gaap": [
                {
                    "id": "asc_944_1",
                    "content": "ASC 944-20: Insurance contracts should be classified as insurance contracts, investment contracts, or service contracts based on the level of insurance risk transferred.",
                    "standard": "ASC 944-20",
                    "topic": "Insurance Contract Classification",
                    "framework": "US_GAAP"
                }
            ],
            "ifrs": [
                {
                    "id": "ifrs_17_1",
                    "content": "IFRS 17 requires insurance contracts to be measured using the General Measurement Model, Variable Fee Approach, or Premium Allocation Approach.",
                    "standard": "IFRS 17",
                    "topic": "Measurement Models",
                    "framework": "IFRS"
                }
            ]
        }
    
    def search(self, query: str, framework: str = "US_GAAP", n_results: int = 5) -> List[Dict[str, Any]]:
        """Basic search for backward compatibility"""
        logger.warning("Using deprecated search method. Migrate to RAGPipeline.")
        
        # Simple keyword matching for compatibility
        results = []
        search_data = []
        
        if framework == "US_GAAP":
            search_data = self.knowledge_data["us_gaap"]
        elif framework == "IFRS":
            search_data = self.knowledge_data["ifrs"]
        else:
            search_data = self.knowledge_data["us_gaap"] + self.knowledge_data["ifrs"]
        
        for item in search_data:
            if query.lower() in item["content"].lower():
                results.append({
                    "content": item["content"],
                    "metadata": {k: v for k, v in item.items() if k != "content"},
                    "distance": 0.5,  # Dummy distance
                    "id": item["id"]
                })
        
        return results[:n_results]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get basic statistics"""
        return {
            "us_gaap_documents": len(self.knowledge_data["us_gaap"]),
            "ifrs_documents": len(self.knowledge_data["ifrs"]),
            "total_documents": len(self.knowledge_data["us_gaap"]) + len(self.knowledge_data["ifrs"]),
            "last_updated": datetime.now().isoformat(),
            "status": "deprecated - use RAGPipeline"
        }