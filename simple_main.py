"""
AAIRE Simple Startup - For testing without all dependencies
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime
import os

app = FastAPI(
    title="AAIRE",
    description="AI-powered conversational assistant for insurance accounting and actuarial questions",
    version="1.0-MVP-Simple",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple models
class ChatRequest(BaseModel):
    query: str = Field(..., max_length=2000, description="User query")
    session_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    citations: List[Dict[str, Any]]
    confidence: float
    session_id: str
    compliance_triggered: bool = False
    processing_time_ms: int

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AAIRE",
        "version": "1.0-MVP-Simple",
        "status": "healthy",
        "description": "Accounting & Actuarial Insurance Resource Expert - Simple Mode",
        "message": "Simple mode active. Configure API keys and restart for full functionality."
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "AAIRE",
        "version": "1.0-MVP-Simple",
        "mode": "simple",
        "environment_check": {
            "openai_api_key": "OPENAI_API_KEY" in os.environ,
            "pinecone_api_key": "PINECONE_API_KEY" in os.environ,
            "fred_api_key": "FRED_API_KEY" in os.environ
        }
    }

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Simple chat endpoint for testing
    """
    start_time = datetime.utcnow()
    
    # Simple compliance check
    blocked_terms = ["tax advice", "legal advice", "investment advice"]
    query_lower = request.query.lower()
    
    for term in blocked_terms:
        if term in query_lower:
            return ChatResponse(
                response=f"I cannot provide {term}. I can explain accounting standards and their requirements, but please consult qualified professionals for {term}.",
                citations=[],
                confidence=1.0,
                session_id=request.session_id or "compliance",
                compliance_triggered=True,
                processing_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
            )
    
    # Simple knowledge base
    knowledge_responses = {
        "ifrs 17": "IFRS 17 requires insurance contracts to be measured using the General Measurement Model, Variable Fee Approach, or Premium Allocation Approach. The standard includes a Contractual Service Margin (CSM) that represents unearned profit.",
        "asc 944": "ASC 944 provides guidance for insurance enterprises. Key areas include contract classification, premium revenue recognition, and reserve calculations.",
        "insurance reserves": "Insurance reserves represent estimates of future claim payments. Under IFRS 17, they include risk adjustment and contractual service margin components.",
        "contractual service margin": "The Contractual Service Margin (CSM) under IFRS 17 represents the unearned profit in insurance contracts and is recognized over the coverage period.",
        "premium allocation approach": "The Premium Allocation Approach (PAA) is a simplified measurement model under IFRS 17 for short-duration contracts or contracts where PAA approximates the general model."
    }
    
    # Simple matching
    response_text = "I understand you're asking about insurance accounting. "
    citations = []
    
    for key, value in knowledge_responses.items():
        if key in query_lower:
            response_text = value
            citations = [{
                "id": 1,
                "text": value[:100] + "...",
                "source": "IFRS 17" if "ifrs" in key else "ASC 944",
                "confidence": 0.9
            }]
            break
    else:
        response_text += f"Your query about '{request.query}' is noted. For full functionality with AI-powered responses, please configure OpenAI and Pinecone API keys and restart with the full system."
    
    processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    
    return ChatResponse(
        response=response_text,
        citations=citations,
        confidence=0.8 if citations else 0.5,
        session_id=request.session_id or "simple",
        compliance_triggered=False,
        processing_time_ms=processing_time
    )

@app.get("/api/v1/knowledge/stats")
async def get_knowledge_stats():
    """Get simple knowledge base statistics"""
    return {
        "status": "Simple mode active",
        "total_responses": 5,
        "frameworks": ["IFRS 17", "ASC 944"],
        "mode": "simple",
        "message": "Configure API keys for full RAG pipeline functionality",
        "environment_check": {
            "openai_api_key": "OPENAI_API_KEY" in os.environ,
            "pinecone_api_key": "PINECONE_API_KEY" in os.environ,
            "fred_api_key": "FRED_API_KEY" in os.environ
        }
    }

if __name__ == "__main__":
    print("🚀 Starting AAIRE in Simple Mode...")
    print("📝 This mode provides basic functionality for testing")
    print("🔧 Configure API keys and use main.py for full features")
    print("📊 Access API docs at: http://localhost:8000/api/docs")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )