"""
AAIRE (Accounting & Actuarial Insurance Resource Expert) - MVP
Main FastAPI application following SRS v2.0 specifications
"""

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import asyncio
import logging
import structlog
from datetime import datetime

from src.rag_pipeline import RAGPipeline
from src.compliance_engine import ComplianceEngine
from src.document_processor import DocumentProcessor
from src.external_apis import ExternalAPIManager
from src.auth import AuthManager
from src.audit_logger import AuditLogger

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

app = FastAPI(
    title="AAIRE",
    description="AI-powered conversational assistant for insurance accounting and actuarial questions",
    version="1.0-MVP",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize core components
rag_pipeline = RAGPipeline()
compliance_engine = ComplianceEngine()
document_processor = DocumentProcessor()
external_api_manager = ExternalAPIManager()
auth_manager = AuthManager()
audit_logger = AuditLogger()

# Request/Response Models per MVP API spec
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

class DocumentUploadRequest(BaseModel):
    title: str
    source_type: str
    effective_date: str
    tags: Optional[List[str]] = []

class DocumentUploadResponse(BaseModel):
    job_id: str
    status: str
    message: str

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AAIRE",
        "version": "1.0-MVP",
        "status": "healthy",
        "description": "Accounting & Actuarial Insurance Resource Expert"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "AAIRE",
        "version": "1.0-MVP"
    }

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Main chat endpoint - MVP-FR-001 through MVP-FR-008
    Implements query processing, retrieval, and response generation
    """
    start_time = datetime.utcnow()
    
    try:
        # Authenticate user
        user = await auth_manager.authenticate(credentials.credentials)
        
        # Audit log
        await audit_logger.log_event(
            event="query_submitted",
            user_id=user.id,
            data={"query": request.query, "session_id": request.session_id}
        )
        
        # Check compliance rules first - MVP-FR-017
        compliance_result = await compliance_engine.check_query(request.query)
        if compliance_result.blocked:
            await audit_logger.log_event(
                event="compliance_triggered",
                user_id=user.id,
                data={"rule": compliance_result.rule_name, "query": request.query}
            )
            
            return ChatResponse(
                response=compliance_result.response,
                citations=[],
                confidence=1.0,
                session_id=request.session_id or "new",
                compliance_triggered=True,
                processing_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
            )
        
        # Process query through RAG pipeline
        rag_response = await rag_pipeline.process_query(
            query=request.query,
            filters=request.filters,
            user_context=user.context
        )
        
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        return ChatResponse(
            response=rag_response.answer,
            citations=rag_response.citations,
            confidence=rag_response.confidence,
            session_id=request.session_id or rag_response.session_id,
            compliance_triggered=False,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error("Error processing chat request", error=str(e), user_id=getattr(user, 'id', 'unknown'))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/documents", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    metadata: str = None,  # JSON string
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Document upload endpoint - MVP-FR-009 through MVP-FR-012
    """
    try:
        # Authenticate and authorize
        user = await auth_manager.authenticate(credentials.credentials)
        if not user.can_upload_documents:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        # Process document upload
        job_id = await document_processor.upload_document(
            file=file,
            metadata=metadata,
            user_id=user.id
        )
        
        await audit_logger.log_event(
            event="document_uploaded",
            user_id=user.id,
            data={"filename": file.filename, "job_id": job_id}
        )
        
        return DocumentUploadResponse(
            job_id=job_id,
            status="accepted",
            message="Document queued for processing"
        )
        
    except Exception as e:
        logger.error("Error uploading document", error=str(e))
        raise HTTPException(status_code=500, detail="Upload failed")

@app.websocket("/api/v1/chat/stream")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for streaming responses
    """
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # Process query and stream response
            async for chunk in rag_pipeline.stream_response(data["query"]):
                await websocket.send_json({
                    "type": "chunk",
                    "content": chunk
                })
            
            await websocket.send_json({
                "type": "complete",
                "citations": []  # Include citations when complete
            })
            
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        await websocket.close()

@app.get("/api/v1/documents/{job_id}/status")
async def get_document_status(
    job_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get document processing status"""
    user = await auth_manager.authenticate(credentials.credentials)
    status = await document_processor.get_status(job_id, user.id)
    return status

@app.get("/api/v1/knowledge/stats")
async def get_knowledge_stats(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get knowledge base statistics - Admin only"""
    user = await auth_manager.authenticate(credentials.credentials)
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return await rag_pipeline.get_stats()

@app.get("/api/v1/external/refresh")
async def refresh_external_data(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Trigger refresh of external data sources - Admin only"""
    user = await auth_manager.authenticate(credentials.credentials)
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    job_id = await external_api_manager.refresh_all()
    return {"job_id": job_id, "status": "started"}

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )