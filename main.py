"""
AAIRE (Accounting & Actuarial Insurance Resource Expert) - MVP
Main FastAPI application following SRS v2.0 specifications
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import asyncio
import logging
import structlog
import json
import os
import re
from datetime import datetime
from pathlib import Path

# Import modules with fallbacks for MVP startup
try:
    from src.rag_pipeline import RAGPipeline
except ImportError:
    RAGPipeline = None

try:
    from src.compliance_engine import ComplianceEngine
except ImportError:
    ComplianceEngine = None

try:
    from src.document_processor import DocumentProcessor
except ImportError:
    DocumentProcessor = None

try:
    from src.external_apis import ExternalAPIManager
except ImportError:
    ExternalAPIManager = None

try:
    from src.auth import AuthManager
except ImportError:
    AuthManager = None

try:
    from src.audit_logger import AuditLogger
except ImportError:
    AuditLogger = None

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

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Create static directory if it doesn't exist
os.makedirs("static/js", exist_ok=True)

# Security
security = HTTPBearer()

# Initialize core components with fallbacks
rag_pipeline = None
compliance_engine = None
document_processor = None
external_api_manager = None
auth_manager = None
audit_logger = None

logger.info("Starting component initialization...")

# Initialize RAG Pipeline first
try:
    if RAGPipeline:
        logger.info("Initializing RAG Pipeline...")
        rag_pipeline = RAGPipeline()
        logger.info("✅ RAG Pipeline initialized successfully")
    else:
        logger.warning("❌ RAGPipeline class not available")
except Exception as e:
    logger.error("❌ RAG Pipeline initialization failed", error=str(e), exc_info=True)

# Initialize other components
try:
    if ComplianceEngine:
        compliance_engine = ComplianceEngine()
        logger.info("✅ Compliance Engine initialized")
    else:
        logger.warning("❌ ComplianceEngine class not available")
except Exception as e:
    logger.error("❌ Compliance Engine initialization failed", error=str(e))

try:
    if DocumentProcessor:
        document_processor = DocumentProcessor(rag_pipeline)
        logger.info("✅ Document Processor initialized")
    else:
        logger.warning("❌ DocumentProcessor class not available")
except Exception as e:
    logger.error("❌ Document Processor initialization failed", error=str(e))

try:
    if ExternalAPIManager:
        external_api_manager = ExternalAPIManager(rag_pipeline)
        logger.info("✅ External API Manager initialized")
    else:
        logger.warning("❌ ExternalAPIManager class not available")
except Exception as e:
    logger.error("❌ External API Manager initialization failed", error=str(e))

try:
    if AuthManager:
        auth_manager = AuthManager()
        logger.info("✅ Auth Manager initialized")
    else:
        logger.warning("❌ AuthManager class not available")
except Exception as e:
    logger.error("❌ Auth Manager initialization failed", error=str(e))

try:
    if AuditLogger:
        audit_logger = AuditLogger()
        logger.info("✅ Audit Logger initialized")
    else:
        logger.warning("❌ AuditLogger class not available")
except Exception as e:
    logger.error("❌ Audit Logger initialization failed", error=str(e))

logger.info("Component initialization complete", 
           rag_pipeline_available=rag_pipeline is not None,
           document_processor_available=document_processor is not None)

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

async def search_uploaded_documents(query: str) -> str:
    """Simple text search in uploaded documents"""
    try:
        uploads_dir = Path("data/uploads")
        if not uploads_dir.exists():
            return ""
        
        query_terms = query.lower().split()
        matches = []
        
        for file_path in uploads_dir.glob("*"):
            if file_path.is_file():
                try:
                    # For now, only handle text-based searches in basic text extraction
                    if file_path.suffix.lower() == '.pdf':
                        content = extract_pdf_text_simple(file_path)
                    elif file_path.suffix.lower() in ['.txt', '.csv']:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    else:
                        continue
                    
                    # Simple keyword search
                    content_lower = content.lower()
                    relevant_sections = []
                    
                    for term in query_terms:
                        if term in content_lower:
                            # Find sentences containing the term
                            sentences = re.split(r'[.!?]+', content)
                            for sentence in sentences:
                                if term in sentence.lower() and len(sentence.strip()) > 10:
                                    relevant_sections.append(sentence.strip()[:200] + "...")
                    
                    if relevant_sections:
                        matches.append(f"From {file_path.name}:\n" + "\n".join(relevant_sections[:3]))
                        
                except Exception as e:
                    logger.warning(f"Error searching file {file_path}: {e}")
                    continue
        
        return "\n\n".join(matches[:2]) if matches else ""
        
    except Exception as e:
        logger.error(f"Error in document search: {e}")
        return ""

def extract_pdf_text_simple(file_path: Path) -> str:
    """Simple PDF text extraction without dependencies"""
    try:
        # Try to import PyPDF2 if available
        import PyPDF2
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_content = []
            for page in pdf_reader.pages:
                try:
                    text_content.append(page.extract_text())
                except:
                    continue
            return "\n".join(text_content)
    except ImportError:
        logger.warning("PyPDF2 not available for PDF text extraction")
        return ""
    except Exception as e:
        logger.warning(f"Error extracting PDF text: {e}")
        return ""

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main frontend"""
    try:
        with open("templates/index.html", "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>AAIRE Frontend Not Found</h1><p>Please ensure templates/index.html exists.</p>",
            status_code=404
        )

@app.get("/app", response_class=HTMLResponse)
async def app_page():
    """Alternative route for the frontend"""
    return await root()

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
async def chat(request: ChatRequest):
    """
    Main chat endpoint - MVP-FR-001 through MVP-FR-008
    Implements query processing, retrieval, and response generation
    """
    start_time = datetime.utcnow()
    
    try:
        # For MVP, skip authentication temporarily
        user_id = "demo-user"
        
        # Log query if audit logger is available
        if audit_logger:
            await audit_logger.log_event(
                event="query_submitted",
                user_id=user_id,
                data={"query": request.query, "session_id": request.session_id}
            )
        
        # Check compliance rules if available
        if compliance_engine:
            compliance_result = await compliance_engine.check_query(request.query)
            if compliance_result.blocked:
                if audit_logger:
                    await audit_logger.log_event(
                        event="compliance_triggered",
                        user_id=user_id,
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
        
        # Process query through RAG pipeline if available
        if rag_pipeline:
            rag_response = await rag_pipeline.process_query(
                query=request.query,
                filters=request.filters,
                user_context={}
            )
            
            return ChatResponse(
                response=rag_response.answer,
                citations=rag_response.citations,
                confidence=rag_response.confidence,
                session_id=request.session_id or rag_response.session_id,
                compliance_triggered=False,
                processing_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
            )
        else:
            # Fallback response - try to search uploaded documents
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Simple text search in uploaded documents
            document_matches = await search_uploaded_documents(request.query)
            
            if document_matches:
                response_text = f"Based on your uploaded documents, here's what I found:\n\n{document_matches}\n\nNote: This is a basic text search. For full AI-powered analysis, please configure OpenAI API key."
                citations = ["Uploaded company documents"]
                confidence = 0.7
            else:
                response_text = f"I searched your uploaded documents but couldn't find specific information about '{request.query}'. For full AI-powered analysis of your documents, please configure OpenAI and Pinecone API keys."
                citations = []
                confidence = 0.3
            
            return ChatResponse(
                response=response_text,
                citations=citations,
                confidence=confidence,
                session_id=request.session_id or "fallback",
                compliance_triggered=False,
                processing_time_ms=processing_time
            )
        
    except Exception as e:
        logger.error("Error processing chat request", error=str(e))
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        return ChatResponse(
            response="I apologize, but I'm experiencing technical difficulties. Please try again later.",
            citations=[],
            confidence=0.0,
            session_id=request.session_id or "error",
            compliance_triggered=False,
            processing_time_ms=processing_time
        )

@app.post("/api/v1/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    metadata: str = Form(default='{"title":"unknown","source_type":"COMPANY","effective_date":"2025-01-24"}')
):
    """
    Document upload endpoint - MVP-FR-009 through MVP-FR-012
    """
    logger.info("Upload request received", filename=file.filename, content_type=file.content_type, metadata=metadata)
    try:
        if not document_processor:
            # Fallback for when document processor isn't available
            # Save file to uploads directory
            os.makedirs("data/uploads", exist_ok=True)
            file_path = f"data/uploads/{file.filename}"
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            return DocumentUploadResponse(
                job_id=f"fallback_{datetime.utcnow().timestamp()}",
                status="accepted",
                message=f"Document {file.filename} uploaded successfully (fallback mode)"
            )
        
        # For MVP, use demo user
        user_id = "demo-user"
        
        # Process document upload
        job_id = await document_processor.upload_document(
            file=file,
            metadata=metadata,
            user_id=user_id
        )
        
        if audit_logger:
            await audit_logger.log_event(
                event="document_uploaded",
                user_id=user_id,
                data={"filename": file.filename, "job_id": job_id}
            )
        
        return DocumentUploadResponse(
            job_id=job_id,
            status="accepted",
            message="Document queued for processing"
        )
        
    except Exception as e:
        logger.error("Error uploading document", error=str(e), filename=file.filename, metadata=metadata)
        raise HTTPException(status_code=422, detail=f"Upload failed: {str(e)}")

@app.websocket("/api/v1/chat/ws")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat
    """
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "query":
                query = data.get("message", "")
                session_id = data.get("session_id", "")
                
                try:
                    if rag_pipeline:
                        # Process with RAG pipeline
                        rag_response = await rag_pipeline.process_query(
                            query=query,
                            filters=None,
                            user_context={}
                        )
                        
                        await websocket.send_json({
                            "type": "response",
                            "message": rag_response.answer,
                            "sources": [cite.get("source", "") for cite in rag_response.citations],
                            "confidence": rag_response.confidence
                        })
                    else:
                        # Fallback response with document search
                        document_matches = await search_uploaded_documents(query)
                        
                        if document_matches:
                            message = f"Based on your uploaded documents:\n\n{document_matches}\n\nNote: This is basic text search. Configure OpenAI API key for full AI analysis."
                            sources = ["Uploaded company documents"]
                            confidence = 0.7
                        else:
                            message = f"I searched your uploaded documents but couldn't find information about '{query}'. Configure OpenAI and Pinecone API keys for full AI functionality."
                            sources = []
                            confidence = 0.3
                        
                        await websocket.send_json({
                            "type": "response",
                            "message": message,
                            "sources": sources,
                            "confidence": confidence
                        })
                        
                except Exception as e:
                    logger.error("Error processing WebSocket query", error=str(e))
                    await websocket.send_json({
                        "type": "error",
                        "message": "I apologize, but I'm experiencing technical difficulties. Please try again later."
                    })
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        try:
            await websocket.close()
        except:
            pass

@app.get("/api/v1/documents/{job_id}/status")
async def get_document_status(job_id: str):
    """Get document processing status"""
    if not document_processor:
        raise HTTPException(status_code=503, detail="Document processing service not available")
    
    # For MVP, use demo user
    user_id = "demo-user"
    status = await document_processor.get_status(job_id, user_id)
    return status

@app.get("/api/v1/knowledge/stats")
async def get_knowledge_stats():
    """Get knowledge base statistics"""
    if rag_pipeline:
        return await rag_pipeline.get_stats()
    else:
        return {
            "status": "RAG pipeline not initialized",
            "message": "Please configure OpenAI and Pinecone API keys",
            "components": {
                "rag_pipeline": rag_pipeline is not None,
                "compliance_engine": compliance_engine is not None,
                "document_processor": document_processor is not None,
                "external_api_manager": external_api_manager is not None
            }
        }

@app.get("/api/v1/external/refresh")
async def refresh_external_data():
    """Trigger refresh of external data sources"""
    if not external_api_manager:
        raise HTTPException(status_code=503, detail="External API manager not available")
    
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
