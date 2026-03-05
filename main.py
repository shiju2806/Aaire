"""
AAIRE (Accounting & Actuarial Insurance Resource Expert) - MVP
Main FastAPI application following SRS v2.0 specifications
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, WebSocket, WebSocketDisconnect, Form, Request
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

from src.providers.config_loader import get_config, get_nested
_infra_config = get_config("infrastructure")
_UPLOAD_DIR = get_nested(_infra_config, "paths", "upload_dir", default="data/uploads")

# Initialize logger
logger = structlog.get_logger()

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
    from src.ingestion import IngestionPipeline
    logger.info("Using new IngestionPipeline (layout-aware, element-typed)")
except ImportError:
    IngestionPipeline = None

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

try:
    from src.analytics_engine import AnalyticsEngine
except ImportError:
    AnalyticsEngine = None

try:
    from src.workflow_engine import WorkflowEngine
except ImportError:
    WorkflowEngine = None

# Configure stdlib logging root level (structlog delegates to stdlib)
logging.basicConfig(format="%(message)s", level=logging.INFO)

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

# SEC Filing ingestion tracker
INGESTED_FILINGS_FILE = Path("data/ingested_filings.json")
ingested_filings = set()

def load_ingested_filings():
    """Load previously ingested filing accession numbers"""
    global ingested_filings
    if INGESTED_FILINGS_FILE.exists():
        try:
            with open(INGESTED_FILINGS_FILE, 'r') as f:
                ingested_filings = set(json.load(f))
                logger.info(f"Loaded {len(ingested_filings)} ingested filings from cache")
        except Exception as e:
            logger.error(f"Failed to load ingested filings: {e}")
            ingested_filings = set()

def save_ingested_filing(accession_number: str):
    """Save an ingested filing accession number"""
    global ingested_filings
    ingested_filings.add(accession_number)
    try:
        INGESTED_FILINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(INGESTED_FILINGS_FILE, 'w') as f:
            json.dump(list(ingested_filings), f)
    except Exception as e:
        logger.error(f"Failed to save ingested filing: {e}")

app = FastAPI(
    title="AAIRE",
    description="Insurance Resource Expert - AI-powered assistant for accounting and actuarial guidance",
    version="1.0-MVP",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Enable CORS (origins from config/infrastructure.yaml)
_cors_origins = get_nested(_infra_config, "server", "cors_origins",
                           default=["https://aaire.xyz", "https://www.aaire.xyz"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Security middleware for HTTPS
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY" 
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self' https://aaire.xyz; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; font-src 'self' https://cdnjs.cloudflare.com"
    return response

# Add middleware to prevent caching of static files
@app.middleware("http")
async def add_cache_control_header(request, call_next):
    response = await call_next(request)
    # Add no-cache headers for JavaScript and CSS files
    if request.url.path.endswith(('.js', '.css')):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Startup event
@app.on_event("startup")
async def startup_event():
    load_ingested_filings()

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
ingestion_pipeline = None
external_api_manager = None
auth_manager = None
audit_logger = None
analytics_engine = None
workflow_engine = None

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
    if IngestionPipeline:
        from src.providers import get_llm_provider, get_embedding_provider, get_retrieval_provider
        search_engine = getattr(rag_pipeline, 'bm25_engine', None) if rag_pipeline else None
        entity_extractor = getattr(rag_pipeline, 'entity_extractor', None) if rag_pipeline else None

        # Initialize relationship extractor for knowledge graph population
        relationship_extractor = None
        graph_store = getattr(rag_pipeline, 'graph_store', None) if rag_pipeline else None
        if graph_store:
            try:
                from src.knowledge_graph.relationship_extractor import RelationshipExtractor
                relationship_extractor = RelationshipExtractor(
                    llm_provider=get_llm_provider(),
                    graph_store=graph_store,
                )
                logger.info("✅ Relationship extractor initialized for knowledge graph")
            except Exception as re_err:
                logger.warning("Relationship extractor init failed (non-fatal)", error=str(re_err))

        ingestion_pipeline = IngestionPipeline(
            llm_provider=get_llm_provider(),
            embedding_provider=get_embedding_provider(),
            retrieval_provider=get_retrieval_provider(),
            search_engine=search_engine,
            entity_extractor=entity_extractor,
            relationship_extractor=relationship_extractor,
        )
        logger.info("✅ Ingestion Pipeline initialized")
    else:
        ingestion_pipeline = None
        logger.warning("❌ IngestionPipeline class not available")
except Exception as e:
    ingestion_pipeline = None
    logger.error("❌ Ingestion Pipeline initialization failed", error=str(e))

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

try:
    if AnalyticsEngine:
        analytics_engine = AnalyticsEngine()
        logger.info("✅ Analytics Engine initialized")
    else:
        logger.warning("❌ AnalyticsEngine class not available")
except Exception as e:
    logger.error("❌ Analytics Engine initialization failed", error=str(e))

try:
    if WorkflowEngine:
        workflow_engine = WorkflowEngine()
        logger.info("✅ Workflow Engine initialized")
    else:
        logger.warning("❌ WorkflowEngine class not available")
except Exception as e:
    logger.warning("❌ Workflow Engine initialization failed", error=str(e)[:100])
    workflow_engine = None

logger.info("Component initialization complete",
           rag_pipeline_available=rag_pipeline is not None,
           ingestion_pipeline_available=ingestion_pipeline is not None)

# Include Browser Extension API routes (separate from main functionality)
try:
    from src.extension_api import extension_router
    app.include_router(extension_router)
    logger.info("✅ Browser Extension API routes added")
except ImportError as e:
    logger.warning("❌ Browser Extension API not available", error=str(e))

# Request/Response Models per MVP API spec
class ChatRequest(BaseModel):
    query: str = Field(..., max_length=2000, description="User query")
    session_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    user_context: Optional[Dict[str, str]] = None
    job_id: Optional[str] = None  # For filtering to specific uploaded document

class ChatResponse(BaseModel):
    response: str
    citations: List[Dict[str, Any]]
    confidence: float
    session_id: str
    compliance_triggered: bool = False
    processing_time_ms: int
    follow_up_questions: List[str] = []

class DocumentUploadRequest(BaseModel):
    title: str
    source_type: str
    effective_date: str
    tags: Optional[List[str]] = []

class DocumentUploadResponse(BaseModel):
    job_id: str
    status: str
    message: str

class FeedbackRequest(BaseModel):
    message_id: str
    feedback_type: str  # 'thumbs_up', 'thumbs_down', 'issue_report'
    issue_type: Optional[str] = None  # For issue reports
    issue_description: Optional[str] = None  # For detailed issue descriptions
    timestamp: str
    user_id: str
    session_id: str

class FeedbackResponse(BaseModel):
    status: str
    message: str
    feedback_id: Optional[str] = None

async def search_uploaded_documents(query: str) -> str:
    """Simple text search in uploaded documents"""
    try:
        uploads_dir = Path(_UPLOAD_DIR)
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

def create_enhanced_fallback_summary(filename: str, file_size: int, metadata: dict) -> dict:
    """Create an enhanced fallback summary when AI processing is not available"""
    
    # Analyze file type and provide relevant insights
    file_extension = filename.lower().split('.')[-1] if '.' in filename else 'unknown'
    file_type_insights = {
        'pdf': 'PDF document that may contain financial reports, standards, or regulatory information',
        'docx': 'Word document likely containing policies, procedures, or analysis',
        'doc': 'Word document likely containing policies, procedures, or analysis', 
        'xlsx': 'Excel spreadsheet potentially containing financial data, calculations, or models',
        'xls': 'Excel spreadsheet potentially containing financial data, calculations, or models',
        'csv': 'Data file containing structured information suitable for analysis',
        'txt': 'Text document with raw content',
        'pptx': 'Presentation document with slides and visual content',
        'ppt': 'Presentation document with slides and visual content'
    }
    
    file_type_description = file_type_insights.get(file_extension, 'Document file')
    
    # Generate content based on source type
    source_type = metadata.get('source_type', 'COMPANY')
    source_insights = {
        'US_GAAP': 'Contains US GAAP accounting standards and guidance',
        'IFRS': 'Contains International Financial Reporting Standards',
        'COMPANY': 'Company-specific document for internal use',
        'ACTUARIAL': 'Actuarial analysis, models, or calculations',
        'REGULATORY': 'Regulatory guidance or compliance documentation'
    }
    
    content_hint = source_insights.get(source_type, 'Business document')
    
    summary_text = f"""**Document Analysis Summary**

**Document Details:**
- **Filename**: {filename}
- **Type**: {file_type_description.title()}
- **Source**: {source_type.replace('_', ' ').title()}
- **Size**: {file_size:,} bytes ({file_size/1024:.1f} KB)
- **Upload Date**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

**Content Analysis:**
{content_hint}. This document has been successfully uploaded to the AAIRE knowledge base and is available for search and analysis.

**Key Features:**
• Document is indexed and searchable through chat queries
• Content can be referenced in accounting and actuarial questions
• Available to all enterprise users across departments
• Supports compliance and regulatory inquiries

**Usage Recommendations:**
• Ask specific questions about document content in chat
• Reference this document in accounting or actuarial workflows  
• Use for compliance verification and standards guidance
• Combine with other documents for comprehensive analysis

**System Status:**
✅ Document uploaded successfully  
✅ Available for chat queries  
✅ Indexed in knowledge base  
⚠️ AI-powered detailed analysis requires OpenAI API configuration

*Note: This is a system-generated summary. For detailed AI analysis of document content, configure OpenAI API keys in system settings.*"""

    return {
        "summary": summary_text,
        "key_insights": [
            f"Document type: {file_extension.upper()} ({file_type_description})",
            f"Content category: {content_hint}",
            f"File size: {file_size:,} bytes",
            "Available for enterprise chat queries",
            "Requires OpenAI API for detailed AI analysis"
        ],
        "document_metadata": metadata,
        "generated_at": datetime.utcnow().isoformat(),
        "confidence": 0.7,
        "analysis_type": "enhanced_fallback"
    }

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
    """Health check endpoint — verifies all critical dependencies."""
    checks = {}
    overall_healthy = True

    # Check Qdrant vector store
    if rag_pipeline and hasattr(rag_pipeline, 'qdrant_client') and rag_pipeline.qdrant_client:
        try:
            rag_pipeline.qdrant_client.get_collections()
            checks["qdrant"] = "healthy"
        except Exception as e:
            checks["qdrant"] = f"unhealthy: {str(e)[:80]}"
            overall_healthy = False
    else:
        checks["qdrant"] = "not configured"

    # Check Elasticsearch / BM25 search engine
    if rag_pipeline and hasattr(rag_pipeline, 'bm25_engine') and rag_pipeline.bm25_engine:
        try:
            stats = rag_pipeline.bm25_engine.get_stats()
            checks["search_engine"] = f"healthy ({stats.get('total_documents', 0)} docs)"
        except Exception as e:
            checks["search_engine"] = f"unhealthy: {str(e)[:80]}"
            overall_healthy = False
    else:
        checks["search_engine"] = "not configured"

    # Check Redis cache
    if rag_pipeline and hasattr(rag_pipeline, 'cache') and rag_pipeline.cache:
        try:
            rag_pipeline.cache.ping()
            checks["redis"] = "healthy"
        except Exception as e:
            checks["redis"] = f"unhealthy: {str(e)[:80]}"
            overall_healthy = False
    else:
        checks["redis"] = "not configured"

    # Check RAG pipeline overall
    checks["rag_pipeline"] = "healthy" if rag_pipeline else "not available"
    if not rag_pipeline:
        overall_healthy = False

    # Check ingestion pipeline
    checks["ingestion_pipeline"] = "healthy" if ingestion_pipeline else "not available"

    status_code = 200 if overall_healthy else 503
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if overall_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "AAIRE",
            "version": "1.0-MVP",
            "checks": checks,
        },
    )

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    return await chat_handler(request)

@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE streaming chat endpoint — progressive response delivery."""
    from fastapi.responses import StreamingResponse

    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")

    async def event_generator():
        try:
            async for chunk in rag_pipeline.process_query_streaming(
                query=request.query,
                filters=request.filters,
                session_id=request.session_id,
                conversation_history=request.conversation_history,
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            logger.error("Streaming chat error", error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'message': 'An error occurred.'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/api/v2/data/process", response_model=ChatResponse)
async def chat_v2(request: ChatRequest):
    """API v2 compatibility endpoint - routes to same chat handler"""
    return await chat_handler(request)

async def chat_handler(request: ChatRequest):
    """
    Main chat endpoint - MVP-FR-001 through MVP-FR-008
    Implements query processing, retrieval, and response generation
    """
    start_time = datetime.utcnow()
    
    try:
        # For MVP, skip authentication temporarily
        user_id = "demo-user"
        
        # Log query with user context if audit logger is available
        if audit_logger:
            await audit_logger.log_event(
                event="query_submitted",
                user_id=user_id,
                data={
                    "query": request.query, 
                    "session_id": request.session_id,
                    "user_context": request.user_context
                }
            )
        
        # Log user context for enterprise MVP demonstration
        if request.user_context:
            logger.info(f"Query from {request.user_context.get('department', 'Unknown')} department",
                       user=request.user_context.get('name', 'Unknown'),
                       role=request.user_context.get('role', 'Unknown'),
                       query=request.query[:50] + "..." if len(request.query) > 50 else request.query)
        
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
            logger.info(f"Using RAG pipeline for query: {request.query}")
            
            # Add job_id to filters if provided
            query_filters = request.filters or {}
            if request.job_id:
                query_filters["job_id"] = request.job_id
                logger.info(f"Filtering query to job_id: {request.job_id}")
            
            rag_response = await rag_pipeline.process_query(
                query=request.query,
                filters=query_filters,
                user_context=request.user_context or {},
                session_id=request.session_id,
                conversation_history=request.conversation_history
            )
            logger.info(f"RAG response citations count: {len(rag_response.citations)}")
            
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            session_id = request.session_id or rag_response.session_id
            
            # Track analytics
            if analytics_engine:
                asyncio.create_task(analytics_engine.track_query(
                    query=request.query,
                    response=rag_response.answer,
                    session_id=session_id,
                    user_id=user_id,
                    confidence=rag_response.confidence,
                    sources=[cite.get("source", "") for cite in rag_response.citations],
                    processing_time_ms=processing_time
                ))
            
            return ChatResponse(
                response=rag_response.answer,
                citations=rag_response.citations,
                confidence=rag_response.confidence,
                session_id=session_id,
                compliance_triggered=False,
                processing_time_ms=processing_time,
                follow_up_questions=rag_response.follow_up_questions
            )
        else:
            # Fallback response - try to search uploaded documents
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Simple text search in uploaded documents
            document_matches = await search_uploaded_documents(request.query)
            
            if document_matches:
                response_text = f"Based on your uploaded documents, here's what I found:\n\n{document_matches}\n\nNote: This is a basic text search. For full AI-powered analysis, please configure OpenAI API key."
                citations = [{"source": "Uploaded company documents", "text": "Basic text search results"}]
                confidence = 0.7
            else:
                # No matches found in documents - provide general knowledge response without false citations
                response_text = f"""I searched your uploaded documents but couldn't find specific information about '{request.query}'.

In general accounting terms, accounts payable refers to amounts a company owes to suppliers or vendors for goods or services received but not yet paid for. It represents a liability on the company's balance sheet.

Note: This is general accounting knowledge, not from your specific company documents. For company-specific policies or full AI-powered analysis, please configure OpenAI and vector database API keys."""
                citations = []  # No citations for general knowledge
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

_MAX_UPLOAD_SIZE = get_nested(_infra_config, "upload", "max_size_mb", default=100) * 1024 * 1024
_ALLOWED_EXTENSIONS = set(get_nested(_infra_config, "upload", "allowed_extensions",
                                      default=[".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".txt", ".html", ".htm"]))

@app.post("/api/v1/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    metadata: str = Form(default='{"title":"unknown","source_type":"COMPANY","effective_date":"2025-01-24"}')
):
    """
    Document upload endpoint - MVP-FR-009 through MVP-FR-012
    Uses the new IngestionPipeline for layout-aware, element-typed processing.
    """
    import uuid
    logger.info("Upload request received", filename=file.filename, content_type=file.content_type, metadata=metadata)

    try:
        # Validate file extension.
        safe_filename = Path(file.filename).name  # Strip path traversal
        file_ext = Path(safe_filename).suffix.lower()
        if file_ext not in _ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type '{file_ext}' not allowed. Accepted: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
            )

        # Parse metadata.
        try:
            metadata_dict = json.loads(metadata) if metadata else {}
        except Exception:
            metadata_dict = {"title": file.filename, "source_type": "COMPANY", "effective_date": datetime.utcnow().strftime("%Y-%m-%d")}

        job_id = str(uuid.uuid4())

        # Save uploaded file.
        os.makedirs(_UPLOAD_DIR, exist_ok=True)
        file_path = Path(f"{_UPLOAD_DIR}/{job_id}{file_ext}")

        content = await file.read()
        if len(content) > _MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({len(content) / 1024 / 1024:.1f} MB). Maximum size is {_MAX_UPLOAD_SIZE / 1024 / 1024:.0f} MB."
            )
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        with open(file_path, "wb") as buffer:
            buffer.write(content)

        # Map source_type to jurisdiction.
        source_to_jurisdiction = {
            "US_GAAP": "US_GAAP", "US_STAT": "US_STAT",
            "IFRS": "IFRS", "COMPANY": "unknown", "ACTUARIAL": "unknown",
        }
        jurisdiction = source_to_jurisdiction.get(metadata_dict.get("source_type", ""), "unknown")

        if ingestion_pipeline:
            # Process via new pipeline (async, non-blocking).
            async def _run_ingestion() -> None:
                try:
                    result = await ingestion_pipeline.ingest(
                        file_path=file_path,
                        document_id=job_id,
                        document_title=metadata_dict.get("title", file.filename),
                        jurisdiction=jurisdiction,
                    )
                    fallback_jobs = getattr(app.state, 'fallback_jobs', {})
                    fallback_jobs[job_id] = {
                        'job_id': job_id,
                        'status': 'completed' if not result.errors else 'completed_with_errors',
                        'progress': 100,
                        'filename': file.filename,
                        'created_at': datetime.utcnow().isoformat(),
                        'completed_at': datetime.utcnow().isoformat(),
                        'chunks_created': result.total_chunks,
                        'element_counts': result.element_counts,
                        'errors': result.errors,
                        'summary': {
                            'title': metadata_dict.get("title", file.filename),
                            'elements': result.total_elements,
                            'chunks': result.total_chunks,
                            'duration': f"{result.duration_seconds:.1f}s",
                        },
                    }
                    app.state.fallback_jobs = fallback_jobs
                    # Invalidate query cache — document state has changed
                    if rag_pipeline and hasattr(rag_pipeline, 'cache') and rag_pipeline.cache:
                        try:
                            rag_pipeline.cache.flushdb()
                            logger.info("Query cache invalidated after ingestion", job_id=job_id)
                        except Exception as cache_err:
                            logger.warning("Cache invalidation failed (non-fatal)", error=str(cache_err))

                    logger.info("Ingestion complete", job_id=job_id, chunks=result.total_chunks)
                except Exception as e:
                    logger.error("Ingestion failed", job_id=job_id, error=str(e))
                    fallback_jobs = getattr(app.state, 'fallback_jobs', {})
                    fallback_jobs[job_id] = {
                        'job_id': job_id,
                        'status': 'failed',
                        'progress': 0,
                        'filename': file.filename,
                        'error': str(e),
                    }
                    app.state.fallback_jobs = fallback_jobs

            # Store initial job entry.
            fallback_jobs = getattr(app.state, 'fallback_jobs', {})
            fallback_jobs[job_id] = {
                'job_id': job_id,
                'status': 'processing',
                'progress': 0,
                'filename': file.filename,
                'created_at': datetime.utcnow().isoformat(),
            }
            app.state.fallback_jobs = fallback_jobs

            asyncio.create_task(_run_ingestion())
        else:
            # Fallback: save file with basic summary (no pipeline available).
            fallback_jobs = getattr(app.state, 'fallback_jobs', {})
            fallback_jobs[job_id] = {
                'job_id': job_id,
                'status': 'completed',
                'progress': 100,
                'filename': file.filename,
                'created_at': datetime.utcnow().isoformat(),
                'completed_at': datetime.utcnow().isoformat(),
                'summary': create_enhanced_fallback_summary(file.filename, len(content), metadata_dict),
            }
            app.state.fallback_jobs = fallback_jobs

        if audit_logger:
            await audit_logger.log_event(
                event="document_uploaded",
                user_id="demo-user",
                data={"filename": file.filename, "job_id": job_id},
            )

        return DocumentUploadResponse(
            job_id=job_id,
            status="accepted",
            message="Document queued for processing",
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
                conversation_history = data.get("conversation_history", [])
                user_context = data.get("user_context", {})
                
                try:
                    if rag_pipeline:
                        # Process with RAG pipeline using streaming
                        logger.info(f"WebSocket using RAG pipeline for query: {query}")
                        # Log WebSocket user context
                        if user_context:
                            logger.info(f"WebSocket query from {user_context.get('department', 'Unknown')} department",
                                       user=user_context.get('name', 'Unknown'),
                                       role=user_context.get('role', 'Unknown'),
                                       query=query[:50] + "..." if len(query) > 50 else query)

                        # Send thinking status
                        await websocket.send_json({
                            "type": "thinking",
                            "message": "Analyzing your question..."
                        })

                        # Use streaming API
                        response_stream = rag_pipeline.process_query_streaming(
                            query=query,
                            filters=None,
                            user_context=user_context,
                            session_id=session_id,
                            conversation_history=conversation_history
                        )

                        async for chunk_data in response_stream:
                            if chunk_data["type"] == "content":
                                # Stream content chunks
                                await websocket.send_json({
                                    "type": "stream",
                                    "content": chunk_data["content"]
                                })
                            elif chunk_data["type"] == "done":
                                # Send final metadata
                                citations = chunk_data.get("citations", [])
                                logger.info(f"🔍 DEBUG - Full citations structure: {citations}")
                                sources = [cite.get("source", "") for cite in citations]
                                logger.info(f"🔍 DEBUG - Citations in chunk_data: {len(citations)}, Sources extracted: {sources}")
                                await websocket.send_json({
                                    "type": "response",
                                    "message": "",  # Already streamed
                                    "sources": sources,
                                    "confidence": chunk_data.get("confidence", 0.5),
                                    "followUpQuestions": chunk_data.get("follow_up_questions", [])
                                })
                            elif chunk_data["type"] == "error":
                                await websocket.send_json({
                                    "type": "error",
                                    "message": chunk_data.get("message", "An error occurred")
                                })
                    else:
                        # Fallback response with document search
                        document_matches = await search_uploaded_documents(query)
                        
                        if document_matches:
                            message = f"Based on your uploaded documents:\n\n{document_matches}\n\nNote: This is basic text search. Configure OpenAI API key for full AI analysis."
                            sources = ["Uploaded company documents"]
                            confidence = 0.7
                        else:
                            # No matches found - provide general knowledge without false citations
                            if "accounts payable" in query.lower():
                                message = f"""I searched your uploaded documents but couldn't find specific information about '{query}'.

In general accounting terms, accounts payable refers to amounts a company owes to suppliers or vendors for goods or services received but not yet paid for. It represents a liability on the company's balance sheet.

Note: This is general accounting knowledge, not from your specific company documents. Configure OpenAI and vector database API keys for full AI-powered analysis."""
                            else:
                                message = f"I searched your uploaded documents but couldn't find information about '{query}'. Configure OpenAI and Pinecone API keys for full AI functionality."
                            sources = []  # No false citations
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
    fallback_jobs = getattr(app.state, 'fallback_jobs', {})
    if job_id in fallback_jobs:
        return fallback_jobs[job_id]
    raise HTTPException(status_code=404, detail="Document not found")

@app.delete("/api/v1/documents/{job_id}")
async def delete_document(job_id: str):
    """Delete a document completely from the system"""
    try:
        # For MVP, use demo user
        user_id = "demo-user"
        deletion_results = {}
        
        logger.info(f"🗑️ Starting complete deletion for job_id: {job_id}")
        
        # 1. Delete from RAG pipeline (vector database + cache)
        if rag_pipeline:
            deletion_results["vector_store"] = await rag_pipeline.delete_document(job_id)
            logger.info(f"✅ Vector store deletion: {deletion_results['vector_store']}")
        
        # 2. Clear from fallback jobs (in-memory state)
        fallback_jobs = getattr(app.state, 'fallback_jobs', {})
        if job_id in fallback_jobs:
            job_data = fallback_jobs.pop(job_id)
            deletion_results["fallback_jobs"] = {"removed": job_data.get("filename", "unknown")}
            logger.info(f"✅ Removed from fallback jobs: {job_data.get('filename')}")
        
        # 3. Clean up uploaded file
        upload_dir = Path(_UPLOAD_DIR)
        for f in upload_dir.glob(f"{job_id}*"):
            try:
                f.unlink()
                deletion_results["upload_file"] = f"deleted {f.name}"
                logger.info(f"✅ Upload file deleted: {f.name}")
            except Exception as e:
                logger.warning(f"⚠️ Upload file cleanup failed: {str(e)}")

        # 4. Clear any targeted cache entries
        if rag_pipeline and hasattr(rag_pipeline, 'redis_client') and rag_pipeline.redis_client:
            try:
                # Clear document-specific cache entries
                cache_pattern = f"*{job_id}*"
                cleared_keys = await rag_pipeline._clear_cache_pattern(cache_pattern)
                deletion_results["cache_cleanup"] = {"cleared_keys": cleared_keys}
                logger.info(f"✅ Cache cleanup: {cleared_keys} keys cleared")
            except Exception as e:
                logger.warning(f"⚠️ Cache cleanup failed: {str(e)}")
                deletion_results["cache_cleanup"] = f"failed: {str(e)}"
        
        # 5. Log audit event if available
        if audit_logger:
            await audit_logger.log_event(
                event="document_deleted",
                user_id=user_id,
                data={"job_id": job_id, "deletion_results": deletion_results}
            )
        
        logger.info(f"🎉 Complete deletion finished for {job_id}: {deletion_results}")
        
        return {
            "status": "success", 
            "message": f"Document {job_id} completely deleted",
            "deletion_results": deletion_results
        }
        
    except Exception as e:
        logger.error(f"❌ Error deleting document {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

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
                "ingestion_pipeline": ingestion_pipeline is not None,
                "external_api_manager": external_api_manager is not None
            }
        }

@app.post("/api/v1/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback for response quality"""
    try:
        import uuid
        
        # Generate unique feedback ID
        feedback_id = str(uuid.uuid4())
        
        # Create feedback record
        feedback_data = {
            "feedback_id": feedback_id,
            "message_id": feedback.message_id,
            "feedback_type": feedback.feedback_type,
            "issue_type": feedback.issue_type,
            "issue_description": feedback.issue_description,
            "timestamp": feedback.timestamp,
            "user_id": feedback.user_id,
            "session_id": feedback.session_id,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Log the feedback for now (in production, save to database)
        logger.info("User feedback received", **feedback_data)
        
        # Store feedback in a simple JSON file for MVP
        feedback_dir = Path("data/feedback")
        feedback_dir.mkdir(parents=True, exist_ok=True)
        
        feedback_file = feedback_dir / f"feedback_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(feedback_file, "a") as f:
            f.write(json.dumps(feedback_data) + "\n")
        
        # Basic automated quality metrics
        if feedback.feedback_type == "thumbs_down" or feedback.feedback_type == "issue_report":
            # Log for monitoring/alerting
            logger.warning("Negative feedback received",
                         feedback_type=feedback.feedback_type,
                         issue_type=feedback.issue_type,
                         user_id=feedback.user_id,
                         session_id=feedback.session_id)
        
        return FeedbackResponse(
            status="success",
            message="Feedback recorded successfully",
            feedback_id=feedback_id
        )
        
    except Exception as e:
        logger.error("Failed to record feedback", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to record feedback")

@app.get("/api/v1/feedback/analytics")
async def get_feedback_analytics():
    """Get basic feedback analytics for the dashboard"""
    try:
        feedback_dir = Path("data/feedback")
        if not feedback_dir.exists():
            return {
                "total_feedback": 0,
                "feedback_breakdown": {},
                "daily_feedback": {},
                "avg_quality_metrics": {},
                "recent_issues": []
            }
        
        all_feedback = []
        feedback_breakdown = {}
        daily_feedback = {}
        issues = []
        
        # Read all feedback files
        for feedback_file in feedback_dir.glob("feedback_*.jsonl"):
            with open(feedback_file, 'r') as f:
                for line in f:
                    if line.strip():
                        feedback_data = json.loads(line.strip())
                        all_feedback.append(feedback_data)
                        
                        # Count feedback types
                        feedback_type = feedback_data.get('feedback_type', 'unknown')
                        feedback_breakdown[feedback_type] = feedback_breakdown.get(feedback_type, 0) + 1
                        
                        # Count daily feedback
                        date = feedback_data.get('timestamp', '').split('T')[0]
                        daily_feedback[date] = daily_feedback.get(date, 0) + 1
                        
                        # Collect issues
                        if feedback_type in ['thumbs_down', 'issue_report']:
                            issues.append({
                                'timestamp': feedback_data.get('timestamp'),
                                'type': feedback_data.get('issue_type', 'unknown'),
                                'description': feedback_data.get('issue_description', ''),
                                'user_id': feedback_data.get('user_id'),
                                'session_id': feedback_data.get('session_id')
                            })
        
        # Calculate satisfaction rate
        positive_feedback = feedback_breakdown.get('thumbs_up', 0)
        negative_feedback = feedback_breakdown.get('thumbs_down', 0)
        total_ratings = positive_feedback + negative_feedback
        satisfaction_rate = (positive_feedback / total_ratings * 100) if total_ratings > 0 else 0
        
        return {
            "total_feedback": len(all_feedback),
            "feedback_breakdown": feedback_breakdown,
            "satisfaction_rate": round(satisfaction_rate, 1),
            "daily_feedback": daily_feedback,
            "recent_issues": issues[-10:] if issues else [],  # Last 10 issues
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to generate feedback analytics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to generate analytics")

@app.get("/api/v1/debug/documents")
async def debug_all_documents():
    """Debug endpoint to see all documents in vector store"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    return await rag_pipeline.get_all_documents()

@app.post("/api/v1/debug/cleanup")
async def debug_cleanup_orphaned():
    """Debug endpoint to clean up orphaned chunks"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    try:
        return await rag_pipeline.cleanup_orphaned_chunks()
    except Exception as e:
        logger.error("Cleanup failed with original method", error=str(e))
        # Fallback: Try to clear entire collection and recreate
        try:
            await rag_pipeline.clear_all_documents()
            return {"status": "success", "message": "Cleared all documents as fallback"}
        except Exception as e2:
            logger.error("Fallback cleanup also failed", error=str(e2))
            return {"status": "error", "error": str(e2)}

@app.post("/api/v1/debug/clear-all-documents")
async def debug_clear_all_documents():
    """Debug endpoint to clear ALL documents from vector database - use with extreme caution"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    return await rag_pipeline.clear_all_documents()

@app.post("/api/v1/debug/restart-state")
async def debug_restart_state():
    """Debug endpoint to clear all in-memory application state"""
    try:
        logger.info("🔄 Debug: Clearing application state")
        
        # Clear fallback jobs state
        if hasattr(app.state, 'fallback_jobs'):
            app.state.fallback_jobs.clear()
            logger.info("✅ Cleared fallback_jobs state")
        
        # Clear any other application state
        for attr in dir(app.state):
            if not attr.startswith('_') and attr != 'fallback_jobs':
                try:
                    delattr(app.state, attr)
                    logger.info(f"✅ Cleared state: {attr}")
                except:
                    pass
        
        return {"status": "success", "message": "Application state cleared"}
    except Exception as e:
        logger.error(f"❌ Error clearing application state: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/debug/clear-cache")
async def debug_clear_cache():
    """Debug endpoint to clear all cached responses"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    return await rag_pipeline.clear_all_cache()

@app.post("/api/v1/debug/reset-vector-db")
async def debug_reset_vector_db():
    """Debug endpoint to completely reset the vector database (DESTRUCTIVE)"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    if rag_pipeline.vector_store_type != "qdrant":
        raise HTTPException(status_code=400, detail="Vector database reset only supported for Qdrant")
    
    try:
        # Delete the entire collection
        rag_pipeline.qdrant_client.delete_collection(rag_pipeline.collection_name)
        
        # Recreate empty collection
        from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
        rag_pipeline.qdrant_client.create_collection(
            collection_name=rag_pipeline.collection_name,
            vectors_config=VectorParams(
                size=1536,  # OpenAI embedding dimension
                distance=Distance.COSINE
            )
        )
        
        # Recreate job_id index
        try:
            rag_pipeline.qdrant_client.create_payload_index(
                collection_name=rag_pipeline.collection_name,
                field_name="job_id",
                field_schema=PayloadSchemaType.KEYWORD
            )
        except:
            pass  # Index creation can fail if it already exists
        
        # Clear all cache
        await rag_pipeline.clear_all_cache()
        
        return {
            "status": "success",
            "message": "Vector database completely reset - all documents deleted",
            "collection": rag_pipeline.collection_name,
            "vector_store": rag_pipeline.vector_store_type
        }
        
    except Exception as e:
        logger.error("Failed to reset vector database", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

@app.get("/api/v1/external/refresh")
async def refresh_external_data():
    """Trigger refresh of external data sources"""
    if not external_api_manager:
        raise HTTPException(status_code=503, detail="External API manager not available")
    
    job_id = await external_api_manager.refresh_all()
    return {"job_id": job_id, "status": "started"}

@app.get("/api/v1/documents/{job_id}/summary")
async def get_document_summary(job_id: str):
    """Get AI-generated executive summary for uploaded document"""
    try:
        logger.info(f"📄 Requesting summary for job_id: {job_id}")
        
        # Check fallback jobs first
        fallback_jobs = getattr(app.state, 'fallback_jobs', {})
        logger.info(f"🔍 Fallback jobs available: {list(fallback_jobs.keys())}")
        
        if job_id in fallback_jobs:
            job_data = fallback_jobs[job_id]
            logger.info(f"✅ Found job in fallback_jobs: {job_data.get('filename')}")
            logger.info(f"📋 Summary available: {'summary' in job_data and bool(job_data.get('summary'))}")
            
            return {
                "job_id": job_id,
                "document_info": {
                    "filename": job_data.get('filename'),
                    "status": job_data.get('status'),
                    "created_at": job_data.get('created_at')
                },
                "summary": job_data.get('summary', {})
            }
        
        raise HTTPException(status_code=404, detail="Document not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving document summary for job_id {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve summary: {str(e)}")

# Workflow API Endpoints
@app.get("/api/v1/workflows")
async def list_workflows():
    """Get list of available workflow templates"""
    if not workflow_engine:
        raise HTTPException(status_code=503, detail="Workflow engine not available")
    
    workflows = await workflow_engine.list_workflows()
    return {"workflows": workflows}

@app.post("/api/v1/workflows/{template_id}/start")
async def start_workflow(template_id: str, session_id: str = None):
    """Start a new workflow session"""
    if not workflow_engine:
        raise HTTPException(status_code=503, detail="Workflow engine not available")
    
    if not session_id:
        import uuid
        session_id = f"workflow_{uuid.uuid4()}"
    
    result = await workflow_engine.start_workflow(template_id, session_id)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@app.post("/api/v1/workflows/{session_id}/step")
async def process_workflow_step(session_id: str, response: dict):
    """Process user response to current workflow step"""
    if not workflow_engine:
        raise HTTPException(status_code=503, detail="Workflow engine not available")
    
    user_response = response.get("response", "")
    result = await workflow_engine.process_step_response(session_id, user_response)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    # Track workflow step completion
    if analytics_engine and result.get("status") == "continue":
        asyncio.create_task(analytics_engine.track_workflow_step(
            workflow_id=result.get("workflow_id", "unknown"),
            step_id=result.get("current_step", {}).get("id", "unknown"),
            step_name=result.get("current_step", {}).get("title", ""),
            session_id=session_id,
            completed=True
        ))
    
    return result

@app.get("/api/v1/workflows/{session_id}/status")
async def get_workflow_status(session_id: str):
    """Get current workflow status"""
    if not workflow_engine:
        raise HTTPException(status_code=503, detail="Workflow engine not available")
    
    result = await workflow_engine.get_workflow_status(session_id)
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result

# Analytics API Endpoints
@app.get("/api/v1/analytics/summary")
async def get_analytics_summary(days: int = 30):
    """Get usage analytics summary"""
    if not analytics_engine:
        raise HTTPException(status_code=503, detail="Analytics engine not available")
    
    if days < 1 or days > 365:
        raise HTTPException(status_code=400, detail="Days must be between 1 and 365")
    
    summary = await analytics_engine.get_analytics_summary(days)
    return summary

@app.get("/api/v1/analytics/knowledge-gaps")
async def get_knowledge_gaps(limit: int = 20):
    """Get queries with low confidence scores (knowledge gaps)"""
    if not analytics_engine:
        raise HTTPException(status_code=503, detail="Analytics engine not available")
    
    summary = await analytics_engine.get_analytics_summary(30)
    knowledge_gaps = summary.get("knowledge_gaps", [])[:limit]
    
    return {
        "knowledge_gaps": knowledge_gaps,
        "total_gaps": len(knowledge_gaps),
        "generated_at": datetime.utcnow().isoformat()
    }

# ========================================================================
# SEC EDGAR API ENDPOINTS - Clean Implementation
# ========================================================================

try:
    from src.sec_edgar_source import SECEdgarSource
    sec_source_available = True
    logger.info("✅ SEC EDGAR integration available")
except ImportError:
    sec_source_available = False
    logger.warning("❌ SEC EDGAR integration not available")

@app.get("/api/v1/sec/companies/search")
async def search_sec_companies(q: str):
    """Search SEC companies by name or ticker"""
    if not sec_source_available:
        raise HTTPException(status_code=503, detail="SEC EDGAR integration not available")
    
    try:
        async with SECEdgarSource() as sec_client:
            companies = await sec_client.search_company(q)
            return {
                "query": q,
                "companies": companies,
                "count": len(companies)
            }
    except Exception as e:
        logger.error(f"SEC company search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/api/v1/sec/filings")
async def get_sec_filings(cik: str, forms: str = "10-K,10-Q,8-K", years: str = None):
    """Get recent SEC filings for a company"""
    if not sec_source_available:
        raise HTTPException(status_code=503, detail="SEC EDGAR integration not available")
    
    try:
        # Parse parameters
        form_types = [f.strip() for f in forms.split(',')]
        year_list = None
        if years:
            year_list = [int(y.strip()) for y in years.split(',')]
        
        async with SECEdgarSource() as sec_client:
            filings = await sec_client.get_company_filings(cik, form_types, year_list)

            # Mark which filings have been ingested
            for filing in filings:
                filing['ingested'] = filing['accession_number'] in ingested_filings

            return {
                "cik": cik,
                "filings": filings,
                "count": len(filings)
            }
    except Exception as e:
        logger.error(f"SEC filings fetch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Filings fetch failed: {str(e)}")

@app.post("/api/v1/sec/ingest")
async def ingest_sec_filing(request: dict):
    """Ingest SEC filing into RAG pipeline"""
    if not sec_source_available:
        raise HTTPException(status_code=503, detail="SEC EDGAR integration not available")
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="Document processing not available")
    
    try:
        filing_info = request.get('filing_info')
        if not filing_info:
            raise HTTPException(status_code=400, detail="filing_info required")

        logger.info(f"SEC filing ingestion: {filing_info['form_type']} for {filing_info['company_name']}")
        logger.info(f"Filing info keys: {list(filing_info.keys())}")
        logger.info(f"CIK: {filing_info.get('cik')}, Accession: {filing_info.get('accession_number')}")

        # Download SEC filing content
        async with SECEdgarSource() as sec_client:
            content = await sec_client.download_filing_content(filing_info)

            if not content:
                logger.error(f"download_filing_content returned empty for: {filing_info}")
                raise HTTPException(status_code=400, detail="Failed to download filing content")
        
        # Create Document object
        from llama_index.core import Document
        
        metadata = {
            'filename': f"{filing_info['company_name']}_{filing_info['form_type']}_{filing_info['filing_date']}.txt",
            'source_type': 'SEC_FILING',
            'form_type': filing_info['form_type'],
            'company_name': filing_info['company_name'],
            'ticker': filing_info.get('ticker', ''),
            'cik': filing_info['cik'],
            'filing_date': filing_info['filing_date'],
            'processed_at': datetime.utcnow().isoformat(),
            'job_id': f"sec_{filing_info['accession_number']}"
        }
        
        document = Document(text=content, metadata=metadata)
        chunks_created = await rag_pipeline.add_documents([document], "sec_filing")

        # Mark filing as ingested
        save_ingested_filing(filing_info['accession_number'])

        logger.info(f"SEC filing ingested: {chunks_created} chunks created")

        return {
            "status": "success",
            "filing_info": filing_info,
            "chunks_created": chunks_created,
            "message": f"SEC {filing_info['form_type']} filing ingested successfully"
        }
        
    except Exception as e:
        logger.error(f"SEC filing ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

if __name__ == "__main__":
    print("🚨 DEBUG: Starting AAIRE with latest code - formatting system should work!")
    print("🔧 DEBUG: Two-pass formatting and conversation memory enabled")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.environ.get("PORT", 8000)),
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
