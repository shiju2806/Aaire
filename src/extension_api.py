"""
Browser Extension API Endpoints
Separate API routes for browser extension functionality
These routes are isolated from main AAIRE functionality to prevent regression
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import uuid
import json
import structlog
from pathlib import Path
import asyncio
from datetime import datetime

try:
    from .enhanced_document_processor import EnhancedDocumentProcessor as DocumentProcessor
except ImportError:
    try:
        from .document_processor import DocumentProcessor
    except ImportError:
        DocumentProcessor = None

try:
    from .rag_pipeline import RAGPipeline
except ImportError:
    RAGPipeline = None

logger = structlog.get_logger()

# Create separate router for extension endpoints
extension_router = APIRouter(prefix="/api/v1/extension", tags=["extension"])

# Track extension upload jobs separately from main system
extension_jobs = {}

@extension_router.post("/upload")
async def extension_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_url: Optional[str] = Form(None),
    page_title: Optional[str] = Form(None),
    extension_version: Optional[str] = Form(None)
):
    """
    Handle file uploads from browser extension
    Separate from main upload endpoint to prevent regression
    """
    
    try:
        # Generate unique job ID for extension uploads
        job_id = str(uuid.uuid4())
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read and validate file content
        content = await file.read()
        file_size = len(content)
        
        # Basic file validation  
        if file_size == 0:
            raise HTTPException(status_code=422, detail="File is empty")
        
        # For text files, be more lenient with small files (VirDocs extractions can be small)
        if file.content_type and 'text' in file.content_type and file_size < 10:
            raise HTTPException(status_code=422, detail="Text file too small (minimum 10 bytes)")
        elif file_size < 5:
            raise HTTPException(status_code=422, detail="File too small")
        
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            raise HTTPException(status_code=422, detail="File too large (max 100MB)")
        
        # Validate filename
        if len(file.filename) > 255:
            raise HTTPException(status_code=422, detail="Filename too long")
        
        # Check for potentially problematic characters
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
        if any(char in file.filename for char in invalid_chars):
            logger.warning("Filename contains potentially problematic characters", filename=file.filename)
        
        # Reset file position for later reading
        await file.seek(0)
        
        # Log first 200 characters of content for debugging (only for text files)
        debug_content = ""
        if file.content_type and 'text' in file.content_type and file_size < 10000:
            try:
                debug_content = content.decode('utf-8')[:200] + ('...' if file_size > 200 else '')
            except:
                debug_content = "Binary content"
        
        logger.info("Extension upload request received", 
                   filename=file.filename,
                   content_type=file.content_type,
                   file_size=file_size,
                   source_url=source_url,
                   page_title=page_title,
                   content_preview=debug_content)
        
        # Create upload metadata with extension context
        upload_metadata = {
            "job_id": job_id,
            "filename": file.filename,
            "source_url": source_url,
            "page_title": page_title,
            "extension_version": extension_version,
            "upload_time": datetime.utcnow().isoformat(),
            "source": "browser_extension",
            "status": "uploading"
        }
        
        # Store job info
        extension_jobs[job_id] = upload_metadata
        
        logger.info(
            "Extension upload started",
            job_id=job_id,
            filename=file.filename,
            source_url=source_url,
            extension_version=extension_version
        )
        
        # Save file to uploads directory (reuse existing infrastructure)
        uploads_dir = Path("data/uploads")
        uploads_dir.mkdir(exist_ok=True)
        
        file_path = uploads_dir / f"{job_id}_{file.filename}"
        
        # Save uploaded file (content already read during validation)
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Update job status
        extension_jobs[job_id]["status"] = "processing"
        extension_jobs[job_id]["file_path"] = str(file_path)
        
        # For now, just mark as completed (processing can be added later)
        extension_jobs[job_id]["status"] = "completed"
        extension_jobs[job_id]["processing_result"] = {
            "success": True,
            "message": "Document uploaded successfully"
        }
        
        return JSONResponse({
            "job_id": job_id,
            "status": "accepted",
            "message": "File uploaded successfully, processing started",
            "metadata": upload_metadata
        })
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error("Extension upload failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@extension_router.get("/status/{job_id}")
async def get_extension_job_status(job_id: str):
    """
    Get processing status for extension upload
    Separate from main status endpoint
    """
    
    if job_id not in extension_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = extension_jobs[job_id]
    
    return JSONResponse({
        "job_id": job_id,
        "status": job_info.get("status", "unknown"),
        "metadata": job_info,
        "progress": job_info.get("progress", {}),
        "error": job_info.get("error")
    })


@extension_router.post("/query")
async def extension_query(
    job_id: str = Form(...),
    query: str = Form(...),
    context_size: Optional[int] = Form(5)
):
    """
    Handle queries for extension-uploaded documents
    Uses same RAG pipeline but with extension context
    """
    
    try:
        if job_id not in extension_jobs:
            raise HTTPException(status_code=404, detail="Document not found")
        
        job_info = extension_jobs[job_id]
        
        if job_info.get("status") != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"Document not ready. Status: {job_info.get('status')}"
            )
        
        if not RAGPipeline:
            raise HTTPException(status_code=503, detail="RAG Pipeline not available")
            
        # Initialize RAG pipeline (reuse existing)
        rag = RAGPipeline()
        
        # Add extension context to query
        enhanced_query = f"""
        Query about document from browser extension:
        Source URL: {job_info.get('source_url', 'Unknown')}
        Page Title: {job_info.get('page_title', 'Unknown')}
        Document: {job_info.get('filename', 'Unknown')}
        
        User Question: {query}
        """
        
        # Process query using existing pipeline
        response = await rag.process_query(
            query=enhanced_query,
            document_ids=[job_id],  # Use job_id as document identifier
            context_size=context_size
        )
        
        logger.info(
            "Extension query processed",
            job_id=job_id,
            query_length=len(query),
            source_url=job_info.get('source_url')
        )
        
        return JSONResponse({
            "job_id": job_id,
            "query": query,
            "response": response,
            "source_context": {
                "url": job_info.get('source_url'),
                "title": job_info.get('page_title'),
                "filename": job_info.get('filename')
            }
        })
        
    except Exception as e:
        logger.error("Extension query failed", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@extension_router.get("/documents")
async def list_extension_documents():
    """
    List all documents uploaded via browser extension
    """
    
    documents = []
    for job_id, job_info in extension_jobs.items():
        if job_info.get("status") == "completed":
            documents.append({
                "job_id": job_id,
                "filename": job_info.get("filename"),
                "source_url": job_info.get("source_url"),
                "page_title": job_info.get("page_title"),
                "upload_time": job_info.get("upload_time"),
                "extension_version": job_info.get("extension_version")
            })
    
    return JSONResponse({
        "documents": documents,
        "total": len(documents)
    })


@extension_router.delete("/document/{job_id}")
async def delete_extension_document(job_id: str):
    """
    Delete extension-uploaded document
    """
    
    if job_id not in extension_jobs:
        raise HTTPException(status_code=404, detail="Document not found")
    
    job_info = extension_jobs[job_id]
    
    # Remove file if it exists
    file_path = job_info.get("file_path")
    if file_path and Path(file_path).exists():
        Path(file_path).unlink()
    
    # Remove from job tracking
    del extension_jobs[job_id]
    
    logger.info("Extension document deleted", job_id=job_id)
    
    return JSONResponse({
        "job_id": job_id,
        "status": "deleted",
        "message": "Document removed successfully"
    })


# TODO: Implement proper document processing
# async def process_extension_document(
#     job_id: str, 
#     file_path: str, 
#     upload_metadata: Dict[str, Any]
# ):
#     """
#     Background task to process extension-uploaded documents
#     Reuses existing document processing pipeline
#     """
#     pass


# Health check endpoint for extension
@extension_router.get("/health")
async def extension_health():
    """
    Health check for extension API
    """
    return JSONResponse({
        "status": "healthy",
        "service": "AAIRE Browser Extension API",
        "active_jobs": len([j for j in extension_jobs.values() if j.get("status") == "processing"]),
        "total_documents": len([j for j in extension_jobs.values() if j.get("status") == "completed"])
    })