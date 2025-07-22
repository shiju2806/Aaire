"""
Document Processor for AAIRE - MVP-FR-009 through MVP-FR-012
Handles document upload, processing, and ingestion into RAG pipeline
"""

import os
import uuid
import json
import asyncio
from typing import Dict, Any, List, Optional, BinaryIO
from datetime import datetime
from pathlib import Path
import tempfile
import mimetypes

import structlog
from fastapi import UploadFile, HTTPException
import PyPDF2
from docx import Document as DocxDocument
import pandas as pd

from llama_index.core import Document
from .rag_pipeline import RAGPipeline

logger = structlog.get_logger()

class DocumentProcessor:
    def __init__(self, rag_pipeline: RAGPipeline = None):
        """Initialize document processor"""
        self.rag_pipeline = rag_pipeline
        self.upload_dir = Path("data/uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Document processing status tracking
        self.processing_jobs = {}
        
        # Supported file types and size limits (from MVP config)
        self.supported_formats = {
            'application/pdf': {'extension': '.pdf', 'max_size_mb': 100},
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': {
                'extension': '.docx', 'max_size_mb': 50
            },
            'text/csv': {'extension': '.csv', 'max_size_mb': 25},
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': {
                'extension': '.xlsx', 'max_size_mb': 25
            }
        }
        
        logger.info("Document processor initialized", 
                   upload_dir=str(self.upload_dir),
                   supported_formats=list(self.supported_formats.keys()))
    
    async def upload_document(
        self, 
        file: UploadFile, 
        metadata: str, 
        user_id: str
    ) -> str:
        """
        Upload and queue document for processing - MVP-FR-009
        Returns job_id for tracking
        """
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        try:
            # Parse metadata
            metadata_dict = json.loads(metadata) if metadata else {}
            
            # Validate file
            await self._validate_file(file, metadata_dict)
            
            # Save file temporarily
            file_path = await self._save_uploaded_file(file, job_id)
            
            # Create processing job
            self.processing_jobs[job_id] = {
                'status': 'queued',
                'user_id': user_id,
                'filename': file.filename,
                'file_path': str(file_path),
                'metadata': metadata_dict,
                'created_at': datetime.utcnow().isoformat(),
                'progress': 0
            }
            
            # Queue processing (in production, this would use Celery)
            asyncio.create_task(self._process_document_async(job_id))
            
            logger.info("Document upload queued", 
                       job_id=job_id, 
                       filename=file.filename,
                       user_id=user_id)
            
            return job_id
            
        except Exception as e:
            logger.error("Document upload failed", 
                        error=str(e), 
                        filename=getattr(file, 'filename', 'unknown'))
            raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")
    
    async def _validate_file(self, file: UploadFile, metadata: Dict[str, Any]):
        """Validate uploaded file - MVP-FR-010"""
        
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        # Detect mime type
        mime_type, _ = mimetypes.guess_type(file.filename)
        if not mime_type or mime_type not in self.supported_formats:
            raise ValueError(f"Unsupported file type: {mime_type}")
        
        # Check size limit
        max_size_bytes = self.supported_formats[mime_type]['max_size_mb'] * 1024 * 1024
        if size > max_size_bytes:
            raise ValueError(f"File too large: {size} bytes > {max_size_bytes} bytes")
        
        # Validate required metadata
        required_fields = ['title', 'source_type', 'effective_date']
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Missing required metadata field: {field}")
        
        # Validate source_type
        valid_source_types = ['US_GAAP', 'IFRS', 'COMPANY', 'ACTUARIAL']
        if metadata['source_type'] not in valid_source_types:
            raise ValueError(f"Invalid source_type: {metadata['source_type']}")
    
    async def _save_uploaded_file(self, file: UploadFile, job_id: str) -> Path:
        """Save uploaded file to temporary location"""
        
        # Create safe filename
        extension = Path(file.filename).suffix
        safe_filename = f"{job_id}{extension}"
        file_path = self.upload_dir / safe_filename
        
        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return file_path
    
    async def _process_document_async(self, job_id: str):
        """Process document asynchronously - MVP-FR-011, MVP-FR-012"""
        
        try:
            # Update status
            self.processing_jobs[job_id]['status'] = 'processing'
            self.processing_jobs[job_id]['progress'] = 10
            
            job = self.processing_jobs[job_id]
            file_path = Path(job['file_path'])
            metadata = job['metadata']
            
            # Extract text based on file type
            text_content = await self._extract_text(file_path)
            
            self.processing_jobs[job_id]['progress'] = 40
            
            # Create document with metadata
            document = Document(
                text=text_content,
                metadata={
                    **metadata,
                    'filename': job['filename'],
                    'job_id': job_id,
                    'processed_at': datetime.utcnow().isoformat(),
                    'user_id': job['user_id']
                }
            )
            
            self.processing_jobs[job_id]['progress'] = 70
            
            # Add to RAG pipeline
            if self.rag_pipeline:
                doc_type = self._map_source_type(metadata['source_type'])
                chunks_created = await self.rag_pipeline.add_documents([document], doc_type)
                
                self.processing_jobs[job_id]['chunks_created'] = chunks_created
            
            self.processing_jobs[job_id]['progress'] = 100
            self.processing_jobs[job_id]['status'] = 'completed'
            self.processing_jobs[job_id]['completed_at'] = datetime.utcnow().isoformat()
            
            # Clean up file
            if file_path.exists():
                file_path.unlink()
            
            logger.info("Document processing completed", 
                       job_id=job_id,
                       chunks_created=self.processing_jobs[job_id].get('chunks_created', 0))
            
        except Exception as e:
            self.processing_jobs[job_id]['status'] = 'failed'
            self.processing_jobs[job_id]['error'] = str(e)
            self.processing_jobs[job_id]['failed_at'] = datetime.utcnow().isoformat()
            
            logger.error("Document processing failed", 
                        job_id=job_id, 
                        error=str(e))
    
    async def _extract_text(self, file_path: Path) -> str:
        """Extract text from various file formats"""
        
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.pdf':
                return await self._extract_from_pdf(file_path)
            elif file_extension == '.docx':
                return await self._extract_from_docx(file_path)
            elif file_extension == '.csv':
                return await self._extract_from_csv(file_path)
            elif file_extension == '.xlsx':
                return await self._extract_from_xlsx(file_path)
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")
                
        except Exception as e:
            logger.error("Text extraction failed", 
                        file_path=str(file_path), 
                        error=str(e))
            raise
    
    async def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        text_content = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(f"[Page {page_num + 1}]\n{page_text}")
                except Exception as e:
                    logger.warning("Failed to extract page", 
                                 page_num=page_num, 
                                 error=str(e))
                    continue
        
        return "\n\n".join(text_content)
    
    async def _extract_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        doc = DocxDocument(file_path)
        text_content = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content.append(paragraph.text)
        
        # Extract tables
        for table in doc.tables:
            table_text = []
            for row in table.rows:
                row_text = " | ".join(cell.text for cell in row.cells)
                table_text.append(row_text)
            
            if table_text:
                text_content.append(f"[TABLE]\n" + "\n".join(table_text))
        
        return "\n\n".join(text_content)
    
    async def _extract_from_csv(self, file_path: Path) -> str:
        """Extract text from CSV file"""
        try:
            df = pd.read_csv(file_path)
            
            # Create structured text representation
            text_parts = [
                f"[CSV FILE - {len(df)} rows, {len(df.columns)} columns]",
                f"Columns: {', '.join(df.columns.tolist())}",
                "\nData Summary:",
                str(df.describe(include='all'))
            ]
            
            # Include first few rows as sample
            if len(df) > 0:
                text_parts.append("\nSample Data (first 5 rows):")
                text_parts.append(df.head().to_string())
            
            return "\n".join(text_parts)
            
        except Exception as e:
            logger.error("CSV extraction failed", error=str(e))
            raise
    
    async def _extract_from_xlsx(self, file_path: Path) -> str:
        """Extract text from XLSX file"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            text_parts = [f"[EXCEL FILE - {len(excel_file.sheet_names)} sheets]"]
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                text_parts.append(f"\n[SHEET: {sheet_name}]")
                text_parts.append(f"Dimensions: {len(df)} rows, {len(df.columns)} columns")
                text_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
                
                if len(df) > 0:
                    text_parts.append("Sample Data:")
                    text_parts.append(df.head(3).to_string())
            
            return "\n".join(text_parts)
            
        except Exception as e:
            logger.error("Excel extraction failed", error=str(e))
            raise
    
    def _map_source_type(self, source_type: str) -> str:
        """Map source type to RAG pipeline document type"""
        mapping = {
            'US_GAAP': 'us_gaap',
            'IFRS': 'ifrs',
            'COMPANY': 'company',
            'ACTUARIAL': 'actuarial'
        }
        return mapping.get(source_type, 'company')
    
    async def get_status(self, job_id: str, user_id: str) -> Dict[str, Any]:
        """Get processing status for a job"""
        
        if job_id not in self.processing_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = self.processing_jobs[job_id]
        
        # Check authorization
        if job['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Unauthorized")
        
        return {
            'job_id': job_id,
            'status': job['status'],
            'progress': job['progress'],
            'filename': job['filename'],
            'created_at': job['created_at'],
            'chunks_created': job.get('chunks_created'),
            'error': job.get('error')
        }
    
    def get_job_stats(self) -> Dict[str, Any]:
        """Get overall job processing statistics"""
        
        statuses = {}
        for job in self.processing_jobs.values():
            status = job['status']
            statuses[status] = statuses.get(status, 0) + 1
        
        return {
            'total_jobs': len(self.processing_jobs),
            'status_breakdown': statuses,
            'active_jobs': [
                job_id for job_id, job in self.processing_jobs.items()
                if job['status'] in ['queued', 'processing']
            ]
        }