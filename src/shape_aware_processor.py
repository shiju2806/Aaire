"""
Shape-Aware Document Processor
Integrates PDF spatial extraction with existing intelligent extraction system
Provides unified interface for shape-aware document processing
"""

import asyncio
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import structlog
from dataclasses import dataclass

logger = structlog.get_logger()

@dataclass
class ProcessingResult:
    """Unified result structure for shape-aware processing"""
    success: bool
    extraction_method: str
    organizational_data: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]
    warnings: List[str]
    fallback_used: bool = False

class ShapeAwareProcessor:
    """
    Unified processor that combines spatial PDF extraction with intelligent extraction
    Provides fallback mechanisms and confidence-based routing
    """
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.confidence_threshold = 0.7
        self.spatial_confidence_threshold = 0.6
        
    async def process_document(self, 
                             file_path: str, 
                             query_context: str = "",
                             prefer_spatial: bool = True) -> ProcessingResult:
        """
        Process document with shape-aware extraction
        
        Args:
            file_path: Path to document file
            query_context: Context about what information is being sought
            prefer_spatial: Whether to prefer spatial extraction over standard
            
        Returns:
            ProcessingResult with extracted organizational data
        """
        
        logger.info(f"Processing document with shape-aware extraction: {file_path}")
        
        file_path_obj = Path(file_path)
        file_ext = file_path_obj.suffix.lower()
        
        # Route based on file type
        if file_ext == '.pdf':
            return await self._process_pdf(file_path, query_context, prefer_spatial)
        elif file_ext in ['.pptx', '.ppt']:
            return await self._process_powerpoint(file_path, query_context)
        else:
            return await self._process_fallback(file_path, query_context)
    
    async def _process_pdf(self, 
                          file_path: str, 
                          query_context: str,
                          prefer_spatial: bool) -> ProcessingResult:
        """Process PDF with spatial extraction and intelligent fallback"""
        
        spatial_result = None
        intelligent_result = None
        
        # Step 1: Try spatial extraction if preferred
        if prefer_spatial:
            try:
                from .pdf_spatial_extractor import extract_pdf_spatial
                
                logger.info("Attempting PDF spatial extraction")
                spatial_data = await extract_pdf_spatial(file_path)
                
                if spatial_data['success'] and spatial_data['organizational_units']:
                    spatial_confidence = spatial_data['metadata']['average_confidence']
                    
                    if spatial_confidence >= self.spatial_confidence_threshold:
                        logger.info(f"Spatial extraction successful with confidence {spatial_confidence:.3f}")
                        
                        return ProcessingResult(
                            success=True,
                            extraction_method="pdf_spatial",
                            organizational_data=spatial_data['organizational_units'],
                            confidence=spatial_confidence,
                            metadata=spatial_data['metadata'],
                            warnings=[],
                            fallback_used=False
                        )
                    else:
                        logger.info(f"Spatial extraction low confidence ({spatial_confidence:.3f}), will try intelligent extraction")
                        spatial_result = spatial_data
                
            except Exception as e:
                logger.warning(f"Spatial extraction failed: {e}")
        
        # Step 2: Try intelligent extraction as fallback or primary method
        try:
            if self.llm:
                logger.info("Attempting intelligent extraction")
                intelligent_result = await self._intelligent_extraction_fallback(file_path, query_context)
                
                if intelligent_result and intelligent_result.confidence >= self.confidence_threshold:
                    logger.info(f"Intelligent extraction successful with confidence {intelligent_result.confidence:.3f}")
                    
                    return ProcessingResult(
                        success=True,
                        extraction_method="intelligent_llm",
                        organizational_data=self._convert_intelligent_to_org_data(intelligent_result),
                        confidence=intelligent_result.confidence,
                        metadata={"extraction_type": "intelligent_fallback"},
                        warnings=intelligent_result.warnings,
                        fallback_used=not prefer_spatial
                    )
        
        except Exception as e:
            logger.warning(f"Intelligent extraction failed: {e}")
        
        # Step 3: Use best available result or create hybrid
        if spatial_result and spatial_result['organizational_units']:
            logger.info("Using spatial extraction result (lower confidence)")
            return ProcessingResult(
                success=True,
                extraction_method="pdf_spatial_low_conf",
                organizational_data=spatial_result['organizational_units'],
                confidence=spatial_result['metadata']['average_confidence'],
                metadata=spatial_result['metadata'],
                warnings=["Low confidence spatial extraction"],
                fallback_used=True
            )
        
        # Step 4: Complete fallback
        logger.warning("All extraction methods failed, using basic text extraction")
        return await self._basic_text_fallback(file_path)
    
    async def _process_powerpoint(self, file_path: str, query_context: str) -> ProcessingResult:
        """Process PowerPoint files (placeholder for future implementation)"""
        
        logger.info("PowerPoint processing not yet implemented, using basic extraction")
        return await self._basic_text_fallback(file_path)
    
    async def _process_fallback(self, file_path: str, query_context: str) -> ProcessingResult:
        """Process other file types with basic extraction"""
        
        logger.info(f"Using basic text extraction for {file_path}")
        return await self._basic_text_fallback(file_path)
    
    async def _intelligent_extraction_fallback(self, file_path: str, query_context: str):
        """Use existing intelligent extraction as fallback"""
        
        try:
            from .intelligent_extractor import IntelligentDocumentExtractor
            
            # Read file content (simplified - would need proper file reading)
            with open(file_path, 'rb') as f:
                # This is a simplified approach - in production would use proper PDF text extraction
                content = f"Document content from {file_path}"  # Placeholder
            
            extractor = IntelligentDocumentExtractor(self.llm)
            result = await extractor.process_document(content, query_context)
            
            return result
            
        except Exception as e:
            logger.error(f"Intelligent extraction fallback failed: {e}")
            return None
    
    def _convert_intelligent_to_org_data(self, intelligent_result) -> List[Dict[str, Any]]:
        """Convert intelligent extraction result to organizational data format"""
        
        org_data = []
        extracted_data = intelligent_result.extracted_data
        
        # Handle different intelligent extraction formats
        if 'financial_roles' in extracted_data:
            for role in extracted_data['financial_roles']:
                org_data.append({
                    'name': role.get('name', ''),
                    'title': role.get('title', ''),
                    'department': role.get('department', 'Not specified'),
                    'confidence': role.get('confidence', 0.5),
                    'source_box': (0, 0, 0, 0),  # No spatial info from intelligent extraction
                    'cluster_id': 'intelligent_extraction',
                    'warnings': []
                })
        
        elif 'employees' in extracted_data:
            for employee in extracted_data['employees']:
                org_data.append({
                    'name': employee.get('name', ''),
                    'title': employee.get('title', ''),
                    'department': employee.get('department', 'Not specified'),
                    'confidence': employee.get('confidence', 0.5),
                    'source_box': (0, 0, 0, 0),
                    'cluster_id': 'intelligent_extraction',
                    'warnings': []
                })
        
        return org_data
    
    async def _basic_text_fallback(self, file_path: str) -> ProcessingResult:
        """Basic text extraction fallback when all else fails"""
        
        try:
            # Simple text extraction (would be enhanced in production)
            content = f"Basic text content from {file_path}"
            
            return ProcessingResult(
                success=True,
                extraction_method="basic_text",
                organizational_data=[],
                confidence=0.3,
                metadata={"method": "basic_fallback", "file_path": file_path},
                warnings=["Using basic text extraction - no organizational structure detected"],
                fallback_used=True
            )
            
        except Exception as e:
            logger.error(f"Basic text fallback failed: {e}")
            
            return ProcessingResult(
                success=False,
                extraction_method="failed",
                organizational_data=[],
                confidence=0.0,
                metadata={"error": str(e)},
                warnings=["All extraction methods failed"],
                fallback_used=True
            )
    
    def get_processing_summary(self, result: ProcessingResult) -> str:
        """Generate human-readable summary of processing results"""
        
        if not result.success:
            return f"❌ Document processing failed: {result.metadata.get('error', 'Unknown error')}"
        
        method_names = {
            "pdf_spatial": "PDF Spatial Analysis",
            "pdf_spatial_low_conf": "PDF Spatial Analysis (Low Confidence)",
            "intelligent_llm": "AI-Powered Extraction",
            "basic_text": "Basic Text Extraction",
            "failed": "Processing Failed"
        }
        
        method_name = method_names.get(result.extraction_method, result.extraction_method)
        
        summary = f"✅ **{method_name}**\n"
        summary += f"   • Confidence: {result.confidence:.1%}\n"
        summary += f"   • Personnel Found: {len(result.organizational_data)}\n"
        
        if result.fallback_used:
            summary += f"   • ⚠️ Fallback method used\n"
        
        if result.warnings:
            summary += f"   • ⚠️ Warnings: {len(result.warnings)}\n"
        
        # Add sample of extracted data
        if result.organizational_data:
            summary += f"   • 📋 **Sample Results:**\n"
            for i, person in enumerate(result.organizational_data[:3]):
                summary += f"      {i+1}. {person['name']} - {person['title']} ({person['department']})\n"
            
            if len(result.organizational_data) > 3:
                summary += f"      ... and {len(result.organizational_data) - 3} more\n"
        
        return summary

# Convenience function for easy integration
async def process_document_shape_aware(file_path: str, 
                                     query_context: str = "",
                                     llm_client=None,
                                     prefer_spatial: bool = True) -> ProcessingResult:
    """
    Convenience function for shape-aware document processing
    
    Args:
        file_path: Path to document file
        query_context: Context about what information is being sought  
        llm_client: LLM client for intelligent extraction fallback
        prefer_spatial: Whether to prefer spatial extraction
        
    Returns:
        ProcessingResult with extracted organizational data
    """
    processor = ShapeAwareProcessor(llm_client)
    return await processor.process_document(file_path, query_context, prefer_spatial)