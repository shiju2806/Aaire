"""
Enhanced Document Processor with Shape-Aware Extraction
Integrates spatial PDF extraction and PowerPoint shape parsing
"""

import os
import structlog
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import asyncio

# Import the existing document processor
try:
    from .document_processor import DocumentProcessor
    from .shape_aware_processor import ShapeAwareProcessor, process_document_shape_aware
    from .rag_pipeline import RAGPipeline
except ImportError:
    # Handle direct imports for testing
    from document_processor import DocumentProcessor
    from shape_aware_processor import ShapeAwareProcessor, process_document_shape_aware
    from rag_pipeline import RAGPipeline

logger = structlog.get_logger()

class EnhancedDocumentProcessor(DocumentProcessor):
    """
    Enhanced document processor that adds shape-aware extraction capabilities
    Extends the existing DocumentProcessor with spatial awareness
    """
    
    def __init__(self, rag_pipeline: RAGPipeline = None):
        """Initialize enhanced document processor with shape-aware capabilities"""
        super().__init__(rag_pipeline)
        
        # Initialize shape-aware processor
        # Use the RAG pipeline's LLM if available
        llm_client = None
        if self.rag_pipeline and hasattr(self.rag_pipeline, 'llm'):
            llm_client = self.rag_pipeline.llm
            logger.info("Shape-aware processor initialized with LLM client")
        else:
            logger.info("Shape-aware processor initialized without LLM (spatial only)")
        
        self.shape_processor = ShapeAwareProcessor(llm_client)
        
        # Track shape-aware extraction statistics
        self.extraction_stats = {
            'spatial_success': 0,
            'spatial_attempts': 0,
            'fallback_used': 0,
            'total_processed': 0
        }
    
    async def _extract_from_pdf(self, file_path: Path) -> str:
        """
        Enhanced PDF extraction with shape-aware spatial processing
        Overrides parent method to add spatial extraction capabilities
        """
        
        logger.info(f"Enhanced PDF extraction starting for: {file_path}")
        self.extraction_stats['total_processed'] += 1
        
        # Step 1: Try shape-aware extraction first
        try:
            self.extraction_stats['spatial_attempts'] += 1
            
            # Process with shape-aware extractor
            shape_result = await self.shape_processor.process_document(
                str(file_path),
                query_context="Extract all organizational information including names, titles, and departments",
                prefer_spatial=True
            )
            
            if shape_result.success and shape_result.organizational_data:
                logger.info(f"Shape-aware extraction successful: {len(shape_result.organizational_data)} units found")
                self.extraction_stats['spatial_success'] += 1
                
                # Convert shape-aware results to text format
                content = self._format_shape_aware_results(shape_result)
                
                # Add metadata about extraction method
                content += f"\n\n[EXTRACTION METADATA]\n"
                content += f"Method: {shape_result.extraction_method}\n"
                content += f"Confidence: {shape_result.confidence:.1%}\n"
                content += f"Personnel Found: {len(shape_result.organizational_data)}\n"
                
                if shape_result.warnings:
                    content += f"Warnings: {', '.join(shape_result.warnings)}\n"
                
                # Log extraction summary
                summary = self.shape_processor.get_processing_summary(shape_result)
                logger.info(f"Shape-aware extraction summary:\n{summary}")
                
                return content
                
            else:
                logger.info("Shape-aware extraction did not produce organizational data, falling back")
                
        except Exception as e:
            logger.warning(f"Shape-aware extraction failed: {e}, falling back to standard extraction")
        
        # Step 2: Fallback to standard PDF extraction
        self.extraction_stats['fallback_used'] += 1
        logger.info("Using standard PDF extraction as fallback")
        
        # Call parent's extraction method
        return await super()._extract_from_pdf(file_path)
    
    def _format_shape_aware_results(self, shape_result) -> str:
        """Format shape-aware extraction results into readable text"""
        
        content_parts = []
        
        # Add header
        content_parts.append("[SHAPE-AWARE ORGANIZATIONAL EXTRACTION]")
        content_parts.append("=" * 50)
        
        # Group by department if available
        dept_groups = {}
        for unit in shape_result.organizational_data:
            dept = unit.get('department', 'Unknown Department')
            if dept not in dept_groups:
                dept_groups[dept] = []
            dept_groups[dept].append(unit)
        
        # Format each department
        for dept, units in dept_groups.items():
            content_parts.append(f"\n{dept}:")
            content_parts.append("-" * len(dept))
            
            for unit in sorted(units, key=lambda x: x.get('confidence', 0), reverse=True):
                name = unit.get('name', 'Unknown')
                title = unit.get('title', 'Unknown Title')
                confidence = unit.get('confidence', 0)
                
                # Format with confidence indicator
                conf_indicator = "***" if confidence > 0.9 else "**" if confidence > 0.7 else "*"
                
                content_parts.append(f"{conf_indicator} {name} - {title}")
                
                # Add source information if available
                if 'cluster_id' in unit:
                    cluster_id = unit['cluster_id']
                    # Extract page number from cluster_id if available
                    if 'page_' in cluster_id:
                        page_num = cluster_id.split('page_')[-1].split('_')[0]
                        content_parts.append(f"   Source: Page {page_num}, {cluster_id}")
                    else:
                        content_parts.append(f"   Source: {cluster_id}")
                
                # Add page information if available
                if 'page' in unit:
                    content_parts.append(f"   Page: {unit['page']}")
                
                # Add warnings if any
                if unit.get('warnings'):
                    for warning in unit['warnings']:
                        content_parts.append(f"   ⚠️ {warning}")
        
        # Add confidence legend
        content_parts.append("\n" + "=" * 50)
        content_parts.append("Confidence: *** High (>90%) | ** Medium (70-90%) | * Lower (<70%)")
        
        return "\n".join(content_parts)
    
    async def _extract_from_pptx(self, file_path: Path) -> str:
        """
        Enhanced PowerPoint extraction with shape-aware processing
        Extends parent method to add shape relationship preservation
        """
        
        logger.info(f"Enhanced PowerPoint extraction starting for: {file_path}")
        self.extraction_stats['total_processed'] += 1
        
        # Step 1: Try shape-aware extraction first
        try:
            self.extraction_stats['spatial_attempts'] += 1
            
            # Process with shape-aware extractor for PowerPoint
            shape_result = await self.shape_processor.process_document(
                str(file_path),
                query_context="Extract all organizational information including names, titles, and departments from PowerPoint",
                prefer_spatial=True
            )
            
            if shape_result.success and shape_result.organizational_data:
                logger.info(f"PowerPoint shape-aware extraction successful: {len(shape_result.organizational_data)} units found")
                self.extraction_stats['spatial_success'] += 1
                
                # Convert shape-aware results to text format
                content = self._format_shape_aware_results(shape_result)
                
                # Add metadata about extraction method
                content += f"\n\n[EXTRACTION METADATA]\n"
                content += f"Method: {shape_result.extraction_method}\n"
                content += f"Confidence: {shape_result.confidence:.1%}\n"
                content += f"Personnel Found: {len(shape_result.organizational_data)}\n"
                content += f"Source: PowerPoint presentation\n"
                
                if shape_result.warnings:
                    content += f"Warnings: {', '.join(shape_result.warnings)}\n"
                
                # Log extraction summary
                summary = self.shape_processor.get_processing_summary(shape_result)
                logger.info(f"PowerPoint shape-aware extraction summary:\n{summary}")
                
                return content
                
            else:
                logger.info("PowerPoint shape-aware extraction did not produce organizational data, falling back")
                
        except Exception as e:
            logger.warning(f"PowerPoint shape-aware extraction failed: {e}, falling back to standard extraction")
        
        # Step 2: Fallback to standard PowerPoint extraction
        self.extraction_stats['fallback_used'] += 1
        logger.info("Using standard PowerPoint extraction as fallback")
        
        # Call parent's extraction method
        content = await super()._extract_from_pptx(file_path)
        
        return content
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get statistics about shape-aware extraction performance"""
        
        stats = self.extraction_stats.copy()
        
        # Calculate success rate
        if stats['spatial_attempts'] > 0:
            stats['spatial_success_rate'] = stats['spatial_success'] / stats['spatial_attempts']
        else:
            stats['spatial_success_rate'] = 0.0
        
        # Calculate fallback rate
        if stats['total_processed'] > 0:
            stats['fallback_rate'] = stats['fallback_used'] / stats['total_processed']
        else:
            stats['fallback_rate'] = 0.0
        
        return stats
    
    async def process_organizational_chart(self, file_path: Path) -> Dict[str, Any]:
        """
        Specialized method for processing organizational charts
        Returns structured data instead of text
        """
        
        logger.info(f"Processing organizational chart: {file_path}")
        
        # Process with shape-aware extractor
        result = await self.shape_processor.process_document(
            str(file_path),
            query_context="Extract complete organizational structure with all personnel",
            prefer_spatial=True
        )
        
        if result.success:
            return {
                'success': True,
                'extraction_method': result.extraction_method,
                'personnel': result.organizational_data,
                'confidence': result.confidence,
                'metadata': result.metadata,
                'warnings': result.warnings
            }
        else:
            return {
                'success': False,
                'error': result.metadata.get('error', 'Unknown error'),
                'personnel': [],
                'warnings': result.warnings
            }

# Factory function to create enhanced processor
def create_enhanced_document_processor(rag_pipeline: RAGPipeline = None) -> EnhancedDocumentProcessor:
    """
    Factory function to create an enhanced document processor
    This maintains compatibility with existing code
    """
    return EnhancedDocumentProcessor(rag_pipeline)