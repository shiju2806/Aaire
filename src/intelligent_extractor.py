"""
Intelligent Document Extraction System
Provides structure-aware document processing for precise information extraction
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger()

class DocumentType(Enum):
    ORGANIZATIONAL_CHART = "org_chart"
    APPROVAL_MATRIX = "approval_matrix"
    FINANCIAL_STRUCTURE = "financial_structure"
    POLICY_DOCUMENT = "policy_document"
    GENERAL = "general"

@dataclass
class ExtractionResult:
    """Result of intelligent document extraction"""
    extracted_data: Dict[str, Any]
    confidence_score: float
    document_type: DocumentType
    extraction_method: str
    warnings: List[str]
    
class IntelligentDocumentExtractor:
    """
    Advanced document extractor that analyzes document structure
    and prevents hallucination through strict extraction rules
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
        self.confidence_threshold = 0.7
        
    async def process_document(self, text: str, query_context: str = "") -> ExtractionResult:
        """Process document with structure-aware extraction"""
        
        # Step 1: Detect document type
        doc_type = self._detect_document_type(text)
        logger.info(f"Detected document type: {doc_type}")
        
        # Step 2: Apply type-specific extraction
        if doc_type == DocumentType.ORGANIZATIONAL_CHART:
            return await self._extract_organizational_data(text, query_context)
        elif doc_type == DocumentType.APPROVAL_MATRIX:
            return await self._extract_approval_matrix(text, query_context)
        elif doc_type == DocumentType.FINANCIAL_STRUCTURE:
            return await self._extract_financial_structure(text, query_context)
        else:
            return await self._extract_general_information(text, query_context)
    
    def _detect_document_type(self, text: str) -> DocumentType:
        """Detect document type based on content patterns"""
        text_lower = text.lower()
        
        # Organizational chart indicators
        org_indicators = [
            'organizational chart', 'org chart', 'reporting structure',
            'reports to', 'manager', 'director', 'supervisor', 'team lead'
        ]
        
        # Approval matrix indicators
        approval_indicators = [
            'approval matrix', 'approval limit', 'authorization limit',
            'spending authority', 'approval threshold', 'signature authority'
        ]
        
        # Financial structure indicators
        financial_indicators = [
            'financial structure', 'finance team', 'accounting structure',
            'cfo', 'treasurer', 'controller', 'financial analyst'
        ]
        
        if any(indicator in text_lower for indicator in org_indicators):
            return DocumentType.ORGANIZATIONAL_CHART
        elif any(indicator in text_lower for indicator in approval_indicators):
            return DocumentType.APPROVAL_MATRIX
        elif any(indicator in text_lower for indicator in financial_indicators):
            return DocumentType.FINANCIAL_STRUCTURE
        else:
            return DocumentType.GENERAL
    
    async def _extract_organizational_data(self, text: str, query_context: str) -> ExtractionResult:
        """Extract organizational chart information"""
        
        extraction_prompt = f"""
You are analyzing an organizational chart document. Extract ONLY the explicitly stated information.

Document text:
{text}

Query context: {query_context}

Rules:
1. ONLY extract job titles and names that are explicitly written in the document
2. Do NOT invent or infer titles that are not clearly stated
3. If a name appears without a title, mark title as "not specified"
4. Do NOT assume hierarchical relationships unless explicitly stated
5. Distinguish between job titles and authority levels

Return a JSON object with:
{{
    "employees": [
        {{"name": "exact name from document", "title": "exact title from document or 'not specified'", "confidence": 0.0-1.0}}
    ],
    "departments": ["only if explicitly mentioned"],
    "reporting_relationships": ["only if explicitly stated"],
    "warnings": ["any ambiguities or unclear information"]
}}

Be extremely conservative. If you're not 100% certain, don't include it.
"""

        try:
            response = await self._query_llm(extraction_prompt)
            result_data = json.loads(response)
            
            # Validate extraction quality
            confidence = self._calculate_confidence(result_data, text)
            warnings = result_data.get('warnings', [])
            
            return ExtractionResult(
                extracted_data=result_data,
                confidence_score=confidence,
                document_type=DocumentType.ORGANIZATIONAL_CHART,
                extraction_method="llm_structured",
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Error in organizational extraction: {e}")
            return self._fallback_extraction(text, DocumentType.ORGANIZATIONAL_CHART)
    
    async def _extract_financial_structure(self, text: str, query_context: str) -> ExtractionResult:
        """Extract financial structure information with emphasis on job titles"""
        
        extraction_prompt = f"""
You are analyzing a financial structure document. Extract ONLY explicitly stated information about job titles and roles.

Document text:
{text}

Query context: {query_context}

Rules:
1. ONLY extract job titles that are explicitly written in the document
2. Do NOT invent variations of titles (e.g., don't assume "Senior Financial Analyst" if only "Financial Analyst" is written)
3. If multiple people have the same title, list them separately
4. Pay special attention to financial roles: CFO, Treasurer, Controller, Financial Analyst, etc.
5. Do NOT make assumptions about responsibilities or reporting structure

Return a JSON object with:
{{
    "financial_roles": [
        {{"name": "exact name", "title": "exact title from document", "department": "if stated", "confidence": 0.0-1.0}}
    ],
    "title_categories": {{"leadership": [], "management": [], "analyst": [], "other": []}},
    "extracted_titles": ["list of all unique titles found"],
    "warnings": ["any unclear or ambiguous information"]
}}

Focus on accuracy over completeness. Better to miss information than to hallucinate it.
"""

        try:
            response = await self._query_llm(extraction_prompt)
            result_data = json.loads(response)
            
            confidence = self._calculate_confidence(result_data, text)
            warnings = result_data.get('warnings', [])
            
            return ExtractionResult(
                extracted_data=result_data,
                confidence_score=confidence,
                document_type=DocumentType.FINANCIAL_STRUCTURE,
                extraction_method="llm_structured",
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Error in financial structure extraction: {e}")
            return self._fallback_extraction(text, DocumentType.FINANCIAL_STRUCTURE)
    
    async def _extract_approval_matrix(self, text: str, query_context: str) -> ExtractionResult:
        """Extract approval matrix information"""
        
        extraction_prompt = f"""
Analyze this approval matrix document and extract ONLY explicitly stated information.

Document text:
{text}

Query context: {query_context}

Extract:
{{
    "approval_levels": [
        {{"title": "exact title", "amount_limit": "if specified", "authority": "if specified"}}
    ],
    "processes": ["only if explicitly described"],
    "thresholds": ["only if explicitly stated"],
    "warnings": ["any ambiguities"]
}}
"""

        try:
            response = await self._query_llm(extraction_prompt)
            result_data = json.loads(response)
            
            confidence = self._calculate_confidence(result_data, text)
            
            return ExtractionResult(
                extracted_data=result_data,
                confidence_score=confidence,
                document_type=DocumentType.APPROVAL_MATRIX,
                extraction_method="llm_structured",
                warnings=result_data.get('warnings', [])
            )
            
        except Exception as e:
            logger.error(f"Error in approval matrix extraction: {e}")
            return self._fallback_extraction(text, DocumentType.APPROVAL_MATRIX)
    
    async def _extract_general_information(self, text: str, query_context: str) -> ExtractionResult:
        """General information extraction"""
        
        extraction_prompt = f"""
Extract information relevant to: {query_context}

Document text:
{text}

Rules:
1. ONLY extract explicitly stated information
2. Do NOT make assumptions or inferences
3. Focus on factual content directly related to the query

Return JSON with extracted information and confidence scores.
"""

        try:
            response = await self._query_llm(extraction_prompt)
            result_data = json.loads(response)
            
            confidence = self._calculate_confidence(result_data, text)
            
            return ExtractionResult(
                extracted_data=result_data,
                confidence_score=confidence,
                document_type=DocumentType.GENERAL,
                extraction_method="llm_general",
                warnings=[]
            )
            
        except Exception as e:
            logger.error(f"Error in general extraction: {e}")
            return self._fallback_extraction(text, DocumentType.GENERAL)
    
    async def _query_llm(self, prompt: str) -> str:
        """Query the LLM with the extraction prompt"""
        try:
            # Use the LlamaIndex OpenAI client
            response = self.llm.complete(prompt)
            return response.text
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            raise
    
    def _calculate_confidence(self, extracted_data: Dict, original_text: str) -> float:
        """Calculate confidence score based on extraction quality"""
        if not extracted_data:
            return 0.0
        
        # Basic confidence scoring
        text_length = len(original_text)
        extraction_depth = len(str(extracted_data))
        
        # More extracted data relative to text length = higher confidence
        confidence = min(extraction_depth / max(text_length * 0.1, 100), 1.0)
        
        # Boost confidence if specific job titles found
        text_lower = original_text.lower()
        job_indicators = ['cfo', 'treasurer', 'controller', 'analyst', 'manager', 'director']
        found_indicators = sum(1 for indicator in job_indicators if indicator in text_lower)
        
        confidence += min(found_indicators * 0.1, 0.3)
        
        return min(confidence, 1.0)
    
    def _fallback_extraction(self, text: str, doc_type: DocumentType) -> ExtractionResult:
        """Fallback extraction when LLM fails"""
        
        # Simple regex-based extraction
        names = re.findall(r'[A-Z][a-z]+ [A-Z][a-z]+', text)
        titles = re.findall(r'(CFO|CEO|CTO|Treasurer|Controller|Manager|Director|Analyst)', text, re.IGNORECASE)
        
        fallback_data = {
            "names_found": list(set(names)),
            "titles_found": list(set(titles)),
            "extraction_method": "regex_fallback"
        }
        
        return ExtractionResult(
            extracted_data=fallback_data,
            confidence_score=0.3,  # Low confidence for fallback
            document_type=doc_type,
            extraction_method="regex_fallback",
            warnings=["LLM extraction failed, using simple pattern matching"]
        )