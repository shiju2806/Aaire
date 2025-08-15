"""
Simplified Intelligent Document Extractor (no spaCy dependency)
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import structlog
from openai import AsyncOpenAI

logger = structlog.get_logger()

@dataclass
class PersonJobTitle:
    """Extracted person and job title information"""
    name: str
    title: str
    department: Optional[str]
    authority_level: Optional[str]
    confidence: float
    context: str
    source_section: str

@dataclass
class ExtractionResult:
    """Result of intelligent extraction"""
    entities: List[PersonJobTitle]
    structure_type: str
    confidence: float
    raw_text: str
    metadata: Dict[str, Any]

class IntelligentDocumentExtractor:
    """Simplified intelligent document analysis"""
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
    
    async def process_document(self, text: str) -> ExtractionResult:
        """Process document with intelligent extraction"""
        try:
            extraction_prompt = f"""Extract ONLY the explicitly stated job titles and names from this document.
Do NOT invent or infer titles that are not clearly written.

Document text:
{text}

Rules:
1. ONLY extract what is explicitly written
2. Do NOT make assumptions or generate content  
3. If a name appears without a title, mark title as "not specified"
4. Distinguish between job titles and authority levels
5. Include the exact text context where each name/title appears

Provide results in JSON format:
{{
    "extractions": [
        {{
            "name": "exact name as written",
            "title": "exact title as written or 'not specified'",
            "authority_level": "if mentioned separately (MVP, VP level, etc.)",
            "department": "if clearly stated",
            "context": "surrounding text where this appears",
            "source_section": "document section where found",
            "confidence": 0.0-1.0
        }}
    ],
    "document_type": "approval_matrix|org_chart|directory|other",
    "extraction_confidence": 0.0-1.0
}}"""
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a precise information extractor. Only extract explicitly stated information. Never invent or assume."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.0,
                max_tokens=2000
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Convert to PersonJobTitle objects
            job_titles = []
            for extraction in result.get("extractions", []):
                job_titles.append(PersonJobTitle(
                    name=extraction.get("name", ""),
                    title=extraction.get("title", "not specified"),
                    department=extraction.get("department"),
                    authority_level=extraction.get("authority_level"),
                    confidence=extraction.get("confidence", 0.5),
                    context=extraction.get("context", ""),
                    source_section=extraction.get("source_section", "")
                ))
            
            return ExtractionResult(
                entities=job_titles,
                structure_type=result.get("document_type", "unknown"),
                confidence=result.get("extraction_confidence", 0.5),
                raw_text=text,
                metadata={
                    "total_found": len(job_titles),
                    "high_confidence_count": len([jt for jt in job_titles if jt.confidence >= 0.8]),
                    "medium_confidence_count": len([jt for jt in job_titles if 0.5 <= jt.confidence < 0.8]),
                    "low_confidence_count": len([jt for jt in job_titles if jt.confidence < 0.5])
                }
            )
            
        except Exception as e:
            logger.error("Document processing failed", error=str(e))
            return ExtractionResult(
                entities=[],
                structure_type="error",
                confidence=0.0,
                raw_text=text,
                metadata={"error": str(e)}
            )
