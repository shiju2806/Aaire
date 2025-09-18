"""
Bridge Adapter for Legacy Integration
Provides backward compatibility interface while using new extraction system internally
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import structlog

from .registry import ExtractorRegistry
from .smart_router import SmartExtractionRouter
from .fingerprinting import DocumentFingerprinter
from .models import MetadataContext, DocumentFingerprint, ExtractionResult

logger = structlog.get_logger()


@dataclass
class PersonJobTitle:
    """Legacy interface compatibility"""
    name: str
    title: str
    department: Optional[str]
    authority_level: Optional[str]
    confidence: float
    context: str
    source_section: str


@dataclass
class LegacyExtractionResult:
    """Legacy extraction result format for backward compatibility"""
    entities: List[PersonJobTitle]
    structure_type: str
    confidence: float
    raw_text: str
    metadata: Dict[str, Any]

    @property
    def confidence_score(self) -> float:
        """Backward compatibility property"""
        return self.confidence


class ExtractionBridgeAdapter:
    """
    Bridge adapter that provides legacy IntelligentDocumentExtractor interface
    while using the new clean extraction system internally
    """

    def __init__(self, llm_client):
        """Initialize with LLM client for compatibility"""
        self.llm_client = llm_client

        # Initialize new extraction system
        self.registry = ExtractorRegistry()
        self.router = SmartExtractionRouter(
            extractor_registry=self.registry,
            routing_config={
                'cache_enabled': True,
                'pattern_matching_threshold': 0.8,
                'light_llm_threshold': 0.6,
                'full_llm_threshold': 0.4,
                'fallback_enabled': True
            }
        )
        self.fingerprinter = DocumentFingerprinter()

        logger.info("Bridge adapter initialized with new extraction system")

    async def process_document(self, text: str, query_context: str = "") -> LegacyExtractionResult:
        """
        Legacy interface: process_document
        Routes to new extraction system internally
        """
        try:
            # Create document fingerprint
            filename = f"bridge_document_{hash(text) % 10000}"
            document_fingerprint = self.fingerprinter.generate_fingerprint(
                content=text,
                metadata={
                    "filename": filename,
                    "document_type": "bridge_processed",
                    "processing_source": "bridge_adapter"
                },
                filename=filename
            )

            # Create metadata context
            context = MetadataContext(
                document_fingerprint=document_fingerprint,
                filename=filename,
                tenant_id="bridge_adapter",
                job_id=f"bridge_{hash(query_context) % 10000}",
                upload_timestamp=None,
                processing_config={"source": "bridge_adapter"}
            )

            # Route through new extraction system
            extraction_result = await self.router.route_extraction(
                content=text,
                document_fingerprint=document_fingerprint,
                context=context,
                llm_client=self.llm_client
            )

            # Convert to legacy format
            return self._convert_to_legacy_format(extraction_result, text)

        except Exception as e:
            logger.error("Bridge adapter extraction failed", error=str(e))
            return self._create_fallback_result(text, str(e))

    def _convert_to_legacy_format(self, result: ExtractionResult, text: str) -> LegacyExtractionResult:
        """Convert new extraction result to legacy format"""

        # Convert entities to legacy PersonJobTitle format
        legacy_entities = []
        for entity in result.entities:
            legacy_entities.append(PersonJobTitle(
                name=entity.name or "",
                title=entity.title or "not specified",
                department=entity.department,
                authority_level=entity.authority_level,
                confidence=entity.confidence,
                context=entity.context or "",
                source_section=entity.source_section or ""
            ))

        # Determine structure type from document type
        structure_type_mapping = {
            "organizational_chart": "org_chart",
            "approval_matrix": "approval_matrix",
            "financial_structure": "financial_structure",
            "general": "other"
        }
        structure_type = structure_type_mapping.get(result.document_type, "other")

        return LegacyExtractionResult(
            entities=legacy_entities,
            structure_type=structure_type,
            confidence=result.confidence,
            raw_text=text,
            metadata={
                "total_found": len(legacy_entities),
                "high_confidence_count": len([e for e in legacy_entities if e.confidence >= 0.8]),
                "medium_confidence_count": len([e for e in legacy_entities if 0.5 <= e.confidence < 0.8]),
                "low_confidence_count": len([e for e in legacy_entities if e.confidence < 0.5]),
                "extraction_method": result.extraction_method.value,
                "new_system_metadata": result.metadata
            }
        )

    def _create_fallback_result(self, text: str, error: str) -> LegacyExtractionResult:
        """Create fallback result for errors"""
        return LegacyExtractionResult(
            entities=[],
            structure_type="error",
            confidence=0.0,
            raw_text=text,
            metadata={"error": error, "fallback_used": True}
        )


# Legacy class alias for drop-in replacement
class IntelligentDocumentExtractor(ExtractionBridgeAdapter):
    """Alias for complete backward compatibility"""
    pass