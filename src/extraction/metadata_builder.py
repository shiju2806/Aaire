"""
Metadata Builder Pattern - Clean, fluent interface for metadata construction
Eliminates redundant metadata creation logic
"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import structlog

from .models import DocumentMetadata, ExtractedEntity, DocumentFingerprint, MetadataContext
from .fingerprinting import DocumentFingerprinter

logger = structlog.get_logger()


class MetadataBuilder:
    """
    Fluent interface for building document metadata
    Follows builder pattern for clean, readable code
    """

    def __init__(self):
        self.metadata: Dict[str, Any] = {}
        self.confidence_scores: Dict[str, float] = {}
        self.warnings: List[str] = []

    def with_document_info(
        self,
        document_fingerprint: DocumentFingerprint,
        filename: Optional[str] = None
    ) -> 'MetadataBuilder':
        """Add basic document information"""
        self.metadata.update({
            'document_id': document_fingerprint.document_id,
            'document_fingerprint': document_fingerprint.composite_fingerprint,
            'source_document': filename or document_fingerprint.document_id,
            'file_size': document_fingerprint.file_size,
            'creation_timestamp': document_fingerprint.creation_timestamp
        })
        return self

    def with_extraction_info(
        self,
        document_type: str,
        extraction_method: str,
        confidence: float
    ) -> 'MetadataBuilder':
        """Add extraction-specific information"""
        self.metadata.update({
            'document_type': document_type,
            'extraction_method': extraction_method,
            'extraction_confidence': confidence,
            'extraction_timestamp': time.time()
        })
        self.confidence_scores['extraction'] = confidence
        return self

    def with_context_info(self, context: MetadataContext) -> 'MetadataBuilder':
        """Add context information from processing"""
        if context.tenant_id:
            self.metadata['tenant_id'] = context.tenant_id
        if context.job_id:
            self.metadata['job_id'] = context.job_id

        self.metadata.update({
            'upload_timestamp': context.upload_timestamp,
            'processing_config': context.processing_config
        })
        return self

    def with_entity_analysis(self, entities: List[ExtractedEntity]) -> 'MetadataBuilder':
        """Add analysis based on extracted entities"""
        if not entities:
            self.warnings.append("No entities found in document")
            return self

        # Analyze entity types and patterns
        entity_types = [e.entity_type for e in entities if e.entity_type]
        titles = [e.title for e in entities if e.title]
        departments = [e.department for e in entities if e.department]
        authority_levels = [e.authority_level for e in entities if e.authority_level]

        # Entity statistics
        entity_stats = {
            'total_entities': len(entities),
            'entities_with_titles': len(titles),
            'entities_with_departments': len(departments),
            'entities_with_authority': len(authority_levels),
            'unique_departments': len(set(departments)) if departments else 0,
            'avg_entity_confidence': sum(e.confidence for e in entities) / len(entities)
        }

        self.metadata['entity_analysis'] = entity_stats
        self.confidence_scores['entity_analysis'] = entity_stats['avg_entity_confidence']

        # Detect content domains based on entities
        detected_domains = self._detect_content_domains(entities)
        if detected_domains:
            self.metadata['content_domains'] = detected_domains

        # Detect primary framework if possible
        primary_framework = self._detect_primary_framework(entities, titles)
        if primary_framework:
            self.metadata['primary_framework'] = primary_framework

        return self

    def with_custom_metadata(self, custom_data: Dict[str, Any]) -> 'MetadataBuilder':
        """Add custom metadata fields"""
        # Namespace custom data to avoid conflicts
        namespaced_data = {f"custom_{key}": value for key, value in custom_data.items()}
        self.metadata.update(namespaced_data)
        return self

    def with_quality_scores(
        self,
        text_quality: Optional[float] = None,
        structure_quality: Optional[float] = None,
        completeness: Optional[float] = None
    ) -> 'MetadataBuilder':
        """Add quality assessment scores"""
        quality_data = {}

        if text_quality is not None:
            quality_data['text_quality'] = text_quality
            self.confidence_scores['text_quality'] = text_quality

        if structure_quality is not None:
            quality_data['structure_quality'] = structure_quality
            self.confidence_scores['structure_quality'] = structure_quality

        if completeness is not None:
            quality_data['completeness'] = completeness
            self.confidence_scores['completeness'] = completeness

        if quality_data:
            self.metadata['quality_assessment'] = quality_data

        return self

    def with_processing_metrics(
        self,
        processing_time_ms: float,
        method_used: str,
        fallback_used: bool = False
    ) -> 'MetadataBuilder':
        """Add processing performance metrics"""
        self.metadata['processing_metrics'] = {
            'processing_time_ms': processing_time_ms,
            'method_used': method_used,
            'fallback_used': fallback_used,
            'processing_timestamp': time.time()
        }
        return self

    def with_warnings(self, warnings: List[str]) -> 'MetadataBuilder':
        """Add warnings from processing"""
        self.warnings.extend(warnings)
        return self

    def build(self) -> DocumentMetadata:
        """
        Build the final DocumentMetadata object
        Calculates aggregate confidence and validates required fields
        """
        # Calculate aggregate confidence from all confidence scores
        if self.confidence_scores:
            aggregate_confidence = sum(self.confidence_scores.values()) / len(self.confidence_scores)
            self.metadata['aggregate_confidence'] = round(aggregate_confidence, 3)
        else:
            self.metadata['aggregate_confidence'] = 0.0

        # Add warnings to metadata
        if self.warnings:
            self.metadata['warnings'] = self.warnings

        # Validate required fields
        required_fields = ['document_id', 'document_fingerprint', 'source_document', 'document_type']
        missing_fields = [field for field in required_fields if field not in self.metadata]

        if missing_fields:
            raise ValueError(f"Missing required metadata fields: {missing_fields}")

        # Create DocumentMetadata object
        try:
            # Determine content_domain from content_domains (use first domain or default to general)
            content_domains = self.metadata.get('content_domains', [])
            content_domain = content_domains[0] if content_domains else 'general'

            document_metadata = DocumentMetadata(
                document_id=self.metadata['document_id'],
                document_fingerprint=self.metadata['document_fingerprint'],
                source_document=self.metadata['source_document'],
                document_type=self.metadata['document_type'],
                primary_framework=self.metadata.get('primary_framework'),
                content_domains=content_domains,
                content_domain=content_domain,
                jurisdiction=self.metadata.get('jurisdiction'),
                extraction_confidence=self.metadata.get('extraction_confidence', 0.0),
                extraction_method=self.metadata.get('extraction_method', 'unknown'),
                extraction_timestamp=self.metadata.get('extraction_timestamp', time.time()),
                tenant_id=self.metadata.get('tenant_id'),
                job_id=self.metadata.get('job_id'),
                version=self.metadata.get('version', 1)
            )

            logger.debug(
                "DocumentMetadata built successfully",
                document_id=document_metadata.document_id,
                document_type=document_metadata.document_type,
                confidence=document_metadata.extraction_confidence,
                domains=document_metadata.content_domains,
                warnings_count=len(self.warnings)
            )

            return document_metadata

        except Exception as e:
            logger.error(f"Failed to build DocumentMetadata: {e}")
            raise ValueError(f"Failed to build metadata: {e}")

    def _detect_content_domains(self, entities: List[ExtractedEntity]) -> List[str]:
        """Detect content domains based on entity analysis"""
        domains = set()

        # Domain detection based on entity types and titles
        for entity in entities:
            if not entity.title:
                continue

            title_lower = entity.title.lower()

            # Financial domain indicators
            financial_indicators = ['cfo', 'treasurer', 'controller', 'financial', 'finance', 'accounting']
            if any(indicator in title_lower for indicator in financial_indicators):
                domains.add('financial')

            # Operations domain indicators
            operations_indicators = ['operations', 'coo', 'manager', 'director', 'supervisor']
            if any(indicator in title_lower for indicator in operations_indicators):
                domains.add('operations')

            # Technical domain indicators
            technical_indicators = ['cto', 'technical', 'engineer', 'developer', 'architect']
            if any(indicator in title_lower for indicator in technical_indicators):
                domains.add('technical')

            # Legal domain indicators
            legal_indicators = ['legal', 'counsel', 'attorney', 'compliance']
            if any(indicator in title_lower for indicator in legal_indicators):
                domains.add('legal')

            # HR domain indicators
            hr_indicators = ['hr', 'human resources', 'people', 'talent']
            if any(indicator in title_lower for indicator in hr_indicators):
                domains.add('human_resources')

        # Default to general if no specific domains detected
        if not domains:
            domains.add('general')

        return list(domains)

    def _detect_primary_framework(
        self,
        entities: List[ExtractedEntity],
        titles: List[str]
    ) -> Optional[str]:
        """Detect primary regulatory/business framework"""
        # This is a simplified version - can be enhanced based on requirements
        title_text = ' '.join(titles).lower()

        # Financial frameworks
        if any(term in title_text for term in ['usstat', 'us statutory']):
            return 'usstat'
        elif any(term in title_text for term in ['ifrs', 'international']):
            return 'ifrs'
        elif any(term in title_text for term in ['gaap', 'generally accepted']):
            return 'gaap'

        # Other frameworks can be added here
        return None

    def get_current_metadata(self) -> Dict[str, Any]:
        """Get current metadata state (for debugging/inspection)"""
        return {
            'metadata': self.metadata.copy(),
            'confidence_scores': self.confidence_scores.copy(),
            'warnings': self.warnings.copy()
        }

    def reset(self) -> 'MetadataBuilder':
        """Reset builder to initial state for reuse"""
        self.metadata.clear()
        self.confidence_scores.clear()
        self.warnings.clear()
        return self


# Factory functions for common use cases
def create_basic_metadata(
    document_fingerprint: DocumentFingerprint,
    document_type: str,
    extraction_method: str,
    confidence: float,
    context: MetadataContext,
    filename: Optional[str] = None
) -> DocumentMetadata:
    """Factory function for basic metadata creation"""
    return (MetadataBuilder()
            .with_document_info(document_fingerprint, filename)
            .with_extraction_info(document_type, extraction_method, confidence)
            .with_context_info(context)
            .build())


def create_entity_enriched_metadata(
    document_fingerprint: DocumentFingerprint,
    document_type: str,
    extraction_method: str,
    confidence: float,
    context: MetadataContext,
    entities: List[ExtractedEntity],
    processing_time_ms: float,
    warnings: List[str] = None,
    filename: Optional[str] = None
) -> DocumentMetadata:
    """Factory function for entity-enriched metadata creation"""
    builder = (MetadataBuilder()
               .with_document_info(document_fingerprint, filename)
               .with_extraction_info(document_type, extraction_method, confidence)
               .with_context_info(context)
               .with_entity_analysis(entities)
               .with_processing_metrics(processing_time_ms, extraction_method))

    if warnings:
        builder.with_warnings(warnings)

    return builder.build()


class MetadataValidator:
    """Validator for metadata completeness and consistency"""

    @staticmethod
    def validate_metadata(metadata: DocumentMetadata) -> List[str]:
        """Validate metadata and return list of issues"""
        issues = []

        # Required field validation
        if not metadata.document_id:
            issues.append("Missing document_id")

        if not metadata.document_fingerprint:
            issues.append("Missing document_fingerprint")

        if not metadata.source_document:
            issues.append("Missing source_document")

        # Confidence validation
        if metadata.extraction_confidence < 0 or metadata.extraction_confidence > 1:
            issues.append(f"Invalid extraction_confidence: {metadata.extraction_confidence}")

        # Timestamp validation
        current_time = time.time()
        if metadata.extraction_timestamp > current_time + 3600:  # 1 hour future tolerance
            issues.append("extraction_timestamp appears to be in future")

        # Content domains validation
        if metadata.content_domains and not isinstance(metadata.content_domains, list):
            issues.append("content_domains must be a list")

        return issues

    @staticmethod
    def is_valid(metadata: DocumentMetadata) -> bool:
        """Check if metadata is valid"""
        return len(MetadataValidator.validate_metadata(metadata)) == 0