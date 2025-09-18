"""
Clean data models for the extraction system
No hard-coded logic, pure data structures
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import time
import uuid


class ExtractionMethod(Enum):
    """Extraction methods in order of preference"""
    CACHE = "cache"
    PATTERN_MATCHING = "pattern_matching"
    LIGHT_LLM = "light_llm"
    FULL_LLM = "full_llm"
    FALLBACK = "fallback"


class ProcessingStage(Enum):
    """Document processing stages"""
    FINGERPRINTING = "fingerprinting"
    DETECTION = "detection"
    EXTRACTION = "extraction"
    VALIDATION = "validation"
    STORAGE = "storage"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExtractedEntity:
    """Generic extracted entity"""
    name: str
    title: Optional[str] = None
    department: Optional[str] = None
    authority_level: Optional[str] = None
    confidence: float = 0.0
    context: str = ""
    source_section: str = ""
    entity_type: str = "person"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result of extraction process"""
    success: bool
    document_id: str
    extraction_method: ExtractionMethod
    entities: List[ExtractedEntity]
    document_type: str
    confidence: float
    processing_time_ms: float
    metadata: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    extraction_timestamp: float = field(default_factory=time.time)


@dataclass
class DocumentFingerprint:
    """Document identification and deduplication"""
    document_id: str
    content_hash: str
    structure_hash: str
    metadata_hash: str
    composite_fingerprint: str
    file_size: int
    creation_timestamp: float = field(default_factory=time.time)


@dataclass
class MetadataContext:
    """Context for metadata extraction"""
    document_fingerprint: DocumentFingerprint
    filename: str
    tenant_id: Optional[str] = None
    job_id: Optional[str] = None
    upload_timestamp: float = field(default_factory=time.time)
    processing_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """Complete processing result"""
    success: bool
    document_id: str
    stage: ProcessingStage
    extraction_result: Optional[ExtractionResult] = None
    error_message: Optional[str] = None
    fallback_used: bool = False
    cache_hit: bool = False
    total_processing_time_ms: float = 0.0
    stage_timings: Dict[str, float] = field(default_factory=dict)


@dataclass
class DocumentMetadata:
    """Clean metadata structure for Qdrant storage"""
    document_id: str
    document_fingerprint: str
    source_document: str
    document_type: str
    primary_framework: Optional[str] = None
    content_domains: List[str] = field(default_factory=list)
    content_domain: str = "general"  # Main content domain for backward compatibility
    jurisdiction: Optional[str] = None  # Jurisdiction for regulatory documents
    extraction_confidence: float = 0.0
    extraction_method: str = "unknown"
    extraction_timestamp: float = field(default_factory=time.time)
    tenant_id: Optional[str] = None
    job_id: Optional[str] = None
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Qdrant storage"""
        return {
            'document_id': self.document_id,
            'document_fingerprint': self.document_fingerprint,
            'source_document': self.source_document,
            'document_type': self.document_type,
            'primary_framework': self.primary_framework,
            'content_domains': self.content_domains,
            'content_domain': self.content_domain,
            'jurisdiction': self.jurisdiction,
            'extraction_confidence': self.extraction_confidence,
            'extraction_method': self.extraction_method,
            'extraction_timestamp': self.extraction_timestamp,
            'tenant_id': self.tenant_id,
            'job_id': self.job_id,
            'version': self.version
        }


@dataclass
class ExtractionConfig:
    """Configuration loaded from YAML"""
    extractors: Dict[str, Dict[str, Any]]
    routing: Dict[str, Any]
    circuit_breaker: Dict[str, Any]
    limits: Dict[str, Any]
    quality: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionConfig':
        """Create from loaded YAML data"""
        extraction_data = data.get('extraction', {})
        return cls(
            extractors=extraction_data.get('extractors', {}),
            routing=extraction_data.get('routing', {}),
            circuit_breaker=extraction_data.get('circuit_breaker', {}),
            limits=extraction_data.get('limits', {}),
            quality=extraction_data.get('quality', {})
        )


@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking"""
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    success_count: int = 0
    total_requests: int = 0


# Legacy compatibility classes for backward compatibility
@dataclass
class QueryIntent:
    """Legacy QueryIntent class for backward compatibility with smart_metadata_analyzer"""
    content_domains: List[str]  # e.g., ["regulatory", "insurance"]
    context_tags: List[str]     # e.g., ["usstat", "reserves"]
    required_filters: Dict[str, Any]  # Must match these
    excluded_filters: Dict[str, Any]  # Must NOT match these
    confidence: float           # 0.0 to 1.0
    reasoning: str              # Why this intent was detected


@dataclass
class LegacyDocumentMetadata:
    """Legacy DocumentMetadata class for backward compatibility with smart_metadata_analyzer"""
    # Core fields (always present)
    source_document: str
    content_domain: str         # regulatory, legal, technical, medical, etc.

    # Dynamic fields (extracted by LLM)
    context_tags: List[str]     # Most flexible - can be anything
    content_type: Optional[str] = None  # contract, regulation, manual, report
    jurisdiction: Optional[str] = None  # us_ca, eu, uk, etc.
    framework: Optional[str] = None     # usstat, ifrs, gaap, etc.
    version: Optional[str] = None       # 2024, v1.28, etc.
    language: str = "en"

    # Open-ended attributes for domain-specific metadata
    attributes: Dict[str, Any] = field(default_factory=dict)