"""
Document Processing Adapter
Replaces the old SmartMetadataAnalyzer with new extraction system integration
Provides optimized LLM calls and proper metadata extraction
"""

import time
from typing import Dict, Any, List, Optional
import structlog

from .smart_router import SmartExtractionRouter
from .registry import ExtractorRegistry
from .fingerprinting import DocumentFingerprinter
from .metadata_builder import MetadataBuilder, create_entity_enriched_metadata
from .repository import DocumentRepository
from .models import MetadataContext, DocumentFingerprint
from .framework_detector import get_framework_detector

logger = structlog.get_logger()


class DocumentProcessingAdapter:
    """
    Adapter that integrates the new extraction system into document processing pipeline
    Replaces SmartMetadataAnalyzer with optimized LLM calls and proper entity extraction
    """

    def __init__(self, qdrant_client=None, llm_client=None):
        """Initialize with extraction system components"""

        # Initialize new extraction system
        self.registry = ExtractorRegistry()
        self.router = SmartExtractionRouter(
            extractor_registry=self.registry,
            routing_config={
                'cache_enabled': True,
                'pattern_matching_threshold': 0.8,
                'light_llm_threshold': 0.6,
                'full_llm_threshold': 0.4,
                'fallback_enabled': True,
                'cache_ttl_hours': 24,
                'circuit_breaker': {
                    'failure_threshold': 5,
                    'timeout_seconds': 60
                }
            }
        )
        self.fingerprinter = DocumentFingerprinter()
        self.framework_detector = get_framework_detector()
        self.llm_client = llm_client

        # Legacy compatibility attributes
        self.smart_filtering_enabled = True  # Enable new smart filtering by default

        # Initialize repository if Qdrant available
        self.repository = None
        if qdrant_client:
            self.repository = DocumentRepository(qdrant_client)

        logger.info("Document processing adapter initialized with new extraction system")

    async def extract_document_level_metadata(self, content: str, filename: str, doc_type: str) -> Dict[str, Any]:
        """
        Extract document-level metadata using direct LLM calls (bypass complex routing)
        Implements proper token limit handling and framework detection
        """
        start_time = time.time()

        try:
            # Create document fingerprint
            fingerprint = self.fingerprinter.generate_fingerprint(
                content=content,
                metadata={
                    "filename": filename,
                    "document_type": doc_type,
                    "processing_source": "document_processing_adapter"
                },
                filename=filename
            )

            # Check content size and apply intelligent sampling if needed
            content_for_analysis = self._ensure_content_within_limits(content, filename)

            # Create processing context
            context = MetadataContext(
                document_fingerprint=fingerprint,
                filename=filename,
                tenant_id="rag_pipeline",
                job_id=f"doc_processing_{int(time.time())}",
                upload_timestamp=time.time(),
                processing_config={
                    "source": "document_processing_adapter",
                    "doc_type": doc_type,
                    "filename": filename
                }
            )

            # Use direct LLM call instead of complex routing to avoid token issues
            extraction_result = await self._direct_llm_metadata_extraction(
                content=content_for_analysis,
                filename=filename,
                doc_type=doc_type,
                fingerprint=fingerprint
            )

            # Build rich metadata using new system
            metadata_builder = MetadataBuilder()
            metadata_builder.with_document_info(fingerprint, filename)
            metadata_builder.with_extraction_info(
                document_type=extraction_result.document_type,
                extraction_method=extraction_result.extraction_method.value,
                confidence=extraction_result.confidence
            )
            metadata_builder.with_context_info(context)

            if extraction_result.entities:
                metadata_builder.with_entity_analysis(extraction_result.entities)

            # Add processing metrics
            processing_time = (time.time() - start_time) * 1000
            metadata_builder.with_processing_metrics(
                processing_time_ms=processing_time,
                method_used=extraction_result.extraction_method.value,
                fallback_used='fallback' in extraction_result.extraction_method.value
            )

            if extraction_result.warnings:
                metadata_builder.with_warnings(extraction_result.warnings)

            # Build final metadata
            document_metadata = metadata_builder.build()

            # Convert to dictionary format expected by RAG pipeline
            result = {
                'source_document': filename,
                'primary_framework': document_metadata.primary_framework or 'unknown',
                'frameworks': document_metadata.content_domains or ['unknown'],
                'document_type': document_metadata.document_type or doc_type,
                'content_domain': document_metadata.content_domain or 'general',
                'jurisdiction': document_metadata.jurisdiction,
                'context_tags': document_metadata.content_domains or [],
                'attributes': {
                    'extraction_confidence': extraction_result.confidence,
                    'extraction_method': extraction_result.extraction_method.value,
                    'entities_found': len(extraction_result.entities),
                    'processing_time_ms': processing_time,
                    'circuit_breaker_state': self.router.circuit_breaker.get_state(),
                    'router_stats': self.router.get_routing_stats()
                }
            }

            # Store in repository if available
            if self.repository and extraction_result.success:
                try:
                    # Create embedding for storage (this would typically be done by the RAG pipeline)
                    # For now, we'll skip embedding storage and just log
                    logger.info("Extraction result ready for storage in repository")
                except Exception as e:
                    logger.warning("Failed to store in repository", error=str(e))

            logger.info(
                "Document-level metadata extracted with new system",
                filename=filename,
                primary_framework=result['primary_framework'],
                entities_found=len(extraction_result.entities),
                confidence=extraction_result.confidence,
                processing_time_ms=processing_time,
                extraction_method=extraction_result.extraction_method.value
            )

            return result

        except Exception as e:
            logger.error("New extraction system failed, using fallback", error=str(e))
            return self._create_fallback_metadata(filename, doc_type)

    async def extract_chunk_metadata(self, chunk_content: str, document_metadata: Dict[str, Any], chunk_index: int) -> Dict[str, Any]:
        """
        Extract chunk-specific metadata using new system
        Replaces SmartMetadataAnalyzer.extract_chunk_metadata method
        """
        try:
            # For chunks, we mainly inherit document-level metadata
            # But we can add chunk-specific analysis if needed

            result = document_metadata.copy()
            result.update({
                'chunk_index': chunk_index,
                'chunk_size': len(chunk_content),
                'chunk_processing_timestamp': time.time()
            })

            # If multiple frameworks in document, we could analyze chunk focus here
            frameworks = document_metadata.get('frameworks', [])
            if len(frameworks) > 1 and self.llm_client:
                # Could add chunk-specific framework analysis here
                pass

            return result

        except Exception as e:
            logger.warning("Chunk metadata extraction failed", error=str(e), chunk_index=chunk_index)
            return document_metadata.copy()

    async def analyze_query_intent(self, query: str) -> 'QueryIntent':
        """
        Analyze query intent - for now, delegate to smart metadata analyzer
        TODO: Replace with new system intent analysis
        """
        # For now, create a simple intent response
        # This could be enhanced with the new system later

        from .models import QueryIntent

        # Basic pattern-based intent analysis
        required_filters = {}
        context_tags = []
        confidence = 0.3

        # Use configurable framework detection for query analysis
        framework_match = self.framework_detector.detect_framework(query)

        if framework_match.confidence > 0.7:
            context_tags.append(framework_match.framework)
            required_filters['primary_framework'] = [framework_match.framework, 'comparison', 'mixed']
            confidence = framework_match.confidence

        return QueryIntent(
            content_domains=['regulatory'] if context_tags else [],
            context_tags=context_tags,
            required_filters=required_filters,
            excluded_filters={},
            confidence=confidence,
            reasoning=f"Basic pattern matching on query: {query[:50]}"
        )

    def _create_fallback_metadata(self, filename: str, doc_type: str) -> Dict[str, Any]:
        """Create fallback metadata when extraction fails"""
        return {
            'source_document': filename,
            'filename': filename,  # CRITICAL: Preserve filename for citations
            'primary_framework': 'unknown',
            'frameworks': ['unknown'],
            'document_type': doc_type,
            'content_domain': 'general',
            'jurisdiction': None,
            'context_tags': [],
            'attributes': {
                'extraction_confidence': 0.0,
                'extraction_method': 'fallback',
                'entities_found': 0,
                'processing_time_ms': 0.0,
                'fallback_used': True
            }
        }

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction system performance statistics"""
        return {
            'router_stats': self.router.get_routing_stats(),
            'circuit_breaker_state': self.router.circuit_breaker.get_state(),
            'registry_info': self.registry.health_check(),
            'cache_size': self.router.fingerprint_cache.size() if hasattr(self.router, 'fingerprint_cache') else 0
        }

    def create_metadata_dict(self, metadata) -> Dict[str, Any]:
        """
        Convert metadata to dictionary for storage
        Backward compatibility method for legacy interface
        """
        # Handle if metadata is already a dictionary (from our new system)
        if isinstance(metadata, dict):
            return metadata

        # Handle if metadata is a LegacyDocumentMetadata object
        if hasattr(metadata, 'source_document'):
            result = {
                'source_document': metadata.source_document,
                'content_domain': getattr(metadata, 'content_domain', 'general'),
                'context_tags': getattr(metadata, 'context_tags', []),
                'language': getattr(metadata, 'language', 'en')
            }

            # Add optional fields if present
            if hasattr(metadata, 'content_type') and metadata.content_type:
                result['content_type'] = metadata.content_type
            if hasattr(metadata, 'jurisdiction') and metadata.jurisdiction:
                result['jurisdiction'] = metadata.jurisdiction
            if hasattr(metadata, 'framework') and metadata.framework:
                result['framework'] = metadata.framework
            if hasattr(metadata, 'version') and metadata.version:
                result['version'] = metadata.version

            # Add custom attributes
            if hasattr(metadata, 'attributes') and metadata.attributes:
                result.update(metadata.attributes)

            return result

        # Fallback: return as dictionary
        return dict(metadata) if metadata else {}

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the new extraction system"""
        try:
            router_health = self.router.health_check()
            registry_health = self.registry.health_check()

            return {
                'status': 'healthy' if router_health.get('status') == 'healthy' and registry_health.get('status') == 'healthy' else 'degraded',
                'router': router_health,
                'registry': registry_health,
                'repository': self.repository.health_check() if self.repository else {'status': 'not_configured'}
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

    def _ensure_content_within_limits(self, content: str, filename: str) -> str:
        """
        Ensure content is within OpenAI token limits (200K/min for gpt-4o-mini)
        Apply hierarchical sampling if content is too large
        """
        # Target ~30K tokens to stay well under 200K/min rate limit
        TARGET_TOKENS = 30000
        CHARS_PER_TOKEN = 4  # Approximate characters per token
        TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN

        if len(content) <= TARGET_CHARS:
            logger.debug(f"Content size OK for {filename}: {len(content)} chars")
            return content

        logger.info(f"Content too large for {filename}: {len(content)} chars, applying hierarchical sampling")

        # Use the same hierarchical sampling logic from RAG pipeline
        import hashlib
        import random

        # Document-specific seed to prevent cross-contamination
        doc_hash = hashlib.md5(f"{filename}_{len(content)}".encode()).hexdigest()
        doc_seed = int(doc_hash[:8], 16) % 2147483647
        random.seed(doc_seed)

        # Use configurable framework detection for targeted sampling
        detected_framework = self.framework_detector.detect_framework(content, filename)

        # Get framework configuration for enhanced sampling
        framework_config = self.framework_detector.get_framework_config(detected_framework.framework)
        if framework_config:
            framework_keywords = {
                detected_framework.framework: framework_config.get('keywords', []) + framework_config.get('content_indicators', [])
            }
        else:
            # Fallback to basic sampling without framework-specific targeting
            framework_keywords = {}

        # Find framework-specific sections
        content_lower = content.lower()
        detected_frameworks = []
        for framework, keywords in framework_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                detected_frameworks.append(framework)

        # Strategic sampling: beginning + targeted middle + end
        total_chars = len(content)
        sections = []

        # Always include beginning (15% or up to 15K chars)
        beginning_size = min(int(total_chars * 0.15), 15000)
        sections.append(content[:beginning_size])

        # Always include end (10% or up to 10K chars)
        end_size = min(int(total_chars * 0.10), 10000)
        sections.append(content[-end_size:])

        # Framework-targeted middle sampling
        remaining_budget = TARGET_CHARS - beginning_size - end_size
        if remaining_budget > 1000 and detected_frameworks:
            middle_start = beginning_size
            middle_end = total_chars - end_size
            middle_content = content[middle_start:middle_end]

            # Sample framework-relevant sections
            framework_sections = []
            for framework in detected_frameworks[:2]:  # Limit to top 2 frameworks
                keywords = framework_keywords[framework]
                for keyword in keywords[:3]:  # Top 3 keywords per framework
                    keyword_pos = middle_content.lower().find(keyword)
                    if keyword_pos != -1:
                        # Extract 2000 chars around keyword
                        section_start = max(0, keyword_pos - 1000)
                        section_end = min(len(middle_content), keyword_pos + 1000)
                        framework_sections.append(middle_content[section_start:section_end])

            # Randomly sample from framework sections
            if framework_sections:
                random.shuffle(framework_sections)
                current_size = 0
                for section in framework_sections:
                    if current_size + len(section) <= remaining_budget:
                        sections.append(section)
                        current_size += len(section)
                    else:
                        # Add partial section to fill remaining budget
                        remaining = remaining_budget - current_size
                        if remaining > 500:  # Only add if meaningful size
                            sections.append(section[:remaining])
                        break

        # Combine sections
        sampled_content = "\n\n[DOCUMENT SECTION]\n\n".join(sections)

        logger.info(f"Sampled content for {filename}: {len(sampled_content)} chars from {len(content)} chars "
                   f"(frameworks: {detected_frameworks})")

        return sampled_content

    async def _direct_llm_metadata_extraction(self, content: str, filename: str, doc_type: str, fingerprint) -> 'ExtractionResult':
        """
        Direct LLM-based metadata extraction with proper framework detection
        Bypasses complex routing to avoid token limit issues
        """
        from .models import ExtractionResult, ExtractionMethod, ExtractedEntity

        if not self.llm_client:
            logger.warning("No LLM client available, using fallback metadata")
            return ExtractionResult(
                success=True,
                document_type=doc_type or 'unknown',
                extraction_method=ExtractionMethod.FALLBACK,
                entities=[],
                confidence=0.3,
                warnings=['No LLM client available']
            )

        try:
            # Get available frameworks from configuration
            available_frameworks = self.framework_detector.get_all_frameworks()
            frameworks_text = "/".join(available_frameworks)

            # Framework-aware metadata extraction prompt
            prompt = f"""Analyze this document content and extract metadata. Focus on identifying:

1. Primary regulatory/accounting framework ({frameworks_text})
2. Document type and subject matter
3. Key entities, concepts, and domain areas

Document filename: {filename}
Content type: {doc_type}

Content:
{content[:50000]}  # Ensure we don't exceed context limits

Respond with JSON containing:
{{
    "primary_framework": "string ({frameworks_text})",
    "content_domains": ["array of relevant domains"],
    "document_type": "string",
    "confidence": 0.0-1.0,
    "key_concepts": ["array of key concepts found"],
    "summary": "brief document summary"
}}"""

            # Validate LLM client before use
            if not self.llm_client:
                logger.debug("No LLM client available for metadata extraction")
                return {}

            if not hasattr(self.llm_client, 'chat') or not hasattr(self.llm_client.chat, 'completions'):
                logger.warning(
                    "Invalid LLM client for metadata extraction",
                    client_type=type(self.llm_client).__name__ if self.llm_client else "None",
                    has_chat=hasattr(self.llm_client, 'chat') if self.llm_client else False
                )
                return {}

            response = await self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )

            # Parse response
            response_text = response.choices[0].message.content.strip()

            # Try to extract JSON
            import json
            import re

            # Find JSON in response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    metadata = json.loads(json_match.group())
                except json.JSONDecodeError:
                    metadata = self._parse_fallback_metadata(response_text, filename)
            else:
                metadata = self._parse_fallback_metadata(response_text, filename)

            # Create extraction result
            return ExtractionResult(
                success=True,
                document_type=metadata.get('document_type', doc_type or 'unknown'),
                extraction_method=ExtractionMethod.LIGHT_LLM,
                entities=self._create_entities_from_metadata(metadata),
                confidence=metadata.get('confidence', 0.7),
                warnings=[],
                raw_metadata=metadata
            )

        except Exception as e:
            logger.error(f"Direct LLM extraction failed for {filename}: {e}")
            return ExtractionResult(
                success=True,
                document_type=doc_type or 'unknown',
                extraction_method=ExtractionMethod.FALLBACK,
                entities=[],
                confidence=0.3,
                warnings=[f'LLM extraction failed: {str(e)}']
            )

    def _parse_fallback_metadata(self, response_text: str, filename: str) -> Dict[str, Any]:
        """Parse metadata from non-JSON LLM response"""
        # Use configurable framework detection as fallback
        framework_match = self.framework_detector.detect_framework(response_text)
        primary_framework = framework_match.framework

        return {
            'primary_framework': primary_framework,
            'content_domains': [primary_framework] if primary_framework != 'general' else ['general'],
            'document_type': 'unknown',
            'confidence': 0.5,
            'key_concepts': [],
            'summary': f'Document analysis for {filename}'
        }

    def _create_entities_from_metadata(self, metadata: Dict[str, Any]) -> List['ExtractedEntity']:
        """Create entity objects from extracted metadata"""
        from .models import ExtractedEntity

        entities = []

        # Create entities from key concepts
        key_concepts = metadata.get('key_concepts', [])
        for i, concept in enumerate(key_concepts[:10]):  # Limit to 10 concepts
            entities.append(ExtractedEntity(
                name=concept,
                entity_type='concept',
                confidence=metadata.get('confidence', 0.7),
                start_position=0,
                end_position=0,
                context=f"Key concept identified in document analysis"
            ))

        return entities