"""
Base extractor using template method pattern
Eliminates code duplication across different document types
"""

import json
import re
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import structlog

from .models import (
    ExtractedEntity,
    ExtractionResult,
    ExtractionMethod,
    DocumentFingerprint,
    MetadataContext
)

logger = structlog.get_logger()


def strip_markdown_json(response_content: str) -> str:
    """Clean utility function to strip JSON from LLM responses"""
    content = response_content.strip()

    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(json_block_pattern, content, re.DOTALL)

    if match:
        content = match.group(1).strip()
    elif content.startswith('```json') and content.endswith('```'):
        content = content[7:-3].strip()
    elif content.startswith('```') and content.endswith('```'):
        content = content[3:-3].strip()

    return content


class BaseExtractor(ABC):
    """
    Template method pattern for document extraction
    Eliminates duplicate code across extractors
    """

    def __init__(self, extractor_config: Dict[str, Any]):
        self.config = extractor_config
        self.patterns = extractor_config.get('patterns', [])
        self.confidence_threshold = extractor_config.get('confidence_threshold', 0.7)
        self.llm_model = extractor_config.get('llm_model', 'gpt-4o-mini')

    async def extract(
        self,
        content: str,
        document_fingerprint: DocumentFingerprint,
        context: MetadataContext,
        llm_client=None
    ) -> ExtractionResult:
        """
        Template method - same flow for all extractors
        Subclasses implement specific steps
        """
        start_time = time.time()

        try:
            # Step 1: Preprocess content (common preprocessing)
            preprocessed_content = self.preprocess_content(content)

            # Step 2: Extract entities (specific to each extractor)
            entities = await self.extract_entities(
                preprocessed_content,
                context,
                llm_client
            )

            # Step 3: Post-process entities (common validation)
            validated_entities = self.validate_entities(entities, preprocessed_content)

            # Step 4: Calculate confidence (can be overridden)
            confidence = self.calculate_confidence(validated_entities, preprocessed_content)

            # Step 5: Build metadata (specific to each extractor)
            metadata = self.build_extraction_metadata(
                validated_entities,
                confidence,
                context
            )

            processing_time = (time.time() - start_time) * 1000

            result = ExtractionResult(
                success=True,
                document_id=document_fingerprint.document_id,
                extraction_method=self.get_extraction_method(),
                entities=validated_entities,
                document_type=self.get_document_type(),
                confidence=confidence,
                processing_time_ms=processing_time,
                metadata=metadata,
                warnings=self.get_validation_warnings(validated_entities)
            )

            logger.info(
                "Extraction completed successfully",
                extractor=self.__class__.__name__,
                document_id=document_fingerprint.document_id,
                entities_found=len(validated_entities),
                confidence=confidence,
                processing_time_ms=f"{processing_time:.2f}"
            )

            return result

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(
                "Extraction failed",
                extractor=self.__class__.__name__,
                document_id=document_fingerprint.document_id,
                error=str(e),
                processing_time_ms=f"{processing_time:.2f}"
            )

            return self._create_error_result(
                document_fingerprint.document_id,
                str(e),
                processing_time
            )

    def preprocess_content(self, content: str) -> str:
        """Common preprocessing - can be overridden"""
        # Remove excessive whitespace
        preprocessed = re.sub(r'\s+', ' ', content.strip())

        # Normalize line breaks for better parsing
        preprocessed = re.sub(r'\r\n|\r', '\n', preprocessed)

        return preprocessed

    @abstractmethod
    async def extract_entities(
        self,
        content: str,
        context: MetadataContext,
        llm_client=None
    ) -> List[ExtractedEntity]:
        """Each extractor implements this differently"""
        pass

    def validate_entities(
        self,
        entities: List[ExtractedEntity],
        content: str
    ) -> List[ExtractedEntity]:
        """Common entity validation"""
        validated = []

        for entity in entities:
            # Skip entities with no name
            if not entity.name or not entity.name.strip():
                continue

            # Clean name
            entity.name = entity.name.strip()

            # Validate confidence
            if entity.confidence < 0.0:
                entity.confidence = 0.0
            elif entity.confidence > 1.0:
                entity.confidence = 1.0

            # Clean title if present
            if entity.title:
                entity.title = entity.title.strip()
                if entity.title.lower() in ['not specified', 'none', 'n/a', '']:
                    entity.title = None

            # Validate entity appears in content
            if self._entity_appears_in_content(entity, content):
                validated.append(entity)
            else:
                logger.warning(
                    "Entity not found in content, skipping",
                    entity_name=entity.name,
                    entity_title=entity.title
                )

        return validated

    def _entity_appears_in_content(self, entity: ExtractedEntity, content: str) -> bool:
        """Check if entity actually appears in the document content"""
        content_lower = content.lower()

        # Check name appears
        name_words = entity.name.lower().split()
        if len(name_words) >= 2:  # Full names
            if entity.name.lower() not in content_lower:
                return False
        else:  # Single names - be more lenient
            if name_words[0] not in content_lower:
                return False

        # If title is specified, it should also appear
        if entity.title and entity.title.lower() not in content_lower:
            # Allow some flexibility for title variations
            title_lower = entity.title.lower()
            title_variations = [
                title_lower,
                title_lower.replace(' ', ''),
                title_lower.replace('chief', '').strip(),
                title_lower.replace('officer', '').strip()
            ]

            if not any(var in content_lower for var in title_variations if var):
                return False

        return True

    def calculate_confidence(
        self,
        entities: List[ExtractedEntity],
        content: str
    ) -> float:
        """Calculate overall extraction confidence - can be overridden"""
        if not entities:
            return 0.0

        # Average entity confidence
        entity_confidences = [e.confidence for e in entities if e.confidence > 0]
        if not entity_confidences:
            return 0.3  # Low confidence if no individual confidences

        avg_entity_confidence = sum(entity_confidences) / len(entity_confidences)

        # Boost confidence based on number of entities found
        entity_count_boost = min(len(entities) * 0.1, 0.3)

        # Pattern matching boost
        pattern_boost = 0.0
        if self.patterns:
            content_lower = content.lower()
            matching_patterns = sum(1 for pattern in self.patterns if pattern.lower() in content_lower)
            pattern_boost = min(matching_patterns * 0.1, 0.2)

        total_confidence = min(
            avg_entity_confidence + entity_count_boost + pattern_boost,
            1.0
        )

        return round(total_confidence, 3)

    @abstractmethod
    def build_extraction_metadata(
        self,
        entities: List[ExtractedEntity],
        confidence: float,
        context: MetadataContext
    ) -> Dict[str, Any]:
        """Build extractor-specific metadata"""
        pass

    @abstractmethod
    def get_document_type(self) -> str:
        """Return the document type this extractor handles"""
        pass

    @abstractmethod
    def get_extraction_method(self) -> ExtractionMethod:
        """Return the extraction method used"""
        pass

    def get_validation_warnings(self, entities: List[ExtractedEntity]) -> List[str]:
        """Generate validation warnings"""
        warnings = []

        if not entities:
            warnings.append("No entities extracted from document")

        low_confidence_entities = [e for e in entities if e.confidence < 0.5]
        if low_confidence_entities:
            warnings.append(f"{len(low_confidence_entities)} entities have low confidence")

        entities_without_titles = [e for e in entities if not e.title]
        if entities_without_titles:
            warnings.append(f"{len(entities_without_titles)} entities missing titles")

        return warnings

    def can_handle_document(self, content: str) -> float:
        """Determine if this extractor can handle the document"""
        if not self.patterns:
            return 0.5  # Default confidence for general extractors

        content_lower = content.lower()
        matching_patterns = sum(1 for pattern in self.patterns if pattern.lower() in content_lower)

        if matching_patterns == 0:
            return 0.0

        # Confidence based on pattern matches
        confidence = min(matching_patterns / len(self.patterns), 1.0)

        # Boost confidence if multiple patterns match
        if matching_patterns > 1:
            confidence = min(confidence + 0.2, 1.0)

        return confidence

    def _create_error_result(
        self,
        document_id: str,
        error_message: str,
        processing_time: float
    ) -> ExtractionResult:
        """Create error result when extraction fails"""
        return ExtractionResult(
            success=False,
            document_id=document_id,
            extraction_method=ExtractionMethod.FALLBACK,
            entities=[],
            document_type=self.get_document_type(),
            confidence=0.0,
            processing_time_ms=processing_time,
            metadata={'error': error_message},
            warnings=[f"Extraction failed: {error_message}"]
        )

    async def _query_llm_with_retry(
        self,
        llm_client,
        prompt: str,
        max_retries: int = 2
    ) -> Optional[str]:
        """Query LLM with retry logic"""
        if not llm_client:
            logger.debug("No LLM client provided, skipping LLM query")
            return None

        # Validate LLM client has the expected interface
        if not hasattr(llm_client, 'chat') or not hasattr(llm_client.chat, 'completions'):
            logger.warning(
                "Invalid LLM client provided",
                client_type=type(llm_client).__name__,
                has_chat=hasattr(llm_client, 'chat'),
                extractor=self.__class__.__name__
            )
            return None

        for attempt in range(max_retries + 1):
            try:
                response = await llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise information extractor. Only extract explicitly stated information."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=2000
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                logger.warning(
                    f"LLM query attempt {attempt + 1} failed",
                    error=str(e),
                    extractor=self.__class__.__name__
                )
                if attempt == max_retries:
                    raise e
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff

        return None

    def _parse_llm_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM"""
        try:
            cleaned_response = strip_markdown_json(response)

            if not cleaned_response:
                return {}

            return json.loads(cleaned_response)

        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse LLM JSON response",
                error=str(e),
                response_preview=response[:200],
                extractor=self.__class__.__name__
            )
            return {}


# Make asyncio available for retry logic
import asyncio