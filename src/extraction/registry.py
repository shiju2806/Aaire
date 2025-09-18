"""
Extractor Registry with Factory Pattern
No hard-coded logic - completely configurable
"""

import time
from typing import Dict, Type, Optional, List, Any
from pathlib import Path
import yaml
import structlog

from .models import ExtractionConfig, ExtractionResult, DocumentFingerprint, MetadataContext
from .base_extractor import BaseExtractor
from .extractors import (
    OrganizationalExtractor,
    FinancialExtractor,
    ApprovalMatrixExtractor,
    GeneralExtractor
)

logger = structlog.get_logger()


class ExtractorRegistry:
    """
    Factory pattern for document extractors
    Completely configurable - no hard-coded logic
    """

    def __init__(self, config_path: str = "config/extraction_config.yaml"):
        self.extractors: Dict[str, BaseExtractor] = {}
        self.config: Optional[ExtractionConfig] = None
        self._load_configuration(config_path)
        self._register_default_extractors()

    def _load_configuration(self, config_path: str):
        """Load configuration from YAML file"""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Config file not found: {config_path}, using defaults")
                self._create_default_config()
                return

            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)

            self.config = ExtractionConfig.from_dict(config_data)

            logger.info(
                "Extraction configuration loaded",
                config_path=config_path,
                extractors_enabled=len([
                    name for name, conf in self.config.extractors.items()
                    if conf.get('enabled', True)
                ])
            )

        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            self._create_default_config()

    def _create_default_config(self):
        """Create default configuration when config file is missing"""
        default_config = {
            'extraction': {
                'extractors': {
                    'general': {
                        'enabled': True,
                        'patterns': [],
                        'confidence_threshold': 0.5,
                        'llm_model': 'gpt-4o-mini'
                    }
                },
                'routing': {
                    'cache_enabled': True,
                    'pattern_matching_threshold': 0.8,
                    'fallback_enabled': True
                },
                'limits': {
                    'max_document_size_mb': 50,
                    'timeout_seconds': 300
                },
                'quality': {
                    'min_confidence_for_storage': 0.3
                }
            }
        }
        self.config = ExtractionConfig.from_dict(default_config)

    def _register_default_extractors(self):
        """Register built-in extractors based on configuration"""
        # Mapping of extractor names to classes
        extractor_classes = {
            'organizational': OrganizationalExtractor,
            'financial': FinancialExtractor,
            'approval_matrix': ApprovalMatrixExtractor,
            'general': GeneralExtractor
        }

        for extractor_name, extractor_config in self.config.extractors.items():
            if not extractor_config.get('enabled', True):
                logger.debug(f"Extractor {extractor_name} disabled in config")
                continue

            extractor_class = extractor_classes.get(extractor_name)
            if extractor_class:
                try:
                    extractor = extractor_class(extractor_config)
                    self.register_extractor(extractor_name, extractor)
                    logger.debug(f"Registered extractor: {extractor_name}")
                except Exception as e:
                    logger.error(f"Failed to register {extractor_name}: {e}")

        # Ensure we always have a general extractor as fallback
        if 'general' not in self.extractors:
            general_config = {
                'patterns': [],
                'confidence_threshold': 0.5,
                'llm_model': 'gpt-4o-mini',
                'enabled': True
            }
            self.register_extractor('general', GeneralExtractor(general_config))

    def register_extractor(self, name: str, extractor: BaseExtractor):
        """Register a new extractor"""
        self.extractors[name] = extractor
        logger.info(f"Extractor registered: {name}")

    def unregister_extractor(self, name: str):
        """Unregister an extractor"""
        if name in self.extractors:
            del self.extractors[name]
            logger.info(f"Extractor unregistered: {name}")

    def detect_best_extractor(self, content: str) -> str:
        """
        Detect the best extractor for the content
        No hard-coded logic - uses extractor's own can_handle_document method
        """
        best_extractor = 'general'  # Default fallback
        best_confidence = 0.0

        for name, extractor in self.extractors.items():
            if name == 'general':
                continue  # Check general last

            try:
                confidence = extractor.can_handle_document(content)
                logger.debug(
                    f"Extractor {name} confidence: {confidence:.3f}",
                    extractor=name,
                    confidence=confidence
                )

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_extractor = name

            except Exception as e:
                logger.warning(f"Error checking {name} extractor: {e}")

        # Use general extractor if no specific extractor has high confidence
        extractor_threshold = self.config.routing.get('pattern_matching_threshold', 0.8)
        if best_confidence < extractor_threshold:
            best_extractor = 'general'

        logger.info(
            "Best extractor selected",
            extractor=best_extractor,
            confidence=best_confidence,
            threshold=extractor_threshold
        )

        return best_extractor

    async def extract(
        self,
        content: str,
        document_fingerprint: DocumentFingerprint,
        context: MetadataContext,
        llm_client=None,
        force_extractor: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract information using the best extractor

        Args:
            content: Document content
            document_fingerprint: Document fingerprint
            context: Metadata context
            llm_client: Optional LLM client
            force_extractor: Force use of specific extractor
        """
        start_time = time.time()

        # Determine which extractor to use
        if force_extractor and force_extractor in self.extractors:
            extractor_name = force_extractor
            logger.info(f"Using forced extractor: {extractor_name}")
        else:
            extractor_name = self.detect_best_extractor(content)

        extractor = self.extractors.get(extractor_name)
        if not extractor:
            # Fallback to general extractor
            extractor_name = 'general'
            extractor = self.extractors['general']
            logger.warning(f"Extractor not found, using general extractor")

        try:
            # Validate document size
            max_size_mb = self.config.limits.get('max_document_size_mb', 50)
            content_size_mb = len(content.encode('utf-8')) / (1024 * 1024)

            if content_size_mb > max_size_mb:
                raise ValueError(f"Document too large: {content_size_mb:.1f}MB > {max_size_mb}MB")

            # Perform extraction
            logger.info(
                "Starting extraction",
                extractor=extractor_name,
                document_id=document_fingerprint.document_id,
                content_size_mb=f"{content_size_mb:.2f}"
            )

            result = await extractor.extract(
                content=content,
                document_fingerprint=document_fingerprint,
                context=context,
                llm_client=llm_client
            )

            processing_time = (time.time() - start_time) * 1000

            # Add registry metadata
            result.metadata['registry_info'] = {
                'selected_extractor': extractor_name,
                'total_processing_time_ms': processing_time,
                'config_version': getattr(self.config, 'version', '1.0')
            }

            logger.info(
                "Extraction completed",
                extractor=extractor_name,
                document_id=document_fingerprint.document_id,
                success=result.success,
                entities_count=len(result.entities),
                confidence=result.confidence,
                total_time_ms=f"{processing_time:.2f}"
            )

            return result

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(
                "Extraction failed",
                extractor=extractor_name,
                document_id=document_fingerprint.document_id,
                error=str(e),
                processing_time_ms=f"{processing_time:.2f}"
            )

            # Return error result
            return ExtractionResult(
                success=False,
                document_id=document_fingerprint.document_id,
                extraction_method=extractor.get_extraction_method() if extractor else None,
                entities=[],
                document_type="error",
                confidence=0.0,
                processing_time_ms=processing_time,
                metadata={
                    'error': str(e),
                    'failed_extractor': extractor_name
                },
                warnings=[f"Extraction failed: {str(e)}"]
            )

    def get_available_extractors(self) -> List[str]:
        """Get list of available extractor names"""
        return list(self.extractors.keys())

    def get_extractor_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific extractor"""
        if name not in self.extractors:
            return None

        extractor = self.extractors[name]
        extractor_config = self.config.extractors.get(name, {})

        return {
            'name': name,
            'class': extractor.__class__.__name__,
            'document_type': extractor.get_document_type(),
            'extraction_method': extractor.get_extraction_method().value,
            'patterns': extractor_config.get('patterns', []),
            'confidence_threshold': extractor_config.get('confidence_threshold', 0.5),
            'enabled': extractor_config.get('enabled', True)
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on registry"""
        try:
            return {
                'status': 'healthy',
                'total_extractors': len(self.extractors),
                'available_extractors': self.get_available_extractors(),
                'config_loaded': self.config is not None,
                'default_extractor': 'general' in self.extractors
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

    def reload_configuration(self, config_path: str = "config/extraction_config.yaml"):
        """Reload configuration and re-register extractors"""
        logger.info("Reloading extraction configuration")

        # Clear existing extractors
        self.extractors.clear()

        # Reload config and extractors
        self._load_configuration(config_path)
        self._register_default_extractors()

        logger.info(
            "Configuration reloaded",
            total_extractors=len(self.extractors)
        )


# Factory function for easy instantiation
def create_extractor_registry(config_path: str = "config/extraction_config.yaml") -> ExtractorRegistry:
    """Factory function to create extractor registry"""
    return ExtractorRegistry(config_path)