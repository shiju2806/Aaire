"""
Configurable Framework Detection Service
Eliminates hard-coded framework logic and provides maintainable framework detection
"""

import re
import yaml
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import structlog

logger = structlog.get_logger()


@dataclass
class FrameworkMatch:
    """Result of framework detection"""
    framework: str
    confidence: float
    matched_keywords: List[str]
    matched_indicators: List[str]
    keyword_score: float
    indicator_score: float


class FrameworkDetector:
    """
    Configurable framework detection service
    Loads detection rules from configuration and provides consistent framework classification
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "/Users/shijuprakash/AAIRE/config/mvp_config.yaml"
        self.framework_config = self._load_framework_config()
        self.enabled = self.framework_config.get('enabled', True)
        self.confidence_threshold = self.framework_config.get('confidence_threshold', 0.7)
        self.min_content_length = self.framework_config.get('min_content_length', 100)
        self.default_framework = self.framework_config.get('default_framework', 'general')

        logger.info(
            "FrameworkDetector initialized",
            enabled=self.enabled,
            frameworks_loaded=len(self.framework_config.get('frameworks', {})),
            confidence_threshold=self.confidence_threshold
        )

    def _load_framework_config(self) -> Dict:
        """Load framework detection configuration from YAML"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config.get('framework_detection', {})
        except Exception as e:
            logger.error(f"Failed to load framework config: {e}")
            return {
                'enabled': False,
                'frameworks': {},
                'default_framework': 'general'
            }

    def detect_framework(self, content: str, filename: Optional[str] = None) -> FrameworkMatch:
        """
        Detect the primary framework for given content

        Args:
            content: Text content to analyze
            filename: Optional filename for additional context

        Returns:
            FrameworkMatch with detection results
        """
        if not self.enabled or len(content) < self.min_content_length:
            return FrameworkMatch(
                framework=self.default_framework,
                confidence=0.0,
                matched_keywords=[],
                matched_indicators=[],
                keyword_score=0.0,
                indicator_score=0.0
            )

        # Prepare content for analysis
        content_lower = content.lower()
        filename_lower = filename.lower() if filename else ""

        # Score each framework
        framework_scores = {}

        frameworks = self.framework_config.get('frameworks', {})
        for framework_name, framework_config in frameworks.items():
            score_result = self._score_framework(
                content_lower,
                filename_lower,
                framework_config,
                framework_name
            )
            framework_scores[framework_name] = score_result

        # Find best match
        if not framework_scores:
            return FrameworkMatch(
                framework=self.default_framework,
                confidence=0.0,
                matched_keywords=[],
                matched_indicators=[],
                keyword_score=0.0,
                indicator_score=0.0
            )

        # Get highest scoring framework
        best_framework = max(framework_scores.items(), key=lambda x: x[1].confidence)
        best_match = best_framework[1]

        # Apply confidence threshold
        if best_match.confidence < self.confidence_threshold:
            return FrameworkMatch(
                framework=self.default_framework,
                confidence=best_match.confidence,
                matched_keywords=best_match.matched_keywords,
                matched_indicators=best_match.matched_indicators,
                keyword_score=best_match.keyword_score,
                indicator_score=best_match.indicator_score
            )

        logger.debug(
            "Framework detected",
            framework=best_match.framework,
            confidence=best_match.confidence,
            keywords=len(best_match.matched_keywords),
            indicators=len(best_match.matched_indicators)
        )

        return best_match

    def _score_framework(
        self,
        content_lower: str,
        filename_lower: str,
        framework_config: Dict,
        framework_name: str
    ) -> FrameworkMatch:
        """Score a specific framework against content"""

        keywords = framework_config.get('keywords', [])
        content_indicators = framework_config.get('content_indicators', [])
        confidence_boost = framework_config.get('confidence_boost', 0.0)

        # Find keyword matches
        matched_keywords = []
        keyword_score = 0.0

        for keyword in keywords:
            if keyword.lower() in content_lower or keyword.lower() in filename_lower:
                matched_keywords.append(keyword)
                # Weight longer keywords higher
                keyword_score += len(keyword.split()) * 0.1

        # Find content indicator matches
        matched_indicators = []
        indicator_score = 0.0

        for indicator in content_indicators:
            if indicator.lower() in content_lower:
                matched_indicators.append(indicator)
                # Indicators are more specific, weight them higher
                indicator_score += len(indicator.split()) * 0.15

        # Calculate base confidence
        base_confidence = min(1.0, keyword_score + indicator_score)

        # Apply confidence boost if matches found
        if matched_keywords or matched_indicators:
            base_confidence += confidence_boost

        # Normalize to [0, 1]
        final_confidence = min(1.0, max(0.0, base_confidence))

        return FrameworkMatch(
            framework=framework_name,  # Use the framework name from the key
            confidence=final_confidence,
            matched_keywords=matched_keywords,
            matched_indicators=matched_indicators,
            keyword_score=keyword_score,
            indicator_score=indicator_score
        )

    def get_all_frameworks(self) -> List[str]:
        """Get list of all configured frameworks"""
        frameworks = self.framework_config.get('frameworks', {})
        return list(frameworks.keys()) + [self.default_framework]

    def get_framework_config(self, framework: str) -> Optional[Dict]:
        """Get configuration for a specific framework"""
        frameworks = self.framework_config.get('frameworks', {})
        return frameworks.get(framework)

    def is_framework_supported(self, framework: str) -> bool:
        """Check if framework is supported"""
        return framework in self.get_all_frameworks()

    def get_detection_stats(self) -> Dict:
        """Get detection statistics and configuration info"""
        frameworks = self.framework_config.get('frameworks', {})
        return {
            'enabled': self.enabled,
            'frameworks_configured': len(frameworks),
            'framework_names': list(frameworks.keys()),
            'confidence_threshold': self.confidence_threshold,
            'default_framework': self.default_framework,
            'min_content_length': self.min_content_length
        }

    def reload_config(self):
        """Reload framework configuration from file"""
        logger.info("Reloading framework detection configuration")
        self.framework_config = self._load_framework_config()
        self.enabled = self.framework_config.get('enabled', True)
        self.confidence_threshold = self.framework_config.get('confidence_threshold', 0.7)
        self.min_content_length = self.framework_config.get('min_content_length', 100)
        self.default_framework = self.framework_config.get('default_framework', 'general')


# Global instance for shared use
_framework_detector_instance = None


def get_framework_detector(config_path: Optional[str] = None) -> FrameworkDetector:
    """Get singleton framework detector instance"""
    global _framework_detector_instance

    if _framework_detector_instance is None:
        _framework_detector_instance = FrameworkDetector(config_path)

    return _framework_detector_instance


def detect_document_framework(content: str, filename: Optional[str] = None) -> str:
    """
    Convenience function for framework detection
    Returns just the framework name
    """
    detector = get_framework_detector()
    match = detector.detect_framework(content, filename)
    return match.framework


def detect_document_framework_with_confidence(
    content: str,
    filename: Optional[str] = None
) -> Tuple[str, float]:
    """
    Convenience function for framework detection with confidence
    Returns framework name and confidence score
    """
    detector = get_framework_detector()
    match = detector.detect_framework(content, filename)
    return match.framework, match.confidence