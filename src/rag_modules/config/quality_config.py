"""
Centralized Quality Configuration Manager
Eliminates hardcoded values and provides environment-aware configuration
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
import structlog

logger = structlog.get_logger()


class QualityConfig:
    """
    Centralized configuration manager for quality validation system.
    Eliminates all hardcoded values and provides environment-aware settings.
    """

    def __init__(self, config_path: Optional[str] = None, environment: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to quality validation config file
            environment: Environment name (development, production, testing)
        """
        self.environment = environment or os.getenv("AAIRE_ENV", "development")

        # Default config path
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "config" / "quality_validation.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._apply_environment_overrides()

        logger.info("Quality configuration loaded",
                   environment=self.environment,
                   config_path=str(self.config_path))

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning("Quality config file not found, using defaults",
                          path=str(self.config_path))
            return self._get_default_config()
        except yaml.YAMLError as e:
            logger.error("Failed to parse quality config file", exception_details=str(e))
            return self._get_default_config()

    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides."""
        env_overrides = self.config.get("environments", {}).get(self.environment, {})

        if env_overrides:
            logger.info("Applying environment overrides",
                       environment=self.environment,
                       overrides=list(env_overrides.keys()))
            self._deep_merge(self.config, env_overrides)

    def _deep_merge(self, base: Dict, override: Dict):
        """Deep merge override config into base config."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file not found."""
        return {
            "thresholds": {
                "semantic_alignment_minimum": 0.65,
                "grounding_score_minimum": 0.6,
                "confidence_minimum": 0.7
            },
            "models": {
                "primary_embedding": "all-MiniLM-L6-v2",
                "primary_llm": "gpt-4o-mini"
            },
            "feature_flags": {
                "advanced_grounding": True,
                "adaptive_thresholds": True
            }
        }

    # Threshold Management
    def get_threshold(self, metric: str) -> float:
        """Get threshold value for a specific metric."""
        return self.config.get("thresholds", {}).get(metric, 0.5)

    def get_semantic_alignment_threshold(self) -> float:
        """Get semantic alignment minimum threshold."""
        return self.get_threshold("semantic_alignment_minimum")

    def get_grounding_threshold(self) -> float:
        """Get grounding score minimum threshold."""
        return self.get_threshold("grounding_score_minimum")

    def get_confidence_threshold(self) -> float:
        """Get confidence minimum threshold."""
        return self.get_threshold("semantic_confidence_minimum")

    def get_evidence_coverage_threshold(self) -> float:
        """Get evidence coverage minimum threshold."""
        return self.get_threshold("evidence_coverage_minimum")

    def get_hallucination_risk_threshold(self) -> float:
        """Get maximum acceptable hallucination risk."""
        return self.get_threshold("hallucination_risk_maximum")

    # Model Configuration
    def get_model_name(self, model_type: str) -> str:
        """Get model name for specific type."""
        return self.config.get("models", {}).get(model_type, "gpt-4o-mini")

    def get_embedding_model(self) -> str:
        """Get primary embedding model name."""
        return self.config.get("models", {}).get("embedding", {}).get("name", "all-MiniLM-L6-v2")

    def get_llm_model(self) -> str:
        """Get primary LLM model name."""
        return self.config.get("models", {}).get("llm", {}).get("name", "gpt-4o-mini")

    def get_advanced_llm_model(self) -> str:
        """Get advanced LLM model name for complex tasks."""
        return self.config.get("models", {}).get("advanced_llm", {}).get("name", "gpt-4o")

    def get_model_params(self, model_type: str) -> Dict[str, Any]:
        """Get parameters for specific model type."""
        model_config = self.config.get("models", {}).get(model_type, {})
        if isinstance(model_config, dict):
            # Return all params except 'name'
            return {k: v for k, v in model_config.items() if k != 'name'}
        return {}

    # Weight Configuration
    def get_validation_weights(self, component: str = "grounding_validation") -> Dict[str, float]:
        """Get validation component weights."""
        return self.config.get("weights", {}).get(component, {})

    def get_grounding_weights(self) -> Dict[str, float]:
        """Get grounding validation weights."""
        return self.get_validation_weights("grounding_validation")

    def get_unified_quality_weights(self) -> Dict[str, float]:
        """Get unified quality system weights."""
        return self.get_validation_weights("unified_quality")

    # Feature Flags
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature flag is enabled."""
        return self.config.get("feature_flags", {}).get(feature, False)

    def is_advanced_grounding_enabled(self) -> bool:
        """Check if advanced grounding validation is enabled."""
        return self.is_feature_enabled("advanced_grounding")

    def is_adaptive_learning_enabled(self) -> bool:
        """Check if adaptive threshold learning is enabled."""
        return self.is_feature_enabled("adaptive_thresholds")

    def is_openai_alignment_enabled(self) -> bool:
        """Check if OpenAI alignment validator is enabled."""
        return self.is_feature_enabled("openai_alignment_validator")

    # Enhanced Whoosh Configuration
    def get_enhanced_whoosh_config(self) -> Dict[str, Any]:
        """Get Enhanced Whoosh configuration."""
        return self.config.get("enhanced_whoosh", {})

    def is_enhanced_whoosh_enabled(self) -> bool:
        """Check if Enhanced Whoosh is enabled."""
        return self.get_enhanced_whoosh_config().get("enabled", True)

    def get_jurisdiction_threshold(self) -> float:
        """Get jurisdiction classification threshold."""
        return self.get_enhanced_whoosh_config().get("jurisdiction_threshold", 0.4)

    def get_product_confidence_minimum(self) -> float:
        """Get minimum product confidence threshold."""
        return self.get_enhanced_whoosh_config().get("product_confidence_minimum", 0.3)

    def should_include_mixed_jurisdictions(self) -> bool:
        """Check if mixed jurisdictions should be included in searches."""
        return self.get_enhanced_whoosh_config().get("include_mixed_jurisdictions", True)

    def get_classification_mode(self) -> str:
        """Get classification mode: flexible, strict, or off."""
        return self.get_enhanced_whoosh_config().get("classification_mode", "flexible")

    def should_fallback_to_vector(self) -> bool:
        """Check if system should fallback to vector search when keyword returns 0 results."""
        return self.get_enhanced_whoosh_config().get("fallback_to_vector", True)

    # Adaptive Learning Configuration
    def get_adaptive_learning_config(self) -> Dict[str, Any]:
        """Get adaptive learning configuration."""
        return self.config.get("adaptive_learning", {})

    def get_minimum_learning_samples(self) -> int:
        """Get minimum samples needed for adaptive learning."""
        return self.get_adaptive_learning_config().get("minimum_samples", 50)

    def get_learning_rate(self) -> float:
        """Get adaptive learning rate."""
        return self.get_adaptive_learning_config().get("learning_rate", 0.1)

    def get_threshold_bounds(self) -> Dict[str, float]:
        """Get bounds for adaptive threshold adjustments."""
        return self.get_adaptive_learning_config().get("threshold_bounds", {
            "minimum": 0.3,
            "maximum": 0.9
        })

    # Performance Configuration
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance-related configuration."""
        return self.config.get("performance", {})

    def get_similarity_config(self) -> Dict[str, Any]:
        """Get similarity calculation configuration."""
        return self.get_performance_config().get("similarity_calculation", {})

    def get_caching_config(self) -> Dict[str, Any]:
        """Get caching configuration."""
        return self.get_performance_config().get("caching", {})

    def is_caching_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self.get_caching_config().get("enabled", True)

    # Dynamic Configuration Updates
    def update_threshold(self, metric: str, value: float):
        """Dynamically update a threshold value."""
        if "thresholds" not in self.config:
            self.config["thresholds"] = {}

        bounds = self.get_threshold_bounds()
        value = max(bounds["minimum"], min(bounds["maximum"], value))

        old_value = self.config["thresholds"].get(metric, 0.5)
        self.config["thresholds"][metric] = value

        logger.info("Threshold updated dynamically",
                   metric=metric,
                   old_value=old_value,
                   new_value=value)

    def update_learned_thresholds(self, threshold_updates: Dict[str, float]):
        """Update multiple thresholds from adaptive learning."""
        for metric, value in threshold_updates.items():
            self.update_threshold(metric, value)

    # Validation and Health Checks
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration integrity and return status."""
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": []
        }

        # Check required sections
        required_sections = ["thresholds", "models", "weights"]
        for section in required_sections:
            if section not in self.config:
                validation_result["errors"].append(f"Missing required section: {section}")
                validation_result["valid"] = False

        # Check threshold ranges
        thresholds = self.config.get("thresholds", {})
        for key, value in thresholds.items():
            if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                validation_result["warnings"].append(f"Threshold {key} should be between 0.0 and 1.0")

        # Check model availability
        models = self.config.get("models", {})
        required_models = ["primary_embedding", "primary_llm"]
        for model in required_models:
            if model not in models:
                validation_result["warnings"].append(f"Missing recommended model config: {model}")

        return validation_result

    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration for monitoring."""
        return {
            "environment": self.environment,
            "config_path": str(self.config_path),
            "thresholds_count": len(self.config.get("thresholds", {})),
            "models_configured": len(self.config.get("models", {})),
            "features_enabled": [k for k, v in self.config.get("feature_flags", {}).items() if v],
            "adaptive_learning": self.is_adaptive_learning_enabled(),
            "validation_status": self.validate_config()
        }


# Global configuration instance
_global_config: Optional[QualityConfig] = None


def get_quality_config(config_path: Optional[str] = None,
                      environment: Optional[str] = None,
                      force_reload: bool = False) -> QualityConfig:
    """
    Get global quality configuration instance.

    Args:
        config_path: Optional path to config file
        environment: Optional environment name
        force_reload: Force reload configuration

    Returns:
        QualityConfig instance
    """
    global _global_config

    if _global_config is None or force_reload:
        _global_config = QualityConfig(config_path, environment)

    return _global_config


def reload_quality_config():
    """Reload quality configuration from file."""
    global _global_config
    if _global_config:
        _global_config = QualityConfig(
            str(_global_config.config_path),
            _global_config.environment
        )
        logger.info("Quality configuration reloaded")