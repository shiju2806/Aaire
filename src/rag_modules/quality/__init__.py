"""
Quality Assessment Module for RAG Pipeline

This module provides comprehensive quality assessment and metrics calculation
for RAG responses, including confidence scoring, adaptive validation, and
intelligent threshold learning.
"""

# Import from the unified quality system
from .unified_quality_system import (
    UnifiedQualitySystem,
    create_unified_quality_system,
    # Backward compatibility imports
    QualityMetricsManager,
    create_quality_metrics_manager,
    IntelligentValidationSystem,
    create_intelligent_validator
)

__all__ = [
    # Primary exports
    'UnifiedQualitySystem',
    'create_unified_quality_system',
    # Backward compatibility exports
    'QualityMetricsManager',
    'create_quality_metrics_manager',
    'IntelligentValidationSystem',
    'create_intelligent_validator'
]