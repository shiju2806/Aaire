#!/usr/bin/env python3
"""
Debug configuration loading
"""

import sys
sys.path.append('/Users/shijuprakash/AAIRE/src')

from rag_modules.config.quality_config import get_quality_config

def debug_config():
    config = get_quality_config()

    print("🔍 Configuration Debug:")
    print(f"Environment: {config.environment}")
    print(f"Config file path: {config.config_path}")

    print("\nAll thresholds from config file:")
    thresholds = config.config.get("thresholds", {})
    for key, value in thresholds.items():
        print(f"  {key}: {value}")

    print("\nDefault fallback values from _get_default_config():")
    defaults = config._get_default_config()
    print(f"  Default thresholds: {defaults['thresholds']}")

    print("\nMethod outputs:")
    print(f"  get_semantic_alignment_threshold(): {config.get_semantic_alignment_threshold()}")
    print(f"  get_confidence_threshold(): {config.get_confidence_threshold()}")
    print(f"  get_threshold('semantic_alignment_minimum'): {config.get_threshold('semantic_alignment_minimum')}")
    print(f"  get_threshold('semantic_confidence_minimum'): {config.get_threshold('semantic_confidence_minimum')}")

if __name__ == "__main__":
    debug_config()