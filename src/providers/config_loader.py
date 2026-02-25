"""
Centralized configuration loader.

Loads and caches YAML configs from the config/ directory.
All hardcoded values should be read through this module.

Usage:
    from .providers.config_loader import get_config

    scoring = get_config("scoring")
    threshold = scoring["relevance"]["boost"]["high_specificity"]  # 0.8

    infra = get_config("infrastructure")
    timeout = infra["timeouts"]["citation_analysis_seconds"]  # 30
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import structlog

logger = structlog.get_logger()

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
_cache: Dict[str, Dict[str, Any]] = {}


def get_config(name: str) -> Dict[str, Any]:
    """Load a YAML config file by name (without .yaml extension).

    Results are cached after first load. Returns empty dict if file
    doesn't exist (with a warning).

    Examples:
        get_config("scoring")          → loads config/scoring.yaml
        get_config("llm")              → loads config/llm.yaml
        get_config("infrastructure")   → loads config/infrastructure.yaml
    """
    if name in _cache:
        return _cache[name]

    path = _CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        logger.warning("Config file not found, using empty config", path=str(path))
        _cache[name] = {}
        return {}

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    _cache[name] = data
    logger.debug("Config loaded", name=name, keys=list(data.keys()))
    return data


def get_nested(config: Dict[str, Any], *keys, default: Any = None) -> Any:
    """Safely traverse nested config keys.

    Usage:
        scoring = get_config("scoring")
        threshold = get_nested(scoring, "relevance", "boost", "high_specificity", default=0.8)
    """
    current = config
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return default
        if current is None:
            return default
    return current


def reload_config(name: Optional[str] = None) -> None:
    """Clear config cache (all or specific name). Next get_config will reload."""
    if name:
        _cache.pop(name, None)
    else:
        _cache.clear()
