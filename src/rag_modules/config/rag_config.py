"""
Centralized Configuration for RAG Pipeline
Provides a single source of truth for all RAG system settings
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class FormattingConfig:
    """Configuration for formatting operations"""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_formatting_retries: int = 2
    formula_cache_ttl: int = 900  # 15 minutes
    enable_regex_preprocessing: bool = True
    enable_deterministic_postprocessing: bool = True


@dataclass
class ResponseGenerationConfig:
    """Configuration for response generation"""
    single_pass_doc_limit: int = 8
    chunk_size_tokens: int = 2000
    parallel_processing_threshold: int = 4
    max_groups_for_chunking: int = 4
    semantic_grouping_threshold: float = 0.3
    enable_dynamic_strategy: bool = True
    simple_pass_token_limit: int = 2000
    enhanced_pass_token_limit: int = 8000


@dataclass
class QualityThresholds:
    """Quality thresholds for validation"""
    min_response_quality: float = 0.7
    min_relevance_score: float = 0.5
    min_completeness_score: float = 0.6
    min_accuracy_score: float = 0.7
    min_formatting_score: float = 0.8


@dataclass
class CitationConfig:
    """Configuration for citation extraction"""
    max_citations: int = 5
    min_usage_score_threshold: float = 0.05
    min_relevance_threshold: float = 0.1
    enable_intelligent_analysis: bool = True
    fallback_to_top_document: bool = True


@dataclass
class CacheConfig:
    """Configuration for caching"""
    enable_response_cache: bool = True
    response_cache_ttl: int = 3600  # 1 hour
    enable_formula_cache: bool = True
    formula_cache_ttl: int = 900  # 15 minutes
    enable_embedding_cache: bool = True
    embedding_cache_ttl: int = 86400  # 24 hours


@dataclass
class RetrievalConfig:
    """Configuration for document retrieval"""
    default_top_k: int = 20
    hybrid_search_alpha: float = 0.7  # Weight for vector search vs keyword
    enable_reranking: bool = True
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    min_similarity_threshold: float = 0.3


@dataclass
class RAGConfig:
    """Main RAG configuration class"""
    # Sub-configurations
    formatting: FormattingConfig = field(default_factory=FormattingConfig)
    response_generation: ResponseGenerationConfig = field(default_factory=ResponseGenerationConfig)
    quality_thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    citations: CitationConfig = field(default_factory=CitationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    # Global settings
    default_llm_model: str = "gpt-4o-mini"
    default_embedding_model: str = "text-embedding-ada-002"
    enable_debug_logging: bool = False
    enable_performance_monitoring: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """Create config from dictionary"""
        config = cls()

        # Update formatting config
        if 'formatting' in config_dict:
            for key, value in config_dict['formatting'].items():
                if hasattr(config.formatting, key):
                    setattr(config.formatting, key, value)

        # Update response generation config
        if 'response_generation' in config_dict:
            for key, value in config_dict['response_generation'].items():
                if hasattr(config.response_generation, key):
                    setattr(config.response_generation, key, value)

        # Update quality thresholds
        if 'quality_thresholds' in config_dict:
            for key, value in config_dict['quality_thresholds'].items():
                if hasattr(config.quality_thresholds, key):
                    setattr(config.quality_thresholds, key, value)

        # Update citation config
        if 'citations' in config_dict:
            for key, value in config_dict['citations'].items():
                if hasattr(config.citations, key):
                    setattr(config.citations, key, value)

        # Update cache config
        if 'cache' in config_dict:
            for key, value in config_dict['cache'].items():
                if hasattr(config.cache, key):
                    setattr(config.cache, key, value)

        # Update retrieval config
        if 'retrieval' in config_dict:
            for key, value in config_dict['retrieval'].items():
                if hasattr(config.retrieval, key):
                    setattr(config.retrieval, key, value)

        # Update global settings
        for key in ['default_llm_model', 'default_embedding_model', 'enable_debug_logging', 'enable_performance_monitoring']:
            if key in config_dict:
                setattr(config, key, config_dict[key])

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'formatting': {
                'model': self.formatting.model,
                'temperature': self.formatting.temperature,
                'max_formatting_retries': self.formatting.max_formatting_retries,
                'formula_cache_ttl': self.formatting.formula_cache_ttl,
                'enable_regex_preprocessing': self.formatting.enable_regex_preprocessing,
                'enable_deterministic_postprocessing': self.formatting.enable_deterministic_postprocessing,
            },
            'response_generation': {
                'single_pass_doc_limit': self.response_generation.single_pass_doc_limit,
                'chunk_size_tokens': self.response_generation.chunk_size_tokens,
                'parallel_processing_threshold': self.response_generation.parallel_processing_threshold,
                'max_groups_for_chunking': self.response_generation.max_groups_for_chunking,
                'semantic_grouping_threshold': self.response_generation.semantic_grouping_threshold,
                'enable_dynamic_strategy': self.response_generation.enable_dynamic_strategy,
                'simple_pass_token_limit': self.response_generation.simple_pass_token_limit,
                'enhanced_pass_token_limit': self.response_generation.enhanced_pass_token_limit,
            },
            'quality_thresholds': {
                'min_response_quality': self.quality_thresholds.min_response_quality,
                'min_relevance_score': self.quality_thresholds.min_relevance_score,
                'min_completeness_score': self.quality_thresholds.min_completeness_score,
                'min_accuracy_score': self.quality_thresholds.min_accuracy_score,
                'min_formatting_score': self.quality_thresholds.min_formatting_score,
            },
            'citations': {
                'max_citations': self.citations.max_citations,
                'min_usage_score_threshold': self.citations.min_usage_score_threshold,
                'min_relevance_threshold': self.citations.min_relevance_threshold,
                'enable_intelligent_analysis': self.citations.enable_intelligent_analysis,
                'fallback_to_top_document': self.citations.fallback_to_top_document,
            },
            'cache': {
                'enable_response_cache': self.cache.enable_response_cache,
                'response_cache_ttl': self.cache.response_cache_ttl,
                'enable_formula_cache': self.cache.enable_formula_cache,
                'formula_cache_ttl': self.cache.formula_cache_ttl,
                'enable_embedding_cache': self.cache.enable_embedding_cache,
                'embedding_cache_ttl': self.cache.embedding_cache_ttl,
            },
            'retrieval': {
                'default_top_k': self.retrieval.default_top_k,
                'hybrid_search_alpha': self.retrieval.hybrid_search_alpha,
                'enable_reranking': self.retrieval.enable_reranking,
                'reranking_model': self.retrieval.reranking_model,
                'min_similarity_threshold': self.retrieval.min_similarity_threshold,
            },
            'default_llm_model': self.default_llm_model,
            'default_embedding_model': self.default_embedding_model,
            'enable_debug_logging': self.enable_debug_logging,
            'enable_performance_monitoring': self.enable_performance_monitoring,
        }


# Singleton instance
_default_config: Optional[RAGConfig] = None


def get_default_config() -> RAGConfig:
    """Get the default configuration instance"""
    global _default_config
    if _default_config is None:
        _default_config = RAGConfig()
    return _default_config


def set_default_config(config: RAGConfig) -> None:
    """Set the default configuration instance"""
    global _default_config
    _default_config = config


def load_config_from_file(filepath: str) -> RAGConfig:
    """Load configuration from a YAML or JSON file"""
    import yaml
    import json

    with open(filepath, 'r') as f:
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            config_dict = yaml.safe_load(f)
        elif filepath.endswith('.json'):
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {filepath}")

    return RAGConfig.from_dict(config_dict)