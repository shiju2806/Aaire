"""
Entropy-based Disambiguation Service
Differentiates semantically similar concepts through corpus-driven mutual exclusion detection
NO HARD-CODED DOMAIN KNOWLEDGE - learns from document corpus statistics
"""

import structlog
import asyncio
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import re
from pathlib import Path
import json
import time
from datetime import datetime, timedelta

# Scientific computation for entropy analysis
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# KeyBERT for concept extraction
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

from ..config.quality_config import QualityConfig
from ..core.dependency_injection import ServiceMixin

logger = structlog.get_logger()


@dataclass
class ConceptPair:
    """Represents two concepts that may be mutually exclusive"""
    concept_a: str
    concept_b: str
    exclusion_confidence: float = 0.0
    co_occurrence_count: int = 0
    separate_occurrence_count: int = 0
    fisher_p_value: float = 1.0
    chi_square_stat: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class DisambiguationResult:
    """Result of disambiguation analysis for a query"""
    original_query: str
    detected_concepts: List[str]
    conflicting_pairs: List[ConceptPair]
    confidence: float
    recommended_disambiguation: Optional[str] = None
    entropy_score: float = 0.0


class EntropyDisambiguationService(ServiceMixin):
    """
    Entropy-based disambiguation service that learns mutual exclusions
    from document corpus without hard-coded domain knowledge.

    Core principles:
    1. Extract concepts using KeyBERT from both queries and corpus
    2. Build co-occurrence matrix from corpus analysis
    3. Detect mutual exclusions through statistical tests
    4. Use confidence thresholds for reliable disambiguation
    """

    def __init__(self, config: Optional[QualityConfig] = None):
        """Initialize entropy disambiguation service"""
        super().__init__()
        from ..config.quality_config import get_quality_config
        self._config = config or get_quality_config()

        # Initialize models
        self.keybert_model = None
        self.embedding_model = None

        # Concept extraction parameters (from config, not hard-coded)
        self.concept_extraction_config = self._config.config.get('entropy_disambiguation', {
            'top_k_concepts': 10,
            'keyphrase_ngram_range': (1, 3),
            'min_concept_frequency': 3,
            'diversity_threshold': 0.5
        })

        # Statistical thresholds (configurable, not hard-coded)
        self.statistical_config = self._config.config.get('statistical_thresholds', {
            'min_confidence': 0.3,
            'high_confidence': 0.6,
            'min_co_occurrence': 5,
            'fisher_significance': 0.05,
            'chi_square_threshold': 3.84  # p < 0.05 for 1 DOF
        })

        # Learned patterns storage
        self.concept_pairs: Dict[str, ConceptPair] = {}
        self.concept_frequencies: Dict[str, int] = defaultdict(int)
        self.document_concept_matrix: Dict[str, Set[str]] = {}

        # Cache management
        self.cache_invalidation_time = timedelta(hours=1)
        self.last_corpus_analysis = None

        logger.info("Entropy disambiguation service initialized",
                   top_k_concepts=self.concept_extraction_config['top_k_concepts'],
                   min_confidence=self.statistical_config['min_confidence'])

    async def initialize_models(self):
        """Initialize KeyBERT and embedding models asynchronously"""
        try:
            # Use existing embedding model from service container
            self.embedding_model = self.get_service('embedding_model', singleton=True)

            # Initialize KeyBERT with the same embedding model for consistency
            self.keybert_model = KeyBERT(model=self.embedding_model)

            logger.info("KeyBERT and embedding models initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize models", error=str(e))
            # Fallback to default models
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.keybert_model = KeyBERT(model=self.embedding_model)
            logger.info("Using fallback models for disambiguation")

    def extract_concepts(self, text: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Extract key concepts from text using KeyBERT

        Args:
            text: Input text to analyze
            top_k: Number of concepts to extract (defaults to config)

        Returns:
            List of (concept, relevance_score) tuples
        """
        if not self.keybert_model:
            logger.warning("KeyBERT model not initialized, attempting initialization")
            asyncio.create_task(self.initialize_models())
            return []

        top_k = top_k or self.concept_extraction_config['top_k_concepts']
        ngram_range = tuple(self.concept_extraction_config['keyphrase_ngram_range'])
        diversity = self.concept_extraction_config['diversity_threshold']

        try:
            # Extract keywords with diversity to avoid redundant concepts
            concepts = self.keybert_model.extract_keywords(
                text,
                keyphrase_ngram_range=ngram_range,
                stop_words='english',
                top_n=top_k,  # Changed from top_k to top_n for newer KeyBERT version
                use_mmr=True,  # Use Maximal Marginal Relevance for diversity
                diversity=diversity
            )

            # Filter by minimum frequency requirement (learned from corpus)
            min_freq = self.concept_extraction_config['min_concept_frequency']
            filtered_concepts = [
                (concept, score) for concept, score in concepts
                if self.concept_frequencies.get(concept, 0) >= min_freq or score > 0.6
            ]

            logger.debug("Extracted concepts",
                        total_concepts=len(concepts),
                        filtered_concepts=len(filtered_concepts),
                        text_preview=text[:100] + "..." if len(text) > 100 else text)

            return filtered_concepts

        except Exception as e:
            logger.error("Concept extraction failed", error=str(e))
            return []

    def analyze_corpus_cooccurrence(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze document corpus to build co-occurrence patterns

        Args:
            documents: List of document dictionaries with 'content' or 'text' fields

        Returns:
            Analysis results including co-occurrence matrix and statistics
        """
        if not documents:
            logger.warning("No documents provided for corpus analysis")
            return {}

        # Extract concepts from all documents
        all_concepts = set()
        doc_concepts = {}

        for i, doc in enumerate(documents):
            content = doc.get('content', '') or doc.get('text', '')
            if not content:
                continue

            concepts = self.extract_concepts(content)
            concept_set = {concept for concept, score in concepts if score > 0.3}

            if concept_set:
                doc_id = doc.get('id', f'doc_{i}')
                doc_concepts[doc_id] = concept_set
                all_concepts.update(concept_set)

                # Update frequency tracking
                for concept in concept_set:
                    self.concept_frequencies[concept] += 1

        # Build co-occurrence matrix
        concept_list = list(all_concepts)
        n_concepts = len(concept_list)
        cooccurrence_matrix = np.zeros((n_concepts, n_concepts))

        # Count co-occurrences
        for doc_concept_set in doc_concepts.values():
            doc_concept_indices = [
                i for i, concept in enumerate(concept_list)
                if concept in doc_concept_set
            ]

            # Update co-occurrence matrix
            for i in doc_concept_indices:
                for j in doc_concept_indices:
                    if i != j:
                        cooccurrence_matrix[i, j] += 1

        # Store document-concept mapping for future analysis
        self.document_concept_matrix.update(doc_concepts)

        analysis_results = {
            'concept_list': concept_list,
            'cooccurrence_matrix': cooccurrence_matrix,
            'total_documents': len(doc_concepts),
            'total_concepts': n_concepts,
            'document_concepts': doc_concepts
        }

        logger.info("Corpus co-occurrence analysis completed",
                   total_docs=len(doc_concepts),
                   total_concepts=n_concepts,
                   avg_concepts_per_doc=np.mean([len(concepts) for concepts in doc_concepts.values()]))

        return analysis_results

    def detect_mutual_exclusions(self, corpus_analysis: Dict[str, Any]) -> List[ConceptPair]:
        """
        Detect mutually exclusive concept pairs using statistical tests

        Args:
            corpus_analysis: Results from analyze_corpus_cooccurrence

        Returns:
            List of ConceptPair objects with exclusion confidence scores
        """
        if not corpus_analysis or 'concept_list' not in corpus_analysis:
            logger.warning("Invalid corpus analysis provided")
            return []

        concept_list = corpus_analysis['concept_list']
        cooccurrence_matrix = corpus_analysis['cooccurrence_matrix']
        total_docs = corpus_analysis['total_documents']

        detected_pairs = []
        min_co_occurrence = self.statistical_config['min_co_occurrence']
        fisher_sig = self.statistical_config['fisher_significance']

        for i, concept_a in enumerate(concept_list):
            for j, concept_b in enumerate(concept_list[i+1:], i+1):

                # Calculate contingency table
                co_occur = cooccurrence_matrix[i, j]
                a_alone = self.concept_frequencies[concept_a] - co_occur
                b_alone = self.concept_frequencies[concept_b] - co_occur
                neither = total_docs - (co_occur + a_alone + b_alone)

                # Skip if insufficient data
                if co_occur + a_alone + b_alone < min_co_occurrence:
                    continue

                # Contingency table: [[co_occur, a_alone], [b_alone, neither]]
                contingency_table = np.array([
                    [co_occur, a_alone],
                    [b_alone, neither]
                ])

                # Perform statistical tests
                try:
                    # Fisher's exact test for small samples
                    if np.sum(contingency_table) < 50:
                        _, fisher_p = fisher_exact(contingency_table)
                        chi_square_stat = 0.0
                    else:
                        # Chi-square test for larger samples
                        chi_square_stat, chi_p, _, _ = chi2_contingency(contingency_table)
                        fisher_p = chi_p

                    # Calculate exclusion confidence
                    # Higher confidence when concepts rarely co-occur but frequently appear separately
                    expected_co_occur = (self.concept_frequencies[concept_a] *
                                       self.concept_frequencies[concept_b]) / total_docs
                    exclusion_ratio = max(0, expected_co_occur - co_occur) / max(1, expected_co_occur)

                    # Statistical significance adjustment
                    significance_boost = 1.0 if fisher_p < fisher_sig else 0.5
                    exclusion_confidence = min(1.0, exclusion_ratio * significance_boost)

                    if exclusion_confidence > 0.1:  # Only keep potentially meaningful pairs
                        pair = ConceptPair(
                            concept_a=concept_a,
                            concept_b=concept_b,
                            exclusion_confidence=exclusion_confidence,
                            co_occurrence_count=int(co_occur),
                            separate_occurrence_count=int(a_alone + b_alone),
                            fisher_p_value=fisher_p,
                            chi_square_stat=chi_square_stat
                        )
                        detected_pairs.append(pair)

                        # Store for future use
                        pair_key = f"{concept_a}|||{concept_b}"
                        self.concept_pairs[pair_key] = pair

                except Exception as e:
                    logger.debug("Statistical test failed for concept pair",
                               concept_a=concept_a, concept_b=concept_b, error=str(e))
                    continue

        # Sort by exclusion confidence
        detected_pairs.sort(key=lambda p: p.exclusion_confidence, reverse=True)

        logger.info("Mutual exclusion detection completed",
                   total_pairs_analyzed=len(concept_list) * (len(concept_list) - 1) // 2,
                   exclusion_pairs_found=len(detected_pairs),
                   high_confidence_pairs=len([p for p in detected_pairs
                                            if p.exclusion_confidence > self.statistical_config['high_confidence']]))

        return detected_pairs

    async def disambiguate_query(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> DisambiguationResult:
        """
        Perform disambiguation analysis on a query using learned mutual exclusions

        Args:
            query: User query to analyze
            retrieved_docs: Retrieved documents for context

        Returns:
            DisambiguationResult with detected conflicts and recommendations
        """
        # Extract concepts from query
        query_concepts = self.extract_concepts(query)
        detected_concepts = [concept for concept, score in query_concepts if score > 0.3]

        if len(detected_concepts) < 2:
            return DisambiguationResult(
                original_query=query,
                detected_concepts=detected_concepts,
                conflicting_pairs=[],
                confidence=1.0,  # No ambiguity detected
                entropy_score=0.0
            )

        # Analyze corpus if not recently done
        if (self.last_corpus_analysis is None or
            datetime.now() - self.last_corpus_analysis > self.cache_invalidation_time):

            logger.info("Performing fresh corpus analysis for disambiguation")
            corpus_analysis = self.analyze_corpus_cooccurrence(retrieved_docs)
            mutual_exclusions = self.detect_mutual_exclusions(corpus_analysis)
            self.last_corpus_analysis = datetime.now()

        # Find conflicts between query concepts
        conflicting_pairs = []
        for i, concept_a in enumerate(detected_concepts):
            for concept_b in detected_concepts[i+1:]:
                # Check both orderings
                pair_key1 = f"{concept_a}|||{concept_b}"
                pair_key2 = f"{concept_b}|||{concept_a}"

                pair = self.concept_pairs.get(pair_key1) or self.concept_pairs.get(pair_key2)
                if pair and pair.exclusion_confidence > self.statistical_config['min_confidence']:
                    conflicting_pairs.append(pair)

        # Calculate overall confidence and entropy
        if conflicting_pairs:
            max_conflict = max(pair.exclusion_confidence for pair in conflicting_pairs)
            confidence = 1.0 - max_conflict

            # Calculate entropy score (higher = more ambiguous)
            concept_probs = np.array([score for _, score in query_concepts[:len(detected_concepts)]])
            concept_probs = concept_probs / np.sum(concept_probs)  # Normalize
            entropy_score = -np.sum(concept_probs * np.log2(concept_probs + 1e-10))
        else:
            confidence = 1.0
            entropy_score = 0.0

        # Generate disambiguation recommendation
        recommendation = None
        if conflicting_pairs and confidence < self.statistical_config['high_confidence']:
            highest_conflict = max(conflicting_pairs, key=lambda p: p.exclusion_confidence)
            recommendation = f"Query contains potentially conflicting concepts: '{highest_conflict.concept_a}' and '{highest_conflict.concept_b}'. Consider specifying which you're interested in."

        result = DisambiguationResult(
            original_query=query,
            detected_concepts=detected_concepts,
            conflicting_pairs=conflicting_pairs,
            confidence=confidence,
            recommended_disambiguation=recommendation,
            entropy_score=entropy_score
        )

        logger.info("Query disambiguation completed",
                   query_preview=query[:50] + "..." if len(query) > 50 else query,
                   concepts_detected=len(detected_concepts),
                   conflicts_found=len(conflicting_pairs),
                   confidence=confidence)

        return result

    def should_apply_disambiguation(self, disambiguation_result: DisambiguationResult) -> bool:
        """
        Determine if disambiguation should be applied based on confidence thresholds

        Args:
            disambiguation_result: Result from disambiguate_query

        Returns:
            True if disambiguation should be applied
        """
        min_conf = self.statistical_config['min_confidence']
        high_conf = self.statistical_config['high_confidence']

        confidence = disambiguation_result.confidence

        if confidence < min_conf:
            # Don't use disambiguation - too unreliable
            logger.debug("Disambiguation confidence too low, skipping", confidence=confidence)
            return False
        elif min_conf <= confidence < high_conf:
            # Use with fallback - apply light disambiguation
            logger.debug("Applying light disambiguation with fallback", confidence=confidence)
            return True
        else:
            # High confidence - apply full disambiguation
            logger.debug("Applying full disambiguation", confidence=confidence)
            return True

    def get_disambiguation_stats(self) -> Dict[str, Any]:
        """Get statistics about learned disambiguation patterns"""
        total_pairs = len(self.concept_pairs)
        high_conf_pairs = len([p for p in self.concept_pairs.values()
                              if p.exclusion_confidence > self.statistical_config['high_confidence']])

        return {
            'total_concept_pairs': total_pairs,
            'high_confidence_pairs': high_conf_pairs,
            'total_concepts_tracked': len(self.concept_frequencies),
            'documents_analyzed': len(self.document_concept_matrix),
            'last_corpus_analysis': self.last_corpus_analysis.isoformat() if self.last_corpus_analysis else None,
            'cache_status': 'fresh' if (self.last_corpus_analysis and
                                      datetime.now() - self.last_corpus_analysis < self.cache_invalidation_time)
                          else 'stale'
        }


def create_entropy_disambiguation_service(config: Optional[QualityConfig] = None) -> EntropyDisambiguationService:
    """Factory function to create entropy disambiguation service"""
    return EntropyDisambiguationService(config)