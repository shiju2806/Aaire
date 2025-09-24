"""
Dynamic Phrase Detection for Search Queries
Eliminates hardcoded phrase lists by using NLP and corpus analysis
"""

import re
import structlog
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import os
from functools import lru_cache

logger = structlog.get_logger()


@dataclass
class PhraseCandidate:
    """A potential phrase detected in text"""
    text: str
    frequency: int
    confidence: float
    sources: Set[str]  # Where this phrase was found


class DynamicPhraseDetector:
    """
    Intelligent phrase detection using NLP and corpus analysis.
    Learns phrases from actual document content and query patterns.
    """

    def __init__(self, min_phrase_freq: int = None, min_confidence: float = None, config=None):
        """
        Initialize dynamic phrase detector.

        Args:
            min_phrase_freq: Minimum frequency for a phrase to be considered
            min_confidence: Minimum confidence score for phrase detection
            config: Quality configuration instance
        """
        # Use configuration if available, otherwise fall back to parameters or defaults
        if config:
            phrase_config = config.config.get("phrase_detection", {})
            self.min_phrase_freq = min_phrase_freq or phrase_config.get("min_phrase_frequency", 3)
            self.min_confidence = min_confidence or phrase_config.get("min_confidence", 0.6)
            # Additional configuration for confidence calculations
            self.high_confidence_threshold = phrase_config.get("high_confidence_threshold", 0.8)
            self.bigram_confidence_factor = phrase_config.get("bigram_confidence_factor", 0.1)
            self.trigram_confidence_factor = phrase_config.get("trigram_confidence_factor", 0.05)
        else:
            self.min_phrase_freq = min_phrase_freq or 3
            self.min_confidence = min_confidence or 0.6
            # Default configuration for confidence calculations
            self.high_confidence_threshold = 0.8
            self.bigram_confidence_factor = 0.1
            self.trigram_confidence_factor = 0.05

        # Cached phrase data
        self.phrase_cache: Dict[str, List[str]] = {}
        self.corpus_phrases: Dict[str, PhraseCandidate] = {}
        self.query_patterns: Counter = Counter()

        # NLP components (lightweight, no external dependencies)
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'how', 'what', 'when', 'where', 'why', 'who',
            'do', 'does', 'did', 'can', 'could', 'should', 'would', 'will',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'is', 'are', 'was', 'were'
        }

        logger.info("Dynamic phrase detector initialized",
                   min_freq=min_phrase_freq, min_confidence=min_confidence)

    def extract_phrases_from_query(self, query: str) -> List[str]:
        """
        Extract meaningful phrases from a search query using NLP techniques.

        Args:
            query: Search query text

        Returns:
            List of detected phrases
        """
        # Cache check
        if query in self.phrase_cache:
            return self.phrase_cache[query]

        phrases = []

        # Method 1: Noun phrase detection (simple patterns)
        noun_phrases = self._extract_noun_phrases(query)
        phrases.extend(noun_phrases)

        # Method 2: Quoted phrases (explicit user intent)
        quoted_phrases = self._extract_quoted_phrases(query)
        phrases.extend(quoted_phrases)

        # Method 3: Domain-specific compound terms
        compound_terms = self._extract_compound_terms(query)
        phrases.extend(compound_terms)

        # Method 4: Check against learned corpus phrases
        corpus_matches = self._match_corpus_phrases(query)
        phrases.extend(corpus_matches)

        # Deduplicate and filter
        unique_phrases = list(set(phrases))
        filtered_phrases = self._filter_phrases(unique_phrases, query)

        # Cache result
        self.phrase_cache[query] = filtered_phrases

        # Record query pattern for learning
        self.query_patterns[query] += 1

        logger.debug("Phrases extracted from query",
                    query=query[:50] + "..." if len(query) > 50 else query,
                    phrases=filtered_phrases)

        return filtered_phrases

    def _extract_noun_phrases(self, text: str) -> List[str]:
        """Extract noun phrases using simple pattern matching"""
        phrases = []

        # Pattern 1: Adjective + Noun combinations
        adj_noun_pattern = r'\b([A-Za-z]+(?:al|ic|ive|ous|ary|able|ible)) ([A-Za-z]+(?:ion|ity|ness|ment|ance|ence|ing|ed)?\b)'
        matches = re.findall(adj_noun_pattern, text, re.IGNORECASE)
        phrases.extend([f"{adj} {noun}" for adj, noun in matches])

        # Pattern 2: Noun + Noun combinations (common in technical domains)
        noun_noun_pattern = r'\b([A-Z][a-z]+) ([A-Z][a-z]+|[a-z]+(?:ion|ity|ness|ment|ance|ence|ing|ed|er|or|ar|al|ic))\b'
        matches = re.findall(noun_noun_pattern, text)
        phrases.extend([f"{n1} {n2}" for n1, n2 in matches])

        # Pattern 3: Common business/technical phrase patterns
        tech_patterns = [
            r'\b(\w+) (calculation|analysis|method|approach|system|process|procedure|policy|rate|ratio|factor|table|reserve|benefit|payment|value|cost|risk|model|assessment|validation|test|compliance|requirement)\b',
            r'\b(net|gross|total|minimum|maximum|statutory|regulatory|actuarial|financial|economic|market|credit|operational) (\w+)\b'
        ]

        for pattern in tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            phrases.extend([f"{w1} {w2}".lower() for w1, w2 in matches])

        return phrases

    def _extract_quoted_phrases(self, text: str) -> List[str]:
        """Extract explicitly quoted phrases"""
        quoted_pattern = r'"([^"]+)"'
        matches = re.findall(quoted_pattern, text)
        return [match.strip() for match in matches if len(match.strip()) > 2]

    def _extract_compound_terms(self, text: str) -> List[str]:
        """Extract compound terms and technical terminology"""
        phrases = []

        # Hyphenated terms
        hyphenated_pattern = r'\b([a-zA-Z]+-[a-zA-Z]+(?:-[a-zA-Z]+)*)\b'
        hyphenated_matches = re.findall(hyphenated_pattern, text)
        phrases.extend(hyphenated_matches)

        # Acronym + word combinations (e.g., "VM 20", "IFRS 17")
        acronym_pattern = r'\b([A-Z]{2,6}) ?(\d+|[A-Z][a-z]+)\b'
        acronym_matches = re.findall(acronym_pattern, text)
        phrases.extend([f"{acr} {num}".strip() for acr, num in acronym_matches])

        # Number + word combinations indicating specific terms
        num_word_pattern = r'\b(\d+[\.\-]?\d*) ?([A-Za-z]+(?:ion|ity|ness|ment|ance|ence|ing|ed|er|or|ar|al|ic)?)\b'
        num_matches = re.findall(num_word_pattern, text)

        # Filter for meaningful combinations
        meaningful_num_combos = []
        for num, word in num_matches:
            if len(word) > 3 and word.lower() not in self.stop_words:
                meaningful_num_combos.append(f"{num} {word}")
        phrases.extend(meaningful_num_combos)

        return phrases

    def _match_corpus_phrases(self, query: str) -> List[str]:
        """Match query against phrases learned from document corpus"""
        matches = []
        query_lower = query.lower()

        for phrase_text, candidate in self.corpus_phrases.items():
            if candidate.confidence >= self.min_confidence:
                # Check if phrase exists in query
                if phrase_text in query_lower:
                    matches.append(phrase_text)

                # Check for partial matches with high-confidence phrases
                elif candidate.confidence > self.high_confidence_threshold:
                    phrase_words = phrase_text.split()
                    if all(word in query_lower for word in phrase_words):
                        matches.append(phrase_text)

        return matches

    def _filter_phrases(self, phrases: List[str], original_query: str) -> List[str]:
        """Filter and rank detected phrases"""
        if not phrases:
            return []

        filtered = []

        for phrase in phrases:
            phrase_clean = phrase.strip().lower()

            # Skip if too short or just numbers
            if len(phrase_clean) < 4 or phrase_clean.isdigit():
                continue

            # Skip if all stop words
            words = phrase_clean.split()
            if all(word in self.stop_words for word in words):
                continue

            # Skip single common words
            if len(words) == 1 and len(phrase_clean) < 6:
                continue

            # Must be multi-word or technical term
            if len(words) >= 2 or self._is_technical_term(phrase_clean):
                filtered.append(phrase_clean)

        # Remove duplicates and sort by relevance
        unique_filtered = list(set(filtered))

        # Sort by phrase quality (longer, more technical phrases first)
        def phrase_quality(phrase):
            words = phrase.split()
            score = len(words) * 10  # Prefer multi-word phrases

            # Boost for corpus matches
            if phrase in self.corpus_phrases:
                score += self.corpus_phrases[phrase].confidence * 20

            # Boost for technical indicators
            if self._is_technical_term(phrase):
                score += 15

            return score

        return sorted(unique_filtered, key=phrase_quality, reverse=True)[:5]  # Top 5 phrases

    def _is_technical_term(self, phrase: str) -> bool:
        """Check if phrase appears to be a technical term"""
        technical_indicators = [
            # Contains technical suffixes
            r'(tion|ity|ness|ment|ance|ence|ing|al|ic|ive|ous|ary|able|ible)$',
            # Contains abbreviations or codes
            r'[A-Z]{2,}',
            # Contains numbers (version numbers, codes, etc.)
            r'\d+',
            # Contains hyphens (compound terms)
            r'-',
            # Financial/actuarial keywords
            r'(rate|ratio|factor|reserve|benefit|premium|policy|value|cost|risk|capital|liability|asset)'
        ]

        return any(re.search(pattern, phrase, re.IGNORECASE) for pattern in technical_indicators)

    def learn_from_documents(self, documents: List[Dict]) -> None:
        """
        Learn common phrases from document corpus.

        Args:
            documents: List of document dictionaries with 'content' field
        """
        logger.info(f"Learning phrases from {len(documents)} documents")

        # Extract n-grams from all documents
        all_bigrams = Counter()
        all_trigrams = Counter()

        for doc in documents:
            content = doc.get('content', '')
            if not content:
                continue

            # Extract bigrams and trigrams
            bigrams = self._extract_ngrams(content, n=2)
            trigrams = self._extract_ngrams(content, n=3)

            all_bigrams.update(bigrams)
            all_trigrams.update(trigrams)

        # Convert frequent n-grams to phrase candidates
        for ngram, freq in all_bigrams.items():
            if freq >= self.min_phrase_freq and self._is_meaningful_phrase(ngram):
                confidence = min(1.0, freq / (len(documents) * self.bigram_confidence_factor))  # Normalize by document count
                self.corpus_phrases[ngram] = PhraseCandidate(
                    text=ngram,
                    frequency=freq,
                    confidence=confidence,
                    sources={'corpus_bigrams'}
                )

        for ngram, freq in all_trigrams.items():
            if freq >= self.min_phrase_freq and self._is_meaningful_phrase(ngram):
                confidence = min(1.0, freq / (len(documents) * self.trigram_confidence_factor))  # Higher threshold for trigrams
                self.corpus_phrases[ngram] = PhraseCandidate(
                    text=ngram,
                    frequency=freq,
                    confidence=confidence,
                    sources={'corpus_trigrams'}
                )

        logger.info(f"Learned {len(self.corpus_phrases)} phrases from corpus")

    def _extract_ngrams(self, text: str, n: int) -> List[str]:
        """Extract n-grams from text"""
        # Clean and tokenize
        text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
        words = [w for w in text_clean.split() if w not in self.stop_words and len(w) > 2]

        # Generate n-grams
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)

        return ngrams

    def _is_meaningful_phrase(self, phrase: str) -> bool:
        """Check if an n-gram represents a meaningful phrase"""
        words = phrase.split()

        # Must have at least one non-stop word
        if all(word in self.stop_words for word in words):
            return False

        # Must have reasonable length
        if len(phrase) < 6 or len(phrase) > 50:
            return False

        # Prefer phrases with technical terms
        return any(self._is_technical_term(word) for word in words) or len(words) >= 2

    @lru_cache(maxsize=1000)
    def get_cached_phrases(self, query: str) -> Tuple[str, ...]:
        """Cached version of phrase extraction for performance"""
        return tuple(self.extract_phrases_from_query(query))


def create_dynamic_phrase_detector(**kwargs) -> DynamicPhraseDetector:
    """Factory function to create phrase detector"""
    return DynamicPhraseDetector(**kwargs)