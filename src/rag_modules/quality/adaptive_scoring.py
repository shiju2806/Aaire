"""
Adaptive Document-Query Relevance Scoring

Dynamic relevance scoring system that learns from user interactions and
adapts scoring weights based on query patterns and feedback.
"""

import re
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
import structlog

logger = structlog.get_logger()


@dataclass
class ScoringFeature:
    """Feature for adaptive scoring"""
    name: str
    weight: float
    learned_weight: float
    performance_history: List[float]
    last_updated: datetime
    sample_count: int


class AdaptiveRelevanceScorer:
    """
    Adaptive document-query relevance scorer that learns optimal feature weights
    from user feedback and query patterns without hardcoded logic.
    """

    def __init__(self, learning_data_path: str = "/tmp/rag_scoring_data.json"):
        """
        Initialize adaptive relevance scorer.

        Args:
            learning_data_path: Path to store learning data and feature weights
        """
        self.learning_data_path = learning_data_path
        self.scoring_features: Dict[str, ScoringFeature] = {}
        self.query_patterns: Dict[str, List[float]] = defaultdict(list)
        self.feedback_history: List[Dict] = []

        # Initialize scoring features
        self._initialize_features()
        self._load_learning_data()

    def _initialize_features(self):
        """Initialize scoring features with adaptive starting weights"""
        now = datetime.now()

        # Start with equal weights that adapt based on performance
        self.scoring_features = {
            "semantic_similarity": ScoringFeature(
                name="semantic_similarity",
                weight=0.3,
                learned_weight=0.3,
                performance_history=[],
                last_updated=now,
                sample_count=0
            ),
            "keyword_overlap": ScoringFeature(
                name="keyword_overlap",
                weight=0.25,
                learned_weight=0.25,
                performance_history=[],
                last_updated=now,
                sample_count=0
            ),
            "document_freshness": ScoringFeature(
                name="document_freshness",
                weight=0.1,
                learned_weight=0.1,
                performance_history=[],
                last_updated=now,
                sample_count=0
            ),
            "content_completeness": ScoringFeature(
                name="content_completeness",
                weight=0.15,
                learned_weight=0.15,
                performance_history=[],
                last_updated=now,
                sample_count=0
            ),
            "query_specificity_match": ScoringFeature(
                name="query_specificity_match",
                weight=0.2,
                learned_weight=0.2,
                performance_history=[],
                last_updated=now,
                sample_count=0
            )
        }

    def calculate_adaptive_relevance_score(self, query: str, document: Dict[str, Any]) -> float:
        """
        Calculate adaptive relevance score using learned feature weights.

        Args:
            query: User query string
            document: Document dictionary with content, metadata, etc.

        Returns:
            Adaptive relevance score between 0.0 and 1.0
        """
        try:
            # Extract features
            features = self._extract_features(query, document)

            # Calculate weighted score using learned weights
            total_score = 0.0
            total_weight = 0.0

            for feature_name, feature_value in features.items():
                if feature_name in self.scoring_features:
                    weight = self.scoring_features[feature_name].learned_weight
                    total_score += feature_value * weight
                    total_weight += weight

            # Normalize score
            final_score = total_score / total_weight if total_weight > 0 else 0.0

            # Apply query pattern adaptation
            pattern_boost = self._get_pattern_boost(query, features)
            final_score = min(1.0, final_score * pattern_boost)

            return round(final_score, 4)

        except Exception as e:
            logger.error("Adaptive relevance scoring failed", error=str(e))
            return 0.0

    def _extract_features(self, query: str, document: Dict[str, Any]) -> Dict[str, float]:
        """Extract relevance features from query-document pair"""
        features = {}

        try:
            doc_content = document.get('content', '').lower()
            query_lower = query.lower()

            # 1. Semantic Similarity (use existing similarity score if available)
            features['semantic_similarity'] = document.get('score', 0.0)

            # 2. Keyword Overlap
            features['keyword_overlap'] = self._calculate_keyword_overlap(query_lower, doc_content)

            # 3. Document Freshness
            features['document_freshness'] = self._calculate_freshness_score(document)

            # 4. Content Completeness
            features['content_completeness'] = self._calculate_completeness_score(doc_content)

            # 5. Query Specificity Match
            features['query_specificity_match'] = self._calculate_specificity_match(query_lower, doc_content)

            return features

        except Exception as e:
            logger.error("Feature extraction failed", error=str(e))
            return {name: 0.0 for name in self.scoring_features.keys()}

    def _calculate_keyword_overlap(self, query: str, content: str) -> float:
        """Calculate dynamic keyword overlap score"""
        # Extract meaningful keywords (not just word count)
        query_words = set(re.findall(r'\b\w{3,}\b', query))
        content_words = set(re.findall(r'\b\w{3,}\b', content))

        # Remove common stop words dynamically
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'man', 'oil', 'sit', 'son', 'top', 'use', 'big', 'far', 'few', 'got', 'own', 'say', 'she', 'too', 'ask', 'run', 'try', 'end', 'why', 'let', 'put', 'tell', 'time', 'does', 'have', 'they', 'this', 'that', 'will', 'would', 'could', 'should', 'with', 'what', 'when', 'where', 'from', 'into', 'upon', 'each', 'some', 'such', 'only', 'more', 'than', 'just', 'very', 'same', 'make', 'made', 'come', 'take', 'know', 'good', 'first', 'never', 'after', 'right', 'think', 'before', 'through', 'between', 'however', 'because', 'therefore', 'although'}

        query_keywords = query_words - stop_words
        content_keywords = content_words - stop_words

        if not query_keywords:
            return 0.5  # Neutral score if no meaningful keywords

        # Calculate weighted overlap (exact matches get higher weight)
        exact_matches = len(query_keywords & content_keywords)
        partial_matches = 0

        # Check for partial matches (stems, variants)
        for qword in query_keywords:
            for cword in content_keywords:
                if qword in cword or cword in qword:
                    partial_matches += 0.5
                    break

        total_overlap = exact_matches + partial_matches
        return min(1.0, total_overlap / len(query_keywords))

    def _calculate_freshness_score(self, document: Dict[str, Any]) -> float:
        """Calculate document freshness score"""
        # Use document metadata if available
        doc_date = document.get('last_modified')
        if not doc_date:
            return 0.5  # Neutral score if no date info

        try:
            if isinstance(doc_date, str):
                doc_datetime = datetime.fromisoformat(doc_date.replace('Z', '+00:00'))
            else:
                doc_datetime = doc_date

            days_old = (datetime.now() - doc_datetime.replace(tzinfo=None)).days

            # Fresher documents get higher scores (exponential decay)
            if days_old <= 30:
                return 1.0
            elif days_old <= 180:
                return 0.8
            elif days_old <= 365:
                return 0.6
            else:
                return 0.4

        except Exception:
            return 0.5

    def _calculate_completeness_score(self, content: str) -> float:
        """Calculate content completeness score"""
        if not content:
            return 0.0

        # Analyze content structure and completeness
        content_length = len(content)
        word_count = len(content.split())

        # Look for structured content indicators
        has_headers = bool(re.search(r'(?:^|\n)#{1,6}\s+\w+', content))
        has_lists = bool(re.search(r'(?:^|\n)\s*[-*•]\s+\w+', content))
        has_numbers = bool(re.search(r'\d+', content))
        has_specifics = bool(re.search(r'(?:section|article|paragraph|clause)\s+\d+', content, re.I))

        completeness_score = 0.0

        # Length-based scoring
        if word_count > 100:
            completeness_score += 0.4
        elif word_count > 50:
            completeness_score += 0.2

        # Structure-based scoring
        if has_headers:
            completeness_score += 0.2
        if has_lists:
            completeness_score += 0.2
        if has_numbers:
            completeness_score += 0.1
        if has_specifics:
            completeness_score += 0.1

        return min(1.0, completeness_score)

    def _calculate_specificity_match(self, query: str, content: str) -> float:
        """Calculate how well document specificity matches query specificity"""
        # Analyze query specificity
        query_specificity = self._analyze_query_specificity(query)

        # Analyze content specificity
        content_specificity = self._analyze_content_specificity(content)

        # Score based on how well they match
        specificity_diff = abs(query_specificity - content_specificity)

        # Closer match gets higher score
        return max(0.0, 1.0 - specificity_diff)

    def _analyze_query_specificity(self, query: str) -> float:
        """Analyze how specific a query is (0.0 = general, 1.0 = very specific)"""
        specificity_indicators = [
            (r'\b\d+(\.\d+)?\b', 0.2),  # Numbers
            (r'\b(?:specific|exact|precise|detailed)\b', 0.3),  # Specificity words
            (r'\b[A-Z]{2,}\b', 0.2),  # Acronyms
            (r'\b(?:section|article|paragraph|clause)\s+\d+', 0.3),  # Specific references
            (r'\b(?:ratio|percentage|formula|calculation)\b', 0.2),  # Technical terms
        ]

        specificity = 0.3  # Base specificity
        for pattern, weight in specificity_indicators:
            if re.search(pattern, query, re.I):
                specificity += weight

        return min(1.0, specificity)

    def _analyze_content_specificity(self, content: str) -> float:
        """Analyze how specific content is"""
        if not content:
            return 0.0

        specificity_indicators = [
            (r'\b\d+(\.\d+)?%?\b', 0.1),  # Numbers/percentages
            (r'\$\d+', 0.1),  # Money amounts
            (r'\b(?:section|article|paragraph|clause)\s+\d+', 0.2),  # Specific references
            (r'\b[A-Z]{2,}\b', 0.1),  # Acronyms
            (r'(?:formula|equation|calculation):', 0.2),  # Technical content
        ]

        specificity = 0.2  # Base specificity
        for pattern, weight in specificity_indicators:
            matches = len(re.findall(pattern, content, re.I))
            specificity += min(weight, matches * weight * 0.1)

        return min(1.0, specificity)

    def _get_pattern_boost(self, query: str, features: Dict[str, float]) -> float:
        """Get adaptive boost based on learned query patterns"""
        # Analyze query pattern
        pattern_key = self._categorize_query(query)

        # Get historical performance for this pattern
        if pattern_key in self.query_patterns and self.query_patterns[pattern_key]:
            avg_performance = np.mean(self.query_patterns[pattern_key][-10:])  # Last 10 samples
            # Boost based on historical success
            return 0.8 + (avg_performance * 0.4)  # Range: 0.8 - 1.2
        else:
            return 1.0  # No boost for unknown patterns

    def _categorize_query(self, query: str) -> str:
        """Categorize query into patterns for learning"""
        query_lower = query.lower()

        if any(word in query_lower for word in ['how to', 'process', 'steps']):
            return 'procedural'
        elif any(word in query_lower for word in ['what is', 'define', 'definition']):
            return 'definitional'
        elif any(word in query_lower for word in ['calculate', 'formula', 'ratio']):
            return 'computational'
        elif any(word in query_lower for word in ['requirement', 'compliance', 'standard']):
            return 'regulatory'
        elif '?' in query:
            return 'question'
        else:
            return 'general'

    def record_scoring_feedback(self, query: str, document: Dict[str, Any],
                              user_satisfaction: float, final_score: float):
        """Record feedback for adaptive learning"""
        feedback = {
            'query': query,
            'document_id': document.get('id', ''),
            'features': self._extract_features(query, document),
            'final_score': final_score,
            'user_satisfaction': user_satisfaction,
            'timestamp': datetime.now().isoformat(),
            'query_pattern': self._categorize_query(query)
        }

        self.feedback_history.append(feedback)

        # Update pattern performance
        pattern = feedback['query_pattern']
        self.query_patterns[pattern].append(user_satisfaction)

        # Trigger learning if we have enough samples
        if len(self.feedback_history) >= 20:
            self._update_feature_weights()

    def _update_feature_weights(self):
        """Update feature weights based on performance feedback"""
        if len(self.feedback_history) < 10:
            return

        try:
            # Analyze feature performance
            feature_performance = defaultdict(list)

            for feedback in self.feedback_history[-50:]:  # Last 50 samples
                satisfaction = feedback['user_satisfaction']
                for feature_name, feature_value in feedback['features'].items():
                    if feature_value > 0.5:  # Only consider when feature was relevant
                        feature_performance[feature_name].append(satisfaction)

            # Update weights based on performance
            total_performance = 0.0
            new_weights = {}

            for feature_name, feature in self.scoring_features.items():
                if feature_name in feature_performance:
                    avg_performance = np.mean(feature_performance[feature_name])
                    feature.performance_history.append(avg_performance)
                    total_performance += avg_performance
                    new_weights[feature_name] = avg_performance
                else:
                    new_weights[feature_name] = feature.learned_weight

            # Normalize weights
            if total_performance > 0:
                for feature_name, performance in new_weights.items():
                    normalized_weight = performance / total_performance

                    # Apply learning rate for gradual adaptation
                    learning_rate = 0.2
                    old_weight = self.scoring_features[feature_name].learned_weight
                    new_weight = old_weight * (1 - learning_rate) + normalized_weight * learning_rate

                    self.scoring_features[feature_name].learned_weight = new_weight
                    self.scoring_features[feature_name].last_updated = datetime.now()
                    self.scoring_features[feature_name].sample_count += 1

            # Save learning data
            self._save_learning_data()

            logger.info("Feature weights updated from feedback",
                       samples_processed=len(self.feedback_history),
                       features_updated=len(new_weights))

        except Exception as e:
            logger.error("Feature weight update failed", error=str(e))

    def get_current_weights(self) -> Dict[str, float]:
        """Get current learned feature weights"""
        return {name: feature.learned_weight for name, feature in self.scoring_features.items()}

    def _save_learning_data(self):
        """Save learning data and feature weights"""
        try:
            data = {
                'features': {name: asdict(feature) for name, feature in self.scoring_features.items()},
                'query_patterns': dict(self.query_patterns),
                'feedback_count': len(self.feedback_history),
                'last_updated': datetime.now().isoformat()
            }

            # Convert datetime objects to strings
            for feature_data in data['features'].values():
                feature_data['last_updated'] = feature_data['last_updated'].isoformat()

            with open(self.learning_data_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error("Failed to save scoring learning data", error=str(e))

    def _load_learning_data(self):
        """Load previously learned weights and patterns"""
        try:
            with open(self.learning_data_path, 'r') as f:
                data = json.load(f)

            # Restore features
            for feature_name, feature_data in data.get('features', {}).items():
                if feature_name in self.scoring_features:
                    feature_data['last_updated'] = datetime.fromisoformat(feature_data['last_updated'])
                    self.scoring_features[feature_name] = ScoringFeature(**feature_data)

            # Restore query patterns
            self.query_patterns = defaultdict(list, data.get('query_patterns', {}))

            logger.info("Loaded scoring learning data",
                       features_loaded=len(data.get('features', {})),
                       patterns_loaded=len(self.query_patterns))

        except FileNotFoundError:
            logger.info("No previous scoring learning data found")
        except Exception as e:
            logger.error("Failed to load scoring learning data", error=str(e))


def create_adaptive_scorer(learning_data_path: str = "/tmp/rag_scoring_data.json") -> AdaptiveRelevanceScorer:
    """Factory function to create adaptive relevance scorer"""
    return AdaptiveRelevanceScorer(learning_data_path)