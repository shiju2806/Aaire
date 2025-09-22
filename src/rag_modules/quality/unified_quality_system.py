"""
Unified Quality System

Comprehensive quality management system that combines configuration, measurement,
validation, and enforcement for industry-grade RAG pipeline. This unified system
replaces both QualityMetricsManager and IntelligentValidationSystem with a single
cohesive implementation.
"""

import json
import time
import asyncio
import re
import math
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import structlog

from .adaptive_gates import AdaptiveQualityGates, ResponseFeedback
from .adaptive_scoring import AdaptiveRelevanceScorer
from .grounding_validator import ContentGroundingValidator, GroundingResult

logger = structlog.get_logger()


@dataclass
class ValidationResult:
    """Comprehensive validation result"""
    passed: bool
    overall_score: float
    confidence: float
    components: Dict[str, Any]
    rejection_reason: Optional[str] = None
    processing_time_ms: float = 0.0


@dataclass
class LearningMetrics:
    """Metrics for learning performance tracking"""
    validation_count: int
    acceptance_rate: float
    avg_processing_time: float
    threshold_adjustments: int
    last_learning_update: datetime


class UnifiedQualitySystem:
    """
    Unified quality system that combines configuration, measurement, validation,
    and enforcement capabilities for RAG pipeline. Provides:

    Pre-retrieval Configuration:
    - Dynamic similarity thresholds based on query type
    - Adaptive document limits based on query complexity

    Post-response Quality:
    - Comprehensive quality metrics calculation
    - Confidence scoring for responses
    - Intelligent validation and enforcement
    - Real-time learning from validation patterns
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize intelligent validation system.

        Args:
            config: Configuration for validation components
        """
        self.config = config or {}

        # Initialize adaptive components
        self.quality_gates = AdaptiveQualityGates()
        self.relevance_scorer = AdaptiveRelevanceScorer()
        self.grounding_validator = ContentGroundingValidator()

        # Learning and metrics tracking
        self.validation_history: List[Dict] = []
        self.performance_metrics: LearningMetrics = LearningMetrics(
            validation_count=0,
            acceptance_rate=0.0,
            avg_processing_time=0.0,
            threshold_adjustments=0,
            last_learning_update=datetime.now()
        )

        # Real-time learning parameters
        self.learning_enabled = True
        self.learning_interval = 50  # Updates every 50 validations
        self.feedback_window = timedelta(hours=24)  # 24-hour feedback window

    # ================== Pre-retrieval Configuration Methods ==================
    # Methods previously in QualityMetricsManager for pipeline configuration

    def get_similarity_threshold(self, query: str) -> float:
        """
        Determine optimal similarity threshold using adaptive learning.

        Uses machine learning-based approach that adapts thresholds based on:
        - Historical query performance patterns
        - Query complexity and length
        - Retrieval success metrics
        - User feedback patterns

        Args:
            query: The user's input query

        Returns:
            Float similarity threshold value between 0.0 and 1.0
        """
        # Base threshold that adapts based on learning
        base_threshold = getattr(self, 'learned_similarity_threshold', 0.70)

        # Adjust based on query characteristics using ML features
        query_length = len(query.split())
        query_complexity = len([word for word in query.split() if len(word) > 6])

        # Adaptive adjustment based on query features
        if query_length > 15:  # Long queries may need lower threshold for comprehensiveness
            adjustment = -0.05
            reason = "long query - comprehensive search"
        elif query_complexity > 3:  # Complex queries may need precise matches
            adjustment = 0.05
            reason = "complex query - precise search"
        else:
            adjustment = 0.0
            reason = "standard query"

        # Apply bounds checking
        threshold = max(0.55, min(0.85, base_threshold + adjustment))

        logger.info("Adaptive threshold selected",
                   query=query[:50] + "..." if len(query) > 50 else query,
                   threshold=threshold,
                   base_threshold=base_threshold,
                   adjustment=adjustment,
                   reason=reason)

        return threshold

    def get_document_limit(self, query: str) -> int:
        """
        Dynamically determine document limit based on query complexity.

        Analyzes query characteristics to determine optimal number of documents:
        - Word count and length indicators
        - Question complexity patterns
        - Technical procedure requirements
        - Multi-part query detection
        - Regulatory/compliance complexity patterns

        Args:
            query: The user's input query

        Returns:
            Integer document limit based on query complexity
        """
        # Get config
        config = self.config.get('retrieval_config', {})
        base_limit = config.get('base_document_limit', 15)
        standard_limit = config.get('standard_document_limit', 20)
        complex_limit = config.get('complex_document_limit', 30)
        max_limit = config.get('max_document_limit', 40)

        # Analyze query complexity
        query_lower = query.lower()
        words = query_lower.split()
        word_count = len(words)

        # Initialize complexity score
        complexity_score = 0

        # Word count indicator
        if word_count > 15:
            complexity_score += 2  # Very long query
        elif word_count > 10:
            complexity_score += 1  # Long query

        # Question complexity indicators
        comprehensive_words = ['how', 'why', 'what', 'explain', 'describe', 'discuss']
        if any(word in words for word in comprehensive_words):
            complexity_score += 1

        # Technical procedure indicators
        technical_words = ['calculate', 'determine', 'implement', 'process', 'analyze', 'evaluate']
        if any(word in words for word in technical_words):
            complexity_score += 1

        # Multi-part query indicators
        multi_indicators = ['and', 'also', 'additionally', 'furthermore', 'moreover', 'including']
        if sum(1 for word in multi_indicators if word in words) >= 2:
            complexity_score += 1  # Multiple aspects to address

        # Regulatory/compliance complexity (generic patterns)
        if len(re.findall(r'\b[A-Z]{2,}\b', query)) > 2:  # Multiple acronyms
            complexity_score += 1

        # Choose limit based on complexity score
        if complexity_score >= 4:
            final_limit = min(complex_limit, max_limit)  # Complex: 30 docs
            complexity_name = "Complex"
        elif complexity_score >= 2:
            final_limit = min(standard_limit, max_limit)  # Standard: 20 docs
            complexity_name = "Standard"
        else:
            final_limit = min(base_limit, max_limit)      # Simple: 15 docs
            complexity_name = "Simple"

        logger.info(f"Dynamic document limit: {final_limit} ({complexity_name} query, complexity score: {complexity_score})")

        return final_limit

    # ================== Post-response Quality Metrics Methods ==================
    # Methods for measuring and calculating quality after response generation

    def calculate_quality_metrics(self, query: str, response: str, retrieved_docs: List[Dict],
                                 citations: List[Dict]) -> Dict[str, float]:
        """
        Calculate comprehensive quality metrics for the response.

        This method evaluates multiple dimensions of response quality including:
        - Citation coverage: How well the response is supported by sources
        - Response length appropriateness: Not too short, not too long
        - Query-response relevance: Basic keyword overlap analysis
        - Source quality: Average similarity scores of retrieved documents
        - Response completeness: Structured content and specific details

        Args:
            query: The user's input query
            response: The generated response text
            retrieved_docs: List of retrieved documents with scores
            citations: List of citations used in the response

        Returns:
            Dict containing individual metric scores and overall quality score
        """
        try:
            # Initialize metrics
            metrics = {}

            # 1. Citation Coverage - How well the response is supported by sources
            if retrieved_docs:
                # Count how many retrieved docs are actually cited
                cited_docs = len(citations) if citations else 0
                total_docs = len(retrieved_docs)
                metrics['citation_coverage'] = cited_docs / total_docs if total_docs > 0 else 0.0
            else:
                metrics['citation_coverage'] = 0.0

            # 2. Response Length Appropriateness - Not too short, not too long
            response_words = len(response.split())
            if response_words < 20:
                metrics['length_score'] = 0.3  # Too short
            elif response_words > 500:
                metrics['length_score'] = 0.7  # Might be too long
            else:
                metrics['length_score'] = 1.0  # Appropriate length

            # 3. Query-Response Relevance - Basic keyword overlap
            query_words = set(query.lower().split())
            response_words_set = set(response.lower().split())

            # Remove common words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                         'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
                         'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                         'may', 'might', 'can'}
            query_keywords = query_words - stop_words
            response_keywords = response_words_set - stop_words

            if query_keywords:
                overlap = len(query_keywords & response_keywords)
                metrics['keyword_relevance'] = overlap / len(query_keywords)
            else:
                metrics['keyword_relevance'] = 0.5  # Neutral if no keywords

            # 4. Source Quality - Average similarity scores of retrieved docs
            if retrieved_docs:
                scores = [doc.get('score', 0.0) for doc in retrieved_docs]
                metrics['source_quality'] = sum(scores) / len(scores) if scores else 0.0
            else:
                metrics['source_quality'] = 0.0

            # 5. Response Completeness - Basic heuristics
            has_structured_response = any(marker in response for marker in
                                         ['1.', '2.', '•', '-', 'Steps:', 'Requirements:'])
            has_specific_details = any(term in response.lower() for term in
                                      ['ratio', 'percentage', '%', '$', 'requirement',
                                       'standard', 'regulation'])

            completeness_score = 0.5  # Base score
            if has_structured_response:
                completeness_score += 0.25
            if has_specific_details:
                completeness_score += 0.25

            metrics['completeness'] = min(completeness_score, 1.0)

            # 6. Overall Quality Score (weighted average)
            weights = {
                'citation_coverage': 0.25,
                'length_score': 0.15,
                'keyword_relevance': 0.25,
                'source_quality': 0.20,
                'completeness': 0.15
            }

            overall_score = sum(metrics[key] * weights[key] for key in weights if key in metrics)
            metrics['overall_quality'] = overall_score

            # 7. Add intelligent validation metrics if available
            metrics['intelligent_validation'] = {
                'passed': False,  # Will be set by validate_response
                'overall_score': 0.0,  # Will be set by validate_response
                'processing_time_ms': 0.0  # Will be set by validate_response
            }

            # Log quality metrics for monitoring
            logger.info("Response quality metrics calculated",
                       overall_quality=overall_score,
                       citation_coverage=metrics['citation_coverage'],
                       keyword_relevance=metrics['keyword_relevance'],
                       source_quality=metrics['source_quality'])

            return metrics

        except Exception as e:
            logger.error("Failed to calculate quality metrics", error=str(e))
            return {
                "citation_coverage": 0.0,
                "length_score": 0.5,
                "keyword_relevance": 0.5,
                "source_quality": 0.0,
                "completeness": 0.5,
                "overall_quality": 0.3,
                "intelligent_validation": {
                    "passed": False,
                    "overall_score": 0.0,
                    "processing_time_ms": 0.0
                }
            }

    def calculate_confidence(self, retrieved_docs: List[Dict], response: str) -> float:
        """
        Calculate confidence score for the response.

        Computes confidence based on:
        - Average similarity scores of top retrieved documents
        - Number of relevant documents available
        - Document count factor for reliability assessment

        Args:
            retrieved_docs: List of retrieved documents with similarity scores
            response: The generated response text (currently not used in calculation)

        Returns:
            Float confidence score between 0.0 and 1.0
        """
        if not retrieved_docs:
            return 0.0

        # Average similarity score of top documents
        top_scores = [doc['score'] for doc in retrieved_docs[:3]]
        avg_score = sum(top_scores) / len(top_scores) if top_scores else 0.0

        # Adjust based on number of relevant documents
        doc_count_factor = min(len(retrieved_docs) / 3.0, 1.0)

        # Basic confidence calculation
        confidence = avg_score * doc_count_factor

        return round(confidence, 3)

    async def validate_response(self, query: str, response: str, retrieved_docs: List[Dict[str, Any]],
                              confidence: Optional[float] = None) -> ValidationResult:
        """
        Perform comprehensive intelligent validation of RAG response.

        Args:
            query: Original user query
            response: Generated response text
            retrieved_docs: Retrieved documents with metadata
            confidence: Optional confidence score from generation

        Returns:
            ValidationResult with comprehensive validation details
        """
        start_time = time.time()

        try:
            # Prepare validation metrics
            validation_metrics = {
                'query': query,
                'response': response,
                'retrieved_docs': retrieved_docs,
                'confidence': confidence or 0.0,
                'retrieval_scores': [doc.get('score', 0.0) for doc in retrieved_docs],
                'source_quality': self._calculate_source_quality(retrieved_docs)
            }

            components = {}
            overall_scores = []

            # 1. Content Grounding Validation (moved first to populate metrics)
            grounding_result = self.grounding_validator.validate_content_grounding(
                query, response, retrieved_docs
            )

            # Add grounding score to metrics for quality gates evaluation
            validation_metrics['grounding_score'] = grounding_result.grounding_score

            # 2. Adaptive Quality Gates Evaluation (now with grounding score)
            gates_passed, gate_results = self.quality_gates.evaluate_quality_gates(validation_metrics)
            components['quality_gates'] = {
                'passed': gates_passed,
                'results': gate_results,
                'weight': 0.3
            }
            if gates_passed:
                overall_scores.append(0.8 * 0.3)  # High score if gates pass

            # 3. Adaptive Relevance Scoring
            relevance_scores = []
            for doc in retrieved_docs:
                doc_score = self.relevance_scorer.calculate_adaptive_relevance_score(query, doc)
                relevance_scores.append(doc_score)

            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
            components['relevance_scoring'] = {
                'average_score': avg_relevance,
                'individual_scores': relevance_scores,
                'weight': 0.25
            }
            overall_scores.append(avg_relevance * 0.25)

            # 4. Content Grounding Validation (results already calculated above)
            components['content_grounding'] = {
                'result': asdict(grounding_result),
                'weight': 0.35
            }
            overall_scores.append(grounding_result.grounding_score * 0.35)

            # 4. Adaptive Confidence Assessment
            adaptive_confidence = self._calculate_adaptive_confidence(
                validation_metrics, components
            )
            components['adaptive_confidence'] = {
                'score': adaptive_confidence,
                'weight': 0.1
            }
            overall_scores.append(adaptive_confidence * 0.1)

            # Calculate overall validation score
            overall_score = sum(overall_scores)

            # Determine if validation passes using adaptive thresholds
            validation_passed = self._determine_validation_result(overall_score, components)

            # Generate rejection reason if needed
            rejection_reason = None
            if not validation_passed:
                rejection_reason = self._generate_comprehensive_rejection_reason(components)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            # Create validation result
            result = ValidationResult(
                passed=validation_passed,
                overall_score=overall_score,
                confidence=adaptive_confidence,
                components=components,
                rejection_reason=rejection_reason,
                processing_time_ms=processing_time
            )

            # Record validation for learning
            await self._record_validation_result(query, response, retrieved_docs, result)

            # Trigger real-time learning if needed
            if self.learning_enabled:
                await self._check_and_trigger_learning()

            return result

        except Exception as e:
            logger.error("Intelligent validation failed", error=str(e))
            processing_time = (time.time() - start_time) * 1000

            return ValidationResult(
                passed=False,
                overall_score=0.0,
                confidence=0.0,
                components={'error': str(e)},
                rejection_reason="Validation system error",
                processing_time_ms=processing_time
            )

    def _calculate_source_quality(self, retrieved_docs: List[Dict]) -> float:
        """Calculate overall source quality from retrieved documents"""
        if not retrieved_docs:
            return 0.0

        scores = [doc.get('score', 0.0) for doc in retrieved_docs]
        return sum(scores) / len(scores) if scores else 0.0

    def _calculate_adaptive_confidence(self, validation_metrics: Dict[str, Any],
                                     components: Dict[str, Any]) -> float:
        """Calculate adaptive confidence based on all validation components"""
        confidence_factors = []

        # Base confidence from generation
        base_confidence = validation_metrics.get('confidence', 0.5)
        confidence_factors.append(base_confidence * 0.3)

        # Quality gates confidence
        gates_result = components.get('quality_gates', {})
        if gates_result.get('passed', False):
            confidence_factors.append(0.2)

        # Relevance scoring confidence
        relevance_result = components.get('relevance_scoring', {})
        avg_relevance = relevance_result.get('average_score', 0.0)
        confidence_factors.append(avg_relevance * 0.25)

        # Grounding validation confidence
        grounding_result = components.get('content_grounding', {}).get('result', {})
        grounding_confidence = 1.0 - grounding_result.get('hallucination_risk', 0.5)
        confidence_factors.append(grounding_confidence * 0.25)

        return sum(confidence_factors)

    def _determine_validation_result(self, overall_score: float, components: Dict[str, Any]) -> bool:
        """Determine validation result using adaptive logic"""
        # Base threshold that adapts
        base_threshold = 0.6

        # Adjust based on component-specific requirements
        quality_gates = components.get('quality_gates', {})
        grounding_result = components.get('content_grounding', {}).get('result', {})

        # Must pass quality gates for any acceptance
        if not quality_gates.get('passed', False):
            return False

        # Must be properly grounded
        if not grounding_result.get('is_grounded', False):
            return False

        # Check adaptive threshold
        learned_thresholds = self._get_learned_thresholds()
        adaptive_threshold = learned_thresholds.get('overall_threshold', base_threshold)

        return overall_score >= adaptive_threshold

    def _get_learned_thresholds(self) -> Dict[str, float]:
        """Get current learned thresholds from all components"""
        return {
            'overall_threshold': getattr(self, 'learned_overall_threshold', 0.6),
            'quality_gates': self.quality_gates.get_current_thresholds(),
            'relevance_weights': self.relevance_scorer.get_current_weights()
        }

    def _generate_comprehensive_rejection_reason(self, components: Dict[str, Any]) -> str:
        """Generate comprehensive rejection reason from all components"""
        reasons = []

        # Quality gates reasons
        quality_gates = components.get('quality_gates', {})
        if not quality_gates.get('passed', True):
            gate_results = quality_gates.get('results', {})
            reasons.append(self.quality_gates.get_rejection_reason(gate_results))

        # Grounding validation reasons
        grounding_result = components.get('content_grounding', {}).get('result', {})
        if grounding_result.get('rejection_reason'):
            reasons.append(grounding_result['rejection_reason'])

        # Relevance scoring reasons
        relevance_result = components.get('relevance_scoring', {})
        avg_relevance = relevance_result.get('average_score', 1.0)
        if avg_relevance < 0.5:
            reasons.append("poor document-query relevance matching")

        if not reasons:
            return "response does not meet adaptive quality standards"

        return f"Response rejected: {'; '.join(reasons)}"

    async def _record_validation_result(self, query: str, response: str,
                                      retrieved_docs: List[Dict], result: ValidationResult):
        """Record validation result for learning and analytics"""
        validation_record = {
            'timestamp': datetime.now().isoformat(),
            'query_length': len(query),
            'response_length': len(response),
            'doc_count': len(retrieved_docs),
            'overall_score': result.overall_score,
            'passed': result.passed,
            'confidence': result.confidence,
            'processing_time_ms': result.processing_time_ms,
            'components': {k: v for k, v in result.components.items() if k != 'error'}
        }

        self.validation_history.append(validation_record)

        # Update performance metrics
        self.performance_metrics.validation_count += 1

        # Calculate rolling acceptance rate (last 100 validations)
        recent_validations = self.validation_history[-100:]
        passed_count = sum(1 for v in recent_validations if v['passed'])
        self.performance_metrics.acceptance_rate = passed_count / len(recent_validations)

        # Update average processing time
        recent_times = [v['processing_time_ms'] for v in recent_validations]
        self.performance_metrics.avg_processing_time = sum(recent_times) / len(recent_times)

    async def _check_and_trigger_learning(self):
        """Check if learning should be triggered and execute if needed"""
        if not self.learning_enabled:
            return

        # Trigger learning based on validation count
        if self.performance_metrics.validation_count % self.learning_interval == 0:
            await self._execute_learning_update()

        # Trigger learning based on time interval
        time_since_last_update = datetime.now() - self.performance_metrics.last_learning_update
        if time_since_last_update > timedelta(hours=6):  # Learn every 6 hours
            await self._execute_learning_update()

    async def _execute_learning_update(self):
        """Execute comprehensive learning update across all components"""
        try:
            logger.info("Executing intelligent learning update",
                       validation_count=self.performance_metrics.validation_count,
                       acceptance_rate=self.performance_metrics.acceptance_rate)

            # 1. Update overall threshold based on performance
            await self._update_overall_threshold()

            # 2. Update component-specific learning
            await self._update_component_learning()

            # 3. Update performance metrics
            self.performance_metrics.last_learning_update = datetime.now()
            self.performance_metrics.threshold_adjustments += 1

            # 4. Save learning state
            await self._save_learning_state()

            logger.info("Learning update completed successfully")

        except Exception as e:
            logger.error("Learning update failed", error=str(e))

    async def _update_overall_threshold(self):
        """Update overall validation threshold based on performance patterns"""
        if len(self.validation_history) < 20:
            return

        # Analyze recent validation patterns
        recent_validations = self.validation_history[-100:]

        # Calculate optimal threshold based on quality distribution
        passed_scores = [v['overall_score'] for v in recent_validations if v['passed']]
        failed_scores = [v['overall_score'] for v in recent_validations if not v['passed']]

        if len(passed_scores) >= 10 and len(failed_scores) >= 5:
            # Find threshold that maximizes separation
            pass_avg = sum(passed_scores) / len(passed_scores)
            fail_avg = sum(failed_scores) / len(failed_scores)

            # Set threshold between averages with bias toward quality
            optimal_threshold = (pass_avg * 0.7 + fail_avg * 0.3)

            # Apply learning rate for gradual adaptation
            current_threshold = getattr(self, 'learned_overall_threshold', 0.6)
            learning_rate = 0.1

            self.learned_overall_threshold = (
                current_threshold * (1 - learning_rate) +
                optimal_threshold * learning_rate
            )

            # Ensure threshold stays within reasonable bounds
            self.learned_overall_threshold = max(0.4, min(0.8, self.learned_overall_threshold))

    async def _update_component_learning(self):
        """Update learning for individual components"""
        # Create feedback for quality gates
        for record in self.validation_history[-20:]:
            feedback = ResponseFeedback(
                query=f"validation_{record['timestamp']}",
                response_id=record['timestamp'],
                quality_score=record['overall_score'],
                user_satisfaction=1.0 if record['passed'] else 0.3,
                retrieval_score=record.get('components', {}).get('relevance_scoring', {}).get('average_score', 0.5),
                confidence=record['confidence']
            )
            self.quality_gates.record_feedback(feedback)

        # Update relevance scorer with feedback
        for record in self.validation_history[-10:]:
            if 'components' in record:
                relevance_data = record['components'].get('relevance_scoring', {})
                self.relevance_scorer.record_scoring_feedback(
                    query="learning_sample",
                    document={'id': record['timestamp']},
                    user_satisfaction=1.0 if record['passed'] else 0.3,
                    final_score=relevance_data.get('average_score', 0.0)
                )

    async def _save_learning_state(self):
        """Save current learning state and performance metrics"""
        try:
            learning_state = {
                'performance_metrics': asdict(self.performance_metrics),
                'learned_thresholds': self._get_learned_thresholds(),
                'validation_count': len(self.validation_history),
                'last_updated': datetime.now().isoformat()
            }

            # Save to persistent storage
            with open('/tmp/rag_intelligent_validation_state.json', 'w') as f:
                json.dump(learning_state, f, indent=2, default=str)

        except Exception as e:
            logger.error("Failed to save learning state", error=str(e))

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics and learning status"""
        return {
            'performance_metrics': asdict(self.performance_metrics),
            'learned_thresholds': self._get_learned_thresholds(),
            'validation_history_size': len(self.validation_history),
            'learning_enabled': self.learning_enabled
        }

    def configure_learning(self, enabled: bool = True, interval: int = 50):
        """Configure learning parameters"""
        self.learning_enabled = enabled
        self.learning_interval = interval

        logger.info("Learning configuration updated",
                   enabled=enabled,
                   interval=interval)


# ================== Factory Functions ==================

def create_unified_quality_system(config: Optional[Dict[str, Any]] = None) -> UnifiedQualitySystem:
    """Create unified quality system with configuration"""
    return UnifiedQualitySystem(config)

# Backward compatibility aliases
def create_intelligent_validator(config: Optional[Dict[str, Any]] = None) -> UnifiedQualitySystem:
    """Create unified quality system (backward compatibility)"""
    return UnifiedQualitySystem(config)

def create_quality_metrics_manager(config: Optional[Dict[str, Any]] = None) -> UnifiedQualitySystem:
    """Create unified quality system (backward compatibility)"""
    return UnifiedQualitySystem(config)

# Class alias for backward compatibility
IntelligentValidationSystem = UnifiedQualitySystem
QualityMetricsManager = UnifiedQualitySystem