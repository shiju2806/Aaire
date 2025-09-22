"""
Adaptive Quality Gates for RAG Pipeline

Industry-grade validation system with ML-based adaptive thresholds and
dynamic quality gates that learn from user feedback and response patterns.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import structlog
from pathlib import Path

logger = structlog.get_logger()


@dataclass
class ResponseFeedback:
    """Structure for capturing response quality feedback"""
    query: str
    response_id: str
    quality_score: float
    user_satisfaction: Optional[float] = None
    retrieval_score: float = 0.0
    confidence: float = 0.0
    timestamp: datetime = None
    feedback_type: str = "implicit"  # implicit, explicit, validation_error

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class QualityGate:
    """Dynamic quality gate configuration"""
    name: str
    current_threshold: float
    learned_threshold: float
    confidence_level: float
    sample_count: int
    last_updated: datetime
    gate_type: str  # 'retrieval', 'confidence', 'relevance', 'grounding'


class AdaptiveQualityGates:
    """
    ML-based adaptive quality gates that learn optimal thresholds from response patterns
    and user feedback without hardcoded logic.
    """

    def __init__(self, learning_data_path: str = "/tmp/rag_learning_data.json"):
        """
        Initialize adaptive quality gates system.

        Args:
            learning_data_path: Path to store learning data and thresholds
        """
        self.learning_data_path = Path(learning_data_path)
        self.feedback_history: List[ResponseFeedback] = []
        self.quality_gates: Dict[str, QualityGate] = {}

        # Initialize base gates with adaptive starting points
        self._initialize_gates()
        self._load_learning_data()

    def _initialize_gates(self):
        """Initialize quality gates with data-driven starting points"""
        now = datetime.now()

        # Start with conservative thresholds that adapt based on data
        self.quality_gates = {
            "retrieval_similarity": QualityGate(
                name="retrieval_similarity",
                current_threshold=0.7,  # Start conservative
                learned_threshold=0.7,
                confidence_level=0.0,
                sample_count=0,
                last_updated=now,
                gate_type="retrieval"
            ),
            "response_confidence": QualityGate(
                name="response_confidence",
                current_threshold=0.6,  # Start conservative
                learned_threshold=0.6,
                confidence_level=0.0,
                sample_count=0,
                last_updated=now,
                gate_type="confidence"
            ),
            "document_relevance": QualityGate(
                name="document_relevance",
                current_threshold=0.65,  # Start conservative
                learned_threshold=0.65,
                confidence_level=0.0,
                sample_count=0,
                last_updated=now,
                gate_type="relevance"
            ),
            "content_grounding": QualityGate(
                name="content_grounding",
                current_threshold=0.5,  # Restored proper threshold after fixing root cause
                learned_threshold=0.5,
                confidence_level=0.0,
                sample_count=0,
                last_updated=now,
                gate_type="grounding"
            )
        }

    def evaluate_quality_gates(self, metrics: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate if response passes all quality gates using adaptive thresholds.

        Args:
            metrics: Quality metrics from retrieval and generation process

        Returns:
            Tuple of (passes_gates, gate_results)
        """
        gate_results = {}
        overall_pass = True

        try:
            # Evaluate retrieval similarity gate
            retrieval_scores = metrics.get('retrieval_scores', [])
            if retrieval_scores:
                max_retrieval_score = max(retrieval_scores)
                gate_results['retrieval_similarity'] = self._evaluate_gate(
                    "retrieval_similarity", max_retrieval_score
                )
                if not gate_results['retrieval_similarity']['passed']:
                    overall_pass = False

            # Evaluate response confidence gate
            confidence = metrics.get('confidence', 0.0)
            gate_results['response_confidence'] = self._evaluate_gate(
                "response_confidence", confidence
            )
            if not gate_results['response_confidence']['passed']:
                overall_pass = False

            # Evaluate document relevance gate
            source_quality = metrics.get('source_quality', 0.0)
            gate_results['document_relevance'] = self._evaluate_gate(
                "document_relevance", source_quality
            )
            if not gate_results['document_relevance']['passed']:
                overall_pass = False

            # Content grounding evaluation delegated to grounding_validator
            # which provides comprehensive weighted analysis
            grounding_score = metrics.get('grounding_score', 0.0)
            gate_results['content_grounding'] = self._evaluate_gate(
                "content_grounding", grounding_score
            )
            if not gate_results['content_grounding']['passed']:
                overall_pass = False

            # Log gate evaluation
            logger.info("Quality gates evaluated",
                       overall_pass=overall_pass,
                       gate_results={k: v['passed'] for k, v in gate_results.items()})

            return overall_pass, gate_results

        except Exception as e:
            logger.error("Quality gate evaluation failed", error=str(e))
            return False, {"error": "Gate evaluation failed"}

    def _evaluate_gate(self, gate_name: str, score: float) -> Dict[str, Any]:
        """Evaluate individual quality gate"""
        gate = self.quality_gates.get(gate_name)
        if not gate:
            return {"passed": True, "reason": "Gate not found"}

        passed = score >= gate.current_threshold

        return {
            "passed": passed,
            "score": score,
            "threshold": gate.current_threshold,
            "confidence": gate.confidence_level,
            "gate_type": gate.gate_type
        }

    # Grounding score calculation removed - delegated to grounding_validator
    # which provides comprehensive weighted analysis with evidence coverage,
    # numerical grounding, concept alignment, and source attribution

    def record_feedback(self, feedback: ResponseFeedback):
        """Record feedback for threshold learning"""
        self.feedback_history.append(feedback)

        # Trigger learning if we have enough samples
        if len(self.feedback_history) >= 10:
            self._update_thresholds()

    def _update_thresholds(self):
        """Update thresholds based on accumulated feedback using ML techniques"""
        if len(self.feedback_history) < 5:
            return

        try:
            # Group feedback by quality score ranges
            high_quality = [f for f in self.feedback_history if f.quality_score >= 0.8]
            low_quality = [f for f in self.feedback_history if f.quality_score < 0.6]

            # Update retrieval similarity threshold
            self._learn_threshold_from_data("retrieval_similarity", high_quality, low_quality)

            # Update confidence threshold
            self._learn_threshold_from_data("response_confidence", high_quality, low_quality)

            # Update relevance threshold
            self._learn_threshold_from_data("document_relevance", high_quality, low_quality)

            # Update grounding threshold
            self._learn_threshold_from_data("content_grounding", high_quality, low_quality)

            # Save learning data
            self._save_learning_data()

            logger.info("Thresholds updated from learning data",
                       sample_count=len(self.feedback_history),
                       high_quality_samples=len(high_quality),
                       low_quality_samples=len(low_quality))

        except Exception as e:
            logger.error("Threshold learning failed", error=str(e))

    def _learn_threshold_from_data(self, gate_name: str, high_quality: List[ResponseFeedback],
                                  low_quality: List[ResponseFeedback]):
        """Learn optimal threshold for a specific gate from quality data"""
        gate = self.quality_gates.get(gate_name)
        if not gate:
            return

        # Extract scores for this gate type
        high_scores = []
        low_scores = []

        for feedback in high_quality:
            if gate_name == "retrieval_similarity":
                high_scores.append(feedback.retrieval_score)
            elif gate_name == "response_confidence":
                high_scores.append(feedback.confidence)
            elif gate_name == "document_relevance":
                high_scores.append(feedback.retrieval_score * 0.9)  # Proxy metric
            elif gate_name == "content_grounding":
                high_scores.append(feedback.quality_score)

        for feedback in low_quality:
            if gate_name == "retrieval_similarity":
                low_scores.append(feedback.retrieval_score)
            elif gate_name == "response_confidence":
                low_scores.append(feedback.confidence)
            elif gate_name == "document_relevance":
                low_scores.append(feedback.retrieval_score * 0.9)
            elif gate_name == "content_grounding":
                low_scores.append(feedback.quality_score)

        # Calculate optimal threshold using statistical methods
        if high_scores and low_scores:
            # Find threshold that maximizes separation between high and low quality
            high_avg = np.mean(high_scores)
            low_avg = np.mean(low_scores)

            # Set threshold at point that minimizes false positives/negatives
            optimal_threshold = (high_avg + low_avg) / 2

            # Apply learning rate for gradual adaptation
            learning_rate = 0.3
            new_threshold = (gate.current_threshold * (1 - learning_rate) +
                           optimal_threshold * learning_rate)

            # Update gate with bounds checking
            gate.learned_threshold = max(0.1, min(0.95, new_threshold))
            gate.current_threshold = gate.learned_threshold
            gate.confidence_level = min(1.0, len(high_scores + low_scores) / 20.0)
            gate.sample_count = len(high_scores + low_scores)
            gate.last_updated = datetime.now()

            logger.info(f"Learned threshold for {gate_name}",
                       old_threshold=gate.current_threshold,
                       new_threshold=gate.learned_threshold,
                       confidence=gate.confidence_level)

    def get_rejection_reason(self, gate_results: Dict[str, Any]) -> str:
        """Generate human-readable rejection reason based on failed gates"""
        failed_gates = [name for name, result in gate_results.items()
                       if isinstance(result, dict) and not result.get('passed', True)]

        if not failed_gates:
            return "Response meets all quality standards"

        reasons = []
        for gate in failed_gates:
            if gate == "retrieval_similarity":
                reasons.append("insufficient document relevance")
            elif gate == "response_confidence":
                reasons.append("low confidence in response accuracy")
            elif gate == "document_relevance":
                reasons.append("poor source document quality")
            elif gate == "content_grounding":
                reasons.append("response not sufficiently grounded in sources")

        return f"Response rejected due to: {', '.join(reasons)}"

    def _save_learning_data(self):
        """Save learning data and current thresholds"""
        try:
            data = {
                'gates': {name: asdict(gate) for name, gate in self.quality_gates.items()},
                'feedback_count': len(self.feedback_history),
                'last_updated': datetime.now().isoformat()
            }

            # Convert datetime objects to strings for JSON serialization
            for gate_data in data['gates'].values():
                gate_data['last_updated'] = gate_data['last_updated'].isoformat()

            with open(self.learning_data_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error("Failed to save learning data", error=str(e))

    def _load_learning_data(self):
        """Load previously learned thresholds"""
        try:
            if not self.learning_data_path.exists():
                return

            with open(self.learning_data_path, 'r') as f:
                data = json.load(f)

            # Restore gates from saved data
            for gate_name, gate_data in data.get('gates', {}).items():
                if gate_name in self.quality_gates:
                    gate_data['last_updated'] = datetime.fromisoformat(gate_data['last_updated'])
                    self.quality_gates[gate_name] = QualityGate(**gate_data)

            logger.info("Loaded learned thresholds",
                       gates_loaded=len(data.get('gates', {})),
                       feedback_count=data.get('feedback_count', 0))

        except Exception as e:
            logger.error("Failed to load learning data", error=str(e))

    def get_current_thresholds(self) -> Dict[str, float]:
        """Get current adaptive thresholds for monitoring"""
        return {name: gate.current_threshold for name, gate in self.quality_gates.items()}

    def reset_learning(self):
        """Reset learning data (for testing or retraining)"""
        self.feedback_history.clear()
        self._initialize_gates()
        if self.learning_data_path.exists():
            self.learning_data_path.unlink()


def create_adaptive_quality_gates(learning_data_path: str = "/tmp/rag_learning_data.json") -> AdaptiveQualityGates:
    """Factory function to create adaptive quality gates"""
    return AdaptiveQualityGates(learning_data_path)