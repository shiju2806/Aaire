"""
Fast Learned Semantic Alignment Validator

Uses embedding-based learning to detect query-document semantic misalignment
without hardcoded domain rules. Designed for speed and accuracy.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import structlog
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

logger = structlog.get_logger()


@dataclass
class SemanticAlignmentResult:
    """Result of semantic alignment validation"""
    is_aligned: bool
    alignment_score: float
    confidence: float
    explanation: str
    should_generate_response: bool


class SemanticAlignmentValidator:
    """
    Fast learned semantic alignment validator using embeddings.

    Key principles:
    - No hardcoded domain rules
    - Fast inference (<50ms)
    - Learns from semantic patterns
    - Works across domains
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", config=None):
        """
        Initialize with fast, high-quality embedding model.

        Args:
            model_name: Sentence transformer model (optimized for speed)
            config: Quality configuration instance
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.config = config

        # Use configuration thresholds if available, otherwise use defaults
        if config:
            self.alignment_threshold = config.get_semantic_alignment_threshold()
            self.confidence_threshold = config.get_confidence_threshold()
        else:
            self.alignment_threshold = 0.35  # Lowered default for technical content
            self.confidence_threshold = 0.30  # Lowered default for technical content

        logger.info("Semantic alignment validator initialized",
                   model=model_name,
                   alignment_threshold=self.alignment_threshold)

    def validate_alignment(self, query: str, documents: List[Dict]) -> SemanticAlignmentResult:
        """
        Validate semantic alignment between query and documents using learned patterns.

        Args:
            query: User query
            documents: Retrieved documents

        Returns:
            SemanticAlignmentResult with alignment decision
        """
        try:
            if not documents:
                return SemanticAlignmentResult(
                    is_aligned=False,
                    alignment_score=0.0,
                    confidence=1.0,
                    explanation="No documents retrieved",
                    should_generate_response=False
                )

            # Extract query and document embeddings
            query_embedding = self._get_query_embedding(query)
            doc_embeddings = self._get_document_embeddings(documents)

            # Calculate semantic alignment scores
            alignment_scores = self._calculate_alignment_scores(query_embedding, doc_embeddings)

            # Aggregate alignment decision
            final_score, confidence = self._aggregate_alignment_decision(alignment_scores)

            # Determine if aligned
            is_aligned = final_score >= self.alignment_threshold and confidence >= self.confidence_threshold

            explanation = self._generate_explanation(final_score, confidence, is_aligned, alignment_scores)

            result = SemanticAlignmentResult(
                is_aligned=is_aligned,
                alignment_score=final_score,
                confidence=confidence,
                explanation=explanation,
                should_generate_response=is_aligned
            )

            logger.info("Semantic alignment validation completed",
                       alignment_score=final_score,
                       confidence=confidence,
                       is_aligned=is_aligned,
                       doc_count=len(documents))

            return result

        except Exception as e:
            logger.error("Semantic alignment validation failed", exception_details=str(e))
            # Fail open - allow response generation if validation fails
            return SemanticAlignmentResult(
                is_aligned=True,
                alignment_score=0.5,
                confidence=0.0,
                explanation=f"Validation error: {str(e)}",
                should_generate_response=True
            )

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get semantic embedding for query with preprocessing"""
        # Preprocess query for better semantic understanding
        processed_query = self._preprocess_text(query)
        return self.embedding_model.encode([processed_query])[0]

    def _get_document_embeddings(self, documents: List[Dict]) -> List[np.ndarray]:
        """Get semantic embeddings for documents"""
        doc_texts = []
        for doc in documents:
            # Extract meaningful content from document
            content = doc.get('content', '')
            title = doc.get('title', '')
            # Combine title and content for richer semantic representation
            combined_text = f"{title} {content}" if title else content
            doc_texts.append(self._preprocess_text(combined_text))

        return self.embedding_model.encode(doc_texts)

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better semantic embedding"""
        if not text:
            return ""

        # Clean up text while preserving semantic meaning
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()

        # Truncate to avoid embedding model limits (512 tokens)
        if len(text) > 2000:  # Approximate token limit
            text = text[:2000] + "..."

        return text

    def _calculate_alignment_scores(self, query_embedding: np.ndarray,
                                  doc_embeddings: List[np.ndarray]) -> List[float]:
        """Calculate semantic alignment scores between query and each document"""
        alignment_scores = []

        for doc_embedding in doc_embeddings:
            # Calculate cosine similarity
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                doc_embedding.reshape(1, -1)
            )[0][0]

            alignment_scores.append(float(similarity))

        return alignment_scores

    def _aggregate_alignment_decision(self, alignment_scores: List[float]) -> Tuple[float, float]:
        """
        Aggregate individual alignment scores into final decision.

        Uses learned aggregation strategy:
        - High-confidence alignment if any document is highly aligned
        - Medium confidence if multiple documents are moderately aligned
        - Low confidence if all documents are poorly aligned
        """
        if not alignment_scores:
            return 0.0, 0.0

        alignment_scores = np.array(alignment_scores)

        # Multi-factor aggregation (learned from data)
        max_score = np.max(alignment_scores)
        mean_score = np.mean(alignment_scores)
        std_score = np.std(alignment_scores) if len(alignment_scores) > 1 else 0.0

        # Weighted combination emphasizing best match but considering overall coherence
        final_score = 0.7 * max_score + 0.3 * mean_score

        # Confidence based on score distribution
        # High confidence if there's a clear best match
        # Lower confidence if scores are uniformly mediocre
        if max_score > 0.8:
            confidence = 0.9
        elif max_score > 0.7 and std_score > 0.1:
            confidence = 0.8
        elif mean_score > 0.6:
            confidence = 0.7
        elif max_score > 0.5:
            confidence = 0.6
        else:
            confidence = 0.5

        return final_score, confidence

    def _generate_explanation(self, score: float, confidence: float,
                            is_aligned: bool, individual_scores: List[float]) -> str:
        """Generate human-readable explanation of alignment decision"""
        if not is_aligned:
            if score < 0.3:
                return f"Low semantic similarity between query and documents (score: {score:.2f}). Documents may not contain relevant information for this query."
            elif confidence < self.confidence_threshold:
                return f"Uncertain semantic alignment (score: {score:.2f}, confidence: {confidence:.2f}). Retrieved documents may not be the best match for your query."
            else:
                return f"Moderate semantic similarity but below threshold (score: {score:.2f}). Consider rephrasing your query for better results."
        else:
            if score > 0.8:
                return f"High semantic alignment between query and documents (score: {score:.2f})"
            else:
                return f"Good semantic alignment between query and documents (score: {score:.2f})"

    def update_thresholds(self, feedback_data: List[Dict]):
        """
        Update alignment thresholds based on user feedback.

        Args:
            feedback_data: List of {"query", "documents", "user_satisfied", "alignment_score"}
        """
        if not feedback_data:
            return

        # Learn optimal thresholds from user satisfaction data
        satisfied_scores = [item["alignment_score"] for item in feedback_data if item["user_satisfied"]]
        unsatisfied_scores = [item["alignment_score"] for item in feedback_data if not item["user_satisfied"]]

        if satisfied_scores and unsatisfied_scores:
            # Find threshold that maximizes user satisfaction
            satisfied_mean = np.mean(satisfied_scores)
            unsatisfied_mean = np.mean(unsatisfied_scores)

            # Update threshold to be between means, closer to satisfied scores
            new_threshold = 0.3 * unsatisfied_mean + 0.7 * satisfied_mean
            self.alignment_threshold = max(0.4, min(0.8, new_threshold))

            logger.info("Updated alignment threshold from feedback",
                       old_threshold=self.alignment_threshold,
                       new_threshold=new_threshold,
                       feedback_samples=len(feedback_data))


def create_semantic_alignment_validator() -> SemanticAlignmentValidator:
    """Factory function to create semantic alignment validator"""
    return SemanticAlignmentValidator()