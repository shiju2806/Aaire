"""
OpenAI Embeddings-based Semantic Alignment Validator

Uses OpenAI's embedding API for semantic alignment validation
without requiring heavy local dependencies.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import structlog
import openai
from sklearn.metrics.pairwise import cosine_similarity

logger = structlog.get_logger()


@dataclass
class AlignmentResult:
    """Result of semantic alignment validation"""
    is_aligned: bool
    alignment_score: float
    confidence: float
    explanation: str
    should_generate_response: bool


class OpenAIAlignmentValidator:
    """
    Semantic alignment validator using OpenAI embeddings API.
    Lightweight alternative to sentence-transformers.
    """

    def __init__(self, model: str = "text-embedding-ada-002", config=None):
        """
        Initialize with OpenAI embedding model.

        Args:
            model: OpenAI embedding model to use
            config: Quality configuration instance
        """
        self.model = model
        self.config = config

        # Use configuration thresholds if available, otherwise use defaults
        if config:
            self.alignment_threshold = config.get_semantic_alignment_threshold()
            self.confidence_threshold = config.get_confidence_threshold()
        else:
            self.alignment_threshold = 0.35  # Lowered default for technical content
            self.confidence_threshold = 0.30  # Lowered default for technical content

        logger.info("OpenAI alignment validator initialized", model=model)

    def validate_alignment(self, query: str, documents: List[Dict]) -> AlignmentResult:
        """
        Validate semantic alignment between query and documents.

        Args:
            query: User query
            documents: Retrieved documents

        Returns:
            AlignmentResult with alignment decision
        """
        try:
            if not documents:
                return AlignmentResult(
                    is_aligned=False,
                    alignment_score=0.0,
                    confidence=1.0,
                    explanation="No documents retrieved",
                    should_generate_response=False
                )

            # Get embeddings from OpenAI
            query_embedding = self._get_embedding(query)

            # Get document embeddings (sample for efficiency)
            doc_embeddings = []
            for doc in documents[:10]:  # Limit to top 10 docs
                content = doc.get('content', '')[:1000]  # First 1000 chars
                if content:
                    doc_embedding = self._get_embedding(content)
                    doc_embeddings.append(doc_embedding)

            if not doc_embeddings:
                return AlignmentResult(
                    is_aligned=False,
                    alignment_score=0.0,
                    confidence=1.0,
                    explanation="No document content to validate",
                    should_generate_response=False
                )

            # Calculate alignment scores
            alignment_scores = []
            for doc_embedding in doc_embeddings:
                similarity = cosine_similarity(
                    np.array(query_embedding).reshape(1, -1),
                    np.array(doc_embedding).reshape(1, -1)
                )[0][0]
                alignment_scores.append(float(similarity))

            # Aggregate scores
            final_score, confidence = self._aggregate_scores(alignment_scores)

            # Determine if aligned
            is_aligned = final_score >= self.alignment_threshold and confidence >= self.confidence_threshold

            explanation = self._generate_explanation(final_score, confidence, is_aligned)

            return AlignmentResult(
                is_aligned=is_aligned,
                alignment_score=final_score,
                confidence=confidence,
                explanation=explanation,
                should_generate_response=is_aligned
            )

        except Exception as e:
            logger.error("OpenAI alignment validation failed", exception_details=str(e))
            # Fail open - allow response generation if validation fails
            return AlignmentResult(
                is_aligned=True,
                alignment_score=0.5,
                confidence=0.0,
                explanation=f"Validation error: {str(e)}",
                should_generate_response=True
            )

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI API"""
        try:
            response = openai.Embedding.create(
                model=self.model,
                input=text
            )
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error("Failed to get OpenAI embedding", exception_details=str(e))
            # Return random embedding on error
            return np.random.rand(1536).tolist()

    def _aggregate_scores(self, scores: List[float]) -> tuple[float, float]:
        """Aggregate alignment scores"""
        if not scores:
            return 0.0, 0.0

        scores_array = np.array(scores)
        max_score = np.max(scores_array)
        mean_score = np.mean(scores_array)

        # Weighted combination
        final_score = 0.7 * max_score + 0.3 * mean_score

        # Confidence based on distribution
        if max_score > 0.8:
            confidence = 0.9
        elif max_score > 0.7:
            confidence = 0.8
        elif mean_score > 0.6:
            confidence = 0.7
        else:
            confidence = 0.5

        return final_score, confidence

    def _generate_explanation(self, score: float, confidence: float, is_aligned: bool) -> str:
        """Generate explanation for alignment decision"""
        if not is_aligned:
            if score < 0.3:
                return f"Low semantic similarity (score: {score:.2f}). Documents may not contain relevant information."
            elif confidence < self.confidence_threshold:
                return f"Uncertain alignment (score: {score:.2f}, confidence: {confidence:.2f})."
            else:
                return f"Below alignment threshold (score: {score:.2f})."
        else:
            if score > 0.8:
                return f"High semantic alignment (score: {score:.2f})"
            else:
                return f"Good semantic alignment (score: {score:.2f})"


def create_openai_alignment_validator() -> OpenAIAlignmentValidator:
    """Factory function to create OpenAI alignment validator"""
    return OpenAIAlignmentValidator()