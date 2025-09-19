"""
RAG Response Data Model
Contains the response structure for RAG pipeline outputs
"""
from typing import List, Dict, Optional


class RAGResponse:
    """
    Structured response from RAG pipeline

    Attributes:
        answer: The generated answer text
        citations: List of citation dictionaries with source information
        confidence: Confidence score (0-1) for the response
        session_id: Unique session identifier
        follow_up_questions: Optional list of suggested follow-up questions
        quality_metrics: Optional dictionary of quality metrics
    """

    def __init__(
        self,
        answer: str,
        citations: List[Dict],
        confidence: float,
        session_id: str,
        follow_up_questions: List[str] = None,
        quality_metrics: Dict[str, float] = None
    ):
        self.answer = answer
        self.citations = citations
        self.confidence = confidence
        self.session_id = session_id
        self.follow_up_questions = follow_up_questions or []
        self.quality_metrics = quality_metrics or {}

    def to_dict(self) -> Dict:
        """Convert response to dictionary format"""
        return {
            'answer': self.answer,
            'citations': self.citations,
            'confidence': self.confidence,
            'session_id': self.session_id,
            'follow_up_questions': self.follow_up_questions,
            'quality_metrics': self.quality_metrics
        }

    def __repr__(self) -> str:
        return f"RAGResponse(session_id={self.session_id}, confidence={self.confidence:.2f})"