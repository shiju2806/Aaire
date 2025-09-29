"""
Semantic Similarity Service

Replaces the broken entropy disambiguation with proper embedding-based semantic similarity.
This service can distinguish between semantically similar but distinct concepts
(e.g., whole life vs universal life) using vector embeddings.
"""

import asyncio
import structlog
import re
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = structlog.get_logger()


class SemanticSimilarityService:
    """
    Query-agnostic semantic similarity service using embeddings.
    Distinguishes between similar concepts without hardcoded rules.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with a lightweight, fast embedding model"""
        self.model_name = model_name
        self.model = None
        self.is_loaded = False
        logger.info(f"Initializing SemanticSimilarityService with model: {model_name}")

    def _load_model(self):
        """Lazy load the embedding model"""
        if not self.is_loaded:
            try:
                self.model = SentenceTransformer(self.model_name)
                self.is_loaded = True
                logger.info(f"Loaded embedding model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for query text"""
        if not self.is_loaded:
            self._load_model()

        try:
            embedding = self.model.encode([query])
            return embedding[0]
        except Exception as e:
            logger.error(f"Failed to encode query: {e}")
            return np.array([])

    def get_document_embeddings(self, documents: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple documents"""
        if not self.is_loaded:
            self._load_model()

        try:
            embeddings = self.model.encode(documents)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode documents: {e}")
            return []

    def calculate_semantic_scores(self, query: str, retrieved_docs: List[Dict]) -> List[Tuple[Dict, float]]:
        """
        Calculate semantic similarity scores with fine-grained disambiguation.
        Combines cosine similarity with term-specific boosting for precise differentiation.
        Returns list of (document, similarity_score) tuples.
        """
        if not retrieved_docs:
            return []

        # Extract document content
        doc_contents = []
        for doc in retrieved_docs:
            content = self._extract_document_content(doc)
            doc_contents.append(content[:500])  # Limit to 500 chars for performance

        # Get embeddings
        query_embedding = self.get_query_embedding(query)
        if query_embedding.size == 0:
            logger.warning("Failed to get query embedding, returning original order")
            return [(doc, 0.0) for doc in retrieved_docs]

        doc_embeddings = self.get_document_embeddings(doc_contents)
        if isinstance(doc_embeddings, list) and len(doc_embeddings) == 0:
            logger.warning("Failed to get document embeddings, returning original order")
            return [(doc, 0.0) for doc in retrieved_docs]
        elif hasattr(doc_embeddings, 'size') and doc_embeddings.size == 0:
            logger.warning("Failed to get document embeddings, returning original order")
            return [(doc, 0.0) for doc in retrieved_docs]

        # Calculate cosine similarity
        try:
            similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

            # Apply query-agnostic fine-grained disambiguation boosting
            enhanced_scores = []
            for i, (doc, base_score) in enumerate(zip(retrieved_docs, similarities)):
                content = doc_contents[i]

                # Calculate dynamic disambiguation boost
                disambiguation_boost = self._calculate_disambiguation_boost(query, content, doc_contents)

                # Combine semantic similarity with disambiguation boost
                # Base score (0.0-1.0) + disambiguation boost (-0.3 to +0.3)
                enhanced_score = float(base_score) + disambiguation_boost

                # Ensure score stays within reasonable bounds
                enhanced_score = max(0.0, min(1.0, enhanced_score))

                enhanced_scores.append((doc, enhanced_score))

                logger.debug(f"Doc {i+1}: base={base_score:.3f}, boost={disambiguation_boost:.3f}, final={enhanced_score:.3f}")

            # Sort by enhanced score (descending)
            enhanced_scores.sort(key=lambda x: x[1], reverse=True)

            logger.info(f"Calculated enhanced semantic scores for {len(enhanced_scores)} documents")
            logger.debug(f"Top 3 enhanced scores: {[score for _, score in enhanced_scores[:3]]}")

            return enhanced_scores

        except Exception as e:
            logger.error(f"Failed to calculate similarities: {e}")
            return [(doc, 0.0) for doc in retrieved_docs]

    def _extract_document_content(self, doc: Dict[str, Any]) -> str:
        """Extract text content from document for embedding"""
        try:
            # Try various content field locations
            if isinstance(doc, dict):
                content_candidates = [
                    doc.get('content'),
                    doc.get('text'),
                    doc.get('_node_content'),
                ]

                # Try metadata fields
                metadata = doc.get('metadata', {})
                if metadata:
                    content_candidates.extend([
                        metadata.get('content'),
                        metadata.get('text'),
                        metadata.get('_node_content'),
                    ])

                # Return first valid content
                for candidate in content_candidates:
                    if candidate and isinstance(candidate, str) and len(candidate.strip()) > 10:
                        return candidate.strip()

                # Fallback to string representation
                return str(doc)
            else:
                return str(doc)

        except Exception as e:
            logger.warning(f"Error extracting document content: {e}")
            return str(doc)

    def enhance_retrieval_with_semantic_similarity(self,
                                                  query: str,
                                                  retrieved_docs: List[Dict],
                                                  similarity_threshold: float = 0.3) -> List[Dict]:
        """
        Main method to enhance retrieval results using semantic similarity.

        Args:
            query: Original search query
            retrieved_docs: Documents from initial retrieval
            similarity_threshold: Minimum similarity score to include document

        Returns:
            Reordered and filtered documents based on semantic similarity
        """
        if not retrieved_docs:
            return retrieved_docs

        start_time = asyncio.get_event_loop().time()

        try:
            # Calculate semantic scores
            doc_scores = self.calculate_semantic_scores(query, retrieved_docs)

            # Filter by similarity threshold and preserve metadata
            enhanced_docs = []
            for doc, score in doc_scores:
                if score >= similarity_threshold:
                    # Add semantic score to document metadata
                    enhanced_doc = doc.copy() if isinstance(doc, dict) else doc
                    if isinstance(enhanced_doc, dict):
                        if 'metadata' not in enhanced_doc:
                            enhanced_doc['metadata'] = {}
                        enhanced_doc['metadata']['semantic_similarity_score'] = score
                        enhanced_doc['semantic_score'] = score  # Also add at root level

                    enhanced_docs.append(enhanced_doc)

            end_time = asyncio.get_event_loop().time()
            processing_time = (end_time - start_time) * 1000  # Convert to milliseconds

            logger.info(f"Semantic similarity enhancement completed",
                       original_count=len(retrieved_docs),
                       enhanced_count=len(enhanced_docs),
                       processing_time_ms=round(processing_time, 2),
                       threshold=similarity_threshold)

            return enhanced_docs

        except Exception as e:
            logger.error(f"Error in semantic similarity enhancement: {e}")
            # Return original documents on error
            return retrieved_docs

    def detect_specificity_mismatch(self, query: str, retrieved_docs: List[Dict]) -> Dict[str, Any]:
        """
        Detect when query specificity doesn't match document specificity.
        Used for content sufficiency analysis.
        """
        if not retrieved_docs:
            return {"mismatch_detected": False, "confidence": 0.0}

        try:
            doc_scores = self.calculate_semantic_scores(query, retrieved_docs[:5])  # Top 5 for speed

            if not doc_scores:
                return {"mismatch_detected": False, "confidence": 0.0}

            # Analyze score distribution
            scores = [score for _, score in doc_scores]
            avg_score = np.mean(scores)
            max_score = max(scores)

            # Detect mismatch patterns
            mismatch_detected = False
            confidence = 0.0

            # Pattern 1: All scores are low (< 0.4)
            if max_score < 0.4:
                mismatch_detected = True
                confidence = 0.8
                reason = "Low semantic similarity across all documents"

            # Pattern 2: High variance in scores (some relevant, some not)
            elif len(scores) > 1 and np.std(scores) > 0.2 and avg_score < 0.5:
                mismatch_detected = True
                confidence = 0.6
                reason = "Mixed document relevance suggests specificity mismatch"

            else:
                reason = "Good semantic alignment detected"

            return {
                "mismatch_detected": mismatch_detected,
                "confidence": confidence,
                "reason": reason,
                "avg_similarity": float(avg_score),
                "max_similarity": float(max_score),
                "score_variance": float(np.std(scores)) if len(scores) > 1 else 0.0
            }

        except Exception as e:
            logger.error(f"Error in specificity mismatch detection: {e}")
            return {"mismatch_detected": False, "confidence": 0.0}


    def _calculate_disambiguation_boost(self, query: str, doc_content: str, all_docs_content: List[str]) -> float:
        """
        Calculate query-agnostic disambiguation boost using dynamic contrastive analysis.

        Instead of hardcoded patterns, this analyzes what makes documents different
        from each other and boosts documents that align with the query's specific intent.
        """
        # Extract noun phrases and entities from query - these are likely to be differentiating terms
        query_terms = self._extract_discriminative_terms(query)

        if not query_terms:
            return 0.0

        boost = 0.0

        # For each potential discriminative term in the query
        for term in query_terms:
            term_lower = term.lower()
            doc_content_lower = doc_content.lower()

            # Check if this document contains the specific term
            term_in_doc = term_lower in doc_content_lower

            # Calculate how discriminative this term is across all documents
            discrimination_power = self._calculate_term_discrimination_power(term_lower, all_docs_content)

            if term_in_doc and discrimination_power > 0.3:  # Only boost if term is discriminative
                # Boost is proportional to how well this term discriminates
                term_boost = discrimination_power * 0.2  # Max 0.2 boost per term
                boost += term_boost
                logger.debug(f"Discriminative term boost '{term}': +{term_boost:.3f} (power: {discrimination_power:.3f})")

            elif not term_in_doc and discrimination_power > 0.3:
                # Small penalty if document lacks a discriminative query term
                term_penalty = discrimination_power * -0.1  # Max -0.1 penalty per term
                boost += term_penalty
                logger.debug(f"Missing discriminative term '{term}': {term_penalty:.3f}")

        # Cap the total boost/penalty
        boost = max(-0.3, min(0.3, boost))

        return boost

    def _extract_discriminative_terms(self, query: str) -> List[str]:
        """
        Extract discriminative terms using dynamic NLP-based n-gram analysis.
        NO HARDCODING - extracts all meaningful n-grams and ranks by semantic importance.
        """
        import re

        query_lower = query.lower()
        discriminative_terms = []

        # Extract ALL n-grams (1 to 3 words) dynamically
        words = re.findall(r'\b[a-zA-Z0-9-]+\b', query_lower)

        # Generate all meaningful n-grams (unigrams, bigrams, trigrams)
        for n in range(3, 0, -1):  # Start with trigrams (most specific)
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])

                # Skip if it's just stop words
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'how', 'do', 'i', 'what', 'is', 'are'}
                ngram_words = ngram.split()
                if all(word in stop_words for word in ngram_words):
                    continue

                # Skip if this n-gram is contained in a longer term we already have
                is_substring = any(ngram in existing_term.lower() and ngram != existing_term.lower()
                                 for existing_term in discriminative_terms)
                if not is_substring:
                    discriminative_terms.append(ngram)

        # Extract technical patterns (numbers, codes, acronyms) - these are always discriminative
        technical_terms = re.findall(r'\b[A-Z]*[0-9]+[A-Z0-9-]*\b|\b[A-Z]{2,}(?:\s+[0-9]+)?\b', query)
        discriminative_terms.extend(technical_terms)

        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in discriminative_terms:
            term_key = term.lower().strip()
            if term_key not in seen and len(term_key) > 1:
                seen.add(term_key)
                unique_terms.append(term)

        logger.debug(f"Extracted {len(unique_terms)} discriminative n-grams: {unique_terms[:5]}...")
        return unique_terms

    def _calculate_term_discrimination_power(self, term: str, all_docs_content: List[str]) -> float:
        """
        Calculate how well a term discriminates between documents.
        Uses EXACT phrase matching for multi-word terms to distinguish similar concepts.
        Returns value between 0 (no discrimination) and 1 (perfect discrimination).
        """
        if not all_docs_content:
            return 0.0

        # For multi-word terms, require EXACT phrase match (not just word presence)
        term_lower = term.lower()
        is_multiword = ' ' in term_lower

        if is_multiword:
            # Count exact phrase occurrences
            docs_with_exact_phrase = sum(1 for doc in all_docs_content if term_lower in doc.lower())

            # Also count "partial matches" where words appear but not as exact phrase
            term_words = set(term_lower.split())
            docs_with_partial_match = sum(
                1 for doc in all_docs_content
                if term_lower not in doc.lower() and all(word in doc.lower() for word in term_words)
            )

            total_docs = len(all_docs_content)

            if docs_with_exact_phrase == 0:
                return 0.0  # Term doesn't appear anywhere

            if docs_with_exact_phrase == total_docs:
                return 0.0  # Appears in all docs, no discrimination

            # Calculate discrimination power based on exact vs partial matches
            exact_frequency = docs_with_exact_phrase / total_docs

            # Multi-word terms that appear in few documents are HIGHLY discriminative
            if exact_frequency <= 0.1:
                return 1.0  # Very rare exact phrase = maximum discrimination
            elif exact_frequency <= 0.2:
                return 0.9  # Rare exact phrase = high discrimination
            elif exact_frequency <= 0.3:
                return 0.7  # Somewhat rare = good discrimination
            else:
                return 0.5  # Common but still useful for multi-word terms

        else:
            # Single word terms - use standard IDF approach
            docs_with_term = sum(1 for doc in all_docs_content if term_lower in doc.lower())
            total_docs = len(all_docs_content)

            if docs_with_term == 0 or docs_with_term == total_docs:
                return 0.0

            term_frequency = docs_with_term / total_docs

            if term_frequency <= 0.1:
                return 1.0
            elif term_frequency <= 0.3:
                return 0.8
            elif term_frequency <= 0.5:
                return 0.4
            else:
                return 0.1


def create_semantic_similarity_service(model_name: str = "all-MiniLM-L6-v2") -> SemanticSimilarityService:
    """Factory function to create semantic similarity service"""
    return SemanticSimilarityService(model_name=model_name)