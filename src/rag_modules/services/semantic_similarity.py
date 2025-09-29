"""
Semantic Similarity Service

Two-stage retrieval system:
1. Bi-encoder (fast) for initial retrieval
2. Cross-encoder (accurate) for reranking to distinguish similar concepts
   like "whole life" vs "universal life"
"""

import asyncio
import structlog
import re
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = structlog.get_logger()


class SemanticSimilarityService:
    """
    Two-stage semantic similarity service:
    - Stage 1: Fast bi-encoder for initial retrieval (semantic search)
    - Stage 2: Accurate cross-encoder for reranking (distinguishes nuances)

    This approach distinguishes similar-but-different concepts without hardcoding.
    """

    def __init__(self,
                 bi_encoder_model: str = "all-MiniLM-L6-v2",
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 use_cross_encoder: bool = True):
        """
        Initialize with bi-encoder (fast) and cross-encoder (accurate)

        Args:
            bi_encoder_model: Fast model for initial retrieval
            cross_encoder_model: Accurate model for reranking
            use_cross_encoder: Enable cross-encoder reranking (recommended)
        """
        self.bi_encoder_model_name = bi_encoder_model
        self.cross_encoder_model_name = cross_encoder_model
        self.use_cross_encoder = use_cross_encoder

        self.bi_encoder = None
        self.cross_encoder = None
        self.is_loaded = False

        logger.info(f"Initializing SemanticSimilarityService",
                   bi_encoder=bi_encoder_model,
                   cross_encoder=cross_encoder_model if use_cross_encoder else "disabled")

    def _load_models(self):
        """Lazy load both bi-encoder and cross-encoder models"""
        if not self.is_loaded:
            try:
                # Load bi-encoder (fast, for initial retrieval)
                logger.info(f"Loading bi-encoder: {self.bi_encoder_model_name}")
                self.bi_encoder = SentenceTransformer(self.bi_encoder_model_name)
                logger.info(f"✅ Bi-encoder loaded: {self.bi_encoder_model_name}")

                # Load cross-encoder (accurate, for reranking)
                if self.use_cross_encoder:
                    logger.info(f"Loading cross-encoder: {self.cross_encoder_model_name}")
                    self.cross_encoder = CrossEncoder(self.cross_encoder_model_name)
                    logger.info(f"✅ Cross-encoder loaded: {self.cross_encoder_model_name}")

                self.is_loaded = True
                logger.info("✅ All models loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load models: {e}")
                raise

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for query text using bi-encoder"""
        if not self.is_loaded:
            self._load_models()

        try:
            embedding = self.bi_encoder.encode([query])
            return embedding[0]
        except Exception as e:
            logger.error(f"Failed to encode query: {e}")
            return np.array([])

    def get_document_embeddings(self, documents: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple documents using bi-encoder"""
        if not self.is_loaded:
            self._load_models()

        try:
            embeddings = self.bi_encoder.encode(documents)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode documents: {e}")
            return []

    def rerank_with_cross_encoder(self, query: str, retrieved_docs: List[Dict], top_k: int = None) -> List[Tuple[Dict, float]]:
        """
        Rerank documents using cross-encoder for precise similarity scoring.

        Cross-encoders process query and document together, allowing them to
        distinguish nuanced differences like "whole life" vs "universal life".

        Args:
            query: Search query
            retrieved_docs: Documents from initial retrieval
            top_k: Number of top results to return (None = return all)

        Returns:
            List of (document, cross_encoder_score) tuples, sorted by score
        """
        if not self.use_cross_encoder or not retrieved_docs:
            logger.warning("Cross-encoder disabled or no documents, skipping reranking")
            return [(doc, 0.0) for doc in retrieved_docs]

        if not self.is_loaded:
            self._load_models()

        try:
            # Prepare query-document pairs for cross-encoder
            doc_contents = []
            for doc in retrieved_docs:
                content = self._extract_document_content(doc)
                # Limit content length for performance (cross-encoder is slower)
                doc_contents.append(content[:512])

            # Create pairs: [(query, doc1), (query, doc2), ...]
            query_doc_pairs = [[query, content] for content in doc_contents]

            # Get cross-encoder scores
            logger.info(f"🔄 Reranking {len(query_doc_pairs)} documents with cross-encoder")
            cross_scores = self.cross_encoder.predict(query_doc_pairs)

            # Combine documents with their cross-encoder scores
            reranked = list(zip(retrieved_docs, cross_scores))

            # Sort by cross-encoder score (descending)
            reranked.sort(key=lambda x: x[1], reverse=True)

            # Log top scores for debugging
            if reranked:
                top_3_scores = [float(score) for _, score in reranked[:3]]
                logger.info(f"✅ Cross-encoder reranking complete. Top 3 scores: {top_3_scores}")

            # Return top_k results if specified
            if top_k:
                reranked = reranked[:top_k]

            return reranked

        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            # Fallback to original order
            return [(doc, 0.0) for doc in retrieved_docs]

    def calculate_semantic_scores(self, query: str, retrieved_docs: List[Dict]) -> List[Tuple[Dict, float]]:
        """
        Two-stage scoring:
        1. Fast bi-encoder for initial similarity (if not already done)
        2. Accurate cross-encoder for reranking (distinguishes nuances)

        Returns list of (document, similarity_score) tuples.
        """
        if not retrieved_docs:
            return []

        # Stage 1: Bi-encoder scoring (fast, semantic similarity)
        logger.info(f"Stage 1: Bi-encoder semantic similarity for {len(retrieved_docs)} documents")

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

            # Combine documents with bi-encoder scores
            bi_encoder_results = list(zip(retrieved_docs, similarities))

            logger.info(f"✅ Stage 1 complete. Bi-encoder scores calculated.")

            # Stage 2: Cross-encoder reranking (accurate, distinguishes nuances)
            if self.use_cross_encoder:
                logger.info(f"Stage 2: Cross-encoder reranking for precise disambiguation")

                # Rerank ALL results with cross-encoder for maximum accuracy
                reranked_results = self.rerank_with_cross_encoder(query, retrieved_docs)

                logger.info(f"✅ Stage 2 complete. Cross-encoder reranking applied.")
                return reranked_results
            else:
                # Cross-encoder disabled, return bi-encoder results
                bi_encoder_results.sort(key=lambda x: x[1], reverse=True)
                return bi_encoder_results

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
                                                  top_k: int = None) -> List[Dict]:
        """
        Main method to enhance retrieval results using cross-encoder reranking.

        Args:
            query: Original search query
            retrieved_docs: Documents from initial retrieval
            top_k: Number of top documents to return (None = return all, reranked)

        Returns:
            Reordered documents based on cross-encoder scores (no filtering by threshold)

        Note: Cross-encoder scores are NOT normalized to [0,1] and can be negative.
              We rerank ALL documents and optionally keep top_k.
        """
        if not retrieved_docs:
            return retrieved_docs

        start_time = asyncio.get_event_loop().time()

        try:
            # Calculate semantic scores (includes cross-encoder reranking)
            doc_scores = self.calculate_semantic_scores(query, retrieved_docs)

            # Rerank and preserve metadata (NO threshold filtering!)
            enhanced_docs = []
            for doc, score in doc_scores:
                # Add semantic score to document metadata
                enhanced_doc = doc.copy() if isinstance(doc, dict) else doc
                if isinstance(enhanced_doc, dict):
                    if 'metadata' not in enhanced_doc:
                        enhanced_doc['metadata'] = {}
                    enhanced_doc['metadata']['semantic_similarity_score'] = float(score)
                    enhanced_doc['semantic_score'] = float(score)  # Also add at root level

                enhanced_docs.append(enhanced_doc)

            # Optionally limit to top_k documents
            if top_k and top_k < len(enhanced_docs):
                enhanced_docs = enhanced_docs[:top_k]

            end_time = asyncio.get_event_loop().time()
            processing_time = (end_time - start_time) * 1000  # Convert to milliseconds

            logger.info(f"Cross-encoder reranking completed",
                       original_count=len(retrieved_docs),
                       reranked_count=len(enhanced_docs),
                       processing_time_ms=round(processing_time, 2),
                       top_k=top_k if top_k else "all")

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


def create_semantic_similarity_service(
    bi_encoder_model: str = "all-MiniLM-L6-v2",
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    use_cross_encoder: bool = True
) -> SemanticSimilarityService:
    """
    Factory function to create semantic similarity service with cross-encoder reranking.

    Args:
        bi_encoder_model: Fast model for initial retrieval
        cross_encoder_model: Accurate model for reranking (distinguishes nuances)
        use_cross_encoder: Enable cross-encoder reranking (recommended for production)

    Returns:
        Configured SemanticSimilarityService instance
    """
    return SemanticSimilarityService(
        bi_encoder_model=bi_encoder_model,
        cross_encoder_model=cross_encoder_model,
        use_cross_encoder=use_cross_encoder
    )