"""
Domain-Agnostic NLP Query Processor
Uses spaCy for semantic understanding and entity extraction without hardcoded rules
"""

import re
import structlog
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

try:
    import spacy
    from spacy.matcher import PhraseMatcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

logger = structlog.get_logger()

@dataclass
class ProcessedQuery:
    """Represents a processed query with extracted semantic information"""
    original_query: str
    key_entities: List[str]
    key_phrases: List[str]
    semantic_keywords: List[str]
    query_intent: str
    processed_tokens: List[str]

class NLPQueryProcessor:
    """Domain-agnostic NLP query processor using lightweight NLP techniques"""

    def __init__(self):
        """Initialize the NLP processor with spaCy if available"""
        self.nlp = None
        self.phrase_matcher = None

        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load('en_core_web_sm')
                self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr='LEMMA')
                logger.info("spaCy model loaded successfully")
            except OSError:
                logger.warning("spaCy model not found, falling back to basic NLP")
                self.nlp = None
        else:
            logger.warning("spaCy not available, using basic NLP processing")

        # Fallback stop words if spaCy not available
        self.fallback_stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these', 'those',
            'i', 'me', 'my', 'myself', 'we', 'us', 'our', 'you', 'your', 'he', 'him',
            'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their', 'be', 'is', 'am',
            'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'will',
            'would', 'should', 'could', 'can', 'may', 'might'
        }

    def _get_lemma(self, word: str) -> str:
        """Get lemma using spaCy or fallback to lowercase"""
        if self.nlp:
            doc = self.nlp(word)
            return doc[0].lemma_.lower() if doc else word.lower()
        return word.lower()

    def _is_stop_word(self, word: str) -> bool:
        """Check if word is a stop word using spaCy or fallback list"""
        if self.nlp:
            doc = self.nlp(word)
            return doc[0].is_stop if doc else False
        return word.lower() in self.fallback_stop_words

    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases using spaCy noun chunks and patterns"""
        phrases = []

        # Look for quoted phrases
        quoted = re.findall(r'"([^"]*)"', text)
        phrases.extend(quoted)

        if self.nlp:
            doc = self.nlp(text)

            # Extract noun chunks (natural phrases)
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) >= 2:
                    phrases.append(chunk.text.lower())

            # Extract compound nouns and technical terms
            for token in doc:
                if (token.pos_ == 'NOUN' and token.dep_ == 'compound' and
                    token.head.pos_ == 'NOUN'):
                    compound_phrase = f"{token.text} {token.head.text}".lower()
                    phrases.append(compound_phrase)

            # Extract multi-word technical terms using dependency patterns
            # Look for ADJ + compound NOUN + NOUN patterns (e.g., "whole life policy", "universal life policy")
            for token in doc:
                if token.pos_ == 'ADJ' and token.head.pos_ == 'NOUN':
                    head = token.head  # This is the final noun (e.g., "policy")

                    # Look for compound noun modifying this head noun
                    compounds = [child for child in head.children if child.dep_ == 'compound' and child.pos_ == 'NOUN']

                    if compounds:
                        # Build ADJ + compound + NOUN phrase (e.g., "whole life policy")
                        for compound in compounds:
                            three_word_phrase = f"{token.text} {compound.text} {head.text}".lower()
                            phrases.append(three_word_phrase)

                    # Also capture ADJ + NOUN pattern (fallback)
                    two_word_phrase = f"{token.text} {head.text}".lower()
                    phrases.append(two_word_phrase)

        else:
            # Fallback to pattern-based extraction
            words = re.findall(r'\b\w+\b', text.lower())
            # Extract 2-word phrases
            for i in range(len(words) - 1):
                if (not self._is_stop_word(words[i]) and not self._is_stop_word(words[i+1])
                    and len(words[i]) > 2 and len(words[i+1]) > 2):
                    phrase = f"{words[i]} {words[i+1]}"
                    phrases.append(phrase)

            # Extract 3-word phrases for technical terms
            for i in range(len(words) - 2):
                if (not self._is_stop_word(words[i]) and not self._is_stop_word(words[i+2])
                    and len(words[i]) > 2 and len(words[i+2]) > 2):
                    phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                    phrases.append(phrase)

        return list(set(phrases))

    def extract_entities(self, text: str) -> List[str]:
        """Extract entities using spaCy NER or pattern matching"""
        entities = []

        if self.nlp:
            doc = self.nlp(text)

            # Extract named entities
            for ent in doc.ents:
                entities.append(ent.text)

            # Extract proper nouns not caught by NER
            for token in doc:
                if token.pos_ == 'PROPN' and token.text not in entities:
                    entities.append(token.text)

        else:
            # Fallback to pattern-based extraction
            # Capitalized words (potential proper nouns)
            capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
            entities.extend(capitalized)

            # Acronyms
            acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
            entities.extend(acronyms)

        # Technical patterns (numbers with units, percentages, etc.)
        technical = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
        entities.extend(technical)

        return list(set(entities))

    def process_query(self, query: str) -> ProcessedQuery:
        """Process query using lightweight NLP techniques"""

        # Clean and tokenize
        tokens = re.findall(r'\b\w+\b', query.lower())

        # Filter meaningful tokens using spaCy or fallback
        meaningful_tokens = []
        semantic_keywords = []

        if self.nlp:
            doc = self.nlp(query)
            for token in doc:
                if (not token.is_stop and not token.is_punct and
                    len(token.text) > 2 and token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']):
                    meaningful_tokens.append(token.text.lower())
                    semantic_keywords.append(token.lemma_.lower())
        else:
            # Fallback processing
            for token in tokens:
                if not self._is_stop_word(token) and len(token) > 2:
                    lemma = self._get_lemma(token)
                    meaningful_tokens.append(token)
                    semantic_keywords.append(lemma)

        # Extract key phrases
        key_phrases = self.extract_key_phrases(query)

        # Extract entities
        key_entities = self.extract_entities(query)

        # Detect intent
        intent = self._detect_intent(query)

        return ProcessedQuery(
            original_query=query,
            key_entities=key_entities,
            key_phrases=key_phrases,
            semantic_keywords=list(set(semantic_keywords)),
            query_intent=intent,
            processed_tokens=meaningful_tokens
        )

    def _clean_phrase_for_search(self, phrase: str) -> str:
        """Clean extracted phrases for better search matching - no hardcoded logic"""
        if not phrase:
            return ""

        # Remove common articles and determiners that don't add search value
        # These are identified dynamically by spaCy or through basic linguistic patterns
        words = phrase.lower().strip().split()

        # Filter out single-character words and common stop words using our existing logic
        cleaned_words = []
        for word in words:
            if len(word) > 1 and not self._is_stop_word(word):
                cleaned_words.append(word)

        return " ".join(cleaned_words)

    def _detect_intent(self, text: str) -> str:
        """Detect query intent using semantic analysis"""
        if self.nlp:
            doc = self.nlp(text.lower())

            # Look for intent-indicating verbs and question words
            calculation_indicators = ['calculate', 'compute', 'determine', 'find']
            definition_indicators = ['define', 'explain', 'describe', 'what']
            comparison_indicators = ['compare', 'difference', 'versus', 'differ']
            procedure_indicators = ['steps', 'process', 'procedure', 'how']

            lemmas = [token.lemma_ for token in doc if not token.is_stop]

            if any(lemma in calculation_indicators for lemma in lemmas):
                return "calculation"
            elif any(lemma in definition_indicators for lemma in lemmas):
                return "definition"
            elif any(lemma in comparison_indicators for lemma in lemmas):
                return "comparison"
            elif any(lemma in procedure_indicators for lemma in lemmas):
                return "procedure"
        else:
            # Fallback to simple keyword matching
            text_lower = text.lower()
            if any(word in text_lower for word in ['how', 'calculate', 'compute', 'determine']):
                return "calculation"
            elif any(word in text_lower for word in ['what', 'define', 'explain']):
                return "definition"
            elif any(word in text_lower for word in ['difference', 'compare', 'versus', 'vs']):
                return "comparison"
            elif any(word in text_lower for word in ['steps', 'process', 'procedure']):
                return "procedure"

        return "general"

    def generate_search_query(self, processed_query: ProcessedQuery,
                            mode: str = "balanced") -> str:
        """Generate optimized search query for different modes"""

        if mode == "precise":
            # Use original query for precise matching
            return processed_query.original_query

        elif mode == "expanded":
            # Use semantic keywords and key phrases
            terms = []
            terms.append(processed_query.original_query)
            terms.extend(processed_query.key_phrases[:3])  # Top 3 phrases
            terms.extend(processed_query.semantic_keywords[:5])  # Top 5 keywords
            return " ".join(terms)

        elif mode == "semantic":
            # Use processed tokens and semantic keywords
            all_terms = processed_query.processed_tokens + processed_query.semantic_keywords
            unique_terms = list(dict.fromkeys(all_terms))  # Preserve order, remove duplicates
            return " ".join(unique_terms[:10])  # Limit to 10 terms

        elif mode == "focused":
            # Simplified approach: use main phrases only, avoid overly restrictive AND logic
            terms = []

            # Use the most important phrase only for initial retrieval
            if processed_query.key_phrases:
                # Find the best technical term phrase (prefer core terms without articles)
                best_phrase = None
                best_score = 0

                for phrase in processed_query.key_phrases:
                    words = phrase.split()
                    if len(words) >= 2:
                        # Score based on relevance: prefer phrases without articles
                        score = len(words)
                        if not phrase.startswith(('a ', 'an ', 'the ')):
                            score += 2  # Boost phrases without articles
                        if score > best_score:
                            best_score = score
                            best_phrase = phrase

                if best_phrase and len(best_phrase.split()) >= 2:
                    terms.append(f'"{best_phrase}"')
                elif best_phrase:
                    terms.append(best_phrase)

            # If no good phrases, use semantic keywords
            elif processed_query.semantic_keywords:
                # Use top 2-3 keywords only
                for keyword in processed_query.semantic_keywords[:3]:
                    if len(keyword) > 2:
                        terms.append(keyword)

            # Final fallback
            if not terms:
                terms = [processed_query.original_query]

            return " ".join(terms)

        else:  # balanced mode
            # Combine original query with top semantic elements
            terms = [processed_query.original_query]
            # Add top entities and phrases
            terms.extend(processed_query.key_entities[:2])
            terms.extend(processed_query.key_phrases[:2])
            return " ".join(terms)

    def validate_content_match(self, processed_query: ProcessedQuery, content: str) -> Dict[str, float]:
        """Post-retrieval validation: check if content actually contains requested phrases"""

        content_lower = content.lower()
        validation_score = 0.0
        phrase_matches = {}

        # Check for exact phrase matches (highest priority)
        for phrase in processed_query.key_phrases:
            if phrase.lower() in content_lower:
                phrase_matches[phrase] = 1.0
                validation_score += 3.0  # High weight for exact phrase match

        # Check for entity matches
        for entity in processed_query.key_entities:
            if entity.lower() in content_lower:
                phrase_matches[entity] = 1.0
                validation_score += 2.0  # Medium weight for entity match

        # Check for semantic keyword matches
        keyword_matches = 0
        for keyword in processed_query.semantic_keywords:
            if keyword.lower() in content_lower:
                keyword_matches += 1
                validation_score += 1.0  # Lower weight for individual keywords

        # Normalize score by content length to avoid bias towards longer documents
        normalized_score = validation_score / max(1.0, len(content) / 1000)

        return {
            "validation_score": normalized_score,
            "phrase_matches": phrase_matches,
            "keyword_match_count": keyword_matches,
            "total_matches": len(phrase_matches) + keyword_matches
        }

    def get_query_variants(self, query: str) -> Dict[str, str]:
        """Get different query variants for testing multiple search strategies"""
        processed = self.process_query(query)

        return {
            "original": query,
            "precise": self.generate_search_query(processed, "precise"),
            "balanced": self.generate_search_query(processed, "balanced"),
            "semantic": self.generate_search_query(processed, "semantic"),
            "expanded": self.generate_search_query(processed, "expanded")
        }