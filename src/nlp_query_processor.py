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
                if len(chunk.text.split()) >= 2:  # Removed restrictive length requirement
                    phrases.append(chunk.text.lower())

            # Extract compound nouns and technical terms
            for token in doc:
                if (token.pos_ == 'NOUN' and token.dep_ == 'compound' and
                    token.head.pos_ == 'NOUN'):
                    compound_phrase = f"{token.text} {token.head.text}".lower()
                    phrases.append(compound_phrase)

        else:
            # Fallback to pattern-based extraction
            words = re.findall(r'\b\w+\b', text.lower())
            for i in range(len(words) - 1):
                if (not self._is_stop_word(words[i]) and not self._is_stop_word(words[i+1])
                    and len(words[i]) > 2 and len(words[i+1]) > 2):
                    phrase = f"{words[i]} {words[i+1]}"
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
            # Use only the most important key phrases and keywords for precision
            terms = []

            # CRITICAL: Always preserve the original query to ensure we don't lose key differentiating terms
            terms.append(processed_query.original_query)

            # Add additional key phrases that might provide more specificity
            if processed_query.key_phrases:
                # Add up to 2 key phrases for additional context
                terms.extend(processed_query.key_phrases[:2])

            # Add entities which often contain important differentiating terms
            if processed_query.key_entities:
                terms.extend(processed_query.key_entities[:2])

            return " ".join(terms)

        else:  # balanced mode
            # Combine original query with top semantic elements
            terms = [processed_query.original_query]
            # Add top entities and phrases
            terms.extend(processed_query.key_entities[:2])
            terms.extend(processed_query.key_phrases[:2])
            return " ".join(terms)

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