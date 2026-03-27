"""
Unified entity extraction for chunk-level and query-level disambiguation.

Three extraction layers (cheapest first):
1. Regex: actuarial domain patterns + acronyms (zero cost)
2. spaCy NER: people, orgs, dates, locations (en_core_web_sm, ~50ms/chunk)
3. c-TF-IDF from Elasticsearch: corpus-discriminative terms (ingestion only)

At query time, only layers 1+2 run (fast path).
At ingestion time, all three layers run (batch tolerant).

Graceful degradation: if spaCy or ES is unavailable, remaining layers still work.
"""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import structlog

logger = structlog.get_logger()


@dataclass
class ExtractedEntities:
    """Entities extracted from a text chunk or query."""

    # NER entities (spaCy)
    persons: List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)

    # Corpus-discriminative terms (c-TF-IDF from ES)
    discriminative_terms: List[str] = field(default_factory=list)

    # Domain-specific entities (regex)
    domain_entities: List[str] = field(default_factory=list)

    # Flattened, normalized list for Qdrant payload filtering
    all_entities: List[str] = field(default_factory=list)

    # Metadata
    extraction_source: str = ""  # e.g. "spacy+regex" or "spacy+es+regex"
    entity_count: int = 0

    def to_qdrant_fields(self, *, hash_pii: bool = False) -> Dict[str, Any]:
        """Return dict of fields to merge into Qdrant payload.

        Args:
            hash_pii: If True, hash person names instead of storing plaintext.
                      For regulated industries (insurance, healthcare) where
                      PII in vector store payloads is a compliance concern.
        """
        import hashlib

        persons = [e.lower() for e in self.persons]
        if hash_pii and persons:
            persons = [
                f"person_{hashlib.sha256(p.encode()).hexdigest()[:12]}"
                for p in persons
            ]

        return {
            "entities": self.all_entities,
            "entity_orgs": [e.lower() for e in self.organizations],
            "entity_persons": persons,
        }


class EntityExtractor:
    """
    Unified entity extractor combining spaCy NER, ES c-TF-IDF, and regex.

    Lazy-loads spaCy model on first use. If spaCy is not installed or the
    model is missing, falls back to regex-only extraction.

    Thread-safe: spaCy model loading protected by a lock.
    """

    # Regex patterns for acronyms and domain terms not covered by
    # ActuarialEntityExtractor (which handles policy types, regulations, etc.)
    _ACRONYM_PATTERN = re.compile(r"\b[A-Z][A-Z&]{1,10}\b")

    # Common 2-3 letter English words that appear in ALL-CAPS headings.
    # These are false positives for acronym extraction.
    _ACRONYM_STOPWORDS: Set[str] = {
        "IT", "OR", "AN", "IS", "BY", "AS", "ON", "AT", "US", "IF", "NO",
        "IN", "TO", "OF", "DO", "SO", "UP", "AM", "BE", "HE", "ME", "WE",
        "MY", "GO", "HI", "OK", "HA", "OH", "OX",
        "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "ANY",
        "CAN", "HAD", "HER", "WAS", "ONE", "OUR", "OUT", "HAS", "HIS",
        "HOW", "ITS", "MAY", "NEW", "NOW", "OLD", "SEE", "WAY", "WHO",
        "DID", "GET", "LET", "SAY", "SHE", "TOO", "USE",
    }

    # Insurance/actuarial regulatory standards and frameworks.
    # These are NOT reliably detected by spaCy en_core_web_sm.
    _INSURANCE_PATTERNS: List[re.Pattern] = [
        # Accounting standards: IFRS 17, IFRS 9, ASU 2018-12, FAS 133
        re.compile(r"\bIFRS[-‐]?\s*\d+\b", re.IGNORECASE),
        re.compile(r"\bASU\s*\d{4}[-‐]\d+\b", re.IGNORECASE),
        re.compile(r"\bFAS\s*\d+\b", re.IGNORECASE),
        re.compile(r"\bASC\s*\d+(?:[-‐]\d+)*\b", re.IGNORECASE),
        # US Statutory: VM-20, VM-21, VM-30, AG 43, RBC C-3
        re.compile(r"\bVM[-‐]\d+\b", re.IGNORECASE),
        re.compile(r"\bAG[-‐]?\s*\d+\b", re.IGNORECASE),
        re.compile(r"\bRBC\s*C[-‐]\d+\b", re.IGNORECASE),
        # Well-known insurance acronyms that spaCy won't detect as entities
        re.compile(r"\b(?:LDTI|ORSA|CFT|NAIC|SOA|CAS|AAA|GAAP|PBR)\b"),
        re.compile(r"\b(?:DAC|VOBA|SOP|AOCI|OCI|OTTI|FIA|UL|VUL|IUL)\b"),
        re.compile(r"\b(?:CSM|BEL|RA|PAA|GMM|VFA)\b"),  # IFRS 17 specific
    ]

    # Multi-word domain concepts that spaCy NER won't detect.
    _DOMAIN_CONCEPT_PATTERNS: List[re.Pattern] = [
        # Capital & solvency
        re.compile(r"\bcapital\s+ratio(?:s)?\b", re.IGNORECASE),
        re.compile(r"\bsolvency\s+(?:ratio|margin|capital)\b", re.IGNORECASE),
        re.compile(r"\brisk[- ]based\s+capital\b", re.IGNORECASE),
        re.compile(r"\bminimum\s+capital\s+(?:test|requirement)\b", re.IGNORECASE),
        re.compile(r"\bLICAT\s*(?:ratio)?\b", re.IGNORECASE),
        # Loss & combined ratios
        re.compile(r"\bloss\s+ratio(?:s)?\b", re.IGNORECASE),
        re.compile(r"\bcombined\s+ratio(?:s)?\b", re.IGNORECASE),
        re.compile(r"\bexpense\s+ratio(?:s)?\b", re.IGNORECASE),
        # Reserves
        re.compile(r"\breserve\s+(?:adequacy|deficiency|margin)\b", re.IGNORECASE),
        re.compile(r"\bclaims?\s+reserve(?:s)?\b", re.IGNORECASE),
        # IFRS 17 concepts
        re.compile(r"\bcontractual\s+service\s+margin\b", re.IGNORECASE),
        re.compile(r"\bbest\s+estimate\s+liabilit(?:y|ies)\b", re.IGNORECASE),
        re.compile(r"\brisk\s+adjustment\b", re.IGNORECASE),
        re.compile(r"\binsurance\s+(?:contract|revenue)\b", re.IGNORECASE),
        re.compile(r"\bbuilding\s+block\s+approach\b", re.IGNORECASE),
        # General insurance/actuarial
        re.compile(r"\bpolicy(?:holder)?\s+(?:dividend|surplus|equity)\b", re.IGNORECASE),
        re.compile(r"\bunderwriting\s+(?:profit|income|result)\b", re.IGNORECASE),
        re.compile(r"\bpremium\s+(?:deficiency|sufficiency)\b", re.IGNORECASE),
    ]

    def __init__(
        self,
        es_engine: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._nlp = None  # Lazy-loaded spaCy model
        self._es = es_engine
        self._config = config or self._load_config()
        self._spacy_available: Optional[bool] = None  # None = not checked
        self._spacy_lock = threading.Lock()  # Protects lazy model loading

    @staticmethod
    def _load_config() -> Dict[str, Any]:
        try:
            from ..providers.config_loader import get_config
            return get_config("entity_extraction")
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        text: str,
        *,
        use_es_tfidf: bool = True,
    ) -> ExtractedEntities:
        """
        Full extraction: spaCy + ES c-TF-IDF + regex.

        Used during ingestion (batch mode, latency-tolerant).

        Args:
            text: Chunk or document text.
            use_es_tfidf: Query ES for corpus-discriminative terms.

        Returns:
            ExtractedEntities with all entity types populated.
        """
        # Guard against empty/null input.
        if not text or not text.strip():
            return ExtractedEntities(extraction_source="empty_input")

        result = ExtractedEntities()
        sources: List[str] = []

        # Layer 1: Regex (always available, zero cost)
        self._extract_regex(text, result)
        sources.append("regex")

        # Layer 2: spaCy NER
        if self._ensure_spacy():
            self._extract_spacy(text, result)
            sources.append("spacy")

        # Layer 3: c-TF-IDF from Elasticsearch
        if use_es_tfidf and self._es:
            self._extract_es_significant_terms(text, result)
            sources.append("es")

        # Build unified entity list
        result.all_entities = self._build_unified_list(result)
        result.entity_count = len(result.all_entities)
        result.extraction_source = "+".join(sources)

        logger.debug(
            "Entity extraction complete",
            source=result.extraction_source,
            count=result.entity_count,
            entities=result.all_entities[:10],
        )
        return result

    def extract_from_query(self, query: str) -> ExtractedEntities:
        """
        Fast-path extraction for query-time use.

        Only runs spaCy + regex (no ES call) to keep latency low.
        """
        return self.extract(query, use_es_tfidf=False)

    def extract_batch(
        self,
        texts: List[str],
        *,
        use_es_tfidf: bool = True,
    ) -> List[ExtractedEntities]:
        """
        Batch extraction for ingestion performance.

        Uses spaCy's nlp.pipe() for efficient batch NER processing
        instead of calling nlp() per text. ~3-5x faster for large batches.

        Args:
            texts: List of chunk texts.
            use_es_tfidf: Query ES for corpus-discriminative terms.

        Returns:
            List of ExtractedEntities, one per input text.
        """
        results: List[ExtractedEntities] = []

        # Pre-filter empty texts.
        valid_indices: List[int] = []
        valid_texts: List[str] = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_indices.append(i)
                valid_texts.append(text)
            else:
                results.append(ExtractedEntities(extraction_source="empty_input"))

        if not valid_texts:
            return results

        # Regex extraction (always runs, fast).
        batch_results: List[ExtractedEntities] = []
        for text in valid_texts:
            r = ExtractedEntities()
            self._extract_regex(text, r)
            batch_results.append(r)

        # spaCy batch NER via nlp.pipe().
        if self._ensure_spacy() and self._nlp:
            max_chars = self._config.get("spacy", {}).get("max_chars", 5000)
            allowed_types = set(
                self._config.get("spacy", {}).get(
                    "entity_types", ["PERSON", "ORG", "DATE", "GPE"]
                )
            )
            truncated = [t[:max_chars] for t in valid_texts]

            label_map_keys = {"PERSON", "ORG", "DATE", "GPE", "LOC"}

            for idx, doc in enumerate(self._nlp.pipe(truncated, batch_size=64)):
                r = batch_results[idx]
                seen: Set[str] = set()
                for ent in doc.ents:
                    if ent.label_ not in allowed_types:
                        continue
                    normalized = ent.text.strip()
                    if not normalized or normalized.lower() in seen:
                        continue
                    seen.add(normalized.lower())
                    if ent.label_ == "PERSON":
                        r.persons.append(normalized)
                    elif ent.label_ == "ORG":
                        r.organizations.append(normalized)
                    elif ent.label_ == "DATE":
                        r.dates.append(normalized)
                    elif ent.label_ in ("GPE", "LOC"):
                        r.locations.append(normalized)

        # ES c-TF-IDF (per-text, no batch API).
        if use_es_tfidf and self._es:
            for idx, text in enumerate(valid_texts):
                self._extract_es_significant_terms(text, batch_results[idx])

        # Build unified lists.
        for r in batch_results:
            r.all_entities = self._build_unified_list(r)
            r.entity_count = len(r.all_entities)
            sources = ["regex"]
            if self._spacy_available:
                sources.append("spacy")
            if use_es_tfidf and self._es:
                sources.append("es")
            r.extraction_source = "+".join(sources)

        # Reassemble results in original order.
        final: List[ExtractedEntities] = [ExtractedEntities()] * len(texts)
        batch_idx = 0
        for i in range(len(texts)):
            if i in valid_indices:
                final[i] = batch_results[batch_idx]
                batch_idx += 1
            else:
                final[i] = ExtractedEntities(extraction_source="empty_input")

        logger.info(
            "Batch entity extraction complete",
            total=len(texts),
            with_entities=sum(1 for r in final if r.entity_count > 0),
        )
        return final

    # ------------------------------------------------------------------
    # Layer 1: Regex
    # ------------------------------------------------------------------

    def _extract_regex(self, text: str, result: ExtractedEntities) -> None:
        """Extract domain entities using regex patterns."""
        # Reuse ActuarialEntityExtractor patterns
        try:
            from ..conversation_memory import ActuarialEntityExtractor
            raw_entities = ActuarialEntityExtractor.extract_entities(text)
            result.domain_entities.extend(raw_entities)
        except ImportError:
            pass

        # Extract acronyms (FP&A, LDTI, VM-20, etc.)
        extract_acronyms = self._config.get("regex", {}).get("extract_acronyms", True)
        min_len = self._config.get("regex", {}).get("min_acronym_length", 2)
        if extract_acronyms:
            acronyms = self._ACRONYM_PATTERN.findall(text)
            for acr in acronyms:
                if len(acr) >= min_len and acr not in self._ACRONYM_STOPWORDS:
                    result.organizations.append(acr)

        # Extract insurance/actuarial regulatory references (IFRS 17, VM-20, etc.)
        for pattern in self._INSURANCE_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                normalized = match.strip()
                if normalized and normalized not in result.domain_entities:
                    result.domain_entities.append(normalized)

        # Extract multi-word domain concepts (capital ratios, risk adjustment, etc.)
        for pattern in self._DOMAIN_CONCEPT_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                normalized = match.strip().lower()
                if normalized and normalized not in result.domain_entities:
                    result.domain_entities.append(normalized)

    # ------------------------------------------------------------------
    # Layer 2: spaCy NER
    # ------------------------------------------------------------------

    def _ensure_spacy(self) -> bool:
        """Lazy-load spaCy model. Returns True if available.

        Thread-safe: uses a lock to prevent double-loading.
        """
        if self._spacy_available is not None:
            return self._spacy_available

        with self._spacy_lock:
            # Double-check after acquiring lock (another thread may have loaded).
            if self._spacy_available is not None:
                return self._spacy_available

            model_name = self._config.get("spacy", {}).get("model", "en_core_web_sm")
            try:
                import spacy
                self._nlp = spacy.load(model_name)
                self._spacy_available = True
                logger.info("spaCy model loaded for entity extraction", model=model_name)
            except (ImportError, OSError) as e:
                self._spacy_available = False
                logger.warning(
                    "spaCy not available, falling back to regex-only",
                    model=model_name,
                    error=str(e),
                )
        return self._spacy_available

    def _extract_spacy(self, text: str, result: ExtractedEntities) -> None:
        """Extract NER entities using spaCy."""
        if not self._nlp:
            return

        max_chars = self._config.get("spacy", {}).get("max_chars", 5000)
        allowed_types = set(
            self._config.get("spacy", {}).get(
                "entity_types", ["PERSON", "ORG", "DATE", "GPE"]
            )
        )

        doc = self._nlp(text[:max_chars])

        label_map = {
            "PERSON": result.persons,
            "ORG": result.organizations,
            "DATE": result.dates,
            "GPE": result.locations,
            "LOC": result.locations,
        }

        seen: Set[str] = set()
        for ent in doc.ents:
            if ent.label_ not in allowed_types:
                continue
            normalized = ent.text.strip()
            if normalized and normalized.lower() not in seen:
                seen.add(normalized.lower())
                target = label_map.get(ent.label_)
                if target is not None:
                    target.append(normalized)

    # ------------------------------------------------------------------
    # Layer 3: Elasticsearch c-TF-IDF
    # ------------------------------------------------------------------

    def _extract_es_significant_terms(
        self, text: str, result: ExtractedEntities
    ) -> None:
        """
        Use ES significant_terms aggregation to find corpus-discriminative terms.

        These are terms that are statistically overrepresented in this text
        relative to the overall corpus. ES uses chi-square scoring internally.
        """
        if not self._es:
            return

        es_config = self._config.get("elasticsearch", {}).get("significant_terms", {})
        top_k = es_config.get("top_k", 10)

        try:
            if hasattr(self._es, "get_significant_terms"):
                terms = self._es.get_significant_terms(
                    text,
                    top_k=top_k,
                    min_doc_count=es_config.get("min_doc_count", 2),
                    max_doc_percent=es_config.get("max_doc_percent", 30.0),
                )
                for term_info in terms:
                    term = term_info.get("term", "")
                    if term and len(term) >= 3:
                        result.discriminative_terms.append(term)
        except Exception as e:
            logger.debug("ES significant_terms extraction failed (non-fatal)", error=str(e))

    # ------------------------------------------------------------------
    # Unified list building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_unified_list(result: ExtractedEntities) -> List[str]:
        """Merge all entity sources into a deduplicated, normalized list."""
        all_entities: List[str] = []
        seen: Set[str] = set()

        # Priority order: orgs first (most useful for filtering), then persons,
        # then discriminative terms, then domain entities
        for entity_list in [
            result.organizations,
            result.persons,
            result.discriminative_terms,
        ]:
            for entity in entity_list:
                normalized = entity.lower().strip()
                if normalized and normalized not in seen and len(normalized) > 1:
                    seen.add(normalized)
                    all_entities.append(normalized)

        # Domain entities have "category:value" format — extract the value
        for entity in result.domain_entities:
            if ":" in entity:
                value = entity.split(":", 1)[1].lower().strip()
            else:
                value = entity.lower().strip()
            if value and value not in seen and len(value) > 1:
                seen.add(value)
                all_entities.append(value)

        return all_entities
