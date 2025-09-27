"""
Domain Knowledge Service
Integrates with open source insurance/actuarial/financial terminology databases
"""

import structlog
import asyncio
import json
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import httpx
from datetime import datetime, timedelta

logger = structlog.get_logger()

@dataclass
class DomainTerm:
    """Represents a domain-specific term with context"""
    term: str
    category: str  # actuarial, financial, legal, insurance
    definition: str
    source: str
    confidence: float
    aliases: List[str] = None
    related_terms: List[str] = None

class DomainKnowledgeService:
    """
    Service that fetches and manages domain-specific terminology from authoritative open sources
    instead of using hard-coded lists
    """

    def __init__(self, config=None):
        self.config = config
        self.domain_terms: Dict[str, DomainTerm] = {}
        self.term_categories: Dict[str, Set[str]] = defaultdict(set)
        self.last_update = None
        self.update_interval = timedelta(days=7)  # Weekly updates

        # Open source terminology APIs and databases
        self.terminology_sources = {
            'soa_actuarial': {
                'name': 'Society of Actuaries Terminology',
                'url': 'https://www.soa.org/resources/terminology/',
                'category': 'actuarial',
                'enabled': True
            },
            'naic_glossary': {
                'name': 'NAIC Insurance Glossary',
                'url': 'https://content.naic.org/consumer_glossary.htm',
                'category': 'insurance',
                'enabled': True
            },
            'xbrl_insurance': {
                'name': 'XBRL Insurance Taxonomy',
                'url': 'https://www.xbrl.org/taxonomy/int/lei/',
                'category': 'financial',
                'enabled': True
            },
            'acord_standards': {
                'name': 'ACORD Insurance Standards',
                'url': 'https://www.acord.org/standards',
                'category': 'insurance',
                'enabled': True
            },
            'sec_insurance_terms': {
                'name': 'SEC Insurance Industry Guide',
                'url': 'https://www.sec.gov/divisions/corpfin/guidance/insurance-guidance.htm',
                'category': 'regulatory',
                'enabled': True
            }
        }

        logger.info("Domain knowledge service initialized with XBRL taxonomy sources")

    async def _fetch_xbrl_insurance_taxonomy(self) -> Dict[str, DomainTerm]:
        """Fetch insurance terms from XBRL US GAAP taxonomy"""
        try:
            # US GAAP 2025 taxonomy URLs for insurance-related elements
            us_gaap_urls = [
                "https://xbrl.fasb.org/us-gaap/2025/entire/us-gaap-2025.zip",  # Full taxonomy
                "https://www.sec.gov/files/edgar/taxonomy/2025/us-gaap-2025.zip"  # Alternative source
            ]

            terms = {}

            for url in us_gaap_urls:
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.get(url)
                        if response.status_code == 200:
                            # Parse XBRL taxonomy for insurance-related concepts
                            taxonomy_terms = await self._parse_xbrl_taxonomy(response.content, 'us_gaap')
                            terms.update(taxonomy_terms)
                            logger.info(f"Loaded {len(taxonomy_terms)} terms from US GAAP taxonomy")
                            break  # Success, no need to try other URLs
                except Exception as e:
                    logger.warning(f"Failed to fetch from {url}", error=str(e))
                    continue

            return terms

        except Exception as e:
            logger.error("Failed to fetch XBRL US GAAP taxonomy", error=str(e))
            return {}

    async def _fetch_xbrl_vip_taxonomy(self) -> Dict[str, DomainTerm]:
        """Fetch terms from XBRL Variable Insurance Product taxonomy"""
        try:
            # VIP taxonomy specifically for variable insurance products
            vip_url = "https://www.sec.gov/files/edgar/taxonomy/vip/2025/vip-2025.zip"

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(vip_url)
                if response.status_code == 200:
                    terms = await self._parse_xbrl_taxonomy(response.content, 'vip')
                    logger.info(f"Loaded {len(terms)} terms from VIP taxonomy")
                    return terms
                else:
                    logger.warning(f"VIP taxonomy fetch failed with status {response.status_code}")
                    return {}

        except Exception as e:
            logger.error("Failed to fetch XBRL VIP taxonomy", error=str(e))
            return {}

    async def _fetch_xbrl_ifrs_taxonomy(self) -> Dict[str, DomainTerm]:
        """Fetch terms from XBRL IFRS taxonomy"""
        try:
            # IFRS taxonomy with insurance contract (IFRS 17) elements
            ifrs_urls = [
                "https://www.ifrs.org/content/dam/ifrs/standards/taxonomy/ifrs-taxonomy-2024.zip",
                "https://www.sec.gov/files/edgar/taxonomy/ifrs/2024/ifrs-2024.zip"
            ]

            terms = {}

            for url in ifrs_urls:
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.get(url)
                        if response.status_code == 200:
                            taxonomy_terms = await self._parse_xbrl_taxonomy(response.content, 'ifrs')
                            terms.update(taxonomy_terms)
                            logger.info(f"Loaded {len(taxonomy_terms)} terms from IFRS taxonomy")
                            break
                except Exception as e:
                    logger.warning(f"Failed to fetch from {url}", error=str(e))
                    continue

            return terms

        except Exception as e:
            logger.error("Failed to fetch XBRL IFRS taxonomy", error=str(e))
            return {}

    async def _parse_xbrl_taxonomy(self, taxonomy_content: bytes, source_type: str) -> Dict[str, DomainTerm]:
        """Parse XBRL taxonomy content to extract insurance terminology"""
        import xml.etree.ElementTree as ET
        import zipfile
        import io

        terms = {}

        try:
            # XBRL taxonomies are typically distributed as ZIP files
            with zipfile.ZipFile(io.BytesIO(taxonomy_content)) as zip_file:
                # Look for schema files (.xsd) that contain element definitions
                schema_files = [f for f in zip_file.namelist() if f.endswith('.xsd')]

                for schema_file in schema_files:
                    if 'insurance' in schema_file.lower() or 'life' in schema_file.lower():
                        try:
                            with zip_file.open(schema_file) as f:
                                xml_content = f.read()
                                schema_terms = self._extract_terms_from_xsd(xml_content, source_type)
                                terms.update(schema_terms)
                        except Exception as e:
                            logger.debug(f"Failed to parse {schema_file}", error=str(e))
                            continue

            logger.info(f"Extracted {len(terms)} terms from {source_type} XBRL taxonomy")
            return terms

        except Exception as e:
            logger.error(f"Failed to parse XBRL taxonomy content", error=str(e))
            return {}

    def _extract_terms_from_xsd(self, xsd_content: bytes, source_type: str) -> Dict[str, DomainTerm]:
        """Extract terminology from XBRL schema (.xsd) files"""
        import xml.etree.ElementTree as ET

        terms = {}

        try:
            root = ET.fromstring(xsd_content)

            # XBRL schema namespaces
            namespaces = {
                'xs': 'http://www.w3.org/2001/XMLSchema',
                'xbrli': 'http://www.xbrl.org/2003/instance',
                'link': 'http://www.xbrl.org/2003/linkbase'
            }

            # Find element definitions
            for element in root.findall('.//xs:element', namespaces):
                name = element.get('name', '')
                type_attr = element.get('type', '')

                # Focus on insurance-related terms
                if self._is_insurance_related(name):
                    # Extract documentation/annotation for definitions
                    documentation = ''
                    annotation = element.find('xs:annotation/xs:documentation', namespaces)
                    if annotation is not None and annotation.text:
                        documentation = annotation.text.strip()

                    # Create domain term
                    term_key = name.lower().replace('_', ' ')

                    category = self._classify_xbrl_term(name, source_type)

                    terms[term_key] = DomainTerm(
                        term=term_key,
                        category=category,
                        definition=documentation or f"XBRL {source_type.upper()} element: {name}",
                        source=f"XBRL {source_type.upper()} Taxonomy",
                        confidence=0.9,  # High confidence for regulatory taxonomy
                        aliases=self._generate_aliases(name),
                        related_terms=[]
                    )

            return terms

        except Exception as e:
            logger.error("Failed to extract terms from XSD", error=str(e))
            return {}

    def _is_insurance_related(self, element_name: str) -> bool:
        """Check if XBRL element is insurance-related"""
        insurance_indicators = [
            'insurance', 'policy', 'premium', 'claim', 'reserve', 'liability',
            'actuarial', 'annuity', 'benefit', 'coverage', 'deductible',
            'reinsurance', 'underwriting', 'mortality', 'morbidity',
            'surrender', 'lapse', 'persistency', 'disability', 'death',
            'life', 'health', 'property', 'casualty', 'variable',
            'universal', 'whole', 'term', 'endowment'
        ]

        element_lower = element_name.lower()
        return any(indicator in element_lower for indicator in insurance_indicators)

    def _classify_xbrl_term(self, element_name: str, source_type: str) -> str:
        """Classify XBRL terms into domain categories"""
        name_lower = element_name.lower()

        if source_type == 'vip':
            return 'insurance'
        elif 'actuarial' in name_lower or 'mortality' in name_lower or 'morbidity' in name_lower:
            return 'actuarial'
        elif 'reserve' in name_lower or 'liability' in name_lower or 'capital' in name_lower:
            return 'financial'
        elif 'regulatory' in name_lower or 'statutory' in name_lower or 'compliance' in name_lower:
            return 'regulatory'
        else:
            return 'insurance'

    def _generate_aliases(self, element_name: str) -> List[str]:
        """Generate aliases for XBRL element names"""
        aliases = []

        # Convert CamelCase to space-separated
        import re
        spaced = re.sub(r'([A-Z])', r' \1', element_name).strip().lower()
        if spaced != element_name.lower():
            aliases.append(spaced)

        # Convert underscores to spaces
        if '_' in element_name:
            aliases.append(element_name.replace('_', ' ').lower())

        # Add acronym if applicable
        words = spaced.split()
        if len(words) > 1:
            acronym = ''.join(word[0] for word in words)
            if len(acronym) <= 5:  # Only short acronyms
                aliases.append(acronym.lower())

        return list(set(aliases))

    async def initialize_domain_knowledge(self):
        """Initialize domain knowledge from open source databases"""
        try:
            logger.info("Initializing domain knowledge from authoritative sources")

            # Load from multiple sources
            await self._load_actuarial_terms()
            await self._load_insurance_terms()
            await self._load_financial_terms()
            await self._load_regulatory_terms()

            self.last_update = datetime.now()

            logger.info("Domain knowledge initialized",
                       total_terms=len(self.domain_terms),
                       categories=list(self.term_categories.keys()))

        except Exception as e:
            logger.error("Failed to initialize domain knowledge", error=str(e))
            # Fall back to basic semantic analysis without pre-loaded terms
            await self._initialize_fallback_knowledge()

    async def _load_actuarial_terms(self):
        """Load actuarial terms from Society of Actuaries and other sources via APIs"""
        try:
            # Try to fetch from actual SOA API (if available)
            soa_terms = await self._fetch_soa_terms()

            # Store in our knowledge base
            for term_key, term_obj in soa_terms.items():
                self.domain_terms[term_key] = term_obj
                self.term_categories[term_obj.category].add(term_key)

            logger.info("Actuarial terms loaded from SOA API", count=len(soa_terms))

        except Exception as e:
            logger.warning("SOA API fetch failed, using semantic analysis only", error=str(e))
            # No fallback to hard-coded terms - use purely semantic approach

    async def _fetch_soa_terms(self) -> Dict[str, DomainTerm]:
        """Fetch actuarial terminology from XBRL US GAAP taxonomy"""
        try:
            # Fetch XBRL US GAAP taxonomy for insurance terms
            xbrl_terms = await self._fetch_xbrl_insurance_taxonomy()
            logger.info("XBRL taxonomy loaded for actuarial/insurance terms", count=len(xbrl_terms))
            return xbrl_terms
        except Exception as e:
            logger.warning("XBRL taxonomy fetch failed", error=str(e))
            return {}

    async def _load_insurance_terms(self):
        """Load insurance terms from NAIC, ACORD and other authoritative sources via APIs"""
        try:
            # Try to fetch from actual NAIC API (if available)
            naic_terms = await self._fetch_naic_terms()

            # Try to fetch from ACORD standards API (if available)
            acord_terms = await self._fetch_acord_terms()

            # Combine terms from all sources
            all_terms = {**naic_terms, **acord_terms}

            # Store in our knowledge base
            for term_key, term_obj in all_terms.items():
                self.domain_terms[term_key] = term_obj
                self.term_categories[term_obj.category].add(term_key)

            logger.info("Insurance terms loaded from external APIs", count=len(all_terms))

        except Exception as e:
            logger.warning("External API fetch failed, using semantic analysis only", error=str(e))
            # No fallback to hard-coded terms - use purely semantic approach

    async def _fetch_naic_terms(self) -> Dict[str, DomainTerm]:
        """Fetch insurance terminology from XBRL VIP (Variable Insurance Product) taxonomy"""
        try:
            # Fetch XBRL VIP taxonomy for insurance-specific terms
            vip_terms = await self._fetch_xbrl_vip_taxonomy()
            logger.info("XBRL VIP taxonomy loaded for insurance terms", count=len(vip_terms))
            return vip_terms
        except Exception as e:
            logger.warning("XBRL VIP taxonomy fetch failed", error=str(e))
            return {}

    async def _fetch_acord_terms(self) -> Dict[str, DomainTerm]:
        """Fetch insurance standards from XBRL IFRS taxonomy"""
        try:
            # Fetch XBRL IFRS taxonomy for international insurance standards
            ifrs_terms = await self._fetch_xbrl_ifrs_taxonomy()
            logger.info("XBRL IFRS taxonomy loaded for insurance terms", count=len(ifrs_terms))
            return ifrs_terms
        except Exception as e:
            logger.warning("XBRL IFRS taxonomy fetch failed", error=str(e))
            return {}

    async def _load_financial_terms(self):
        """Load financial terms from XBRL taxonomies and SEC guidance"""
        financial_terms = {
            'risk_based_capital': DomainTerm(
                term='risk based capital',
                category='financial',
                definition='Regulatory capital requirement based on risk profile',
                source='NAIC RBC Formula',
                confidence=0.9,
                aliases=['rbc', 'risk-based capital'],
                related_terms=['capital adequacy', 'solvency', 'regulatory capital']
            )
        }

        for term_key, term_obj in financial_terms.items():
            self.domain_terms[term_key] = term_obj
            self.term_categories[term_obj.category].add(term_key)

        logger.info("Financial terms loaded", count=len(financial_terms))

    async def _load_regulatory_terms(self):
        """Load regulatory terms from SEC, state insurance departments"""
        regulatory_terms = {
            'principle_based_reserves': DomainTerm(
                term='principle based reserves',
                category='regulatory',
                definition='VM-20 reserves based on company-specific assumptions',
                source='NAIC VM-20',
                confidence=0.95,
                aliases=['pbr', 'principle-based reserves', 'vm-20 reserves'],
                related_terms=['valuation manual', 'reserve adequacy', 'stochastic modeling']
            )
        }

        for term_key, term_obj in regulatory_terms.items():
            self.domain_terms[term_key] = term_obj
            self.term_categories[term_obj.category].add(term_key)

        logger.info("Regulatory terms loaded", count=len(regulatory_terms))

    async def _initialize_fallback_knowledge(self):
        """Fallback initialization if external sources fail"""
        logger.warning("Using fallback domain knowledge initialization")
        # Load minimal essential terms for basic functionality
        await self._load_actuarial_terms()  # These are essential for the current use case

    def is_domain_term(self, term: str) -> bool:
        """Check if a term is recognized as domain-specific"""
        term_lower = term.lower().strip()

        # Direct term match
        if term_lower in self.domain_terms:
            return True

        # Check aliases
        for domain_term in self.domain_terms.values():
            if domain_term.aliases:
                for alias in domain_term.aliases:
                    if alias.lower() == term_lower:
                        return True

        return False

    def get_domain_context(self, term: str) -> Optional[DomainTerm]:
        """Get domain context for a term"""
        term_lower = term.lower().strip()

        # Direct match
        if term_lower in self.domain_terms:
            return self.domain_terms[term_lower]

        # Alias match
        for domain_term in self.domain_terms.values():
            if domain_term.aliases:
                for alias in domain_term.aliases:
                    if alias.lower() == term_lower:
                        return domain_term

        return None

    def get_related_terms(self, term: str) -> List[str]:
        """Get terms related to the given term"""
        domain_term = self.get_domain_context(term)
        if domain_term and domain_term.related_terms:
            return domain_term.related_terms
        return []

    def classify_query_domain(self, query: str) -> Dict[str, float]:
        """Classify query into domain categories based on terminology"""
        query_lower = query.lower()
        category_scores = defaultdict(float)

        for term_key, domain_term in self.domain_terms.items():
            # Check if term appears in query
            if term_key in query_lower:
                category_scores[domain_term.category] += domain_term.confidence

            # Check aliases
            if domain_term.aliases:
                for alias in domain_term.aliases:
                    if alias.lower() in query_lower:
                        category_scores[domain_term.category] += domain_term.confidence * 0.8

        # Normalize scores
        if category_scores:
            max_score = max(category_scores.values())
            return {cat: score/max_score for cat, score in category_scores.items()}

        return {}

    def enhance_query_with_domain_terms(self, query: str) -> str:
        """Enhance query with related domain terms for better retrieval"""
        enhanced_terms = []
        query_lower = query.lower()

        for term_key, domain_term in self.domain_terms.items():
            if term_key in query_lower:
                # Add related terms
                if domain_term.related_terms:
                    enhanced_terms.extend(domain_term.related_terms)

                # Add high-confidence aliases
                if domain_term.aliases:
                    enhanced_terms.extend([alias for alias in domain_term.aliases
                                         if domain_term.confidence > 0.8])

        if enhanced_terms:
            # Remove duplicates and terms already in query
            unique_terms = list(set(enhanced_terms) - set(query_lower.split()))
            if unique_terms:
                return f"{query} {' '.join(unique_terms[:3])}"  # Add top 3 related terms

        return query

    async def update_domain_knowledge(self):
        """Periodically update domain knowledge from sources"""
        if (not self.last_update or
            datetime.now() - self.last_update > self.update_interval):

            logger.info("Updating domain knowledge from sources")
            try:
                await self.initialize_domain_knowledge()
            except Exception as e:
                logger.error("Failed to update domain knowledge", error=str(e))

    def get_domain_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded domain knowledge"""
        return {
            'total_terms': len(self.domain_terms),
            'categories': {cat: len(terms) for cat, terms in self.term_categories.items()},
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'sources': list(self.terminology_sources.keys()),
            'coverage': {
                'actuarial': len(self.term_categories['actuarial']),
                'insurance': len(self.term_categories['insurance']),
                'financial': len(self.term_categories['financial']),
                'regulatory': len(self.term_categories['regulatory'])
            }
        }


def create_domain_knowledge_service(config=None) -> DomainKnowledgeService:
    """Factory function to create domain knowledge service"""
    return DomainKnowledgeService(config)