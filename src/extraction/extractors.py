"""
Specific extractors using base template - no code duplication
Clean implementation following single responsibility principle
"""

import re
from typing import List, Dict, Any
import structlog

from .base_extractor import BaseExtractor
from .models import (
    ExtractedEntity,
    ExtractionMethod,
    MetadataContext
)

logger = structlog.get_logger()


class OrganizationalExtractor(BaseExtractor):
    """Extract organizational chart information"""

    async def extract_entities(
        self,
        content: str,
        context: MetadataContext,
        llm_client=None
    ) -> List[ExtractedEntity]:
        """Extract organizational entities using LLM with fallback"""

        # Try LLM extraction first
        if llm_client:
            try:
                return await self._extract_with_llm(content, llm_client)
            except Exception as e:
                logger.warning(f"LLM extraction failed, using fallback: {e}")

        # Fallback to pattern matching
        return self._extract_with_patterns(content)

    async def _extract_with_llm(self, content: str, llm_client) -> List[ExtractedEntity]:
        """Extract using LLM"""
        prompt = f"""Extract ONLY explicitly stated job titles and names from this organizational document.

Document text:
{content}

Rules:
1. ONLY extract what is explicitly written
2. Do NOT invent or assume titles
3. If a name appears without a title, mark title as null
4. Distinguish between job titles and authority levels

Return JSON:
{{
    "entities": [
        {{
            "name": "exact name as written",
            "title": "exact title or null",
            "department": "if clearly stated or null",
            "authority_level": "if mentioned separately or null",
            "confidence": 0.0-1.0,
            "context": "surrounding text",
            "source_section": "document section"
        }}
    ]
}}"""

        response = await self._query_llm_with_retry(llm_client, prompt)
        if not response:
            return []

        result_data = self._parse_llm_json_response(response)
        return self._convert_to_entities(result_data.get('entities', []))

    def _extract_with_patterns(self, content: str) -> List[ExtractedEntity]:
        """Pattern-based fallback extraction"""
        entities = []

        # Find name-title patterns
        name_title_patterns = [
            r'([A-Z][a-z]+ [A-Z][a-z]+),?\s+([A-Z][a-zA-Z\s]+?)(?=\n|\.|,)',
            r'([A-Z][a-zA-Z\s]{2,}?):\s*([A-Z][a-z]+ [A-Z][a-z]+)',
            r'([A-Z][a-z]+ [A-Z][a-z]+)\s*-\s*([A-Z][a-zA-Z\s]+)',
        ]

        for pattern in name_title_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                name, title = match[0].strip(), match[1].strip()
                if len(name.split()) >= 2 and len(title) > 2:
                    entities.append(ExtractedEntity(
                        name=name,
                        title=title,
                        confidence=0.6,
                        context=f"Pattern match: {name} - {title}",
                        source_section="pattern_extraction",
                        entity_type="person"
                    ))

        return entities

    def _convert_to_entities(self, entity_data: List[Dict]) -> List[ExtractedEntity]:
        """Convert LLM response to ExtractedEntity objects"""
        entities = []
        for data in entity_data:
            if data.get('name'):
                entities.append(ExtractedEntity(
                    name=data['name'],
                    title=data.get('title'),
                    department=data.get('department'),
                    authority_level=data.get('authority_level'),
                    confidence=data.get('confidence', 0.5),
                    context=data.get('context', ''),
                    source_section=data.get('source_section', ''),
                    entity_type="person"
                ))
        return entities

    def build_extraction_metadata(
        self,
        entities: List[ExtractedEntity],
        confidence: float,
        context: MetadataContext
    ) -> Dict[str, Any]:
        """Build organizational-specific metadata"""
        return {
            'extraction_type': 'organizational',
            'total_entities': len(entities),
            'entities_with_titles': len([e for e in entities if e.title]),
            'departments_found': list(set(e.department for e in entities if e.department)),
            'authority_levels': list(set(e.authority_level for e in entities if e.authority_level)),
            'avg_entity_confidence': sum(e.confidence for e in entities) / len(entities) if entities else 0.0
        }

    def get_document_type(self) -> str:
        return "organizational_chart"

    def get_extraction_method(self) -> ExtractionMethod:
        return ExtractionMethod.FULL_LLM


class FinancialExtractor(BaseExtractor):
    """Extract financial structure information"""

    async def extract_entities(
        self,
        content: str,
        context: MetadataContext,
        llm_client=None
    ) -> List[ExtractedEntity]:
        """Extract financial entities"""

        if llm_client:
            try:
                return await self._extract_with_llm(content, llm_client)
            except Exception as e:
                logger.warning(f"Financial LLM extraction failed: {e}")

        return self._extract_financial_patterns(content)

    async def _extract_with_llm(self, content: str, llm_client) -> List[ExtractedEntity]:
        """LLM-based financial extraction"""
        prompt = f"""Extract financial roles and titles from this document.

Document text:
{content}

Focus on financial positions: CFO, Treasurer, Controller, Financial Analyst, etc.

Return JSON:
{{
    "entities": [
        {{
            "name": "person name",
            "title": "financial title",
            "department": "department if stated",
            "confidence": 0.0-1.0,
            "context": "surrounding text"
        }}
    ]
}}"""

        response = await self._query_llm_with_retry(llm_client, prompt)
        if not response:
            return []

        result_data = self._parse_llm_json_response(response)
        return self._convert_to_entities(result_data.get('entities', []))

    def _extract_financial_patterns(self, content: str) -> List[ExtractedEntity]:
        """Pattern-based financial extraction"""
        entities = []

        # Financial title patterns
        financial_titles = [
            'Chief Financial Officer', 'CFO', 'Treasurer', 'Controller',
            'Financial Analyst', 'Finance Director', 'Finance Manager'
        ]

        for title in financial_titles:
            # Find names associated with these titles
            patterns = [
                rf'([A-Z][a-z]+ [A-Z][a-z]+),?\s+{re.escape(title)}',
                rf'{re.escape(title)}:?\s+([A-Z][a-z]+ [A-Z][a-z]+)',
                rf'([A-Z][a-z]+ [A-Z][a-z]+)\s*-\s*{re.escape(title)}'
            ]

            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    name = match.strip() if isinstance(match, str) else match[0].strip()
                    if len(name.split()) >= 2:
                        entities.append(ExtractedEntity(
                            name=name,
                            title=title,
                            confidence=0.7,
                            context=f"Financial pattern: {name} - {title}",
                            source_section="financial_pattern",
                            entity_type="financial_person"
                        ))

        return entities

    def _convert_to_entities(self, entity_data: List[Dict]) -> List[ExtractedEntity]:
        """Convert to ExtractedEntity objects"""
        entities = []
        for data in entity_data:
            if data.get('name'):
                entities.append(ExtractedEntity(
                    name=data['name'],
                    title=data.get('title'),
                    department=data.get('department'),
                    confidence=data.get('confidence', 0.5),
                    context=data.get('context', ''),
                    entity_type="financial_person"
                ))
        return entities

    def build_extraction_metadata(
        self,
        entities: List[ExtractedEntity],
        confidence: float,
        context: MetadataContext
    ) -> Dict[str, Any]:
        """Build financial-specific metadata"""
        financial_titles = [e.title for e in entities if e.title]
        leadership_titles = [t for t in financial_titles if any(x in t.lower() for x in ['cfo', 'chief', 'director'])]

        return {
            'extraction_type': 'financial',
            'total_entities': len(entities),
            'financial_titles': financial_titles,
            'leadership_roles': leadership_titles,
            'departments_mentioned': list(set(e.department for e in entities if e.department))
        }

    def get_document_type(self) -> str:
        return "financial_structure"

    def get_extraction_method(self) -> ExtractionMethod:
        return ExtractionMethod.FULL_LLM


class ApprovalMatrixExtractor(BaseExtractor):
    """Extract approval matrix information"""

    async def extract_entities(
        self,
        content: str,
        context: MetadataContext,
        llm_client=None
    ) -> List[ExtractedEntity]:
        """Extract approval matrix entities"""

        if llm_client:
            try:
                return await self._extract_with_llm(content, llm_client)
            except Exception as e:
                logger.warning(f"Approval matrix LLM extraction failed: {e}")

        return self._extract_approval_patterns(content)

    async def _extract_with_llm(self, content: str, llm_client) -> List[ExtractedEntity]:
        """LLM-based approval matrix extraction"""
        prompt = f"""Extract approval authorities and limits from this document.

Document text:
{content}

Focus on who can approve what amounts or processes.

Return JSON:
{{
    "entities": [
        {{
            "name": "person or role name",
            "title": "job title if specified",
            "authority_level": "approval limit or authority type",
            "confidence": 0.0-1.0,
            "context": "approval context"
        }}
    ]
}}"""

        response = await self._query_llm_with_retry(llm_client, prompt)
        if not response:
            return []

        result_data = self._parse_llm_json_response(response)
        return self._convert_to_entities(result_data.get('entities', []))

    def _extract_approval_patterns(self, content: str) -> List[ExtractedEntity]:
        """Pattern-based approval extraction"""
        entities = []

        # Approval patterns
        approval_patterns = [
            r'([A-Z][a-z]+ [A-Z][a-z]+).*?approve.*?\$?([\d,]+)',
            r'([A-Z][a-zA-Z\s]+?):\s*\$?([\d,]+)',
            r'\$?([\d,]+).*?([A-Z][a-zA-Z\s]+?)(?=\n|\.)'
        ]

        for pattern in approval_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if len(match) == 2:
                    name_or_role, amount = match
                    entities.append(ExtractedEntity(
                        name=name_or_role.strip(),
                        authority_level=f"${amount}",
                        confidence=0.6,
                        context=f"Approval authority: {name_or_role} - ${amount}",
                        entity_type="approval_authority"
                    ))

        return entities

    def _convert_to_entities(self, entity_data: List[Dict]) -> List[ExtractedEntity]:
        """Convert to ExtractedEntity objects"""
        entities = []
        for data in entity_data:
            if data.get('name'):
                entities.append(ExtractedEntity(
                    name=data['name'],
                    title=data.get('title'),
                    authority_level=data.get('authority_level'),
                    confidence=data.get('confidence', 0.5),
                    context=data.get('context', ''),
                    entity_type="approval_authority"
                ))
        return entities

    def build_extraction_metadata(
        self,
        entities: List[ExtractedEntity],
        confidence: float,
        context: MetadataContext
    ) -> Dict[str, Any]:
        """Build approval matrix metadata"""
        authority_levels = [e.authority_level for e in entities if e.authority_level]

        return {
            'extraction_type': 'approval_matrix',
            'total_authorities': len(entities),
            'authority_levels': authority_levels,
            'has_monetary_limits': any('$' in str(level) for level in authority_levels)
        }

    def get_document_type(self) -> str:
        return "approval_matrix"

    def get_extraction_method(self) -> ExtractionMethod:
        return ExtractionMethod.FULL_LLM


class GeneralExtractor(BaseExtractor):
    """General purpose extractor for any document type"""

    async def extract_entities(
        self,
        content: str,
        context: MetadataContext,
        llm_client=None
    ) -> List[ExtractedEntity]:
        """General entity extraction"""

        # Try LLM if available
        if llm_client:
            try:
                return await self._extract_with_llm(content, llm_client)
            except Exception as e:
                logger.warning(f"General LLM extraction failed: {e}")

        # Fallback to basic name extraction
        return self._extract_general_patterns(content)

    async def _extract_with_llm(self, content: str, llm_client) -> List[ExtractedEntity]:
        """LLM-based general extraction"""
        prompt = f"""Extract any names, titles, and roles from this document.

Document text:
{content}

Extract any people mentioned with their roles or titles.

Return JSON:
{{
    "entities": [
        {{
            "name": "person name",
            "title": "title or role if mentioned",
            "confidence": 0.0-1.0,
            "context": "context where found"
        }}
    ]
}}"""

        response = await self._query_llm_with_retry(llm_client, prompt)
        if not response:
            return []

        result_data = self._parse_llm_json_response(response)
        return self._convert_to_entities(result_data.get('entities', []))

    def _extract_general_patterns(self, content: str) -> List[ExtractedEntity]:
        """Basic pattern extraction for names"""
        entities = []

        # Simple name patterns
        name_patterns = [
            r'([A-Z][a-z]+ [A-Z][a-z]+)',  # Two-word names
            r'([A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+)',  # Name with middle initial
        ]

        names_found = set()
        for pattern in name_patterns:
            matches = re.findall(pattern, content)
            for name in matches:
                if name not in names_found and len(name.split()) >= 2:
                    names_found.add(name)
                    entities.append(ExtractedEntity(
                        name=name,
                        confidence=0.4,
                        context=f"Name found in text: {name}",
                        entity_type="person"
                    ))

        return entities[:10]  # Limit to prevent spam

    def _convert_to_entities(self, entity_data: List[Dict]) -> List[ExtractedEntity]:
        """Convert to ExtractedEntity objects"""
        entities = []
        for data in entity_data:
            if data.get('name'):
                entities.append(ExtractedEntity(
                    name=data['name'],
                    title=data.get('title'),
                    confidence=data.get('confidence', 0.5),
                    context=data.get('context', ''),
                    entity_type="person"
                ))
        return entities

    def build_extraction_metadata(
        self,
        entities: List[ExtractedEntity],
        confidence: float,
        context: MetadataContext
    ) -> Dict[str, Any]:
        """Build general metadata"""
        return {
            'extraction_type': 'general',
            'total_entities': len(entities),
            'extraction_method': 'general_purpose'
        }

    def get_document_type(self) -> str:
        return "general"

    def get_extraction_method(self) -> ExtractionMethod:
        return ExtractionMethod.LIGHT_LLM