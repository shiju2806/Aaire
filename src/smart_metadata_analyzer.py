"""
Smart Metadata Analyzer for AAIRE RAG System
============================================

Provides flexible, domain-agnostic metadata extraction and query intent analysis
to prevent cross-contamination in document retrieval.

Features:
- LLM-powered metadata extraction from document content
- Query intent analysis for dynamic filtering
- Domain-agnostic design (works with regulatory, legal, technical, medical docs)
- Graceful fallbacks and error handling
"""

import os
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import structlog
from openai import AsyncOpenAI

logger = structlog.get_logger()


def strip_markdown_json(response_content: str) -> str:
    """
    Strip markdown code blocks from OpenAI response to extract raw JSON
    Handles cases where OpenAI returns JSON wrapped in ```json ... ```
    """
    content = response_content.strip()

    # Find JSON code block using regex to handle text before the block
    import re

    # Pattern to match ```json ... ``` or ``` ... ``` blocks
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(json_block_pattern, content, re.DOTALL)

    if match:
        # Extract content from within the code block
        content = match.group(1).strip()
    else:
        # Fallback: try simple start/end patterns
        if content.startswith('```json') and content.endswith('```'):
            content = content[7:-3].strip()
        elif content.startswith('```') and content.endswith('```'):
            content = content[3:-3].strip()

    return content


@dataclass
class QueryIntent:
    """Represents the detected intent from a user query"""
    content_domains: List[str]  # e.g., ["regulatory", "insurance"]
    context_tags: List[str]     # e.g., ["usstat", "reserves"]
    required_filters: Dict[str, Any]  # Must match these
    excluded_filters: Dict[str, Any]  # Must NOT match these
    confidence: float           # 0.0 to 1.0
    reasoning: str              # Why this intent was detected


@dataclass
class DocumentMetadata:
    """Flexible metadata structure for any document type"""
    # Core fields (always present)
    source_document: str
    content_domain: str         # regulatory, legal, technical, medical, etc.

    # Dynamic fields (extracted by LLM)
    context_tags: List[str]     # Most flexible - can be anything
    content_type: Optional[str] = None  # contract, regulation, manual, report
    jurisdiction: Optional[str] = None  # us_ca, eu, uk, etc.
    framework: Optional[str] = None     # usstat, ifrs, gaap, etc.
    version: Optional[str] = None       # 2024, v1.28, etc.
    language: str = "en"

    # Open-ended attributes for domain-specific metadata
    attributes: Dict[str, Any] = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


class SmartMetadataAnalyzer:
    """
    LLM-powered metadata extraction and query intent analysis
    """

    def __init__(self, openai_api_key: str = None):
        """Initialize with OpenAI client for LLM analysis"""
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.warning("No OpenAI API key provided - metadata analysis will be limited")
            self.llm_client = None
        else:
            self.llm_client = AsyncOpenAI(api_key=self.openai_api_key)

        # Enable/disable smart filtering (for gradual rollout)
        self.smart_filtering_enabled = os.getenv("ENABLE_SMART_FILTERING", "true").lower() == "true"

        logger.info("Smart metadata analyzer initialized",
                   llm_available=bool(self.llm_client),
                   smart_filtering=self.smart_filtering_enabled)

    def extract_basic_metadata(self, content: str, filename: str, doc_type: str) -> DocumentMetadata:
        """
        Extract basic metadata without LLM (fast fallback)
        Uses pattern matching and heuristics
        """
        try:
            # Initialize with defaults
            metadata = DocumentMetadata(
                source_document=filename,
                content_domain=self._detect_domain_basic(content, filename),
                context_tags=[]
            )

            # Extract basic patterns
            context_tags = set()

            # Regulatory patterns
            if any(term in content.lower() for term in ['usstat', 'us statutory', 'valuation manual']):
                context_tags.add('usstat')
                metadata.framework = 'usstat'
                metadata.content_domain = 'regulatory'

            if any(term in content.lower() for term in ['ifrs', 'international financial reporting']):
                context_tags.add('ifrs')
                metadata.framework = 'ifrs'
                metadata.content_domain = 'regulatory'

            if any(term in content.lower() for term in ['gaap', 'generally accepted accounting']):
                context_tags.add('gaap')
                metadata.framework = 'gaap'
                metadata.content_domain = 'regulatory'

            # Legal patterns
            if any(term in content.lower() for term in ['contract', 'agreement', 'terms and conditions']):
                metadata.content_domain = 'legal'
                metadata.content_type = 'contract'

            # Technical patterns
            if any(term in content.lower() for term in ['api', 'technical specification', 'user manual']):
                metadata.content_domain = 'technical'

            # Extract version patterns
            version_match = re.search(r'version\s+(\d+\.?\d*)', content.lower())
            if version_match:
                metadata.version = version_match.group(1)

            # Extract year patterns
            year_match = re.search(r'20\d{2}', content)
            if year_match:
                context_tags.add(year_match.group(0))

            metadata.context_tags = list(context_tags)

            logger.debug("Basic metadata extracted",
                        domain=metadata.content_domain,
                        tags=metadata.context_tags,
                        framework=metadata.framework)

            return metadata

        except Exception as e:
            logger.error("Error in basic metadata extraction", exception_details=str(e))
            # Return minimal metadata as fallback
            return DocumentMetadata(
                source_document=filename,
                content_domain="unknown",
                context_tags=[]
            )

    async def extract_document_level_metadata(self, content: str, filename: str, doc_type: str) -> Dict[str, Any]:
        """
        Extract document-level metadata from filename and document overview
        This creates the primary framework classification that chunks inherit
        """
        if not self.llm_client:
            return self._extract_document_metadata_basic(content, filename)

        try:
            # Use first 2000 chars for document-level analysis
            document_overview = content[:2000] if len(content) > 2000 else content

            prompt = f"""
Analyze this document to determine its primary regulatory framework(s) and metadata.
This will be applied to ALL chunks from this document.

Document filename: {filename}
Document overview:
{document_overview}

Determine the document-level metadata in JSON format:
{{
    "primary_framework": "single_primary_framework|comparison|mixed|unknown",
    "frameworks": ["list", "of", "all", "frameworks", "mentioned"],
    "document_type": "regulation|comparison_guide|manual|report|contract|other",
    "content_domain": "regulatory|legal|technical|financial|other",
    "jurisdiction": "us|eu|uk|ca|international|other|null",
    "context_tags": ["broad", "document", "level", "keywords"],
    "attributes": {{"any": "additional_metadata"}}
}}

Guidelines:
- primary_framework: The MAIN framework (usstat, ifrs, gaap) or "comparison" if comparing multiple, "mixed" if unclear
- frameworks: ALL frameworks mentioned in the document (even if comparing)
- document_type: What kind of document this is
- context_tags: Important keywords at document level
- Be conservative - if unsure, use "unknown" or "mixed"

Examples:
- USSTAT manual → primary_framework: "usstat", frameworks: ["usstat"]
- IFRS vs USSTAT comparison → primary_framework: "comparison", frameworks: ["ifrs", "usstat"]
- Mixed regulatory document → primary_framework: "mixed", frameworks: ["usstat", "ifrs", "gaap"]
"""

            response = await self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400
            )

            response_content = response.choices[0].message.content.strip()
            if not response_content:
                raise ValueError("Empty response from OpenAI")

            try:
                cleaned_content = strip_markdown_json(response_content)
                doc_metadata = json.loads(cleaned_content)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed for document metadata: {e}")
                logger.warning(f"Response content: {response_content[:200]}...")
                raise ValueError(f"Invalid JSON response: {e}")

            logger.info("Document-level metadata extracted",
                       filename=filename,
                       primary_framework=doc_metadata.get("primary_framework"),
                       frameworks=doc_metadata.get("frameworks", []))

            return doc_metadata

        except Exception as e:
            logger.warning("Document-level metadata extraction failed, using basic", exception_details=str(e))
            return self._extract_document_metadata_basic(content, filename)

    async def extract_chunk_metadata(self, chunk_content: str, document_metadata: Dict[str, Any],
                                   chunk_index: int) -> DocumentMetadata:
        """
        Extract chunk-specific metadata that inherits from document-level metadata
        with optional refinement based on chunk content
        """
        if not self.llm_client:
            return self._create_chunk_metadata_basic(chunk_content, document_metadata, chunk_index)

        try:
            # Only analyze chunk focus if document has multiple frameworks
            frameworks = document_metadata.get("frameworks", [])
            if len(frameworks) <= 1:
                # Single framework document - just inherit
                return self._create_chunk_metadata_inherited(document_metadata, chunk_index)

            # Multi-framework document - analyze chunk focus
            prompt = f"""
This chunk is from a document with multiple frameworks: {frameworks}
Document primary framework: {document_metadata.get("primary_framework", "unknown")}

Analyze which specific framework(s) this chunk focuses on:

Chunk content:
{chunk_content[:1000]}

Respond in JSON format:
{{
    "chunk_focus": "primary_framework_for_this_chunk|mixed",
    "confidence": 0.85,
    "reasoning": "brief explanation"
}}

Guidelines:
- chunk_focus: Which framework this specific chunk discusses most
- If chunk discusses multiple equally, use "mixed"
- If unsure, use the document's primary_framework
- confidence: 0.0 to 1.0
"""

            response = await self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )

            response_content = response.choices[0].message.content.strip()
            if not response_content:
                raise ValueError("Empty response from OpenAI")

            try:
                cleaned_content = strip_markdown_json(response_content)
                chunk_analysis = json.loads(cleaned_content)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed for chunk metadata: {e}")
                logger.warning(f"Response content: {response_content[:200]}...")
                raise ValueError(f"Invalid JSON response: {e}")

            # Create chunk metadata with inheritance + refinement
            metadata = DocumentMetadata(
                source_document=document_metadata.get("source_document", "unknown"),
                content_domain=document_metadata.get("content_domain", "regulatory"),
                context_tags=document_metadata.get("context_tags", []),
                content_type=document_metadata.get("document_type"),
                jurisdiction=document_metadata.get("jurisdiction"),
                framework=document_metadata.get("primary_framework"),  # Inherit primary
                attributes={
                    # Inherit document-level metadata
                    "primary_framework": document_metadata.get("primary_framework"),
                    "frameworks": frameworks,
                    "document_type": document_metadata.get("document_type"),
                    # Chunk-specific metadata
                    "chunk_focus": chunk_analysis.get("chunk_focus"),
                    "chunk_index": chunk_index,
                    "analysis_confidence": chunk_analysis.get("confidence", 0.5)
                }
            )

            logger.debug("Chunk metadata created with refinement",
                        chunk_index=chunk_index,
                        chunk_focus=chunk_analysis.get("chunk_focus"),
                        primary_framework=document_metadata.get("primary_framework"))

            return metadata

        except Exception as e:
            logger.warning("Chunk metadata refinement failed, using inheritance",
                          error=str(e), chunk_index=chunk_index)
            return self._create_chunk_metadata_inherited(document_metadata, chunk_index)

    async def extract_smart_metadata(self, content: str, filename: str, doc_type: str) -> DocumentMetadata:
        """
        Legacy method for backward compatibility
        Now uses document-level detection for single-chunk documents
        """
        doc_metadata = await self.extract_document_level_metadata(content, filename, doc_type)
        return await self.extract_chunk_metadata(content, doc_metadata, 0)

    async def analyze_query_intent(self, query: str) -> QueryIntent:
        """
        Analyze user query to determine filtering intent
        """
        if not self.llm_client or not self.smart_filtering_enabled:
            # Fallback to basic pattern matching
            return self._analyze_intent_basic(query)

        try:
            prompt = f"""
Analyze this user query to determine what content filtering should be applied.
Focus on the new enhanced metadata structure with document-level primary frameworks.

Query: "{query}"

Enhanced Metadata Structure:
- primary_framework: single main framework (usstat, ifrs, gaap, comparison, mixed, unknown)
- frameworks: all frameworks mentioned in document (e.g., ["usstat", "ifrs"])
- chunk_focus: specific framework focus for individual chunks
- document_type: type of document (regulation, comparison_guide, manual, etc.)

Determine filtering strategy:
1. What content domains are relevant? (regulatory, legal, technical, etc.)
2. What specific frameworks are mentioned? (usstat, ifrs, gaap)
3. Should we filter by primary_framework or allow comparison documents?
4. Should we REQUIRE certain metadata matches?
5. Should we EXCLUDE certain metadata?
6. How confident are you in this analysis? (0.0 to 1.0)

Respond in JSON format:
{{
    "content_domains": ["list", "of", "relevant", "domains"],
    "context_tags": ["specific", "keywords", "mentioned"],
    "required_filters": {{"field": "value", "that": "must_match"}},
    "excluded_filters": {{"field": "value", "that": "must_not_match"}},
    "confidence": 0.85,
    "reasoning": "Why you made these decisions"
}}

Enhanced Examples:
- "usstat reserves" → require primary_framework in ["usstat", "comparison"]
- "universal life policy reserves in usstat" → require primary_framework in ["usstat", "comparison"]
- "ifrs accounting standards" → require primary_framework in ["ifrs", "comparison"]
- "compare usstat vs ifrs" → require primary_framework="comparison" OR frameworks contains both
- "general insurance question" → low confidence, minimal filtering
- "mixed regulatory approaches" → allow primary_framework="mixed"
"""

            response = await self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400
            )

            response_content = response.choices[0].message.content.strip()
            if not response_content:
                raise ValueError("Empty response from OpenAI")

            try:
                cleaned_content = strip_markdown_json(response_content)
                intent_data = json.loads(cleaned_content)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed for query intent: {e}")
                logger.warning(f"Response content: {response_content[:200]}...")
                raise ValueError(f"Invalid JSON response: {e}")

            intent = QueryIntent(
                content_domains=intent_data.get("content_domains", []),
                context_tags=intent_data.get("context_tags", []),
                required_filters=intent_data.get("required_filters", {}),
                excluded_filters=intent_data.get("excluded_filters", {}),
                confidence=intent_data.get("confidence", 0.5),
                reasoning=intent_data.get("reasoning", "")
            )

            logger.info("Query intent analyzed",
                       query=query[:50],
                       confidence=intent.confidence,
                       domains=intent.content_domains,
                       tags=intent.context_tags)

            return intent

        except Exception as e:
            logger.warning("Smart intent analysis failed, using basic analysis", exception_details=str(e))
            return self._analyze_intent_basic(query)

    def _detect_domain_basic(self, content: str, filename: str) -> str:
        """Basic domain detection using patterns"""
        content_lower = content.lower()
        filename_lower = filename.lower()

        # Regulatory indicators
        if any(term in content_lower for term in ['usstat', 'ifrs', 'gaap', 'regulation', 'compliance']):
            return 'regulatory'

        # Legal indicators
        if any(term in content_lower for term in ['contract', 'agreement', 'legal', 'terms']):
            return 'legal'

        # Technical indicators
        if any(term in content_lower for term in ['api', 'technical', 'specification', 'manual']):
            return 'technical'

        # Medical indicators
        if any(term in content_lower for term in ['medical', 'patient', 'diagnosis', 'treatment']):
            return 'medical'

        return 'general'

    def _analyze_intent_basic(self, query: str) -> QueryIntent:
        """Basic intent analysis using pattern matching"""
        query_lower = query.lower()

        context_tags = []
        required_filters = {}
        excluded_filters = {}
        confidence = 0.3  # Lower confidence for basic analysis

        # Check for specific regulatory frameworks
        if 'usstat' in query_lower:
            context_tags.append('usstat')
            required_filters['primary_framework'] = ['usstat', 'comparison', 'mixed']
            confidence = 0.9

        if 'ifrs' in query_lower:
            context_tags.append('ifrs')
            required_filters['primary_framework'] = ['ifrs', 'comparison', 'mixed']
            confidence = 0.9

        if 'gaap' in query_lower:
            context_tags.append('gaap')
            required_filters['primary_framework'] = ['gaap', 'comparison', 'mixed']
            confidence = 0.9

        # Check for comparison queries (should allow comparison documents)
        if any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference']):
            required_filters['primary_framework'] = ['comparison', 'mixed']  # Focus on comparison docs
            confidence = 0.8

        # Detect content domains
        content_domains = []
        if any(term in query_lower for term in ['regulation', 'statutory', 'compliance']):
            content_domains.append('regulatory')
        if any(term in query_lower for term in ['contract', 'legal', 'law']):
            content_domains.append('legal')

        return QueryIntent(
            content_domains=content_domains,
            context_tags=context_tags,
            required_filters=required_filters,
            excluded_filters=excluded_filters,
            confidence=confidence,
            reasoning=f"Basic pattern matching on query: {query[:50]}"
        )

    def _extract_document_metadata_basic(self, content: str, filename: str) -> Dict[str, Any]:
        """Basic document-level metadata extraction using patterns"""
        content_lower = content.lower()
        filename_lower = filename.lower()

        frameworks = []
        primary_framework = "unknown"

        # Detect frameworks
        if any(term in content_lower for term in ['usstat', 'us statutory', 'valuation manual']):
            frameworks.append('usstat')
            primary_framework = 'usstat'

        if any(term in content_lower for term in ['ifrs', 'international financial reporting']):
            frameworks.append('ifrs')
            if primary_framework == "unknown":
                primary_framework = 'ifrs'

        if any(term in content_lower for term in ['gaap', 'generally accepted accounting']):
            frameworks.append('gaap')
            if primary_framework == "unknown":
                primary_framework = 'gaap'

        # Determine if it's a comparison document
        if len(frameworks) > 1 or any(word in content_lower for word in ['compare', 'vs', 'versus', 'difference']):
            primary_framework = "comparison"

        # Detect document type
        document_type = "other"
        if any(term in filename_lower for term in ['manual', 'guide']):
            document_type = "manual"
        elif any(term in content_lower for term in ['comparison', 'compare']):
            document_type = "comparison_guide"
        elif any(term in content_lower for term in ['regulation', 'statutory']):
            document_type = "regulation"

        return {
            "source_document": filename,
            "primary_framework": primary_framework,
            "frameworks": frameworks if frameworks else ["unknown"],
            "document_type": document_type,
            "content_domain": "regulatory" if frameworks else "general",
            "jurisdiction": "us" if any(f in ["usstat", "gaap"] for f in frameworks) else None,
            "context_tags": frameworks + [str(year) for year in re.findall(r'20\d{2}', content)],
            "attributes": {}
        }

    def _create_chunk_metadata_inherited(self, document_metadata: Dict[str, Any], chunk_index: int) -> DocumentMetadata:
        """Create chunk metadata by inheriting document-level metadata"""
        return DocumentMetadata(
            source_document=document_metadata.get("source_document", "unknown"),
            content_domain=document_metadata.get("content_domain", "regulatory"),
            context_tags=document_metadata.get("context_tags", []),
            content_type=document_metadata.get("document_type"),
            jurisdiction=document_metadata.get("jurisdiction"),
            framework=document_metadata.get("primary_framework"),
            attributes={
                "primary_framework": document_metadata.get("primary_framework"),
                "frameworks": document_metadata.get("frameworks", []),
                "document_type": document_metadata.get("document_type"),
                "chunk_index": chunk_index,
                "chunk_focus": document_metadata.get("primary_framework")  # Same as primary for single-framework docs
            }
        )

    def _create_chunk_metadata_basic(self, chunk_content: str, document_metadata: Dict[str, Any], chunk_index: int) -> DocumentMetadata:
        """Create chunk metadata using basic patterns when LLM unavailable"""
        return self._create_chunk_metadata_inherited(document_metadata, chunk_index)

    def create_metadata_dict(self, metadata: DocumentMetadata) -> Dict[str, Any]:
        """Convert DocumentMetadata to dictionary for storage"""
        result = {
            'source_document': metadata.source_document,
            'content_domain': metadata.content_domain,
            'context_tags': metadata.context_tags,
            'language': metadata.language
        }

        # Add optional fields if present
        if metadata.content_type:
            result['content_type'] = metadata.content_type
        if metadata.jurisdiction:
            result['jurisdiction'] = metadata.jurisdiction
        if metadata.framework:
            result['framework'] = metadata.framework
        if metadata.version:
            result['version'] = metadata.version

        # Add custom attributes
        if metadata.attributes:
            result.update(metadata.attributes)

        return result