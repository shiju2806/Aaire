"""
Document fingerprinting for deduplication and version tracking
Industry-standard content-based hashing approach
"""

import hashlib
import re
import time
from typing import Dict, Any, Optional
import structlog

from .models import DocumentFingerprint

logger = structlog.get_logger()


class DocumentFingerprinter:
    """
    Generate unique fingerprints for documents to prevent duplicates
    Uses composite hashing: content + structure + metadata
    """

    def __init__(self):
        self.normalization_patterns = [
            (r'\s+', ' '),  # Normalize whitespace
            (r'[^\w\s\-_.]', ''),  # Remove special chars except common ones
            (r'\d{4}-\d{2}-\d{2}', 'DATE'),  # Normalize dates
            (r'\d+\.\d+', 'NUMBER'),  # Normalize numbers
        ]

    def generate_fingerprint(
        self,
        content: str,
        metadata: Dict[str, Any],
        filename: Optional[str] = None
    ) -> DocumentFingerprint:
        """
        Generate comprehensive document fingerprint

        Args:
            content: Document text content
            metadata: Document metadata dictionary
            filename: Optional filename

        Returns:
            DocumentFingerprint with all hash components
        """
        start_time = time.time()

        # Generate individual hash components
        content_hash = self._generate_content_hash(content)
        structure_hash = self._generate_structure_hash(content)
        metadata_hash = self._generate_metadata_hash(metadata)

        # Create composite fingerprint
        composite_data = f"{content_hash}:{structure_hash}:{metadata_hash}"
        composite_fingerprint = hashlib.sha256(composite_data.encode()).hexdigest()[:32]

        # Generate unique document ID
        document_id = self._generate_document_id(composite_fingerprint, filename)

        processing_time = (time.time() - start_time) * 1000

        fingerprint = DocumentFingerprint(
            document_id=document_id,
            content_hash=content_hash,
            structure_hash=structure_hash,
            metadata_hash=metadata_hash,
            composite_fingerprint=composite_fingerprint,
            file_size=len(content.encode('utf-8'))
        )

        logger.debug(
            "Document fingerprint generated",
            document_id=document_id,
            content_size=len(content),
            processing_time_ms=f"{processing_time:.2f}"
        )

        return fingerprint

    def _generate_content_hash(self, content: str) -> str:
        """Generate hash based on normalized content"""
        normalized_content = self._normalize_content(content)
        return hashlib.sha256(normalized_content.encode('utf-8')).hexdigest()[:16]

    def _generate_structure_hash(self, content: str) -> str:
        """Generate hash based on document structure (headings, lists, etc.)"""
        structure_elements = []

        # Extract structural elements
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect headers (lines that are all caps or start with numbers)
            if line.isupper() and len(line) > 5:
                structure_elements.append('HEADER')
            elif re.match(r'^\d+\.?\s', line):
                structure_elements.append('NUMBERED_ITEM')
            elif re.match(r'^[•\-*]\s', line):
                structure_elements.append('BULLET_ITEM')
            elif len(line) < 50 and ':' in line:
                structure_elements.append('LABEL')

        structure_signature = ''.join(structure_elements)
        return hashlib.md5(structure_signature.encode('utf-8')).hexdigest()[:12]

    def _generate_metadata_hash(self, metadata: Dict[str, Any]) -> str:
        """Generate hash from relevant metadata fields"""
        # Only include stable metadata fields that indicate content changes
        relevant_fields = ['filename', 'document_type', 'primary_framework', 'content_domains']

        relevant_metadata = {}
        for field in relevant_fields:
            if field in metadata and metadata[field]:
                relevant_metadata[field] = metadata[field]

        # Sort keys for consistent hashing
        metadata_str = str(sorted(relevant_metadata.items()))
        return hashlib.md5(metadata_str.encode('utf-8')).hexdigest()[:8]

    def _generate_document_id(self, composite_fingerprint: str, filename: Optional[str] = None) -> str:
        """Generate unique document ID"""
        if filename:
            # Use filename + fingerprint for readability
            clean_filename = re.sub(r'[^\w\-_.]', '', filename.split('.')[0])[:20]
            return f"{clean_filename}_{composite_fingerprint[:12]}"
        else:
            return f"doc_{composite_fingerprint[:16]}"

    def _normalize_content(self, content: str) -> str:
        """Normalize content for consistent hashing"""
        normalized = content.lower().strip()

        # Apply normalization patterns
        for pattern, replacement in self.normalization_patterns:
            normalized = re.sub(pattern, replacement, normalized)

        return normalized

    def is_similar_fingerprint(
        self,
        fingerprint1: DocumentFingerprint,
        fingerprint2: DocumentFingerprint,
        similarity_threshold: float = 0.95
    ) -> bool:
        """
        Check if two documents are similar based on their fingerprints

        Args:
            fingerprint1: First document fingerprint
            fingerprint2: Second document fingerprint
            similarity_threshold: Similarity threshold (0.0-1.0)

        Returns:
            True if documents are similar
        """
        # Compare structure similarity (exact match for structure)
        structure_match = fingerprint1.structure_hash == fingerprint2.structure_hash

        # Compare content similarity using Hamming distance on hex strings
        content_similarity = self._calculate_hex_similarity(
            fingerprint1.content_hash,
            fingerprint2.content_hash
        )

        # Metadata should be similar for similar documents
        metadata_match = fingerprint1.metadata_hash == fingerprint2.metadata_hash

        # Calculate overall similarity
        overall_similarity = (
            (1.0 if structure_match else 0.0) * 0.3 +
            content_similarity * 0.6 +
            (1.0 if metadata_match else 0.0) * 0.1
        )

        logger.debug(
            "Document similarity calculated",
            doc1_id=fingerprint1.document_id,
            doc2_id=fingerprint2.document_id,
            structure_match=structure_match,
            content_similarity=content_similarity,
            metadata_match=metadata_match,
            overall_similarity=overall_similarity
        )

        return overall_similarity >= similarity_threshold

    def _calculate_hex_similarity(self, hex1: str, hex2: str) -> float:
        """Calculate similarity between two hex strings"""
        if len(hex1) != len(hex2):
            return 0.0

        matches = sum(c1 == c2 for c1, c2 in zip(hex1, hex2))
        return matches / len(hex1)

    def detect_version_change(
        self,
        original_fingerprint: DocumentFingerprint,
        new_fingerprint: DocumentFingerprint
    ) -> Dict[str, Any]:
        """
        Detect what changed between two versions of a document

        Returns:
            Dictionary with change detection results
        """
        changes = {
            'is_new_version': False,
            'content_changed': False,
            'structure_changed': False,
            'metadata_changed': False,
            'severity': 'none'
        }

        # Check for content changes
        if original_fingerprint.content_hash != new_fingerprint.content_hash:
            changes['content_changed'] = True
            changes['is_new_version'] = True

        # Check for structure changes
        if original_fingerprint.structure_hash != new_fingerprint.structure_hash:
            changes['structure_changed'] = True
            changes['is_new_version'] = True

        # Check for metadata changes
        if original_fingerprint.metadata_hash != new_fingerprint.metadata_hash:
            changes['metadata_changed'] = True

        # Determine severity
        if changes['structure_changed']:
            changes['severity'] = 'major'
        elif changes['content_changed']:
            changes['severity'] = 'moderate'
        elif changes['metadata_changed']:
            changes['severity'] = 'minor'

        logger.info(
            "Document version change detected",
            original_doc=original_fingerprint.document_id,
            new_doc=new_fingerprint.document_id,
            **changes
        )

        return changes


class FingerprintCache:
    """In-memory cache for document fingerprints with TTL"""

    def __init__(self, ttl_hours: int = 24):
        self.cache: Dict[str, tuple] = {}  # fingerprint -> (data, timestamp)
        self.ttl_seconds = ttl_hours * 3600

    def get(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        """Get cached data for fingerprint"""
        if fingerprint in self.cache:
            data, timestamp = self.cache[fingerprint]
            if time.time() - timestamp < self.ttl_seconds:
                return data
            else:
                # Expired, remove from cache
                del self.cache[fingerprint]
        return None

    def put(self, fingerprint: str, data: Dict[str, Any]):
        """Cache data for fingerprint"""
        self.cache[fingerprint] = (data, time.time())

    def clear_expired(self):
        """Remove expired entries from cache"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp >= self.ttl_seconds
        ]
        for key in expired_keys:
            del self.cache[key]

    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)