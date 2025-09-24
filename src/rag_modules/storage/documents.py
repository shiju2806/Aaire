"""
Document Management Module for RAG Pipeline

This module extracts document management functionality from the monolithic RAG pipeline
into a clean, reusable DocumentManager class. It handles all document CRUD operations
including adding, deleting, and querying documents in the vector store.

Extracted from RAG pipeline as part of Phase 4 refactoring (~400 lines of functionality).
"""

import os
import uuid
import asyncio
import shutil
import hashlib
import random
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

# LlamaIndex imports
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser

# Logger import
import structlog
logger = structlog.get_logger(__name__)


class DocumentManager:
    """
    Handles all document management operations for the RAG pipeline.

    This class provides a clean interface for:
    - Adding documents with metadata extraction
    - Deleting documents and their chunks
    - Clearing all documents
    - Cleaning up orphaned chunks
    - Getting document statistics

    Dependencies are injected to allow for flexible testing and deployment.
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        node_parser: SimpleNodeParser,
        metadata_analyzer,
        whoosh_engine,
        cache=None,
        vector_store_type: str = "qdrant",
        qdrant_client=None,
        collection_name: str = "aaire_docs"
    ):
        """
        Initialize DocumentManager with injected dependencies.

        Args:
            index: LlamaIndex VectorStoreIndex for document storage
            node_parser: Parser for breaking documents into chunks
            metadata_analyzer: Service for extracting document/chunk metadata
            whoosh_engine: Search engine for keyword-based search
            cache: Redis cache for query caching (optional)
            vector_store_type: Type of vector store ("qdrant" or "local")
            qdrant_client: QdrantClient instance (required if vector_store_type="qdrant")
            collection_name: Name of Qdrant collection
        """
        self.index = index
        self.node_parser = node_parser
        self.metadata_analyzer = metadata_analyzer
        self.whoosh_engine = whoosh_engine
        self.cache = cache
        self.vector_store_type = vector_store_type
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name

        # Validate required dependencies based on vector store type
        if self.vector_store_type == "qdrant" and not self.qdrant_client:
            raise ValueError("qdrant_client is required when vector_store_type is 'qdrant'")

    async def add_documents(self, documents: List[Document], doc_type: str = "company") -> int:
        """
        Add documents to the single index with document type metadata

        Args:
            documents: List of Document objects to add
            doc_type: Type of documents being added (default: "company")

        Returns:
            Number of chunks/nodes created from the documents

        Raises:
            Exception: If document processing fails
        """
        try:
            # Add document type metadata to each document
            for doc in documents:
                if not doc.metadata:
                    doc.metadata = {}
                doc.metadata['doc_type'] = doc_type
                doc.metadata['added_at'] = datetime.utcnow().isoformat()

            # Parse documents into nodes
            nodes = self.node_parser.get_nodes_from_documents(documents)

            # Extract document-level metadata for each document using new approach
            logger.info(f"Extracting document-level metadata for {len(documents)} documents")
            document_metadata_cache = {}

            for doc in documents:
                try:
                    filename = doc.metadata.get('filename', 'Unknown') if doc.metadata else 'Unknown'

                    # Step 1: Extract document-level metadata (primary framework detection)
                    # Use hierarchical sampling instead of full document to avoid OpenAI rate limits
                    sampled_content = self._sample_document_for_metadata(doc.text, filename)
                    doc_level_metadata = await self.metadata_analyzer.extract_document_level_metadata(
                        content=sampled_content,
                        filename=filename,
                        doc_type=doc_type
                    )

                    # Store for chunk inheritance
                    document_metadata_cache[filename] = doc_level_metadata

                    # Add document-level metadata to the document
                    if not doc.metadata:
                        doc.metadata = {}
                    doc.metadata.update(doc_level_metadata)

                    logger.info(f"Document-level metadata extracted for {filename}",
                               primary_framework=doc_level_metadata.get('primary_framework'),
                               frameworks=doc_level_metadata.get('frameworks', []),
                               document_type=doc_level_metadata.get('document_type'))

                except Exception as e:
                    logger.warning(f"Document-level metadata extraction failed for document",
                                 filename=filename, exception_details=str(e))
                    # Create fallback document metadata
                    document_metadata_cache[filename] = {
                        'source_document': filename,
                        'primary_framework': 'unknown',
                        'frameworks': ['unknown'],
                        'document_type': 'other',
                        'content_domain': 'general',
                        'context_tags': []
                    }

            # Step 2: Process chunks with document-level inheritance + chunk-level refinement
            logger.info(f"Processing {len(nodes)} chunks with inheritance + refinement")
            processed_chunks = 0

            # Parallel processing with controlled concurrency
            parallel_limit = 8  # Process 8 chunks concurrently
            semaphore = asyncio.Semaphore(parallel_limit)

            async def process_chunk(chunk_index, node):
                async with semaphore:
                    if not node.metadata:
                        node.metadata = {}

                    # Find the source document for this chunk
                    source_doc = None
                    if hasattr(node, 'ref_doc_id') and documents:
                        # Try to match by document ID
                        for doc in documents:
                            if hasattr(doc, 'doc_id') and doc.doc_id == node.ref_doc_id:
                                source_doc = doc
                                break
                        # Fallback to first document if no match found
                        if source_doc is None:
                            source_doc = documents[0]
                    elif documents:
                        source_doc = documents[0]

                    # Get document-level metadata
                    filename = source_doc.metadata.get('filename', 'Unknown') if source_doc and source_doc.metadata else 'Unknown'
                    doc_metadata = document_metadata_cache.get(filename, {})

                    try:
                        # Generate chunk metadata with inheritance + refinement
                        chunk_metadata_obj = await self.metadata_analyzer.extract_chunk_metadata(
                            chunk_content=getattr(node, 'text', '') or getattr(node, 'content', ''),
                            document_metadata=doc_metadata,
                            chunk_index=chunk_index
                        )

                        # Convert chunk metadata to dictionary
                        chunk_metadata_dict = self.metadata_analyzer.create_metadata_dict(chunk_metadata_obj)

                        # Add chunk metadata to node
                        node.metadata.update(chunk_metadata_dict)

                        # Add system metadata
                        node.metadata['doc_type'] = doc_type
                        node.metadata['added_at'] = datetime.utcnow().isoformat()

                        # Ensure job_id and filename are preserved for deletion tracking
                        if source_doc and source_doc.metadata:
                            if 'job_id' in source_doc.metadata:
                                node.metadata['job_id'] = source_doc.metadata['job_id']
                            if 'filename' in source_doc.metadata:
                                node.metadata['filename'] = source_doc.metadata['filename']

                        # Log chunk processing (every 20th chunk or if it's refined)
                        if chunk_index % 20 == 0 or (hasattr(chunk_metadata_obj, 'attributes') and
                                                    chunk_metadata_obj.attributes.get('chunk_focus') !=
                                                    doc_metadata.get('primary_framework')):
                            logger.debug(f"Chunk {chunk_index} processed",
                                       filename=filename,
                                       primary_framework=doc_metadata.get('primary_framework'),
                                       chunk_focus=chunk_metadata_obj.attributes.get('chunk_focus') if hasattr(chunk_metadata_obj, 'attributes') else None)

                        return 1  # Success count

                    except Exception as e:
                        logger.warning(f"Chunk metadata processing failed for chunk {chunk_index}",
                                     filename=filename, exception_details=str(e))
                        return 0  # Failure count

            # Process all chunks in parallel
            logger.info(f"🚀 Starting parallel processing with {parallel_limit} concurrent workers")
            tasks = [process_chunk(chunk_index, node) for chunk_index, node in enumerate(nodes)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count successful processes
            processed_chunks = sum(r for r in results if isinstance(r, int) and r > 0)
            failed_chunks = len(results) - processed_chunks

            if failed_chunks > 0:
                logger.warning(f"⚠️ {failed_chunks} chunks failed processing")

            logger.info(f"✅ Parallel processing completed: {processed_chunks}/{len(nodes)} chunks processed successfully")

            # Process fallback for chunks that didn't get metadata
            for chunk_index, node in enumerate(nodes):
                if 'chunk_index' not in node.metadata:
                    # This chunk didn't get processed, apply fallback metadata
                    if document_metadata_cache:
                        first_doc_metadata = next(iter(document_metadata_cache.values()))
                        node.metadata.update(first_doc_metadata)
                    node.metadata['doc_type'] = doc_type
                    node.metadata['added_at'] = datetime.utcnow().isoformat()
                    node.metadata['chunk_index'] = chunk_index

                    # Ensure job_id and filename are preserved
                    if documents and documents[0].metadata:
                        if 'job_id' in documents[0].metadata:
                            node.metadata['job_id'] = documents[0].metadata['job_id']
                        if 'filename' in documents[0].metadata:
                            node.metadata['filename'] = documents[0].metadata['filename']

                # Preserve page information if available in node start_char_idx
                if hasattr(node, 'start_char_idx') and hasattr(node, 'ref_doc_id'):
                    # Try to estimate page number from character position
                    # This is approximate but better than no page info
                    char_idx = getattr(node, 'start_char_idx', 0)
                    # Rough estimate: 2000 characters per page
                    estimated_page = max(1, (char_idx // 2000) + 1)
                    node.metadata['estimated_page'] = estimated_page

                # Check if the node content contains page information from shape-aware extraction
                node_content = getattr(node, 'text', '') or getattr(node, 'content', '')
                if 'Source: Page' in node_content:
                    page_match = re.search(r'Source: Page (\d+)', node_content)
                    if page_match:
                        node.metadata['page'] = int(page_match.group(1))

            # Add to single index
            if self.index is not None:
                self.index.insert_nodes(nodes)
            else:
                # Initialize index if not present
                logger.warning("Index not initialized, attempting to create index")
                if self.vector_store_type == "qdrant":
                    self.index = self._init_qdrant_indexes()
                else:
                    self.index = self._init_local_index()

                if self.index is not None:
                    self.index.insert_nodes(nodes)
                else:
                    raise ValueError("Failed to initialize index for document insertion")

            # Update BM25 index for hybrid search
            self._update_whoosh_index(nodes)

            # Invalidate cache for this document type
            if self.cache:
                pattern = f"query_cache:{doc_type}:*"
                for key in self.cache.scan_iter(match=pattern):
                    self.cache.delete(key)

            # Log summary of document-level metadata processing
            logger.info(f"Document-level metadata processing completed",
                       documents_processed=len(documents),
                       chunks_processed=processed_chunks,
                       total_chunks=len(nodes))

            logger.info(f"Added {len(documents)} documents to index",
                       doc_type=doc_type,
                       total_nodes=len(nodes),
                       whoosh_docs=self.whoosh_engine.get_document_count() if self.whoosh_engine else 0)

            return len(nodes)

        except Exception as e:
            logger.error("Failed to add documents", exception_details=str(e), doc_type=doc_type)
            raise

    async def clear_all_documents(self) -> Dict[str, Any]:
        """Clear all documents from the vector store - use with caution"""
        try:
            if self.vector_store_type == "qdrant":
                # Delete and recreate the entire collection
                self.qdrant_client.delete_collection(self.collection_name)

                # Recreate the collection
                from qdrant_client.models import Distance, VectorParams
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )

                # Reinitialize the index
                self._init_qdrant_indexes()

                # Clear Whoosh index as well
                self._clear_whoosh_index()

                logger.info("Successfully cleared all documents from Qdrant and Whoosh")
                return {"status": "success", "message": "All documents cleared", "method": "qdrant_recreate"}
            else:
                # For local storage, recreate the index
                self._init_local_index()

                # Clear Whoosh index as well
                self._clear_whoosh_index()

                logger.info("Successfully cleared all documents from local storage and Whoosh")
                return {"status": "success", "message": "All documents cleared", "method": "local_recreate"}

        except Exception as e:
            logger.error("Failed to clear all documents", exception_details=str(e))
            return {"status": "error", "error": str(e)}

    async def delete_document(self, job_id: str) -> Dict[str, Any]:
        """Delete all chunks associated with a document from the vector store"""
        try:
            deleted_count = 0

            if self.vector_store_type == "qdrant":
                # Delete from Qdrant using metadata filter
                from qdrant_client.models import Filter, FieldCondition, MatchValue

                # Search for all points with this job_id
                search_result = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="job_id",
                                match=MatchValue(value=job_id)
                            )
                        ]
                    ),
                    limit=1000  # Get all chunks for this document
                )

                # Extract point IDs to delete
                point_ids = [point.id for point in search_result[0]]

                if point_ids:
                    # Delete the points
                    self.qdrant_client.delete(
                        collection_name=self.collection_name,
                        points_selector=point_ids
                    )
                    deleted_count = len(point_ids)
                    logger.info(f"Deleted {deleted_count} chunks from Qdrant for job_id: {job_id}")
                else:
                    logger.warning(f"No chunks found in Qdrant for job_id: {job_id}")


            else:
                # Local index doesn't support deletion by metadata easily
                logger.warning("Local index deletion not implemented - rebuild index recommended")

            # Clear cache entries related to this document
            if self.cache:
                # Clear all cache entries (simple approach for now)
                pattern = "query_cache:*"
                for key in self.cache.scan_iter(match=pattern):
                    self.cache.delete(key)
                logger.info("Cleared query cache after document deletion")

            return {
                "status": "success",
                "deleted_chunks": deleted_count,
                "job_id": job_id,
                "vector_store": self.vector_store_type
            }

        except Exception as e:
            logger.error(f"Failed to delete document from vector store", exception_details=str(e), job_id=job_id)
            return {
                "status": "error",
                "error": str(e),
                "job_id": job_id
            }

    async def cleanup_orphaned_chunks(self) -> Dict[str, Any]:
        """Clean up chunks that don't have valid job_ids (legacy data)"""
        try:
            cleaned_count = 0

            if self.vector_store_type == "qdrant":
                # Get all points without job_id
                from qdrant_client.models import Filter, IsNullCondition

                # Search for points without job_id
                search_result = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[
                            IsNullCondition(
                                key="job_id",
                                is_null=True
                            )
                        ]
                    ),
                    limit=1000
                )

                # Extract point IDs to delete
                point_ids = [point.id for point in search_result[0]]

                if point_ids:
                    # Delete the orphaned points
                    self.qdrant_client.delete(
                        collection_name=self.collection_name,
                        points_selector=point_ids
                    )
                    cleaned_count = len(point_ids)
                    logger.info(f"Cleaned {cleaned_count} orphaned chunks from Qdrant")

            return {
                "status": "success",
                "cleaned_chunks": cleaned_count,
                "vector_store": self.vector_store_type
            }

        except Exception as e:
            logger.error(f"Failed to cleanup orphaned chunks", exception_details=str(e))
            return {
                "status": "error",
                "error": str(e)
            }

    async def get_all_documents(self) -> Dict[str, Any]:
        """Get all documents currently in the vector store for debugging"""
        try:
            documents = []

            if self.vector_store_type == "qdrant":
                # Get all points in the collection
                search_result = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    limit=1000  # Adjust if you have more documents
                )

                for point in search_result[0]:
                    if point.payload:
                        documents.append({
                            "point_id": point.id,
                            "filename": point.payload.get("filename", "Unknown"),
                            "job_id": point.payload.get("job_id", "No job_id"),
                            "doc_type": point.payload.get("doc_type", "Unknown"),
                            "added_at": point.payload.get("added_at", "Unknown"),
                            "text_preview": self._extract_text_content(point.payload)[:100] + "..." if self._extract_text_content(point.payload) else ""
                        })

            return {
                "status": "success",
                "total_documents": len(documents),
                "documents": documents,
                "vector_store": self.vector_store_type
            }

        except Exception as e:
            logger.error(f"Failed to get all documents", exception_details=str(e))
            return {
                "status": "error",
                "error": str(e)
            }

    def _extract_text_content(self, payload):
        """Extract text content from Qdrant payload, handling JSON structure properly."""
        import json

        # Try direct text field first
        if payload.get('text'):
            return payload.get('text')
        if payload.get('content'):
            return payload.get('content')

        # Handle _node_content JSON structure
        if payload.get('_node_content'):
            try:
                if isinstance(payload['_node_content'], str):
                    node_data = json.loads(payload['_node_content'])
                    return node_data.get('text', '')
                elif isinstance(payload['_node_content'], dict):
                    return payload['_node_content'].get('text', '')
            except (json.JSONDecodeError, AttributeError):
                pass

        # Fallback to string representation
        return str(payload)

    def _sample_document_for_metadata(self, doc_text: str, filename: str) -> str:
        """
        Sample document content for metadata extraction to avoid OpenAI rate limits.
        Reduces from 237K tokens to ~30K tokens using strategic hierarchical sampling.

        Industry benchmark approach:
        - First 10 pages (executive summary, introduction)
        - Random strategic samples from middle
        - Last few pages (conclusions)
        - Key section headers and structured content
        """
        # Target ~30K tokens (vs 237K full document) to stay under 200K/min rate limit
        TARGET_TOKENS = 30000
        CHARS_PER_TOKEN = 4  # Rough approximation for English text
        TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN

        logger.info(f"Sampling document for metadata extraction",
                   filename=filename,
                   original_size=len(doc_text),
                   target_size=TARGET_CHARS)

        # If document is already small enough, return as-is
        if len(doc_text) <= TARGET_CHARS:
            logger.info("Document already within target size", size=len(doc_text))
            return doc_text

        # Split document into logical sections
        lines = doc_text.split('\n')
        total_lines = len(lines)

        # Calculate proportional sampling
        beginning_lines = int(total_lines * 0.15)  # First 15% (introduction, TOC)
        ending_lines = int(total_lines * 0.10)     # Last 10% (conclusions)
        middle_sample_lines = int(total_lines * 0.15) # 15% random sample from middle

        sampled_sections = []

        # 1. Beginning section (critical for framework identification)
        beginning_section = lines[:beginning_lines]
        sampled_sections.append("=== DOCUMENT BEGINNING ===")
        sampled_sections.extend(beginning_section)

        # 2. Strategic middle sampling (look for key sections)
        middle_start = beginning_lines
        middle_end = total_lines - ending_lines
        middle_lines = lines[middle_start:middle_end]

        # Dynamic content analysis for intelligent sampling
        content_lower = doc_text.lower()

        # Detect document frameworks dynamically
        framework_keywords = {
            'usstat': ['usstat', 'statutory', 'naic', 'reserve', 'rbc'],
            'ifrs': ['ifrs', 'international', 'ias', 'fair value'],
            'gaap': ['gaap', 'generally accepted', 'accounting principles'],
            'actuarial': ['mortality', 'morbidity', 'lapse', 'policyholder', 'benefit'],
            'valuation': ['valuation', 'methodology', 'assumption', 'discount rate'],
            'regulatory': ['regulation', 'compliance', 'requirement', 'standard']
        }

        detected_frameworks = []
        for framework, keywords in framework_keywords.items():
            if any(kw in content_lower for kw in keywords):
                detected_frameworks.append(framework)

        # Build dynamic keyword list based on detected frameworks
        dynamic_keywords = ['methodology', 'calculation', 'framework', 'standard', 'requirement', 'principle']
        for framework in detected_frameworks:
            dynamic_keywords.extend(framework_keywords[framework])

        # Find important middle sections with dynamic content awareness
        important_middle = []
        framework_sections = []  # Prioritize framework-specific content

        for i, line in enumerate(middle_lines):
            line_clean = line.strip()

            # Structural patterns (always important)
            is_structural = (line_clean and (
                re.match(r'^[A-Z][A-Z\s]{10,}$', line_clean) or  # ALL CAPS headers
                re.match(r'^\d+\.', line_clean) or                # Numbered sections
                re.match(r'^[IVXLC]+\.', line_clean) or          # Roman numerals
                re.match(r'^Table \d+', line_clean) or           # Tables
                re.match(r'^Section \d+', line_clean) or         # Sections
                re.match(r'^Appendix', line_clean)               # Appendices
            ))

            # Framework-specific content (highest priority)
            is_framework_specific = any(keyword in line_clean.lower() for keyword in dynamic_keywords)

            if is_structural or is_framework_specific:
                # Include this line and context
                start_idx = max(0, i-1)
                end_idx = min(len(middle_lines), i+4)
                section_lines = middle_lines[start_idx:end_idx]

                if is_framework_specific:
                    framework_sections.extend(section_lines)
                else:
                    important_middle.extend(section_lines)

        # Prioritize framework-specific content
        important_middle = framework_sections + important_middle

        # Remove duplicates while preserving order
        seen = set()
        deduplicated_middle = []
        for line in important_middle:
            if line not in seen:
                seen.add(line)
                deduplicated_middle.append(line)
        important_middle = deduplicated_middle

        # Use document-specific seed to prevent cross-contamination
        doc_hash = hashlib.md5(f"{filename}_{len(doc_text)}".encode()).hexdigest()

        # Add document-specific random samples from middle if we haven't captured enough
        if len(important_middle) < middle_sample_lines:
            remaining_middle = [line for line in middle_lines if line not in important_middle]

            doc_seed = int(doc_hash[:8], 16) % 2147483647  # Convert to valid seed
            random.seed(doc_seed)

            additional_samples = random.sample(
                remaining_middle,
                min(middle_sample_lines - len(important_middle), len(remaining_middle))
            )
            important_middle.extend(additional_samples)

        sampled_sections.append("\n=== DOCUMENT MIDDLE (KEY SECTIONS) ===")
        sampled_sections.extend(important_middle[:middle_sample_lines])

        # 3. Ending section (conclusions, references)
        ending_section = lines[-ending_lines:]
        sampled_sections.append("\n=== DOCUMENT ENDING ===")
        sampled_sections.extend(ending_section)

        # Combine and check size
        sampled_content = '\n'.join(sampled_sections)

        # If still too large, do secondary trimming
        if len(sampled_content) > TARGET_CHARS:
            # Truncate each section proportionally
            sections = sampled_content.split('=== DOCUMENT')
            trimmed_sections = []

            chars_per_section = TARGET_CHARS // len(sections)
            for section in sections:
                if len(section) > chars_per_section:
                    # Keep first part of each section (most important)
                    trimmed_sections.append(section[:chars_per_section] + "...[TRIMMED]")
                else:
                    trimmed_sections.append(section)

            sampled_content = '=== DOCUMENT'.join(trimmed_sections)

        logger.info(f"Document sampling completed",
                   filename=filename,
                   original_size=len(doc_text),
                   sampled_size=len(sampled_content),
                   reduction_ratio=f"{len(sampled_content) / len(doc_text) * 100:.1f}%",
                   detected_frameworks=detected_frameworks)

        return sampled_content

    def _update_whoosh_index(self, nodes):
        """Update Whoosh index with new document nodes"""
        try:
            if not self.whoosh_engine:
                logger.warning("Whoosh engine not available for indexing")
                return

            # Convert nodes to Whoosh document format
            whoosh_docs = []
            for node in nodes:
                text = node.get_content() if hasattr(node, 'get_content') else str(node.text)
                node_id = node.node_id if hasattr(node, 'node_id') else str(uuid.uuid4())
                metadata = node.metadata or {}

                whoosh_doc = {
                    'doc_id': node_id,
                    'content': text,
                    'title': metadata.get('filename', 'Unknown'),
                    'metadata': {
                        'node_id': node_id,
                        'filename': metadata.get('filename', 'Unknown'),
                        'doc_type': metadata.get('doc_type', 'company'),
                        'added_at': metadata.get('added_at', ''),
                        'page': metadata.get('page', 0),
                        'primary_framework': metadata.get('primary_framework', 'unknown'),
                        'content_domains': metadata.get('content_domains', []),
                        'document_type': metadata.get('document_type', 'unknown'),
                        'file_path': metadata.get('filename', 'Unknown'),
                        'confidence_score': metadata.get('confidence_score', 0.5),
                        **metadata  # Include all existing metadata
                    }
                }
                whoosh_docs.append(whoosh_doc)

            # Index documents in Whoosh (incremental)
            if whoosh_docs:
                indexed_count = self.whoosh_engine.add_documents(whoosh_docs, batch_processing=True)
                logger.info(f"✅ Whoosh index updated with {indexed_count} documents")
        except Exception as e:
            logger.error(f"Failed to update Whoosh index: {str(e)}")

    def _clear_whoosh_index(self):
        """Clear the Whoosh search index by deleting and recreating the index directory"""
        try:
            if self.whoosh_engine:
                whoosh_index_dir = self.whoosh_engine.index_dir

                # Delete the index directory if it exists
                if whoosh_index_dir.exists():
                    shutil.rmtree(str(whoosh_index_dir))
                    logger.info(f"Deleted Whoosh index directory: {whoosh_index_dir}")

                # Reinitialize the Whoosh engine
                self.whoosh_engine.initialize_index()
                logger.info("Reinitialized Whoosh search engine")
        except Exception as e:
            logger.error(f"Failed to clear Whoosh index: {str(e)}")

    def _init_qdrant_indexes(self):
        """Initialize Qdrant collection and indexes"""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                # Create collection if it doesn't exist
                from qdrant_client.models import Distance, VectorParams
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")

            # Create and return VectorStoreIndex with Qdrant
            from llama_index.core import VectorStoreIndex, StorageContext
            from llama_index.vector_stores.qdrant import QdrantVectorStore

            vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name
            )

            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )

            # Create index with the storage context
            self.index = VectorStoreIndex.from_documents(
                documents=[],  # Start with empty documents
                storage_context=storage_context
            )

            logger.info(f"Created VectorStoreIndex for Qdrant collection: {self.collection_name}")
            return self.index

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant indexes: {str(e)}")
            raise

    def _init_local_index(self):
        """Initialize local vector store as fallback"""
        # Create a simple in-memory vector store
        from llama_index.core import VectorStoreIndex
        self.index = VectorStoreIndex(
            nodes=[]
        )
        logger.info("Initialized local vector store")
        return self.index


def create_document_manager(
    index: VectorStoreIndex,
    node_parser: SimpleNodeParser,
    metadata_analyzer,
    whoosh_engine,
    cache=None,
    vector_store_type: str = "qdrant",
    qdrant_client=None,
    collection_name: str = "aaire_docs"
) -> DocumentManager:
    """
    Factory function to create a DocumentManager instance.

    Args:
        index: LlamaIndex VectorStoreIndex for document storage
        node_parser: Parser for breaking documents into chunks
        metadata_analyzer: Service for extracting document/chunk metadata
        whoosh_engine: Search engine for keyword-based search
        cache: Redis cache for query caching (optional)
        vector_store_type: Type of vector store ("qdrant" or "local")
        qdrant_client: QdrantClient instance (required if vector_store_type="qdrant")
        collection_name: Name of Qdrant collection

    Returns:
        Configured DocumentManager instance

    Raises:
        ValueError: If required dependencies are missing
    """
    return DocumentManager(
        index=index,
        node_parser=node_parser,
        metadata_analyzer=metadata_analyzer,
        whoosh_engine=whoosh_engine,
        cache=cache,
        vector_store_type=vector_store_type,
        qdrant_client=qdrant_client,
        collection_name=collection_name
    )