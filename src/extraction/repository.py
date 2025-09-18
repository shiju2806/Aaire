"""
Repository Pattern for Qdrant Integration
Clean abstraction over Qdrant operations with comprehensive error handling
"""

import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, Filter, FieldCondition, MatchValue, MatchAny,
    VectorParams, Distance, OptimizersConfig, HnswConfigDiff,
    PayloadSchemaType, SearchParams, CreateCollection, UpdateCollection
)
import structlog

from .models import DocumentMetadata, ExtractionResult, DocumentFingerprint
from .metadata_builder import MetadataValidator

logger = structlog.get_logger()


class DocumentRepository:
    """
    Repository pattern for clean Qdrant data access
    Abstracts away Qdrant-specific operations and provides clean interface
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str = "aaire-documents",
        qdrant_config: Optional[Dict[str, Any]] = None
    ):
        self.client = qdrant_client
        self.collection_name = collection_name
        self.config = qdrant_config or {}
        self._setup_collection()

    def _setup_collection(self):
        """Setup collection with proper configuration"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(col.name == self.collection_name for col in collections)

            if not collection_exists:
                logger.info(f"Creating new collection: {self.collection_name}")
                self._create_collection()
            else:
                logger.info(f"Using existing collection: {self.collection_name}")

            # Setup indexes for filtering
            self._setup_indexes()

        except Exception as e:
            logger.error(f"Failed to setup collection: {e}")
            raise

    def _create_collection(self):
        """Create Qdrant collection with optimized configuration"""
        vector_config = VectorParams(
            size=self.config.get('vector_size', 1536),
            distance=Distance.COSINE,
            on_disk=True  # Store vectors on disk for cost efficiency
        )

        # Optimized HNSW parameters
        hnsw_config = HnswConfigDiff(
            m=self.config.get('hnsw_m', 16),
            ef_construct=self.config.get('hnsw_ef_construct', 100),
            full_scan_threshold=self.config.get('hnsw_full_scan_threshold', 10000)
        )

        # Optimizer configuration
        optimizer_config = OptimizersConfig(
            deleted_threshold=0.2,
            vacuum_min_vector_number=1000,
            default_segment_number=2,
            max_segment_size=None,
            memmap_threshold=None,
            indexing_threshold=self.config.get('indexing_threshold', 20000),
            flush_interval_sec=5,
            max_optimization_threads=None
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vector_config,
            optimizers_config=optimizer_config,
            hnsw_config=hnsw_config
        )

    def _setup_indexes(self):
        """Setup payload indexes for efficient filtering"""
        index_configs = self.config.get('indexes', [])

        # Default indexes if not configured
        if not index_configs:
            index_configs = [
                {'field': 'document_id', 'type': 'keyword'},
                {'field': 'document_fingerprint', 'type': 'keyword'},
                {'field': 'job_id', 'type': 'keyword'},
                {'field': 'primary_framework', 'type': 'keyword'},
                {'field': 'content_domains', 'type': 'keyword'},
                {'field': 'document_type', 'type': 'keyword'},
                {'field': 'tenant_id', 'type': 'keyword'},
                {'field': 'extraction_timestamp', 'type': 'integer'}
            ]

        for index_config in index_configs:
            try:
                field_name = index_config['field']
                field_type = getattr(PayloadSchemaType, index_config['type'].upper())

                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
                logger.debug(f"Created index for field: {field_name}")

            except Exception as e:
                # Index might already exist
                logger.debug(f"Index creation for {index_config['field']} skipped: {e}")

    async def save_extraction_result(
        self,
        extraction_result: ExtractionResult,
        content: str,
        embedding: List[float],
        metadata: DocumentMetadata
    ) -> str:
        """
        Save extraction result to Qdrant

        Args:
            extraction_result: The extraction result
            content: Document content
            embedding: Document embedding vector
            metadata: Document metadata

        Returns:
            Point ID of stored document
        """
        try:
            # Validate metadata
            validation_issues = MetadataValidator.validate_metadata(metadata)
            if validation_issues:
                logger.warning(
                    "Metadata validation issues",
                    document_id=metadata.document_id,
                    issues=validation_issues
                )

            # Generate point ID
            point_id = str(uuid.uuid4())

            # Prepare payload
            payload = self._build_payload(extraction_result, content, metadata)

            # Create point
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )

            # Upsert to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            logger.info(
                "Document saved to Qdrant",
                point_id=point_id,
                document_id=metadata.document_id,
                document_type=metadata.document_type,
                entities_count=len(extraction_result.entities)
            )

            return point_id

        except Exception as e:
            logger.error(
                "Failed to save extraction result",
                document_id=extraction_result.document_id,
                error=str(e)
            )
            raise

    def _build_payload(
        self,
        extraction_result: ExtractionResult,
        content: str,
        metadata: DocumentMetadata
    ) -> Dict[str, Any]:
        """Build Qdrant payload from extraction result and metadata"""
        # Start with metadata
        payload = metadata.to_dict()

        # Add extraction-specific fields
        payload.update({
            'content': content,
            'extraction_success': extraction_result.success,
            'extraction_confidence': extraction_result.confidence,
            'extraction_method': extraction_result.extraction_method.value,
            'extraction_processing_time_ms': extraction_result.processing_time_ms,
            'extraction_warnings': extraction_result.warnings,
            'storage_timestamp': time.time()
        })

        # Add entity information
        if extraction_result.entities:
            entities_data = []
            for entity in extraction_result.entities:
                entities_data.append({
                    'name': entity.name,
                    'title': entity.title,
                    'department': entity.department,
                    'authority_level': entity.authority_level,
                    'confidence': entity.confidence,
                    'entity_type': entity.entity_type
                })

            payload['extracted_entities'] = entities_data
            payload['entity_count'] = len(entities_data)

            # Create searchable entity fields
            payload['entity_names'] = [e.name for e in extraction_result.entities if e.name]
            payload['entity_titles'] = [e.title for e in extraction_result.entities if e.title]

        else:
            payload['entity_count'] = 0

        return payload

    async def find_by_fingerprint(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        """Find document by fingerprint"""
        try:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_fingerprint",
                        match=MatchValue(value=fingerprint)
                    )
                ]
            )

            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=search_filter,
                limit=1,
                with_payload=True,
                with_vectors=False
            )

            if results[0]:  # results is tuple (points, next_page_offset)
                point = results[0][0]  # First point from first page
                return {
                    'point_id': point.id,
                    'payload': point.payload
                }

            return None

        except Exception as e:
            logger.error(f"Failed to find document by fingerprint: {e}")
            return None

    async def find_by_document_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Find document by document ID"""
        try:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )

            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=search_filter,
                limit=1,
                with_payload=True,
                with_vectors=False
            )

            if results[0]:
                point = results[0][0]
                return {
                    'point_id': point.id,
                    'payload': point.payload
                }

            return None

        except Exception as e:
            logger.error(f"Failed to find document by ID: {e}")
            return None

    async def search_with_filters(
        self,
        query_vector: List[float],
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search documents with vector similarity and filters

        Args:
            query_vector: Query embedding vector
            filters: Optional filters dictionary
            limit: Maximum results to return
            score_threshold: Minimum similarity score

        Returns:
            List of matching documents
        """
        try:
            # Build search filter
            search_filter = self._build_search_filter(filters) if filters else None

            # Search parameters
            search_params = SearchParams(
                hnsw_ef=self.config.get('search_hnsw_ef', 128),
                exact=limit < self.config.get('exact_search_threshold', 1000)
            )

            # Perform search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit,
                score_threshold=score_threshold,
                search_params=search_params,
                with_payload=True,
                with_vectors=False
            )

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'point_id': result.id,
                    'score': result.score,
                    'payload': result.payload
                })

            logger.debug(
                "Search completed",
                results_count=len(formatted_results),
                filters=filters,
                score_threshold=score_threshold
            )

            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _build_search_filter(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from filters dictionary"""
        must_conditions = []

        # Document isolation filters (highest priority)
        if 'document_ids' in filters:
            document_ids = filters['document_ids']
            if isinstance(document_ids, list):
                must_conditions.append(
                    FieldCondition(key="document_id", match=MatchAny(any=document_ids))
                )
            else:
                must_conditions.append(
                    FieldCondition(key="document_id", match=MatchValue(value=document_ids))
                )

        # Tenant isolation
        if 'tenant_id' in filters:
            must_conditions.append(
                FieldCondition(key="tenant_id", match=MatchValue(value=filters['tenant_id']))
            )

        # Job isolation
        if 'job_id' in filters:
            must_conditions.append(
                FieldCondition(key="job_id", match=MatchValue(value=filters['job_id']))
            )

        # Framework filtering
        if 'primary_framework' in filters:
            frameworks = filters['primary_framework']
            if isinstance(frameworks, list):
                must_conditions.append(
                    FieldCondition(key="primary_framework", match=MatchAny(any=frameworks))
                )
            else:
                must_conditions.append(
                    FieldCondition(key="primary_framework", match=MatchValue(value=frameworks))
                )

        # Content domain filtering
        if 'content_domains' in filters:
            domains = filters['content_domains']
            if isinstance(domains, list):
                must_conditions.append(
                    FieldCondition(key="content_domains", match=MatchAny(any=domains))
                )
            else:
                must_conditions.append(
                    FieldCondition(key="content_domains", match=MatchValue(value=domains))
                )

        # Document type filtering
        if 'document_type' in filters:
            must_conditions.append(
                FieldCondition(key="document_type", match=MatchValue(value=filters['document_type']))
            )

        return Filter(must=must_conditions) if must_conditions else None

    async def delete_by_document_id(self, document_id: str) -> bool:
        """Delete document by ID"""
        try:
            # Find the point first
            document = await self.find_by_document_id(document_id)
            if not document:
                logger.warning(f"Document not found for deletion: {document_id}")
                return False

            # Delete the point
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[document['point_id']]
            )

            logger.info(f"Document deleted: {document_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    async def update_metadata(self, document_id: str, metadata_updates: Dict[str, Any]) -> bool:
        """Update document metadata"""
        try:
            # Find existing document
            document = await self.find_by_document_id(document_id)
            if not document:
                return False

            # Update payload
            updated_payload = document['payload'].copy()
            updated_payload.update(metadata_updates)
            updated_payload['last_updated'] = time.time()

            # Update in Qdrant
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=metadata_updates,
                points=[document['point_id']]
            )

            logger.info(f"Metadata updated for document: {document_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update metadata for {document_id}: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information and statistics"""
        try:
            collection_info = self.client.get_collection(self.collection_name)

            return {
                'collection_name': self.collection_name,
                'points_count': collection_info.points_count,
                'segments_count': collection_info.segments_count,
                'vector_size': collection_info.config.params.vectors.size,
                'distance_metric': collection_info.config.params.vectors.distance.name,
                'status': collection_info.status,
                'optimizer_status': collection_info.optimizer_status,
                'indexed_vectors_count': collection_info.indexed_vectors_count
            }

        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {'error': str(e)}

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on repository"""
        try:
            # Test basic operations
            info = self.get_collection_info()

            # Test search capability
            test_vector = [0.1] * self.config.get('vector_size', 1536)
            test_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=test_vector,
                limit=1
            )

            return {
                'status': 'healthy',
                'collection_info': info,
                'search_functional': len(test_results) >= 0  # Search worked
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }