"""
RAG Pipeline using LlamaIndex - MVP Core Implementation
Following SRS v2.0 specifications for weeks 3-4
"""

import os
import yaml
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from datetime import datetime
import asyncio

# Load environment variables early to ensure API keys are available
from dotenv import load_dotenv
load_dotenv()
import uuid
import re
import json
import numpy as np
from collections import defaultdict
# LlamaIndex imports - current version structure  
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    Settings,
    StorageContext
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.indices.base_retriever import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from openai import AsyncOpenAI

# Vector stores - Qdrant only (Pinecone removed for simplicity)

# Qdrant
try:
    from qdrant_client import QdrantClient
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantVectorStore = None
    QdrantClient = None

import redis
import structlog
from .relevance_engine import RelevanceEngine
from .conversation_memory import ConversationMemoryManager
from .extraction.document_processing_adapter import DocumentProcessingAdapter

# Import modular components
from .rag_modules.core.response import RAGResponse
from .rag_modules.analysis.citations import CitationAnalyzer
from .rag_modules.cache.manager import CacheManager
from .rag_modules.formatting import FormattingManager, create_formatting_manager
from .rag_modules.query import QueryAnalyzer, create_query_analyzer
from .rag_modules.query.insurance_taxonomy_extractor import InsuranceTaxonomyExtractor
from .rag_modules.quality import QualityMetricsManager, create_quality_metrics_manager
from .rag_modules.services import DocumentRetriever, create_document_retriever
from .rag_modules.services import ResponseGenerator, create_response_generator
from .rag_modules.services import SemanticSimilarityService, create_semantic_similarity_service
from .rag_modules.search import create_search_engine
from .rag_modules.storage import DocumentManager, create_document_manager
from .providers.config_loader import get_config, get_nested

# Phase 2-4 modules
from .providers import get_llm_provider
from .retrieval.intent_cache import IntentCache
from .retrieval.element_aware_retriever import ElementAwareRetriever
from .ingestion.chunk_schema import StructuredStore
from .generation.context_assembler import ContextAssembler
from .generation.verification import VerificationPipeline
from .generation.compliance_check import ComplianceChecker
from .generation.citation_builder import CitationBuilder

logger = structlog.get_logger()

class RAGPipeline:
    def __init__(self, config_path: str = "config/mvp_config.yaml"):
        """Initialize RAG pipeline with LlamaIndex and Qdrant"""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize OpenAI components
        # Allow environment variable override for model
        model_name = os.getenv("OPENAI_MODEL", self.config['llm_config']['model'])
        
        # Debug confirmed this llama-index version supports gpt-4o-mini directly
        llama_index_model = model_name
        logger.info(f"🎯 Using model directly: {model_name}")
        
        # Initialize OpenAI LLM with version compatibility
        try:
            self.llm = OpenAI(
                model=llama_index_model,
                temperature=self.config['llm_config']['temperature'],
                max_tokens=self.config['llm_config']['max_tokens']
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI with model parameter: {e}")
            # Try older initialization pattern
            try:
                self.llm = OpenAI(
                    temperature=self.config['llm_config']['temperature'],
                    max_tokens=self.config['llm_config']['max_tokens']
                )
                # Set model after initialization if possible
                if hasattr(self.llm, 'model'):
                    self.llm.model = llama_index_model
            except Exception as e2:
                logger.error(f"Failed to initialize OpenAI with fallback: {e2}")
                raise e2
        
        # Store the actual model name for API calls
        self.actual_model = model_name
        
        # Verify the model field is set correctly (no override needed since we use direct initialization)
        if hasattr(self.llm, 'model'):
            logger.info(f"✅ Model field confirmed: {self.llm.model}")
        else:
            logger.warning("OpenAI object has no model field")
        
        logger.info(f"✅ OpenAI LLM initialized successfully with model: {self.llm.model if hasattr(self.llm, 'model') else 'unknown'}")
        
        logger.info(f"Using OpenAI model: {model_name}")
        
        from .providers import get_embedding_provider
        self._embedding_provider = get_embedding_provider()
        self.embedding_model = self._embedding_provider.get_llama_index_embedding()

        # Initialize AsyncOpenAI client for parallel processing
        self.async_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        logger.info("✅ AsyncOpenAI client initialized for parallel processing")

        # Configure global settings (replaces ServiceContext in 0.10.x)
        Settings.llm = self.llm
        Settings.embed_model = self.embedding_model
        
        # Keep reference for backward compatibility
        self.service_context = None
        
        # Initialize node parser with simple chunking (hierarchical not available in 0.9.x)
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.config['chunking_strategies']['default']['chunk_size'],
            chunk_overlap=self.config['chunking_strategies']['default']['overlap']
        )
        
        # Initialize vector store: Qdrant primary, local fallback
        self.vector_store_type = None
        self.index_name = None
        _infra = get_config("infrastructure")
        self.collection_name = get_nested(_infra, "qdrant", "collection_name", default="aaire-documents")

        # Try Qdrant first - just test the connection, don't init indexes yet
        if self._try_qdrant():
            self.vector_store_type = "qdrant"
            self.index_name = self.collection_name
            logger.info("Using Qdrant vector store")
        # Fall back to local storage if Qdrant unavailable - defer init
        else:
            self.vector_store_type = "local"
            self.index_name = "local"
            logger.info("Using local vector store")
        
        # Initialize Redis for caching
        self._init_cache()
        
        # Initialize hybrid search components
        self._init_hybrid_search()
        
        # Initialize advanced relevance engine
        self.relevance_engine = RelevanceEngine()
        
        # Initialize conversation memory manager
        memory_config = self.config.get('memory_config', {})
        self.memory_manager = ConversationMemoryManager(
            redis_client=self.cache,
            config=memory_config
        )

        # Initialize new extraction system for document processing
        self.metadata_analyzer = DocumentProcessingAdapter(
            qdrant_client=self.qdrant_client if hasattr(self, 'qdrant_client') else None,
            llm_client=self.async_client  # Use AsyncOpenAI client instead of LlamaIndex wrapper
        )

        # Initialize modular components
        self.citation_analyzer = CitationAnalyzer()
        self.cache_manager = CacheManager(self.cache)

        # Initialize complete insurance taxonomy extractor with XBRL + ACORD + document extraction
        self.taxonomy_extractor = InsuranceTaxonomyExtractor(
            llm_client=self.async_client,
            min_term_frequency=2
        )
        self.taxonomy = {}  # Will be populated after documents are loaded
        logger.info("✅ Complete taxonomy extractor initialized (XBRL + ACORD + document extraction)")

        # Initialize new extracted modules
        self.formatting_manager = create_formatting_manager(llm_client=self.llm)
        self.query_analyzer = create_query_analyzer(llm=self.llm, taxonomy_extractor=self.taxonomy_extractor)
        self.quality_metrics_manager = create_quality_metrics_manager(self.config.get('retrieval_config', {}))

        # Initialize semantic similarity service for enhanced retrieval
        self.semantic_similarity_service = create_semantic_similarity_service()
        logger.info("✅ Semantic similarity service initialized for query-agnostic disambiguation")

        # Initialize Phase 3 services modules (index will be set later)
        self.document_retriever = create_document_retriever(
            vector_index=None,  # Will be set after index creation
            bm25_engine=self.bm25_engine,
            relevance_engine=self.relevance_engine,
            metadata_analyzer=self.metadata_analyzer,
            quality_metrics_manager=self.quality_metrics_manager,
            config=self.config
        )

        self.response_generator = create_response_generator(
            llm_client=self.llm,
            async_client=self.async_client,
            memory_manager=self.memory_manager,
            formatting_manager=self.formatting_manager,
            query_analyzer=self.query_analyzer,
            config=self.config
        )


        # Initialize Phase 2-4 modules
        self._llm_provider = get_llm_provider()
        self.intent_cache = IntentCache()
        self.structured_store = StructuredStore()
        self.element_aware_retriever = ElementAwareRetriever(
            document_retriever=self.document_retriever,
            structured_store=self.structured_store,
        )
        self.context_assembler = ContextAssembler()
        self.verification_pipeline = VerificationPipeline(
            llm_provider=self._llm_provider,
        )
        self.compliance_checker = ComplianceChecker(
            llm_provider=self._llm_provider,
        )
        self.citation_builder = CitationBuilder()

        # Initialize entity extraction for disambiguation
        try:
            from .extraction.entity_extractor import EntityExtractor
            self.entity_extractor = EntityExtractor(es_engine=self.bm25_engine)
            self.element_aware_retriever._entity_extractor = self.entity_extractor
            logger.info("Entity extractor initialized for retrieval disambiguation")
        except Exception as e:
            self.entity_extractor = None
            logger.warning("Entity extractor initialization failed (non-fatal)", error=str(e))

        # Initialize knowledge graph for entity disambiguation
        self.graph_store = None
        try:
            from .knowledge_graph.graph_store import GraphStore
            es_client = getattr(self.bm25_engine, 'client', None)
            if es_client:
                self.graph_store = GraphStore(
                    es_client=es_client,
                    embedding_provider=self.semantic_similarity_service,
                )
                self.element_aware_retriever._graph_store = self.graph_store
                logger.info("Knowledge graph initialized for entity disambiguation")
            else:
                logger.info("No ES client available, knowledge graph disabled")
        except Exception as e:
            logger.warning("Knowledge graph initialization failed (non-fatal)", error=str(e))

        logger.info("Phase 2-4 modules initialized (element-aware retrieval, entity extraction, knowledge graph, context assembly, verification, compliance, citations)")

        # Initialize document manager (will create the index)
        self.document_manager = create_document_manager(
            index=None,  # Will be created by document manager
            node_parser=self.node_parser,
            metadata_analyzer=self.metadata_analyzer,
            bm25_engine=self.bm25_engine,
            cache=self.cache,
            vector_store_type=self.vector_store_type,
            qdrant_client=self.qdrant_client if hasattr(self, 'qdrant_client') else None,
            collection_name=self.collection_name if hasattr(self, 'collection_name') else None
        )

        # Now perform deferred index initialization and get reference
        if self.vector_store_type == "qdrant":
            self.index = self.document_manager._init_qdrant_indexes()
        else:
            self.index = self.document_manager._init_local_index()

        # Update document retriever with the created index
        self.document_retriever.index = self.index

        # Ensure entity payload indexes exist in Qdrant (idempotent)
        if self.vector_store_type == "qdrant" and hasattr(self, 'qdrant_client'):
            try:
                from qdrant_client.models import PayloadSchemaType
                for entity_field in ["entities", "entity_orgs", "entity_persons", "document_title",
                                     "doc_content_hash", "content_hash"]:
                    self.qdrant_client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=entity_field,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                logger.info("Payload indexes ensured (including dedup fields)")
            except Exception as e:
                logger.debug("Entity index creation skipped (may already exist)", error=str(e))

        logger.info("RAG Pipeline initialized",
                   model=self.config['llm_config']['model'],
                   embedding_model=self.config['embedding_config']['model'],
                   memory_enabled=self.cache is not None,
                   smart_filtering=self.metadata_analyzer.smart_filtering_enabled,
                   structured_response_enabled=True)
    
    def _try_qdrant(self) -> bool:
        """Try to initialize Qdrant vector store"""
        if not QDRANT_AVAILABLE:
            logger.warning("Qdrant client not available")
            return False
            
        try:
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            
            logger.info(f"Attempting Qdrant initialization with URL: {qdrant_url}")
            
            if not qdrant_url:
                logger.info("QDRANT_URL not set, skipping Qdrant")
                return False
                
            # Initialize Qdrant client
            logger.info("Creating Qdrant client...")
            self.qdrant_client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key
            )
            
            # Test connection
            logger.info("Testing Qdrant connection...")
            collections = self.qdrant_client.get_collections()
            logger.info("✅ Connected to Qdrant successfully")
            
            # Ensure collection exists (name set in __init__ from config)
            existing = [c.name for c in collections.collections]
            if self.collection_name not in existing:
                from qdrant_client.models import Distance, VectorParams
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self._embedding_provider.dimension, distance=Distance.COSINE),
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")

            logger.info(f"Initializing QdrantVectorStore with collection: {self.collection_name}")
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name
            )
            
            logger.info("✅ Qdrant connection established")
            return True
            
        except Exception as e:
            logger.error("❌ Qdrant initialization failed", error=str(e), exc_info=True)
            return False
    
    def _init_cache(self):
        """Initialize Redis cache"""
        try:
            self.cache = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=0,
                decode_responses=True
            )
            # Test connection
            self.cache.ping()
            logger.info("Redis cache initialized")
        except Exception as e:
            logger.info("Redis cache not available, continuing without cache", error=str(e)[:50])
            self.cache = None
    
    def _init_hybrid_search(self):
        """Initialize keyword search for hybrid retrieval (Elasticsearch or BM25 fallback)"""
        try:
            self.bm25_engine = create_search_engine()
            self.keyword_search_ready = False
            logger.info("Search engine initialized", engine=type(self.bm25_engine).__name__)

            # Start sync in background thread
            import threading
            self.backfill_thread = threading.Thread(
                target=self._sync_search_index,
                daemon=True
            )
            self.backfill_thread.start()
            logger.info("Search index sync started in background")

        except Exception as e:
            logger.error("Failed to initialize search engine", error=str(e))
            self.bm25_engine = None
            self.keyword_search_ready = False

    def _sync_search_index(self):
        """Sync search index with Qdrant documents. Skips if already in sync (Elasticsearch persistence)."""
        try:
            if not hasattr(self, 'qdrant_client') or not self.qdrant_client or not self.bm25_engine or not hasattr(self, 'collection_name'):
                logger.info("Qdrant or search engine not available, skipping sync")
                return

            # If engine supports doc_count (Elasticsearch), check if already synced.
            if hasattr(self.bm25_engine, 'doc_count'):
                es_count = self.bm25_engine.doc_count()
                try:
                    qdrant_info = self.qdrant_client.get_collection(self.collection_name)
                    qdrant_count = qdrant_info.points_count or 0
                except Exception:
                    qdrant_count = 0

                if es_count > 0 and abs(es_count - qdrant_count) <= 5:
                    logger.info(
                        "Search index already synced, skipping backfill",
                        es_docs=es_count,
                        qdrant_points=qdrant_count,
                    )
                    self.keyword_search_ready = True
                    self._build_taxonomy_from_documents()
                    return

            logger.info("Starting search index sync from Qdrant documents...")

            batch_size = 50
            documents_processed = 0
            offset = None

            while True:
                try:
                    response = self.qdrant_client.scroll(
                        collection_name=self.collection_name,
                        limit=batch_size,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )

                    points = response[0]
                    if not points:
                        break

                    batch_docs = []
                    for point in points:
                        payload = point.payload
                        text_content = (
                            payload.get('display_text') or
                            payload.get('text') or
                            payload.get('content') or
                            str(payload)
                        )
                        doc_name = (
                            payload.get('document_title') or
                            payload.get('filename') or
                            'Unknown'
                        )

                        if text_content and len(text_content.strip()) > 10:
                            batch_docs.append({
                                'doc_id': str(point.id),
                                'content': text_content,
                                'title': doc_name,
                                'metadata': {
                                    'point_id': str(point.id),
                                    'title': doc_name,
                                    'filename': doc_name,
                                    'doc_type': payload.get('doc_type', 'company'),
                                    'added_at': payload.get('added_at', ''),
                                    'page': payload.get('page', 0),
                                    'primary_framework': payload.get('primary_framework', payload.get('jurisdiction', 'unknown')),
                                    'content_domains': payload.get('content_domains', []),
                                    'document_type': payload.get('document_type', payload.get('element_type', 'unknown')),
                                    'file_path': doc_name,
                                    'confidence_score': payload.get('confidence_score', 0.5),
                                    **payload
                                }
                            })

                    if batch_docs:
                        self.bm25_engine.add_documents(batch_docs)
                        documents_processed += len(batch_docs)
                        logger.info(f"Indexed {documents_processed} documents in search engine...")

                    offset = response[1]
                    if len(points) < batch_size:
                        break

                except Exception as batch_error:
                    logger.error(f"Error processing search sync batch: {str(batch_error)}")
                    break

            logger.info(f"Search index sync completed: {documents_processed} documents indexed")
            self.keyword_search_ready = True
            self._build_taxonomy_from_documents()

        except Exception as e:
            logger.error(f"Search index sync failed: {str(e)}")
            import traceback
            logger.error(f"Full error trace: {traceback.format_exc()}")
            self.keyword_search_ready = True

    def _calculate_document_hash(self, documents: List[Dict]) -> str:
        """
        Calculate a hash of the document set for taxonomy cache invalidation.

        Args:
            documents: List of document dicts with metadata

        Returns:
            MD5 hash of sorted filenames and doc IDs
        """
        import hashlib

        # Create a stable representation of the document set
        doc_identifiers = []
        for doc in documents:
            # Use filename and doc_id as unique identifiers
            filename = doc.get('metadata', {}).get('filename', 'unknown')
            doc_id = doc.get('metadata', {}).get('doc_id', '')
            doc_identifiers.append(f"{filename}:{doc_id}")

        # Sort for consistent hashing
        doc_identifiers.sort()

        # Create hash
        content = '|'.join(doc_identifiers).encode('utf-8')
        return hashlib.md5(content).hexdigest()

    def _build_taxonomy_from_documents(self):
        """Build taxonomy from existing documents for query enhancement with cache invalidation."""
        try:
            logger.info("🔍 Building domain taxonomy from documents...")

            # If no Qdrant client, can't extract from documents
            if not hasattr(self, 'qdrant_client') or not self.qdrant_client:
                logger.warning("Qdrant client not available, skipping taxonomy extraction")
                return

            # Fetch all documents from Qdrant to calculate current hash
            documents = []
            offset = None
            batch_size = 100

            while len(documents) < 200:  # Limit to first 200 docs for taxonomy extraction
                try:
                    response = self.qdrant_client.scroll(
                        collection_name=self.collection_name,
                        limit=batch_size,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False
                    )

                    points = response[0]
                    if not points:
                        break

                    for point in points:
                        payload = point.payload
                        text_content = (
                            payload.get('text') or
                            payload.get('content') or
                            payload.get('_node_content') or
                            ''
                        )

                        if text_content and len(text_content.strip()) > 50:
                            documents.append({
                                'content': text_content,
                                'metadata': {
                                    'filename': payload.get('filename', 'Unknown'),
                                    'doc_id': str(point.id)
                                }
                            })

                    offset = response[1]
                    if len(points) < batch_size:
                        break

                except Exception as e:
                    logger.error(f"Error fetching documents for taxonomy: {e}")
                    break

            if not documents:
                logger.warning("No documents found for taxonomy extraction")
                return

            # Calculate hash of current document set
            current_doc_hash = self._calculate_document_hash(documents)

            # Try to load existing taxonomy and check if it's still valid
            taxonomy_path = "data/taxonomy.json"
            taxonomy_valid = False

            if self.taxonomy_extractor.load_taxonomy(taxonomy_path):
                # Check if taxonomy has metadata with document hash
                cached_taxonomy = self.taxonomy_extractor.acronyms  # Access loaded data

                # Load the raw taxonomy file to check metadata
                try:
                    import json
                    with open(taxonomy_path, 'r') as f:
                        taxonomy_data = json.load(f)

                    cached_doc_hash = taxonomy_data.get('metadata', {}).get('document_hash', '')

                    if cached_doc_hash == current_doc_hash:
                        logger.info(f"✅ Loaded existing taxonomy from {taxonomy_path} (hash match: {current_doc_hash[:8]})")
                        taxonomy_valid = True
                    else:
                        logger.info(f"🔄 Document set changed (old: {cached_doc_hash[:8]}, new: {current_doc_hash[:8]}), rebuilding taxonomy...")
                        taxonomy_valid = False
                except Exception as e:
                    logger.warning(f"Could not verify taxonomy hash: {e}, rebuilding...")
                    taxonomy_valid = False

            # If taxonomy is valid, we're done
            if taxonomy_valid:
                return

            # Build new taxonomy
            logger.info(f"📄 Extracting taxonomy from {len(documents)} documents...")

            # Build taxonomy using pattern extraction
            self.taxonomy = self.taxonomy_extractor.build_taxonomy(documents)

            # Add document hash to taxonomy metadata
            if 'metadata' not in self.taxonomy:
                self.taxonomy['metadata'] = {}
            self.taxonomy['metadata']['document_hash'] = current_doc_hash

            # Save taxonomy for future use (pass the taxonomy dict to preserve metadata)
            self.taxonomy_extractor.save_taxonomy(taxonomy_path, self.taxonomy)

            logger.info(
                f"✅ Taxonomy built and cached (hash: {current_doc_hash[:8]}): "
                f"{len(self.taxonomy.get('acronyms', {}))} acronyms, "
                f"{len(self.taxonomy.get('hierarchies', {}))} hierarchies, "
                f"{len(self.taxonomy.get('synonyms', {}))} synonym groups"
            )

        except Exception as e:
            logger.error(f"Failed to build taxonomy: {e}")
            import traceback
            logger.error(f"Taxonomy error trace: {traceback.format_exc()}")

    async def add_documents(self, documents: List[Document], doc_type: str = "company"):
        """Add documents using the document manager"""
        return await self.document_manager.add_documents(documents, doc_type)
    
    async def process_query(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> RAGResponse:
        """
        Process a user query through the RAG pipeline with conversation memory
        """
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Record user message in conversation memory
        if self.memory_manager:
            await self.memory_manager.add_message(session_id, 'user', query)
        
        try:
            # Check cache first (but skip cache for debugging if needed)
            cache_key = self.cache_manager.get_cache_key(query, filters, self.vector_store, self.index_name)
            use_cache = (self.cache and 
                        self.config['retrieval_config']['use_cache'] and
                        not os.getenv('DISABLE_CACHE', '').lower() in ('true', '1', 'yes'))
            
            if use_cache:
                cached_response = self.cache.get(cache_key)
                if cached_response:
                    logger.info("Returning cached response", query_hash=cache_key[:8])
                    return self.cache_manager.deserialize_response(cached_response, session_id)
            
            # Check if query is within AAIRE's domain expertise
            logger.info(f"🔍 Classifying query topic: '{query[:50]}...'")
            topic_check = await self.query_analyzer.classify_query_topic(query)
            logger.info(f"🎯 Topic classification result: {topic_check}")
            
            if not topic_check['is_relevant']:
                logger.info(f"❌ Query rejected as off-topic: '{query[:50]}...'")
                return RAGResponse(
                    answer=topic_check['polite_response'],
                    citations=[],
                    confidence=1.0,  # High confidence in polite rejection
                    session_id=session_id,
                    follow_up_questions=[]
                )
            
            # Determine document type filter
            doc_type_filter = self._get_doc_type_filter(filters)
            
            # Intelligent semantic query enhancement for better concept retrieval
            taxonomy_terms = []
            try:
                enhancement_result = await self.query_analyzer.enhance_query_semantically(query)
                expanded_query = enhancement_result['enhanced_query']
                # Extract taxonomy terms for LLM guidance
                if 'enhancements' in enhancement_result and 'taxonomy_expansion' in enhancement_result['enhancements']:
                    taxonomy_terms = enhancement_result['enhancements']['taxonomy_expansion']
                    logger.info(f"🎯 Extracted {len(taxonomy_terms)} taxonomy terms for LLM guidance: {taxonomy_terms}")
                else:
                    logger.warning(f"⚠️ No taxonomy terms found in enhancement_result")
                logger.info(f"🚀 Query semantically enhanced: {enhancement_result['enhancement_count']} concepts added")
            except Exception as e:
                logger.warning(f"Semantic enhancement failed, using basic expansion: {e}")
                expanded_query = self.query_analyzer.expand_query(query)
            
            # Get adaptive similarity threshold
            similarity_threshold = self.quality_metrics_manager.get_similarity_threshold(query)

            # --- Phase 3: Intent cache fast-path ---
            cached_intent = self.intent_cache.classify_fast(expanded_query)
            if cached_intent:
                logger.info("Intent fast-path matched", query_type=cached_intent.query_type, jurisdiction=cached_intent.jurisdiction)

            # --- Phase 3: Element-aware retrieval with parallel search + early exit ---
            enriched_results = await self.element_aware_retriever.retrieve(
                query=expanded_query,
                doc_type_filter=doc_type_filter,
                similarity_threshold=similarity_threshold,
                filters=filters,
                fetch_originals=True,
            )

            # --- Phase 3: Cross-encoder reranking with entity-aware composite scoring ---
            _query_entities = None
            if self.entity_extractor:
                try:
                    _query_entities = self.entity_extractor.extract_from_query(query)
                except Exception:
                    pass
            if enriched_results and self.semantic_similarity_service:
                enriched_results = self.semantic_similarity_service.rerank_enriched_results(
                    query, enriched_results, query_entities=_query_entities
                )

            # Convert reranked results to dict format for backward-compatible consumers
            retrieved_docs = [self._enriched_to_dict(er) for er in enriched_results]

            # Apply diversity selection to spread context across source documents
            retrieved_docs = self.document_retriever.get_diverse_context_documents(retrieved_docs)

            # Check if we found relevant documents after reranking
            if retrieved_docs:
                logger.info(f"Found {len(retrieved_docs)} relevant documents for query: '{query[:50]}...'")

                doc_sources = [(doc['metadata'].get('document_title', 'Unknown'),
                              doc.get('rerank_score', doc.get('score', 0)))
                             for doc in retrieved_docs[:5]]
                logger.info(f"Top document sources with scores: {doc_sources}")

                # --- Phase 4: Assemble rich context ---
                assembled = self.context_assembler.assemble(retrieved_docs, query)

                # --- Phase 4.1: Inject knowledge graph entity context ---
                graph_context = ""
                resolved_nodes = getattr(self.element_aware_retriever, '_last_resolved_nodes', [])
                if self.graph_store and resolved_nodes:
                    try:
                        from .providers.config_loader import get_config as _get_config, get_nested as _get_nested
                        kg_cfg = _get_config("knowledge_graph")
                        if _get_nested(kg_cfg, "retrieval", "inject_entity_context", default=True):
                            graph_context = self.graph_store.get_entity_context(
                                [n.entity_id for n in resolved_nodes]
                            )
                            if graph_context:
                                assembled.text = graph_context + "\n\n" + assembled.text
                                logger.info("Graph entity context injected into prompt",
                                            resolved_entities=len(resolved_nodes))
                    except Exception as e:
                        logger.debug("Graph context injection failed (non-fatal)", error=str(e))

                # --- Phase 4.2: Emit retrieval audit trail ---
                try:
                    from .knowledge_graph.audit import RetrievalAudit
                    graph_chunk_ids = getattr(self.element_aware_retriever, '_last_graph_chunk_ids', [])
                    _audit_query_entities = []
                    if _query_entities and hasattr(_query_entities, 'all_entities'):
                        _audit_query_entities = _query_entities.all_entities
                    audit = RetrievalAudit(
                        query=query,
                        query_entities=_audit_query_entities,
                        resolved_entities=[n.entity_id for n in resolved_nodes],
                        graph_connected_chunks=graph_chunk_ids,
                        hybrid_search_chunks=[er.node_id for er in enriched_results[:20]],
                        final_ranked_chunks=[er.node_id for er in enriched_results[:10]],
                        rerank_scores={er.node_id: er.rerank_score for er in enriched_results[:10]
                                       if er.rerank_score is not None},
                        graph_context_injected=bool(graph_context),
                    )
                    audit.log()
                except Exception:
                    pass  # Audit must never block the pipeline

                # --- Phase 4: Generate with verification pipeline (2-3 LLM calls max) ---
                conversation_context = ""
                if conversation_history:
                    recent = conversation_history[-6:]  # Last 3 exchanges
                    conversation_context = "\n".join(
                        f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in recent
                    )

                verification_result = await self.verification_pipeline.run(
                    query=query,
                    context_text=assembled.text,
                    conversation_history=conversation_context,
                )
                response = verification_result.response
                logger.info(
                    "Verification complete",
                    verified=verification_result.verified,
                    llm_calls=verification_result.llm_calls,
                    correction_applied=verification_result.correction_applied,
                )

                # --- Phase 4: Post-generation compliance check ---
                compliance_result = await self.compliance_checker.check(response)
                response = compliance_result.response
                if compliance_result.disclaimer_added:
                    logger.info("Compliance disclaimer added", issues=compliance_result.issues)

                # --- Phase 4: Inline citation extraction with refusal guard ---
                if self._detect_refusal(response):
                    logger.info("Refusal detected — suppressing citations")
                    citations = []
                    confidence = 0.1
                else:
                    built_citations = self.citation_builder.extract_inline_citations(
                        response, assembled.source_map
                    )
                    citations = [c.to_dict() for c in built_citations]
                    confidence = self.quality_metrics_manager.calculate_confidence(retrieved_docs, response)
            else:
                # No relevant documents found - check if this could be relevant general knowledge
                is_general_query = self.query_analyzer.is_general_knowledge_query(query)
                
                # Even if it's a general query, it must still be within AAIRE's domain
                if is_general_query:
                    # Re-check topic relevance for general knowledge questions
                    topic_check = await self.query_analyzer.classify_query_topic(query)
                    if not topic_check['is_relevant']:
                        return RAGResponse(
                            answer=topic_check['polite_response'],
                            citations=[],
                            confidence=1.0,
                            session_id=session_id,
                            follow_up_questions=[]
                        )
                
                if is_general_query:
                    # Use general knowledge response
                    logger.info(f"No relevant documents found, using general knowledge for: '{query[:50]}...'")
                    response = await self.response_generator.generate_response(query, [], user_context, conversation_history, session_id)


                    response = self.formatting_manager.format_response(response)
                    response = self.citation_analyzer.remove_citations_from_response(response)
                    citations = []
                    confidence = 0.3  # Low confidence for general knowledge responses
                else:
                    # Specific query but no documents found - provide detailed feedback
                    logger.warning(f"No relevant documents found for specific query: '{query[:50]}...'")
                    
                    # Check what documents we do have available
                    available_docs = []
                    try:
                        if hasattr(self, 'vector_store') and self.vector_store:
                            # Try to get some info about available documents
                            sample_docs = await self.document_retriever.vector_search("document", None, 0.1)  # Very low threshold
                            available_docs = list(set([doc['metadata'].get('filename', 'Unknown') for doc in sample_docs[:5]]))
                    except:
                        pass
                    
                    if available_docs:
                        response = f"I couldn't find specific information about '{query}' in the uploaded documents. The available documents include: {', '.join(available_docs)}. Please verify that the document containing this information has been successfully uploaded and processed."
                    else:
                        response = f"I couldn't find specific information about '{query}' in the uploaded documents. Please ensure the relevant document has been uploaded and processed successfully."
                    
                    citations = []
                    confidence = 0.1  # Very low confidence when we can't find specific content
            
            # Response formatting handled by prompt engineering
            
            # Generate contextual follow-up questions
            follow_up_questions = await self.response_generator.generate_follow_up_questions(query, response, retrieved_docs)
            
            # Calculate quality metrics
            quality_metrics = self.quality_metrics_manager.calculate_quality_metrics(query, response, retrieved_docs, citations)
            
            rag_response = RAGResponse(
                answer=response,
                citations=citations,
                confidence=confidence,
                session_id=session_id,
                follow_up_questions=follow_up_questions,
                quality_metrics=quality_metrics
            )
            
            # Cache the response
            if self.cache:
                self.cache.setex(
                    cache_key, 
                    self.config['retrieval_config']['cache_ttl'],
                    self.cache_manager.serialize_response(rag_response)
                )
            
            # Record assistant response in conversation memory
            if self.memory_manager:
                await self.memory_manager.add_message(session_id, 'assistant', response)
            
            return rag_response

        except Exception as e:
            logger.error("Failed to process query", error=str(e), query=query[:100])
            raise

    async def process_query_streaming(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None
    ):
        """
        Process a user query with streaming response — uses the same
        Phase 3-4 pipeline as process_query() (element-aware retrieval,
        context assembly, verification, compliance, precise citations).

        Yields: response chunks, then final metadata.
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        if self.memory_manager:
            await self.memory_manager.add_message(session_id, 'user', query)

        try:
            logger.info("Streaming query processing started", query=query, session_id=session_id)

            # --- Topic gate ---
            topic_result = await self.query_analyzer.classify_query_topic(query)
            if not topic_result['is_relevant']:
                yield {"type": "content", "content": topic_result['polite_response']}
                yield {"type": "done", "session_id": session_id, "citations": [], "confidence": 1.0, "follow_up_questions": []}
                return

            # --- Semantic enhancement ---
            doc_type_filter = self._get_doc_type_filter(filters)
            try:
                enhancement_result = await self.query_analyzer.enhance_query_semantically(query)
                expanded_query = enhancement_result['enhanced_query']
            except Exception as e:
                logger.warning(f"Semantic enhancement failed: {e}")
                expanded_query = self.query_analyzer.expand_query(query)

            similarity_threshold = self.quality_metrics_manager.get_similarity_threshold(query)

            # --- Phase 3: Intent cache fast-path ---
            cached_intent = self.intent_cache.classify_fast(expanded_query)
            if cached_intent:
                logger.info("Intent fast-path matched", query_type=cached_intent.query_type)

            # --- Phase 3: Element-aware retrieval ---
            enriched_results = await self.element_aware_retriever.retrieve(
                query=expanded_query,
                doc_type_filter=doc_type_filter,
                similarity_threshold=similarity_threshold,
                filters=filters,
                fetch_originals=True,
            )

            # --- Phase 3: Cross-encoder reranking with entity-aware composite scoring ---
            _query_entities_stream = None
            if self.entity_extractor:
                try:
                    _query_entities_stream = self.entity_extractor.extract_from_query(query)
                except Exception:
                    pass
            if enriched_results and self.semantic_similarity_service:
                enriched_results = self.semantic_similarity_service.rerank_enriched_results(
                    query, enriched_results, query_entities=_query_entities_stream
                )

            # Convert reranked results to dict format
            retrieved_docs = [self._enriched_to_dict(er) for er in enriched_results]

            # Apply diversity selection to spread context across source documents
            retrieved_docs = self.document_retriever.get_diverse_context_documents(retrieved_docs)

            if retrieved_docs:
                logger.info(f"Found {len(retrieved_docs)} relevant documents")

                # --- Phase 4: Assemble context ---
                assembled = self.context_assembler.assemble(retrieved_docs, query)

                # --- Phase 4: Verification pipeline (generate + verify + correct) ---
                conversation_context = ""
                if conversation_history:
                    recent = conversation_history[-6:]
                    conversation_context = "\n".join(
                        f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in recent
                    )

                verification_result = await self.verification_pipeline.run(
                    query=query,
                    context_text=assembled.text,
                    conversation_history=conversation_context,
                )
                response = verification_result.response

                # --- Phase 4: Compliance check ---
                compliance_result = await self.compliance_checker.check(response)
                response = compliance_result.response

                # Stream the response progressively (word-level chunks)
                words = response.split(' ')
                buffer = []
                for word in words:
                    buffer.append(word)
                    if len(buffer) >= 5:
                        yield {"type": "content", "content": ' '.join(buffer) + ' '}
                        buffer = []
                        await asyncio.sleep(0.03)
                if buffer:
                    yield {"type": "content", "content": ' '.join(buffer)}

                # --- Phase 4: Inline citation extraction with refusal guard ---
                if self._detect_refusal(response):
                    logger.info("Refusal detected — suppressing citations")
                    citations = []
                    confidence = 0.1
                else:
                    built_citations = self.citation_builder.extract_inline_citations(
                        response, assembled.source_map
                    )
                    citations = [c.to_dict() for c in built_citations]
                    confidence = self.quality_metrics_manager.calculate_confidence(retrieved_docs, response)
                follow_up_questions = await self.response_generator.generate_follow_up_questions(query, response, retrieved_docs)

                if self.memory_manager:
                    await self.memory_manager.add_message(session_id, 'assistant', response)

                yield {
                    "type": "done",
                    "session_id": session_id,
                    "citations": citations,
                    "confidence": confidence,
                    "follow_up_questions": follow_up_questions,
                }
            else:
                # No documents found
                is_general_query = self.query_analyzer.is_general_knowledge_query(query)
                if is_general_query:
                    response_stream = await self.response_generator.generate_response(
                        query, [], user_context, conversation_history, session_id, stream=True
                    )
                    full_response = ""
                    async for chunk in response_stream:
                        full_response += chunk
                        yield {"type": "content", "content": chunk}

                    if self.memory_manager:
                        await self.memory_manager.add_message(session_id, 'assistant', full_response)
                    yield {"type": "done", "session_id": session_id, "citations": [], "confidence": 0.3, "follow_up_questions": []}
                else:
                    error_msg = f"I couldn't find specific information about '{query}' in the uploaded documents."
                    yield {"type": "content", "content": error_msg}
                    yield {"type": "done", "session_id": session_id, "citations": [], "confidence": 0.1, "follow_up_questions": []}

        except Exception as e:
            logger.error("Failed to process streaming query", error=str(e), query=query[:100])
            yield {"type": "error", "message": "I apologize, but I encountered an error processing your request."}
            raise

    @staticmethod
    def _enriched_to_dict(er) -> Dict[str, Any]:
        """Convert an EnrichedResult to a backward-compatible dict."""
        return {
            'content': er.original_content or er.content,
            'metadata': er.metadata,
            'score': er.score,
            'relevance_score': er.score,
            'rerank_score': er.rerank_score,
            'node_id': er.node_id,
            'search_type': er.search_type,
        }

    @staticmethod
    def _detect_refusal(response: str) -> bool:
        """Detect if the LLM response is a refusal / 'I can't find it' answer.

        If the LLM refused to answer, we suppress all citations to avoid
        the contradiction of 'I have no info' + 5 source citations.
        """
        if not response:
            return True
        response_lower = response.lower()
        refusal_patterns = [
            "i couldn't find",
            "i could not find",
            "could not find specific",
            "couldn't find specific",
            "does not contain",
            "doesn't contain",
            "no relevant information",
            "no specific information",
            "insufficient information",
            "not enough information",
            "unable to answer",
            "cannot answer",
            "i'm sorry, but the provided context",
            "i apologize, but the provided context",
            "no information available",
            "not available in the provided",
        ]
        return any(p in response_lower for p in refusal_patterns)

    def _get_doc_type_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[List[str]]:
        """Get document types to filter by based on filters"""
        if not filters or not filters.get('source_type'):
            return None  # No filter, search all document types
        
        source_types = filters['source_type']
        if isinstance(source_types, str):
            source_types = [source_types]
        
        doc_types = []
        for source_type in source_types:
            if source_type == "US_GAAP":
                doc_types.append("us_gaap")
            elif source_type == "IFRS":
                doc_types.append("ifrs")
            elif source_type == "COMPANY":
                doc_types.append("company")
            elif source_type == "ACTUARIAL":
                doc_types.append("actuarial")
        
        return doc_types if doc_types else None
    
    async def delete_document(self, job_id: str) -> Dict[str, Any]:
        """Delete all chunks associated with a document using the document manager"""
        return await self.document_manager.delete_document(job_id)
    
    
    async def cleanup_orphaned_chunks(self) -> Dict[str, Any]:
        """Clean up chunks that don't have valid job_ids (legacy data)"""
        try:
            cleaned_count = 0
            
            if self.vector_store_type == "qdrant":
                # Get all points without job_id
                from qdrant_client.models import Filter, FieldCondition, IsNullCondition
                
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
            logger.error(f"Failed to cleanup orphaned chunks", error=str(e))
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
                            "text_preview": point.payload.get("text", "")[:100] + "..." if point.payload.get("text") else ""
                        })
                        
            return {
                "status": "success",
                "total_documents": len(documents),
                "documents": documents,
                "vector_store": self.vector_store_type
            }
            
        except Exception as e:
            logger.error(f"Failed to get all documents", error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def clear_all_cache(self) -> Dict[str, Any]:
        """Clear all cached responses"""
        try:
            cleared_count = 0
            if self.cache:
                # Clear all cache entries
                pattern = "*"
                keys = list(self.cache.scan_iter(match=pattern))
                if keys:
                    cleared_count = self.cache.delete(*keys)
                logger.info(f"Cleared {cleared_count} cache entries")
            
            return {
                "status": "success",
                "cleared_entries": cleared_count
            }
        except Exception as e:
            logger.error("Failed to clear cache", error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get RAG pipeline statistics"""
        stats = {
            "index": {},
            "cache_stats": {},
            "configuration": {
                "model": self.config['llm_config']['model'],
                "embedding_model": self.config['embedding_config']['model'],
                "similarity_threshold": self.config['retrieval_config']['similarity_threshold'],
                "index_name": self.index_name
            }
        }
        
        # Get single index statistics
        try:
            stats["index"] = {
                "name": self.index_name,
                "status": "active",
                "last_updated": datetime.utcnow().isoformat(),
                "note": "Single index with document type metadata filtering"
            }
        except Exception as e:
            stats["index"] = {
                "name": self.index_name,
                "status": "error",
                "error": str(e)
            }
        
        # Get cache statistics
        if self.cache:
            try:
                cache_info = self.cache.info()
                stats["cache_stats"] = {
                    "connected_clients": cache_info.get("connected_clients", 0),
                    "used_memory": cache_info.get("used_memory_human", "0B"),
                    "hits": cache_info.get("keyspace_hits", 0),
                    "misses": cache_info.get("keyspace_misses", 0)
                }
            except:
                stats["cache_stats"] = {"status": "unavailable"}
        
        return stats
    
    async def clear_all_documents(self) -> Dict[str, Any]:
        """Clear all documents from Qdrant database"""
        try:
            if not hasattr(self, 'qdrant_client') or not self.qdrant_client:
                return {
                    "status": "error",
                    "message": "Qdrant client not initialized"
                }

            # Get current document count before clearing
            doc_count_before = 0
            try:
                search_result = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    limit=1000
                )
                doc_count_before = len(search_result[0])
            except Exception as e:
                logger.warning(f"Could not get document count before clearing: {e}")

            # Delete the collection and recreate it
            logger.info(f"🗑️ Clearing all documents from Qdrant collection: {self.collection_name}")

            # Delete collection
            self.qdrant_client.delete_collection(self.collection_name)
            logger.info(f"✅ Deleted collection: {self.collection_name}")

            # Recreate collection with same configuration
            from qdrant_client.models import Distance, VectorParams
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self._embedding_provider.dimension,  # OpenAI embedding dimension
                    distance=Distance.COSINE
                )
            )
            logger.info(f"✅ Recreated empty collection: {self.collection_name}")

            # Recreate payload indexes for filtering
            try:
                from qdrant_client.models import PayloadSchemaType
                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="filename",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="doc_type",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="job_id",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                # Entity + dedup payload indexes
                for entity_field in ["entities", "entity_orgs", "entity_persons",
                                     "doc_content_hash", "content_hash"]:
                    self.qdrant_client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=entity_field,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                logger.info("Recreated payload indexes (including dedup fields)")
            except Exception as e:
                logger.warning(f"Could not recreate payload indexes: {e}")

            # Clear BM25 search index as well
            if hasattr(self, 'document_manager') and self.document_manager:
                self.document_manager._clear_bm25_index()

            # Clear cache
            await self.clear_all_cache()

            logger.info(f"🎯 Successfully cleared all documents from Qdrant database")

            return {
                "status": "success",
                "message": f"Successfully cleared all documents from {self.collection_name}",
                "documents_cleared": doc_count_before,
                "collection_name": self.collection_name,
                "bm25_index_cleared": True,
                "cache_cleared": True
            }

        except Exception as e:
            logger.error(f"❌ Failed to clear Qdrant documents: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to clear documents: {str(e)}"
            }
