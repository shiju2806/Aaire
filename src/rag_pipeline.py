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
from src.enhanced_whoosh_engine import EnhancedWhooshEngine, EnhancedSearchResult
from src.intelligent_query_analyzer import IntelligentQueryAnalyzer, QueryIntent, JurisdictionHint, ProductHint

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
from src.relevance_engine import RelevanceEngine
from extraction.bridge_adapter import IntelligentDocumentExtractor
from enhanced_query_handler import EnhancedQueryHandler
from conversation_memory import ConversationMemoryManager
from extraction.document_processing_adapter import DocumentProcessingAdapter
from extraction.models import QueryIntent, LegacyDocumentMetadata as DocumentMetadata

# Import modular components
from rag_modules.core.response import RAGResponse
from rag_modules.analysis.citations import CitationAnalyzer
from rag_modules.cache.manager import CacheManager
from rag_modules.formatting import FormattingManager, create_formatting_manager
from rag_modules.query import QueryAnalyzer, create_query_analyzer
# Quality services now handled via dependency injection
from rag_modules.services import DocumentRetriever, create_document_retriever
from rag_modules.services import ResponseGenerator, create_response_generator
from rag_modules.storage import DocumentManager, create_document_manager

# Import enhanced modules
from rag_modules.enhanced_pipeline import EnhancedRAGPipeline

logger = structlog.get_logger()

class RAGPipeline:
    def __init__(self, config_path: str = "config/mvp_config.yaml"):
        """Initialize RAG pipeline with LlamaIndex and Qdrant"""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize collection_name early to prevent AttributeError
        self.collection_name = "aaire-documents"
        
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
        
        self.embedding_model = OpenAIEmbedding(
            model=self.config['embedding_config']['model']
        )

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

        # Initialize new extracted modules using dependency injection
        from rag_modules.core.dependency_injection import get_container
        container = get_container()
        self.formatting_manager = container.get_singleton('formatting_manager')
        self.query_analyzer = create_query_analyzer(llm=self.llm)

        # Initialize intelligent query analyzer for jurisdiction/product awareness
        self.intelligent_query_analyzer = IntelligentQueryAnalyzer()

        # Initialize quality services via dependency injection
        self.quality_metrics_service = container.get_singleton('quality_metrics_service')
        self.validation_service = container.get_singleton('validation_service')

        # Initialize Phase 3 services modules (index will be set later)
        self.document_retriever = create_document_retriever(
            vector_index=None,  # Will be set after index creation
            whoosh_engine=self.whoosh_engine,
            relevance_engine=self.relevance_engine,
            metadata_analyzer=self.metadata_analyzer,
            quality_metrics_manager=self.quality_metrics_service,
            config=self.config,
            llm_client=self.async_client  # Pass LLM client for framework detection
        )

        self.response_generator = create_response_generator(
            llm_client=self.llm,
            async_client=self.async_client,
            memory_manager=self.memory_manager,
            formatting_manager=self.formatting_manager,
            query_analyzer=self.query_analyzer,
            config=self.config
        )

        # Initialize document manager (will create the index)
        self.document_manager = create_document_manager(
            index=None,  # Will be created by document manager
            node_parser=self.node_parser,
            metadata_analyzer=self.metadata_analyzer,
            whoosh_engine=self.whoosh_engine,
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

        # Critical fix: Update document manager's index reference
        self.document_manager.index = self.index

        # Update document retriever with the created index
        self.document_retriever.index = self.index

        # Initialize Enhanced RAG Pipeline for advanced features
        try:
            self.enhanced_pipeline = EnhancedRAGPipeline(
                llm_client=self.async_client,
                config_dir="/Users/shijuprakash/AAIRE/config"
            )
            self.enhanced_features_enabled = True
            logger.info("✅ Enhanced RAG Pipeline initialized with advanced features")
        except Exception as e:
            logger.warning(f"Enhanced RAG Pipeline initialization failed, using standard features: {e}")
            self.enhanced_pipeline = None
            self.enhanced_features_enabled = False

        logger.info("RAG Pipeline initialized",
                   model=self.config['llm_config']['model'],
                   embedding_model=self.config['embedding_config']['model'],
                   memory_enabled=self.cache is not None,
                   smart_filtering=self.metadata_analyzer.smart_filtering_enabled,
                   structured_response_enabled=True,
                   enhanced_features_enabled=self.enhanced_features_enabled)
    
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
            
            # Initialize Qdrant vector store
            self.collection_name = "aaire-documents"
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
        """Initialize Whoosh keyword search for hybrid retrieval"""
        try:
            # Initialize Enhanced Whoosh search engine with jurisdiction/product awareness
            from pathlib import Path
            self.whoosh_engine = EnhancedWhooshEngine(
                index_dir=Path("enhanced_search_index")
            )
            self.keyword_search_ready = False
            logger.info("✅ Whoosh search engine initialized")

            # Start Whoosh backfill in background thread to avoid startup delays
            import threading
            self.backfill_thread = threading.Thread(
                target=self._populate_whoosh_from_existing_documents,
                daemon=False  # Changed from True to ensure proper completion
            )
            self.backfill_thread.start()
            logger.info("🚀 Whoosh backfill started in background")

        except Exception as e:
            logger.error("Failed to initialize Whoosh search", error=str(e))
            # Set fallback values
            self.whoosh_engine = None
            self.keyword_search_ready = False

    def _populate_whoosh_from_existing_documents(self):
        """Populate Whoosh index with existing documents from Qdrant on startup (optimized)"""
        try:
            # Only proceed if we have Qdrant and Whoosh available
            if not hasattr(self, 'qdrant_client') or not self.qdrant_client or not self.whoosh_engine:
                logger.info("🔄 Qdrant or Whoosh not available, skipping Whoosh backfill")
                return

            logger.info("🔄 Starting Whoosh backfill from existing Qdrant documents...")

            # Use smaller batches for better performance
            batch_size = 50
            documents_processed = 0
            offset = None

            # Process documents in batches
            while True:
                try:
                    # Fetch batch of documents
                    response = self.qdrant_client.scroll(
                        collection_name=self.collection_name,
                        limit=batch_size,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False  # We only need the text content
                    )

                    points = response[0]
                    if not points:
                        break

                    # Prepare documents for Whoosh indexing
                    batch_docs = []

                    for point in points:
                        payload = point.payload

                        # Extract text content with proper JSON parsing for _node_content
                        def extract_text_content(payload):
                            import json
                            # Try direct text fields first
                            if payload.get('text'):
                                return payload.get('text')
                            if payload.get('content'):
                                return payload.get('content')

                            # Handle _node_content JSON field
                            if payload.get('_node_content'):
                                try:
                                    if isinstance(payload['_node_content'], str):
                                        node_data = json.loads(payload['_node_content'])
                                        return node_data.get('text', '')
                                    elif isinstance(payload['_node_content'], dict):
                                        return payload['_node_content'].get('text', '')
                                except (json.JSONDecodeError, AttributeError):
                                    pass

                            return str(payload)

                        text_content = extract_text_content(payload)

                        if text_content and len(text_content.strip()) > 10:  # Only meaningful content
                            logger.debug(f"🔍 WHOOSH DEBUG: Adding document {str(point.id)[:8]}... with {len(text_content)} chars")
                            # Convert to Whoosh document format
                            whoosh_doc = {
                                'doc_id': str(point.id),
                                'content': text_content,
                                'title': payload.get('filename', 'Unknown'),
                                'metadata': {
                                    'point_id': str(point.id),
                                    'filename': payload.get('filename', 'Unknown'),
                                    'doc_type': payload.get('doc_type', 'company'),
                                    'added_at': payload.get('added_at', ''),
                                    'page': payload.get('page', 0),
                                    'primary_framework': payload.get('primary_framework', 'unknown'),
                                    'content_domains': payload.get('content_domains', []),
                                    'document_type': payload.get('document_type', 'unknown'),
                                    'file_path': payload.get('filename', 'Unknown'),
                                    'confidence_score': payload.get('confidence_score', 0.5),
                                    # Include all existing metadata for smart filtering
                                    **payload
                                }
                            }
                            batch_docs.append(whoosh_doc)

                    # Index batch in Whoosh
                    if batch_docs:
                        logger.debug(f"🔍 WHOOSH DEBUG: Attempting to index {len(batch_docs)} documents")
                        indexed_count = self.whoosh_engine.add_documents(batch_docs, batch_processing=True)
                        documents_processed += indexed_count
                        logger.info(f"📄 Indexed {documents_processed} documents in Whoosh...")
                        logger.debug(f"🔍 WHOOSH DEBUG: Total docs in index now: {self.whoosh_engine.get_document_count()}")

                    # Update offset for next batch
                    offset = response[1]

                    if len(points) < batch_size:
                        break

                except Exception as batch_error:
                    logger.error(f"Error processing Whoosh batch: {str(batch_error)}")
                    break

            # Force final commit to ensure all documents are written to disk
            if hasattr(self.whoosh_engine, 'index') and self.whoosh_engine.index:
                try:
                    with self.whoosh_engine.index.writer() as writer:
                        writer.commit()
                    logger.info("✅ Final Whoosh index commit successful")
                except Exception as commit_error:
                    logger.error(f"❌ Final commit failed: {commit_error}")

            logger.info(f"📚 Whoosh backfill completed: {documents_processed} documents indexed")
            logger.info("🎯 Hybrid search (vector + keyword) now available for ALL documents")
            self.keyword_search_ready = True

        except Exception as e:
            logger.error(f"❌ Whoosh backfill failed: {str(e)}")
            # Don't crash the system, just log the error
            import traceback
            logger.error(f"Full error trace: {traceback.format_exc()}")
            self.keyword_search_ready = True  # Mark as ready even if failed
    
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
            
            # Expand query for better retrieval
            expanded_query = self.query_analyzer.expand_query(query)
            
            # Get adaptive similarity threshold
            similarity_threshold = self.quality_metrics_service.get_similarity_threshold(query)
            
            # Store current query for citation filtering
            self._current_query = query
            
            # ALWAYS search uploaded documents first
            retrieved_docs = await self.document_retriever.retrieve_documents(expanded_query, doc_type_filter, similarity_threshold, filters)
            
            # Check if we found relevant documents in uploaded content
            if retrieved_docs and len(retrieved_docs) > 0:
                # Found relevant documents - use them for response
                logger.info(f"Found {len(retrieved_docs)} relevant documents for query: '{query[:50]}...'")
                
                # Log document sources for transparency
                doc_sources = [(doc['metadata'].get('filename', 'Unknown'), 
                              doc.get('relevance_score', doc.get('score', 0))) 
                             for doc in retrieved_docs[:5]]
                logger.info(f"Top document sources with scores: {doc_sources}")
                
                # Check for potential document confusion issues
                unique_sources = set([source[0] for source in doc_sources])
                if len(unique_sources) > 1:
                    logger.info(f"Multiple document sources found for query, applying strict citation filters")
                
                response = await self.response_generator.generate_response(query, retrieved_docs, user_context, conversation_history, session_id)


                # Pass 2: Apply unified intelligent formatting (single LLM call)
                response = await self.formatting_manager.apply_unified_intelligent_formatting(response, query, retrieved_docs)

                # Check if this is an "I don't know" response - if so, don't generate citations
                if self.citation_analyzer.is_insufficient_information_response(response):
                    logger.info("🚫 Skipping citation generation for insufficient information response")
                    citations = []
                else:
                    citations = self.citation_analyzer.extract_citations(retrieved_docs, query, response)

                confidence = self.quality_metrics_service.calculate_confidence(retrieved_docs, response)
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


                    response = await self.formatting_manager.apply_unified_intelligent_formatting(response, query, [])
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
            quality_metrics = self.quality_metrics_service.calculate_quality_metrics(query, response, retrieved_docs, citations)

            # Apply intelligent validation to the response
            try:
                validation_result = await self.validation_service.validate_response(
                    query=query,
                    response=response,
                    retrieved_docs=retrieved_docs,
                    confidence=confidence
                )

                # If validation fails, use rejection message instead
                if not validation_result.passed:
                    logger.warning(f"Response failed intelligent validation: {validation_result.rejection_reason}")
                    response = f"I don't have sufficient information to provide a reliable answer to your question. {validation_result.rejection_reason}"
                    citations = []
                    confidence = 0.1
                    # Update quality metrics for the rejection
                    quality_metrics = self.quality_metrics_service.calculate_quality_metrics(query, response, [], [])

                # Add validation results to quality metrics
                quality_metrics['intelligent_validation'] = {
                    'passed': validation_result.passed,
                    'overall_score': validation_result.overall_score,
                    'processing_time_ms': validation_result.processing_time_ms,
                    'components': validation_result.components
                }

            except Exception as e:
                logger.error(f"Intelligent validation failed: {e}")
                # Continue with original response if validation fails

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

    async def process_query_enhanced(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
        query_intent: Optional[Any] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query using Enhanced RAG Pipeline with advanced features

        Args:
            query: User query
            filters: Optional document filters
            user_context: Additional context for processing
            session_id: Session identifier
            conversation_history: Previous conversation context
            options: Processing options like max_iterations, use_reasoning

        Returns:
            Enhanced response with advanced retrieval and self-correction
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        if not self.enhanced_features_enabled or not self.enhanced_pipeline:
            logger.warning("Enhanced features not available, falling back to standard processing")
            standard_response = await self.process_query(query, filters, user_context, session_id, conversation_history)
            return {
                "response": standard_response.answer,
                "documents": [{"content": c.content, "metadata": c.metadata} for c in standard_response.citations] if standard_response.citations else [],
                "reasoning_chain": None,
                "verification_result": None,
                "metadata": {
                    "processing_mode": "standard_fallback",
                    "enhanced_features": {"advanced_retrieval": False, "self_correction": False}
                }
            }

        try:
            logger.info(f"🚀 Processing query with enhanced features: '{query[:50]}...'")

            # Define base retrieval function that uses our existing system
            async def base_retrieval_func(search_query: str) -> List[Dict]:
                """Wrapper for existing retrieval system"""
                # Get adaptive similarity threshold
                similarity_threshold = self.quality_metrics_service.get_similarity_threshold(search_query)

                # Determine document type filter
                doc_type_filter = self._get_doc_type_filter(filters)

                # Expand query for better retrieval
                expanded_query = self.query_analyzer.expand_query(search_query)

                # Retrieve documents using existing system with query intent
                retrieved_docs = await self.document_retriever.retrieve_documents(
                    expanded_query, doc_type_filter, similarity_threshold, filters, query_intent
                )

                # Convert to format expected by enhanced pipeline
                formatted_docs = []
                for doc in retrieved_docs:
                    formatted_docs.append({
                        'content': doc.get('content', ''),
                        'metadata': doc.get('metadata', {}),
                        'score': doc.get('relevance_score', doc.get('score', 0.0))
                    })

                return formatted_docs

            # Define base generation function that uses our existing system
            async def base_generation_func(gen_query: str, context: str) -> str:
                """Wrapper for existing generation system"""
                # Convert context back to document format for existing system
                context_docs = [{'content': context, 'metadata': {'source': 'enhanced_context'}}]

                # Use existing response generator
                response = await self.response_generator.generate_response(
                    gen_query, context_docs, user_context, conversation_history, session_id
                )

                return response

            # Process with enhanced pipeline
            enhanced_result = await self.enhanced_pipeline.enhanced_rag_query(
                query=query,
                base_retrieval_func=base_retrieval_func,
                base_generation_func=base_generation_func,
                document_store=None,  # Could integrate document store if needed
                context=user_context,
                options=options or {}
            )

            # Record conversation history if enabled
            if self.memory_manager:
                await self.memory_manager.add_message(session_id, 'user', query)
                await self.memory_manager.add_message(session_id, 'assistant', enhanced_result['response'])

            logger.info("✅ Enhanced query processing completed successfully",
                       retrieval_strategy=enhanced_result['metadata'].get('retrieval_strategy'),
                       correction_applied=enhanced_result['metadata'].get('correction_applied'),
                       processing_time_ms=enhanced_result['metadata'].get('processing_time_ms'))

            return enhanced_result

        except Exception as e:
            logger.error(f"Enhanced query processing failed: {e}")
            logger.info("Falling back to standard processing")

            # Fallback to standard processing
            standard_response = await self.process_query(query, filters, user_context, session_id, conversation_history)
            return {
                "response": standard_response.answer,
                "documents": [{"content": c.content, "metadata": c.metadata} for c in standard_response.citations] if standard_response.citations else [],
                "reasoning_chain": None,
                "verification_result": None,
                "metadata": {
                    "processing_mode": "standard_fallback",
                    "error": str(e),
                    "enhanced_features": {"advanced_retrieval": False, "self_correction": False}
                }
            }

    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get status of enhanced features"""
        if not self.enhanced_features_enabled or not self.enhanced_pipeline:
            return {
                "enabled": False,
                "reason": "Enhanced pipeline not initialized",
                "features": {"advanced_retrieval": False, "self_correction": False}
            }

        return {
            "enabled": True,
            "features": self.enhanced_pipeline.get_status(),
            "configuration": self.enhanced_pipeline.get_configuration()
        }

    def configure_enhanced_features(self, module: str, config_updates: Dict[str, Any]) -> bool:
        """Configure enhanced features dynamically"""
        if not self.enhanced_features_enabled or not self.enhanced_pipeline:
            logger.warning("Enhanced features not available for configuration")
            return False

        return self.enhanced_pipeline.update_configuration(module, config_updates)

    async def _generate_extraction_follow_ups(self, query: str, extraction_result) -> List[str]:
        """Generate relevant follow-up questions for extraction results"""
        try:
            if not extraction_result.entities:
                return []
            
            follow_up_prompt = f"""Based on this organizational information extraction, suggest 3 relevant follow-up questions:
            
    Original query: {query}
    Document type: {extraction_result.structure_type}
    People found: {len(extraction_result.entities)}
    
    Create questions that would:
    1. Dive deeper into specific roles or departments
    2. Explore relationships or reporting structures  
    3. Clarify authority levels or responsibilities
    
    Return only the questions, one per line."""
            
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Generate relevant follow-up questions for organizational analysis."},
                    {"role": "user", "content": follow_up_prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            questions = [q.strip() for q in response.choices[0].message.content.strip().split("\n") if q.strip()]
            return questions[:3]  # Limit to 3 questions
            
        except Exception as e:
            logger.error("Follow-up generation failed", error=str(e))
            return []
    

    async def stream_response(self, query: str) -> AsyncGenerator[str, None]:
        """Stream response generation for real-time UI updates"""
        # This is a simplified streaming implementation
        # In production, you'd want proper streaming from the LLM
        response = await self.process_query(query)
        
        # Simulate streaming by yielding chunks
        words = response.answer.split()
        chunk_size = 5
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if i + chunk_size < len(words):
                chunk += " "
            yield chunk
            await asyncio.sleep(0.1)  # Small delay for demo purposes
    
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
    
    def _create_semantic_document_groups(self, documents: List[Dict]) -> List[List[Dict]]:
        """Group documents by semantic similarity without hard-coded topics"""
        try:
            # Use LLM to analyze document themes and create groups
            doc_summaries = []
            for i, doc in enumerate(documents[:15], 1):  # Limit for analysis efficiency
                content_sample = doc['content'][:300].replace('\n', ' ')
                doc_summaries.append(f"Doc {i}: {content_sample}...")
            
            grouping_prompt = f"""Analyze these document excerpts and identify natural groupings based on their content themes.

Documents:
{chr(10).join(doc_summaries)}

Create 2-4 logical groups where documents with similar themes are together.
Output format:
Group 1: Doc 1, Doc 3, Doc 7 (theme: [describe])
Group 2: Doc 2, Doc 5, Doc 9 (theme: [describe])
etc.

Group documents by content similarity:"""
            
            response = self.llm.complete(grouping_prompt)
            grouping_result = response.text.strip()
            
            logger.info(f"📊 SEMANTIC GROUPING RESULT:\n{grouping_result}")
            
            # Parse the grouping response to create actual groups
            groups = self._parse_document_groupings(grouping_result, documents[:15])
            
            # Add remaining documents to smallest groups
            if len(documents) > 15:
                remaining_docs = documents[15:]
                for doc in remaining_docs:
                    smallest_group = min(groups, key=len)
                    smallest_group.append(doc)
            
            logger.info(f"📚 Created {len(groups)} semantic document groups:")
            for i, group in enumerate(groups, 1):
                logger.info(f"  Group {i}: {len(group)} documents")
            return groups
            
        except Exception as e:
            logger.warning(f"Failed to create semantic groups, using simple split: {str(e)}")
            # Fallback: simple split into 3 groups
            group_size = len(documents) // 3
            return [
                documents[:group_size],
                documents[group_size:2*group_size],
                documents[2*group_size:]
            ]
    
    def _parse_document_groupings(self, grouping_text: str, documents: List[Dict]) -> List[List[Dict]]:
        """Parse LLM grouping output into actual document groups"""
        import re
        
        groups = []
        lines = grouping_text.split('\n')
        
        for line in lines:
            # Look for patterns like "Group 1: Doc 1, Doc 3, Doc 7"
            match = re.search(r'Group \d+:.*?Doc (\d+(?:, Doc \d+)*)', line)
            if match:
                doc_nums_str = match.group(1)
                doc_nums = [int(num.strip()) for num in re.findall(r'\d+', doc_nums_str)]
                
                group_docs = []
                for doc_num in doc_nums:
                    if 1 <= doc_num <= len(documents):
                        group_docs.append(documents[doc_num - 1])
                
                if group_docs:
                    groups.append(group_docs)
        
        # Ensure we have at least 2 groups
        if len(groups) < 2:
            mid = len(documents) // 2
            return [documents[:mid], documents[mid:]]
        
        return groups
    
    async def _process_with_chunked_enhancement(self, query: str, retrieved_docs: List[Dict], conversation_context: str) -> str:
        """Main processing method that combines all our enhancements"""
        try:
            # 🔧 APPLY ENHANCED METHODOLOGICAL RANKING
            enhanced_docs = self._apply_methodological_ranking(query, retrieved_docs)

            # Get diverse documents for processing
            diverse_docs = self._get_diverse_context_documents(enhanced_docs)
            logger.info(f"📚 Processing {len(diverse_docs)} diverse documents (out of {len(enhanced_docs)} total)")
            
            # Check for organizational queries first
            if self.query_analyzer.is_organizational_query(query, diverse_docs):
                return self.query_analyzer.generate_organizational_response(query, diverse_docs, conversation_context)
            
            # Use chunked processing for comprehensive coverage
            if len(diverse_docs) <= 8:
                # Small document set - enhanced single pass
                response = self._generate_enhanced_single_pass(query, diverse_docs, conversation_context)
            else:
                # Large document set - semantic chunking
                response = await self._generate_chunked_response(query, diverse_docs, conversation_context)
            
            # Apply enhancements
            # OPTIMIZATION: Disable completeness check for faster response times
            # enhanced_response = self._completeness_check(query, response, diverse_docs)
            logger.info("⚡ Completeness check disabled for speed optimization")

            # Note: Structured response generation is already handled in the individual methods above
            # Removing duplicate call to prevent double processing and API errors

            return response  # CRITICAL FIX: Return the successful response!

        except Exception as e:
            logger.error(f"Enhanced processing failed: {str(e)}")
            # Fallback to simple approach
            return self._generate_enhanced_single_pass(query, diverse_docs[:5], conversation_context)
    
    async def _generate_chunked_response(self, query: str, documents: List[Dict], conversation_context: str) -> str:
        """Generate response using semantic chunking for large document sets"""
        # Delegate to ResponseGenerator for consistent formatting
        if hasattr(self, 'response_generator') and self.response_generator:
            return await self.response_generator.generate_chunked_response(query, documents, conversation_context)

        # Fallback to local implementation if ResponseGenerator not available
        # Create semantic groups
        document_groups = self._create_semantic_document_groups(documents)
        response_parts = []

        # Process all groups in parallel using async for faster response
        async def process_group_async(group_index: int, doc_group: List[Dict]) -> str:
            group_context = "\n\n".join([doc['content'] for doc in doc_group])

            group_prompt = f"""You are answering: {query}

This is document group {group_index} of {len(document_groups)}. Focus on these documents:

{group_context}

FORMATTING REQUIREMENTS (write clean, professional responses):
- Use numbered sections: 1. Section Title (no bold, no markdown)
- Use numbered subsections: 1.1 Subsection Title
- Use bullet points (•) for lists
- Keep formatting clean and readable without markdown symbols

EXAMPLE FORMAT:
1. Main Section Title

Content paragraph with details from documents.

1.1 Subsection Title

• First bullet point
• Second bullet point
• Third bullet point

2. Next Main Section

More content here.

CRITICAL FORMATTING RULES:
- Main headings: 1. Title, 2. Title, 3. Title (clean numbering)
- Sub-headings: 1.1 Title, 1.2 Title (clean sub-numbering)
- Use bullet points (•) not dashes for lists
- NO markdown symbols like ** or ###
- Always double line break between sections
- Write formulas clearly: use simple notation like (A + B)/C
- Keep everything clean and professional

CONTENT REQUIREMENTS:
- Include ALL relevant formulas, calculations, and mathematical expressions
- Preserve specific numerical values like 90%, $2.50 per $1,000, etc.
- Copy EXACT formulas from documents
- Include ALL calculation methods and procedures
- Maintain technical accuracy and detail

Provide a detailed response covering all information that relates to the question using proper markdown formatting."""

            # Use AsyncOpenAI for true parallel processing
            response = await self.async_client.chat.completions.create(
                model=self.actual_model,
                messages=[{"role": "user", "content": group_prompt}],
                temperature=0,
                max_tokens=4000
            )

            logger.info(f"⚡ Processed group {group_index}/{len(document_groups)} (async)")
            return response.choices[0].message.content.strip()

        # Process all groups concurrently using asyncio.gather for true parallelism
        logger.info(f"⚡ Starting parallel processing of {len(document_groups)} groups with AsyncOpenAI")
        response_parts = await asyncio.gather(*[
            process_group_async(i+1, doc_group)
            for i, doc_group in enumerate(document_groups)
        ])
        
        # Use smart continuation approach (clean joining - each part has proper formatting)
        logger.info("📋 Using smart continuation approach for seamless formatting")

        # Filter out empty parts and clean whitespace
        cleaned_parts = [part.strip() for part in response_parts if part and part.strip()]

        # Simple clean joining - each part already has correct numbering from ChatGPT-style prompts
        final_response = "\n\n".join(cleaned_parts)

        # Apply normalization for consistent spacing
        return self.formatting_manager.normalize_spacing(final_response)
    
    
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
                self.document_manager._init_qdrant_indexes()

                # Clear Whoosh index as well
                self._clear_whoosh_index()

                logger.info("Successfully cleared all documents from Qdrant and Whoosh")
                return {"status": "success", "message": "All documents cleared", "method": "qdrant_recreate"}
            else:
                # For local storage, recreate the index
                self.document_manager._init_local_index()

                # Clear Whoosh index as well
                self._clear_whoosh_index()

                logger.info("Successfully cleared all documents from local storage and Whoosh")
                return {"status": "success", "message": "All documents cleared", "method": "local_recreate"}
                
        except Exception as e:
            logger.error("Failed to clear all documents", error=str(e))
            return {"status": "error", "error": str(e)}

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
    
    async def generate_document_summary(self, document_content: str, doc_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intelligent executive summary of uploaded documents"""
        try:
            # Create specialized prompt for document summarization
            summary_prompt = f"""You are AAIRE, a senior AI consultant specializing in insurance accounting, actuarial analysis, and financial compliance. You have deep expertise in US GAAP, IFRS, actuarial standards, and insurance regulations.

**Document Analysis Request:**

**Document Profile:**
- Title: {doc_metadata.get('title', 'Unknown')}
- Type: {doc_metadata.get('source_type', 'Unknown')}
- Effective Date: {doc_metadata.get('effective_date', 'Unknown')}
- Analysis Purpose: Executive Summary for Accounting/Actuarial Review

**Document Content to Analyze:**
{document_content[:6000]}

**Required Analysis Framework:**

**📋 EXECUTIVE SUMMARY**
Provide a 2-3 sentence high-level overview of the document's purpose and significance.

**🔍 KEY ACCOUNTING & ACTUARIAL IMPACTS**
- Identify specific accounting standards referenced (ASC, IFRS, etc.)
- Highlight changes to accounting treatments or methodologies
- Assess impact on financial statement presentation
- Note any actuarial assumption changes or valuation impacts

**⚖️ REGULATORY & COMPLIANCE IMPLICATIONS**
- Identify regulatory requirements and deadlines
- Assess compliance obligations and reporting changes
- Highlight any new disclosure requirements
- Note potential audit or examination impacts

**💰 FINANCIAL IMPACT ANALYSIS**
- Quantify financial impacts where possible
- Identify affected financial statement line items
- Assess materiality and significance
- Highlight cash flow or capital implications

**⚠️ RISK ASSESSMENT & CONTROLS**
- Identify compliance risks and mitigation strategies
- Assess implementation challenges and timeline risks
- Highlight areas requiring additional controls or procedures
- Note potential reputational or regulatory penalties

**🎯 STRATEGIC RECOMMENDATIONS**
Provide 5-7 specific, actionable recommendations including:
- Immediate actions required (with timelines)
- Stakeholder communication needs
- System/process changes required
- Training or resource needs
- Monitoring and ongoing compliance requirements

**📊 KEY METRICS & BENCHMARKS**
Extract and highlight:
- Important dates and deadlines
- Financial figures and thresholds
- Percentage impacts or changes
- Comparative data or benchmarks

**🔗 INTERCONNECTED IMPACTS**
- How this affects other accounting areas
- Integration with existing policies/procedures
- Coordination needs across departments
- Potential conflicts with other standards

Use professional, precise language with specific details. Include relevant accounting citations and technical terms. Focus on actionable insights that enable informed business decisions."""

            # Generate summary using the LLM
            response = self.llm.complete(summary_prompt)
            summary_text = response.text.strip()
            
            # Extract key metrics and insights
            key_insights = self._extract_key_insights(summary_text, document_content)
            
            return {
                "summary": summary_text,
                "key_insights": key_insights,
                "document_metadata": doc_metadata,
                "generated_at": datetime.utcnow().isoformat(),
                "confidence": 0.85  # Base confidence for summarization
            }
            
        except Exception as e:
            logger.error("Failed to generate document summary", error=str(e))
            return {
                "summary": "Unable to generate summary at this time. The document has been processed and indexed for search.",
                "key_insights": [],
                "document_metadata": doc_metadata,
                "generated_at": datetime.utcnow().isoformat(),
                "confidence": 0.0
            }
    
    def _extract_key_insights(self, summary_text: str, document_content: str) -> List[Dict[str, Any]]:
        """Extract key insights and metrics from document analysis"""
        insights = []
        
        # Extract potential accounting standards mentioned
        standards_mentioned = []
        accounting_keywords = ['ASC', 'FASB', 'IFRS', 'GAAP', 'CECL', 'LDTI', 'ASU', 'FAS']
        for keyword in accounting_keywords:
            if keyword in document_content.upper():
                standards_mentioned.append(keyword)
        
        if standards_mentioned:
            insights.append({
                "type": "accounting_standards",
                "value": standards_mentioned,
                "description": "Accounting standards referenced in document"
            })
        
        # Extract dates and deadlines
        import re
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
            r'\b\d{4}-\d{2}-\d{2}\b',      # YYYY-MM-DD
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        ]
        
        dates_found = []
        for pattern in date_patterns:
            dates_found.extend(re.findall(pattern, document_content, re.IGNORECASE))
        
        if dates_found:
            insights.append({
                "type": "important_dates",
                "value": dates_found[:5],  # Limit to first 5 dates
                "description": "Important dates mentioned in document"
            })
        
        # Extract financial terms
        financial_terms = ['reserve', 'liability', 'asset', 'premium', 'claim', 'policyholder', 'actuarial', 'valuation']
        terms_found = [term for term in financial_terms if term in document_content.lower()]
        
        if terms_found:
            insights.append({
                "type": "financial_concepts",
                "value": terms_found,
                "description": "Key financial concepts discussed"
            })
        
        return insights

    async def process_query_with_intelligence(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        session_id: str = "default",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        use_strict_mode: bool = True
    ) -> RAGResponse:
        """
        Enhanced query processing with intelligent extraction capabilities
        Routes queries to appropriate processing method based on content analysis
        """
        try:
            # Import the enhanced components
            from extraction.bridge_adapter import IntelligentDocumentExtractor

            # Initialize components
            query_handler = EnhancedQueryHandler(self.llm)
            intelligent_extractor = IntelligentDocumentExtractor(self.llm)
            
            logger.info("Enhanced query processing started", 
                       query=query[:100], 
                       session_id=session_id)
            
            # Step 1: Intelligent query analysis for jurisdiction/product awareness
            query_intent = self.intelligent_query_analyzer.analyze_query(query)

            logger.info("Query intent analysis completed",
                       jurisdiction=query_intent.jurisdiction_hint.value,
                       product=query_intent.product_hint.value,
                       jurisdiction_confidence=query_intent.jurisdiction_confidence,
                       product_confidence=query_intent.product_confidence,
                       disambiguation_needed=query_intent.disambiguation_needed)

            # Step 2: Analyze query to determine processing strategy
            routing_decision = await query_handler.route_query(query, user_context)
            
            logger.info("Query routing decision made",
                       method=routing_decision['method'],
                       extraction_type=routing_decision['extraction_type'],
                       confidence=routing_decision['confidence'])
            
            # Step 2: Route to appropriate processing method
            if routing_decision['method'] == 'intelligent_extraction':
                return await self._process_with_intelligent_extraction(
                    query, routing_decision, intelligent_extractor,
                    filters, user_context, session_id, conversation_history
                )
            elif routing_decision['method'] == 'enhanced_rag' or await self._should_use_enhanced_rag(query, routing_decision):
                # Use Enhanced RAG Pipeline for complex actuarial queries
                logger.info("Using Enhanced RAG processing",
                           confidence=routing_decision.get('confidence', 0.0),
                           enhanced_features_available=self.enhanced_features_enabled)

                if self.enhanced_features_enabled:
                    enhanced_response = await self.process_query_enhanced(
                        query=query,
                        filters=filters,
                        user_context=user_context,
                        session_id=session_id,
                        conversation_history=conversation_history,
                        query_intent=query_intent,
                        options={}
                    )

                    # Convert enhanced response format to RAGResponse format
                    citations = []
                    if enhanced_response.get('documents'):
                        from models import Citation
                        for doc in enhanced_response['documents']:
                            citations.append(Citation(
                                content=doc.get('content', ''),
                                metadata=doc.get('metadata', {}),
                                score=doc.get('metadata', {}).get('score', 0.0)
                            ))

                    # Create RAGResponse with enhanced metadata
                    from models import RAGResponse
                    return RAGResponse(
                        answer=enhanced_response.get('response', ''),
                        citations=citations,
                        session_id=session_id,
                        confidence=enhanced_response.get('metadata', {}).get('confidence', routing_decision.get('confidence', 0.0)),
                        metadata={
                            'processing_mode': 'enhanced_rag',
                            'routing_confidence': routing_decision.get('confidence', 0.0),
                            'enhanced_features': enhanced_response.get('metadata', {}).get('enhanced_features', {}),
                            'reasoning_chain': enhanced_response.get('reasoning_chain'),
                            'verification_result': enhanced_response.get('verification_result')
                        }
                    )
                else:
                    logger.warning("Enhanced RAG requested but not available, falling back to standard")
                    return await self.process_query(
                        query, filters, user_context, session_id, conversation_history
                    )
            else:
                # Fall back to standard RAG processing
                logger.info("Using standard RAG processing")
                return await self.process_query(
                    query, filters, user_context, session_id, conversation_history
                )
                
        except Exception as e:
            logger.error("Enhanced query processing failed", 
                        error=str(e), 
                        query=query[:50],
                        event="Enhanced query processing failed")
            
            # Fallback to standard processing
            logger.info("Falling back to standard RAG processing")
            return await self.process_query(
                query, filters, user_context, session_id, conversation_history
            )
    
    async def _process_with_intelligent_extraction(
        self,
        query: str,
        routing_decision: Dict[str, Any],
        intelligent_extractor,
        filters: Optional[Dict[str, Any]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        session_id: str = "default",
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> RAGResponse:
        """Process query using intelligent document extraction"""
        
        logger.info("Starting intelligent extraction processing",
                   extraction_type=routing_decision['extraction_type'])
        
        try:
            # Step 1: Retrieve relevant documents using standard RAG
            rag_response = await self.process_query(
                query, filters, user_context, session_id, conversation_history
            )
            
            # Step 2: Apply intelligent extraction to retrieved documents
            extracted_insights = []
            
            if hasattr(rag_response, 'citations') and rag_response.citations:
                for citation in rag_response.citations[:3]:  # Limit to top 3 docs
                    try:
                        # Get document content for extraction
                        doc_content = citation.get('content', '')
                        if doc_content:
                            extraction_result = await intelligent_extractor.process_document(
                                doc_content, query
                            )
                            
                            if extraction_result.confidence_score > 0.5:
                                extracted_insights.append({
                                    'source': citation.get('source', 'Unknown'),
                                    'extraction_data': extraction_result.extracted_data,
                                    'confidence': extraction_result.confidence_score,
                                    'document_type': extraction_result.document_type.value,
                                    'warnings': extraction_result.warnings
                                })
                                
                    except Exception as e:
                        logger.warning(f"Extraction failed for document: {e}")
                        continue
            
            # Step 3: Enhanced response generation with extracted insights
            enhanced_response = await self._generate_enhanced_response(
                query, rag_response, extracted_insights, routing_decision
            )
            
            # Step 4: Update response with enhanced information
            rag_response.response = enhanced_response
            rag_response.follow_up_questions = self._generate_extraction_followups(
                routing_decision['extraction_type'], extracted_insights
            )
            
            # Add extraction metadata
            if not hasattr(rag_response, 'metadata'):
                rag_response.metadata = {}
            
            rag_response.metadata['intelligent_extraction'] = {
                'extraction_type': routing_decision['extraction_type'],
                'insights_count': len(extracted_insights),
                'confidence': routing_decision['confidence'],
                'processing_method': 'enhanced'
            }
            
            logger.info("Intelligent extraction completed successfully",
                       insights_found=len(extracted_insights),
                       extraction_type=routing_decision['extraction_type'])
            
            return rag_response
            
        except Exception as e:
            logger.error("Intelligent extraction processing failed", error=str(e))
            # Return the basic RAG response if enhancement fails
            return await self.process_query(
                query, filters, user_context, session_id, conversation_history
            )
    
    async def _generate_enhanced_response(
        self,
        query: str,
        rag_response: RAGResponse,
        extracted_insights: List[Dict[str, Any]],
        routing_decision: Dict[str, Any]
    ) -> str:
        """Generate enhanced response incorporating intelligent extraction results"""
        
        if not extracted_insights:
            return rag_response.response
        
        # Build enhancement prompt
        insights_summary = []
        for insight in extracted_insights:
            insights_summary.append(f"From {insight['source']}: {insight['extraction_data']}")
        
        enhancement_prompt = f"""
The user asked: "{query}"

Original response:
{rag_response.response}

Additional intelligent extraction results:
{chr(10).join(insights_summary)}

Extraction type: {routing_decision['extraction_type']}

Instructions:
1. Enhance the original response with the specific extracted information
2. For job title queries, provide a clear breakdown of roles and people
3. Include confidence levels where appropriate
4. Highlight any discrepancies or unclear information
5. Keep the response professional and well-structured
6. Focus on accuracy - only include information that was explicitly extracted

Generate an enhanced response that combines the original information with the extracted insights:
"""

        try:
            enhanced_response = self.llm.complete(enhancement_prompt)
            return enhanced_response.text
        except Exception as e:
            logger.error(f"Failed to generate enhanced response: {e}")
            return rag_response.response
    
    def _generate_extraction_followups(
        self, 
        extraction_type: str, 
        extracted_insights: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate follow-up questions based on extraction results"""
        
        followups = []
        
        if extraction_type == 'job_titles':
            followups.extend([
                "Can you provide more details about the reporting structure?",
                "What are the responsibilities for each role?",
                "Are there any vacant positions or recent changes?",
                "How do these roles relate to the overall organizational structure?"
            ])
        elif extraction_type == 'financial_roles':
            followups.extend([
                "What are the specific responsibilities of each financial role?",
                "How is the finance team structured hierarchically?",
                "What approval authorities do these roles have?",
                "Are there any recent changes in the finance organization?"
            ])
        elif extraction_type == 'organizational':
            followups.extend([
                "Can you explain the reporting relationships in more detail?",
                "What departments are represented in this structure?",
                "How does this structure support business operations?",
                "Are there any upcoming organizational changes planned?"
            ])
        
        # Add specific followups based on extracted data
        if extracted_insights:
            insight_count = sum(len(insight.get('extraction_data', {})) for insight in extracted_insights)
            if insight_count > 0:
                followups.append(f"Can you provide more context about the {insight_count} items identified?")
        
        return followups[:3]  # Limit to 3 follow-ups for optimal user experience
    
    
    async def _apply_professional_formatting(self, response: str, query: str) -> str:
        """Apply ChatGPT-style professional formatting to responses"""
        try:
            logger.info("📝 Applying professional formatting to response")
            
            # Use LLM to reformat the content professionally
            system_prompt = """You are a formatting expert. Rewrite the provided technical content with:

1. Clear visual hierarchy using emojis as section markers (🔹 for main points, ✅ for summary, 📌 for key points)
2. Numbered sections with proper spacing
3. Sub-points using (a), (b), (c) or bullet points
4. Bold only for key terms and headers (use sparingly)
5. Clean spacing between sections
6. Professional, conversational tone
7. Examples where helpful
8. A clear summary at the end

Format like high-quality ChatGPT responses - clean, organized, and easy to scan.
Keep all technical accuracy but improve readability dramatically.
Do NOT add unnecessary information - only reformat what's provided."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}\n\nContent to reformat:\n{response}"}
            ]
            
            formatted_response = await self.llm_client.achat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.3,
                max_tokens=4000
            )
            
            result = formatted_response.choices[0].message.content
            
            # Apply final clean-up
            result = self.formatting_manager.final_formatting_cleanup(result)
            
            logger.info("✅ Professional formatting applied successfully")
            return result
            
        except Exception as e:
            logger.warning(f"Could not apply professional formatting: {e}")
            # Fall back to basic cleanup
            return self.formatting_manager.basic_professional_format(response)

    async def _should_use_enhanced_rag(self, query: str, routing_decision: Dict[str, Any]) -> bool:
        """
        Determine if Enhanced RAG Pipeline should be used based on LLM analysis
        Uses AI to understand query complexity without hardcoded rules
        """
        # Check if enhanced features are available
        if not self.enhanced_features_enabled:
            return False

        try:
            # Use LLM to analyze if query needs enhanced features
            prompt = """Analyze this query and determine if it requires enhanced RAG features.

Enhanced RAG should be used for:
1. Complex actuarial/accounting calculations requiring multi-step reasoning
2. Queries about specific regulatory frameworks (US STAT, IFRS, GAAP)
3. Technical questions requiring deep domain expertise
4. Multi-part questions needing query decomposition
5. Questions where standard retrieval might miss nuanced information

Query: {query}

Return JSON with:
{{
    "use_enhanced": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "detected_complexity": ["list", "of", "complexity", "indicators"]
}}""".format(query=query)

            response = await self.async_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            import json
            analysis = json.loads(response.choices[0].message.content)

            # Also check for framework detection using LLM
            framework_detected = False
            try:
                from rag_modules.filtering import LLMFrameworkDetector
                detector = LLMFrameworkDetector(self.async_client)
                framework_detection = await detector.detect_framework(query)
                framework_detected = framework_detection.primary_framework is not None
            except:
                pass

            # Make final decision - be more selective about Enhanced RAG
            should_use_enhanced = (
                analysis.get('use_enhanced', False) and
                analysis.get('confidence', 0.0) > 0.95 and  # Only for very high confidence LLM recommendations
                framework_detected  # AND framework must be detected
                # Removed low confidence fallback to Enhanced RAG (was causing expensive processing)
            )

            logger.info("LLM-based enhanced RAG routing",
                       should_use_enhanced=should_use_enhanced,
                       llm_recommendation=analysis.get('use_enhanced'),
                       confidence=analysis.get('confidence', 0.0),
                       reasoning=analysis.get('reasoning'),
                       complexity_indicators=analysis.get('detected_complexity', []),
                       framework_detected=framework_detected,
                       standard_routing_confidence=routing_decision.get('confidence', 0.0))

            return should_use_enhanced

        except Exception as e:
            logger.warning(f"LLM routing analysis failed: {e}, using fallback")
            # Simple fallback - use enhanced for low confidence queries
            return routing_decision.get('confidence', 0.0) < 0.3

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
                    size=1536,  # OpenAI embedding dimension
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
                logger.info("✅ Recreated payload indexes")
            except Exception as e:
                logger.warning(f"Could not recreate payload indexes: {e}")

            # Clear Whoosh search index as well
            self._clear_whoosh_index()

            # Clear cache
            self.clear_cache()

            logger.info(f"🎯 Successfully cleared all documents from Qdrant database")

            return {
                "status": "success",
                "message": f"Successfully cleared all documents from {self.collection_name}",
                "documents_cleared": doc_count_before,
                "collection_name": self.collection_name,
                "whoosh_index_cleared": True,
                "cache_cleared": True
            }

        except Exception as e:
            logger.error(f"❌ Failed to clear Qdrant documents: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to clear documents: {str(e)}"
            }
