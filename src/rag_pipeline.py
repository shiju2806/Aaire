"""
RAG Pipeline using LlamaIndex - MVP Core Implementation
Following SRS v2.0 specifications for weeks 3-4
"""

import os
import yaml
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
import asyncio
import uuid
import re
import json
import numpy as np
from collections import defaultdict
from rank_bm25 import BM25Okapi

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
from .intelligent_extractor_simple import IntelligentDocumentExtractor
from .enhanced_query_handler_simple import EnhancedQueryHandler

logger = structlog.get_logger()

class RAGResponse:
    def __init__(self, answer: str, citations: List[Dict], confidence: float, session_id: str, follow_up_questions: List[str] = None, quality_metrics: Dict[str, float] = None):
        self.answer = answer
        self.citations = citations
        self.confidence = confidence
        self.session_id = session_id
        self.follow_up_questions = follow_up_questions or []
        self.quality_metrics = quality_metrics or {}

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
        
        self.embedding_model = OpenAIEmbedding(
            model=self.config['embedding_config']['model']
        )
        
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
        
        # Try Qdrant
        if self._try_qdrant():
            self.vector_store_type = "qdrant"
            self.index_name = self.collection_name
            logger.info("Using Qdrant vector store")
        # Fall back to local storage if Qdrant unavailable
        else:
            self._init_local_index()
            self.vector_store_type = "local"
            self.index_name = "local"
            logger.info("Using local vector store")
        
        # Initialize Redis for caching
        self._init_cache()
        
        # Initialize hybrid search components
        self._init_hybrid_search()
        
        # Initialize advanced relevance engine
        self.relevance_engine = RelevanceEngine()
        
        logger.info("RAG Pipeline initialized", 
                   model=self.config['llm_config']['model'],
                   embedding_model=self.config['embedding_config']['model'])
    
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
            
            logger.info("Initializing Qdrant indexes...")
            self._init_qdrant_indexes()
            logger.info("✅ Qdrant initialization complete")
            return True
            
        except Exception as e:
            logger.error("❌ Qdrant initialization failed", error=str(e), exc_info=True)
            return False
    
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
                    vectors_config=VectorParams(
                        size=1536,  # OpenAI embedding dimension
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            
            # Create index for job_id field to enable filtered deletion
            try:
                from qdrant_client.models import PayloadSchemaType
                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="job_id",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                logger.info("Created index for job_id field")
            except Exception as e:
                # Index might already exist, which is fine
                logger.info(f"job_id index status: {str(e)[:50]}")
            
            # Initialize storage context with Qdrant
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # Create or load index with Qdrant
            try:
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store
                )
                logger.info("Loaded existing Qdrant index")
            except:
                self.index = VectorStoreIndex(
                    nodes=[],
                    storage_context=self.storage_context
                )
                logger.info("Created new Qdrant index")
            
            logger.info("Qdrant indexes initialized")
            
        except Exception as e:
            logger.error("Failed to initialize Qdrant indexes", error=str(e))
            raise
    
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
        """Initialize BM25 keyword search for hybrid retrieval"""
        try:
            # Initialize BM25 index (will be populated when documents are added)
            self.bm25_index = None
            self.bm25_documents = []  # Store document texts for BM25
            self.bm25_metadata = []   # Store metadata for BM25 documents
            logger.info("✅ Hybrid search components initialized")
        except Exception as e:
            logger.error("Failed to initialize hybrid search", error=str(e))
            # Set fallback values
            self.bm25_index = None
            self.bm25_documents = []
            self.bm25_metadata = []
    
    def _update_bm25_index(self, nodes):
        """Update BM25 index with new document nodes"""
        try:
            # Add node texts and metadata to BM25 storage
            for node in nodes:
                text = node.get_content() if hasattr(node, 'get_content') else str(node.text)
                self.bm25_documents.append(text)
                self.bm25_metadata.append({
                    'node_id': node.node_id if hasattr(node, 'node_id') else str(uuid.uuid4()),
                    'metadata': node.metadata or {},
                    'text': text
                })
            
            # Rebuild BM25 index with all documents
            if self.bm25_documents:
                # Tokenize documents for BM25 (simple word splitting)
                tokenized_docs = [self._tokenize_text(doc) for doc in self.bm25_documents]
                self.bm25_index = BM25Okapi(tokenized_docs)
                logger.info(f"✅ BM25 index updated with {len(self.bm25_documents)} documents")
        except Exception as e:
            logger.error("Failed to update BM25 index", error=str(e))
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        # Convert to lowercase and split on non-alphanumeric characters
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _init_local_index(self):
        """Initialize local vector store as fallback"""
        # Create a simple in-memory vector store
        self.index = VectorStoreIndex(
            nodes=[]
        )
        logger.info("Initialized local vector store")
    
    async def add_documents(self, documents: List[Document], doc_type: str = "company"):
        """
        Add documents to the single index with document type metadata
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
            
            # Ensure nodes inherit the document metadata including job_id
            for node in nodes:
                if not node.metadata:
                    node.metadata = {}
                # Preserve important metadata from parent document
                node.metadata['doc_type'] = doc_type
                node.metadata['added_at'] = datetime.utcnow().isoformat()
                # Ensure job_id is preserved for deletion tracking
                if documents and documents[0].metadata and 'job_id' in documents[0].metadata:
                    node.metadata['job_id'] = documents[0].metadata['job_id']
                    node.metadata['filename'] = documents[0].metadata.get('filename', 'Unknown')
                
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
                    import re
                    page_match = re.search(r'Source: Page (\d+)', node_content)
                    if page_match:
                        node.metadata['page'] = int(page_match.group(1))
            
            # Add to single index
            self.index.insert_nodes(nodes)
            
            # Update BM25 index for hybrid search
            self._update_bm25_index(nodes)
            
            # Invalidate cache for this document type
            if self.cache:
                pattern = f"query_cache:{doc_type}:*"
                for key in self.cache.scan_iter(match=pattern):
                    self.cache.delete(key)
            
            logger.info(f"Added {len(documents)} documents to index",
                       doc_type=doc_type,
                       total_nodes=len(nodes),
                       bm25_documents=len(self.bm25_documents))
            
            return len(nodes)
            
        except Exception as e:
            logger.error("Failed to add documents", error=str(e), doc_type=doc_type)
            raise
    
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
        
        try:
            # Check cache first (but skip cache for debugging if needed)
            cache_key = self._get_cache_key(query, filters)
            use_cache = (self.cache and 
                        self.config['retrieval_config']['use_cache'] and
                        not os.getenv('DISABLE_CACHE', '').lower() in ('true', '1', 'yes'))
            
            if use_cache:
                cached_response = self.cache.get(cache_key)
                if cached_response:
                    logger.info("Returning cached response", query_hash=cache_key[:8])
                    return self._deserialize_response(cached_response, session_id)
            
            # Check if query is within AAIRE's domain expertise
            logger.info(f"🔍 Classifying query topic: '{query[:50]}...'")
            topic_check = await self._classify_query_topic(query)
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
            expanded_query = self._expand_query(query)
            
            # Get adaptive similarity threshold
            similarity_threshold = self._get_similarity_threshold(query)
            
            # Store current query for citation filtering
            self._current_query = query
            
            # ALWAYS search uploaded documents first
            retrieved_docs = await self._retrieve_documents(expanded_query, doc_type_filter, similarity_threshold, filters)
            
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
                
                response = await self._generate_response(query, retrieved_docs, user_context, conversation_history)
                
                # Post-process response to fix citation format and clean up text
                response = self._fix_citation_format(response, retrieved_docs)
                response = self._basic_text_cleanup(response)
                
                citations = self._extract_citations(retrieved_docs, query)
                confidence = self._calculate_confidence(retrieved_docs, response)
            else:
                # No relevant documents found - check if this could be relevant general knowledge
                is_general_query = self._is_general_knowledge_query(query)
                
                # Even if it's a general query, it must still be within AAIRE's domain
                if is_general_query:
                    # Re-check topic relevance for general knowledge questions
                    topic_check = await self._classify_query_topic(query)
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
                    response = await self._generate_response(query, [], user_context, conversation_history)
                    response = self._remove_citations_from_response(response)
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
                            sample_docs = await self._vector_search("document", None, 0.1)  # Very low threshold
                            available_docs = list(set([doc['metadata'].get('filename', 'Unknown') for doc in sample_docs[:5]]))
                    except:
                        pass
                    
                    if available_docs:
                        response = f"I couldn't find specific information about '{query}' in the uploaded documents. The available documents include: {', '.join(available_docs)}. Please verify that the document containing this information has been successfully uploaded and processed."
                    else:
                        response = f"I couldn't find specific information about '{query}' in the uploaded documents. Please ensure the relevant document has been uploaded and processed successfully."
                    
                    citations = []
                    confidence = 0.1  # Very low confidence when we can't find specific content
            
            # Clean up formatting before finalizing
            response = self._clean_formulas(response)
            
            # Generate contextual follow-up questions
            follow_up_questions = await self._generate_follow_up_questions(query, response, retrieved_docs)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(query, response, retrieved_docs, citations)
            
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
                    self._serialize_response(rag_response)
                )
            
            return rag_response
            
        except Exception as e:
            logger.error("Failed to process query", error=str(e), query=query[:100])
            raise
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
    
    async def _retrieve_documents(self, query: str, doc_type_filter: Optional[List[str]], similarity_threshold: Optional[float] = None, filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Hybrid retrieval: combines vector search with BM25 keyword search using buffer approach"""
        
        # Get the target document limit
        doc_limit = self._get_document_limit(query)
        
        # Use buffer approach: both searches get the full limit to ensure we don't miss important documents
        # This allows the best documents from either method to compete fairly
        vector_results = await self._vector_search(query, doc_type_filter, similarity_threshold, filters, buffer_limit=doc_limit)
        keyword_results = await self._keyword_search(query, doc_type_filter, filters, buffer_limit=doc_limit)
        
        logger.info(f"Buffer approach: Retrieved {len(vector_results)} vector + {len(keyword_results)} keyword results, target limit: {doc_limit}")
        
        # Combine results using advanced relevance engine
        all_results = vector_results + keyword_results
        
        # Remove duplicates while preserving all scoring info
        unique_results = {}
        for result in all_results:
            node_id = result['node_id']
            if node_id not in unique_results:
                unique_results[node_id] = result
            else:
                # Merge scoring information from both searches
                existing = unique_results[node_id]
                existing['search_type'] = 'hybrid'
                # Keep the higher score
                if result['score'] > existing['score']:
                    existing['score'] = result['score']
        
        # Use advanced relevance engine for ranking
        combined_results = self.relevance_engine.rank_documents(query, list(unique_results.values()))
        
        # Apply the document limit to the final combined results
        if len(combined_results) > doc_limit:
            logger.info(f"Trimming combined results from {len(combined_results)} to {doc_limit}")
            combined_results = combined_results[:doc_limit]
        
        return combined_results
    
    async def _vector_search(self, query: str, doc_type_filter: Optional[List[str]], similarity_threshold: Optional[float] = None, filters: Optional[Dict[str, Any]] = None, buffer_limit: Optional[int] = None) -> List[Dict]:
        """Original vector-based semantic search with filtering support"""
        all_results = []
        
        try:
            # Use buffer limit if provided (for combined ranking), otherwise use dynamic calculation
            if buffer_limit is not None:
                vector_limit = buffer_limit
            else:
                # Get dynamic document limit based on query
                doc_limit = self._get_document_limit(query)
                # For standalone search, allocate 70% to vector search
                vector_limit = int(doc_limit * 0.7)
            
            # Create retriever from single index with dynamic limit
            retriever = self.index.as_retriever(
                similarity_top_k=vector_limit
            )
            
            # Retrieve documents
            nodes = retriever.retrieve(query)
            
            # Use adaptive threshold if provided, otherwise fall back to config
            threshold = similarity_threshold if similarity_threshold is not None else self.config['retrieval_config']['similarity_threshold']
            
            for node in nodes:
                if node.score >= threshold:
                    # Apply job_id filter if specified (highest priority)
                    if filters and 'job_id' in filters:
                        node_job_id = node.metadata.get('job_id') if node.metadata else None
                        if node_job_id != filters['job_id']:
                            continue  # Skip nodes from different documents
                    
                    # Apply document type filter if specified
                    if doc_type_filter:
                        node_doc_type = node.metadata.get('doc_type') if node.metadata else None
                        if node_doc_type not in doc_type_filter:
                            continue  # Skip nodes that don't match filter
                    
                    all_results.append({
                        'content': node.text,
                        'metadata': node.metadata or {},
                        'score': node.score,
                        'source_type': node.metadata.get('doc_type', 'unknown') if node.metadata else 'unknown',
                        'node_id': node.id_,
                        'search_type': 'vector'
                    })
                    
        except Exception as e:
            logger.warning("Failed to retrieve from index", error=str(e))
        
        # Sort by relevance score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        # Note: vector_limit is already applied in retriever, but trimming here for safety
        return all_results[:vector_limit] if 'vector_limit' in locals() else all_results
    
    async def _keyword_search(self, query: str, doc_type_filter: Optional[List[str]], filters: Optional[Dict[str, Any]] = None, buffer_limit: Optional[int] = None) -> List[Dict]:
        """BM25-based keyword search"""
        results = []
        
        try:
            if not self.bm25_index or not self.bm25_documents:
                logger.info("BM25 index not available, skipping keyword search")
                return results
            
            # Use buffer limit if provided (for combined ranking), otherwise use dynamic calculation
            if buffer_limit is not None:
                keyword_limit = buffer_limit
            else:
                # Get dynamic document limit based on query
                doc_limit = self._get_document_limit(query)
                # For standalone search, allocate 30% to keyword search
                keyword_limit = int(doc_limit * 0.3)
            
            # Tokenize query for BM25
            query_tokens = self._tokenize_text(query)
            
            # Get BM25 scores for all documents
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            
            # Create results with scores
            for i, score in enumerate(bm25_scores):
                if score > 0 and i < len(self.bm25_metadata):  # Only include docs with positive scores
                    doc_metadata = self.bm25_metadata[i]
                    
                    # Apply job_id filter if specified (highest priority)
                    if filters and 'job_id' in filters:
                        doc_job_id = doc_metadata['metadata'].get('job_id')
                        if doc_job_id != filters['job_id']:
                            continue  # Skip docs from different documents
                    
                    # Apply document type filter if specified
                    if doc_type_filter:
                        doc_type = doc_metadata['metadata'].get('doc_type')
                        if doc_type not in doc_type_filter:
                            continue
                    
                    results.append({
                        'content': doc_metadata['text'],
                        'metadata': doc_metadata['metadata'],
                        'score': float(score),  # BM25 score
                        'source_type': doc_metadata['metadata'].get('doc_type', 'unknown'),
                        'node_id': doc_metadata['node_id'],
                        'search_type': 'keyword'
                    })
            
            # Sort by BM25 score and take top results
            results.sort(key=lambda x: x['score'], reverse=True)
            results = results[:keyword_limit]
            
            logger.info(f"BM25 keyword search found {len(results)} results")
            
        except Exception as e:
            logger.error("Failed to perform BM25 keyword search", error=str(e))
        
        return results
    
    def _combine_search_results(self, vector_results: List[Dict], keyword_results: List[Dict], query: str) -> List[Dict]:
        """Combine and rerank results from vector and keyword search with exact match boosting"""
        
        try:
            # Create a combined results dictionary to avoid duplicates
            combined_dict = {}
            
            # Check for exact matches in query (like ASC codes)
            exact_match_patterns = re.findall(r'\b(ASC\s+\d{3}-\d{2}-\d{2}-\d{1,2})\b', query, re.IGNORECASE)
            has_exact_patterns = len(exact_match_patterns) > 0
            
            # Normalize scores and add vector results
            max_vector_score = max([r['score'] for r in vector_results], default=1.0)
            for result in vector_results:
                node_id = result['node_id']
                normalized_score = result['score'] / max_vector_score if max_vector_score > 0 else 0
                
                # Check for exact pattern matches in content
                exact_match_bonus = 0.0
                if has_exact_patterns:
                    content = result.get('content', '')
                    for pattern in exact_match_patterns:
                        if re.search(re.escape(pattern), content, re.IGNORECASE):
                            exact_match_bonus += 0.5  # Significant boost for exact matches
                            logger.info(f"Exact match bonus applied for '{pattern}' in {result['metadata'].get('filename', 'Unknown')}")
                
                combined_dict[node_id] = result.copy()
                combined_dict[node_id]['vector_score'] = normalized_score
                combined_dict[node_id]['keyword_score'] = 0.0
                combined_dict[node_id]['exact_match_bonus'] = exact_match_bonus
                combined_dict[node_id]['combined_score'] = (normalized_score * 0.6) + exact_match_bonus  # Vector + exact match bonus
            
            # Normalize scores and add/update keyword results
            max_keyword_score = max([r['score'] for r in keyword_results], default=1.0)
            for result in keyword_results:
                node_id = result['node_id']
                normalized_score = result['score'] / max_keyword_score if max_keyword_score > 0 else 0
                
                # Check for exact matches in keyword results too
                exact_match_bonus = 0.0
                if has_exact_patterns:
                    content = result.get('content', '')
                    for pattern in exact_match_patterns:
                        if re.search(re.escape(pattern), content, re.IGNORECASE):
                            exact_match_bonus += 0.5
                
                if node_id in combined_dict:
                    # Update existing result with keyword score
                    combined_dict[node_id]['keyword_score'] = normalized_score
                    # Recalculate with all components
                    combined_dict[node_id]['combined_score'] = (
                        combined_dict[node_id]['vector_score'] * 0.6 + 
                        normalized_score * 0.4 + 
                        combined_dict[node_id]['exact_match_bonus']  # Keep existing bonus
                    )
                    combined_dict[node_id]['search_type'] = 'hybrid'
                else:
                    # Add new keyword-only result
                    combined_dict[node_id] = result.copy()
                    combined_dict[node_id]['vector_score'] = 0.0
                    combined_dict[node_id]['keyword_score'] = normalized_score
                    combined_dict[node_id]['exact_match_bonus'] = exact_match_bonus
                    combined_dict[node_id]['combined_score'] = (normalized_score * 0.4) + exact_match_bonus
            
            # Convert back to list and sort by combined score
            final_results = list(combined_dict.values())
            final_results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Take top results and clean up temporary scoring fields
            doc_limit = self._get_document_limit(query)
            final_results = final_results[:doc_limit]
            for result in final_results:
                result['score'] = result['combined_score']  # Set final score
                # Keep detailed scores for debugging but rename
                result.pop('combined_score', None)
                # Optionally remove detailed scores to clean up
                # result.pop('vector_score', None)
                # result.pop('keyword_score', None)
            
            logger.info(f"Hybrid search combined {len(vector_results)} vector + {len(keyword_results)} keyword results into {len(final_results)} final results")
            
            return final_results
            
        except Exception as e:
            logger.error("Failed to combine search results", error=str(e))
            # Fallback to vector results only with dynamic limit
            doc_limit = self._get_document_limit(query)
            return vector_results[:doc_limit]
    
    
    def _get_diverse_context_documents(self, documents: List[Dict]) -> List[Dict]:
        """Select diverse documents for context to ensure comprehensive coverage"""
        # FIXED: Return all documents for maximum comprehensive coverage
        # The issue was complex deduplication logic that was too aggressive
        logger.info(f"🔍 DIVERSE SELECTION DEBUG: Input={len(documents)}, Output={len(documents)} (returning all)")
        return documents
    
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
    
    def _generate_single_pass_response(self, query: str, documents: List[Dict], conversation_context: str) -> str:
        """Fallback single-pass response for smaller document sets"""
        context = "\n\n".join([f"[Doc {i+1}]\n{doc['content']}" for i, doc in enumerate(documents)])
        
        prompt = f"""You are AAIRE, an expert in insurance accounting.
{conversation_context}
Question: {query}

Documents:
{context}

Provide a comprehensive response using all the information in the documents above.

FORMATTING REQUIREMENTS:
- Use **bold** ONLY for section headings and key terms (NO # symbols)

🚨 CRITICAL NUMBERED LIST FORMATTING (FOLLOW EXACTLY):
Never write title**1.**content or text**2.**moretext on same line.

❌ WRONG: "Universal Life Policy in ULSG**1.**Determine the Adjusted"
✅ CORRECT: "Universal Life Policy in ULSG

**1.** Determine the Adjusted"

ALWAYS separate numbered items from preceding text:

1. First item content here

2. Second item content here  

3. Third item content here

- Use bullet points (-) with proper spacing:

- First bullet point

- Second bullet point

- Regular text should NOT be bold
- Add two line breaks between major sections  
- CRITICAL: Never concatenate numbered items with preceding text
- Include all formulas, calculations, and requirements found"""
        
        response = self.llm.complete(prompt)
        return response.text.strip()
    
    def _merge_response_parts(self, query: str, response_parts: List[str]) -> str:
        """Merge multiple response parts into a coherent final response"""
        if len(response_parts) == 1:
            return response_parts[0]
        
        merge_prompt = f"""Combine these related responses into a single, well-organized answer to: {query}

Response parts to merge:
{chr(10).join([f"PART {i+1}:\n{part}\n" for i, part in enumerate(response_parts)])}

FORMATTING REQUIREMENTS:
- Use proper markdown formatting: # for main sections, ## for subsections
- Use numbered lists (1., 2., 3.) with proper spacing
- Use bullet points (-) for sub-items
- Use **bold** only for emphasis within text, not for headers
- Ensure clear spacing between sections and lists

CONTENT REQUIREMENTS:
- Eliminate any redundancy between parts
- Organize information logically with clear markdown headers
- Maintain ALL technical details, formulas, and calculations from each part
- Keep EXACT formulas and preserve specific values like 90%, $2.50 per $1,000, etc.
- Ensure the response flows naturally as a complete answer

Create a cohesive response using proper markdown formatting:"""
        
        response = self.llm.complete(merge_prompt)
        return response.text.strip()
    
    def _completeness_check(self, query: str, response: str, documents: List[Dict]) -> str:
        """Check if response missed important content and add it"""
        try:
            # Optimize: Only check top 20 most relevant documents for completeness
            # These contain the most important information and reduce processing time
            top_docs = documents[:20] if len(documents) > 20 else documents
            
            # Create condensed view of top document content
            all_content_snippets = []
            for i, doc in enumerate(top_docs, 1):
                snippet = doc['content'][:500].replace('\n', ' ')
                all_content_snippets.append(f"Doc {i}: {snippet}...")
            
            logger.info(f"Completeness check: Analyzing {len(top_docs)} top documents (out of {len(documents)} total)")
            
            check_prompt = f"""Compare this response against the source documents to identify missing content.

User Question: {query}

Current Response:
{response[:1500]}...

Source Documents:
{chr(10).join(all_content_snippets)}

Identify any important concepts, methods, calculations, or considerations mentioned in the documents that are missing from the response.
Focus on content that would be relevant to answering the user's question.

If significant content is missing, list it. If the response is complete, respond with "COMPLETE".

Missing content:"""
            
            missing_check = self.llm.complete(check_prompt)
            missing_content = missing_check.text.strip()
            
            logger.info(f"🔍 COMPLETENESS CHECK RESULT: {missing_content[:200]}...")
            
            # Enhanced debugging for completeness check
            logger.info(f"🔍 DEBUG: missing_content length: {len(missing_content) if missing_content else 0}")
            logger.info(f"🔍 DEBUG: missing_content truthy: {bool(missing_content)}")
            logger.info(f"🔍 DEBUG: 'COMPLETE' in upper: {'COMPLETE' in missing_content.upper() if missing_content else 'N/A'}")
            
            # Check if the response is exactly "COMPLETE" (not just containing the word)
            if missing_content and missing_content.strip():
                if missing_content.strip().upper() == "COMPLETE" or missing_content.strip().upper().startswith("COMPLETE."):
                    logger.info("✅ Response marked as COMPLETE by completeness check")
                else:
                    logger.info("✅ Missing content identified, enhancing response")
                    # Add missing content
                    enhancement_prompt = f"""Enhance this response by adding the missing content identified below.

Original Response:
{response}

Missing Content to Add:
{missing_content}

Create an enhanced response that includes the original content plus the missing information.

FORMATTING REQUIREMENTS:
- Use **bold** ONLY for section headings and key terms (NO # symbols)
- Put each numbered list item on its own line (1. on one line, 2. on next line, etc.)
- Use proper line breaks between sections for readability
- Regular text should NOT be bold
- Add blank lines between major sections
- Use bullet points (-) with proper spacing

Enhanced Response:"""
                    
                    enhanced = self.llm.complete(enhancement_prompt)
                    logger.info("✅ Added missing content via completeness check")
                    return enhanced.text.strip()
            else:
                logger.info("❌ No missing content response received")
            
            return response
            
        except Exception as e:
            logger.warning(f"Completeness check failed: {str(e)}")
            return response
    
    def _fix_reserve_terminology(self, response: str) -> str:
        """Fix common terminology errors in reserve calculations"""
        import re
        
        try:
            # Fix Deferred Reserve -> Deterministic Reserve
            response = re.sub(r'\bDeferred Reserve\b', 'Deterministic Reserve', response, flags=re.IGNORECASE)
            response = re.sub(r'\bDeferred Reserves\b', 'Deterministic Reserves', response, flags=re.IGNORECASE)
            
            # Ensure DR is correctly defined
            response = re.sub(r'\bDR\s*=\s*Deferred', 'DR = Deterministic', response, flags=re.IGNORECASE)
            
            # Fix Scenario Reserve -> Stochastic Reserve (if needed)
            # Note: Scenario Reserve is sometimes acceptable, but Stochastic is more precise
            response = re.sub(r'\bScenario Reserve\s*\(SR\)', 'Stochastic Reserve (SR)', response)
            
            # Fix inconsistent header formatting
            response = re.sub(r'\*\*([A-Z][a-z]+.*?):\*\*\s*([A-Z])', r'\n**\1**\n\2', response)
            response = re.sub(r'•\s*\*\*(.*?):\*\*', r'• **\1:**', response)
            
            logger.info("Fixed reserve terminology and formatting")
            return response
            
        except Exception as e:
            logger.warning(f"Failed to fix terminology: {str(e)}")
            return response
    
    def _clean_formulas(self, response: str) -> str:
        """Normalize LaTeX math for Markdown renderers (KaTeX/MathJax).

        - Preserve existing \[ ... \] and \( ... \) blocks
        - Convert $$ ... $$ to display math \[ ... \]
        - Convert inline $ ... $ to \( ... \) while avoiding currency amounts
        - Do NOT alter LaTeX commands or subscripts inside math
        """
        import re

        try:
            logger.info("🧮 Normalizing formula formatting for math rendering")

            text = response

            # 1) Normalize display math: $$...$$ -> \[ ... \]
            def replace_display(match):
                content = match.group(1).strip()
                return f"\\[{content}\\]"

            text = re.sub(r"\$\$(.+?)\$\$", replace_display, text, flags=re.DOTALL)

            # 2) Normalize inline math: $...$ -> \( ... \), avoid currency like $1,000 or $2.5 million
            inline_pattern = re.compile(r"(?<!\$)\$(?!\$)([^\n$]+?)(?<!\$)\$(?!\$)")

            def is_currency(s: str) -> bool:
                s_strip = s.strip()
                # Pure numbers with commas/decimals and optional unit words
                if re.fullmatch(r"[\d,]+(?:\.\d+)?(?:\s*(?:k|m|bn|million|billion|thousand))?", s_strip, flags=re.IGNORECASE):
                    return True
                # Starts or ends with only digits/commas/decimals
                if re.fullmatch(r"[\d,]+(?:\.\d+)?", s_strip):
                    return True
                return False

            def replace_inline(match):
                content = match.group(1)
                if is_currency(content):
                    # Leave as-is (likely currency in prose, not math)
                    return match.group(0)
                return f"\\({content.strip()}\\)"

            text = inline_pattern.sub(replace_inline, text)

            # 3) Preserve existing \[ ... \] and \( ... \) blocks as-is; avoid destructive replacements

            # 4) Light whitespace cleanup without touching math delimiters
            # Collapse spaces but preserve newlines and not inside math delimiters (best-effort global cleanup)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\*\*\s+", "**", text)
            text = re.sub(r"\s+\*\*", "**", text)

            logger.info("✅ Formula normalization complete")
            return text

        except Exception as e:
            logger.warning(f"Formula cleaning failed: {str(e)}")
            return response
    
    def _preserve_formulas(self, response: str, documents: List[Dict]) -> str:
        """Extract and preserve mathematical formulas from documents"""
        try:
            # Extract all mathematical content from documents
            all_content = " ".join([doc['content'] for doc in documents])
            
            formula_prompt = f"""Extract ALL mathematical formulas, equations, and calculations from this content.

Content:
{all_content[:4000]}

Find every formula, equation, calculation method, or mathematical expression.
Preserve the exact notation including LaTeX markup, variables, subscripts, etc.

Output each formula with a brief description:
1. [Description]: [Exact formula as written]
2. [Description]: [Exact formula as written]

If no formulas found, respond with "NO_FORMULAS".

Extracted formulas:"""
            
            formula_response = self.llm.complete(formula_prompt)
            extracted_formulas = formula_response.text.strip()
            
            logger.info(f"🧮 FORMULA EXTRACTION RESULT: {extracted_formulas[:300]}...")
            
            if extracted_formulas and "NO_FORMULAS" not in extracted_formulas:
                # Check if formulas are already well-represented in response
                formula_check_prompt = f"""Are these mathematical formulas adequately represented in the response?

Response:
{response[:1000]}...

Formulas from documents:
{extracted_formulas}

If key formulas are missing or oversimplified, respond with "ADD_FORMULAS".
If formulas are well-represented, respond with "FORMULAS_ADEQUATE".

Assessment:"""
                
                formula_check = self.llm.complete(formula_check_prompt)
                
                logger.info(f"🧮 FORMULA CHECK RESULT: {formula_check.text.strip()}")
                
                if "ADD_FORMULAS" in formula_check.text:
                    # Add mathematical formulas section
                    enhanced_response = f"""{response}

## Mathematical Formulas and Calculations

{extracted_formulas}"""
                    
                    logger.info("✅ Added mathematical formulas section")
                    return enhanced_response
                else:
                    logger.info("❌ Formulas determined to be adequately represented")
            else:
                logger.info("❌ No formulas extracted or NO_FORMULAS returned")
            
            return response
            
        except Exception as e:
            logger.warning(f"Formula preservation failed: {str(e)}")
            return response
    
    async def _process_with_chunked_enhancement(self, query: str, retrieved_docs: List[Dict], conversation_context: str) -> str:
        """Main processing method that combines all our enhancements"""
        try:
            # Get diverse documents for processing
            diverse_docs = self._get_diverse_context_documents(retrieved_docs)
            logger.info(f"📚 Processing {len(diverse_docs)} diverse documents (out of {len(retrieved_docs)} total)")
            
            # Check for organizational queries first
            if self._is_organizational_query(query, diverse_docs):
                return self._generate_organizational_response(query, diverse_docs, conversation_context)
            
            # Use chunked processing for comprehensive coverage
            if len(diverse_docs) <= 8:
                # Small document set - enhanced single pass
                response = self._generate_enhanced_single_pass(query, diverse_docs, conversation_context)
            else:
                # Large document set - semantic chunking
                response = await self._generate_chunked_response(query, diverse_docs, conversation_context)
            
            # Apply enhancements
            enhanced_response = self._completeness_check(query, response, diverse_docs)
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Enhanced processing failed: {str(e)}")
            # Fallback to simple approach
            return self._generate_enhanced_single_pass(query, diverse_docs[:5], conversation_context)
    
    def _is_organizational_query(self, query: str, documents: List[Dict]) -> bool:
        """Check if this is an organizational structure query"""
        org_terms = ['breakdown by job', 'organizational structure', 'job titles', 'hierarchy']
        has_org_query = any(term in query.lower() for term in org_terms)
        
        if has_org_query:
            # Check if documents contain spatial extraction data
            sample_content = " ".join([doc['content'][:300] for doc in documents[:3]])
            return '[SHAPE-AWARE ORGANIZATIONAL EXTRACTION]' in sample_content
        
        return False
    
    def _generate_organizational_response(self, query: str, documents: List[Dict], conversation_context: str) -> str:
        """Generate response for organizational structure queries"""
        context = "\n\n".join([f"[Doc {i+1}]\n{doc['content']}" for i, doc in enumerate(documents)])
        
        prompt = f"""You are AAIRE, an expert in insurance accounting and actuarial matters.
{conversation_context}
Question: {query}

Organizational data:
{context}

Provide a clear organizational breakdown based on the spatial extraction data found in the documents.
Use appropriate headings and structure the information clearly."""
        
        response = self.llm.complete(prompt)
        return response.text.strip()
    
    def _generate_enhanced_single_pass(self, query: str, documents: List[Dict], conversation_context: str) -> str:
        """Enhanced single-pass response for smaller document sets"""
        context = "\n\n".join([f"[Doc {i+1}]\n{doc['content']}" for i, doc in enumerate(documents)])
        
        prompt = f"""You are AAIRE, an expert in insurance accounting.
{conversation_context}
Question: {query}

Documents:
{context}

Create a comprehensive response that addresses ALL aspects covered in the retrieved documents.

FORMATTING REQUIREMENTS:
- Use proper markdown formatting with headers: # for main sections, ## for subsections
- Use numbered lists (1., 2., 3.) with proper line breaks
- Use bullet points (-) for sub-items
- Keep mathematical formulas and expressions clear and readable
- Use **bold** only for emphasis within text, not for headers
- Ensure proper spacing between sections and lists

CONTENT REQUIREMENTS:
- Include ALL relevant regulatory sections, formulas, and calculations
- Address ALL distinct concepts and methodologies mentioned
- Preserve ALL technical details and specific requirements
- Convert complex mathematical notation to readable text format (e.g., 𝐸𝑥+𝑡 = 𝑉𝑁𝑃𝑅⦁𝑎̈𝑥+𝑡:𝑣−𝑡| becomes E(x+t) = VNPR × annuity(x+t):v-t)
- Replace complex Unicode symbols with readable text
- Convert actuarial notation to plain English equivalents
- Maintain mathematical accuracy while ensuring accessibility

Structure your response to systematically cover every major topic found in the source material using clear markdown formatting."""
        
        response = self.llm.complete(prompt)
        return response.text.strip()
    
    async def _generate_chunked_response(self, query: str, documents: List[Dict], conversation_context: str) -> str:
        """Generate response using semantic chunking for large document sets"""
        # Create semantic groups
        document_groups = self._create_semantic_document_groups(documents)
        response_parts = []
        
        # Process all groups in parallel using threading for faster response
        def process_group(group_index: int, doc_group: List[Dict]) -> str:
            group_context = "\n\n".join([f"[Doc {i+1}]\n{doc['content']}" for i, doc in enumerate(doc_group)])
            
            group_prompt = f"""You are answering: {query}

This is document group {group_index} of {len(document_groups)}. Focus on these documents:

{group_context}

FORMATTING REQUIREMENTS:
- Use proper markdown formatting: # for main sections, ## for subsections  
- Use numbered lists (1., 2., 3.) with proper spacing
- Use bullet points (-) for sub-items
- Use **bold** only for emphasis within text, not for headers
- Ensure clear spacing between sections and lists

CONTENT REQUIREMENTS:
- Include ALL relevant formulas, calculations, and mathematical expressions
- Preserve specific numerical values like 90%, $2.50 per $1,000, etc.
- Copy EXACT formulas from documents 
- Include ALL calculation methods and procedures
- Maintain technical accuracy and detail

Provide a detailed response covering all information that relates to the question using proper markdown formatting."""
            
            group_response = self.llm.complete(group_prompt)
            logger.info(f"Processed group {group_index}/{len(document_groups)}")
            return group_response.text.strip()
        
        # Process all groups concurrently using ThreadPoolExecutor
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_group, i+1, doc_group) for i, doc_group in enumerate(document_groups)]
            response_parts = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Temporarily disable structured JSON approach - has parsing issues
        # TODO: Fix JSON parsing and markdown conversion in structured approach
        logger.info("📋 Using enhanced chunked response approach")
        
        # Fallback to existing chunked approach
        # Merge all parts
        merged_response = self._merge_response_parts(query, response_parts)
        
        # Apply basic normalization only (no heavy post-processing since structured failed)
        return self._normalize_spacing(merged_response)
    
    
    async def _generate_response(
        self, 
        query: str, 
        retrieved_docs: List[Dict],
        user_context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """Generate response using retrieved documents and conversation context"""
        
        # Build conversation context if available
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            # Use config settings for better memory retention
            max_history = self.config.get('conversation_config', {}).get('max_history_messages', 20)
            max_msg_length = self.config.get('conversation_config', {}).get('max_message_length', 500)
            
            conversation_context = f"\n\nConversation History (last {max_history//2} exchanges):\n"
            
            # Get recent history based on config
            recent_history = conversation_history[-max_history:] if len(conversation_history) > max_history else conversation_history
            
            for msg in recent_history:
                role = "User" if msg.get('sender') == 'user' else "Assistant"
                content = msg.get('content', '')[:max_msg_length]  # Use configurable truncation
                conversation_context += f"{role}: {content}\n"
        
        # Check if we have relevant documents
        if not retrieved_docs:
            # No relevant documents found - check topic classification before general knowledge response
            topic_check = await self._classify_query_topic(query)
            if not topic_check['is_relevant']:
                logger.info(f"❌ General knowledge request rejected as off-topic: '{query[:50]}...'")
                return topic_check['polite_response']
                
            # Query is relevant, provide general knowledge response
            # Check if this is a calculation request
            calc_config = self.config.get('calculation_config', {})
            calc_enhancement = ""
            if calc_config.get('enable_structured_calculations') and any(kw in query.lower() for kw in ['calculate', 'amortization', 'schedule', 'table', 'payment', 'journal']):
                calc_enhancement = f"\n\nCalculation Instructions:\n{calc_config.get('calculation_instructions', '')}"
            
            prompt = f"""You are AAIRE, an expert in insurance accounting and actuarial matters.
You provide accurate information based on US GAAP, IFRS, and general accounting principles.
{conversation_context}
Current User Question: {query}

This appears to be a general accounting question. I will provide a standard accounting explanation.{calc_enhancement}

Instructions:
- Consider the conversation history to provide contextual answers
- Provide a helpful general answer based on standard accounting and actuarial principles
- This is general accounting knowledge, not from any specific company documents
- Mention relevant accounting standards (US GAAP, IFRS) where applicable
- Never provide tax or legal advice
- CRITICAL: If performing ANY calculations, double-check ALL arithmetic (25×19,399×45=21,823,875 NOT 21,074,875)
- Show step-by-step calculations with accurate intermediate results
- Do NOT include any citation numbers like [1], [2], etc.
- Do NOT reference any specific documents or sources
- Make it clear this is general knowledge, not company-specific information

Response:"""
        else:
            # Use dynamic chunked processing for all non-general queries
            return await self._process_with_chunked_enhancement(query, retrieved_docs, conversation_context)
        
        # General knowledge response
        response = self.llm.complete(prompt)
        return response.text.strip()
    
    def _determine_question_categories(self, query: str, response: str, retrieved_docs: List[Dict]) -> List[str]:
        """Determine appropriate question categories based on context"""
        categories = []
        
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Check for specific topics and suggest relevant categories
        if any(term in query_lower for term in ['gaap', 'ifrs', 'standard', 'compliance']):
            categories.extend(['comparison', 'compliance'])
        
        if any(term in query_lower for term in ['reserve', 'calculation', 'premium', 'claim']):
            categories.extend(['examples', 'technical'])
        
        if any(term in query_lower for term in ['audit', 'test', 'review']):
            categories.extend(['application', 'compliance'])
        
        if any(term in response_lower for term in ['require', 'must', 'shall']):
            categories.append('clarification')
        
        # Default categories if none detected
        if not categories:
            categories = ['clarification', 'examples', 'application']
        
        return list(set(categories))  # Remove duplicates
    
    def _get_category_examples(self, categories: List[str]) -> Dict[str, List[str]]:
        """Get example questions for each category"""
        category_questions = {
            'clarification': [
                "Can you explain this in simpler terms?",
                "What does this mean in practice?",
                "Could you break this down further?"
            ],
            'examples': [
                "Can you provide a real-world example?",
                "How would this work for a life insurance company?",
                "What would this look like in financial statements?"
            ],
            'comparison': [
                "How does this differ under IFRS vs US GAAP?",
                "What are the key differences from previous standards?",
                "How does this compare to industry practice?"
            ],
            'technical': [
                "What are the detailed calculation steps?",
                "What assumptions are typically used?",
                "How do you handle edge cases?"
            ],
            'application': [
                "How do companies typically implement this?",
                "What systems support this process?",
                "How often should this be performed?"
            ],
            'compliance': [
                "What are the audit requirements?",
                "How do regulators typically examine this?",
                "What documentation is needed?"
            ]
        }
        
        return {cat: category_questions.get(cat, []) for cat in categories}

    async def _generate_follow_up_questions(self, query: str, response: str, retrieved_docs: List[Dict]) -> List[str]:
        """Generate contextual follow-up questions based on the query and response"""
        
        # Determine appropriate question categories
        categories = self._determine_question_categories(query, response, retrieved_docs)
        category_examples = self._get_category_examples(categories)
        
        # Analyze actual document content for specific follow-up opportunities
        content_insights = self._analyze_document_content_for_followups(retrieved_docs[:2])  # Analyze top 2 docs
        
        # Build rich context from document content analysis
        topic_context = ""
        if content_insights:
            context_parts = []
            for insight_type, items in content_insights.items():
                if items and insight_type != 'source_docs':
                    context_parts.append(f"{insight_type}: {', '.join(items[:3])}")  # Top 3 items per type
            
            if context_parts:
                topic_context = f"Document content analysis:\n" + "\n".join(context_parts)
            
            # Add source document info
            source_docs = content_insights.get('source_docs', [])
            if source_docs:
                topic_context += f"\nSource documents: {', '.join(source_docs[:2])}"
        
        # Build category guidance
        category_guidance = ""
        for cat, examples in category_examples.items():
            category_guidance += f"\n{cat.title()}: {', '.join(examples[:2])}"
        
        # Create content-specific guidance for better follow-ups
        content_guidance = ""
        if content_insights:
            guidance_parts = []
            
            if content_insights.get('standards_mentioned'):
                guidance_parts.append(f"Consider asking about other standards: {', '.join(content_insights['standards_mentioned'][:3])}")
            
            if content_insights.get('examples_found'):
                guidance_parts.append("Ask about specific examples or scenarios mentioned in the documents")
            
            if content_insights.get('tables_data'):
                guidance_parts.append(f"Reference specific data: {', '.join(content_insights['tables_data'][:2])}")
            
            if content_insights.get('key_concepts'):
                guidance_parts.append(f"Explore concepts like: {', '.join(content_insights['key_concepts'][:3])}")
            
            if content_insights.get('implementation_terms'):
                guidance_parts.append("Ask about implementation, transition, or adoption aspects")
            
            if guidance_parts:
                content_guidance = f"\nContent-specific opportunities:\n" + "\n".join([f"- {part}" for part in guidance_parts])

        # Extract specific elements from the actual response to create targeted follow-ups
        response_elements = self._extract_response_elements(response)
        document_specifics = self._get_document_specifics(retrieved_docs[:2])
        
        prompt = f"""Generate 2-3 highly specific follow-up questions based on this EXACT conversation and documents.

USER ASKED: "{query}"

MY RESPONSE: "{response[:500]}..." 

SPECIFIC DOCUMENT CONTENT USED:
{document_specifics}

RESPONSE ANALYSIS:
{response_elements}

CRITICAL INSTRUCTIONS:
- Questions must be DIRECTLY related to what I just explained to the user
- Reference SPECIFIC information from the documents that were actually cited
- Build upon the EXACT conversation context and dig deeper into specifics
- NO generic business/insurance questions
- Focus on specific metrics, segments, time periods, or data points I mentioned
- If discussing financial results, ask about specific components or related metrics
- If discussing business segments, ask about other segments or comparative performance
- If discussing time periods, ask about trends or comparisons to other periods

Examples of GOOD contextual questions for financial discussions:
- "What drove the unfavorable claims experience in U.S. Traditional that you mentioned?"
- "How did the other business segments perform compared to the 14.3% ROE you cited?"
- "What specific factors contributed to the $276 million capital deployment figure?"
- "Can you break down the components of the variable investment income mentioned?"

Examples of GOOD contextual questions for technical documents:
- "What does the PWC document say about the implementation timeline for this standard?"
- "How does the calculation method in section 3.2 apply to different scenarios?"
- "What are the disclosure requirements mentioned alongside this guidance?"

Examples of BAD generic questions to COMPLETELY AVOID:
- "How do claims impact profitability?" 
- "What strategies do insurers use for capital management?"
- "Can you explain adjusted operating income?"
- "What are the benefits of this approach?"

Generate exactly 2-3 contextual follow-up questions that dig deeper into the specific information I just provided:"""

        try:
            response_obj = self.llm.complete(prompt)
            questions_text = response_obj.text.strip()
            
            # Parse the response into individual questions
            questions = []
            logger.info(f"🎯 AI generated follow-up questions: {questions_text}")
            for line in questions_text.split('\n'):
                line = line.strip()
                if line and len(line) > 10:  # Filter out empty or very short lines
                    # Clean up any unwanted formatting - remove numbers, bullets, quotes
                    clean_question = line.strip('- •').strip()
                    # Remove numbering like "1. " or "2. "
                    clean_question = re.sub(r'^\d+\.\s*', '', clean_question)
                    # Remove surrounding quotes
                    clean_question = clean_question.strip('"\'').strip()
                    
                    if clean_question.endswith('?') and len(clean_question) > 10:
                        # Validate question is contextual (not generic)
                        is_contextual = self._is_contextual_question(clean_question, query, response)
                        logger.info(f"🔍 Question validation: '{clean_question}' -> {'✅' if is_contextual else '❌'}")
                        if is_contextual:
                            questions.append(clean_question)
            
            # Return max 3 questions, fallback if none are contextual
            if questions:
                return questions[:3]
            else:
                logger.warning("No contextual follow-up questions generated, using response-based fallback")
                return self._generate_response_based_fallback(query, response, retrieved_docs)
            
        except Exception as e:
            logger.error("Failed to generate follow-up questions", error=str(e))
            # Return fallback questions if generation fails
            return [
                "Can you explain this in more detail?",
                "What are the practical implications?",
                "How does this apply in practice?"
            ]
    
    def _analyze_document_content_for_followups(self, retrieved_docs: List[Dict]) -> Dict[str, List[str]]:
        """Analyze document content to extract specific elements for targeted follow-up questions"""
        insights = {
            'standards_mentioned': [],
            'examples_found': [],
            'tables_data': [],
            'implementation_terms': [],
            'key_concepts': [],
            'cross_references': [],
            'source_docs': []
        }
        
        try:
            for doc in retrieved_docs:
                content = doc.get('content', '').lower()
                filename = doc.get('metadata', {}).get('filename', 'Unknown')
                insights['source_docs'].append(filename)
                
                # Extract accounting standards (ASC, IFRS, etc.)
                standards = re.findall(r'\b(?:asc|ifrs|ias|fas)\s+\d+(?:[-\.\s]\d+)*\b', content)
                insights['standards_mentioned'].extend([std.upper() for std in standards])
                
                # Find examples in the document
                examples = re.findall(r'example\s+\d+[:\s][^\.]*\.', content)
                insights['examples_found'].extend([ex.strip()[:50] + "..." for ex in examples[:2]])
                
                # Detect tables and data references
                table_refs = re.findall(r'table\s+\d+|schedule\s+\d+|appendix\s+[a-z]', content)
                insights['tables_data'].extend([ref.title() for ref in table_refs])
                
                # Find implementation-related terms
                impl_patterns = [
                    r'transition\s+requirements?',
                    r'implementation\s+guidance',
                    r'effective\s+date',
                    r'adoption\s+process',
                    r'system\s+changes?'
                ]
                for pattern in impl_patterns:
                    matches = re.findall(pattern, content)
                    insights['implementation_terms'].extend([match.title() for match in matches])
                
                # Extract key financial/actuarial concepts
                concept_patterns = [
                    r'present\s+value',
                    r'discount\s+rate',
                    r'fair\s+value',
                    r'amortization',
                    r'reserve\s+adequacy',
                    r'capital\s+ratio',
                    r'risk\s+adjustment'
                ]
                for pattern in concept_patterns:
                    matches = re.findall(pattern, content)
                    insights['key_concepts'].extend([match.title() for match in matches])
                
                # Find cross-references to other standards/sections
                cross_refs = re.findall(r'see\s+(?:also\s+)?(?:asc|ifrs|section|paragraph)\s+\d+(?:[-\.\s]\d+)*', content)
                insights['cross_references'].extend([ref.upper() for ref in cross_refs])
            
            # Clean up and deduplicate
            for key in insights:
                if key != 'source_docs':
                    insights[key] = list(set(insights[key]))[:5]  # Max 5 unique items per category
            
            logger.info(f"📊 Content analysis extracted: {sum(len(v) if isinstance(v, list) else 0 for v in insights.values())} content elements")
            return insights
            
        except Exception as e:
            logger.error(f"Failed to analyze document content: {e}")
            return {'source_docs': [doc.get('metadata', {}).get('filename', 'Unknown') for doc in retrieved_docs]}

    def _extract_response_elements(self, response: str) -> str:
        """Extract specific elements mentioned in the response for targeted follow-ups"""
        elements = []
        response_lower = response.lower()
        
        try:
            # Extract specific financial metrics and dollar amounts
            financial_metrics = []
            dollar_amounts = re.findall(r'\$[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand)?', response_lower)
            percentages = re.findall(r'\d+\.\d+%', response_lower)
            
            if dollar_amounts:
                elements.append(f"Financial amounts: {', '.join(dollar_amounts[:3])}")
            if percentages:
                elements.append(f"Performance metrics: {', '.join(percentages[:3])}")
            
            # Extract specific business segments mentioned
            segments = re.findall(r'(?:u\.s\.|us|canada|emea|latin america)\s+(?:traditional|group|individual life|financial solutions)', response_lower)
            if segments:
                elements.append(f"Business segments: {', '.join(set([s.title() for s in segments[:3]]))}")
            
            # Extract time periods and quarters
            periods = re.findall(r'q[1-4]\s+(?:20\d{2}|results?)|(?:second|first|third|fourth)\s+quarter', response_lower)
            if periods:
                elements.append(f"Time periods discussed: {', '.join(set([p.upper() for p in periods]))}")
            
            # Extract company/entity names
            companies = re.findall(r'\b(?:rga|reinsurance group|equitable holdings?)\b', response_lower)
            if companies:
                elements.append(f"Companies mentioned: {', '.join(set([c.upper() for c in companies]))}")
            
            # Extract performance trends and outcomes
            performance_terms = []
            if 'favorable' in response_lower:
                favorable_items = re.findall(r'favorable\s+(?:\w+\s+){0,2}(?:experience|performance|results?|investment)', response_lower)
                performance_terms.extend(favorable_items)
            if 'unfavorable' in response_lower:
                unfavorable_items = re.findall(r'unfavorable\s+(?:\w+\s+){0,2}(?:experience|claims?|results?)', response_lower)
                performance_terms.extend(unfavorable_items)
            
            if performance_terms:
                elements.append(f"Performance trends: {', '.join(set(performance_terms[:3]))}")
            
            # Extract specific standards mentioned
            standards_mentioned = re.findall(r'\b(?:asc|ifrs|ias|fas)\s+\d+(?:[-\.\s]\d+)*\b', response_lower)
            if standards_mentioned:
                elements.append(f"Standards referenced: {', '.join(set([s.upper() for s in standards_mentioned]))}")
            
            # Extract key financial concepts
            financial_concepts = []
            concept_patterns = [
                r'adjusted\s+operating\s+income',
                r'return\s+on\s+equity',
                r'excess\s+capital',
                r'variable\s+investment\s+income',
                r'claims?\s+experience',
                r'premium\s+growth',
                r'reserve\s+adequacy',
                r'capital\s+deployment'
            ]
            for pattern in concept_patterns:
                matches = re.findall(pattern, response_lower)
                financial_concepts.extend([match.title() for match in matches])
            
            if financial_concepts:
                elements.append(f"Key concepts: {', '.join(set(financial_concepts[:3]))}")
            
            return "\n".join([f"- {element}" for element in elements]) if elements else "- General financial/business information provided"
            
        except Exception as e:
            logger.error(f"Failed to extract response elements: {e}")
            return "- Unable to analyze response elements"
    
    def _get_document_specifics(self, retrieved_docs: List[Dict]) -> str:
        """Extract specific information from the documents that were actually used"""
        specifics = []
        
        try:
            for i, doc in enumerate(retrieved_docs[:2], 1):
                filename = doc.get('metadata', {}).get('filename', 'Unknown')
                content = doc.get('content', '')[:400]  # First 400 chars
                
                specifics.append(f"Document {i} ({filename}):")
                specifics.append(f"Content excerpt: \"{content}...\"")
                
                # Extract specific elements from this document
                content_lower = content.lower()
                doc_elements = []
                
                # Standards in this specific document
                doc_standards = re.findall(r'\b(?:asc|ifrs|ias|fas)\s+\d+(?:[-\.\s]\d+)*\b', content_lower)
                if doc_standards:
                    doc_elements.append(f"Standards: {', '.join(set([s.upper() for s in doc_standards]))}")
                
                # Tables or data references
                tables = re.findall(r'table\s+\d+|schedule\s+\d+|appendix\s+[a-z]', content_lower)
                if tables:
                    doc_elements.append(f"Data references: {', '.join(set(tables))}")
                
                if doc_elements:
                    specifics.append(f"Specific elements: {'; '.join(doc_elements)}")
                
                specifics.append("")  # Add blank line between documents
            
            return "\n".join(specifics) if specifics else "No specific document content available"
            
        except Exception as e:
            logger.error(f"Failed to get document specifics: {e}")
            return "Unable to analyze document specifics"
    
    def _is_contextual_question(self, question: str, original_query: str, response: str) -> bool:
        """Validate that a follow-up question is contextual to the conversation"""
        question_lower = question.lower()
        response_lower = response.lower()
        query_lower = original_query.lower()
        
        # Generic phrases that indicate non-contextual questions
        generic_phrases = [
            'how do insurers typically',
            'what strategies do insurers',
            'how does claims impact profitability',
            'what are some strategies for',
            'how do companies typically handle',
            'what are the benefits of this approach',
            'how can organizations improve',
            'what factors influence profitability',
            'what are some common practices',
            'how can we improve our',
            'what does this mean for the industry',
            'what are the implications for',
            'how should companies approach',
            'what best practices should',
            'how does this compare to industry standards'
        ]
        
        # Check if question contains generic phrases
        if any(phrase in question_lower for phrase in generic_phrases):
            return False
        
        # Extract specific metrics, amounts, or data points from response
        response_specifics = []
        response_specifics.extend(re.findall(r'\$[\d,]+(?:\.\d+)?\s*(?:million|billion)', response_lower))
        response_specifics.extend(re.findall(r'\d+\.\d+%', response_lower))
        response_specifics.extend(re.findall(r'q[1-4]\s+(?:20\d{2}|results?)', response_lower))
        response_specifics.extend(re.findall(r'(?:u\.s\.|canada|emea)\s+(?:traditional|group)', response_lower))
        
        # Check if question references something mentioned in the response or query
        contextual_indicators = [
            # References specific financial amounts/metrics from response
            any(specific in question_lower for specific in response_specifics),
            # References specific elements from response/query
            any(word in question_lower for word in ['mentioned', 'explained', 'described', 'discussed', 'cited']),
            # References specific standards that appear in response
            bool(re.search(r'\b(?:asc|ifrs|ias|fas)\s+\d+', question_lower)) and bool(re.search(r'\b(?:asc|ifrs|ias|fas)\s+\d+', response_lower)),
            # References document-specific terms
            any(term in question_lower for term in ['document', 'section', 'example', 'table', 'schedule', 'presentation']),
            # References calculation or specific concept from response
            any(term in question_lower and term in response_lower for term in ['calculation', 'method', 'approach', 'guidance', 'component', 'breakdown']),
            # References specific company/business terms that appear in both
            any(term in question_lower and term in response_lower for term in ['rga', 'equitable', 'segment', 'traditional', 'group']),
            # References specific time periods or comparative language
            any(term in question_lower for term in ['compared to', 'other segments', 'different', 'breakdown', 'components']),
        ]
        
        # Must have at least one strong contextual indicator
        return any(contextual_indicators)
    
    def _generate_response_based_fallback(self, query: str, response: str, retrieved_docs: List[Dict]) -> List[str]:
        """Generate simple, contextual follow-ups based on the actual response content"""
        fallback_questions = []
        
        try:
            # Extract key terms from the response to create specific follow-ups
            response_lower = response.lower()
            
            # If response mentions specific standards, ask about related ones
            standards = re.findall(r'\b(?:asc|ifrs|ias|fas)\s+\d+(?:[-\.\s]\d+)*\b', response_lower)
            if standards:
                fallback_questions.append(f"How does {standards[0].upper()} relate to other accounting standards?")
            
            # If response mentions calculations, ask for details
            if any(word in response_lower for word in ['calculate', 'computation', 'formula']):
                fallback_questions.append("Can you walk through the calculation steps in more detail?")
            
            # If response mentions examples, ask for more
            if 'example' in response_lower:
                fallback_questions.append("Can you provide another example of this concept?")
            
            # If response mentions implementation, ask about challenges
            if any(word in response_lower for word in ['implement', 'apply', 'adopt']):
                fallback_questions.append("What are the main challenges in implementing this?")
            
            # Document-specific fallback
            if retrieved_docs:
                filename = retrieved_docs[0].get('metadata', {}).get('filename', '')
                if filename:
                    fallback_questions.append(f"What else does the {filename} document cover on this topic?")
            
            # If no specific fallbacks, use minimal contextual ones
            if not fallback_questions:
                fallback_questions = [
                    "Can you clarify any part of this explanation?",
                    "Are there related concepts I should understand?",
                    "How would this apply in practice?"
                ]
            
            return fallback_questions[:3]
            
        except Exception as e:
            logger.error(f"Failed to generate fallback questions: {e}")
            return [
                "Can you elaborate on this topic?",
                "What are the key takeaways?",
                "How does this relate to our previous discussion?"
            ]
    
    async def _classify_query_topic(self, query: str) -> Dict[str, Any]:
        """Classify whether the query is within AAIRE's domain expertise"""
        
        logger.info(f"📋 Starting topic classification for: '{query[:30]}...'")
        
        # Define relevant financial/insurance/accounting domains
        relevant_keywords = {
            'financial': [
                'financial', 'finance', 'revenue', 'profit', 'loss', 'earnings', 'income', 'expense',
                'assets', 'liabilities', 'equity', 'balance sheet', 'cash flow', 'statement',
                'budget', 'forecast', 'valuation', 'investment', 'portfolio', 'returns',
                'capital', 'funding', 'financing', 'debt', 'credit', 'loan', 'mortgage'
            ],
            'accounting': [
                'accounting', 'gaap', 'ifrs', 'asc', 'fas', 'ias', 'standard', 'compliance',
                'audit', 'auditing', 'journal', 'ledger', 'depreciation', 'amortization',
                'accrual', 'recognition', 'measurement', 'disclosure', 'reporting',
                'consolidation', 'segment', 'fair value', 'impairment', 'tax'
            ],
            'insurance': [
                'insurance', 'insurer', 'policy', 'premium', 'claim', 'coverage', 'underwriting',
                'reinsurance', 'actuarial', 'risk', 'reserve', 'liability', 'benefit',
                'annuity', 'life insurance', 'health insurance', 'property', 'casualty',
                'solvency', 'capital adequacy', 'licat', 'regulatory'
            ],
            'banking': [
                'bank', 'banking', 'deposit', 'withdrawal', 'account', 'lending', 'borrowing',
                'interest rate', 'mortgage', 'loan', 'credit', 'debit', 'payment',
                'financial institution', 'federal reserve', 'monetary policy', 'currency'
            ],
            'investment': [
                'investment', 'investing', 'stock', 'bond', 'security', 'portfolio',
                'mutual fund', 'etf', 'dividend', 'yield', 'return', 'risk', 'volatility',
                'market', 'trading', 'hedge fund', 'private equity', 'venture capital'
            ],
            'mathematical': [
                'calculation', 'formula', 'equation', 'mathematical', 'statistics', 'probability',
                'model', 'modeling', 'quantitative', 'analysis', 'ratio', 'percentage',
                'present value', 'future value', 'discount rate', 'compound', 'regression'
            ],
            'economics': [
                'economic', 'economics', 'inflation', 'deflation', 'gdp', 'recession',
                'growth', 'unemployment', 'monetary', 'fiscal', 'policy', 'market',
                'supply', 'demand', 'price', 'cost', 'microeconomic', 'macroeconomic'
            ]
        }
        
        # Quick keyword-based check first
        query_lower = query.lower()
        has_relevant_keywords = False
        
        for domain, keywords in relevant_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                has_relevant_keywords = True
                break
        
        # If obvious keywords found, likely relevant
        if has_relevant_keywords:
            return {'is_relevant': True, 'confidence': 0.9}
        
        # Use AI classification for ambiguous cases
        classification_prompt = f"""Determine if this question is relevant to AAIRE, an AI assistant specialized in:
- Financial analysis and reporting
- Accounting standards (GAAP, IFRS, ASC, etc.)
- Insurance and actuarial topics
- Banking and investment concepts  
- Mathematical and statistical analysis
- Economics and business finance

Question: "{query}"

Is this question within AAIRE's expertise domain?

Respond with either:
"RELEVANT" - if the question relates to finance, accounting, insurance, banking, investments, mathematics/statistics, or economics
"NOT_RELEVANT" - if the question is about other topics like sports, entertainment, cooking, travel, personal relationships, general knowledge, etc.

Answer:"""

        try:
            response = self.llm.complete(classification_prompt)
            classification = response.text.strip().upper()
            
            if "RELEVANT" in classification:
                return {'is_relevant': True, 'confidence': 0.8}
            else:
                polite_responses = [
                    "I'm AAIRE, a specialized AI assistant focused on financial, accounting, insurance, and actuarial topics. I'd be happy to help you with questions related to these areas instead!",
                    "I specialize in financial, accounting, insurance, banking, and related business topics. Could you ask me something within these domains? I'd love to help!",
                    "As an insurance and financial industry specialist, I'm designed to assist with accounting standards, financial analysis, insurance topics, and related mathematical concepts. How can I help you with these areas?",
                    "I'm focused on providing expert assistance with financial, accounting, actuarial, and insurance-related questions. Is there something in these areas I can help you with instead?"
                ]
                
                import random
                selected_response = random.choice(polite_responses)
                
                return {
                    'is_relevant': False, 
                    'polite_response': selected_response,
                    'confidence': 0.8
                }
                
        except Exception as e:
            logger.error(f"Failed to classify query topic: {e}")
            # Default to allowing the query if classification fails
            return {'is_relevant': True, 'confidence': 0.3}

    def _expand_query(self, query: str) -> str:
        """Expand general queries with specific domain terms for better retrieval"""
        query_lower = query.lower()
        
        # Domain-specific term mappings for insurance and accounting
        expansion_mappings = {
            # Capital and financial health terms
            'capital health': 'capital health LICAT ratio core ratio total ratio capital adequacy',
            'company capital': 'company capital LICAT ratio core ratio total ratio regulatory capital',
            'assess capital': 'assess capital LICAT ratio core ratio total ratio capital adequacy',
            'financial strength': 'financial strength LICAT ratio core ratio total ratio capital adequacy',
            'capital adequacy': 'capital adequacy LICAT ratio core ratio total ratio regulatory capital',
            
            # Insurance specific expansions
            'insurance': 'insurance LICAT OSFI regulatory capital solvency',
            'regulatory': 'regulatory OSFI LICAT compliance capital requirements',
            'solvency': 'solvency LICAT ratio capital adequacy regulatory capital',
            
            # Accounting standard expansions
            'accounting': 'accounting GAAP IFRS standards disclosure requirements',
            'financial reporting': 'financial reporting GAAP IFRS disclosure standards',
            'compliance': 'compliance regulatory requirements OSFI GAAP IFRS',
            
            # Risk management expansions
            'risk': 'risk management capital risk regulatory risk operational risk',
            'management': 'management risk management capital management regulatory management'
        }
        
        # Apply expansions
        expanded_query = query
        for general_term, expansion in expansion_mappings.items():
            if general_term in query_lower:
                # Add specific terms to the query
                specific_terms = expansion.replace(general_term, '').strip()
                if specific_terms:
                    expanded_query = f"{query} {specific_terms}"
                break
        
        # Log expansion for debugging
        if expanded_query != query:
            logger.info("Query expanded", 
                       original=query, 
                       expanded=expanded_query)
        
        return expanded_query
    
    def _calculate_quality_metrics(self, query: str, response: str, retrieved_docs: List[Dict], citations: List[Dict]) -> Dict[str, float]:
        """Calculate automated quality metrics for the response"""
        try:
            # Initialize metrics
            metrics = {}
            
            # 1. Citation Coverage - How well the response is supported by sources
            if retrieved_docs:
                # Count how many retrieved docs are actually cited
                cited_docs = len(citations) if citations else 0
                total_docs = len(retrieved_docs)
                metrics['citation_coverage'] = cited_docs / total_docs if total_docs > 0 else 0.0
            else:
                metrics['citation_coverage'] = 0.0
            
            # 2. Response Length Appropriateness - Not too short, not too long
            response_words = len(response.split())
            if response_words < 20:
                metrics['length_score'] = 0.3  # Too short
            elif response_words > 500:
                metrics['length_score'] = 0.7  # Might be too long
            else:
                metrics['length_score'] = 1.0  # Appropriate length
            
            # 3. Query-Response Relevance - Basic keyword overlap
            query_words = set(query.lower().split())
            response_words_set = set(response.lower().split())
            
            # Remove common words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can'}
            query_keywords = query_words - stop_words
            response_keywords = response_words_set - stop_words
            
            if query_keywords:
                overlap = len(query_keywords & response_keywords)
                metrics['keyword_relevance'] = overlap / len(query_keywords)
            else:
                metrics['keyword_relevance'] = 0.5  # Neutral if no keywords
            
            # 4. Source Quality - Average similarity scores of retrieved docs
            if retrieved_docs:
                scores = [doc.get('score', 0.0) for doc in retrieved_docs]
                metrics['source_quality'] = sum(scores) / len(scores) if scores else 0.0
            else:
                metrics['source_quality'] = 0.0
            
            # 5. Response Completeness - Basic heuristics
            has_structured_response = any(marker in response for marker in ['1.', '2.', '•', '-', 'Steps:', 'Requirements:'])
            has_specific_details = any(term in response.lower() for term in ['ratio', 'percentage', '%', '$', 'requirement', 'standard', 'regulation'])
            
            completeness_score = 0.5  # Base score
            if has_structured_response:
                completeness_score += 0.25
            if has_specific_details:
                completeness_score += 0.25
            
            metrics['completeness'] = min(completeness_score, 1.0)
            
            # 6. Overall Quality Score (weighted average)
            weights = {
                'citation_coverage': 0.25,
                'length_score': 0.15,
                'keyword_relevance': 0.25,
                'source_quality': 0.20,
                'completeness': 0.15
            }
            
            overall_score = sum(metrics[key] * weights[key] for key in weights if key in metrics)
            metrics['overall_quality'] = overall_score
            
            # Log quality metrics for monitoring
            logger.info("Response quality metrics calculated",
                       overall_quality=overall_score,
                       citation_coverage=metrics['citation_coverage'],
                       keyword_relevance=metrics['keyword_relevance'],
                       source_quality=metrics['source_quality'])
            
            return metrics
            
        except Exception as e:
            logger.error("Failed to calculate quality metrics", error=str(e))
            return {
                "citation_coverage": 0.0,
                "length_score": 0.5,
                "keyword_relevance": 0.5,
                "source_quality": 0.0,
                "completeness": 0.5,
                "overall_quality": 0.3
            }
    
    def _get_similarity_threshold(self, query: str) -> float:
        """Determine optimal similarity threshold based on query type"""
        query_lower = query.lower()
        
        # Use stricter threshold for specific/critical queries that need precision
        specific_indicators = [
            'specific', 'exact', 'precise', 'what is the', 'define',
            'calculation', 'formula', 'ratio', 'compliance requirement',
            'regulatory requirement', 'standard requires', 'rule states',
            'policy says', 'according to', 'as per', 'mandate'
        ]
        
        # Use relaxed threshold for general/exploratory queries that need comprehensiveness
        general_indicators = [
            'how to', 'what are ways', 'assess', 'evaluate', 'overview',
            'explain', 'understand', 'approach', 'methods', 'strategies',
            'best practices', 'considerations', 'factors', 'guidance',
            'help me', 'show me how'
        ]
        
        # Check for specific indicators first
        if any(indicator in query_lower for indicator in specific_indicators):
            threshold = 0.75  # Stricter for precision
            reason = "specific query"
        elif any(indicator in query_lower for indicator in general_indicators):
            threshold = 0.65  # Relaxed for comprehensiveness  
            reason = "general query"
        else:
            threshold = 0.70  # Balanced middle ground
            reason = "neutral query"
        
        logger.info("Adaptive threshold selected", 
                   query=query[:50] + "..." if len(query) > 50 else query,
                   threshold=threshold,
                   reason=reason)
        
        return threshold
    
    def _get_document_limit(self, query: str) -> int:
        """Dynamically determine document limit based on query complexity"""
        
        # Get config
        config = self.config.get('retrieval_config', {})
        base_limit = config.get('base_document_limit', 25)
        standard_limit = config.get('standard_document_limit', 35)  
        complex_limit = config.get('complex_document_limit', 45)
        max_limit = config.get('max_document_limit', 60)
        
        # Analyze query complexity
        query_lower = query.lower()
        words = query_lower.split()
        word_count = len(words)
        
        # Initialize complexity score
        complexity_score = 0
        
        # Word count indicator
        if word_count > 15:
            complexity_score += 2  # Very long query
        elif word_count > 10:
            complexity_score += 1  # Long query
        
        # Question complexity indicators  
        comprehensive_words = ['how', 'why', 'what', 'explain', 'describe', 'discuss']
        if any(word in words for word in comprehensive_words):
            complexity_score += 1
        
        # Technical procedure indicators
        technical_words = ['calculate', 'determine', 'implement', 'process', 'analyze', 'evaluate']
        if any(word in words for word in technical_words):
            complexity_score += 1
        
        # Multi-part query indicators
        multi_indicators = ['and', 'also', 'additionally', 'furthermore', 'moreover', 'including']
        if sum(1 for word in multi_indicators if word in words) >= 2:
            complexity_score += 1  # Multiple aspects to address
        
        # Regulatory/compliance complexity (generic patterns)
        import re
        if len(re.findall(r'\b[A-Z]{2,}\b', query)) > 2:  # Multiple acronyms
            complexity_score += 1
        
        # Choose limit based on complexity score
        if complexity_score >= 4:
            final_limit = min(complex_limit, max_limit)  # Complex: 45 docs
            complexity_name = "Complex"
        elif complexity_score >= 2:
            final_limit = min(standard_limit, max_limit)  # Standard: 35 docs  
            complexity_name = "Standard"
        else:
            final_limit = min(base_limit, max_limit)      # Simple: 25 docs
            complexity_name = "Simple"
        
        logger.info(f"Dynamic document limit: {final_limit} ({complexity_name} query, complexity score: {complexity_score})")
        
        return final_limit
    
    def _is_general_knowledge_query(self, query: str) -> bool:
        """Check if query is asking for general knowledge vs specific document content"""
        query_lower = query.lower()
        
        # First check for document-specific indicators (these override general patterns)
        document_indicators = [
            r'\bour company\b',
            r'\bthe uploaded\b',
            r'\bthe document\b',
            r'\bin the document\b',
            r'\bshow me\b',
            r'\bfind\b.*\bin\b',
            r'\banalyze\b',
            r'\bspecific\b.*\bmentioned\b',
            r'\bpolicy\b',
            r'\bprocedure\b',
            r'\bin the.*image\b',  # "in the chatgpt image"
            r'\bfrom the.*image\b',  # "from the image"
            r'\bthe.*chart\b',  # "the revenue chart"
            r'\buploaded.*image\b',  # "uploaded image"
            r'\bASC\s+\d{3}-\d{2}-\d{2}-\d{2}\b',  # ASC codes like "ASC 255-10-50-51"
            r'\bFASB\b',  # FASB references
            r'\bGAAP\b',  # GAAP references
            r'\bIFRS\b'   # IFRS references
        ]
        
        import re
        for pattern in document_indicators:
            if re.search(pattern, query_lower):
                return False  # Document-specific query
        
        # Common general knowledge question patterns (only if no document indicators)
        general_patterns = [
            r'^\s*what is\s+[a-z\s]+\??$',  # Simple "what is X?" questions
            r'^\s*define\s+[a-z\s]+\??$',   # Simple "define X" questions
            r'^\s*what\s+does\s+[a-z\s]+\s+mean\??$',  # Simple "what does X mean" questions
            r'^\s*what\s+are\s+the\s+types\s+of\s+[a-z\s?]+\??$',  # "what are the types of X" questions
            r'^\s*how\s+does\s+[a-z\s]+\s+work\??$'  # Simple "how does X work" questions
        ]
        
        for pattern in general_patterns:
            if re.search(pattern, query_lower):
                return True
                
        return False
    
    def _extract_citations(self, retrieved_docs: List[Dict], query: str = "") -> List[Dict[str, Any]]:
        """Extract citation information - if document was used for response, it should be cited"""
        citations = []
        
        if not retrieved_docs:
            logger.warning("❌ NO CITATIONS GENERATED - no retrieved documents")
            return citations
        
        logger.info(f"🎯 CITATION EXTRACTION: Processing {len(retrieved_docs)} documents")
        logger.info("📋 All retrieved documents:")
        for i, doc in enumerate(retrieved_docs):
            filename = doc['metadata'].get('filename', 'Unknown')
            relevance_score = doc.get('relevance_score', doc.get('score', 0.0))
            logger.info(f"  Doc {i+1}: {filename} - relevance: {relevance_score:.3f}")
        
        # CORE PRINCIPLE: If a document contributed to the response, it deserves citation
        # Use the top documents that were actually used for response generation
        max_citations = 3
        
        for i, doc in enumerate(retrieved_docs[:max_citations]):
            relevance_score = doc.get('relevance_score', doc.get('score', 0.0))
            filename = doc['metadata'].get('filename', 'Unknown')
            
            logger.info(f"📄 Processing citation {i+1}: {filename}, relevance_score={relevance_score:.3f}")
            
            # Simple quality filter - only skip obviously bad documents
            if relevance_score < 0.1:  # Very permissive threshold
                logger.info(f"❌ SKIPPING - Extremely low relevance: {relevance_score:.3f}")
                continue
            
            # Skip obvious generic responses only
            content_lower = doc.get('content', '').lower()
            if any(phrase in content_lower for phrase in [
                'how can i assist you today',
                'feel free to share',
                'what can i help you with'
            ]):
                logger.info(f"❌ SKIPPING - Generic assistant response")
                continue
                
            # Get filename for source
            filename = doc['metadata'].get('filename', 'Unknown')
            
            # Extract page information if available
            page_info = ""
            if 'page' in doc['metadata']:
                page_info = f", Page {doc['metadata']['page']}"
            elif 'page_label' in doc['metadata']:
                page_info = f", Page {doc['metadata']['page_label']}"
            elif hasattr(doc, 'node_id') and 'page_' in str(doc.get('node_id', '')):
                # Extract page from node_id like "page_1_chunk_2"
                try:
                    page_num = str(doc.get('node_id', '')).split('page_')[1].split('_')[0]
                    page_info = f", Page {page_num}"
                except:
                    pass
            
            # Check if content contains page references from shape-aware extraction
            content = doc.get('content', '')
            if 'Source: Page' in content:
                # Extract page number from content like "Source: Page 2, cluster_1_page_2"
                import re
                page_match = re.search(r'Source: Page (\d+)', content)
                if page_match:
                    page_info = f", Page {page_match.group(1)}"
            
            citation = {
                "id": len(citations) + 1,  # Use actual citation count, not doc index
                "text": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                "source": f"{filename}{page_info}",
                "source_type": doc['source_type'],
                "confidence": round(relevance_score, 3)  # Use relevance score instead of original score
            }
            
            # Add additional metadata if available
            if 'page' in doc['metadata']:
                citation['page'] = doc['metadata']['page']
            if 'section' in doc['metadata']:
                citation['section'] = doc['metadata']['section']
            if 'standard' in doc['metadata']:
                citation['standard'] = doc['metadata']['standard']
                
            citations.append(citation)
            logger.info(f"✅ ADDED citation from: {filename} (relevance: {relevance_score:.3f})")
        
        logger.info(f"🎯 FINAL RESULT: Generated {len(citations)} citations from {len(retrieved_docs)} retrieved documents")
        
        # DEBUG: Print citation details for troubleshooting
        if citations:
            for i, citation in enumerate(citations):
                logger.info(f"Citation {i+1}: source={citation.get('source')}, confidence={citation.get('confidence')}")
        else:
            logger.warning("❌ NO CITATIONS GENERATED - this explains missing citation display")
        
        return citations
    
    def _infer_document_domain(self, filename: str, content: str) -> str:
        """Dynamically infer document domain from filename and content"""
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        # Insurance/Regulatory domain
        if any(term in filename_lower for term in ['licat', 'insurance', 'regulatory', 'capital']):
            return 'insurance'
        
        # Accounting standards domain
        if any(term in filename_lower for term in ['pwc', 'asc', 'ifrs', 'gaap']) or \
           any(term in content_lower for term in ['asc ', 'ifrs', 'accounting standard']):
            return 'accounting_standards'
        
        # Foreign currency domain
        if any(term in filename_lower for term in ['foreign', 'currency', 'fx']) or \
           any(term in content_lower for term in ['foreign currency', 'exchange rate']):
            return 'foreign_currency'
        
        # Actuarial domain
        if any(term in filename_lower for term in ['actuarial', 'valuation', 'reserves']) or \
           any(term in content_lower for term in ['actuarial', 'present value', 'discount rate']):
            return 'actuarial'
        
        return 'general'
    
    def _check_domain_compatibility(self, query_domain: str, doc_domain: str, doc_filename: str) -> Dict[str, Any]:
        """Check if query domain is compatible with document domain"""
        
        # Define domain compatibility matrix - make more permissive for debugging
        compatibility_matrix = {
            'accounting_standards': ['accounting_standards', 'foreign_currency', 'general', 'accounting'],
            'foreign_currency': ['foreign_currency', 'accounting_standards', 'general', 'accounting'],
            'insurance': ['insurance', 'actuarial', 'general'],
            'actuarial': ['actuarial', 'insurance', 'general'],
            'accounting': ['accounting', 'accounting_standards', 'foreign_currency', 'general'],
            'general': ['general', 'accounting_standards', 'foreign_currency', 'insurance', 'actuarial', 'accounting'],
            None: ['general', 'accounting_standards', 'foreign_currency', 'insurance', 'actuarial', 'accounting']  # Handle None domain
        }
        
        # Get compatible domains for query
        compatible_domains = compatibility_matrix.get(query_domain, ['general'])
        
        # Check compatibility
        if doc_domain in compatible_domains:
            return {
                'compatible': True,
                'reason': f"Query domain '{query_domain}' compatible with document domain '{doc_domain}'"
            }
        else:
            return {
                'compatible': False,
                'reason': f"Query domain '{query_domain}' incompatible with document domain '{doc_domain}' (file: {doc_filename})"
            }
    
    def _calculate_confidence(self, retrieved_docs: List[Dict], response: str) -> float:
        """Calculate confidence score for the response"""
        if not retrieved_docs:
            return 0.0
        
        # Average similarity score of top documents
        top_scores = [doc['score'] for doc in retrieved_docs[:3]]
        avg_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
        
        # Adjust based on number of relevant documents
        doc_count_factor = min(len(retrieved_docs) / 3.0, 1.0)
        
        # Basic confidence calculation
        confidence = avg_score * doc_count_factor
        
        return round(confidence, 3)
    
    def _get_cache_key(self, query: str, filters: Optional[Dict]) -> str:
        """Generate cache key for query that includes document state"""
        import hashlib
        
        # Include document state in cache key so it invalidates when documents change
        try:
            # Get document count and last modification from vector store
            doc_count = 0
            last_modified = "unknown"
            
            if hasattr(self, 'vector_store') and self.vector_store:
                try:
                    # Try to get basic stats from vector store
                    if hasattr(self.vector_store, 'client'):
                        collection_info = self.vector_store.client.get_collection(self.index_name)
                        if collection_info:
                            doc_count = collection_info.vectors_count or 0
                except:
                    # Fallback to zero if we can't get collection info
                    doc_count = 0
            
            cache_data = {
                'query': query,
                'filters': filters or {},
                'doc_count': doc_count,  # Cache invalidates when document count changes
                'version': '2.0'  # Manual version bump to invalidate old cache entries
            }
        except Exception as e:
            # Fallback cache key if we can't get document state
            cache_data = {
                'query': query,
                'filters': filters or {},
                'version': '2.0'  # This will invalidate all old cache entries
            }
        
        return hashlib.md5(str(cache_data).encode()).hexdigest()
    
    def _serialize_response(self, response: RAGResponse) -> str:
        """Serialize response for caching"""
        import json
        return json.dumps({
            'answer': response.answer,
            'citations': response.citations,
            'confidence': response.confidence,
            'follow_up_questions': response.follow_up_questions
        })
    
    def _deserialize_response(self, cached_data: str, session_id: str) -> RAGResponse:
        """Deserialize cached response"""
        import json
        data = json.loads(cached_data)
        return RAGResponse(
            answer=data['answer'],
            citations=data['citations'],
            confidence=data['confidence'],
            session_id=session_id,
            follow_up_questions=data.get('follow_up_questions', []),
            quality_metrics=data.get('quality_metrics', {})
        )
    
    def _remove_citations_from_response(self, response: str) -> str:
        """Remove any citation numbers [1], [2], etc. from response text"""
        import re
        # Remove citation patterns like [1], [2], [1][2][3], etc.
        cleaned_response = re.sub(r'\[[\d\s,]+\]', '', response)
        # Clean up any double spaces left behind
        cleaned_response = re.sub(r'\s+', ' ', cleaned_response)
        return cleaned_response.strip()
    
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
                
                logger.info("Successfully cleared all documents from Qdrant")
                return {"status": "success", "message": "All documents cleared", "method": "qdrant_recreate"}
            else:
                # For local storage, recreate the index
                self._init_local_index()
                logger.info("Successfully cleared all documents from local storage")
                return {"status": "success", "message": "All documents cleared", "method": "local_recreate"}
                
        except Exception as e:
            logger.error("Failed to clear all documents", error=str(e))
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
            logger.error(f"Failed to delete document from vector store", error=str(e), job_id=job_id)
            return {
                "status": "error",
                "error": str(e),
                "job_id": job_id
            }
    
    async def _clear_cache_pattern(self, pattern: str) -> int:
        """Clear cache entries matching a specific pattern"""
        try:
            if not self.cache:
                return 0
            
            cleared_count = 0
            cache_pattern = f"query_cache:*{pattern}*"
            
            # Use scan_iter to find matching keys
            matching_keys = []
            for key in self.cache.scan_iter(match=cache_pattern):
                matching_keys.append(key)
            
            # Delete matching keys
            if matching_keys:
                cleared_count = self.cache.delete(*matching_keys)
                logger.info(f"Cleared {cleared_count} cache entries matching pattern: {pattern}")
            
            return cleared_count
            
        except Exception as e:
            logger.error(f"Failed to clear cache pattern {pattern}: {str(e)}")
            return 0
    
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
            from .enhanced_query_handler import EnhancedQueryHandler
            from .intelligent_extractor import IntelligentDocumentExtractor
            
            # Initialize components
            query_handler = EnhancedQueryHandler(self.llm)
            intelligent_extractor = IntelligentDocumentExtractor(self.llm)
            
            logger.info("Enhanced query processing started", 
                       query=query[:100], 
                       session_id=session_id)
            
            # Step 1: Analyze query to determine processing strategy
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
    
    def _fix_citation_format(self, response: str, retrieved_docs: List[Dict]) -> str:
        """Post-process response to replace [1], [2] citations with proper source names and page numbers"""
        if not retrieved_docs:
            return response
        
        import re
        
        # Create mapping of citation numbers to proper source names
        citation_map = {}
        for i, doc in enumerate(retrieved_docs[:10]):  # Handle up to 10 citations
            citation_num = i + 1
            filename = doc['metadata'].get('filename', 'Unknown')
            
            # Try to extract page number
            page_info = ""
            if 'page' in doc['metadata']:
                page_info = f", Page {doc['metadata']['page']}"
            elif 'estimated_page' in doc['metadata']:
                page_info = f", Page {doc['metadata']['estimated_page']}"
            elif 'Source: Page' in doc.get('content', ''):
                page_match = re.search(r'Source: Page (\d+)', doc.get('content', ''))
                if page_match:
                    page_info = f", Page {page_match.group(1)}"
            
            # Create proper citation
            proper_citation = f"({filename}{page_info})"
            citation_map[f"[{citation_num}]"] = proper_citation
        
        # Replace all [1], [2], etc. with proper citations
        fixed_response = response
        for old_citation, new_citation in citation_map.items():
            fixed_response = fixed_response.replace(old_citation, new_citation)
        
        # Also handle "Document X" references
        for i in range(1, 11):  # Handle Document 1-10
            doc_ref = f"Document {i}"
            if doc_ref in fixed_response and i <= len(retrieved_docs):
                doc = retrieved_docs[i-1]
                filename = doc['metadata'].get('filename', 'Unknown')
                page_info = ""
                if 'page' in doc['metadata']:
                    page_info = f", Page {doc['metadata']['page']}"
                elif 'estimated_page' in doc['metadata']:
                    page_info = f", Page {doc['metadata']['estimated_page']}"
                
                proper_ref = f"{filename}{page_info}"
                fixed_response = fixed_response.replace(doc_ref, proper_ref)
        
        return fixed_response
    
    def _basic_text_cleanup(self, response: str) -> str:
        """Basic text cleanup - minimal whitespace normalization only"""
        import re
        
        # Only do minimal cleanup - no hardcoded patterns
        # Clean up excessive newlines (but keep double newlines for spacing)
        cleaned = re.sub(r'\n{4,}', '\n\n\n', response)
        
        # Clean up whitespace at start and end
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _validate_and_fix_formatting(self, response: str) -> str:
        """Validate formatting and fix common issues using LLM-based correction"""
        try:
            # Check for common formatting issues
            issues = []
            
            # Check for missing line breaks before numbered items
            if '**1.' in response and not '\n\n**1.' in response and not response.startswith('**1.'):
                issues.append("Missing line breaks before numbered items")
            
            # Check for run-on formatting
            if '1. ' in response and '2. ' in response:
                # Check if numbered items are on same line
                lines = response.split('\n')
                for line in lines:
                    if '1. ' in line and ('2. ' in line or '3. ' in line):
                        issues.append("Numbered items running together on same line")
                        break
            
            # Check for "Where:" without line break
            if 'Where: -' in response or 'Where:-' in response:
                issues.append("Missing line break after 'Where:'")
                
            # Check for bold formatting artifacts
            if '**includes' in response or '**excludes' in response:
                issues.append("Bold formatting artifacts in text")
            
            # If issues found, apply LLM-based correction
            if issues:
                logger.info(f"🔧 Formatting issues detected: {issues}")
                return self._apply_llm_formatting_fix(response, issues)
            else:
                logger.info("✅ Response formatting validated - no major issues found")
                return response
                
        except Exception as e:
            logger.warning(f"Formatting validation failed: {e}")
            return response
    
    def _apply_llm_formatting_fix(self, response: str, detected_issues: list) -> str:
        """Apply LLM-based formatting correction for detected issues"""
        
        issues_description = ', '.join(detected_issues)
        
        correction_prompt = f"""Fix the formatting issues in this insurance/actuarial response.

DETECTED ISSUES: {issues_description}

SPECIFIC FIXES NEEDED:
1. Put blank line BEFORE each **numbered item** (like **1.** or **2.**)
2. Separate numbered list items (1. 2. 3.) onto different lines  
3. Add line break after "Where:" before definitions
4. Fix bold formatting artifacts like **includes or **excludes
5. Keep ALL formulas and mathematical content intact

EXAMPLE OF CORRECT FORMAT:
**Section Title**

Regular paragraph text here.

**1.** First numbered point

**2.** Second numbered point

Formula: NPR = APV(Benefits) - APV(Premiums)

Where:
- NPR = Net Premium Reserve
- APV = Actuarial Present Value

Original text to fix:
{response}

Provide the corrected version:"""
        
        try:
            corrected = self.llm.complete(correction_prompt, temperature=0.1)
            logger.info("🔧 Applied LLM-based formatting correction")
            return corrected.text.strip()
        except Exception as e:
            logger.warning(f"LLM formatting correction failed: {e}")
            return response
    
    def _generate_structured_response(self, query: str, context: str) -> str:
        """Generate response with structured JSON output for consistent formatting"""
        
        structured_prompt = f"""You are AAIRE, an insurance accounting expert.
        
Question: {query}

Context: {context}

Generate a response in this EXACT JSON structure (be precise with formulas):
{{
    "summary": "Brief 2-3 sentence overview",
    "sections": [
        {{
            "title": "Section heading",
            "content": "Detailed explanation",
            "formulas": [
                {{
                    "name": "Reserve Calculation",
                    "latex": "R_t = PV(benefits_t) - PV(premiums_t)",
                    "readable": "R(t) = PV(benefits at time t) - PV(premiums at time t)",
                    "components": {{
                        "R_t": "Reserve at time t",
                        "PV": "Present Value function"
                    }}
                }}
            ],
            "numbered_items": ["Step 1 description", "Step 2 description"]
        }}
    ],
    "key_values": {{
        "rates": ["90% confidence level", "2.5% discount rate"],
        "references": ["ASC 944-40-25-25", "IFRS 17.32"]
    }}
}}

Ensure ALL mathematical notation is included in both latex and readable formats.
Only return the JSON - no other text."""
        
        try:
            response = self.llm.complete(structured_prompt, temperature=0.2)
            structured_data = json.loads(response.text.strip())
            logger.info("✅ Successfully generated structured JSON response")
            return self._structured_to_markdown(structured_data)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            # Fallback to existing correction method
            return self._apply_llm_formatting_fix(response.text, ["JSON structure invalid"])
        except Exception as e:
            logger.warning(f"Structured generation failed: {e}")
            raise
    
    def _structured_to_markdown(self, data: Dict) -> str:
        """Convert structured JSON to formatted markdown with proper formula handling"""
        
        output = []
        
        # Summary
        if 'summary' in data:
            output.append(f"{data['summary']}\n")
        
        # Process each section
        for section in data.get('sections', []):
            # Section title
            output.append(f"\n**{section['title']}**\n")
            
            # Content
            if 'content' in section:
                output.append(f"{section['content']}\n")
            
            # Formulas - with special handling
            if 'formulas' in section:
                output.append("\n**Key Formulas:**\n")
                for formula in section['formulas']:
                    # Use readable format as primary
                    output.append(f"\n• {formula['name']}:\n")
                    output.append(f"  {formula['readable']}\n")
                    
                    # Add component definitions if present
                    if 'components' in formula:
                        output.append("  Where:\n")
                        for var, desc in formula['components'].items():
                            output.append(f"  - {var} = {desc}\n")
            
            # Numbered items with proper spacing
            if 'numbered_items' in section:
                output.append("\n")
                for i, item in enumerate(section['numbered_items'], 1):
                    output.append(f"**{i}.** {item}\n\n")
        
        # Key values
        if 'key_values' in data:
            output.append("\n**Important Values:**\n")
            for category, values in data['key_values'].items():
                for value in values:
                    output.append(f"• {value}\n")
        
        return ''.join(output)
    
    def _validate_formula_formatting(self, response: str) -> bool:
        """Check if formulas are properly formatted"""
        
        validation_prompt = f"""Check if this text has properly formatted formulas:

{response[:1000]}

Look for:
1. LaTeX notation that wasn't converted (\\sum, \\times, _{{subscript}})
2. Unreadable mathematical expressions
3. Complex subscripts not converted to parentheses

Reply with just: VALID or NEEDS_FIXING"""
        
        try:
            result = self.llm.complete(validation_prompt, temperature=0)
            return "VALID" in result.text.upper()
        except Exception as e:
            logger.warning(f"Formula validation failed: {e}")
            return True  # Default to assuming it's valid
    
    def _normalize_spacing(self, response: str) -> str:
        """Enhanced cleanup to fix persistent formatting issues: bold headers, line breaks, formulas"""
        import re
        
        
        # Step 1: Preserve and protect formulas/mathematical content before cleanup
        formula_patterns = [
            (r'(\$[\d,]+(?:\.\d+)?)', r'FORMULA_DOLLAR_\1_END'),  # Dollar amounts
            (r'(\d+%)', r'FORMULA_PERCENT_\1_END'),  # Percentages  
            (r'(\d+\.?\d*\s*[×*]\s*\d+\.?\d*)', r'FORMULA_MULT_\1_END'),  # Multiplication
            (r'(NPV|PV|FV|PMT|RATE|NPER)', r'FORMULA_FUNC_\1_END'),  # Financial functions
            (r'(\w+\s*=\s*[\w\d\s\+\-\*/\(\)\.]+)', r'FORMULA_EQ_\1_END'),  # Equations
            (r'([A-Z]\([^)]+\))', r'FORMULA_NOTATION_\1_END'),  # Function notation like E(x+t)
        ]
        
        formula_map = {}
        result = response
        
        for i, (pattern, template) in enumerate(formula_patterns):
            matches = re.findall(pattern, result)
            for j, match in enumerate(matches):
                placeholder_key = f'FORMULA_{i}_{j}_PLACEHOLDER'
                formula_map[placeholder_key] = match
                result = result.replace(match, placeholder_key, 1)
        
        # Step 2: Convert **bold headers** to proper markdown headers
        # Main section headers (longer titles with key words)
        result = re.sub(r'\*\*(.*?(?:Calculating|Determine|Calculate|Final|Minimum|Step|Method|Example|Overview|Summary|Introduction|Conclusion)[^*]{5,}?)\*\*', r'# \1', result)
        
        # Numbered section headers: **1.** -> ## 1.
        result = re.sub(r'\*\*(\d+\..*?)\*\*', r'## \1', result)
        
        # Subsection headers with key terms
        result = re.sub(r'\*\*((?:Step|Section|Part|Phase|Component|Element|Factor|Requirement|Condition|Assumption|Variable|Formula|Calculation|Procedure|Process|Method)[^*]*?)\*\*', r'### \1', result)
        
        # Step 3: Fix line breaks around numbered sections
        # Ensure space before numbered sections (1., 2., etc.)
        result = re.sub(r'([^\n])\n(\d+\.)', r'\1\n\n\2', result)
        result = re.sub(r'(\d+\.)\s*([A-Z])', r'\1 \2', result)  # Ensure space after number
        
        # Fix subsection numbering (4.1, 4.2, etc.)
        result = re.sub(r'([^\n])\n(\d+\.\d+)', r'\1\n\n\2', result)
        
        # Step 4: Clean up whitespace
        result = re.sub(r'[ \t]+', ' ', result)
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        # Step 5: Ensure proper spacing around headers
        result = re.sub(r'\n(#{1,6}\s)', r'\n\n\1', result)  # Space before headers
        result = re.sub(r'(#{1,6}[^\n]+)\n([^\n#])', r'\1\n\n\2', result)  # Space after headers
        
        # Step 6: Fix list formatting
        result = re.sub(r'\n(-\s)', r'\n\n- ', result)  # Space before bullet lists
        result = re.sub(r'\n(\d+\.)\s*([^\n])', r'\n\n\1 \2', result)  # Space before numbered lists
        
        # Step 7: Clean up excessive bold formatting in body text
        # Remove **bold** from short phrases that shouldn't be emphasized
        result = re.sub(r'\*\*([^*]{1,20})\*\*(?=\s[a-z])', r'\1', result)  # Short bold followed by lowercase
        
        # Step 8: Restore preserved formulas
        for placeholder_key, original_formula in formula_map.items():
            result = result.replace(placeholder_key, original_formula)
        
        result = result.strip()
        return result
    
    def clear_cache(self):
        """Clear the response cache to force fresh responses"""
        try:
            if self.cache:
                self.cache.flushdb()
                logger.info("✅ Response cache cleared successfully")
                return True
            else:
                logger.info("ℹ️ No cache available to clear")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to clear cache: {e}")
            return False
