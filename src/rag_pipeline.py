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
            # Check cache first
            cache_key = self._get_cache_key(query, filters)
            if self.cache and self.config['retrieval_config']['use_cache']:
                cached_response = self.cache.get(cache_key)
                if cached_response:
                    logger.info("Returning cached response", query_hash=cache_key[:8])
                    return self._deserialize_response(cached_response, session_id)
            
            # Determine document type filter
            doc_type_filter = self._get_doc_type_filter(filters)
            
            # Expand query for better retrieval
            expanded_query = self._expand_query(query)
            
            # Get adaptive similarity threshold
            similarity_threshold = self._get_similarity_threshold(query)
            
            # Store current query for citation filtering
            self._current_query = query
            
            # Retrieve relevant documents using expanded query and adaptive threshold
            retrieved_docs = await self._retrieve_documents(expanded_query, doc_type_filter, similarity_threshold)
            
            # Check if this is a general knowledge query
            is_general_query = self._is_general_knowledge_query(query)
            
            # For general knowledge queries, don't pass any documents to avoid citation generation
            if is_general_query:
                logger.info(f"General knowledge query detected: '{query}' - using general knowledge response")
                response = await self._generate_response(query, [], user_context, conversation_history)  # Empty docs list
                # Force remove any citations from general knowledge response
                response = self._remove_citations_from_response(response)
                citations = []
                confidence = 0.3  # Low confidence for general knowledge responses
            else:
                # Generate response with retrieved documents
                response = await self._generate_response(query, retrieved_docs, user_context, conversation_history)
                
                # Extract citations only if we have relevant documents
                if retrieved_docs:
                    citations = self._extract_citations(retrieved_docs, query)
                    confidence = self._calculate_confidence(retrieved_docs, response)
                else:
                    # No relevant documents found - no citations and low confidence
                    citations = []
                    confidence = 0.3  # Low confidence for general knowledge responses
            
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
    
    async def _retrieve_documents(self, query: str, doc_type_filter: Optional[List[str]], similarity_threshold: Optional[float] = None) -> List[Dict]:
        """Hybrid retrieval: combines vector search with BM25 keyword search"""
        
        # Get results from both search methods
        vector_results = await self._vector_search(query, doc_type_filter, similarity_threshold)
        keyword_results = await self._keyword_search(query, doc_type_filter)
        
        # Combine and rerank results
        combined_results = self._combine_search_results(vector_results, keyword_results, query)
        
        return combined_results
    
    async def _vector_search(self, query: str, doc_type_filter: Optional[List[str]], similarity_threshold: Optional[float] = None) -> List[Dict]:
        """Original vector-based semantic search"""
        all_results = []
        
        try:
            # Create retriever from single index
            retriever = self.index.as_retriever(
                similarity_top_k=self.config['retrieval_config']['max_results']
            )
            
            # Retrieve documents
            nodes = retriever.retrieve(query)
            
            # Use adaptive threshold if provided, otherwise fall back to config
            threshold = similarity_threshold if similarity_threshold is not None else self.config['retrieval_config']['similarity_threshold']
            
            for node in nodes:
                if node.score >= threshold:
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
        return all_results[:self.config['retrieval_config']['max_results']]
    
    async def _keyword_search(self, query: str, doc_type_filter: Optional[List[str]]) -> List[Dict]:
        """BM25-based keyword search"""
        results = []
        
        try:
            if not self.bm25_index or not self.bm25_documents:
                logger.info("BM25 index not available, skipping keyword search")
                return results
            
            # Tokenize query for BM25
            query_tokens = self._tokenize_text(query)
            
            # Get BM25 scores for all documents
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            
            # Create results with scores
            for i, score in enumerate(bm25_scores):
                if score > 0 and i < len(self.bm25_metadata):  # Only include docs with positive scores
                    doc_metadata = self.bm25_metadata[i]
                    
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
            results = results[:self.config['retrieval_config']['max_results']]
            
            logger.info(f"BM25 keyword search found {len(results)} results")
            
        except Exception as e:
            logger.error("Failed to perform BM25 keyword search", error=str(e))
        
        return results
    
    def _combine_search_results(self, vector_results: List[Dict], keyword_results: List[Dict], query: str) -> List[Dict]:
        """Combine and rerank results from vector and keyword search"""
        
        try:
            # Create a combined results dictionary to avoid duplicates
            combined_dict = {}
            
            # Normalize scores and add vector results
            max_vector_score = max([r['score'] for r in vector_results], default=1.0)
            for result in vector_results:
                node_id = result['node_id']
                normalized_score = result['score'] / max_vector_score if max_vector_score > 0 else 0
                
                combined_dict[node_id] = result.copy()
                combined_dict[node_id]['vector_score'] = normalized_score
                combined_dict[node_id]['keyword_score'] = 0.0
                combined_dict[node_id]['combined_score'] = normalized_score * 0.6  # Weight vector search at 60%
            
            # Normalize scores and add/update keyword results
            max_keyword_score = max([r['score'] for r in keyword_results], default=1.0)
            for result in keyword_results:
                node_id = result['node_id']
                normalized_score = result['score'] / max_keyword_score if max_keyword_score > 0 else 0
                
                if node_id in combined_dict:
                    # Update existing result with keyword score
                    combined_dict[node_id]['keyword_score'] = normalized_score
                    combined_dict[node_id]['combined_score'] = (
                        combined_dict[node_id]['vector_score'] * 0.6 + normalized_score * 0.4
                    )
                    combined_dict[node_id]['search_type'] = 'hybrid'
                else:
                    # Add new keyword-only result
                    combined_dict[node_id] = result.copy()
                    combined_dict[node_id]['vector_score'] = 0.0
                    combined_dict[node_id]['keyword_score'] = normalized_score
                    combined_dict[node_id]['combined_score'] = normalized_score * 0.4  # Weight keyword-only at 40%
            
            # Convert back to list and sort by combined score
            final_results = list(combined_dict.values())
            final_results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Take top results and clean up temporary scoring fields
            final_results = final_results[:self.config['retrieval_config']['max_results']]
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
            # Fallback to vector results only
            return vector_results[:self.config['retrieval_config']['max_results']]
    
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
            # No relevant documents found - provide general knowledge response
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
            # Build context from retrieved documents
            context_parts = []
            for i, doc in enumerate(retrieved_docs[:5]):  # Use top 5 docs
                context_parts.append(f"[{i+1}] {doc['content']}")
            
            context = "\n\n".join(context_parts)
            
            # Build prompt with document context
            prompt = f"""You are AAIRE, an expert in insurance accounting and actuarial matters.
You provide accurate information based on US GAAP, IFRS, and company policies.
{conversation_context}
Current User Question: {query}

Relevant Information from Company Documents:
{context}

Instructions:
- Consider the conversation history to provide contextual and relevant answers
- Provide a comprehensive answer based ONLY on the relevant information provided above
- Always cite your sources using [1], [2], etc. format when referencing the provided information
- If the provided information is insufficient to fully answer the question, clearly state this
- You may supplement with general accounting knowledge, but clearly distinguish between document-based and general information
- Never provide tax or legal advice
- Focus on accounting and actuarial standards
- Build upon previous conversation context when appropriate

Response:"""

        try:
            # Generate response using OpenAI
            response = self.llm.complete(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error("Failed to generate response", error=str(e))
            return "I apologize, but I'm unable to generate a response at this time. Please try again."
    
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
        
        # Determine context for better follow-up questions
        topic_context = ""
        if retrieved_docs:
            # Use document context for specific follow-ups
            topics = []
            for doc in retrieved_docs[:3]:  # Use top 3 documents for context
                if 'source' in doc.get('metadata', {}):
                    topics.append(doc['metadata']['source'])
            if topics:
                topic_context = f"Related to documents: {', '.join(topics[:2])}"
        
        # Build category guidance
        category_guidance = ""
        for cat, examples in category_examples.items():
            category_guidance += f"\n{cat.title()}: {', '.join(examples[:2])}"
        
        prompt = f"""As AAIRE, an insurance accounting and actuarial expert, generate 2-3 specific follow-up questions that would help the user understand this topic better.

Original Question: {query}

My Response: {response}

{topic_context}

Suggested Question Categories:{category_guidance}

Instructions:
- Generate exactly 2-3 concise, specific follow-up questions
- Use different categories from the suggestions above for variety
- Make questions relevant to insurance accounting, actuarial science, or compliance
- Keep questions under 15 words each
- Format as a simple list, one question per line
- No numbering, bullets, or extra formatting
- Questions should encourage deeper understanding and practical application

Follow-up Questions:"""

        try:
            response_obj = self.llm.complete(prompt)
            questions_text = response_obj.text.strip()
            
            # Parse the response into individual questions
            questions = []
            for line in questions_text.split('\n'):
                line = line.strip()
                if line and len(line) > 10:  # Filter out empty or very short lines
                    # Clean up any unwanted formatting
                    clean_question = line.strip('- •').strip()
                    if clean_question.endswith('?'):
                        questions.append(clean_question)
            
            # Return max 3 questions
            return questions[:3]
            
        except Exception as e:
            logger.error("Failed to generate follow-up questions", error=str(e))
            # Return fallback questions if generation fails
            return [
                "Can you explain this in more detail?",
                "What are the practical implications?",
                "How does this apply in practice?"
            ]
    
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
            r'\buploaded.*image\b'  # "uploaded image"
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
        """Extract citation information from retrieved documents"""
        citations = []
        
        # Use strict citation threshold to prevent irrelevant citations
        # Only show citations for highly relevant documents
        if retrieved_docs:
            top_scores = [doc['score'] for doc in retrieved_docs[:3]]
            min_top_score = min(top_scores) if top_scores else 0.75
            
            # Special handling for specific ASC code queries
            import re
            if re.search(r'ASC \d{3}-\d{2}', query):
                CITATION_THRESHOLD = max(0.5, min_top_score - 0.1)  # More lenient for ASC queries
                logger.info(f"ASC code query detected, using lower threshold")
            else:
                # More strict threshold - only show citations for truly relevant docs
                CITATION_THRESHOLD = max(0.75, min_top_score - 0.02)  # Much stricter
        else:
            CITATION_THRESHOLD = 0.75  # Fallback to original threshold
        
        logger.info(f"Citation threshold calculated: {CITATION_THRESHOLD}")
        
        for i, doc in enumerate(retrieved_docs[:5]):
            # Log all document scores for debugging
            logger.info(f"Document {i+1}: score={doc['score']}, filename={doc['metadata'].get('filename', 'Unknown')}")
            
            # Skip documents with low relevance scores
            if doc['score'] < CITATION_THRESHOLD:
                logger.info(f"SKIPPING citation for low-relevance document (score: {doc['score']}) - threshold: {CITATION_THRESHOLD}")
                continue
            
            # Additional filter: Skip documents that seem to be general/irrelevant responses
            # This helps prevent old cached documents from appearing as citations
            content_lower = doc['content'].lower()
            filename_lower = doc['metadata'].get('filename', '').lower()
            
            # Skip generic responses
            if any(phrase in content_lower for phrase in [
                'how can i assist you today',
                'feel free to share',
                'what can i help you with',
                'how may i help',
                'hello! how can i assist'
            ]):
                logger.info(f"SKIPPING citation for generic response document: {doc['metadata'].get('filename', 'Unknown')}")
                continue
            
            # Skip completely unrelated document types for image-specific queries
            query_lower = getattr(self, '_current_query', '').lower()
            if any(term in query_lower for term in ['image', 'chart', 'graph', 'figure']):
                # If asking about images but document is about taxes/personal matters, skip it
                if any(term in filename_lower for term in ['tax', 'personal', 'canada', 'ey-managing']) and not any(term in content_lower for term in ['revenue', 'fy23', 'financial', 'earnings']):
                    logger.info(f"SKIPPING citation for unrelated document in image query: {filename_lower}")
                    continue
                
            # Get filename for source
            filename = doc['metadata'].get('filename', 'Unknown')
            
            citation = {
                "id": len(citations) + 1,  # Use actual citation count, not doc index
                "text": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                "source": filename,
                "source_type": doc['source_type'],
                "confidence": round(doc['score'], 3)
            }
            
            # Add additional metadata if available
            if 'page' in doc['metadata']:
                citation['page'] = doc['metadata']['page']
            if 'section' in doc['metadata']:
                citation['section'] = doc['metadata']['section']
            if 'standard' in doc['metadata']:
                citation['standard'] = doc['metadata']['standard']
                
            citations.append(citation)
        
        logger.info(f"Generated {len(citations)} citations from {len(retrieved_docs)} retrieved documents")
        return citations
    
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
        """Generate cache key for query"""
        import hashlib
        cache_data = {
            'query': query,
            'filters': filters or {}
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
