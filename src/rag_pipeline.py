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

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    ServiceContext,
    StorageContext
)
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.pinecone import PineconeVectorStore

import pinecone
from pinecone import Pinecone
import redis
import structlog

logger = structlog.get_logger()

class RAGResponse:
    def __init__(self, answer: str, citations: List[Dict], confidence: float, session_id: str):
        self.answer = answer
        self.citations = citations
        self.confidence = confidence
        self.session_id = session_id

class RAGPipeline:
    def __init__(self, config_path: str = "config/mvp_config.yaml"):
        """Initialize RAG pipeline with LlamaIndex and Pinecone"""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize OpenAI components
        self.llm = OpenAI(
            model=self.config['llm_config']['model'],
            temperature=self.config['llm_config']['temperature'],
            max_tokens=self.config['llm_config']['max_tokens']
        )
        
        self.embedding_model = OpenAIEmbedding(
            model=self.config['embedding_config']['model'],
            dimensions=self.config['embedding_config']['dimensions']
        )
        
        # Initialize Pinecone
        self._init_pinecone()
        
        # Initialize Redis for caching
        self._init_cache()
        
        # Initialize service context
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embedding_model
        )
        
        # Initialize node parser with hierarchical chunking
        self.node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=self.config['chunking_strategies']['default']['chunk_size'],
            chunk_overlap=self.config['chunking_strategies']['default']['overlap']
        )
        
        # Initialize indexes
        self.indexes = {}
        self._init_indexes()
        
        logger.info("RAG Pipeline initialized", 
                   model=self.config['llm_config']['model'],
                   embedding_model=self.config['embedding_config']['model'])
    
    def _init_pinecone(self):
        """Initialize Pinecone vector database"""
        try:
            # Initialize Pinecone client
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            
            # Define index names for different document types
            self.index_names = {
                "us_gaap": "aaire-us-gaap",
                "ifrs": "aaire-ifrs", 
                "company": "aaire-company",
                "actuarial": "aaire-actuarial"
            }
            
            # Create indexes if they don't exist
            existing_indexes = [index.name for index in pc.list_indexes()]
            
            for index_name in self.index_names.values():
                if index_name not in existing_indexes:
                    pc.create_index(
                        name=index_name,
                        dimension=self.config['embedding_config']['dimensions'],
                        metric="cosine",
                        spec={
                            "serverless": {
                                "cloud": "aws",
                                "region": "us-east-1"
                            }
                        }
                    )
                    logger.info(f"Created Pinecone index: {index_name}")
            
            # Connect to indexes
            self.pinecone_indexes = {
                key: pc.Index(index_name) 
                for key, index_name in self.index_names.items()
            }
            
        except Exception as e:
            logger.error("Failed to initialize Pinecone", error=str(e))
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
            logger.warning("Redis cache not available", error=str(e))
            self.cache = None
    
    def _init_indexes(self):
        """Initialize LlamaIndex vector store indexes"""
        for doc_type, pinecone_index in self.pinecone_indexes.items():
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Try to load existing index or create new one
            try:
                self.indexes[doc_type] = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    service_context=self.service_context
                )
                logger.info(f"Loaded existing index for {doc_type}")
            except:
                self.indexes[doc_type] = VectorStoreIndex(
                    nodes=[],
                    storage_context=storage_context,
                    service_context=self.service_context
                )
                logger.info(f"Created new index for {doc_type}")
    
    async def add_documents(self, documents: List[Document], doc_type: str = "company"):
        """
        Add documents to the appropriate index with hierarchical chunking
        """
        try:
            if doc_type not in self.indexes:
                raise ValueError(f"Unknown document type: {doc_type}")
            
            # Parse documents into hierarchical nodes
            nodes = self.node_parser.get_nodes_from_documents(documents)
            
            # Add to index
            self.indexes[doc_type].insert_nodes(nodes)
            
            # Invalidate cache for this document type
            if self.cache:
                pattern = f"query_cache:{doc_type}:*"
                for key in self.cache.scan_iter(match=pattern):
                    self.cache.delete(key)
            
            logger.info(f"Added {len(documents)} documents to {doc_type} index",
                       total_nodes=len(nodes))
            
            return len(nodes)
            
        except Exception as e:
            logger.error("Failed to add documents", error=str(e), doc_type=doc_type)
            raise
    
    async def process_query(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        """
        Process a user query through the RAG pipeline
        """
        session_id = str(uuid.uuid4())
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(query, filters)
            if self.cache and self.config['retrieval_config']['use_cache']:
                cached_response = self.cache.get(cache_key)
                if cached_response:
                    logger.info("Returning cached response", query_hash=cache_key[:8])
                    return self._deserialize_response(cached_response, session_id)
            
            # Determine which indexes to search
            search_indexes = self._get_search_indexes(filters)
            
            # Retrieve relevant documents
            retrieved_docs = await self._retrieve_documents(query, search_indexes)
            
            # Generate response
            response = await self._generate_response(query, retrieved_docs, user_context)
            
            # Extract citations
            citations = self._extract_citations(retrieved_docs)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(retrieved_docs, response)
            
            rag_response = RAGResponse(
                answer=response,
                citations=citations,
                confidence=confidence,
                session_id=session_id
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
    
    def _get_search_indexes(self, filters: Optional[Dict[str, Any]]) -> List[str]:
        """Determine which indexes to search based on filters"""
        if not filters or not filters.get('source_type'):
            return list(self.indexes.keys())  # Search all indexes
        
        source_types = filters['source_type']
        if isinstance(source_types, str):
            source_types = [source_types]
        
        search_indexes = []
        for source_type in source_types:
            if source_type == "US_GAAP":
                search_indexes.append("us_gaap")
            elif source_type == "IFRS":
                search_indexes.append("ifrs")
            elif source_type == "COMPANY":
                search_indexes.append("company")
            elif source_type == "ACTUARIAL":
                search_indexes.append("actuarial")
        
        return search_indexes or list(self.indexes.keys())
    
    async def _retrieve_documents(self, query: str, search_indexes: List[str]) -> List[Dict]:
        """Retrieve relevant documents from specified indexes"""
        all_results = []
        
        for index_name in search_indexes:
            if index_name not in self.indexes:
                continue
                
            try:
                # Create retriever
                retriever = VectorIndexRetriever(
                    index=self.indexes[index_name],
                    similarity_top_k=self.config['retrieval_config']['max_results'] // len(search_indexes)
                )
                
                # Retrieve documents
                nodes = await retriever.aretrieve(query)
                
                for node in nodes:
                    if node.score >= self.config['retrieval_config']['similarity_threshold']:
                        all_results.append({
                            'content': node.text,
                            'metadata': node.metadata or {},
                            'score': node.score,
                            'source_type': index_name,
                            'node_id': node.id_
                        })
                        
            except Exception as e:
                logger.warning(f"Failed to retrieve from {index_name}", error=str(e))
                continue
        
        # Sort by relevance score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:self.config['retrieval_config']['max_results']]
    
    async def _generate_response(
        self, 
        query: str, 
        retrieved_docs: List[Dict],
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate response using retrieved documents"""
        
        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5]):  # Use top 5 docs
            context_parts.append(f"[{i+1}] {doc['content']}")
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        prompt = f"""You are AAIRE, an expert in insurance accounting and actuarial matters.
You provide accurate information based on US GAAP, IFRS, and company policies.

User Question: {query}

Relevant Information:
{context}

Instructions:
- Provide a comprehensive answer based on the relevant information
- Always cite your sources using [1], [2], etc. format
- If information is insufficient, clearly state what additional details are needed
- Never provide tax or legal advice
- Focus on accounting and actuarial standards

Response:"""

        try:
            # Generate response using OpenAI
            response = await self.llm.acomplete(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error("Failed to generate response", error=str(e))
            return "I apologize, but I'm unable to generate a response at this time. Please try again."
    
    def _extract_citations(self, retrieved_docs: List[Dict]) -> List[Dict[str, Any]]:
        """Extract citation information from retrieved documents"""
        citations = []
        
        for i, doc in enumerate(retrieved_docs[:5]):
            citation = {
                "id": i + 1,
                "text": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                "source": doc['metadata'].get('source', 'Unknown'),
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
            'confidence': response.confidence
        })
    
    def _deserialize_response(self, cached_data: str, session_id: str) -> RAGResponse:
        """Deserialize cached response"""
        import json
        data = json.loads(cached_data)
        return RAGResponse(
            answer=data['answer'],
            citations=data['citations'],
            confidence=data['confidence'],
            session_id=session_id
        )
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get RAG pipeline statistics"""
        stats = {
            "indexes": {},
            "cache_stats": {},
            "configuration": {
                "model": self.config['llm_config']['model'],
                "embedding_model": self.config['embedding_config']['model'],
                "similarity_threshold": self.config['retrieval_config']['similarity_threshold']
            }
        }
        
        # Get index statistics
        for index_name, index in self.indexes.items():
            try:
                # Note: LlamaIndex doesn't directly expose document counts
                # This would need to be tracked separately in production
                stats["indexes"][index_name] = {
                    "status": "active",
                    "last_updated": datetime.utcnow().isoformat()
                }
            except Exception as e:
                stats["indexes"][index_name] = {
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