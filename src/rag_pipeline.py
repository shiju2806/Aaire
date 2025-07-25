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

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    ServiceContext,
    StorageContext
)
from llama_index.node_parser import SimpleNodeParser
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response.pprint_utils import pprint_response
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import OpenAI
from llama_index.vector_stores import PineconeVectorStore

# Import vector store options
import pinecone
try:
    from qdrant_client import QdrantClient
    from llama_index.vector_stores import QdrantVectorStore
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

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
            model=self.config['embedding_config']['model']
        )
        
        # Initialize service context
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embedding_model
        )
        
        # Initialize node parser with simple chunking (hierarchical not available in 0.9.x)
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.config['chunking_strategies']['default']['chunk_size'],
            chunk_overlap=self.config['chunking_strategies']['default']['overlap']
        )
        
        # Try vector stores in priority order: Qdrant > Pinecone > Local
        self.vector_store_type = None
        self.index_name = None  # Will be set based on vector store type
        
        # Try Qdrant first (better free tier)
        if self._try_qdrant():
            self.vector_store_type = "qdrant"
            self.index_name = self.collection_name  # Use Qdrant collection name
            logger.info("Using Qdrant vector store")
        # Fall back to Pinecone
        elif self._try_pinecone():
            self.vector_store_type = "pinecone" 
            logger.info("Using Pinecone vector store")
        # Fall back to local storage
        else:
            self._init_local_index()
            self.vector_store_type = "local"
            self.index_name = "local"
            logger.info("Using local vector store")
        
        # Initialize Redis for caching
        self._init_cache()
        
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
            
            if not qdrant_url:
                logger.info("QDRANT_URL not set, skipping Qdrant")
                return False
                
            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key
            )
            
            # Test connection
            collections = self.qdrant_client.get_collections()
            logger.info("Connected to Qdrant successfully")
            
            # Initialize Qdrant vector store
            self.collection_name = "aaire-documents"
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name
            )
            
            self._init_qdrant_indexes()
            return True
            
        except Exception as e:
            logger.info("Qdrant initialization failed, trying Pinecone", error=str(e)[:100])
            return False
    
    def _try_pinecone(self) -> bool:
        """Try to initialize Pinecone vector store"""
        try:
            self._init_pinecone()
            self._init_indexes()
            return True
        except Exception as e:
            logger.info("Pinecone initialization failed, using local storage", error=str(e)[:100])
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
            
            # Initialize storage context with Qdrant
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # Create or load index with Qdrant
            try:
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    service_context=self.service_context
                )
                logger.info("Loaded existing Qdrant index")
            except:
                self.index = VectorStoreIndex(
                    nodes=[],
                    storage_context=self.storage_context,
                    service_context=self.service_context
                )
                logger.info("Created new Qdrant index")
            
            logger.info("Qdrant indexes initialized")
            
        except Exception as e:
            logger.error("Failed to initialize Qdrant indexes", error=str(e))
            raise
    
    def _init_pinecone(self):
        """Initialize Pinecone vector database using v2.x API for llama-index 0.9.x compatibility"""
        try:
            import pinecone
            
            # Initialize Pinecone with v2.x API format (requires environment)
            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
            )
            
            # Single index for Pinecone free tier
            self.index_name = "aaire-main"
            
            # List existing indexes using v2.x API
            existing_indexes = pinecone.list_indexes()
            
            if self.index_name not in existing_indexes:
                # Create index using v2.x API format
                pinecone.create_index(
                    name=self.index_name,
                    dimension=1536,  # Standard OpenAI embedding dimension
                    metric="cosine"
                )
                logger.info(f"Created Pinecone index: {self.index_name}")
            
            # Connect to the index using v2.x API
            self.pinecone_index = pinecone.Index(self.index_name)
            
            logger.info("Pinecone initialized successfully with v2.x API")
            
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
            logger.info("Redis cache not available, continuing without cache", error=str(e)[:50])
            self.cache = None
    
    def _init_indexes(self):
        """Initialize single LlamaIndex vector store index"""
        vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Try to load existing index or create new one
        try:
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                service_context=self.service_context
            )
            logger.info("Loaded existing Pinecone index")
        except:
            self.index = VectorStoreIndex(
                nodes=[],
                storage_context=storage_context,
                service_context=self.service_context
            )
            logger.info("Created new Pinecone index")
    
    def _init_local_index(self):
        """Initialize local vector store as fallback"""
        # Create a simple in-memory vector store
        self.index = VectorStoreIndex(
            nodes=[],
            service_context=self.service_context
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
            
            # Ensure nodes inherit the document type metadata
            for node in nodes:
                if not node.metadata:
                    node.metadata = {}
                node.metadata['doc_type'] = doc_type
                node.metadata['added_at'] = datetime.utcnow().isoformat()
            
            # Add to single index
            self.index.insert_nodes(nodes)
            
            # Invalidate cache for this document type
            if self.cache:
                pattern = f"query_cache:{doc_type}:*"
                for key in self.cache.scan_iter(match=pattern):
                    self.cache.delete(key)
            
            logger.info(f"Added {len(documents)} documents to index",
                       doc_type=doc_type,
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
            
            # Determine document type filter
            doc_type_filter = self._get_doc_type_filter(filters)
            
            # Retrieve relevant documents
            retrieved_docs = await self._retrieve_documents(query, doc_type_filter)
            
            # Generate response
            response = await self._generate_response(query, retrieved_docs, user_context)
            
            # Extract citations only if we have relevant documents
            if retrieved_docs:
                citations = self._extract_citations(retrieved_docs)
                confidence = self._calculate_confidence(retrieved_docs, response)
            else:
                # No relevant documents found - no citations and low confidence
                citations = []
                confidence = 0.3  # Low confidence for general knowledge responses
            
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
    
    async def _retrieve_documents(self, query: str, doc_type_filter: Optional[List[str]]) -> List[Dict]:
        """Retrieve relevant documents from single index with optional document type filtering"""
        all_results = []
        
        try:
            # Create retriever from single index
            retriever = self.index.as_retriever(
                similarity_top_k=self.config['retrieval_config']['max_results']
            )
            
            # Retrieve documents
            nodes = retriever.retrieve(query)
            
            for node in nodes:
                if node.score >= self.config['retrieval_config']['similarity_threshold']:
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
                        'node_id': node.id_
                    })
                    
        except Exception as e:
            logger.warning("Failed to retrieve from index", error=str(e))
        
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
        
        # Check if we have relevant documents
        if not retrieved_docs:
            # No relevant documents found - provide general knowledge response
            prompt = f"""You are AAIRE, an expert in insurance accounting and actuarial matters.
You provide accurate information based on US GAAP, IFRS, and general accounting principles.

User Question: {query}

No specific documents were found in your company's knowledge base that directly address this question.

Instructions:
- Provide a helpful general answer based on standard accounting and actuarial principles
- Clearly state that this is general information, not from specific company documents
- Mention relevant accounting standards (US GAAP, IFRS) where applicable
- Never provide tax or legal advice
- Suggest that the user consult company-specific policies or seek professional advice for specific situations
- Do NOT reference any specific documents or sources

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

User Question: {query}

Relevant Information from Company Documents:
{context}

Instructions:
- Provide a comprehensive answer based ONLY on the relevant information provided above
- Always cite your sources using [1], [2], etc. format when referencing the provided information
- If the provided information is insufficient to fully answer the question, clearly state this
- You may supplement with general accounting knowledge, but clearly distinguish between document-based and general information
- Never provide tax or legal advice
- Focus on accounting and actuarial standards

Response:"""

        try:
            # Generate response using OpenAI
            response = self.llm.complete(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error("Failed to generate response", error=str(e))
            return "I apologize, but I'm unable to generate a response at this time. Please try again."
    
    def _extract_citations(self, retrieved_docs: List[Dict]) -> List[Dict[str, Any]]:
        """Extract citation information from retrieved documents"""
        citations = []
        
        # Only include citations for documents with sufficient relevance score
        CITATION_THRESHOLD = 0.85  # High threshold to prevent false citations
        
        for i, doc in enumerate(retrieved_docs[:5]):
            # Log all document scores for debugging
            logger.info(f"Document {i+1}: score={doc['score']}, filename={doc['metadata'].get('filename', 'Unknown')}")
            
            # Skip documents with low relevance scores
            if doc['score'] < CITATION_THRESHOLD:
                logger.info(f"SKIPPING citation for low-relevance document (score: {doc['score']}) - threshold: {CITATION_THRESHOLD}")
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
