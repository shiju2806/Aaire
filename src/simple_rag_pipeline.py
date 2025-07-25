"""
Simplified RAG Pipeline using direct OpenAI API
Works without complex llama-index dependencies
"""

import os
import yaml
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import uuid
import openai
from openai import OpenAI as OpenAIClient

import structlog
logger = structlog.get_logger()

class SimpleRAGResponse:
    def __init__(self, answer: str, citations: List[Dict], confidence: float, session_id: str):
        self.answer = answer
        self.citations = citations
        self.confidence = confidence
        self.session_id = session_id

class SimpleRAGPipeline:
    def __init__(self, config_path: str = "config/mvp_config.yaml"):
        """Initialize simple RAG pipeline using direct OpenAI API"""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize OpenAI client
        self.client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Get model configuration
        self.model_name = os.getenv("OPENAI_MODEL", self.config['llm_config']['model'])
        self.temperature = self.config['llm_config']['temperature']
        self.max_tokens = self.config['llm_config']['max_tokens']
        
        logger.info(f"Simple RAG Pipeline initialized with model: {self.model_name}")
        
        # Simple in-memory document store
        self.documents = []
        self.vector_store_type = "simple"
        self.index_name = "simple"
    
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
            r'\bprocedure\b'
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
    
    async def process_query(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> SimpleRAGResponse:
        """Process a user query through the simple RAG pipeline"""
        session_id = str(uuid.uuid4())
        
        try:
            # Check if this is a general knowledge query
            is_general_query = self._is_general_knowledge_query(query)
            
            if is_general_query:
                logger.info(f"General knowledge query detected: '{query}'")
                # Generate general knowledge response
                prompt = f"""You are AAIRE, an expert in insurance accounting and actuarial matters.

User Question: {query}

This appears to be a general accounting question. Provide a helpful answer based on standard accounting principles.
Do NOT include any citation numbers like [1], [2], etc.
Make it clear this is general knowledge, not from specific documents.

Response:"""
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "system", "content": "You are an expert in insurance accounting."}, 
                              {"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                return SimpleRAGResponse(
                    answer=response.choices[0].message.content,
                    citations=[],
                    confidence=0.3,
                    session_id=session_id
                )
            else:
                # For document-specific queries, search documents if any exist
                if self.documents:
                    # Simple keyword search
                    relevant_docs = self._search_documents(query)
                    
                    if relevant_docs:
                        context = "\n\n".join([f"[{i+1}] {doc['content']}" for i, doc in enumerate(relevant_docs[:3])])
                        prompt = f"""You are AAIRE, an expert in insurance accounting and actuarial matters.

User Question: {query}

Relevant Information from Company Documents:
{context}

Provide a comprehensive answer based on the relevant information above.
Always cite your sources using [1], [2], etc. format.

Response:"""
                    else:
                        prompt = f"""You are AAIRE, an expert in insurance accounting and actuarial matters.

User Question: {query}

No specific documents were found that directly address this question.
Provide a helpful general answer and suggest consulting company-specific policies.

Response:"""
                else:
                    prompt = f"""You are AAIRE, an expert in insurance accounting and actuarial matters.

User Question: {query}

No documents have been uploaded yet. Please upload relevant documents to get specific answers.
For now, I can provide general accounting guidance.

Response:"""
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "system", "content": "You are an expert in insurance accounting."}, 
                              {"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # Extract citations if we had relevant docs
                citations = []
                if self.documents and 'relevant_docs' in locals() and relevant_docs:
                    for i, doc in enumerate(relevant_docs[:3]):
                        citations.append({
                            "id": i + 1,
                            "text": doc['content'][:200] + "...",
                            "source": doc.get('filename', 'Unknown'),
                            "source_type": "company",
                            "confidence": 0.8
                        })
                
                return SimpleRAGResponse(
                    answer=response.choices[0].message.content,
                    citations=citations,
                    confidence=0.8 if citations else 0.3,
                    session_id=session_id
                )
                
        except Exception as e:
            logger.error("Failed to process query", error=str(e))
            return SimpleRAGResponse(
                answer="I apologize, but I'm experiencing technical difficulties. Please try again.",
                citations=[],
                confidence=0.0,
                session_id=session_id
            )
    
    def _search_documents(self, query: str) -> List[Dict]:
        """Simple keyword search in documents"""
        query_words = query.lower().split()
        results = []
        
        for doc in self.documents:
            content_lower = doc['content'].lower()
            score = sum(1 for word in query_words if word in content_lower) / len(query_words)
            if score > 0.3:
                results.append({**doc, 'score': score})
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:5]
    
    async def add_documents(self, documents: List[Any], doc_type: str = "company"):
        """Add documents to simple store"""
        for doc in documents:
            self.documents.append({
                'content': doc.text if hasattr(doc, 'text') else str(doc),
                'filename': doc.metadata.get('filename', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown',
                'job_id': doc.metadata.get('job_id', str(uuid.uuid4())) if hasattr(doc, 'metadata') else str(uuid.uuid4()),
                'doc_type': doc_type,
                'added_at': datetime.utcnow().isoformat()
            })
        return len(documents)
    
    async def delete_document(self, job_id: str) -> Dict[str, Any]:
        """Delete document by job_id"""
        before_count = len(self.documents)
        self.documents = [doc for doc in self.documents if doc.get('job_id') != job_id]
        deleted_count = before_count - len(self.documents)
        
        return {
            "status": "success",
            "deleted_chunks": deleted_count,
            "job_id": job_id,
            "vector_store": self.vector_store_type
        }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get simple pipeline statistics"""
        return {
            "index": {
                "name": "simple",
                "status": "active",
                "document_count": len(self.documents)
            },
            "configuration": {
                "model": self.model_name,
                "index_name": "simple"
            }
        }
    
    async def clear_all_cache(self) -> Dict[str, Any]:
        """Clear cache (no-op for simple pipeline)"""
        return {"status": "success", "cleared_entries": 0}
    
    async def generate_document_summary(self, document_content: str, doc_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate document summary using OpenAI"""
        try:
            prompt = f"""Summarize this document for accounting and actuarial professionals:

Document: {doc_metadata.get('title', 'Unknown')}
Type: {doc_metadata.get('source_type', 'Unknown')}

Content (first 3000 chars):
{document_content[:3000]}

Provide:
1. Executive summary (2-3 sentences)
2. Key accounting/actuarial impacts
3. Important dates or deadlines
4. Action items

Summary:"""

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            return {
                "summary": response.choices[0].message.content,
                "key_insights": [],
                "document_metadata": doc_metadata,
                "generated_at": datetime.utcnow().isoformat(),
                "confidence": 0.8
            }
        except Exception as e:
            logger.error("Failed to generate summary", error=str(e))
            return {
                "summary": "Summary generation failed. Document has been processed.",
                "key_insights": [],
                "document_metadata": doc_metadata,
                "generated_at": datetime.utcnow().isoformat(),
                "confidence": 0.0
            }