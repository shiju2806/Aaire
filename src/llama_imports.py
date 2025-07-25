"""
Compatibility layer for llama-index imports
Handles different versions and import structures
"""

# Try different import structures for llama-index compatibility
try:
    # Try llama-index 0.12.x structure first
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, StorageContext, Settings
    from llama_index.core.node_parser import SimpleNodeParser
    from llama_index.core.indices.base_retriever import BaseRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    
    # Check if Settings exists (0.12.x uses this instead of ServiceContext)
    HAS_SETTINGS = True
    ServiceContext = None  # Not used in 0.12.x
    
    print("✅ Using llama-index 0.12.x import structure")
    
except ImportError as e:
    print(f"⚠️ llama-index 0.12.x imports failed: {e}")
    try:
        # Try older llama-index structure (0.10.x)
        from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document, ServiceContext, StorageContext
        from llama_index.node_parser import SimpleNodeParser
        from llama_index.indices.base_retriever import BaseRetriever
        from llama_index.query_engine import RetrieverQueryEngine
        
        # Older versions use ServiceContext
        HAS_SETTINGS = False
        Settings = None
        
        print("✅ Using llama-index 0.10.x import structure")
        
    except ImportError as e2:
        print(f"❌ Both import structures failed: {e2}")
        raise ImportError("Unable to import required llama-index components")

# LLM and Embedding imports
try:
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    print("✅ OpenAI imports successful")
except ImportError as e:
    print(f"⚠️ Trying alternative OpenAI imports: {e}")
    try:
        from llama_index.llms import OpenAI
        from llama_index.embeddings import OpenAIEmbedding
        print("✅ Alternative OpenAI imports successful")
    except ImportError as e2:
        print(f"❌ OpenAI imports failed: {e2}")
        raise

# Vector store imports
try:
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    print("✅ Qdrant vector store import successful")
except ImportError as e:
    print(f"⚠️ Qdrant import failed: {e}")
    try:
        from llama_index.vector_stores import QdrantVectorStore
        print("✅ Alternative Qdrant import successful")
    except ImportError as e2:
        print(f"❌ Qdrant imports failed: {e2}")
        QdrantVectorStore = None

# Export all the imported classes
__all__ = [
    'VectorStoreIndex',
    'SimpleDirectoryReader', 
    'Document',
    'StorageContext',
    'Settings',
    'ServiceContext',
    'SimpleNodeParser',
    'BaseRetriever',
    'RetrieverQueryEngine',
    'OpenAI',
    'OpenAIEmbedding',
    'QdrantVectorStore',
    'HAS_SETTINGS'
]

print("🎯 llama-index compatibility layer loaded successfully")