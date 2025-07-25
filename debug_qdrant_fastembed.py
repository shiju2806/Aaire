#!/usr/bin/env python3
"""
Deep dive debug for Qdrant fastembed issue
"""

print("🔍 Qdrant Fastembed Deep Dive Debug")
print("=" * 50)

# Test qdrant_client direct import
try:
    import qdrant_client
    print(f"✅ qdrant_client version: {qdrant_client.__version__}")
except Exception as e:
    print(f"❌ qdrant_client import failed: {e}")

# Test fastembed direct import
try:
    import fastembed
    print(f"✅ fastembed version: {fastembed.__version__}")
except Exception as e:
    print(f"❌ fastembed import failed: {e}")

# Test the specific problematic import
print("\n🔍 Testing specific import paths...")

try:
    from qdrant_client.qdrant_fastembed import IDF_EMBEDDING_MODELS
    print("✅ IDF_EMBEDDING_MODELS import successful")
except ImportError as e:
    print(f"❌ IDF_EMBEDDING_MODELS import failed: {e}")
    
    # Check what's actually available in qdrant_fastembed
    try:
        import qdrant_client.qdrant_fastembed as qf
        available_attrs = [attr for attr in dir(qf) if not attr.startswith('_')]
        print(f"📋 Available in qdrant_fastembed: {available_attrs}")
    except Exception as e2:
        print(f"❌ Cannot inspect qdrant_fastembed: {e2}")

# Test llama-index vector store import step by step
print("\n🔍 Testing llama-index vector store import step by step...")

try:
    print("1. Testing llama_index.vector_stores import...")
    import llama_index.vector_stores
    print("✅ llama_index.vector_stores imported")
    
    print("2. Testing llama_index.vector_stores.qdrant import...")
    import llama_index.vector_stores.qdrant
    print("✅ llama_index.vector_stores.qdrant imported")
    
    print("3. Testing QdrantVectorStore class import...")
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    print("✅ QdrantVectorStore imported successfully!")
    
except Exception as e:
    print(f"❌ Step-by-step import failed at: {e}")
    
    # Try to find alternative import path
    print("\n🔍 Trying alternative import paths...")
    
    alternatives = [
        "llama_index.vector_stores.QdrantVectorStore",
        "llama_index.storage.QdrantVectorStore", 
        "llama_index.core.vector_stores.QdrantVectorStore"
    ]
    
    for alt in alternatives:
        try:
            parts = alt.split('.')
            module_name = '.'.join(parts[:-1])
            class_name = parts[-1]
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✅ Alternative found: {alt}")
            break
        except Exception:
            print(f"❌ {alt} not available")

# Check if we can work around the issue
print("\n🔍 Testing workaround options...")

try:
    # Try importing qdrant without fastembed features
    from qdrant_client import QdrantClient
    print("✅ QdrantClient direct import works")
    
    # Try creating a simple vector store without problematic imports
    print("✅ Basic Qdrant functionality available")
    
except Exception as e:
    print(f"❌ Even basic Qdrant functionality failed: {e}")

print("\n🎯 Diagnosis complete!")
print("If QdrantVectorStore import still fails, we can:")
print("1. Use QdrantClient directly without llama-index wrapper")
print("2. Patch the import issue in our code")
print("3. Use a different qdrant/fastembed version combination")