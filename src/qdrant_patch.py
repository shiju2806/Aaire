"""
Patch for Qdrant fastembed compatibility issue
Fixes IDF_EMBEDDING_MODELS import error
"""

def patch_qdrant_fastembed():
    """
    Patch the missing IDF_EMBEDDING_MODELS in qdrant_client.qdrant_fastembed
    """
    try:
        import qdrant_client.qdrant_fastembed as qf
        
        # Check if IDF_EMBEDDING_MODELS is missing
        if not hasattr(qf, 'IDF_EMBEDDING_MODELS'):
            # Check if SUPPORTED_EMBEDDING_MODELS exists
            if hasattr(qf, 'SUPPORTED_EMBEDDING_MODELS'):
                # Create IDF_EMBEDDING_MODELS as alias for SUPPORTED_EMBEDDING_MODELS
                qf.IDF_EMBEDDING_MODELS = qf.SUPPORTED_EMBEDDING_MODELS
                print("✅ Patched IDF_EMBEDDING_MODELS in qdrant_client.qdrant_fastembed")
                return True
            else:
                # Create a minimal fallback
                qf.IDF_EMBEDDING_MODELS = []
                print("⚠️ Created empty IDF_EMBEDDING_MODELS fallback")
                return True
        else:
            print("✅ IDF_EMBEDDING_MODELS already exists")
            return True
            
    except Exception as e:
        print(f"❌ Failed to patch qdrant_fastembed: {e}")
        return False

def try_import_qdrant_vector_store():
    """
    Try to import QdrantVectorStore with patching
    """
    try:
        # First apply the patch
        if patch_qdrant_fastembed():
            # Now try the import
            from llama_index.vector_stores.qdrant import QdrantVectorStore
            print("✅ QdrantVectorStore imported successfully after patch!")
            return QdrantVectorStore
        else:
            print("❌ Patch failed, cannot import QdrantVectorStore")
            return None
    except Exception as e:
        print(f"❌ QdrantVectorStore import still failed after patch: {e}")
        return None

# Auto-apply patch when imported
if __name__ == "__main__":
    print("🔧 Testing Qdrant patch...")
    QdrantVectorStore = try_import_qdrant_vector_store()
    if QdrantVectorStore:
        print("🎉 Patch successful!")
    else:
        print("💔 Patch failed")
else:
    # Auto-patch when imported
    patch_qdrant_fastembed()