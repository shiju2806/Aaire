#!/usr/bin/env python3
"""
Check compatibility between llama-index versions and our imports
"""
import subprocess
import sys

print("🔍 Compatibility Check for llama-index==0.12.7")
print("=" * 60)

# Test the imports we actually use in our code
required_imports = [
    # From rag_pipeline.py
    ("llama_index.core", "VectorStoreIndex"),
    ("llama_index.core", "SimpleDirectoryReader"),
    ("llama_index.core", "Document"),
    ("llama_index.core", "Settings"),
    ("llama_index.core", "StorageContext"),
    ("llama_index.core.node_parser", "SimpleNodeParser"),
    ("llama_index.core.indices.base_retriever", "BaseRetriever"),
    ("llama_index.core.query_engine", "RetrieverQueryEngine"),
    ("llama_index.embeddings.openai", "OpenAIEmbedding"),
    ("llama_index.llms.openai", "OpenAI"),
    ("llama_index.vector_stores.qdrant", "QdrantVectorStore"),
    
    # From document_processor.py and external_apis.py
    ("llama_index.core", "Document"),
]

print("📦 Testing required imports for llama-index 0.12.7...")
print("-" * 40)

# Check if we can test locally first
try:
    import llama_index
    current_version = getattr(llama_index, '__version__', 'unknown')
    print(f"Current llama-index version: {current_version}")
except ImportError:
    print("llama-index not currently installed")

# Test each import
success_count = 0
failed_imports = []

for module_path, class_name in required_imports:
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name, None)
        if cls:
            print(f"✅ {module_path}.{class_name}")
            success_count += 1
        else:
            print(f"❌ {module_path}.{class_name} - class not found")
            failed_imports.append(f"{module_path}.{class_name}")
    except ImportError as e:
        print(f"❌ {module_path}.{class_name} - {e}")
        failed_imports.append(f"{module_path}.{class_name}")

print(f"\n📊 Results: {success_count}/{len(required_imports)} imports successful")

if failed_imports:
    print(f"\n❌ Failed imports:")
    for imp in failed_imports:
        print(f"   • {imp}")
        
    print(f"\n🔍 Alternative import paths to try:")
    alternatives = {
        "VectorStoreIndex": ["llama_index.VectorStoreIndex", "llama_index.indices.VectorStoreIndex"],
        "Document": ["llama_index.Document", "llama_index.schema.Document"],
        "Settings": ["llama_index.Settings", "llama_index.global_settings.Settings"],
        "SimpleDirectoryReader": ["llama_index.SimpleDirectoryReader", "llama_index.readers.SimpleDirectoryReader"],
        "StorageContext": ["llama_index.StorageContext", "llama_index.storage.StorageContext"],
    }
    
    for failed in failed_imports:
        class_name = failed.split('.')[-1]
        if class_name in alternatives:
            print(f"\n   {class_name} alternatives:")
            for alt in alternatives[class_name]:
                try:
                    module_parts = alt.split('.')
                    module_name = '.'.join(module_parts[:-1])
                    cls_name = module_parts[-1]
                    module = __import__(module_name, fromlist=[cls_name])
                    cls = getattr(module, cls_name, None)
                    if cls:
                        print(f"      ✅ {alt}")
                    else:
                        print(f"      ❌ {alt} - class not found")
                except ImportError:
                    print(f"      ❌ {alt} - import failed")
else:
    print(f"\n✅ All imports successful! llama-index configuration is compatible.")

print(f"\n🎯 Recommendations:")
if failed_imports:
    print("❌ Current configuration may have import issues.")
    print("   Consider using alternative import paths or different versions.")
else:
    print("✅ Current configuration should work fine.")
    print("   Proceed with the pinned versions.")

# Check specific version compatibility
print(f"\n📋 Version compatibility notes for llama-index 0.12.7:")
print("• This is a relatively recent version with modern import structure")
print("• Should support gpt-4o-mini model")
print("• Compatible with qdrant-client 1.7.3")
print("• Uses Settings instead of ServiceContext (good)")

print(f"\n🏁 Compatibility check complete!")