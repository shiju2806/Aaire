#!/usr/bin/env python3
"""
Comprehensive debugging script to test all imports
"""
import os
import sys
import traceback

print("🔍 AAIRE Import Deep Dive Analysis")
print("=" * 60)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
print("✅ Environment loaded")

# Test each import individually
imports_to_test = [
    ("RAGPipeline", "src.rag_pipeline", "RAGPipeline"),
    ("DocumentProcessor", "src.document_processor", "DocumentProcessor"), 
    ("ExternalAPIManager", "src.external_apis", "ExternalAPIManager"),
    ("ComplianceEngine", "src.compliance_engine", "ComplianceEngine"),
    ("AuthManager", "src.auth", "AuthManager"),
]

results = {}

print("\n📦 Testing Individual Imports...")
print("-" * 40)

for name, module_path, class_name in imports_to_test:
    print(f"\n🧪 Testing {name}...")
    try:
        # Try importing the module
        print(f"  📝 Importing module: {module_path}")
        module = __import__(module_path, fromlist=[class_name])
        print(f"  ✅ Module imported successfully")
        
        # Try getting the class
        print(f"  📝 Getting class: {class_name}")
        cls = getattr(module, class_name)
        print(f"  ✅ Class found: {cls}")
        
        # Try basic instantiation test (if possible)
        print(f"  📝 Testing class instantiation...")
        if name == "RAGPipeline":
            # Special handling for RAGPipeline as it needs config
            try:
                instance = cls("config/mvp_config.yaml")
                print(f"  ✅ RAGPipeline instantiated successfully")
                results[name] = "SUCCESS"
            except Exception as e:
                print(f"  ⚠️  RAGPipeline instantiation failed: {e}")
                print(f"  📋 But import was successful")
                results[name] = "IMPORT_OK_INIT_FAILED"
        else:
            print(f"  ✅ Class accessible (not testing instantiation)")
            results[name] = "SUCCESS"
            
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        print(f"  📋 Full traceback:")
        traceback.print_exc()
        results[name] = f"IMPORT_FAILED: {e}"
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        print(f"  📋 Full traceback:")
        traceback.print_exc()
        results[name] = f"ERROR: {e}"

print("\n" + "=" * 60)
print("📊 SUMMARY RESULTS")
print("=" * 60)

for name, result in results.items():
    status = "✅" if "SUCCESS" in result else "⚠️" if "IMPORT_OK" in result else "❌"
    print(f"{status} {name:20} -> {result}")

print("\n🔍 DETAILED DEPENDENCY CHECK")
print("-" * 40)

# Check critical dependencies
critical_deps = [
    ("llama-index.core", "llama_index.core"),
    ("llama-index.llms.openai", "llama_index.llms.openai"),
    ("llama-index.embeddings.openai", "llama_index.embeddings.openai"),
    ("llama-index.vector_stores.qdrant", "llama_index.vector_stores.qdrant"),
    ("qdrant_client", "qdrant_client"),
    ("openai", "openai"),
    ("yaml", "yaml"),
    ("structlog", "structlog"),
]

print("\n📋 Testing critical dependencies:")
for dep_name, import_path in critical_deps:
    try:
        __import__(import_path)
        print(f"  ✅ {dep_name}")
    except ImportError as e:
        print(f"  ❌ {dep_name}: {e}")

print("\n🎯 NEXT STEPS RECOMMENDATIONS:")
print("-" * 40)

failed_imports = [name for name, result in results.items() if "FAILED" in result or "ERROR" in result]
if failed_imports:
    print("❌ Failed imports found. Recommended actions:")
    for name in failed_imports:
        print(f"   • Fix {name}: {results[name]}")
else:
    print("✅ All imports successful!")
    
init_failed = [name for name, result in results.items() if "INIT_FAILED" in result]
if init_failed:
    print("⚠️  Import OK but initialization failed:")
    for name in init_failed:
        print(f"   • Check {name} configuration requirements")

print("\n🏁 Deep dive complete!")