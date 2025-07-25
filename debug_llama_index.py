#!/usr/bin/env python3
"""
Debug llama-index actual module structure
"""
import sys

print("🔍 LlamaIndex Module Structure Analysis")
print("=" * 50)

# Test what's actually available
test_imports = [
    # Core imports
    ("llama_index", "llama_index"),
    ("llama_index.core", "llama_index.core"),
    
    # Try different paths for VectorStoreIndex
    ("llama_index.VectorStoreIndex", "llama_index", "VectorStoreIndex"),
    ("llama_index.core.VectorStoreIndex", "llama_index.core", "VectorStoreIndex"), 
    ("llama_index.indices.VectorStoreIndex", "llama_index.indices", "VectorStoreIndex"),
    
    # Try different paths for Document
    ("llama_index.Document", "llama_index", "Document"),
    ("llama_index.core.Document", "llama_index.core", "Document"),
    ("llama_index.schema.Document", "llama_index.schema", "Document"),
    
    # Vector stores
    ("llama_index.vector_stores", "llama_index.vector_stores"),
    ("llama_index.vector_stores.qdrant", "llama_index.vector_stores.qdrant"),
    
    # Other core components
    ("llama_index.ServiceContext", "llama_index", "ServiceContext"),
    ("llama_index.StorageContext", "llama_index", "StorageContext"),
    ("llama_index.Settings", "llama_index", "Settings"),
    ("llama_index.core.Settings", "llama_index.core", "Settings"),
]

print("📦 Testing available imports...")

working_imports = {}
failed_imports = {}

for test_name, module_path, *class_name in test_imports:
    try:
        if class_name:
            # Test importing a specific class
            module = __import__(module_path, fromlist=[class_name[0]])
            cls = getattr(module, class_name[0])
            print(f"✅ {test_name} -> {cls}")
            working_imports[test_name] = (module_path, class_name[0])
        else:
            # Test importing a module
            module = __import__(module_path)
            print(f"✅ {test_name} -> module available")
            working_imports[test_name] = (module_path, None)
    except (ImportError, AttributeError) as e:
        print(f"❌ {test_name} -> {e}")
        failed_imports[test_name] = str(e)

print("\n" + "=" * 50)
print("📋 WORKING IMPORTS SUMMARY")
print("=" * 50)

for name, (module_path, class_name) in working_imports.items():
    if class_name:
        print(f"✅ from {module_path} import {class_name}")
    else:
        print(f"✅ import {module_path}")

print("\n📦 Try exploring llama_index structure...")
try:
    import llama_index
    print(f"llama_index.__file__ = {getattr(llama_index, '__file__', 'N/A')}")
    print(f"llama_index.__version__ = {getattr(llama_index, '__version__', 'N/A')}")
    
    # List available attributes
    attrs = [attr for attr in dir(llama_index) if not attr.startswith('_')]
    print(f"Available in llama_index: {attrs[:10]}...")  # Show first 10
    
    # Check for core classes
    core_classes = ['VectorStoreIndex', 'Document', 'ServiceContext', 'StorageContext', 'Settings']
    for cls_name in core_classes:
        if hasattr(llama_index, cls_name):
            print(f"✅ llama_index.{cls_name} available")
        else:
            print(f"❌ llama_index.{cls_name} not found")
            
except ImportError as e:
    print(f"❌ Cannot import llama_index: {e}")

print("\n🔍 Checking for specific modules...")
import pkgutil
try:
    import llama_index
    for importer, modname, ispkg in pkgutil.iter_modules(llama_index.__path__, 'llama_index.'):
        if 'core' in modname or 'vector' in modname or 'indices' in modname:
            print(f"📦 Found: {modname}")
except:
    print("❌ Cannot explore llama_index modules")

print("\n🏁 Analysis complete!")