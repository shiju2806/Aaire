#!/usr/bin/env python3
"""
Phase 4 Complete RAG Pipeline Test
Tests the fully refactored modular RAG pipeline after cleanup
"""

import sys
import os
sys.path.append('src')

def test_pipeline_import():
    """Test that the refactored pipeline imports successfully"""
    print("🧪 Testing pipeline import...")

    try:
        from src.rag_pipeline import RAGPipeline
        print("✅ RAGPipeline imports successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_modular_components_import():
    """Test that all modular components import successfully"""
    print("\n🧪 Testing modular components import...")

    try:
        # Core components
        from src.rag_modules.core.response import RAGResponse
        print("✅ RAGResponse imports successfully")

        # Analysis components
        from src.rag_modules.analysis.citations import CitationAnalyzer
        print("✅ CitationAnalyzer imports successfully")

        # Cache components
        from src.rag_modules.cache.manager import CacheManager
        print("✅ CacheManager imports successfully")

        # Formatting components
        from src.rag_modules.formatting import FormattingManager
        print("✅ FormattingManager imports successfully")

        # Query components
        from src.rag_modules.query import QueryAnalyzer
        print("✅ QueryAnalyzer imports successfully")

        # Quality components
        from src.rag_modules.quality import QualityMetricsManager
        print("✅ QualityMetricsManager imports successfully")

        # Services components
        from src.rag_modules.services import DocumentRetriever, ResponseGenerator
        print("✅ DocumentRetriever and ResponseGenerator import successfully")

        # Storage components
        from src.rag_modules.storage import DocumentManager
        print("✅ DocumentManager imports successfully")

        return True

    except Exception as e:
        print(f"❌ Modular component import failed: {e}")
        return False

def test_pipeline_structure():
    """Test the pipeline class structure and methods"""
    print("\n🔍 Testing pipeline structure...")

    try:
        from src.rag_pipeline import RAGPipeline

        # Check critical methods exist
        critical_methods = [
            '__init__',
            'process_query',
            '_init_hybrid_search',
            '_populate_whoosh_from_existing_documents'
        ]

        for method in critical_methods:
            if hasattr(RAGPipeline, method):
                print(f"✅ {method} found")
            else:
                print(f"❌ {method} missing")
                return False

        # Check that old extracted methods are gone
        old_methods = [
            '_extract_citations',
            '_format_response',
            '_generate_response',
            '_retrieve_documents',
            '_calculate_confidence'
        ]

        for method in old_methods:
            if hasattr(RAGPipeline, method):
                print(f"⚠️ Old method {method} still present (should be removed)")
                return False
            else:
                print(f"✅ Old method {method} successfully removed")

        return True

    except Exception as e:
        print(f"❌ Pipeline structure test failed: {e}")
        return False

def test_file_sizes():
    """Test that file sizes are reasonable after refactoring"""
    print("\n📊 Testing file sizes...")

    try:
        pipeline_path = "/Users/shijuprakash/AAIRE/src/rag_pipeline.py"
        if os.path.exists(pipeline_path):
            with open(pipeline_path, 'r') as f:
                lines = len(f.readlines())
            print(f"📄 Main pipeline: {lines} lines")

            if lines < 2000:
                print("✅ Main pipeline successfully reduced in size")
            else:
                print(f"⚠️ Main pipeline still large: {lines} lines")
                return False

        # Check modular components exist and have reasonable sizes
        module_paths = [
            "/Users/shijuprakash/AAIRE/src/rag_modules/core/response.py",
            "/Users/shijuprakash/AAIRE/src/rag_modules/analysis/citations.py",
            "/Users/shijuprakash/AAIRE/src/rag_modules/cache/manager.py",
            "/Users/shijuprakash/AAIRE/src/rag_modules/formatting/manager.py",
            "/Users/shijuprakash/AAIRE/src/rag_modules/query/analyzer.py",
            "/Users/shijuprakash/AAIRE/src/rag_modules/quality/metrics.py",
            "/Users/shijuprakash/AAIRE/src/rag_modules/services/retrieval.py",
            "/Users/shijuprakash/AAIRE/src/rag_modules/services/generation.py",
            "/Users/shijuprakash/AAIRE/src/rag_modules/storage/documents.py"
        ]

        total_modular_lines = 0
        for path in module_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    lines = len(f.readlines())
                total_modular_lines += lines
                module_name = os.path.basename(path)
                print(f"📄 {module_name}: {lines} lines")

        print(f"📄 Total modular components: {total_modular_lines} lines")
        print(f"📊 Refactoring ratio: {lines} main + {total_modular_lines} modular = {lines + total_modular_lines} total")

        return True

    except Exception as e:
        print(f"❌ File size test failed: {e}")
        return False

def test_module_structure():
    """Test the overall modular structure"""
    print("\n🏗️ Testing module structure...")

    try:
        module_dirs = [
            "/Users/shijuprakash/AAIRE/src/rag_modules/core",
            "/Users/shijuprakash/AAIRE/src/rag_modules/analysis",
            "/Users/shijuprakash/AAIRE/src/rag_modules/cache",
            "/Users/shijuprakash/AAIRE/src/rag_modules/formatting",
            "/Users/shijuprakash/AAIRE/src/rag_modules/query",
            "/Users/shijuprakash/AAIRE/src/rag_modules/quality",
            "/Users/shijuprakash/AAIRE/src/rag_modules/services",
            "/Users/shijuprakash/AAIRE/src/rag_modules/storage"
        ]

        for module_dir in module_dirs:
            if os.path.exists(module_dir):
                init_file = os.path.join(module_dir, "__init__.py")
                if os.path.exists(init_file):
                    print(f"✅ {os.path.basename(module_dir)} module properly structured")
                else:
                    print(f"⚠️ {os.path.basename(module_dir)} missing __init__.py")
            else:
                print(f"❌ {os.path.basename(module_dir)} module missing")
                return False

        return True

    except Exception as e:
        print(f"❌ Module structure test failed: {e}")
        return False

def main():
    """Run comprehensive Phase 4 test suite"""
    print("🚀 Starting Phase 4 Complete RAG Pipeline Test Suite")
    print("=" * 70)

    tests = [
        test_pipeline_import,
        test_modular_components_import,
        test_pipeline_structure,
        test_file_sizes,
        test_module_structure
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_func.__name__} PASSED")
            else:
                print(f"❌ {test_func.__name__} FAILED")
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED with exception: {e}")

    print("\n" + "=" * 70)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 Phase 4 refactoring COMPLETE! All tests passed!")
        print("🏆 RAG pipeline successfully modularized and cleaned up!")
        return True
    else:
        print("⚠️ Some tests failed. Phase 4 needs attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)