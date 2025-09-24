#!/usr/bin/env python3
"""
Debug script to investigate what documents are being retrieved for whole life queries
"""

import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("🔍 DOCUMENT RETRIEVAL DEBUG FOR WHOLE LIFE QUERY")
    print("=" * 70)

    try:
        # Import and initialize the RAG pipeline
        from rag_pipeline import RAGPipeline

        print("📊 Initializing RAG Pipeline...")
        rag = RAGPipeline()

        # Test query
        query = "how do I calculate the reserves for a whole life policy in usstat"
        print(f"🔍 Testing query: '{query}'")

        # Get the vector store directly to see what it retrieves
        print(f"\n📚 Vector search results:")
        vector_results = rag.vector_store.similarity_search_with_score(query, k=10)

        print(f"   Found {len(vector_results)} vector results:")

        for i, (doc, score) in enumerate(vector_results):
            print(f"\n   [{i+1}] Score: {score:.3f}")
            print(f"       Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"       Source Type: {doc.metadata.get('source_type', 'Unknown')}")
            print(f"       Page: {doc.metadata.get('page', 'N/A')}")
            print(f"       Content (first 200 chars): {doc.page_content[:200]}...")

            # Check for specific terms
            content_lower = doc.page_content.lower()
            metadata_text = str(doc.metadata).lower()

            flags = []
            if 'ifrs' in content_lower or 'ifrs' in metadata_text:
                flags.append("🚨 IFRS")
            if 'usstat' in content_lower or 'usstat' in metadata_text:
                flags.append("✅ USSTAT")
            if 'universal life' in content_lower:
                flags.append("🔄 Universal Life")
            if 'whole life' in content_lower:
                flags.append("✅ Whole Life")
            if 'whole contract' in content_lower:
                flags.append("🚨 Whole Contract (UL concept)")
            if 'core cash flow' in content_lower:
                flags.append("🚨 Core Cash Flow (UL concept)")

            if flags:
                print(f"       Flags: {' | '.join(flags)}")

        # Test Whoosh search separately
        print(f"\n🔍 Whoosh search results:")
        try:
            whoosh_results = rag.whoosh_engine.search(query, limit=10)
            print(f"   Found {len(whoosh_results)} whoosh results:")

            for i, result in enumerate(whoosh_results):
                print(f"   [{i+1}] {result.get('title', 'No title')[:50]}...")
                print(f"       Source: {result.get('source', 'Unknown')}")

        except Exception as e:
            print(f"   ❌ Whoosh search failed: {e}")

        # Test the complete pipeline to see final aggregation
        print(f"\n🔄 Complete pipeline test:")
        try:
            response = rag.query(query)
            print(f"   Response confidence: {response.get('confidence', 'N/A')}")
            print(f"   Citations count: {len(response.get('citations', []))}")

            for i, citation in enumerate(response.get('citations', [])):
                print(f"   Citation [{i+1}]: {citation.get('source', 'Unknown')} (confidence: {citation.get('confidence', 'N/A')})")
                print(f"       Text preview: {citation.get('text', '')[:100]}...")

        except Exception as e:
            print(f"   ❌ Complete pipeline failed: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()