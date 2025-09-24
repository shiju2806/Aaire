#!/usr/bin/env python3
"""
Debug script to investigate Whoosh indexing issues
"""

import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from whoosh_search_engine import WhooshSearchEngine
from pathlib import Path

def main():
    print("🔍 WHOOSH DEBUG INVESTIGATION")
    print("=" * 50)

    # Initialize Whoosh search engine
    index_dir = Path("whoosh_index")
    whoosh = WhooshSearchEngine(index_dir=index_dir)

    print(f"📁 Whoosh index directory: {index_dir}")
    print(f"📊 Document count: {whoosh.get_document_count()}")

    # Test a simple search
    test_queries = [
        "universal life",
        "usstat",
        "reserve",
        "valuation manual",
        "VM-20"
    ]

    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        results = whoosh.search(query, limit=5)
        print(f"   Results: {len(results)}")

        if results:
            for i, result in enumerate(results[:2]):
                print(f"   [{i+1}] {result.get('title', 'No title')[:50]}...")
                content_preview = result.get('content', '')[:100]
                print(f"       Content: {content_preview}...")
        else:
            print("   ❌ No results found")

    # Check if we can access the index directly
    print(f"\n📚 Direct index investigation:")
    try:
        if whoosh.index and whoosh.index.doc_count() > 0:
            with whoosh.index.searcher() as searcher:
                print(f"   Total indexed documents: {searcher.doc_count()}")

                # Sample a few documents
                print("   Sample documents:")
                for i, doc in enumerate(searcher.documents()):
                    if i >= 3:  # Only show first 3
                        break
                    title = doc.get('title', 'No title')
                    content = doc.get('content', '')
                    print(f"   [{i+1}] Title: {title[:50]}...")
                    print(f"       Content length: {len(content)}")
                    print(f"       Content preview: {content[:100]}...")
                    print()
        else:
            print("   ❌ Index is empty or unavailable")

    except Exception as e:
        print(f"   ❌ Error accessing index: {e}")

if __name__ == "__main__":
    main()