#!/usr/bin/env python3
"""
Debug Whoosh Parser Bug
Test the MultifieldParser with NLP-generated queries to find the exact issue
"""

import sys
import os
from whoosh import index
from whoosh.qparser import MultifieldParser

def test_whoosh_parser():
    print("🔍 Testing Whoosh Parser with NLP Queries")
    print("=" * 60)

    # Try to open the Whoosh index
    try:
        index_path = "/Users/shijuprakash/AAIRE/simple_search_index"
        if not os.path.exists(index_path):
            print(f"❌ Index not found at: {index_path}")
            return

        ix = index.open_dir(index_path)
        print(f"✅ Opened Whoosh index: {index_path}")
        print(f"📊 Index schema fields: {list(ix.schema.names())}")

    except Exception as e:
        print(f"❌ Failed to open index: {e}")
        return

    # Test queries that are causing issues
    test_queries = [
        # Simple queries (should work)
        "whole life policy",
        "universal life policy",

        # NLP-generated queries (likely to fail)
        '"life policy" usstat reserve calculate policy',
        '"life policy" "universal life policy" usstat reserve universal calculate',

        # Malformed queries from logs
        '"life policy" "universal calculate reserve" ""life policy" policy" policy universal calculate reserve'
    ]

    with ix.searcher() as searcher:
        parser = MultifieldParser(["title", "content"], ix.schema)

        for query_text in test_queries:
            print(f"\n📝 Testing Query: '{query_text}'")

            try:
                # Try to parse the query
                parsed_query = parser.parse(query_text)
                print(f"✅ Parsed successfully: {parsed_query}")

                # Try to execute the search
                results = searcher.search(parsed_query, limit=5)
                print(f"🔍 Search results: {len(results)} found")

                for i, result in enumerate(results[:2]):
                    title = result.get('title', 'No title')[:50]
                    score = result.score
                    print(f"   {i+1}. {title}... (score: {score:.3f})")

            except Exception as e:
                print(f"❌ Parser/Search failed: {str(e)}")
                print(f"   Error type: {type(e).__name__}")

if __name__ == "__main__":
    test_whoosh_parser()