#!/usr/bin/env python3
"""
Debug NLP Quote Handling Bug
Test the NLP processor to find where malformed quotes are being generated
"""

import sys
sys.path.append('/Users/shijuprakash/AAIRE/src')

from nlp_query_processor import NLPQueryProcessor

def test_nlp_quote_handling():
    print("🔍 Testing NLP Quote Handling Bug")
    print("=" * 50)

    processor = NLPQueryProcessor()

    test_queries = [
        "how do I calculate the reserves for a whole life policy in usstat",
        "how do I calculate the reserves for a universal life policy in usstat"
    ]

    for query in test_queries:
        print(f"\n📝 Original Query: '{query}'")

        # Process the query
        processed = processor.process_query(query)

        print(f"🧠 Key Phrases: {processed.key_phrases}")
        print(f"🔍 Key Entities: {processed.key_entities}")
        print(f"📋 Semantic Keywords: {processed.semantic_keywords}")
        print(f"🎯 Intent: {processed.query_intent}")

        # Generate search queries in different modes
        modes = ["focused", "balanced", "semantic", "precise"]
        for mode in modes:
            search_query = processor.generate_search_query(processed, mode)
            print(f"🔎 {mode.upper()}: '{search_query}'")

        print("-" * 40)

if __name__ == "__main__":
    test_nlp_quote_handling()