#!/usr/bin/env python3
"""Test the enhanced RAG features with two queries"""

import requests
import json
import time

# API endpoint
url = "http://localhost:8000/api/v1/chat"

# Test queries
queries = [
    {
        "name": "Universal Life (should work well)",
        "query": "how do I calculate the reserves for a universal life policy in usstat",
        "session_id": "test-universal-life"
    },
    {
        "name": "Whole Life (should have limited/no content)",
        "query": "how do I calculate the reserves for a whole life policy in usstat",
        "session_id": "test-whole-life"
    }
]

print("="*80)
print("TESTING ENHANCED RAG FEATURES")
print("="*80)

for test_case in queries:
    print(f"\n📝 Testing: {test_case['name']}")
    print(f"Query: {test_case['query']}")
    print("-"*40)

    # Make request
    response = requests.post(
        url,
        json={
            "query": test_case["query"],
            "session_id": test_case["session_id"]
        }
    )

    if response.status_code == 200:
        result = response.json()

        print(f"✅ Response received successfully")
        print(f"Confidence: {result.get('confidence', 'N/A')}")

        # Check if citations were found
        citations = result.get('citations', [])
        print(f"Citations found: {len(citations)}")
        if citations:
            for cite in citations[:3]:  # Show first 3 citations
                print(f"  - {cite.get('source', 'Unknown source')}")

        # Show first part of response
        answer = result.get('answer', 'No answer')
        print(f"\nResponse preview (first 500 chars):")
        print(answer[:500] + "..." if len(answer) > 500 else answer)

        # Check for enhanced features metadata
        if 'metadata' in result:
            print(f"\nEnhanced features used: {result['metadata']}")

    else:
        print(f"❌ Request failed with status: {response.status_code}")
        print(f"Error: {response.text}")

    print("\n" + "="*80)
    time.sleep(2)  # Small delay between requests

print("\n✅ Test complete!")