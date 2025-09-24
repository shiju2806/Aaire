#!/usr/bin/env python3
"""
Debug script to investigate Qdrant text extraction for Whoosh indexing
"""

import os
import sys
import json
from qdrant_client import QdrantClient

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def extract_text_content(payload):
    """Extract text content from Qdrant payload, handling JSON structure properly."""
    # Try direct text field first
    if payload.get('text'):
        return payload.get('text')
    if payload.get('content'):
        return payload.get('content')

    # Handle _node_content JSON structure
    if payload.get('_node_content'):
        try:
            if isinstance(payload['_node_content'], str):
                node_data = json.loads(payload['_node_content'])
                return node_data.get('text', '')
            elif isinstance(payload['_node_content'], dict):
                return payload['_node_content'].get('text', '')
        except (json.JSONDecodeError, AttributeError):
            pass

    # Fallback to string representation
    return str(payload)

def main():
    print("🔍 QDRANT -> WHOOSH TEXT EXTRACTION DEBUG")
    print("=" * 60)

    # Connect to Qdrant
    try:
        client = QdrantClient(
            host="ce8b5f05-c0a2-47b1-a761-c2f9e6f73817.europe-west3-0.gcp.cloud.qdrant.io",
            port=6333,
            https=True
        )

        collection_name = "aaire-documents"
        print(f"📊 Connected to Qdrant collection: {collection_name}")

        # Get collection info
        collection_info = client.get_collection(collection_name)
        print(f"📈 Collection has {collection_info.points_count} points")

        # Sample a few documents to see text extraction
        response, _ = client.scroll(
            collection_name=collection_name,
            limit=10,
            with_payload=True,
            with_vectors=False
        )

        print(f"\n🔍 Analyzing {len(response)} sample documents:")
        print("-" * 60)

        meaningful_count = 0
        empty_count = 0
        short_count = 0

        for i, point in enumerate(response):
            payload = point.payload

            print(f"\n📄 Document {i+1}:")
            print(f"   Point ID: {point.id}")
            print(f"   Filename: {payload.get('filename', 'Unknown')}")

            # Check what fields exist
            print(f"   Available fields: {list(payload.keys())}")

            # Try our text extraction
            extracted_text = extract_text_content(payload)
            text_length = len(extracted_text.strip()) if extracted_text else 0

            print(f"   Extracted text length: {text_length}")

            if text_length > 10:
                meaningful_count += 1
                print(f"   ✅ MEANINGFUL CONTENT (first 100 chars):")
                print(f"      {extracted_text[:100]}...")
            elif text_length > 0:
                short_count += 1
                print(f"   ⚠️  SHORT CONTENT ({text_length} chars): {extracted_text}")
            else:
                empty_count += 1
                print(f"   ❌ NO TEXT CONTENT")

                # Debug what's in _node_content
                if payload.get('_node_content'):
                    node_content = payload['_node_content']
                    print(f"      _node_content type: {type(node_content)}")
                    if isinstance(node_content, str):
                        print(f"      _node_content preview: {node_content[:200]}...")
                    else:
                        print(f"      _node_content keys: {list(node_content.keys()) if isinstance(node_content, dict) else 'Not a dict'}")

        print(f"\n📊 SUMMARY:")
        print(f"   ✅ Meaningful content (>10 chars): {meaningful_count}")
        print(f"   ⚠️  Short content (1-10 chars): {short_count}")
        print(f"   ❌ No content (0 chars): {empty_count}")
        print(f"   📈 Success rate: {meaningful_count/len(response)*100:.1f}%")

        if meaningful_count == 0:
            print(f"\n🚨 PROBLEM IDENTIFIED: No documents have meaningful text content!")
            print(f"   This explains why Whoosh has 0 documents.")

    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")

if __name__ == "__main__":
    main()