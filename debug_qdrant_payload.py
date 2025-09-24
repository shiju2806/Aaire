#!/usr/bin/env python3
"""
Debug script to examine exactly what fields are in Qdrant payload
"""
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

load_dotenv()

from qdrant_client import QdrantClient
import json

def debug_qdrant_payload():
    """Debug what's actually stored in payload"""
    print("🔍 DEBUGGING QDRANT PAYLOAD FIELDS")
    print("=" * 60)

    try:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url:
            print("❌ QDRANT_URL not set in environment")
            return

        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        collection_name = "aaire-documents"

        # Get first few points to inspect all payload fields
        search_result = client.scroll(
            collection_name=collection_name,
            limit=3
        )

        points = search_result[0]
        print(f"📊 Inspecting {len(points)} chunks for all payload fields:")

        for i, point in enumerate(points, 1):
            print(f"\n--- CHUNK {i} ---")
            print(f"Point ID: {point.id}")

            if point.payload:
                print("📋 ALL PAYLOAD FIELDS:")
                for key, value in point.payload.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"  {key}: '{value[:100]}...' (length: {len(value)})")
                    else:
                        print(f"  {key}: {value}")

                # Check all possible text field names
                possible_text_fields = ['text', 'content', '_node_content', 'node_content', 'body', 'raw_text']
                print(f"\n🔍 CHECKING POSSIBLE TEXT FIELDS:")
                for field in possible_text_fields:
                    if field in point.payload:
                        value = point.payload[field]
                        print(f"  ✅ {field}: present, length={len(str(value))}")
                        if len(str(value)) > 50:
                            print(f"    Preview: '{str(value)[:50]}...'")
                    else:
                        print(f"  ❌ {field}: not found")
            else:
                print("❌ No payload found")

        # Check collection info
        print(f"\n🗂️ COLLECTION INFO:")
        collection_info = client.get_collection(collection_name)
        print(f"  Total vectors: {collection_info.vectors_count}")
        print(f"  Status: {collection_info.status}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_qdrant_payload()