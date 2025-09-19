#!/usr/bin/env python3
"""
Inspect the actual content in Qdrant to understand what's stored
"""
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

load_dotenv()

from qdrant_client import QdrantClient

def inspect_qdrant_content():
    """Inspect actual content in Qdrant"""
    print("🔍 INSPECTING QDRANT CONTENT")
    print("=" * 60)

    try:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url:
            print("❌ QDRANT_URL not set in environment")
            return

        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        collection_name = "aaire-documents"

        # Get first few points to inspect content
        search_result = client.scroll(
            collection_name=collection_name,
            limit=10
        )

        points = search_result[0]
        print(f"📊 Inspecting first {len(points)} chunks:")

        for i, point in enumerate(points, 1):
            print(f"\n--- CHUNK {i} ---")
            print(f"Point ID: {point.id}")

            if point.payload:
                filename = point.payload.get("filename", "Unknown")
                job_id = point.payload.get("job_id", "No job_id")
                doc_type = point.payload.get("doc_type", "Unknown")
                text = point.payload.get("text", "")

                print(f"Filename: {filename}")
                print(f"Job ID: {job_id}")
                print(f"Doc Type: {doc_type}")
                print(f"Text length: {len(text)} characters")
                print(f"Text preview (first 200 chars): {text[:200]}...")

                # Check for reserve terms in this specific chunk
                text_lower = text.lower()
                reserve_terms_found = []

                terms_to_check = [
                    'deterministic', 'stochastic', 'reserve', 'reserves',
                    'dr', 'sr', 'npr', 'premium', 'net premium'
                ]

                for term in terms_to_check:
                    if term in text_lower:
                        reserve_terms_found.append(term)

                if reserve_terms_found:
                    print(f"🎯 RESERVE TERMS FOUND: {', '.join(reserve_terms_found)}")
                    # Show context around terms
                    for term in reserve_terms_found[:3]:  # First 3 terms
                        start = text_lower.find(term)
                        if start != -1:
                            context_start = max(0, start - 50)
                            context_end = min(len(text), start + len(term) + 50)
                            context = text[context_start:context_end]
                            print(f"   '{term}' context: ...{context}...")
                else:
                    print("❌ No reserve terms found in this chunk")
            else:
                print("❌ No payload found")

        # Try to search for chunks that might contain reserve terms
        print(f"\n🔍 SEARCHING ALL CHUNKS FOR RESERVE TERMS:")

        all_result = client.scroll(
            collection_name=collection_name,
            limit=1000  # Get more chunks
        )

        all_points = all_result[0]
        print(f"📊 Total chunks to search: {len(all_points)}")

        chunks_with_reserves = []

        for point in all_points:
            if point.payload and point.payload.get("text"):
                text = point.payload.get("text", "").lower()
                filename = point.payload.get("filename", "Unknown")

                # Check for any reserve-related terms
                if any(term in text for term in ['deterministic', 'stochastic', 'reserve', 'dr', 'sr', 'npr']):
                    chunks_with_reserves.append({
                        'point_id': point.id,
                        'filename': filename,
                        'text': point.payload.get("text", "")
                    })

        if chunks_with_reserves:
            print(f"✅ Found {len(chunks_with_reserves)} chunks with reserve terms!")

            for i, chunk in enumerate(chunks_with_reserves[:5], 1):  # Show first 5
                print(f"\n📄 Chunk {i}:")
                print(f"   Point ID: {chunk['point_id']}")
                print(f"   Filename: {chunk['filename']}")
                print(f"   Text preview: {chunk['text'][:300]}...")
        else:
            print("❌ No chunks found with reserve terms")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_qdrant_content()