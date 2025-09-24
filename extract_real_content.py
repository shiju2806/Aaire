#!/usr/bin/env python3
"""
Extract and search the real text content from _node_content field
"""
import os
import sys
import json
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

load_dotenv()

from qdrant_client import QdrantClient

def extract_and_search_content():
    """Extract real text content from _node_content and search for Universal Life terms"""
    print("🔍 EXTRACTING REAL TEXT CONTENT FROM _NODE_CONTENT")
    print("=" * 60)

    try:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url:
            print("❌ QDRANT_URL not set in environment")
            return

        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        collection_name = "aaire-documents"

        # Get more points to search
        search_result = client.scroll(
            collection_name=collection_name,
            limit=100  # Get first 100 chunks
        )

        points = search_result[0]
        print(f"📊 Searching {len(points)} chunks for Universal Life content:")

        # Search terms for Universal Life
        search_terms = [
            'universal life', 'vm-20', 'deterministic reserve', 'stochastic reserve',
            'dr', 'sr', 'npr', 'net premium reserve', 'principle-based reserves',
            'pbr', 'valuation manual', 'statutory reserve'
        ]

        chunks_with_ul_content = []
        total_chunks_with_text = 0

        for i, point in enumerate(points):
            if point.payload and '_node_content' in point.payload:
                try:
                    # Parse the JSON content
                    node_data = json.loads(point.payload['_node_content'])
                    text_content = node_data.get('text', '')

                    if text_content:
                        total_chunks_with_text += 1
                        text_lower = text_content.lower()
                        filename = point.payload.get('filename', 'Unknown')

                        # Check for Universal Life related terms
                        found_terms = []
                        for term in search_terms:
                            if term.lower() in text_lower:
                                found_terms.append(term)

                        if found_terms:
                            chunks_with_ul_content.append({
                                'point_id': point.id,
                                'filename': filename,
                                'job_id': point.payload.get('job_id', 'No job_id'),
                                'text': text_content,
                                'text_length': len(text_content),
                                'found_terms': found_terms,
                                'chunk_index': point.payload.get('chunk_index', 'Unknown')
                            })

                            print(f"✅ FOUND UNIVERSAL LIFE CONTENT in chunk {i+1}:")
                            print(f"   File: {filename}")
                            print(f"   Terms found: {', '.join(found_terms)}")
                            print(f"   Text length: {len(text_content)} chars")
                            print(f"   Text preview: {text_content[:200]}...")
                            print()

                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error for point {point.id}: {e}")
                except Exception as e:
                    print(f"❌ Error processing point {point.id}: {e}")

        print(f"\n📊 SUMMARY:")
        print(f"   Total chunks examined: {len(points)}")
        print(f"   Chunks with text content: {total_chunks_with_text}")
        print(f"   Chunks with Universal Life content: {len(chunks_with_ul_content)}")

        if chunks_with_ul_content:
            print(f"\n📄 UNIVERSAL LIFE CHUNKS FOUND:")
            for i, chunk in enumerate(chunks_with_ul_content[:5], 1):  # Show first 5
                print(f"\n{i}. File: {chunk['filename']}")
                print(f"   Chunk index: {chunk['chunk_index']}")
                print(f"   Terms: {', '.join(chunk['found_terms'])}")
                print(f"   Length: {chunk['text_length']} chars")

                # Show relevant snippets
                text_lower = chunk['text'].lower()
                for term in chunk['found_terms'][:2]:  # Show context for first 2 terms
                    start_idx = text_lower.find(term.lower())
                    if start_idx != -1:
                        context_start = max(0, start_idx - 100)
                        context_end = min(len(chunk['text']), start_idx + len(term) + 100)
                        context = chunk['text'][context_start:context_end]
                        print(f"   '{term}' context: ...{context}...")

        # Check if we need to scroll more documents
        if len(points) >= 100:
            print(f"\n🔄 NOTE: Only checked first 100 chunks. There may be more content.")
            print(f"   Total chunks in collection: Use scroll with offset to get more.")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    extract_and_search_content()