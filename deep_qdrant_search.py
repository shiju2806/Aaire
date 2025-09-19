#!/usr/bin/env python3
"""
Deep search of Qdrant to find text containing reserve terminology
"""
import os
import sys
import re
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

load_dotenv()

from qdrant_client import QdrantClient

def deep_search_qdrant():
    """Deep search Qdrant for any mention of reserve terms"""
    print("🔍 DEEP SEARCH OF QDRANT FOR RESERVE TERMINOLOGY")
    print("=" * 80)

    try:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url:
            print("❌ QDRANT_URL not set in environment")
            return

        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        collection_name = "aaire-documents"

        # Get all points
        search_result = client.scroll(
            collection_name=collection_name,
            limit=1000
        )

        points = search_result[0]
        print(f"📊 Searching {len(points)} documents in Qdrant...")

        # Broader search terms
        search_words = [
            'deterministic',
            'stochastic',
            'reserve',
            'reserves',
            'DR',
            'SR',
            'NPR',
            'premium'
        ]

        # Track findings
        findings = {}
        reserve_documents = []

        for point in points:
            if point.payload and point.payload.get("text"):
                text = point.payload.get("text", "")
                text_lower = text.lower()
                filename = point.payload.get("filename", "Unknown")
                job_id = point.payload.get("job_id", "No job_id")

                # Check for reserve-related content
                has_reserve_terms = False
                found_terms = []

                for word in search_words:
                    if word.lower() in text_lower:
                        found_terms.append(word)
                        has_reserve_terms = True

                if has_reserve_terms:
                    # Look for specific combinations
                    combinations_found = []

                    # Check for deterministic + reserve
                    if 'deterministic' in text_lower and 'reserve' in text_lower:
                        combinations_found.append('deterministic+reserve')

                    # Check for stochastic + reserve
                    if 'stochastic' in text_lower and 'reserve' in text_lower:
                        combinations_found.append('stochastic+reserve')

                    # Check for DR mentions
                    if ' dr ' in text_lower or text_lower.startswith('dr ') or text_lower.endswith(' dr'):
                        combinations_found.append('DR_standalone')

                    # Check for SR mentions
                    if ' sr ' in text_lower or text_lower.startswith('sr ') or text_lower.endswith(' sr'):
                        combinations_found.append('SR_standalone')

                    if combinations_found:
                        doc_info = {
                            'filename': filename,
                            'job_id': job_id,
                            'point_id': point.id,
                            'combinations_found': combinations_found,
                            'text_snippet': text[:500] + "..." if len(text) > 500 else text,
                            'found_terms': found_terms
                        }
                        reserve_documents.append(doc_info)

                        # Track by filename
                        if filename not in findings:
                            findings[filename] = {
                                'job_id': job_id,
                                'chunks': [],
                                'all_combinations': set(),
                                'all_terms': set()
                            }

                        findings[filename]['chunks'].append(doc_info)
                        findings[filename]['all_combinations'].update(combinations_found)
                        findings[filename]['all_terms'].update(found_terms)

        # Report findings
        if findings:
            print(f"\n✅ Found {len(findings)} documents with reserve-related content:")
            print(f"📊 Total chunks with reserve content: {len(reserve_documents)}")

            print("\n📋 DOCUMENTS WITH RESERVE TERMINOLOGY:")
            print("=" * 60)

            for filename, data in findings.items():
                print(f"\n📄 {filename}")
                print(f"   Job ID: {data['job_id']}")
                print(f"   Chunks with reserve content: {len(data['chunks'])}")
                print(f"   Combinations found: {', '.join(sorted(data['all_combinations']))}")
                print(f"   All terms found: {', '.join(sorted(data['all_terms']))}")

                # Show a few example chunks
                print(f"   Example chunks:")
                for i, chunk in enumerate(data['chunks'][:2], 1):
                    print(f"   {i}. Point ID: {chunk['point_id']}")
                    print(f"      Combinations: {', '.join(chunk['combinations_found'])}")
                    print(f"      Text preview: {chunk['text_snippet'][:200]}...")
                    print()

                if len(data['chunks']) > 2:
                    print(f"   ... and {len(data['chunks']) - 2} more chunks")

                print("-" * 40)

            # Summary by combination type
            print(f"\n📊 SUMMARY BY COMBINATION TYPE:")
            combination_counts = {}
            for doc_data in findings.values():
                for combo in doc_data['all_combinations']:
                    combination_counts[combo] = combination_counts.get(combo, 0) + 1

            for combo, count in sorted(combination_counts.items()):
                print(f"   {combo}: {count} documents")

        else:
            print("❌ No documents found containing reserve-related combinations")

        # Also check for simpler patterns
        print(f"\n🔍 CHECKING FOR BROADER PATTERNS:")
        simple_counts = {}
        for word in ['reserve', 'reserves', 'premium', 'deterministic', 'stochastic']:
            count = 0
            for point in points:
                if point.payload and point.payload.get("text"):
                    text = point.payload.get("text", "").lower()
                    if word.lower() in text:
                        count += 1
            simple_counts[word] = count
            print(f"   '{word}': found in {count} chunks")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    deep_search_qdrant()