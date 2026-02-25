#!/usr/bin/env python3
"""
Targeted search for specific reserve terminology to provide exact counts and examples
"""
import os
import sys
import re
from dotenv import load_dotenv
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

load_dotenv()

from qdrant_client import QdrantClient
from whoosh_search_engine import WhooshSearchEngine

def search_exact_terms():
    """Search for exact reserve terminology phrases"""
    print("🔍 TARGETED SEARCH FOR EXACT RESERVE TERMINOLOGY")
    print("=" * 80)

    try:
        # Initialize Whoosh search engine
        search_engine = WhooshSearchEngine(index_dir="search_index")

        # Exact terms to search for
        exact_terms = [
            '"deterministic reserve"',
            '"stochastic reserve"',
            '"deterministic reserves"',
            '"stochastic reserves"',
            '"DR"',
            '"SR"',
            '"net premium reserve"',
            '"NPR"'
        ]

        all_findings = {}

        for term in exact_terms:
            print(f"\n🎯 Searching for exact phrase: {term}")

            # Use simpler query for exact matches
            clean_term = term.replace('"', '')
            results = search_engine.search(clean_term, limit=50, highlight=True)

            if results:
                print(f"   ✅ Found {len(results)} documents")

                # Collect top results with highlights
                top_results = []
                for result in results[:10]:  # Top 10 results
                    doc_info = {
                        'doc_id': result.doc_id,
                        'score': result.score,
                        'content_preview': result.content[:200] + "..." if len(result.content) > 200 else result.content,
                        'highlights': result.highlights,
                        'metadata': result.metadata
                    }
                    top_results.append(doc_info)

                all_findings[clean_term] = {
                    'count': len(results),
                    'top_results': top_results
                }
            else:
                print(f"   ❌ No results for {term}")
                all_findings[clean_term] = {
                    'count': 0,
                    'top_results': []
                }

        # Summary report
        print("\n" + "=" * 80)
        print("📊 SUMMARY OF FINDINGS")
        print("=" * 80)

        total_dr_docs = 0
        total_sr_docs = 0
        total_npr_docs = 0

        # Count documents for each category
        dr_terms = ["deterministic reserve", "deterministic reserves", "DR"]
        sr_terms = ["stochastic reserve", "stochastic reserves", "SR"]
        npr_terms = ["net premium reserve", "NPR"]

        for term, data in all_findings.items():
            print(f"\n📋 {term.upper()}: {data['count']} documents")

            if term.lower() in [t.lower() for t in dr_terms]:
                total_dr_docs += data['count']
            elif term.lower() in [t.lower() for t in sr_terms]:
                total_sr_docs += data['count']
            elif term.lower() in [t.lower() for t in npr_terms]:
                total_npr_docs += data['count']

            # Show example documents
            if data['top_results']:
                print(f"   Top documents:")
                for i, result in enumerate(data['top_results'][:3], 1):
                    file_path = result['metadata'].get('file_path', 'Unknown file')
                    print(f"   {i}. {file_path} (Score: {result['score']:.2f})")
                    if result['highlights']:
                        print(f"      Highlight: {result['highlights'][:100]}...")

        print(f"\n🎯 CATEGORY TOTALS:")
        print(f"   📄 Deterministic Reserve (DR) related: {total_dr_docs} document instances")
        print(f"   📄 Stochastic Reserve (SR) related: {total_sr_docs} document instances")
        print(f"   📄 Net Premium Reserve (NPR) related: {total_npr_docs} document instances")

        # Find unique documents containing these terms
        unique_docs = set()
        for term_data in all_findings.values():
            for result in term_data['top_results']:
                file_path = result['metadata'].get('file_path', 'Unknown')
                unique_docs.add(file_path)

        print(f"\n📚 UNIQUE DOCUMENTS: {len(unique_docs)} unique source documents")
        for doc in sorted(unique_docs):
            print(f"   • {doc}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

def search_qdrant_exact():
    """Search Qdrant for exact occurrences of reserve terms"""
    print("\n" + "=" * 80)
    print("🔍 SEARCHING QDRANT FOR EXACT OCCURRENCES")
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

        # Exact search patterns
        patterns = {
            'deterministic reserve': re.compile(r'\bdeterministic\s+reserve\b', re.IGNORECASE),
            'stochastic reserve': re.compile(r'\bstochastic\s+reserve\b', re.IGNORECASE),
            'deterministic reserves': re.compile(r'\bdeterministic\s+reserves\b', re.IGNORECASE),
            'stochastic reserves': re.compile(r'\bstochastic\s+reserves\b', re.IGNORECASE),
            'DR reserve': re.compile(r'\bDR\s+reserve\b', re.IGNORECASE),
            'SR reserve': re.compile(r'\bSR\s+reserve\b', re.IGNORECASE),
            'net premium reserve': re.compile(r'\bnet\s+premium\s+reserve\b', re.IGNORECASE),
            'NPR': re.compile(r'\bNPR\b', re.IGNORECASE)
        }

        findings = {}

        for point in points:
            if point.payload and point.payload.get("text"):
                text = point.payload.get("text", "")
                filename = point.payload.get("filename", "Unknown")
                job_id = point.payload.get("job_id", "No job_id")

                for term, pattern in patterns.items():
                    matches = pattern.findall(text)
                    if matches:
                        if term not in findings:
                            findings[term] = {
                                'total_matches': 0,
                                'documents': {},
                                'examples': []
                            }

                        if filename not in findings[term]['documents']:
                            findings[term]['documents'][filename] = {
                                'job_id': job_id,
                                'match_count': 0,
                                'chunks': []
                            }

                        findings[term]['total_matches'] += len(matches)
                        findings[term]['documents'][filename]['match_count'] += len(matches)
                        findings[term]['documents'][filename]['chunks'].append({
                            'point_id': point.id,
                            'matches': matches,
                            'context': text[:300] + "..." if len(text) > 300 else text
                        })

                        # Store examples
                        for match in matches[:2]:  # First 2 matches
                            start = text.lower().find(match.lower())
                            if start != -1:
                                context = text[max(0, start-50):start+len(match)+50]
                                findings[term]['examples'].append({
                                    'filename': filename,
                                    'match': match,
                                    'context': context
                                })

        # Report findings
        if findings:
            print(f"\n✅ Found exact matches in Qdrant:")

            for term, data in findings.items():
                print(f"\n📋 '{term.upper()}':")
                print(f"   Total matches: {data['total_matches']}")
                print(f"   Documents: {len(data['documents'])}")

                # Show document breakdown
                for filename, doc_data in list(data['documents'].items())[:3]:  # Top 3 docs
                    print(f"   • {filename}: {doc_data['match_count']} matches (job_id: {doc_data['job_id']})")

                # Show examples with context
                print(f"   Examples:")
                for example in data['examples'][:2]:  # Top 2 examples
                    print(f"   → {example['filename']}")
                    print(f"     Match: '{example['match']}'")
                    print(f"     Context: ...{example['context']}...")
                    print()

        else:
            print("❌ No exact matches found in Qdrant")

    except Exception as e:
        print(f"❌ Error searching Qdrant: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    search_exact_terms()
    search_qdrant_exact()

if __name__ == "__main__":
    main()