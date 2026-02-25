#!/usr/bin/env python3
"""
Search script to find documents containing deterministic reserve (DR) and stochastic reserve (SR) terms
in both Qdrant vector database and Whoosh search index
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

def search_qdrant_for_reserves():
    """Search Qdrant vector database for reserve-related terms"""
    print("=" * 60)
    print("🔍 SEARCHING QDRANT VECTOR DATABASE")
    print("=" * 60)

    try:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url:
            print("❌ QDRANT_URL not set in environment")
            return

        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        collection_name = "aaire-documents"

        print(f"🔗 Connecting to Qdrant at {qdrant_url}")

        # Check if collection exists
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if collection_name not in collection_names:
            print(f"❌ Collection '{collection_name}' not found")
            print(f"Available collections: {collection_names}")
            return

        print(f"✅ Found collection '{collection_name}'")

        # Get all points
        search_result = client.scroll(
            collection_name=collection_name,
            limit=1000  # Increase limit to get more documents
        )

        points = search_result[0]
        print(f"📊 Total documents found: {len(points)}")

        # Search terms
        search_terms = [
            "deterministic reserve",
            "stochastic reserve",
            "deterministic reserves",
            "stochastic reserves",
            "DR reserve",
            "SR reserve",
            "net premium reserve",
            "NPR",
            "reserve methodology",
            "reserve calculation",
            "reserve valuation"
        ]

        found_documents = {}

        print(f"\n🔍 Searching for reserve-related terms in {len(points)} documents...")

        for point in points:
            if point.payload and point.payload.get("text"):
                text = point.payload.get("text", "").lower()
                filename = point.payload.get("filename", "Unknown")
                job_id = point.payload.get("job_id", "No job_id")

                for term in search_terms:
                    if term.lower() in text:
                        if filename not in found_documents:
                            found_documents[filename] = {
                                "job_id": job_id,
                                "terms_found": set(),
                                "chunks": [],
                                "point_ids": []
                            }

                        found_documents[filename]["terms_found"].add(term)
                        found_documents[filename]["chunks"].append({
                            "point_id": point.id,
                            "text_preview": text[:200] + "..." if len(text) > 200 else text,
                            "term": term
                        })
                        found_documents[filename]["point_ids"].append(point.id)

        # Report findings
        if found_documents:
            print(f"\n✅ Found {len(found_documents)} documents containing reserve-related terms:")
            print("=" * 60)

            for filename, info in found_documents.items():
                print(f"\n📄 Document: {filename}")
                print(f"   Job ID: {info['job_id']}")
                print(f"   Terms found: {', '.join(sorted(info['terms_found']))}")
                print(f"   Matching chunks: {len(info['chunks'])}")

                # Show first few matching chunks
                for i, chunk in enumerate(info['chunks'][:3]):
                    print(f"   \nChunk {i+1} (Point ID: {chunk['point_id']}):")
                    print(f"   Term: '{chunk['term']}'")
                    print(f"   Preview: {chunk['text_preview']}")

                if len(info['chunks']) > 3:
                    print(f"   ... and {len(info['chunks']) - 3} more chunks")

                print("-" * 40)
        else:
            print("\n❌ No documents found containing reserve-related terms in Qdrant")

    except Exception as e:
        print(f"❌ Error searching Qdrant: {str(e)}")
        import traceback
        traceback.print_exc()

def search_whoosh_for_reserves():
    """Search Whoosh index for reserve-related terms"""
    print("\n" + "=" * 60)
    print("🔍 SEARCHING WHOOSH SEARCH INDEX")
    print("=" * 60)

    try:
        # Initialize Whoosh search engine
        search_engine = WhooshSearchEngine(index_dir="search_index")

        print(f"📊 Total documents in Whoosh: {search_engine.get_document_count()}")

        if search_engine.get_document_count() == 0:
            print("❌ Whoosh search index is empty")
            return

        # Search terms
        search_queries = [
            "deterministic reserve",
            "stochastic reserve",
            "deterministic reserves",
            "stochastic reserves",
            "DR reserve",
            "SR reserve",
            "net premium reserve",
            "NPR",
            "reserve methodology",
            "reserve calculation",
            "reserve valuation"
        ]

        all_results = {}

        for query in search_queries:
            print(f"\n🔍 Searching for: '{query}'")

            results = search_engine.search(query, limit=50, highlight=True)

            if results:
                print(f"   ✅ Found {len(results)} results")

                for result in results:
                    doc_id = result.doc_id
                    if doc_id not in all_results:
                        all_results[doc_id] = {
                            "content": result.content,
                            "metadata": result.metadata,
                            "score": result.score,
                            "queries_matched": set(),
                            "highlights": []
                        }

                    all_results[doc_id]["queries_matched"].add(query)
                    if result.highlights:
                        all_results[doc_id]["highlights"].append(result.highlights)
            else:
                print(f"   ❌ No results for '{query}'")

        # Report consolidated findings
        if all_results:
            print(f"\n✅ Found {len(all_results)} unique documents containing reserve-related terms:")
            print("=" * 60)

            for doc_id, info in sorted(all_results.items(), key=lambda x: x[1]["score"], reverse=True):
                print(f"\n📄 Document ID: {doc_id}")
                print(f"   Score: {info['score']:.2f}")
                print(f"   Queries matched: {', '.join(sorted(info['queries_matched']))}")

                # Show metadata
                metadata = info['metadata']
                if 'file_path' in metadata:
                    print(f"   File path: {metadata['file_path']}")
                if 'primary_framework' in metadata:
                    print(f"   Framework: {metadata['primary_framework']}")
                if 'document_type' in metadata:
                    print(f"   Document type: {metadata['document_type']}")

                # Show content preview
                content_preview = info['content'][:300] + "..." if len(info['content']) > 300 else info['content']
                print(f"   Content preview: {content_preview}")

                # Show highlights if available
                if info['highlights']:
                    print(f"   Highlights:")
                    for highlight in info['highlights'][:2]:  # Show first 2 highlights
                        if highlight:
                            print(f"     {highlight}")

                print("-" * 40)
        else:
            print("\n❌ No documents found containing reserve-related terms in Whoosh")

    except Exception as e:
        print(f"❌ Error searching Whoosh: {str(e)}")
        import traceback
        traceback.print_exc()

def search_files_directly():
    """Search document files directly for reserve terms"""
    print("\n" + "=" * 60)
    print("🔍 SEARCHING FILES DIRECTLY")
    print("=" * 60)

    # Look for document directories
    potential_dirs = [
        "documents",
        "docs",
        "data",
        "uploads",
        "files"
    ]

    search_terms = [
        r"\bdeterministic\s+reserve\b",
        r"\bstochastic\s+reserve\b",
        r"\bdeterministic\s+reserves\b",
        r"\bstochastic\s+reserves\b",
        r"\bDR\s+reserve\b",
        r"\bSR\s+reserve\b",
        r"\bnet\s+premium\s+reserve\b",
        r"\bNPR\b",
        r"\breserve\s+methodology\b",
        r"\breserve\s+calculation\b",
        r"\breserve\s+valuation\b"
    ]

    found_files = {}

    for dir_name in potential_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"📁 Searching in directory: {dir_path.absolute()}")

            # Search for text files
            for file_path in dir_path.rglob("*.txt"):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().lower()

                    for pattern in search_terms:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            if str(file_path) not in found_files:
                                found_files[str(file_path)] = {
                                    "terms_found": set(),
                                    "matches": []
                                }

                            found_files[str(file_path)]["terms_found"].update(matches)
                            found_files[str(file_path)]["matches"].extend(matches)

                except Exception as e:
                    print(f"   ❌ Error reading {file_path}: {e}")

    if found_files:
        print(f"\n✅ Found {len(found_files)} files containing reserve-related terms:")

        for file_path, info in found_files.items():
            print(f"\n📄 File: {file_path}")
            print(f"   Terms found: {', '.join(sorted(info['terms_found']))}")
            print(f"   Total matches: {len(info['matches'])}")
    else:
        print("\n❌ No files found containing reserve-related terms")

def main():
    """Main function to run all searches"""
    print("🔍 AAIRE RESERVE TERMINOLOGY SEARCH")
    print("Searching for deterministic reserve (DR) and stochastic reserve (SR) terms")
    print("=" * 80)

    # Search Qdrant vector database
    search_qdrant_for_reserves()

    # Search Whoosh index
    search_whoosh_for_reserves()

    # Search files directly
    search_files_directly()

    print("\n" + "=" * 80)
    print("🔍 SEARCH COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()