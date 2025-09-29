#!/usr/bin/env python3
"""
Comprehensive search for VM-20 reserve calculation methodology terms
including Stochastic Reserve (SR), Deterministic Reserve (DR), Net Premium Reserve (NPR),
Monte Carlo simulation, CTE, prescribed adverse scenarios, and Deferred Premium Asset (DPA)
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

def search_vm20_terms():
    """Search for comprehensive VM-20 methodology terms"""
    print("🔍 VM-20 RESERVE METHODOLOGY COMPREHENSIVE SEARCH")
    print("=" * 80)

    try:
        # Initialize Whoosh search engine
        search_engine = WhooshSearchEngine(index_dir="search_index")

        print(f"📊 Total documents in Whoosh: {search_engine.get_document_count()}")

        if search_engine.get_document_count() == 0:
            print("❌ Whoosh search index is empty")
            return

        # Comprehensive VM-20 terms to search for
        vm20_search_terms = {
            "Reserve Types": [
                "Stochastic Reserve",
                "SR",
                "Deterministic Reserve",
                "DR",
                "Net Premium Reserve",
                "NPR"
            ],
            "Calculation Methods": [
                "Monte Carlo simulation",
                "Monte Carlo",
                "CTE",
                "Conditional Tail Expectation",
                "prescribed adverse scenarios",
                "adverse scenarios"
            ],
            "VM-20 Specific": [
                "Deferred Premium Asset",
                "DPA",
                "VM-20",
                "VM20",
                "Principle-Based Reserves",
                "PBR"
            ],
            "Technical Terms": [
                "stochastic modeling",
                "deterministic scenario",
                "tail risk",
                "confidence level",
                "percentile",
                "scenario testing",
                "cash flow projection",
                "economic scenario generator",
                "ESG"
            ]
        }

        all_results = {}
        category_results = {}

        for category, terms in vm20_search_terms.items():
            print(f"\n📂 Searching category: {category}")
            category_results[category] = {}

            for term in terms:
                print(f"  🔍 Searching for: '{term}'")

                results = search_engine.search(term, limit=50, highlight=True)

                if results:
                    print(f"    ✅ Found {len(results)} results")
                    category_results[category][term] = len(results)

                    for result in results:
                        doc_id = result.doc_id
                        if doc_id not in all_results:
                            all_results[doc_id] = {
                                "content": result.content,
                                "metadata": result.metadata,
                                "score": result.score,
                                "terms_matched": set(),
                                "categories": set(),
                                "highlights": []
                            }

                        all_results[doc_id]["terms_matched"].add(term)
                        all_results[doc_id]["categories"].add(category)
                        if result.highlights:
                            all_results[doc_id]["highlights"].append({
                                "term": term,
                                "highlight": result.highlights
                            })
                else:
                    print(f"    ❌ No results for '{term}'")
                    category_results[category][term] = 0

        # Analysis and reporting
        print("\n" + "=" * 80)
        print("📊 VM-20 METHODOLOGY SEARCH RESULTS ANALYSIS")
        print("=" * 80)

        # Category summary
        for category, terms_data in category_results.items():
            total_in_category = sum(terms_data.values())
            print(f"\n📂 {category}:")
            print(f"   Total document instances: {total_in_category}")

            for term, count in terms_data.items():
                if count > 0:
                    print(f"   ✅ '{term}': {count} documents")
                else:
                    print(f"   ❌ '{term}': 0 documents")

        # Document analysis
        if all_results:
            print(f"\n📚 UNIQUE DOCUMENTS CONTAINING VM-20 TERMS: {len(all_results)}")
            print("=" * 60)

            # Sort by score and number of terms matched
            sorted_docs = sorted(all_results.items(),
                               key=lambda x: (len(x[1]["terms_matched"]), x[1]["score"]),
                               reverse=True)

            for doc_id, info in sorted_docs[:10]:  # Top 10 documents
                print(f"\n📄 Document ID: {doc_id}")
                print(f"   Score: {info['score']:.2f}")
                print(f"   Terms matched ({len(info['terms_matched'])}): {', '.join(sorted(info['terms_matched']))}")
                print(f"   Categories: {', '.join(sorted(info['categories']))}")

                # Show metadata
                metadata = info['metadata']
                if 'file_path' in metadata:
                    print(f"   File: {metadata['file_path']}")
                if 'document_type' in metadata:
                    print(f"   Type: {metadata['document_type']}")

                # Show content preview
                content_preview = info['content'][:200] + "..." if len(info['content']) > 200 else info['content']
                print(f"   Preview: {content_preview}")

                # Show key highlights
                if info['highlights']:
                    print(f"   Key highlights:")
                    for highlight_info in info['highlights'][:3]:  # Top 3 highlights
                        print(f"     [{highlight_info['term']}]: {highlight_info['highlight']}")

                print("-" * 60)

            # Summary statistics
            print(f"\n🎯 SUMMARY STATISTICS:")

            # Most comprehensive documents (containing multiple VM-20 terms)
            comprehensive_docs = [doc for doc, info in all_results.items()
                                if len(info['terms_matched']) >= 3]

            print(f"   📄 Documents with 3+ VM-20 terms: {len(comprehensive_docs)}")

            # Count by category coverage
            category_coverage = {}
            for doc_id, info in all_results.items():
                num_categories = len(info['categories'])
                if num_categories not in category_coverage:
                    category_coverage[num_categories] = 0
                category_coverage[num_categories] += 1

            print(f"   📊 Category coverage:")
            for num_cats, count in sorted(category_coverage.items(), reverse=True):
                print(f"     {num_cats} categories: {count} documents")

            # Identify unique source files
            unique_files = set()
            for info in all_results.values():
                file_path = info['metadata'].get('file_path', 'Unknown')
                unique_files.add(file_path)

            print(f"   📁 Unique source files: {len(unique_files)}")
            for file_path in sorted(unique_files):
                print(f"     • {file_path}")

        else:
            print("\n❌ No documents found containing VM-20 methodology terms")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

def search_qdrant_vm20():
    """Search Qdrant for VM-20 terms"""
    print("\n" + "=" * 80)
    print("🔍 SEARCHING QDRANT FOR VM-20 TERMS")
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

        # VM-20 search patterns
        vm20_patterns = {
            'Stochastic Reserve': re.compile(r'\bstochastic\s+reserve\b', re.IGNORECASE),
            'SR': re.compile(r'\bSR\b', re.IGNORECASE),
            'Deterministic Reserve': re.compile(r'\bdeterministic\s+reserve\b', re.IGNORECASE),
            'DR': re.compile(r'\bDR\b', re.IGNORECASE),
            'Net Premium Reserve': re.compile(r'\bnet\s+premium\s+reserve\b', re.IGNORECASE),
            'NPR': re.compile(r'\bNPR\b', re.IGNORECASE),
            'Monte Carlo simulation': re.compile(r'\bmonte\s+carlo\s+simulation\b', re.IGNORECASE),
            'Monte Carlo': re.compile(r'\bmonte\s+carlo\b', re.IGNORECASE),
            'CTE': re.compile(r'\bCTE\b', re.IGNORECASE),
            'Conditional Tail Expectation': re.compile(r'\bconditional\s+tail\s+expectation\b', re.IGNORECASE),
            'prescribed adverse scenarios': re.compile(r'\bprescribed\s+adverse\s+scenarios\b', re.IGNORECASE),
            'adverse scenarios': re.compile(r'\badverse\s+scenarios\b', re.IGNORECASE),
            'Deferred Premium Asset': re.compile(r'\bdeferred\s+premium\s+asset\b', re.IGNORECASE),
            'DPA': re.compile(r'\bDPA\b', re.IGNORECASE),
            'VM-20': re.compile(r'\bVM-20\b|\bVM20\b', re.IGNORECASE),
            'Principle-Based Reserves': re.compile(r'\bprinciple-based\s+reserves\b|\bprinciple\s+based\s+reserves\b', re.IGNORECASE),
            'PBR': re.compile(r'\bPBR\b', re.IGNORECASE)
        }

        findings = {}

        for point in points:
            if point.payload and point.payload.get("text"):
                text = point.payload.get("text", "")
                filename = point.payload.get("filename", "Unknown")
                job_id = point.payload.get("job_id", "No job_id")

                for term, pattern in vm20_patterns.items():
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

                        # Store examples with context
                        for match in matches[:2]:  # First 2 matches
                            start = text.lower().find(match.lower())
                            if start != -1:
                                context = text[max(0, start-75):start+len(match)+75]
                                findings[term]['examples'].append({
                                    'filename': filename,
                                    'match': match,
                                    'context': context
                                })

        # Report findings
        if findings:
            print(f"\n✅ Found VM-20 terms in Qdrant:")

            # Sort terms by number of matches
            sorted_terms = sorted(findings.items(), key=lambda x: x[1]['total_matches'], reverse=True)

            for term, data in sorted_terms:
                print(f"\n📋 '{term}':")
                print(f"   Total matches: {data['total_matches']}")
                print(f"   Documents: {len(data['documents'])}")

                # Show top documents
                sorted_docs = sorted(data['documents'].items(),
                                   key=lambda x: x[1]['match_count'], reverse=True)

                for filename, doc_data in sorted_docs[:3]:  # Top 3 docs
                    print(f"   • {filename}: {doc_data['match_count']} matches")

                # Show examples with context
                if data['examples']:
                    print(f"   Examples:")
                    for example in data['examples'][:2]:  # Top 2 examples
                        print(f"   → {example['filename']}")
                        print(f"     Context: ...{example['context']}...")
                        print()

        else:
            print("❌ No VM-20 terms found in Qdrant")

    except Exception as e:
        print(f"❌ Error searching Qdrant: {str(e)}")
        import traceback
        traceback.print_exc()

def search_file_system():
    """Search the file system directly for VM-20 terms"""
    print("\n" + "=" * 80)
    print("🔍 SEARCHING FILE SYSTEM FOR VM-20 TERMS")
    print("=" * 80)

    # Look for document directories
    potential_dirs = ["data", "documents", "docs", "uploads", "files"]

    vm20_terms_regex = [
        r'\bstochastic\s+reserve\b',
        r'\bdeterministic\s+reserve\b',
        r'\bnet\s+premium\s+reserve\b',
        r'\bmonte\s+carlo\s+simulation\b',
        r'\bmonte\s+carlo\b',
        r'\bCTE\b',
        r'\bconditional\s+tail\s+expectation\b',
        r'\bprescribed\s+adverse\s+scenarios\b',
        r'\badverse\s+scenarios\b',
        r'\bdeferred\s+premium\s+asset\b',
        r'\bVM-20\b',
        r'\bVM20\b',
        r'\bprinciple-based\s+reserves\b',
        r'\bPBR\b',
        r'\bSR\b',
        r'\bDR\b',
        r'\bNPR\b',
        r'\bDPA\b'
    ]

    found_files = {}

    for dir_name in potential_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"📁 Searching in directory: {dir_path.absolute()}")

            # Search for various file types
            file_patterns = ["*.txt", "*.pdf", "*.docx", "*.doc"]

            for pattern in file_patterns:
                for file_path in dir_path.rglob(pattern):
                    if file_path.suffix.lower() == '.txt':
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()

                            for term_pattern in vm20_terms_regex:
                                matches = re.findall(term_pattern, content, re.IGNORECASE)
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
        print(f"\n✅ Found {len(found_files)} files containing VM-20 terms:")

        for file_path, info in found_files.items():
            print(f"\n📄 File: {file_path}")
            print(f"   Terms found: {', '.join(sorted(info['terms_found']))}")
            print(f"   Total matches: {len(info['matches'])}")
    else:
        print("\n❌ No files found containing VM-20 terms in file system")

def main():
    """Main function to run comprehensive VM-20 search"""
    print("🔍 COMPREHENSIVE VM-20 RESERVE METHODOLOGY SEARCH")
    print("Searching for Stochastic Reserve (SR), Deterministic Reserve (DR),")
    print("Net Premium Reserve (NPR), Monte Carlo simulation, CTE,")
    print("prescribed adverse scenarios, and Deferred Premium Asset (DPA)")
    print("=" * 80)

    # Search Whoosh index
    search_vm20_terms()

    # Search Qdrant vector database
    search_qdrant_vm20()

    # Search file system directly
    search_file_system()

    print("\n" + "=" * 80)
    print("🔍 VM-20 COMPREHENSIVE SEARCH COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()