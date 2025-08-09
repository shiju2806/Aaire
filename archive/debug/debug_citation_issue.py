#!/usr/bin/env python3
"""
Debug why LICAT.pdf is being cited for ASC 255-10-50-51 queries
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.relevance_engine import RelevanceEngine
import re

def debug_citation_contamination():
    """Debug why irrelevant documents get cited"""
    
    print("=== DEBUGGING CITATION CONTAMINATION ===")
    print("Analyzing why LICAT.pdf appears in ASC 255-10-50-51 citations")
    print("="*60)
    
    # Initialize relevance engine
    relevance_engine = RelevanceEngine()
    
    # Test query
    test_query = "what is ASC 255-10-50-51"
    
    # Analyze the query
    query_analysis = relevance_engine.analyze_query(test_query)
    
    print(f"Query: {test_query}")
    print(f"Query Type: {query_analysis.query_type.value}")
    print(f"Entities: {query_analysis.entities}")
    print(f"Domain: {query_analysis.domain}")
    print(f"Specificity Score: {query_analysis.specificity_score:.3f}")
    print(f"Search Weights: {query_analysis.search_weights}")
    
    # Simulate document analysis
    print(f"\n=== SIMULATING DOCUMENT SCORING ===")
    
    # Mock documents that might be in search results
    mock_documents = [
        {
            'node_id': 'pwc_doc_1',
            'content': 'ASC 255-10-50-51 provides guidance on nonmonetary items. The economic significance of nonmonetary items depends heavily on the value of specific goods and services. Nonmonetary assets include: a. Goods held primarily for resale or assets held primarily for direct use in providing services for the business of the entity.',
            'metadata': {'filename': 'pwcforeigncurrency0522.pdf', 'source_type': 'COMPANY'},
            'score': 0.85  # High vector similarity
        },
        {
            'node_id': 'licat_doc_1', 
            'content': 'LICAT Chapter 1: Overview of regulatory capital requirements. The framework establishes minimum capital requirements for life insurance companies. Section 2.5.5 discusses actuarial reserves and their impact on total ratio calculations. Financial reporting requirements include disclosure of assets and liabilities.',
            'metadata': {'filename': 'LICAT.pdf', 'source_type': 'ACTUARIAL'},
            'score': 0.45  # Lower vector similarity but still present
        }
    ]
    
    for doc in mock_documents:
        print(f"\n--- Document: {doc['metadata']['filename']} ---")
        relevance = relevance_engine.score_document_relevance(query_analysis, doc)
        
        print(f"Vector Score: {doc['score']:.3f}")
        print(f"Exact Match Score: {relevance.exact_match_score:.3f}")
        print(f"Context Score: {relevance.context_score:.3f}")
        print(f"Entity Coverage Score: {relevance.entity_coverage_score:.3f}")
        print(f"Final Relevance Score: {relevance.final_score:.3f}")
        print(f"Explanation: {relevance.explanation}")
        
        # Check what entities are being matched
        content_lower = doc['content'].lower()
        for entity in query_analysis.entities:
            if entity.lower() in content_lower:
                print(f"  ✅ Entity '{entity}' found in content")
            else:
                print(f"  ❌ Entity '{entity}' NOT found in content")
        
        # Check for false positive matches
        if '255' in doc['content']:
            positions = [m.start() for m in re.finditer('255', doc['content'])]
            print(f"  ⚠️  Found '255' at positions: {positions}")
            for pos in positions:
                context = doc['content'][max(0, pos-20):pos+30]
                print(f"    Context: ...{context}...")
    
    # Analyze citation thresholds
    print(f"\n=== CITATION THRESHOLD ANALYSIS ===")
    if query_analysis.query_type.value == "specific_reference":
        threshold = 0.3
        print(f"Specific reference query - citation threshold: {threshold}")
        print(f"⚠️  This threshold might be too permissive!")
        
        # Suggest better threshold
        print(f"\nRECOMMENDATIONS:")
        print(f"1. Increase threshold to 0.5 for specific reference queries")
        print(f"2. Add negative scoring for document type mismatch")
        print(f"3. Require minimum exact match score for specific queries")
        print(f"4. Add domain alignment filter (PWC doc = foreign_currency, LICAT = insurance)")

if __name__ == "__main__":
    debug_citation_contamination()