#!/usr/bin/env python3
"""
Debug citation extraction when response clearly uses document content
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def analyze_citation_flow():
    """Analyze the citation extraction flow"""
    
    print("=== CITATION EXTRACTION FLOW ANALYSIS ===")
    print("Response clearly uses PWC document content but no citation shown")
    print("="*60)
    
    print("PIPELINE FLOW:")
    print("1. ✅ Query Analysis: 'what is ASC 255-10-50-51?' → specific_reference")
    print("2. ✅ Document Search: Found PWC document with ASC content")
    print("3. ✅ Response Generation: Used document content in response")
    print("4. ❌ Citation Display: Citation [1] missing in UI")
    
    print("\nPOSSIBLE CAUSES:")
    print("A. Citation extraction logic is failing")
    print("B. Document passes search but fails citation threshold")
    print("C. Citation display/formatting issue in frontend")
    print("D. Logging issue - citations created but not shown")
    
    print("\nCHECK POINTS:")
    print("1. Are citations being created in _extract_citations()?")
    print("2. Are citations being passed to the response?")
    print("3. Is the frontend receiving and displaying citations?")
    print("4. Are server logs showing citation creation messages?")
    
    print("\nEXPECTED SERVER LOGS:")
    print("- 'Found X relevant documents for query'")
    print("- 'Top document sources with scores: [....]'")
    print("- 'Document 1: relevance_score=X.XXX, filename=pwcforeigncurrency0522.pdf'")
    print("- '✅ ADDED citation from: pwcforeigncurrency0522.pdf (relevance: X.XXX)'")
    print("- '🎯 FINAL RESULT: Generated X citations from Y retrieved documents'")
    
    print("\nDEBUGGING STEPS:")
    print("1. Check server logs for citation creation messages")
    print("2. Verify retrieved_docs contains PWC document")
    print("3. Check if document passes all citation filters")
    print("4. Verify citations array is populated and returned")
    print("5. Test if frontend is receiving citations in API response")
    
    # Simulate the citation extraction logic
    print("\n=== SIMULATING CITATION EXTRACTION ===")
    
    # Mock what should be happening
    mock_retrieved_docs = [
        {
            'node_id': 'pwc_doc_1',
            'content': 'ASC 255-10-50-51 provides guidance on nonmonetary items...',
            'metadata': {'filename': 'pwcforeigncurrency0522.pdf', 'source_type': 'COMPANY'},
            'score': 0.85,
            'relevance_score': 0.735,
            'relevance_breakdown': {
                'exact_match': 0.750,
                'semantic': 0.850,
                'context': 0.154,
                'entity_coverage': 1.000
            },
            'source_type': 'company'
        }
    ]
    
    print(f"Mock retrieved docs: {len(mock_retrieved_docs)} documents")
    
    # Test the citation thresholds
    query_type = "specific_reference"
    top_score = 0.735
    
    if top_score > 0.7:
        score_threshold = 0.5
        print(f"Dynamic threshold: {score_threshold} (high-quality match)")
    else:
        score_threshold = 0.3
        print(f"Dynamic threshold: {score_threshold} (inclusive)")
    
    doc = mock_retrieved_docs[0]
    relevance_score = doc['relevance_score']
    
    print(f"\nTesting document: {doc['metadata']['filename']}")
    print(f"Relevance score: {relevance_score}")
    print(f"Threshold: {score_threshold}")
    print(f"Passes threshold: {relevance_score >= score_threshold}")
    
    # Test entity coverage
    entity_coverage = doc['relevance_breakdown']['entity_coverage']
    print(f"Entity coverage: {entity_coverage} (>= 0.1: {entity_coverage >= 0.1})")
    
    # Test exact match
    exact_match = doc['relevance_breakdown']['exact_match']
    print(f"Exact match: {exact_match} (>= 0.05: {exact_match >= 0.05})")
    
    print(f"\n🎯 CONCLUSION:")
    if (relevance_score >= score_threshold and 
        entity_coverage >= 0.1 and 
        exact_match >= 0.05):
        print("✅ Document should generate a citation!")
        print("Issue is likely in:")
        print("- Citation creation code has a bug")
        print("- Frontend not displaying citations")
        print("- Server logs will show if citations are being created")
    else:
        print("❌ Document fails citation filters")

if __name__ == "__main__":
    analyze_citation_flow()