#!/usr/bin/env python3
"""
Debug why citations disappeared after implementing robust filtering
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.relevance_engine import RelevanceEngine

def debug_missing_citations():
    """Debug why no citations are showing for ASC 255-10-50-51"""
    
    print("=== DEBUGGING MISSING CITATIONS ===")
    print("Analyzing why ASC 255-10-50-51 now has no citations")
    print("="*60)
    
    # Initialize relevance engine
    relevance_engine = RelevanceEngine()
    
    # Test query (exactly as user entered)
    test_query = "what is ASC 255-10-50-51?"
    
    # Analyze the query
    query_analysis = relevance_engine.analyze_query(test_query)
    
    print(f"Query: {test_query}")
    print(f"Query Type: {query_analysis.query_type.value}")
    print(f"Entities: {query_analysis.entities}")
    print(f"Domain: {query_analysis.domain}")  # Key: Is this 'foreign_currency'?
    print(f"Specificity Score: {query_analysis.specificity_score:.3f}")
    print(f"Search Weights: {query_analysis.search_weights}")
    
    # Test the specific filtering logic
    print(f"\n=== TESTING CITATION FILTERS ===")
    
    # Mock the PWC document as it should appear after search
    mock_pwc_doc = {
        'node_id': 'pwc_doc_1',
        'content': 'When determining whether an asset or liability is monetary or nonmonetary, a reporting entity should consider the guidance in ASC 830-10-45-18 and ASC 255, Changing Prices. While ASC 255 was not specifically created to be applied to foreign currency transactions, the information is helpful in distinguishing assets and liabilities as monetary or nonmonetary. ASC 255-10-20 also provides definitions of monetary assets and liabilities while ASC 255-10-50-51 and 50-52 provide examples. Definition from ASC 255-10-20 Monetary Assets: Money or a claim to receive a sum of money the amount of which is fixed or determinable without reference to future prices of specific goods or services. Monetary Liability: An obligation to pay a sum of money the amount of which is fixed or determinable without reference to future prices of specific goods and services. Excerpt from ASC 255-10-50-51 The economic significance of nonmonetary items depends heavily on the value of specific goods and services. Nonmonetary assets include all of the following: a. Goods held primarily for resale or assets held primarily for direct use in providing services for the business of the entity. b. Claims to cash in amounts dependent on future prices of specific goods or services. c. Residual rights such as goodwill or equity interests.',
        'metadata': {'filename': 'pwcforeigncurrency0522.pdf', 'source_type': 'COMPANY'},
        'score': 0.85  # High vector similarity
    }
    
    # Score the document using relevance engine
    relevance = relevance_engine.score_document_relevance(query_analysis, mock_pwc_doc)
    
    print(f"PWC Document Scoring:")
    print(f"  Vector Score: {mock_pwc_doc['score']:.3f}")
    print(f"  Exact Match Score: {relevance.exact_match_score:.3f}")
    print(f"  Context Score: {relevance.context_score:.3f}")
    print(f"  Entity Coverage Score: {relevance.entity_coverage_score:.3f}")
    print(f"  Final Relevance Score: {relevance.final_score:.3f}")
    
    # Add relevance breakdown to document (as the pipeline would)
    mock_pwc_doc['relevance_score'] = relevance.final_score
    mock_pwc_doc['relevance_breakdown'] = {
        'exact_match': relevance.exact_match_score,
        'semantic': relevance.semantic_score,
        'context': relevance.context_score,
        'entity_coverage': relevance.entity_coverage_score
    }
    
    # Test citation filtering logic
    print(f"\n=== TESTING CITATION FILTERING LOGIC ===")
    
    # 1. Query type check
    if query_analysis.query_type.value == "specific_reference":
        print(f"✅ Query classified as specific_reference")
        
        # 2. Dynamic threshold calculation
        top_scores = [mock_pwc_doc.get('relevance_score', mock_pwc_doc.get('score', 0))]
        if top_scores and max(top_scores) > 0.7:
            score_threshold = 0.5
            print(f"✅ High-quality match found - using selective threshold: {score_threshold}")
        else:
            score_threshold = 0.3
            print(f"⚠️  No high-quality matches - using inclusive threshold: {score_threshold}")
        
        # 3. Check if document passes threshold
        relevance_score = mock_pwc_doc.get('relevance_score', mock_pwc_doc.get('score', 0))
        if relevance_score >= score_threshold:
            print(f"✅ Document passes threshold: {relevance_score:.3f} >= {score_threshold}")
        else:
            print(f"❌ Document FAILS threshold: {relevance_score:.3f} < {score_threshold}")
            return
        
        # 4. Entity coverage check
        entity_coverage = mock_pwc_doc.get('relevance_breakdown', {}).get('entity_coverage', 0.0)
        if entity_coverage >= 0.1:
            print(f"✅ Entity coverage sufficient: {entity_coverage:.3f} >= 0.1")
        else:
            print(f"❌ Entity coverage FAILS: {entity_coverage:.3f} < 0.1")
            return
        
        # 5. Exact match check
        exact_match_score = mock_pwc_doc.get('relevance_breakdown', {}).get('exact_match', 0.0)
        if len(query_analysis.entities) > 0 and exact_match_score >= 0.05:
            print(f"✅ Exact match sufficient: {exact_match_score:.3f} >= 0.05")
        elif len(query_analysis.entities) == 0:
            print(f"✅ No entities to match")
        else:
            print(f"❌ Exact match FAILS: {exact_match_score:.3f} < 0.05")
            return
    
    # 6. Domain mismatch check
    doc_filename = mock_pwc_doc['metadata'].get('filename', '').lower()
    if query_analysis.domain == 'foreign_currency' and 'licat' in doc_filename:
        print(f"❌ Domain mismatch: foreign_currency query but LICAT document")
        return
    elif 'licat' in doc_filename:
        print(f"⚠️  Document is LICAT but query domain is: {query_analysis.domain}")
    else:
        print(f"✅ No domain mismatch detected")
    
    print(f"\n🎯 CONCLUSION:")
    print(f"Document should pass all citation filters!")
    
    # Check for other potential issues
    print(f"\n=== OTHER POTENTIAL ISSUES ===")
    print(f"1. Check if document is making it through search pipeline")
    print(f"2. Check if there are generic response filters blocking it")
    print(f"3. Check server logs for filtering messages")
    print(f"4. Verify PWC document is actually indexed and searchable")

if __name__ == "__main__":
    debug_missing_citations()