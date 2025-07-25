#!/usr/bin/env python3
"""
Test the general knowledge query detection
"""
import re

def _is_general_knowledge_query(query: str) -> bool:
    """Check if query is asking for general knowledge vs specific document content"""
    query_lower = query.lower()
    
    # First check for document-specific indicators (these override general patterns)
    document_indicators = [
        r'\bour company\b',
        r'\bthe uploaded\b',
        r'\bthe document\b',
        r'\bin the document\b',
        r'\bshow me\b',
        r'\bfind\b.*\bin\b',
        r'\banalyze\b',
        r'\bspecific\b.*\bmentioned\b',
        r'\bpolicy\b',
        r'\bprocedure\b'
    ]
    
    for pattern in document_indicators:
        if re.search(pattern, query_lower):
            return False  # Document-specific query
    
    # Common general knowledge question patterns (only if no document indicators)
    general_patterns = [
        r'^\s*what is\s+[a-z\s]+\??$',  # Simple "what is X?" questions
        r'^\s*define\s+',
        r'^\s*explain\s+[a-z\s]+\s+(concept|principle|term)s?\??$',
        r'^\s*how\s+does\s+.*\s+work\??$',
        r'^\s*what\s+does\s+.*\s+mean\??$',
        r'^\s*what\s+are\s+.*\s+(concept|principle)s?\??$'
    ]
    
    for pattern in general_patterns:
        if re.search(pattern, query_lower):
            return True
            
    return False

def test_queries():
    test_cases = [
        # General knowledge queries (should return True)
        ("what is accounts payable", True),
        ("what is accounts receivable?", True),
        ("What is accounts payable?", True),
        ("define accounts payable", True),
        ("define accounts receivable", True),
        ("explain accounting principles", True),
        ("how does accounting work?", True),
        ("what does accrual mean", True),
        ("what are basic accounting concepts", True),
        
        # Specific document queries (should return False)
        ("what is our company's accounts payable policy", False),
        ("show me the accounts payable procedures", False),
        ("find accounts payable in the uploaded document", False),
        ("can you explain the csm rules for re-insurance? how is it different from underlying csm?", False),
        ("what are the specific accounting standards mentioned in the document", False),
        ("analyze the financial statements", False),
    ]
    
    print("🧪 Testing general knowledge query detection:\n")
    
    all_passed = True
    for query, expected in test_cases:
        result = _is_general_knowledge_query(query)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"{status} '{query}' -> {result} (expected {expected})")
        if result != expected:
            all_passed = False
    
    print(f"\n🎯 Overall result: {'All tests passed!' if all_passed else 'Some tests failed!'}")
    return all_passed

if __name__ == "__main__":
    test_queries()