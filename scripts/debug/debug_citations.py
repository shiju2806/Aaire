#!/usr/bin/env python3
"""
Debug citation fixing function
"""

# Test the citation post-processing function
test_response = """
The supervisory target for this ratio is 100%, with a minimum requirement of 90% [2].

The supervisory target for the Core Ratio is 70%, with a minimum requirement of 55% for 2025 and beyond [2].

Regulatory bodies monitor these ratios closely [4].

As mentioned in Document 2, the Base Solvency Buffer is important.
"""

# Mock retrieved docs
mock_docs = [
    {
        'metadata': {
            'filename': 'LICAT.pdf',
            'page': 2
        },
        'content': 'Some content with Source: Page 2, cluster info'
    },
    {
        'metadata': {
            'filename': 'LICAT.pdf', 
            'estimated_page': 4
        },
        'content': 'More content'
    }
]

def fix_citations(response, retrieved_docs):
    """Test citation fixing function"""
    import re
    
    print("BEFORE:")
    print(response)
    print("\n" + "="*50 + "\n")
    
    if not retrieved_docs:
        return response
    
    # Create mapping of citation numbers to proper source names
    citation_map = {}
    for i, doc in enumerate(retrieved_docs[:10]):
        citation_num = i + 1
        filename = doc['metadata'].get('filename', 'Unknown')
        
        # Try to extract page number
        page_info = ""
        if 'page' in doc['metadata']:
            page_info = f", Page {doc['metadata']['page']}"
        elif 'estimated_page' in doc['metadata']:
            page_info = f", Page {doc['metadata']['estimated_page']}"
        elif 'Source: Page' in doc.get('content', ''):
            page_match = re.search(r'Source: Page (\d+)', doc.get('content', ''))
            if page_match:
                page_info = f", Page {page_match.group(1)}"
        
        # Create proper citation
        proper_citation = f"({filename}{page_info})"
        citation_map[f"[{citation_num}]"] = proper_citation
        
        print(f"Mapping: [{ citation_num}] -> {proper_citation}")
    
    print(f"\nCitation map: {citation_map}")
    
    # Replace all [1], [2], etc. with proper citations
    fixed_response = response
    for old_citation, new_citation in citation_map.items():
        print(f"Replacing '{old_citation}' with '{new_citation}'")
        fixed_response = fixed_response.replace(old_citation, new_citation)
    
    # Also handle "Document X" references
    for i in range(1, 11):
        doc_ref = f"Document {i}"
        if doc_ref in fixed_response and i <= len(retrieved_docs):
            doc = retrieved_docs[i-1]
            filename = doc['metadata'].get('filename', 'Unknown')
            page_info = ""
            if 'page' in doc['metadata']:
                page_info = f", Page {doc['metadata']['page']}"
            elif 'estimated_page' in doc['metadata']:
                page_info = f", Page {doc['metadata']['estimated_page']}"
            
            proper_ref = f"{filename}{page_info}"
            print(f"Replacing '{doc_ref}' with '{proper_ref}'")
            fixed_response = fixed_response.replace(doc_ref, proper_ref)
    
    print("\nAFTER:")
    print(fixed_response)
    
    return fixed_response

if __name__ == "__main__":
    fix_citations(test_response, mock_docs)