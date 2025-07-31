#!/usr/bin/env python3
"""
Check what documents are in the RAG index
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline
from src.config import get_settings

async def check_documents():
    """Check documents in the index"""
    settings = get_settings()
    
    # Initialize RAG pipeline
    print("Initializing RAG pipeline...")
    rag = RAGPipeline(settings)
    
    # Search for all documents mentioning ASC
    print("\nSearching for documents containing 'ASC'...")
    results = await rag.search("ASC", k=20)
    
    # Group by filename
    documents = {}
    for result in results:
        metadata = result.get('metadata', {})
        filename = metadata.get('filename', 'Unknown')
        if filename not in documents:
            documents[filename] = []
        documents[filename].append(result)
    
    print(f"\nFound content in {len(documents)} documents:")
    for filename, doc_results in documents.items():
        print(f"\n📄 {filename}")
        print(f"   Chunks: {len(doc_results)}")
        
        # Show sample content
        for i, result in enumerate(doc_results[:2]):  # Show first 2 chunks
            text = result.get('text', '')
            # Look for ASC references
            if 'ASC' in text:
                import re
                asc_refs = re.findall(r'ASC \d{3}-\d{2}-\d{2}-\d{2}', text)
                if asc_refs:
                    print(f"   Found ASC references: {', '.join(set(asc_refs))}")
    
    # Specifically search for ASC 255
    print("\n" + "="*60)
    print("Searching specifically for 'ASC 255'...")
    asc_255_results = await rag.search("ASC 255", k=10)
    
    print(f"\nFound {len(asc_255_results)} results for ASC 255:")
    for i, result in enumerate(asc_255_results):
        print(f"\n--- Result {i+1} ---")
        print(f"Score: {result.get('score', 0):.4f}")
        text = result.get('text', '')
        # Extract ASC 255 context
        if 'ASC 255' in text:
            start = text.find('ASC 255')
            context = text[max(0, start-50):start+200]
            print(f"Context: ...{context}...")

if __name__ == "__main__":
    asyncio.run(check_documents())