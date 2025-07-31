#!/usr/bin/env python3
"""
Debug script to test document search and retrieval
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline
from src.config import get_settings

async def test_search():
    """Test search functionality"""
    settings = get_settings()
    
    # Initialize RAG pipeline
    print("Initializing RAG pipeline...")
    rag = RAGPipeline(settings)
    
    # Test queries
    test_queries = [
        "ASC 255-10-50-51",
        "ASC 255-10-50-51 nonmonetary items",
        "nonmonetary assets include goods held for resale",
        "what is ASC 255-10-50-51"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)
        
        # Perform search
        results = await rag.search(query, k=5)
        
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"Score: {result.get('score', 0):.4f}")
            print(f"Node ID: {result.get('node_id', 'N/A')}")
            print(f"Content preview: {result.get('text', '')[:200]}...")
            
            metadata = result.get('metadata', {})
            print(f"Filename: {metadata.get('filename', 'N/A')}")
            print(f"Page: {metadata.get('page_num', 'N/A')}")
    
    # Also test keyword search directly
    print(f"\n{'='*60}")
    print("DIRECT KEYWORD SEARCH TEST")
    print('='*60)
    
    keyword_results = await rag._keyword_search("ASC 255-10-50-51", k=5)
    print(f"\nKeyword search found {len(keyword_results)} results")
    for i, result in enumerate(keyword_results[:3]):
        print(f"\n--- Keyword Result {i+1} ---")
        print(f"Score: {result.get('score', 0):.4f}")
        print(f"Content preview: {result.get('text', '')[:200]}...")

if __name__ == "__main__":
    asyncio.run(test_search())