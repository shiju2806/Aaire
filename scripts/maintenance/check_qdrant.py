#!/usr/bin/env python3
"""
Simple script to check what's in Qdrant
"""
import os
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient

def main():
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
            limit=100
        )
        
        points = search_result[0]
        print(f"📊 Total documents found: {len(points)}")
        
        if len(points) > 0:
            print("\n📋 Documents in vector store:")
            unique_files = {}
            
            for i, point in enumerate(points, 1):
                if point.payload:
                    filename = point.payload.get("filename", "Unknown")
                    job_id = point.payload.get("job_id", "No job_id")
                    doc_type = point.payload.get("doc_type", "Unknown")
                    text_preview = point.payload.get("text", "")[:50] + "..." if point.payload.get("text") else ""
                    
                    # Group by filename for summary
                    if filename not in unique_files:
                        unique_files[filename] = {
                            "job_id": job_id,
                            "doc_type": doc_type,
                            "chunk_count": 0,
                            "points": []
                        }
                    unique_files[filename]["chunk_count"] += 1
                    unique_files[filename]["points"].append(point.id)
                    
                    if len(unique_files) <= 5:  # Show first 5 unique files
                        if unique_files[filename]["chunk_count"] == 1:  # Only show on first chunk
                            print(f"  📄 {filename}")
                            print(f"     Job ID: {job_id}")
                            print(f"     Type: {doc_type}")
                            print(f"     Preview: {text_preview}")
                            print()
            
            print(f"📈 Summary by document:")
            for filename, info in unique_files.items():
                print(f"  • {filename}: {info['chunk_count']} chunks (job_id: {info['job_id']})")
            
            # Check for orphaned chunks (no job_id)
            orphaned_count = sum(1 for point in points if not point.payload or not point.payload.get("job_id"))
            if orphaned_count > 0:
                print(f"\n⚠️  Found {orphaned_count} orphaned chunks (no job_id)")
                
            # Show which documents might be causing false citations
            print(f"\n🔍 Documents that might cause false citations for 'accounts payable':")
            for filename, info in unique_files.items():
                print(f"  - {filename} ({info['chunk_count']} chunks)")
                
        else:
            print("✅ Vector store is empty")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()