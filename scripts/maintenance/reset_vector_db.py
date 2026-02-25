#!/usr/bin/env python3
"""
Complete vector database reset script - removes ALL documents from Qdrant
"""
import os
import sys
import asyncio
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

async def main():
    """Complete reset of the vector database"""
    try:
        if not QDRANT_AVAILABLE:
            print("❌ Qdrant client not available. Install with: pip install qdrant-client")
            return
            
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url:
            print("❌ QDRANT_URL not set in environment")
            print("Please check your .env file and ensure Qdrant configuration is uncommented")
            return
            
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        collection_name = "aaire-documents"
        
        print(f"🔗 Connecting to Qdrant at {qdrant_url}")
        
        # Check if collection exists
        try:
            collections = client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if collection_name not in collection_names:
                print(f"ℹ️  Collection '{collection_name}' doesn't exist - nothing to clean")
                return
                
            print(f"📊 Found collection '{collection_name}'")
            
            # Get current document count
            collection_info = client.get_collection(collection_name)
            point_count = collection_info.points_count
            print(f"📈 Current documents in collection: {point_count}")
            
            if point_count == 0:
                print("✅ Collection is already empty")
                return
            
            # Confirm deletion
            print(f"\n⚠️  This will DELETE ALL {point_count} documents from the vector store!")
            print("This action cannot be undone.")
            confirm = input("Are you sure you want to proceed? Type 'DELETE ALL' to confirm: ").strip()
            
            if confirm != "DELETE ALL":
                print("❌ Operation cancelled")
                return
            
            print(f"\n🗑️  Deleting collection '{collection_name}'...")
            client.delete_collection(collection_name)
            print(f"✅ Collection '{collection_name}' deleted successfully")
            
            print(f"\n🔨 Recreating empty collection '{collection_name}'...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI embedding dimension
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Empty collection '{collection_name}' created")
            
            # Create index for job_id field
            try:
                from qdrant_client.models import PayloadSchemaType
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name="job_id",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                print("✅ Created job_id index for future document tracking")
            except Exception as e:
                print(f"⚠️  Could not create job_id index: {e}")
            
            print(f"\n🎉 Vector database reset complete!")
            print(f"📊 The collection is now empty and ready for new documents")
            
        except Exception as e:
            print(f"❌ Error accessing collection: {str(e)}")
            return
            
    except Exception as e:
        print(f"❌ Error during reset: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())