#!/usr/bin/env python3
"""
Test if RAG pipeline can access environment variables and generate embeddings
"""

import os
import asyncio
from dotenv import load_dotenv

# Load environment like main.py does
load_dotenv()

async def test_rag_embedding():
    """Test RAG pipeline embedding generation"""

    print("🔍 Testing RAG pipeline embedding access...")

    # Check environment
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No OPENAI_API_KEY in environment")
        return False

    print(f"✅ OpenAI API key available: {api_key[:10]}...")

    try:
        # Import and initialize like the main app does
        from src.rag_pipeline import RAGPipeline

        print("📝 Initializing RAG pipeline...")

        # Initialize RAG pipeline (this should set up embedding model)
        rag = RAGPipeline(config_path="config/mvp_config.yaml")

        print("✅ RAG pipeline initialized")

        # Check if embedding model is available
        if hasattr(rag, 'embedding_model'):
            print("✅ Embedding model found in RAG pipeline")

            # Try to generate an embedding
            test_text = "whole life insurance reserves calculation"
            print(f"📝 Testing embedding generation for: {test_text}")

            embedding = await rag.embedding_model.aget_text_embedding(test_text)

            if embedding and len(embedding) > 0:
                print(f"✅ Embedding generated! Dimensions: {len(embedding)}")
                print(f"   First 5 values: {embedding[:5]}")
                return True
            else:
                print("❌ Empty embedding returned")
                return False

        else:
            print("❌ No embedding_model attribute found")
            return False

    except Exception as e:
        print(f"❌ Error testing RAG embedding: {e}")
        return False

async def main():
    """Main test"""
    print("🚀 Testing RAG pipeline embedding access...\n")

    result = await test_rag_embedding()

    print(f"\n📊 Result: {'✅ SUCCESS' if result else '❌ FAILED'}")

    if not result:
        print("\n💡 The embedding issue is NOT fixed. Environment variables")
        print("   may be loaded in main process but not accessible to RAG pipeline.")

if __name__ == "__main__":
    asyncio.run(main())