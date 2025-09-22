#!/usr/bin/env python3
"""
Test script to verify OpenAI embedding generation is working
"""

import os
import asyncio
from openai import AsyncOpenAI

async def test_embedding_generation():
    """Test if we can generate embeddings using OpenAI API"""
    print("🔍 Testing OpenAI embedding generation...")

    # Check if API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No OPENAI_API_KEY environment variable found")
        return False

    print(f"✅ OpenAI API key found: {api_key[:10]}...")

    try:
        # Initialize client
        client = AsyncOpenAI(api_key=api_key)

        # Test text for embedding
        test_text = "This is a test document about whole life insurance reserves calculation in US STAT framework."

        print(f"📝 Testing with text: {test_text}")

        # Generate embedding
        response = await client.embeddings.create(
            model="text-embedding-ada-002",
            input=test_text
        )

        # Check result
        if response.data and len(response.data) > 0:
            embedding = response.data[0].embedding
            print(f"✅ Embedding generated successfully!")
            print(f"   - Dimensions: {len(embedding)}")
            print(f"   - First 5 values: {embedding[:5]}")
            print(f"   - Token usage: {response.usage.total_tokens}")
            return True
        else:
            print("❌ Empty embedding response")
            return False

    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return False

async def test_llamaindex_embedding():
    """Test LlamaIndex embedding model"""
    print("\n🔍 Testing LlamaIndex embedding model...")

    try:
        from llama_index.embeddings.openai import OpenAIEmbedding

        # Initialize embedding model
        embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

        # Test text
        test_text = "Universal life insurance policy reserves calculation"

        print(f"📝 Testing with text: {test_text}")

        # Generate embedding
        embedding = await embed_model.aget_text_embedding(test_text)

        if embedding and len(embedding) > 0:
            print(f"✅ LlamaIndex embedding generated successfully!")
            print(f"   - Dimensions: {len(embedding)}")
            print(f"   - First 5 values: {embedding[:5]}")
            return True
        else:
            print("❌ Empty LlamaIndex embedding")
            return False

    except Exception as e:
        print(f"❌ Error with LlamaIndex embedding: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting embedding generation tests...\n")

    # Test direct OpenAI API
    api_works = await test_embedding_generation()

    # Test LlamaIndex wrapper
    llamaindex_works = await test_llamaindex_embedding()

    print(f"\n📊 Results:")
    print(f"   - Direct OpenAI API: {'✅ Working' if api_works else '❌ Failed'}")
    print(f"   - LlamaIndex wrapper: {'✅ Working' if llamaindex_works else '❌ Failed'}")

    if api_works and llamaindex_works:
        print("\n🎉 Both embedding methods work - the issue is elsewhere!")
    elif api_works and not llamaindex_works:
        print("\n⚠️  OpenAI API works but LlamaIndex wrapper fails")
    elif not api_works:
        print("\n❌ OpenAI API authentication/access issue")

if __name__ == "__main__":
    asyncio.run(main())