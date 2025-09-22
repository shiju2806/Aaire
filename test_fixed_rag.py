#!/usr/bin/env python3
"""
Test the fixed RAG system to ensure no hallucination on LICAT query
"""

import requests
import json

def test_fixed_rag():
    """Test the capital health query that was previously hallucinating"""
    print("🔍 Testing fixed RAG system with capital health query...")

    url = "https://34706080ef01.ngrok-free.app/api/v1/chat"
    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true"
    }
    payload = {
        "query": "what are the ratios to assess a companies capital health",
        "session_id": "test-fixed-rag",
        "user_id": "demo-user"
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            data = response.json()
            response_text = data.get('response', '')

            print(f"✅ Response received (length: {len(response_text)} chars)")
            print(f"📊 Confidence: {data.get('confidence', 'N/A')}")
            print(f"📚 Sources: {data.get('sources', [])}")

            # Check if it's a proper "I don't know" response
            insufficient_indicators = [
                "don't have sufficient information",
                "don't have any relevant information",
                "not contain information",
                "insufficient information"
            ]

            is_proper_rejection = any(indicator in response_text.lower() for indicator in insufficient_indicators)

            # Check if it contains generic ratio definitions (hallucination)
            hallucination_indicators = [
                "current ratio",
                "quick ratio",
                "debt to equity",
                "return on equity",
                "1., 2., 3."  # Generic numbered list format
            ]

            contains_hallucination = any(indicator in response_text.lower() for indicator in hallucination_indicators)

            print(f"\n📖 Response preview:\n{response_text}")

            print(f"\n🔍 Analysis:")
            print(f"   ✅ Proper rejection: {is_proper_rejection}")
            print(f"   ❌ Contains hallucination: {contains_hallucination}")

            if is_proper_rejection and not contains_hallucination:
                print(f"\n🎉 SUCCESS: RAG properly rejected query without hallucination!")
                return True
            elif contains_hallucination:
                print(f"\n❌ FAILED: RAG still hallucinating generic content!")
                return False
            else:
                print(f"\n⚠️  UNCLEAR: Response format unexpected")
                return False
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    """Test fixed RAG system"""
    print("🚀 Testing Fixed RAG System...\n")

    result = test_fixed_rag()

    print(f"\n📊 Result: {'✅ FIXED' if result else '❌ STILL BROKEN'}")

if __name__ == "__main__":
    main()