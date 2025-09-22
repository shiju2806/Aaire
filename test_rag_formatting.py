#!/usr/bin/env python3
"""
Test the new RAG-optimized formatting system
"""

import requests
import json

def test_rag_formatting():
    """Test the new RAG-optimized formatting with markdown artifact removal"""
    print("🔍 Testing RAG-optimized formatting system...")

    url = "https://7a15851e7f63.ngrok-free.app/api/v1/chat"
    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true"
    }
    payload = {
        "query": "what are the ratios to assess a company's capital health in ifrs 17",
        "session_id": "test-rag-formatting",
        "user_id": "demo-user"
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            data = response.json()
            response_text = data.get('response', '')

            print(f"✅ Response received")
            print(f"📊 Length: {len(response_text)} chars")
            print(f"📚 Sources: {len(data.get('sources', []))}")

            # Check for markdown artifacts that should be removed
            markdown_artifacts = [
                "**",  # Bold markers
                "##",  # Headers
                "###", # Subheaders
                "```", # Code blocks
                "*"    # Asterisks (when used for emphasis)
            ]

            artifact_count = 0
            for artifact in markdown_artifacts:
                count = response_text.count(artifact)
                if count > 0:
                    print(f"❌ Found {count} instances of '{artifact}'")
                    artifact_count += count

            print(f"\n📖 Response:")
            print("="*80)
            print(response_text)
            print("="*80)

            # Assessment
            no_artifacts = artifact_count == 0
            has_content = len(response_text) > 100
            reasonable_length = len(response_text) < 3000

            print(f"\n🔍 RAG Formatting Analysis:")
            print(f"   ✅ No markdown artifacts: {no_artifacts} (found {artifact_count})")
            print(f"   ✅ Has content: {has_content}")
            print(f"   ✅ Reasonable length: {reasonable_length}")

            if no_artifacts and has_content:
                print(f"\n🎉 SUCCESS: RAG-optimized formatting working correctly!")
                return True
            else:
                print(f"\n⚠️  NEEDS IMPROVEMENT: Still contains markdown artifacts")
                return False
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    test_rag_formatting()