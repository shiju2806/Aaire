#!/usr/bin/env python3
"""
Test the improved brevity and response merging
"""

import requests
import json

def test_brevity():
    """Test condensed response format"""
    print("🔍 Testing improved brevity and response merging...")

    url = "https://34706080ef01.ngrok-free.app/api/v1/chat"
    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true"
    }
    payload = {
        "query": "what are the ratios to assess a company's capital health in ifrs 17",
        "session_id": "test-brevity",
        "user_id": "demo-user"
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            data = response.json()
            response_text = data.get('response', '')

            print(f"✅ Response received")
            print(f"📊 Length: {len(response_text)} chars (vs 5000+ before)")
            print(f"📚 Sources: {len(data.get('sources', []))}")

            # Count sections
            section_count = response_text.count('## Section')
            no_info_count = response_text.lower().count('does not contain information')

            print(f"📋 Sections: {section_count}")
            print(f"❌ 'No info' sections: {no_info_count}")

            print(f"\n📖 Response:")
            print(response_text)

            # Assessment
            is_concise = len(response_text) < 2000
            no_empty_sections = no_info_count == 0
            has_content = len(response_text) > 200

            print(f"\n🔍 Analysis:")
            print(f"   ✅ Concise (<2000 chars): {is_concise}")
            print(f"   ✅ No empty sections: {no_empty_sections}")
            print(f"   ✅ Has content: {has_content}")

            if is_concise and no_empty_sections and has_content:
                print(f"\n🎉 SUCCESS: Response is concise and well-formatted!")
                return True
            else:
                print(f"\n⚠️  NEEDS IMPROVEMENT")
                return False
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    test_brevity()