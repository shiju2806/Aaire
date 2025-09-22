#!/usr/bin/env python3
"""
Test specific queries to analyze content retrieval
"""

import requests
import json

def test_query(query, description):
    """Test a query and return the response"""
    print(f"\n🔍 Testing: {description}")
    print(f"Query: {query}")

    url = "https://34706080ef01.ngrok-free.app/api/v1/chat"
    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true"
    }
    payload = {
        "query": query,
        "session_id": f"test-{description.lower().replace(' ', '-')}",
        "user_id": "demo-user"
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response received (length: {len(data.get('response', ''))} chars)")
            print(f"📊 Confidence: {data.get('confidence', 'N/A')}")
            print(f"📚 Sources: {data.get('sources', [])}")

            # Check for DR and SR mentions
            response_text = data.get('response', '').lower()
            dr_mentions = response_text.count('deterministic reserve') + response_text.count(' dr ')
            sr_mentions = response_text.count('stochastic reserve') + response_text.count(' sr ')

            print(f"🔍 DR mentions: {dr_mentions}")
            print(f"🔍 SR mentions: {sr_mentions}")

            # Show first part of response
            response_preview = data.get('response', '')[:500]
            print(f"📖 Response preview:\n{response_preview}...")

            return data
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def main():
    """Test both queries"""
    print("🚀 Testing query content analysis...\n")

    # Test 1: Universal Life
    ul_response = test_query(
        "how do I calculate the reserves for a universal life policy in usstat",
        "Universal Life Reserves"
    )

    print("\n" + "="*80)

    # Test 2: Whole Life
    wl_response = test_query(
        "how do I calculate the reserves for a whole life policy in usstat",
        "Whole Life Reserves"
    )

    print("\n" + "="*80)
    print("📊 ANALYSIS SUMMARY:")

    if ul_response and wl_response:
        ul_text = ul_response.get('response', '').lower()
        wl_text = wl_response.get('response', '').lower()

        print(f"Universal Life response: {len(ul_text)} chars")
        print(f"Whole Life response: {len(wl_text)} chars")

        print(f"Universal Life DR/SR mentions: {ul_text.count('deterministic') + ul_text.count(' dr ') + ul_text.count('stochastic') + ul_text.count(' sr ')}")
        print(f"Whole Life DR/SR mentions: {wl_text.count('deterministic') + wl_text.count(' dr ') + wl_text.count('stochastic') + wl_text.count(' sr ')}")

if __name__ == "__main__":
    main()