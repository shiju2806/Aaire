#!/usr/bin/env python3
"""
Test the new intelligent validation system with adaptive quality gates
"""

import requests
import json

def test_intelligent_validation():
    """Test the intelligent validation system with various query types"""
    print("🔍 Testing intelligent validation system...")

    # Use a fresh ngrok link (you'll need to update this)
    url = "http://localhost:8009/api/v1/chat"  # Will use local first, then ngrok
    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true"
    }

    # Test queries that should trigger different validation scenarios
    test_queries = [
        {
            "name": "Capital Health Query (Previously Hallucinating)",
            "query": "what are the ratios to assess a company's capital health per ifrs 17",
            "expected": "Should be rejected with insufficient information"
        },
        {
            "name": "Generic Financial Query",
            "query": "what is the current ratio",
            "expected": "Should be rejected as generic content"
        },
        {
            "name": "Valid Document Query",
            "query": "how do I calculate the reserves for a universal life policy",
            "expected": "Should pass if information exists in documents"
        },
        {
            "name": "Specific Regulatory Query",
            "query": "what are the specific disclosure requirements under IFRS 17 section 100",
            "expected": "Should be handled based on document content"
        }
    ]

    for test in test_queries:
        print(f"\n🧪 Testing: {test['name']}")
        print(f"Query: {test['query']}")
        print(f"Expected: {test['expected']}")

        payload = {
            "query": test['query'],
            "session_id": f"test-intelligent-validation-{test['name'].lower().replace(' ', '-')}",
            "user_id": "test-user"
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                data = response.json()
                response_text = data.get('response', '')
                quality_metrics = data.get('quality_metrics', {})
                validation_results = quality_metrics.get('intelligent_validation', {})

                print(f"✅ Response received")
                print(f"📊 Length: {len(response_text)} chars")
                print(f"🔒 Validation passed: {validation_results.get('passed', 'N/A')}")
                print(f"📈 Overall score: {validation_results.get('overall_score', 'N/A')}")
                print(f"⏱️ Processing time: {validation_results.get('processing_time_ms', 'N/A')}ms")

                # Show validation components
                components = validation_results.get('components', {})
                if components:
                    print("🔍 Validation Components:")
                    for component, details in components.items():
                        if isinstance(details, dict):
                            if 'passed' in details:
                                status = "✅" if details['passed'] else "❌"
                                print(f"   {status} {component}: {details.get('score', 'N/A')}")

                print(f"📖 Response preview (first 200 chars):")
                print(f"   {response_text[:200]}...")

                # Check for hallucination indicators
                hallucination_indicators = [
                    "current ratio", "quick ratio", "debt to equity", "return on equity"
                ]
                found_indicators = [ind for ind in hallucination_indicators if ind.lower() in response_text.lower()]
                if found_indicators:
                    print(f"⚠️  Potential hallucination indicators found: {found_indicators}")

            else:
                print(f"❌ Error: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"❌ Exception: {e}")

        print("-" * 80)

def main():
    """Test intelligent validation system"""
    print("🚀 Testing Intelligent Validation System...\n")
    test_intelligent_validation()
    print(f"\n📊 Testing completed!")

if __name__ == "__main__":
    main()