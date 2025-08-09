#!/usr/bin/env python3
"""
Test if citations are being included in API responses
"""

import requests
import json

def test_citation_api():
    """Test if the API is returning citations"""
    
    print("=== TESTING CITATION API RESPONSE ===")
    print("Testing if citations are included in API response")
    print("="*60)
    
    # Test query
    query = "what is ASC 255-10-50-51?"
    
    # API endpoint (adjust if needed)
    api_url = "http://localhost:8000/api/v1/query"  # or whatever your API endpoint is
    
    payload = {
        "query": query,
        "user_id": "test_user"
    }
    
    try:
        print(f"Sending request to: {api_url}")
        print(f"Query: {query}")
        
        response = requests.post(api_url, json=payload, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"Response keys: {list(data.keys())}")
            
            # Check if response has answer
            if 'answer' in data:
                answer = data['answer']
                print(f"Answer length: {len(answer)} characters")
                print(f"Answer preview: {answer[:100]}...")
            
            # Check if response has citations
            if 'citations' in data:
                citations = data['citations']
                print(f"Citations count: {len(citations)}")
                
                for i, citation in enumerate(citations):
                    print(f"Citation {i+1}:")
                    print(f"  Source: {citation.get('source', 'Unknown')}")
                    print(f"  Confidence: {citation.get('confidence', 'Unknown')}")
                    print(f"  Text preview: {citation.get('text', '')[:100]}...")
            else:
                print("❌ No 'citations' key in response")
            
            # Check confidence
            if 'confidence' in data:
                print(f"Confidence: {data['confidence']}")
            
            # Full response structure
            print(f"\nFull response structure:")
            print(json.dumps(data, indent=2)[:500] + "..." if len(str(data)) > 500 else json.dumps(data, indent=2))
            
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API - is the server running?")
        print("Make sure AAIRE is running on localhost:8000")
    except Exception as e:
        print(f"❌ Error testing API: {e}")
    
    print(f"\n=== ALTERNATIVE DEBUGGING ===")
    print("If API test fails, check:")
    print("1. Server logs for citation creation messages")
    print("2. Browser developer tools network tab")
    print("3. Frontend citation rendering code")
    print("4. Try different query to see if issue is specific to ASC queries")

if __name__ == "__main__":
    test_citation_api()