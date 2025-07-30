#!/usr/bin/env python3
"""
Test script to debug OCR/chart analysis issue
"""

import requests
import json
import time
import base64
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"

def test_image_upload():
    """Test uploading an image and checking if OCR is triggered"""
    
    print("=== Testing Image Upload and OCR Processing ===\n")
    
    # Test with a simple image file (you can replace with actual ChatGPT image path)
    test_image_path = "chatgpt_revenue_chart.png"  # Replace with actual path
    
    # First, let's check if the server is running
    try:
        health_response = requests.get(f"{BASE_URL}/health")
        print(f"✅ Server is running: {health_response.json()}")
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        return
    
    # Check if test image exists
    if not Path(test_image_path).exists():
        print(f"⚠️  Test image not found: {test_image_path}")
        print("Creating a dummy test to check the flow...")
        
        # Test with a text file to see the flow
        test_file_path = "test_doc.txt"
        with open(test_file_path, "w") as f:
            f.write("FY23 Revenue: $100B\nFY24 Revenue: $150B\n")
        
        with open(test_file_path, "rb") as f:
            files = {"file": ("test_doc.txt", f, "text/plain")}
            metadata = {
                "title": "Test Revenue Document",
                "source_type": "COMPANY",
                "effective_date": "2025-01-30"
            }
            
            response = requests.post(
                f"{BASE_URL}/api/v1/upload",
                files=files,
                data={"metadata": json.dumps(metadata)}
            )
            
            print(f"\nUpload response: {response.status_code}")
            print(f"Response data: {json.dumps(response.json(), indent=2)}")
            
            if response.status_code == 200:
                job_id = response.json()["job_id"]
                print(f"\n✅ Document uploaded with job_id: {job_id}")
                
                # Wait for processing
                time.sleep(2)
                
                # Check job status
                status_response = requests.get(f"{BASE_URL}/api/v1/job/{job_id}")
                print(f"\nJob status: {json.dumps(status_response.json(), indent=2)}")
        
        # Clean up
        Path(test_file_path).unlink()
        return
    
    # Upload actual image
    print(f"\nUploading image: {test_image_path}")
    
    with open(test_image_path, "rb") as f:
        files = {"file": (test_image_path, f, "image/png")}
        metadata = {
            "title": "ChatGPT Revenue Chart",
            "source_type": "COMPANY",  
            "effective_date": "2025-01-30"
        }
        
        print(f"Metadata: {json.dumps(metadata, indent=2)}")
        
        response = requests.post(
            f"{BASE_URL}/api/v1/upload",
            files=files,
            data={"metadata": json.dumps(metadata)}
        )
        
        print(f"\nUpload response: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response data: {json.dumps(result, indent=2)}")
            job_id = result["job_id"]
            
            # Wait and check job status
            print("\n⏳ Waiting for processing...")
            for i in range(5):
                time.sleep(2)
                status_response = requests.get(f"{BASE_URL}/api/v1/job/{job_id}")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"\nJob status ({i+1}/5): {status_data.get('status', 'unknown')}")
                    print(f"Progress: {status_data.get('progress', 0)}%")
                    
                    if status_data.get('status') == 'completed':
                        print("\n✅ Processing completed!")
                        if 'summary' in status_data:
                            print("\nSummary:")
                            print(json.dumps(status_data['summary'], indent=2))
                        break
                    elif status_data.get('status') == 'failed':
                        print(f"\n❌ Processing failed: {status_data.get('error', 'Unknown error')}")
                        break
        else:
            print(f"❌ Upload failed: {response.text}")

def test_query():
    """Test querying about the uploaded image"""
    
    print("\n\n=== Testing Query About Image ===\n")
    
    query = "What was the FY23 revenue from the ChatGPT image?"
    
    print(f"Query: {query}")
    
    request_data = {
        "query": query,
        "session_id": "test_session_001",
        "filters": None,
        "conversation_history": [],
        "user_context": {
            "name": "Test User",
            "department": "Engineering",
            "role": "Developer"
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/chat",
        json=request_data
    )
    
    print(f"\nQuery response: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nResponse: {result['response']}")
        print(f"\nCitations: {len(result.get('citations', []))} found")
        if result.get('citations'):
            for i, citation in enumerate(result['citations'], 1):
                print(f"\nCitation {i}:")
                print(f"  Source: {citation.get('source', 'Unknown')}")
                print(f"  Confidence: {citation.get('confidence', 0)}")
                print(f"  Text: {citation.get('text', '')[:100]}...")
        else:
            print("❌ No citations found!")
        
        print(f"\nConfidence: {result.get('confidence', 0)}")
        print(f"Processing time: {result.get('processing_time_ms', 0)}ms")
    else:
        print(f"❌ Query failed: {response.text}")

def check_ocr_availability():
    """Check if OCR processors are available"""
    
    print("\n\n=== Checking OCR Availability ===\n")
    
    # This would need a debug endpoint to check, but we can infer from logs
    print("Check the server console/logs for:")
    print("- 'Using Tesseract OCR processor'")
    print("- 'OCR processor initialized'")
    print("- '[OCR] Processing image'")
    print("- '[Tesseract] Raw extraction'")
    print("\nIf these messages don't appear when uploading an image, OCR is not being triggered.")

if __name__ == "__main__":
    test_image_upload()
    test_query()
    check_ocr_availability()
    
    print("\n\n=== Debugging Steps ===")
    print("1. Check server console for OCR-related logs")
    print("2. Verify Tesseract is installed: 'tesseract --version'")
    print("3. Check if image file path is correct in the test")
    print("4. Look for any error messages in the server logs")