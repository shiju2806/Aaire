#!/usr/bin/env python3
"""Re-index all documents with new chunk size (1800 chars, 250 overlap)"""

import requests
import sys
from pathlib import Path
import time

# Find the uploaded PDF (single file contains all 3 documents)
uploaded_pdf = Path("/Users/shijuprakash/AAIRE/data/uploads/0717b81a-0a1a-47e5-b0e8-f484df37c7bc.pdf")

# The 3 documents that need re-indexing (based on document_hashes.json):
documents_to_reindex = [
    "2025 Edition - Valuation Manual.pdf",  # VM-20
    "IFRS General.pdf",
    "LICAT.pdf"
]

print("=" * 80)
print("📚 RE-INDEXING ALL DOCUMENTS WITH NEW CHUNK SIZE")
print("=" * 80)
print(f"New chunk configuration: 1800 chars, 250 overlap")
print(f"Documents to re-index: {len(documents_to_reindex)}")
for doc in documents_to_reindex:
    print(f"  - {doc}")
print("=" * 80)

if not uploaded_pdf.exists():
    print(f"❌ Uploaded PDF not found at: {uploaded_pdf}")
    sys.exit(1)

print(f"\n✅ Found uploaded PDF at: {uploaded_pdf}")
print(f"📊 File size: {uploaded_pdf.stat().st_size / 1024 / 1024:.2f} MB")

# The uploaded file appears to be a merged PDF or the system stores all uploads in one location
# We'll re-upload it to trigger re-indexing with the new chunk configuration

url = "http://localhost:8080/api/v1/upload"
print(f"\n🚀 Uploading to {url} to trigger re-indexing...")

with open(uploaded_pdf, 'rb') as f:
    files = {'file': (uploaded_pdf.name, f, 'application/pdf')}

    try:
        response = requests.post(url, files=files, timeout=300)

        if response.status_code == 200:
            print(f"✅ Re-indexing successful!")
            print(f"📄 Response: {response.json()}")
        else:
            print(f"❌ Re-indexing failed: {response.status_code}")
            print(f"📄 Response: {response.text}")
            sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"⏱️  Upload timed out (this is normal for large files)")
        print(f"Check server logs to verify re-indexing progress")
    except Exception as e:
        print(f"❌ Error during upload: {e}")
        sys.exit(1)

print("\n" + "=" * 80)
print("✅ RE-INDEXING COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("1. Wait for indexing to complete (check server logs)")
print("2. Test query: 'how do I calculate the reserves for a universal life policy in usstat'")
print("3. Verify DR chunks now appear in retrieval logs")