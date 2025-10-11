#!/usr/bin/env python3
"""Re-index all 3 documents with new chunk size (1800 chars, 250 overlap)"""

import requests
import sys
from pathlib import Path
import time

# The 3 PDF files to re-index
pdf_files = [
    Path("/Users/shijuprakash/Downloads/2025 Edition - Valuation Manual.pdf"),
    Path("/Users/shijuprakash/Downloads/IFRS General.pdf"),
    Path("/Users/shijuprakash/Downloads/LICAT.pdf")
]

print("=" * 80)
print("📚 RE-INDEXING ALL 3 DOCUMENTS WITH NEW CHUNK SIZE")
print("=" * 80)
print(f"New chunk configuration: 1800 chars, 250 overlap")
print(f"Documents to re-index: {len(pdf_files)}")
print("=" * 80)

# Verify all files exist
for pdf_file in pdf_files:
    if not pdf_file.exists():
        print(f"❌ File not found: {pdf_file}")
        sys.exit(1)
    size_mb = pdf_file.stat().st_size / 1024 / 1024
    print(f"✅ {pdf_file.name} ({size_mb:.2f} MB)")

url = "http://localhost:8080/api/v1/upload"
print(f"\n🚀 Uploading to {url}...")
print("=" * 80)

# Upload each file
successful_uploads = []
failed_uploads = []

for idx, pdf_file in enumerate(pdf_files, 1):
    print(f"\n📄 [{idx}/3] Uploading: {pdf_file.name}")

    try:
        with open(pdf_file, 'rb') as f:
            files = {'file': (pdf_file.name, f, 'application/pdf')}
            response = requests.post(url, files=files, timeout=300)

        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success! Job ID: {result.get('job_id', 'N/A')}")
            successful_uploads.append(pdf_file.name)
        else:
            print(f"   ❌ Failed: {response.status_code}")
            print(f"   Response: {response.text}")
            failed_uploads.append(pdf_file.name)

    except requests.exceptions.Timeout:
        print(f"   ⏱️  Upload timed out (may still be processing in background)")
        successful_uploads.append(f"{pdf_file.name} (timeout - check logs)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        failed_uploads.append(pdf_file.name)

    # Brief pause between uploads
    if idx < len(pdf_files):
        time.sleep(2)

print("\n" + "=" * 80)
print("📊 RE-INDEXING SUMMARY")
print("=" * 80)
print(f"✅ Successful: {len(successful_uploads)}")
for name in successful_uploads:
    print(f"   - {name}")

if failed_uploads:
    print(f"\n❌ Failed: {len(failed_uploads)}")
    for name in failed_uploads:
        print(f"   - {name}")
    sys.exit(1)
else:
    print("\n" + "=" * 80)
    print("✅ ALL 3 DOCUMENTS QUEUED FOR RE-INDEXING")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Wait for indexing to complete (check server logs)")
    print("2. Test query: 'how do I calculate the reserves for a universal life policy in usstat'")
    print("3. Verify DR chunks now appear in retrieval logs")
    print("4. Check that chunk sizes are ~1800 chars instead of ~1024")