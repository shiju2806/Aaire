#!/usr/bin/env python3
"""Re-index VM-20 with new chunk size"""

import requests
import sys
from pathlib import Path

# Find the VM-20 PDF
vm20_path = Path("/Users/shijuprakash/AAIRE/data/2025 Edition - Valuation Manual.pdf")

if not vm20_path.exists():
    print(f"❌ VM-20 PDF not found at: {vm20_path}")
    sys.exit(1)

print(f"✅ Found VM-20 PDF at: {vm20_path}")
print(f"📊 File size: {vm20_path.stat().st_size / 1024 / 1024:.2f} MB")

# Upload to trigger re-indexing
url = "http://localhost:8080/api/v1/upload"
print(f"\n🚀 Uploading to {url}...")

with open(vm20_path, 'rb') as f:
    files = {'file': (vm20_path.name, f, 'application/pdf')}
    response = requests.post(url, files=files)

if response.status_code == 200:
    print(f"✅ Re-indexing successful!")
    print(f"📄 Response: {response.json()}")
else:
    print(f"❌ Re-indexing failed: {response.status_code}")
    print(f"📄 Response: {response.text}")
    sys.exit(1)