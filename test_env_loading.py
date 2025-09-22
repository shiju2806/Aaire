#!/usr/bin/env python3
"""
Test if dotenv loading works correctly
"""

import os
from dotenv import load_dotenv

print("🔍 Testing environment variable loading...")

# Check current working directory
print(f"Current working directory: {os.getcwd()}")

# Check if .env file exists
env_file_path = os.path.join(os.getcwd(), '.env')
print(f".env file exists: {os.path.exists(env_file_path)}")

# Load dotenv
result = load_dotenv()
print(f"load_dotenv() result: {result}")

# Check if OPENAI_API_KEY is loaded
api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    print(f"✅ OPENAI_API_KEY loaded: {api_key[:10]}...")
else:
    print("❌ OPENAI_API_KEY not found after load_dotenv()")

# Check other env vars from .env
qdrant_url = os.getenv('QDRANT_URL')
if qdrant_url:
    print(f"✅ QDRANT_URL loaded: {qdrant_url[:30]}...")
else:
    print("❌ QDRANT_URL not found")

print("\n📋 Environment variables test completed.")