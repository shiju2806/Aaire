#!/usr/bin/env python3
"""
Test ASC 255-10-50-51 query using the main application
"""

import asyncio
import json
from pathlib import Path

def test_simple_query():
    """Test if we can run a simple ASC query"""
    
    print("Testing ASC 255-10-50-51 query...")
    print("="*60)
    
    # Check if there are any files that might contain ASC codes
    uploads_dir = Path("data/uploads")
    
    print("Current uploaded files:")
    for file_path in uploads_dir.iterdir():
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024*1024)
            print(f"  {file_path.name} ({size_mb:.2f} MB)")
    
    print(f"\nLooking for job ID: 67fedc40-10e0-4d3e-8bbe-eca257af662f")
    
    # Try to find any files with similar timestamps or patterns
    for file_path in uploads_dir.iterdir():
        if file_path.is_file() and file_path.name.endswith('.pdf'):
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    if len(pdf_reader.pages) > 0:
                        first_page = pdf_reader.pages[0].extract_text()
                        if "foreign" in first_page.lower() or "currency" in first_page.lower() or "pwc" in first_page.lower():
                            print(f"\n🎯 Potential match: {file_path.name}")
                            print(f"First page content: {first_page[:200]}...")
                            
                            # Search for ASC codes in full document
                            full_text = ""
                            for page in pdf_reader.pages:
                                full_text += page.extract_text()
                            
                            if "255-10-50-51" in full_text:
                                print("✅ Found ASC 255-10-50-51 in this document!")
                                return file_path.name
                            else:
                                print("❌ ASC 255-10-50-51 not found in this document")
            except Exception as e:
                print(f"Error checking {file_path.name}: {e}")
    
    print("\n❌ Could not find the pwcforeigncurrency0522.pdf file")
    print("This suggests:")
    print("1. The upload may still be in progress")
    print("2. The file failed to save to the uploads directory") 
    print("3. The upload was successful but the file is stored elsewhere")
    
    return None

if __name__ == "__main__":
    test_simple_query()