#!/usr/bin/env python3
"""
Simple check for ASC 255 in uploaded documents
"""

import os
import re
from pathlib import Path

def check_documents_for_asc():
    """Check documents for ASC codes without using full pipeline"""
    uploads_dir = Path("data/uploads")
    
    if not uploads_dir.exists():
        print(f"Upload directory not found: {uploads_dir}")
        return
    
    print(f"Checking documents in: {uploads_dir}")
    print("="*60)
    
    asc_pattern = re.compile(r'ASC\s+\d{3}-\d{2}-\d{2}-\d{2}', re.IGNORECASE)
    asc_255_pattern = re.compile(r'ASC\s+255[^0-9]', re.IGNORECASE)
    
    found_any_asc = False
    found_asc_255 = False
    
    for file_path in uploads_dir.iterdir():
        if file_path.is_file():
            print(f"\n📄 Checking: {file_path.name}")
            
            try:
                # Try to read as text
                if file_path.suffix.lower() in ['.txt', '.csv']:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Search for ASC patterns
                    asc_matches = asc_pattern.findall(content)
                    asc_255_matches = asc_255_pattern.findall(content)
                    
                    if asc_matches:
                        found_any_asc = True
                        print(f"   ✅ Found ASC codes: {', '.join(set(asc_matches))}")
                    
                    if asc_255_matches:
                        found_asc_255 = True
                        print(f"   🎯 Found ASC 255 references!")
                        # Show context around ASC 255
                        for match in asc_255_matches:
                            start_pos = content.find(match)
                            if start_pos != -1:
                                context_start = max(0, start_pos - 100)
                                context_end = min(len(content), start_pos + 200)
                                context = content[context_start:context_end]
                                print(f"      Context: ...{context}...")
                    
                    # Search specifically for "ASC 255-10-50-51"
                    if "255-10-50-51" in content:
                        print(f"   🔍 Found exact ASC 255-10-50-51!")
                        start_pos = content.find("255-10-50-51")
                        context_start = max(0, start_pos - 150)
                        context_end = min(len(content), start_pos + 300)
                        context = content[context_start:context_end]
                        print(f"      Full context: ...{context}...")
                    
                    if not asc_matches and not asc_255_matches:
                        print(f"   ❌ No ASC codes found")
                        
                elif file_path.suffix.lower() == '.pdf':
                    print(f"   📖 PDF file - would need OCR/PDF processing")
                else:
                    print(f"   ❓ Unsupported file type: {file_path.suffix}")
                    
            except Exception as e:
                print(f"   ⚠️  Error reading file: {e}")
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"Found any ASC codes: {found_any_asc}")
    print(f"Found ASC 255 references: {found_asc_255}")
    
    if not found_any_asc:
        print("\n🔍 No ASC codes found in any documents.")
        print("This suggests either:")
        print("1. No documents with ASC codes have been uploaded")
        print("2. ASC codes are in PDF files that need special processing")
        print("3. The search/indexing system has an issue")

if __name__ == "__main__":
    check_documents_for_asc()