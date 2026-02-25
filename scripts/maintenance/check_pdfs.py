#!/usr/bin/env python3
"""
Check PDF files for ASC codes using PyPDF2
"""

import re
from pathlib import Path
import PyPDF2

def check_pdf_files():
    """Check PDF files for ASC codes"""
    uploads_dir = Path("data/uploads")
    
    pdf_files = list(uploads_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to check")
    print("="*60)
    
    asc_pattern = re.compile(r'ASC\s+\d{3}-\d{2}-\d{2}-\d{2}', re.IGNORECASE)
    
    for pdf_path in pdf_files:
        print(f"\n📄 Checking PDF: {pdf_path.name}")
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                print(f"   Pages: {len(pdf_reader.pages)}")
                
                full_text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        full_text += f"[Page {page_num + 1}]\n{page_text}\n\n"
                        print(f"   Page {page_num + 1}: {len(page_text)} characters")
                    except Exception as e:
                        print(f"   Page {page_num + 1}: Error - {e}")
                
                # Search for ASC patterns
                asc_matches = asc_pattern.findall(full_text)
                if asc_matches:
                    print(f"   ✅ Found ASC codes: {', '.join(set(asc_matches))}")
                
                # Search specifically for "ASC 255"
                if "ASC 255" in full_text.upper() or "255-10-50-51" in full_text:
                    print(f"   🎯 Found ASC 255 or 255-10-50-51!")
                    
                    # Find context around ASC 255
                    if "255-10-50-51" in full_text:
                        start_pos = full_text.find("255-10-50-51")
                        context_start = max(0, start_pos - 200)
                        context_end = min(len(full_text), start_pos + 400)
                        context = full_text[context_start:context_end]
                        print(f"      Context: ...{context}...")
                
                if not asc_matches and "255" not in full_text:
                    print(f"   ❌ No ASC codes found")
                    # Show a sample of the text to verify extraction worked
                    sample = full_text[:300].replace('\n', ' ')
                    print(f"   Sample text: {sample}...")
                
        except Exception as e:
            print(f"   ⚠️  Error reading PDF: {e}")

if __name__ == "__main__":
    check_pdf_files()