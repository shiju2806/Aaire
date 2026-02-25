#!/usr/bin/env python3
"""
Check the newly uploaded PDF for ASC 255-10-50-51
"""

import re
from pathlib import Path
import PyPDF2

def check_new_pdf():
    """Check the newly uploaded PDF for ASC codes"""
    uploads_dir = Path("data/uploads")
    
    # Look for the specific file
    target_file = None
    for pdf_file in uploads_dir.glob("*.pdf"):
        if pdf_file.stat().st_size > 1000000:  # Look for files > 1MB (1.95MB file)
            target_file = pdf_file
            break
    
    if not target_file:
        print("Could not find the 1.95MB PDF file")
        return
    
    print(f"📄 Checking: {target_file.name}")
    print(f"File size: {target_file.stat().st_size / (1024*1024):.2f} MB")
    print("="*60)
    
    try:
        with open(target_file, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            print(f"Pages: {len(pdf_reader.pages)}")
            
            full_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    full_text += f"[Page {page_num + 1}]\n{page_text}\n\n"
                except Exception as e:
                    print(f"Error on page {page_num + 1}: {e}")
            
            print(f"Total extracted text: {len(full_text)} characters")
            
            # Search for ASC patterns
            asc_pattern = re.compile(r'ASC\s+\d{3}-\d{2}-\d{2}-\d{2}', re.IGNORECASE)
            asc_matches = asc_pattern.findall(full_text)
            
            if asc_matches:
                print(f"\n✅ Found ASC codes: {', '.join(set(asc_matches))}")
            
            # Search specifically for "ASC 255-10-50-51"
            if "255-10-50-51" in full_text:
                print(f"\n🎯 Found ASC 255-10-50-51!")
                
                # Find all occurrences and show context
                positions = []
                start = 0
                while True:
                    pos = full_text.find("255-10-50-51", start)
                    if pos == -1:
                        break
                    positions.append(pos)
                    start = pos + 1
                
                print(f"Found {len(positions)} occurrences of '255-10-50-51'")
                
                for i, pos in enumerate(positions[:3]):  # Show first 3 occurrences
                    context_start = max(0, pos - 200)
                    context_end = min(len(full_text), pos + 400)
                    context = full_text[context_start:context_end]
                    print(f"\n--- Occurrence {i+1} ---")
                    print(f"Context: ...{context}...")
            
            # Search for "nonmonetary" to verify content
            if "nonmonetary" in full_text.lower():
                print(f"\n✅ Found 'nonmonetary' content!")
                pos = full_text.lower().find("nonmonetary")
                context_start = max(0, pos - 150)
                context_end = min(len(full_text), pos + 300)
                context = full_text[context_start:context_end]
                print(f"Context: ...{context}...")
            
            if not asc_matches and "255" not in full_text:
                print(f"\n❌ No ASC codes found")
                # Show sample to verify extraction
                sample = full_text[:500].replace('\n', ' ')
                print(f"Sample: {sample}...")
                
    except Exception as e:
        print(f"Error processing PDF: {e}")

if __name__ == "__main__":
    check_new_pdf()