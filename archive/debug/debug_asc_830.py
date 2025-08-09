#!/usr/bin/env python3
"""
Debug ASC 830-20-35-8 search in uploaded documents
"""

import re
from pathlib import Path
import PyPDF2

def search_asc_830():
    """Search for ASC 830-20-35-8 in uploaded documents"""
    uploads_dir = Path("data/uploads")
    
    print("Searching for ASC 830-20-35-8 in uploaded documents...")
    print("="*60)
    
    # Look for files that might contain foreign currency content
    for file_path in uploads_dir.glob("*.pdf"):
        print(f"\n📄 Checking: {file_path.name}")
        print(f"Size: {file_path.stat().st_size / (1024*1024):.2f} MB")
        
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                print(f"Pages: {len(pdf_reader.pages)}")
                
                full_text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        full_text += f"[Page {page_num + 1}]\n{page_text}\n\n"
                    except Exception as e:
                        print(f"Error on page {page_num + 1}: {e}")
                
                # Search for ASC 830 references
                asc_830_pattern = re.compile(r'ASC\s+830[^0-9]', re.IGNORECASE)
                asc_830_matches = asc_830_pattern.findall(full_text)
                
                if asc_830_matches:
                    print(f"✅ Found ASC 830 references: {len(asc_830_matches)}")
                    
                    # Search specifically for ASC 830-20-35-8
                    if "830-20-35-8" in full_text:
                        print(f"🎯 Found ASC 830-20-35-8!")
                        
                        # Find context around it
                        pos = full_text.find("830-20-35-8")
                        context_start = max(0, pos - 300)
                        context_end = min(len(full_text), pos + 500)
                        context = full_text[context_start:context_end]
                        print(f"Context:\n{context}")
                        
                        return file_path.name
                    else:
                        print(f"❌ ASC 830-20-35-8 not found, but found other ASC 830 references")
                        
                        # Show what ASC 830 content is available
                        asc_830_sections = re.findall(r'ASC\s+830-\d{2}-\d{2}-\d{1,2}', full_text, re.IGNORECASE)
                        if asc_830_sections:
                            print(f"Available ASC 830 sections: {set(asc_830_sections)}")
                else:
                    print(f"❌ No ASC 830 references found")
                    
                    # Check if this looks like the foreign currency document
                    if any(keyword in full_text.lower() for keyword in ['foreign currency', 'exchange rate', 'pwc']):
                        print(f"✅ This appears to be a foreign currency document")
                        # Show first few ASC references to see what's available
                        asc_pattern = re.compile(r'ASC\s+\d{3}-\d{2}-\d{2}-\d{1,2}', re.IGNORECASE)
                        asc_matches = asc_pattern.findall(full_text)
                        if asc_matches:
                            unique_asc = list(set(asc_matches))[:10]
                            print(f"ASC codes found: {unique_asc}")
                    
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
    
    print(f"\n❌ ASC 830-20-35-8 not found in any uploaded documents")
    return None

if __name__ == "__main__":
    search_asc_830()