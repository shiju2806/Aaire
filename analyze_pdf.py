#!/usr/bin/env python3
"""
Analyze PDF files for organizational chart content
"""

import os
import sys

def analyze_pdf_simple(pdf_path):
    """Simple PDF analysis without dependencies"""
    try:
        # Try PyPDF2 first
        import PyPDF2
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages[:3]:  # First 3 pages
                text += page.extract_text()
            
            # Look for organizational chart indicators
            indicators = [
                "director", "manager", "ceo", "cfo", "president", 
                "senior", "vice", "head", "chief", "officer",
                "department", "finance", "accounting", "audit",
                "structure", "organizational", "chart", "reporting"
            ]
            
            found_indicators = [ind for ind in indicators if ind.lower() in text.lower()]
            
            print(f"\n📄 File: {os.path.basename(pdf_path)}")
            print(f"📊 Pages: {len(reader.pages)}")
            print(f"🔍 Text sample: {text[:200]}...")
            print(f"🎯 Job/Org indicators found: {found_indicators}")
            
            if len(found_indicators) >= 3:
                print("✅ Likely contains organizational information")
                return True
            else:
                print("❌ Probably not organizational chart")
                return False
                
    except Exception as e:
        print(f"❌ Could not analyze {pdf_path}: {e}")
        return False

def main():
    upload_dir = "/Users/shijuprakash/AAIRE/data/uploads"
    pdf_files = [f for f in os.listdir(upload_dir) if f.endswith('.pdf')]
    
    print("🔍 **ANALYZING PDF FILES FOR ORGANIZATIONAL CONTENT**\n")
    
    candidates = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(upload_dir, pdf_file)
        if analyze_pdf_simple(pdf_path):
            candidates.append(pdf_file)
    
    print(f"\n🎯 **LIKELY FINANCE STRUCTURES CANDIDATES:**")
    for candidate in candidates:
        print(f"   • {candidate}")
    
    if not candidates:
        print("❌ No organizational charts found in uploaded PDFs")
        print("📤 Please upload the Finance Structures PDF to the web interface")

if __name__ == "__main__":
    main()