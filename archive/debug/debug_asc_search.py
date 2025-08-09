#!/usr/bin/env python3
"""
Debug why ASC 255-10-50-51 search returns wrong document
"""

import re
from pathlib import Path
import PyPDF2

def debug_asc_search():
    """Debug which documents contain ASC references and why search might be confused"""
    uploads_dir = Path("data/uploads")
    
    print("=== DEBUG: ASC Search Issues ===")
    print("Analyzing all documents for ASC content...")
    print("="*60)
    
    docs_analysis = {}
    
    for file_path in uploads_dir.glob("*.pdf"):
        print(f"\n📄 Analyzing: {file_path.name}")
        
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                full_text = ""
                for page in pdf_reader.pages:
                    try:
                        full_text += page.extract_text() + "\n"
                    except:
                        continue
                
                # Analysis for this document
                analysis = {
                    'filename': file_path.name,
                    'size_mb': file_path.stat().st_size / (1024*1024),
                    'pages': len(pdf_reader.pages),
                    'total_chars': len(full_text),
                    'asc_references': [],
                    'has_255': False,
                    'has_830': False,
                    'accounting_keywords': 0,
                    'insurance_keywords': 0
                }
                
                # Find all ASC references
                asc_pattern = re.compile(r'ASC\s+\d{3}(?:-\d{2}(?:-\d{2}(?:-\d{1,2})?)?)?', re.IGNORECASE)
                asc_matches = asc_pattern.findall(full_text)
                analysis['asc_references'] = list(set(asc_matches))
                
                # Check for specific ASC codes
                analysis['has_255'] = '255' in full_text
                analysis['has_830'] = '830' in full_text
                
                # Count accounting vs insurance keywords
                accounting_keywords = ['gaap', 'fasb', 'ifrs', 'revenue', 'liability', 'asset', 'equity', 'nonmonetary', 'monetary', 'foreign currency']
                insurance_keywords = ['licat', 'actuarial', 'premium', 'policy', 'insurer', 'regulatory capital', 'solvency']
                
                text_lower = full_text.lower()
                for keyword in accounting_keywords:
                    analysis['accounting_keywords'] += text_lower.count(keyword)
                
                for keyword in insurance_keywords:
                    analysis['insurance_keywords'] += text_lower.count(keyword)
                
                docs_analysis[file_path.name] = analysis
                
                # Print summary for this doc
                print(f"  Size: {analysis['size_mb']:.2f} MB, Pages: {analysis['pages']}")
                print(f"  ASC references: {analysis['asc_references']}")
                print(f"  Has 255: {analysis['has_255']}, Has 830: {analysis['has_830']}")
                print(f"  Accounting keywords: {analysis['accounting_keywords']}")
                print(f"  Insurance keywords: {analysis['insurance_keywords']}")
                
                # Check for exact strings
                if "255-10-50-51" in full_text:
                    print(f"  ✅ Contains EXACT '255-10-50-51'")
                    # Show context
                    pos = full_text.find("255-10-50-51")
                    context = full_text[max(0, pos-100):pos+200]
                    print(f"  Context: ...{context}...")
                
                if "830-20-35-8" in full_text:
                    print(f"  ✅ Contains EXACT '830-20-35-8'")
                
        except Exception as e:
            print(f"  ❌ Error processing: {e}")
    
    print("\n" + "="*60)
    print("SUMMARY ANALYSIS:")
    print("="*60)
    
    # Find the best document for ASC 255-10-50-51
    best_for_255 = None
    best_score_255 = 0
    
    for filename, analysis in docs_analysis.items():
        score = 0
        if analysis['has_255']:
            score += 10
        if any('255' in ref for ref in analysis['asc_references']):
            score += 20
        score += analysis['accounting_keywords'] * 0.1
        
        print(f"\n{filename}:")
        print(f"  Score for ASC 255: {score}")
        print(f"  Primary content: {'Insurance' if analysis['insurance_keywords'] > analysis['accounting_keywords'] else 'Accounting'}")
        
        if score > best_score_255:
            best_score_255 = score
            best_for_255 = filename
    
    print(f"\n🎯 BEST DOCUMENT for ASC 255-10-50-51: {best_for_255} (score: {best_score_255})")
    
    # Identify why search might be returning wrong document
    print(f"\n🔍 POTENTIAL SEARCH ISSUES:")
    if len(docs_analysis) > 1:
        sorted_docs = sorted(docs_analysis.items(), key=lambda x: x[1]['accounting_keywords'], reverse=True)
        print(f"1. Most accounting content: {sorted_docs[0][0]} ({sorted_docs[0][1]['accounting_keywords']} keywords)")
        
        sorted_by_asc = sorted(docs_analysis.items(), key=lambda x: len(x[1]['asc_references']), reverse=True)
        print(f"2. Most ASC references: {sorted_by_asc[0][0]} ({len(sorted_by_asc[0][1]['asc_references'])} references)")
        
        if best_for_255 != sorted_docs[0][0]:
            print(f"⚠️  WARNING: Best document for ASC 255 is NOT the one with most accounting content!")
            print(f"   This could cause search confusion in vector/semantic search")

if __name__ == "__main__":
    debug_asc_search()