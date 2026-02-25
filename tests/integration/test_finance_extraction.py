#!/usr/bin/env python3
"""
Test Finance Structures document extraction without full system
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_document_extraction(file_path):
    """Test extraction on a specific PDF file"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
        
    print(f"🔍 **TESTING EXTRACTION FOR:** {os.path.basename(file_path)}")
    print("=" * 60)
    
    try:
        # Try basic PDF extraction first
        import PyPDF2
        
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            print(f"📊 PDF Info:")
            print(f"   • Pages: {len(reader.pages)}")
            print(f"   • File size: {os.path.getsize(file_path):,} bytes")
            
            # Extract text from first few pages
            print(f"\n📄 **TEXT EXTRACTION SAMPLE:**")
            for page_num in range(min(3, len(reader.pages))):
                page = reader.pages[page_num]
                text = page.extract_text()
                
                print(f"\n--- Page {page_num + 1} ---")
                print(f"Characters: {len(text)}")
                
                # Look for organizational indicators
                lines = text.split('\n')
                org_lines = []
                for line in lines[:20]:  # First 20 lines
                    line = line.strip()
                    if line and any(keyword in line.lower() for keyword in [
                        'director', 'manager', 'ceo', 'cfo', 'president', 
                        'senior', 'vice', 'head', 'chief', 'officer',
                        'finance', 'accounting', 'audit', 'department'
                    ]):
                        org_lines.append(line)
                
                if org_lines:
                    print(f"🎯 **POTENTIAL JOB TITLES/ROLES FOUND:**")
                    for line in org_lines:
                        print(f"   • {line}")
                else:
                    print(f"❌ No obvious job titles found on page {page_num + 1}")
                
                # Show text sample
                print(f"\n📝 **TEXT SAMPLE:**")
                print(text[:300] + "..." if len(text) > 300 else text)
                
    except Exception as e:
        print(f"❌ Error extracting text: {e}")
        
    # Try the shape-aware extraction if available
    try:
        from src.pdf_spatial_extractor import PDFSpatialExtractor
        
        print(f"\n🔧 **TESTING SHAPE-AWARE EXTRACTION:**")
        extractor = PDFSpatialExtractor()
        result = extractor.extract_with_coordinates(file_path)
        
        if result and 'organizational_units' in result:
            units = result['organizational_units']
            print(f"✅ Shape-aware extraction successful!")
            print(f"📊 Extracted {len(units)} organizational units")
            
            for i, unit in enumerate(units[:5]):  # Show first 5
                print(f"\n  Unit {i+1}:")
                print(f"    Text: {unit.get('text', 'Unknown')}")
                print(f"    Type: {unit.get('line_type', 'Unknown')}")
                print(f"    Coordinates: ({unit.get('x0', 0):.1f}, {unit.get('y0', 0):.1f})")
                print(f"    Confidence: {unit.get('confidence', 0):.2f}")
        elif result:
            print(f"✅ Shape-aware extraction successful!")
            print(f"📊 Result keys: {list(result.keys())}")
            print(f"📄 Sample result: {str(result)[:200]}...")
        else:
            print(f"❌ Shape-aware extraction failed or found no data")
            
    except ImportError:
        print(f"⚠️  Shape-aware extraction not available (missing dependencies)")
    except Exception as e:
        print(f"❌ Shape-aware extraction error: {e}")

def main():
    print("📋 **FINANCE STRUCTURES EXTRACTION TEST**\n")
    
    # Check for common file locations
    possible_files = [
        "/Users/shijuprakash/AAIRE/finance_structures.pdf",
        "/Users/shijuprakash/AAIRE/data/uploads/finance_structures.pdf",
        "/Users/shijuprakash/Downloads/finance_structures.pdf",
        "/Users/shijuprakash/Desktop/finance_structures.pdf"
    ]
    
    # Check uploaded PDFs
    upload_dir = "/Users/shijuprakash/AAIRE/data/uploads"
    if os.path.exists(upload_dir):
        pdf_files = [f for f in os.listdir(upload_dir) if f.endswith('.pdf')]
        for pdf_file in pdf_files:
            possible_files.append(os.path.join(upload_dir, pdf_file))
    
    print("🔍 **LOOKING FOR FINANCE STRUCTURES DOCUMENT:**")
    found_files = [f for f in possible_files if os.path.exists(f)]
    
    if not found_files:
        print("❌ No PDF files found. Please:")
        print("   1. Copy your Finance Structures PDF to this directory")
        print("   2. Name it 'finance_structures.pdf'")
        print("   3. Run this script again")
        print("\n   Or provide the full path as argument:")
        print("   python3 test_finance_extraction.py /path/to/your/file.pdf")
        return
        
    if len(sys.argv) > 1:
        # File path provided as argument
        test_document_extraction(sys.argv[1])
    else:
        # Test all found PDFs
        print(f"✅ Found {len(found_files)} PDF files to test:")
        for file_path in found_files:
            print(f"   • {os.path.basename(file_path)}")
        
        print("\n" + "="*60)
        for file_path in found_files:
            test_document_extraction(file_path)
            print("\n" + "="*60)

if __name__ == "__main__":
    main()