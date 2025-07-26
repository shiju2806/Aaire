#!/usr/bin/env python3
"""
Debug script to test document upload and processing flow
"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append('src')

from src.rag_pipeline import RAGPipeline
from src.document_processor import DocumentProcessor


async def test_document_processing():
    """Test the full document processing pipeline"""
    
    print("🔍 Testing Document Processing Pipeline")
    print("=" * 50)
    
    # Initialize components
    print("\n1. Initializing components...")
    try:
        rag_pipeline = RAGPipeline()
        print("✅ RAG Pipeline initialized")
        
        document_processor = DocumentProcessor(rag_pipeline)
        print("✅ Document Processor initialized")
        
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        return
    
    # Create a test PDF file (create a simple PDF using reportlab if available)
    print("\n2. Creating test document...")
    test_content = """
    ACCOUNTING POLICY DOCUMENT
    
    This is a sample accounting policy document for testing purposes.
    
    Key Topics:
    - Revenue Recognition under ASC 606
    - Lease Accounting under ASC 842
    - Financial Instruments under ASC 326
    
    Effective Date: January 1, 2024
    Last Updated: December 2023
    
    This document contains important guidance on revenue recognition principles
    and should be reviewed by all accounting personnel.
    """
    
    # Create a temporary PDF file for testing
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            temp_file_path = f.name
        
        # Create a simple PDF
        c = canvas.Canvas(temp_file_path, pagesize=letter)
        
        # Add content to PDF
        y = 750
        for line in test_content.strip().split('\n'):
            if line.strip():
                c.drawString(50, y, line.strip())
                y -= 20
                if y < 50:  # Start new page if needed
                    c.showPage()
                    y = 750
        
        c.save()
        print(f"✅ Test PDF created: {temp_file_path}")
        
    except ImportError:
        # Fallback: create a CSV file instead (supported format)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Category,Description,Date,Status\n")
            f.write("Revenue Recognition,ASC 606 Implementation,2024-01-01,Active\n")
            f.write("Lease Accounting,ASC 842 Compliance,2024-01-01,Active\n") 
            f.write("Financial Instruments,ASC 326 Updates,2024-01-01,Pending\n")
            temp_file_path = f.name
        print(f"✅ Test CSV created: {temp_file_path}")
    
    print(f"✅ Test file created: {temp_file_path}")
    
    # Create a mock UploadFile-like object
    class MockUploadFile:
        def __init__(self, file_path):
            # Determine filename and content type based on extension
            path_obj = Path(file_path)
            self.filename = f"test_accounting_policy{path_obj.suffix}"
            
            if path_obj.suffix == '.pdf':
                self.content_type = "application/pdf" 
            elif path_obj.suffix == '.csv':
                self.content_type = "text/csv"
            else:
                self.content_type = "text/plain"
                
            self.file_path = file_path
            self.file = open(file_path, 'rb')  # Keep file handle open
            
        async def read(self):
            with open(self.file_path, 'rb') as f:
                return f.read()
        
        def close(self):
            if hasattr(self, 'file') and self.file:
                self.file.close()
    
    mock_file = MockUploadFile(temp_file_path)
    
    # Test metadata
    metadata = '{"title":"Test Accounting Policy","source_type":"COMPANY","effective_date":"2024-01-01","tags":["accounting","policy","revenue"]}'
    
    print("\n3. Testing document upload...")
    try:
        job_id = await document_processor.upload_document(
            file=mock_file,
            metadata=metadata,
            user_id="test-user"
        )
        print(f"✅ Document upload initiated with job_id: {job_id}")
        
    except Exception as e:
        print(f"❌ Document upload failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Wait for processing to complete
    print("\n4. Waiting for processing to complete...")
    max_wait = 30  # seconds
    wait_time = 0
    
    while wait_time < max_wait:
        try:
            status = await document_processor.get_status(job_id, "test-user")
            print(f"   Status: {status['status']} - Progress: {status['progress']}%")
            
            if status['status'] in ['completed', 'failed']:
                break
                
        except Exception as e:
            print(f"   Error checking status: {e}")
            break
            
        await asyncio.sleep(2)
        wait_time += 2
    
    # Get final status
    print("\n5. Final processing results...")
    try:
        final_status = await document_processor.get_status(job_id, "test-user")
        
        print(f"Status: {final_status['status']}")
        print(f"Progress: {final_status['progress']}%")
        print(f"Chunks created: {final_status.get('chunks_created', 'N/A')}")
        
        if final_status.get('error'):
            print(f"❌ Error: {final_status['error']}")
        
        if final_status.get('summary'):
            summary = final_status['summary']
            print(f"✅ Summary generated: {summary.get('analysis_type', 'unknown')}")
            print(f"   Confidence: {summary.get('confidence', 0)}")
            
    except Exception as e:
        print(f"❌ Error getting final status: {e}")
    
    # Test vector store status
    print("\n6. Checking vector store...")
    try:
        if hasattr(rag_pipeline, 'get_stats'):
            stats = await rag_pipeline.get_stats()
            print(f"✅ Vector store stats: {stats}")
        else:
            print("❌ get_stats method not available")
            
    except Exception as e:
        print(f"❌ Error getting vector store stats: {e}")
    
    # Cleanup
    try:
        os.unlink(temp_file_path)
        print(f"\n✅ Cleanup completed")
    except:
        pass
    
    print("\n" + "=" * 50)
    print("🏁 Test completed")


if __name__ == "__main__":
    asyncio.run(test_document_processing())