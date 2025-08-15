#!/usr/bin/env python3
"""
Test script for the intelligent extraction system
"""

import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

async def test_extraction():
    """Test the intelligent extraction components"""
    
    # Test document with job titles
    test_document = """
    Finance Team Structure Report
    
    The finance team consists of the following members:
    
    John Smith - Chief Financial Officer (CFO)
    Sarah Johnson - Treasurer  
    Mike Wilson - Financial Controller
    Lisa Chen - Senior Financial Analyst
    David Brown - Financial Analyst
    Emily Davis - Junior Accountant
    Robert Taylor - Staff Accountant
    
    The CFO oversees all financial operations and reports to the CEO.
    The Treasurer manages cash flow and banking relationships.
    The Controller ensures accurate financial reporting and compliance.
    """
    
    try:
        from enhanced_query_handler import EnhancedQueryHandler
        from intelligent_extractor import IntelligentDocumentExtractor
        
        # Mock LLM client
        class MockLLM:
            def complete(self, prompt):
                class MockResponse:
                    def __init__(self, text):
                        self.text = text
                
                # Simple mock response for testing
                return MockResponse('''
                {
                    "financial_roles": [
                        {"name": "John Smith", "title": "Chief Financial Officer", "department": "Finance", "confidence": 0.9},
                        {"name": "Sarah Johnson", "title": "Treasurer", "department": "Finance", "confidence": 0.9},
                        {"name": "Mike Wilson", "title": "Financial Controller", "department": "Finance", "confidence": 0.9},
                        {"name": "Lisa Chen", "title": "Senior Financial Analyst", "department": "Finance", "confidence": 0.8},
                        {"name": "David Brown", "title": "Financial Analyst", "department": "Finance", "confidence": 0.8}
                    ],
                    "title_categories": {
                        "leadership": ["Chief Financial Officer"],
                        "management": ["Treasurer", "Financial Controller"],
                        "analyst": ["Senior Financial Analyst", "Financial Analyst"],
                        "other": ["Junior Accountant", "Staff Accountant"]
                    },
                    "extracted_titles": ["Chief Financial Officer", "Treasurer", "Financial Controller", "Senior Financial Analyst", "Financial Analyst", "Junior Accountant", "Staff Accountant"],
                    "warnings": []
                }
                ''')
        
        mock_llm = MockLLM()
        
        # Test query handler
        print("🔍 Testing Enhanced Query Handler...")
        query_handler = EnhancedQueryHandler(mock_llm)
        
        test_queries = [
            "Who are the finance team members?",
            "List all job titles in the finance department",
            "What is the organizational structure?",
            "How many employees work in finance?"
        ]
        
        for query in test_queries:
            analysis = query_handler.analyze_query(query)
            print(f"Query: '{query}'")
            print(f"  - Needs extraction: {analysis.needs_intelligent_extraction}")
            print(f"  - Extraction type: {analysis.extraction_type}")
            print(f"  - Confidence: {analysis.confidence:.3f}")
            print(f"  - Reasoning: {analysis.reasoning}")
            print()
        
        # Test intelligent extractor
        print("🧠 Testing Intelligent Document Extractor...")
        extractor = IntelligentDocumentExtractor(mock_llm)
        
        result = await extractor.process_document(test_document, "List all job titles")
        
        print(f"Document type detected: {result.document_type}")
        print(f"Extraction confidence: {result.confidence_score:.3f}")
        print(f"Extraction method: {result.extraction_method}")
        print(f"Extracted data: {result.extracted_data}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        
        print("\n✅ Intelligent extraction system test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the intelligent extraction files are in the src directory")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_extraction())