#!/usr/bin/env python3
"""
Script to clean up the Qdrant vector database
"""
import os
import sys
import asyncio
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.rag_pipeline import RAGPipeline
import structlog

logger = structlog.get_logger()

async def main():
    """Clean up the vector database"""
    try:
        print("🔧 Initializing RAG Pipeline...")
        rag_pipeline = RAGPipeline()
        print("✅ RAG Pipeline initialized successfully")
        
        print("\n📊 Checking current documents in vector store...")
        documents_result = await rag_pipeline.get_all_documents()
        
        if documents_result["status"] == "success":
            total_docs = documents_result["total_documents"]
            print(f"📈 Total documents found: {total_docs}")
            
            if total_docs > 0:
                print("\n📋 Documents currently in vector store:")
                for i, doc in enumerate(documents_result["documents"][:10], 1):  # Show first 10
                    print(f"{i:2d}. {doc['filename']} (job_id: {doc['job_id']}) - {doc['doc_type']}")
                    if doc['text_preview']:
                        print(f"    Preview: {doc['text_preview']}")
                
                if total_docs > 10:
                    print(f"    ... and {total_docs - 10} more documents")
                
                # Ask user if they want to clean up
                print(f"\n🧹 Found {total_docs} documents in vector store.")
                cleanup_choice = input("Do you want to clean up orphaned chunks (documents without job_id)? [y/N]: ").strip().lower()
                
                if cleanup_choice in ['y', 'yes']:
                    print("\n🧹 Cleaning up orphaned chunks...")
                    cleanup_result = await rag_pipeline.cleanup_orphaned_chunks()
                    
                    if cleanup_result["status"] == "success":
                        cleaned_count = cleanup_result["cleaned_chunks"]
                        print(f"✅ Cleaned up {cleaned_count} orphaned chunks")
                    else:
                        print(f"❌ Cleanup failed: {cleanup_result.get('error', 'Unknown error')}")
                else:
                    print("ℹ️  Skipping cleanup")
                
                # Option to delete specific documents
                delete_choice = input("\nDo you want to delete specific documents? [y/N]: ").strip().lower()
                if delete_choice in ['y', 'yes']:
                    print("\nEnter the filename(s) of documents to delete (or 'all' to delete everything):")
                    files_to_delete = input("Filenames (comma-separated): ").strip()
                    
                    if files_to_delete.lower() == 'all':
                        confirm = input("⚠️  This will delete ALL documents from the vector store. Are you sure? [y/N]: ").strip().lower()
                        if confirm in ['y', 'yes']:
                            # Delete all documents by finding all unique job_ids
                            job_ids = set()
                            for doc in documents_result["documents"]:
                                if doc['job_id'] != 'No job_id':
                                    job_ids.add(doc['job_id'])
                            
                            for job_id in job_ids:
                                delete_result = await rag_pipeline.delete_document(job_id)
                                if delete_result["status"] == "success":
                                    print(f"✅ Deleted document with job_id: {job_id}")
                                else:
                                    print(f"❌ Failed to delete job_id {job_id}: {delete_result.get('error', 'Unknown error')}")
                    else:
                        filenames = [f.strip() for f in files_to_delete.split(',')]
                        for filename in filenames:
                            # Find job_id for this filename
                            job_id = None
                            for doc in documents_result["documents"]:
                                if doc['filename'].lower() == filename.lower():
                                    job_id = doc['job_id']
                                    break
                            
                            if job_id and job_id != 'No job_id':
                                delete_result = await rag_pipeline.delete_document(job_id)
                                if delete_result["status"] == "success":
                                    print(f"✅ Deleted {filename} (job_id: {job_id})")
                                else:
                                    print(f"❌ Failed to delete {filename}: {delete_result.get('error', 'Unknown error')}")
                            else:
                                print(f"❌ Could not find {filename} in vector store")
            else:
                print("✅ Vector store is empty - no cleanup needed")
        else:
            print(f"❌ Failed to check documents: {documents_result.get('error', 'Unknown error')}")
        
        print("\n🎉 Cleanup script completed!")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())