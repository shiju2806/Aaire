"""
Document Deduplication Service
Handles content-based deduplication to prevent duplicate document processing
"""

import hashlib
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import structlog

logger = structlog.get_logger()

class DocumentDeduplicator:
    """
    Service to handle document deduplication based on content hash
    """

    def __init__(self, storage_path: str = "data/document_hashes.json"):
        """
        Initialize document deduplicator

        Args:
            storage_path: Path to store document hash mappings
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.hash_db = self._load_hash_database()

    def _load_hash_database(self) -> Dict[str, Any]:
        """Load existing hash database"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load hash database: {e}")

        return {}

    def _save_hash_database(self):
        """Save hash database to disk"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.hash_db, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save hash database: {e}")

    def calculate_content_hash(self, file_content: bytes) -> str:
        """
        Calculate SHA-256 hash of file content

        Args:
            file_content: Raw file content bytes

        Returns:
            Hex string of content hash
        """
        return hashlib.sha256(file_content).hexdigest()

    def is_duplicate(self, content_hash: str) -> bool:
        """
        Check if document with given hash already exists

        Args:
            content_hash: SHA-256 hash of document content

        Returns:
            True if document already exists
        """
        return content_hash in self.hash_db

    def get_existing_document(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get existing document info by content hash

        Args:
            content_hash: SHA-256 hash of document content

        Returns:
            Document info dict or None if not found
        """
        return self.hash_db.get(content_hash)

    def register_document(self, content_hash: str, job_id: str, filename: str,
                         file_size: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new document in the hash database

        Args:
            content_hash: SHA-256 hash of document content
            job_id: Unique job ID for this document
            filename: Original filename
            file_size: File size in bytes
            metadata: Document metadata

        Returns:
            Document info dict
        """
        doc_info = {
            "job_id": job_id,
            "filename": filename,
            "file_size": file_size,
            "metadata": metadata,
            "first_uploaded": datetime.utcnow().isoformat(),
            "upload_count": 1,
            "last_uploaded": datetime.utcnow().isoformat()
        }

        self.hash_db[content_hash] = doc_info
        self._save_hash_database()

        logger.info(f"Document registered with hash: {content_hash[:8]}...",
                   job_id=job_id, filename=filename)

        return doc_info

    def update_duplicate_upload(self, content_hash: str, new_job_id: str,
                               new_filename: str) -> Dict[str, Any]:
        """
        Update existing document entry when duplicate is uploaded

        Args:
            content_hash: SHA-256 hash of document content
            new_job_id: New job ID for duplicate upload
            new_filename: New filename (may be different)

        Returns:
            Updated document info dict
        """
        doc_info = self.hash_db[content_hash]
        doc_info["upload_count"] += 1
        doc_info["last_uploaded"] = datetime.utcnow().isoformat()
        doc_info["duplicate_job_ids"] = doc_info.get("duplicate_job_ids", [])
        doc_info["duplicate_job_ids"].append({
            "job_id": new_job_id,
            "filename": new_filename,
            "uploaded_at": datetime.utcnow().isoformat()
        })

        self._save_hash_database()

        logger.info(f"Duplicate upload recorded for hash: {content_hash[:8]}...",
                   original_job_id=doc_info["job_id"],
                   duplicate_job_id=new_job_id,
                   upload_count=doc_info["upload_count"])

        return doc_info

    def get_duplicate_stats(self) -> Dict[str, Any]:
        """
        Get statistics about duplicate uploads

        Returns:
            Statistics dict
        """
        total_hashes = len(self.hash_db)
        duplicates = sum(1 for doc in self.hash_db.values() if doc["upload_count"] > 1)
        total_uploads = sum(doc["upload_count"] for doc in self.hash_db.values())

        duplicate_details = []
        for content_hash, doc_info in self.hash_db.items():
            if doc_info["upload_count"] > 1:
                duplicate_details.append({
                    "content_hash": content_hash[:8] + "...",
                    "original_job_id": doc_info["job_id"],
                    "filename": doc_info["filename"],
                    "upload_count": doc_info["upload_count"],
                    "file_size_mb": round(doc_info["file_size"] / (1024 * 1024), 1),
                    "first_uploaded": doc_info["first_uploaded"],
                    "last_uploaded": doc_info["last_uploaded"]
                })

        return {
            "total_unique_documents": total_hashes,
            "documents_with_duplicates": duplicates,
            "total_uploads": total_uploads,
            "potential_savings": total_uploads - total_hashes,
            "duplicate_details": duplicate_details,
            "generated_at": datetime.utcnow().isoformat()
        }

    def cleanup_orphaned_entries(self, valid_job_ids: List[str]) -> Dict[str, Any]:
        """
        Remove entries from hash database for documents that no longer exist

        Args:
            valid_job_ids: List of job IDs that still exist in the system

        Returns:
            Cleanup results
        """
        orphaned = []
        cleaned_hashes = []

        for content_hash, doc_info in list(self.hash_db.items()):
            if doc_info["job_id"] not in valid_job_ids:
                orphaned.append({
                    "content_hash": content_hash[:8] + "...",
                    "job_id": doc_info["job_id"],
                    "filename": doc_info["filename"]
                })
                del self.hash_db[content_hash]
                cleaned_hashes.append(content_hash)

        if cleaned_hashes:
            self._save_hash_database()
            logger.info(f"Cleaned {len(cleaned_hashes)} orphaned entries from hash database")

        return {
            "orphaned_entries_removed": len(orphaned),
            "orphaned_details": orphaned,
            "cleaned_at": datetime.utcnow().isoformat()
        }

    def rebuild_from_existing_files(self, uploads_dir: Path) -> Dict[str, Any]:
        """
        Rebuild hash database by scanning existing uploaded files

        Args:
            uploads_dir: Directory containing uploaded files

        Returns:
            Rebuild results
        """
        if not uploads_dir.exists():
            return {"error": "Uploads directory does not exist"}

        rebuilt_count = 0
        errors = []

        # Clear existing database
        self.hash_db = {}

        for file_path in uploads_dir.iterdir():
            if file_path.is_file():
                try:
                    # Handle different filename formats:
                    # 1. {job_id}_{original_filename} (new format)
                    # 2. {job_id}.{ext} (current format)

                    filename_parts = file_path.name.split('_', 1)
                    if len(filename_parts) == 2:
                        # Format: {job_id}_{original_filename}
                        job_id = filename_parts[0]
                        original_filename = filename_parts[1]
                    else:
                        # Format: {job_id}.{ext} - extract job_id and use generic name
                        filename_without_ext = file_path.stem  # removes .pdf/.txt etc
                        job_id = filename_without_ext
                        original_filename = f"document{file_path.suffix}"

                    # Calculate content hash
                    with open(file_path, 'rb') as f:
                        content = f.read()

                    content_hash = self.calculate_content_hash(content)
                    file_size = len(content)

                    # Register or update
                    if content_hash in self.hash_db:
                        self.update_duplicate_upload(content_hash, job_id, original_filename)
                    else:
                        self.register_document(
                            content_hash,
                            job_id,
                            original_filename,
                            file_size,
                            {"reconstructed": True}
                        )

                    rebuilt_count += 1

                except Exception as e:
                    errors.append(f"Error processing {file_path.name}: {str(e)}")

        logger.info(f"Rebuilt hash database: {rebuilt_count} files processed, {len(errors)} errors")

        return {
            "files_processed": rebuilt_count,
            "unique_documents": len(self.hash_db),
            "errors": errors,
            "rebuilt_at": datetime.utcnow().isoformat()
        }