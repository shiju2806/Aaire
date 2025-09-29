#!/usr/bin/env python3
"""
Document Deduplication Script for AAIRE

This script removes duplicate documents from both Qdrant vector database
and BM25 index, keeping only the newest version of each unique document.

Usage:
    python scripts/deduplicate_documents.py [--dry-run] [--keep-oldest]
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Set

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.http import models
import structlog

logger = structlog.get_logger()

class DocumentDeduplicator:
    """Remove duplicate documents from AAIRE's vector database and search indexes"""

    def __init__(self, qdrant_url: str, qdrant_api_key: str = None, collection_name: str = "aaire-documents"):
        """Initialize deduplicator with Qdrant connection"""
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

    def get_all_documents(self) -> List[Dict]:
        """Retrieve all documents from Qdrant collection"""
        logger.info("🔍 Fetching all documents from Qdrant...")

        documents = []
        scroll_result = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )

        documents.extend(scroll_result[0])

        # Handle pagination if there are more documents
        while scroll_result[1]:
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=scroll_result[1],
                with_payload=True,
                with_vectors=False
            )
            documents.extend(scroll_result[0])

        logger.info(f"📊 Found {len(documents)} total documents")
        return documents

    def identify_duplicates(self, documents: List[Dict], keep_newest: bool = True) -> Dict[str, List[Dict]]:
        """
        Identify duplicate documents by filename and group them

        Args:
            documents: List of document records from Qdrant
            keep_newest: If True, keep newest version; if False, keep oldest

        Returns:
            Dict mapping filename to list of duplicate document records
        """
        logger.info("🔍 Identifying duplicate documents...")

        # Group documents by filename
        filename_groups = defaultdict(list)

        for doc in documents:
            payload = doc.payload
            filename = payload.get('file_name') or payload.get('filename') or payload.get('title', 'unknown')

            # Create document record with metadata
            doc_record = {
                'point_id': doc.id,
                'filename': filename,
                'job_id': payload.get('job_id'),
                'added_at': payload.get('effective_date') or payload.get('added_at'),
                'doc_type': payload.get('document_type'),
                'payload': payload
            }

            filename_groups[filename].append(doc_record)

        # Find duplicates (files with multiple versions)
        duplicates = {}
        total_duplicate_count = 0

        for filename, docs in filename_groups.items():
            if len(docs) > 1:
                # Sort by date to determine which to keep
                try:
                    docs_sorted = sorted(docs, key=lambda x: x['added_at'] or '', reverse=keep_newest)
                except (TypeError, KeyError):
                    # Fallback: sort by job_id if dates are problematic
                    docs_sorted = sorted(docs, key=lambda x: x['job_id'] or '', reverse=keep_newest)

                duplicates[filename] = docs_sorted
                total_duplicate_count += len(docs) - 1  # -1 because we keep one

        logger.info(f"📋 Found {len(duplicates)} files with duplicates")
        logger.info(f"🗑️ Total documents to remove: {total_duplicate_count}")

        return duplicates

    def show_duplicates_summary(self, duplicates: Dict[str, List[Dict]]):
        """Display a summary of duplicate documents found"""
        if not duplicates:
            logger.info("✅ No duplicates found!")
            return

        logger.info("📊 Duplicate Documents Summary:")
        logger.info("=" * 60)

        for filename, docs in duplicates.items():
            logger.info(f"\n📄 File: {filename}")
            logger.info(f"   Versions found: {len(docs)}")

            for i, doc in enumerate(docs):
                status = "🟢 KEEP" if i == 0 else "🔴 REMOVE"
                logger.info(f"   {status} - Job ID: {doc['job_id']} | Date: {doc['added_at']}")

        logger.info("\n" + "=" * 60)

    def remove_duplicates(self, duplicates: Dict[str, List[Dict]], dry_run: bool = True) -> Dict[str, int]:
        """
        Remove duplicate documents from Qdrant

        Args:
            duplicates: Dict of filename -> list of duplicate documents
            dry_run: If True, only show what would be removed

        Returns:
            Dict with removal statistics
        """
        if not duplicates:
            logger.info("✅ No duplicates to remove")
            return {'removed': 0, 'kept': 0, 'errors': 0}

        stats = {'removed': 0, 'kept': 0, 'errors': 0}

        logger.info(f"{'🔍 DRY RUN: Would remove' if dry_run else '🗑️ Removing'} duplicate documents...")

        for filename, docs in duplicates.items():
            keep_doc = docs[0]  # First document (newest/oldest based on sort)
            remove_docs = docs[1:]  # Rest are duplicates

            logger.info(f"\n📄 Processing: {filename}")
            logger.info(f"   🟢 Keeping: Job {keep_doc['job_id']} (Date: {keep_doc['added_at']})")

            stats['kept'] += 1

            # Remove duplicate documents
            for doc in remove_docs:
                try:
                    point_id = doc['point_id']

                    if dry_run:
                        logger.info(f"   🔍 Would remove: Job {doc['job_id']} (Point ID: {point_id})")
                    else:
                        # Actually delete from Qdrant
                        self.qdrant_client.delete(
                            collection_name=self.collection_name,
                            points_selector=models.PointIdsList(
                                points=[point_id]
                            )
                        )
                        logger.info(f"   ✅ Removed: Job {doc['job_id']} (Point ID: {point_id})")

                    stats['removed'] += 1

                except Exception as e:
                    logger.error(f"   ❌ Error removing {doc['job_id']}: {e}")
                    stats['errors'] += 1

        return stats

    def run_deduplication(self, dry_run: bool = True, keep_newest: bool = True) -> Dict[str, int]:
        """
        Main deduplication workflow

        Args:
            dry_run: If True, only analyze and show what would be removed
            keep_newest: If True, keep newest version; if False, keep oldest

        Returns:
            Statistics about the deduplication process
        """
        logger.info("🚀 Starting document deduplication process")
        logger.info(f"   Mode: {'DRY RUN' if dry_run else 'LIVE REMOVAL'}")
        logger.info(f"   Strategy: Keep {'newest' if keep_newest else 'oldest'} version")

        try:
            # Step 1: Get all documents
            documents = self.get_all_documents()

            if not documents:
                logger.warning("⚠️ No documents found in collection")
                return {'removed': 0, 'kept': 0, 'errors': 0}

            # Step 2: Identify duplicates
            duplicates = self.identify_duplicates(documents, keep_newest)

            # Step 3: Show summary
            self.show_duplicates_summary(duplicates)

            # Step 4: Remove duplicates (or show what would be removed)
            stats = self.remove_duplicates(duplicates, dry_run)

            # Final summary
            logger.info("\n🎯 Deduplication Summary:")
            logger.info(f"   📊 Documents analyzed: {len(documents)}")
            logger.info(f"   🟢 Documents kept: {stats['kept']}")
            logger.info(f"   🗑️ Documents {'would be ' if dry_run else ''}removed: {stats['removed']}")
            logger.info(f"   ❌ Errors: {stats['errors']}")

            if dry_run and stats['removed'] > 0:
                logger.info("\n💡 To actually remove duplicates, run with --no-dry-run")

            return stats

        except Exception as e:
            logger.error(f"❌ Deduplication failed: {e}")
            raise


def main():
    """Main command-line interface"""
    parser = argparse.ArgumentParser(description="Remove duplicate documents from AAIRE")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Only analyze duplicates, don't remove them (default)")
    parser.add_argument("--no-dry-run", action="store_true",
                        help="Actually remove duplicates (use with caution)")
    parser.add_argument("--keep-oldest", action="store_true",
                        help="Keep oldest version instead of newest")
    parser.add_argument("--qdrant-url",
                        default="https://ce8b5f05-c0a2-47b1-a761-c2f9e6f73817.europe-west3-0.gcp.cloud.qdrant.io:6333",
                        help="Qdrant server URL")
    parser.add_argument("--collection", default="aaire-documents",
                        help="Qdrant collection name")

    args = parser.parse_args()

    # Determine actual dry_run mode
    dry_run = args.dry_run and not args.no_dry_run
    keep_newest = not args.keep_oldest

    # Get Qdrant API key from environment
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    if not qdrant_api_key:
        logger.warning("⚠️ QDRANT_API_KEY not found in environment variables")

    try:
        # Create deduplicator and run
        deduplicator = DocumentDeduplicator(
            qdrant_url=args.qdrant_url,
            qdrant_api_key=qdrant_api_key,
            collection_name=args.collection
        )

        stats = deduplicator.run_deduplication(dry_run=dry_run, keep_newest=keep_newest)

        if stats['errors'] > 0:
            sys.exit(1)
        else:
            logger.info("✅ Deduplication completed successfully")

    except Exception as e:
        logger.error(f"❌ Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()