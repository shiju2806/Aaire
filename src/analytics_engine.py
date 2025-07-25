"""
Usage Analytics Engine for AAIRE
Tracks user interactions, questions, and knowledge gaps
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import structlog
from pathlib import Path

logger = structlog.get_logger()

class AnalyticsEngine:
    def __init__(self, data_dir: str = "data/analytics"):
        """Initialize analytics engine"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Analytics files
        self.queries_file = self.data_dir / "queries.jsonl"
        self.sessions_file = self.data_dir / "sessions.jsonl"
        self.workflows_file = self.data_dir / "workflows.jsonl"
        
        # In-memory caches for performance
        self.query_cache = []
        self.session_cache = {}
        
        logger.info("Analytics engine initialized", data_dir=str(self.data_dir))
    
    async def track_query(
        self, 
        query: str, 
        response: str, 
        session_id: str,
        user_id: str = "demo-user",
        confidence: float = 0.0,
        sources: List[str] = None,
        processing_time_ms: int = 0
    ):
        """Track a user query and response"""
        try:
            query_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": session_id,
                "user_id": user_id,
                "query": query,
                "query_length": len(query),
                "query_words": len(query.split()),
                "response_length": len(response),
                "confidence": confidence,
                "sources": sources or [],
                "processing_time_ms": processing_time_ms,
                "query_type": self._classify_query(query),
                "complexity": self._assess_complexity(query),
                "accounting_topics": self._extract_topics(query)
            }
            
            # Append to file
            await self._append_to_file(self.queries_file, query_data)
            
            # Update session
            await self._update_session(session_id, user_id, "query")
            
            logger.info("Query tracked", 
                       session_id=session_id, 
                       query_type=query_data["query_type"],
                       confidence=confidence)
            
        except Exception as e:
            logger.error("Failed to track query", error=str(e))
    
    async def track_document_upload(
        self,
        filename: str,
        file_type: str,
        file_size: int,
        session_id: str,
        user_id: str = "demo-user",
        processing_success: bool = True
    ):
        """Track document upload events"""
        try:
            upload_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "event": "document_upload",
                "session_id": session_id,
                "user_id": user_id,
                "filename": filename,
                "file_type": file_type,
                "file_size": file_size,
                "success": processing_success
            }
            
            await self._append_to_file(self.queries_file, upload_data)
            await self._update_session(session_id, user_id, "upload")
            
        except Exception as e:
            logger.error("Failed to track document upload", error=str(e))
    
    async def track_workflow_step(
        self,
        workflow_id: str,
        step_id: str,
        step_name: str,
        session_id: str,
        user_id: str = "demo-user",
        completed: bool = False,
        skipped: bool = False
    ):
        """Track workflow progression"""
        try:
            workflow_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "workflow_id": workflow_id,
                "step_id": step_id,
                "step_name": step_name,
                "session_id": session_id,
                "user_id": user_id,
                "completed": completed,
                "skipped": skipped
            }
            
            await self._append_to_file(self.workflows_file, workflow_data)
            
        except Exception as e:
            logger.error("Failed to track workflow step", error=str(e))
    
    async def get_analytics_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Load recent data
            queries = await self._load_recent_data(self.queries_file, cutoff_date)
            workflows = await self._load_recent_data(self.workflows_file, cutoff_date)
            
            # Calculate metrics
            total_queries = len([q for q in queries if q.get('query')])
            total_uploads = len([q for q in queries if q.get('event') == 'document_upload'])
            
            # Query analysis
            query_types = Counter([q.get('query_type') for q in queries if q.get('query_type')])
            avg_confidence = sum([q.get('confidence', 0) for q in queries if q.get('confidence')]) / max(len(queries), 1)
            
            # Topic analysis
            all_topics = []
            for q in queries:
                topics = q.get('accounting_topics', [])
                all_topics.extend(topics)
            top_topics = Counter(all_topics).most_common(10)
            
            # Session analysis
            sessions = set([q.get('session_id') for q in queries if q.get('session_id')])
            
            # Knowledge gaps (low confidence queries)
            knowledge_gaps = [
                {
                    "query": q.get('query', ''),
                    "confidence": q.get('confidence', 0),
                    "topics": q.get('accounting_topics', [])
                }
                for q in queries 
                if q.get('confidence', 0) < 0.5 and q.get('query')
            ]
            
            # Workflow analysis
            workflow_completion = defaultdict(list)
            for w in workflows:
                workflow_completion[w.get('workflow_id', '')].append(w)
            
            return {
                "period_days": days,
                "generated_at": datetime.utcnow().isoformat(),
                "overview": {
                    "total_queries": total_queries,
                    "total_uploads": total_uploads,
                    "unique_sessions": len(sessions),
                    "avg_confidence": round(avg_confidence, 3)
                },
                "query_analysis": {
                    "by_type": dict(query_types),
                    "top_topics": top_topics,
                    "knowledge_gaps_count": len(knowledge_gaps)
                },
                "knowledge_gaps": knowledge_gaps[:10],  # Top 10 gaps
                "workflow_usage": {
                    "active_workflows": len(workflow_completion),
                    "total_steps": len(workflows)
                },
                "trends": await self._calculate_trends(queries, days)
            }
            
        except Exception as e:
            logger.error("Failed to generate analytics summary", error=str(e))
            return {"error": str(e)}
    
    def _classify_query(self, query: str) -> str:
        """Classify query into categories"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how to', 'how do i', 'steps', 'process']):
            return 'how_to'
        elif any(word in query_lower for word in ['what is', 'define', 'definition', 'meaning']):
            return 'definition'
        elif any(word in query_lower for word in ['calculate', 'computation', 'formula']):
            return 'calculation'
        elif any(word in query_lower for word in ['compliance', 'requirement', 'regulation']):
            return 'compliance'
        elif any(word in query_lower for word in ['example', 'sample', 'template']):
            return 'example'
        else:
            return 'general'
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity"""
        word_count = len(query.split())
        
        if word_count < 5:
            return 'simple'
        elif word_count < 15:
            return 'medium'
        else:
            return 'complex'
    
    def _extract_topics(self, query: str) -> List[str]:
        """Extract accounting topics from query"""
        topics = []
        query_lower = query.lower()
        
        # Accounting standards
        if any(std in query_lower for std in ['asc', 'fasb', 'gaap']):
            topics.append('US_GAAP')
        if any(std in query_lower for std in ['ifrs', 'ias']):
            topics.append('IFRS')
        
        # Topic areas
        topic_keywords = {
            'revenue_recognition': ['revenue', 'asc 606', 'contract'],
            'leases': ['lease', 'asc 842', 'rental'],
            'financial_instruments': ['derivative', 'investment', 'fair value'],
            'insurance': ['insurance', 'premium', 'claim', 'policyholder'],
            'reserves': ['reserve', 'provision', 'allowance'],
            'consolidation': ['consolidation', 'subsidiary', 'intercompany'],
            'cash_flow': ['cash flow', 'statement of cash flows'],
            'depreciation': ['depreciation', 'amortization', 'asset']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    async def _append_to_file(self, file_path: Path, data: Dict[str, Any]):
        """Append data to JSONL file"""
        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            logger.error("Failed to append to file", file=str(file_path), error=str(e))
    
    async def _update_session(self, session_id: str, user_id: str, event_type: str):
        """Update session information"""
        try:
            session_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": session_id,
                "user_id": user_id,
                "event_type": event_type,
                "last_activity": datetime.utcnow().isoformat()
            }
            
            await self._append_to_file(self.sessions_file, session_data)
            
        except Exception as e:
            logger.error("Failed to update session", error=str(e))
    
    async def _load_recent_data(self, file_path: Path, cutoff_date: datetime) -> List[Dict[str, Any]]:
        """Load data from JSONL file after cutoff date"""
        data = []
        
        if not file_path.exists():
            return data
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            record_date = datetime.fromisoformat(record.get('timestamp', '1970-01-01'))
                            if record_date >= cutoff_date:
                                data.append(record)
                        except (json.JSONDecodeError, ValueError):
                            continue
        except Exception as e:
            logger.error("Failed to load data", file=str(file_path), error=str(e))
        
        return data
    
    async def _calculate_trends(self, queries: List[Dict[str, Any]], days: int) -> Dict[str, Any]:
        """Calculate usage trends"""
        try:
            # Group by day
            daily_counts = defaultdict(int)
            for query in queries:
                if query.get('timestamp'):
                    date = query['timestamp'][:10]  # YYYY-MM-DD
                    daily_counts[date] += 1
            
            # Calculate average
            avg_daily = sum(daily_counts.values()) / max(days, 1)
            
            return {
                "avg_queries_per_day": round(avg_daily, 2),
                "peak_day": max(daily_counts.items(), key=lambda x: x[1]) if daily_counts else None,
                "daily_breakdown": dict(daily_counts)
            }
            
        except Exception as e:
            logger.error("Failed to calculate trends", error=str(e))
            return {}