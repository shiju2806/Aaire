"""
Audit Logger for AAIRE - MVP-FR-024
Comprehensive audit trail for compliance and security
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import structlog

from config.compliance import AUDIT_EVENTS, LogLevel

logger = structlog.get_logger()

class AuditEventType(Enum):
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    QUERY_SUBMITTED = "query_submitted"
    DOCUMENT_UPLOADED = "document_uploaded"
    COMPLIANCE_TRIGGERED = "compliance_triggered"
    ADMIN_ACTION = "admin_action"
    ERROR_OCCURRED = "error_occurred"
    SYSTEM_EVENT = "system_event"

@dataclass
class AuditEvent:
    event_id: str
    event_type: str
    user_id: str
    timestamp: str
    data: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    risk_level: str = "low"

class AuditLogger:
    def __init__(self):
        """Initialize audit logger without starting async tasks"""
        
        # In-memory storage for MVP (replace with persistent storage in production)
        self.audit_events = []
        
        # Configuration from compliance module
        self.event_config = AUDIT_EVENTS
        
        # Hot storage limit (90 days as specified)
        self.hot_storage_days = 90
        
        # Background cleanup task (will be started lazily)
        self._cleanup_task = None
        self._cleanup_started = False
        
        logger.info("Audit logger initialized")
    
    def _ensure_cleanup_task(self):
        """Start cleanup task if not already started (lazy initialization)"""
        if not self._cleanup_started and not self._cleanup_task:
            try:
                loop = asyncio.get_running_loop()
                self._cleanup_task = loop.create_task(self._cleanup_loop())
                self._cleanup_started = True
                logger.info("Audit cleanup task started")
            except RuntimeError:
                # No event loop running yet, will try again later
                pass
    
    async def _cleanup_loop(self):
        """Background task for log cleanup"""
        while True:
            try:
                await self._cleanup_expired_logs()
                # Run cleanup every 24 hours
                await asyncio.sleep(24 * 3600)
            except asyncio.CancelledError:
                logger.info("Audit cleanup task cancelled")
                break
            except Exception as e:
                logger.error("Audit log cleanup failed", error=str(e))
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    async def log_event(
        self,
        event: str,
        user_id: str,
        data: Dict[str, Any],
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        risk_level: str = "low"
    ):
        """
        Log an audit event - MVP-FR-024
        """
        
        # Ensure cleanup task is running
        self._ensure_cleanup_task()
        
        # Generate unique event ID
        import uuid
        event_id = str(uuid.uuid4())
        
        # Create audit event
        audit_event = AuditEvent(
            event_id=event_id,
            event_type=event,
            user_id=user_id,
            timestamp=datetime.utcnow().isoformat(),
            data=data,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            risk_level=risk_level
        )
        
        # Store event
        self.audit_events.append(audit_event)
        
        # Get log level for this event type
        event_config = self.event_config.get(event, {})
        log_level = event_config.get("log_level", LogLevel.INFO)
        
        # Structure log data
        log_data = {
            "audit_event_id": event_id,
            "event_type": event,
            "user_id": user_id,
            "data": data,
            "ip_address": ip_address,
            "risk_level": risk_level
        }
        
        # Log to structured logger
        if log_level == LogLevel.ERROR:
            logger.error("Audit event", **log_data)
        elif log_level == LogLevel.WARNING:
            logger.warning("Audit event", **log_data)
        else:
            logger.info("Audit event", **log_data)
        
        # Check for high-risk events
        if risk_level in ["high", "critical"]:
            await self._handle_high_risk_event(audit_event)
    
    async def _handle_high_risk_event(self, event: AuditEvent):
        """Handle high-risk security events"""
        
        logger.warning("High-risk audit event detected", 
                      event_id=event.event_id,
                      event_type=event.event_type,
                      user_id=event.user_id,
                      risk_level=event.risk_level)
        
        # In production, this would:
        # - Send alerts to security team
        # - Integrate with SIEM systems
        # - Trigger automated responses
        # - Create security incidents
    
    async def get_user_events(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit events for a specific user"""
        
        filtered_events = []
        
        for event in self.audit_events:
            # Filter by user
            if event.user_id != user_id:
                continue
            
            # Filter by date range
            event_time = datetime.fromisoformat(event.timestamp)
            if start_date and event_time < start_date:
                continue
            if end_date and event_time > end_date:
                continue
            
            # Filter by event types
            if event_types and event.event_type not in event_types:
                continue
            
            filtered_events.append(asdict(event))
            
            if len(filtered_events) >= limit:
                break
        
        # Sort by timestamp (newest first)
        filtered_events.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return filtered_events
    
    async def get_system_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        risk_levels: Optional[List[str]] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get system-wide audit events (admin only)"""
        
        filtered_events = []
        
        for event in self.audit_events:
            # Filter by date range
            event_time = datetime.fromisoformat(event.timestamp)
            if start_date and event_time < start_date:
                continue
            if end_date and event_time > end_date:
                continue
            
            # Filter by risk levels
            if risk_levels and event.risk_level not in risk_levels:
                continue
            
            filtered_events.append(asdict(event))
            
            if len(filtered_events) >= limit:
                break
        
        # Sort by timestamp (newest first)
        filtered_events.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return filtered_events
    
    async def get_compliance_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get compliance-related events for audit purposes"""
        
        compliance_event_types = [
            AuditEventType.COMPLIANCE_TRIGGERED.value,
            AuditEventType.DOCUMENT_UPLOADED.value,
            AuditEventType.ADMIN_ACTION.value
        ]
        
        return await self.get_system_events(
            start_date=start_date,
            end_date=end_date,
            limit=10000  # No limit for compliance reports
        )
    
    async def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit log statistics"""
        
        # Count events by type
        event_counts = {}
        risk_counts = {}
        user_counts = {}
        
        # Calculate date ranges
        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)
        last_week = now - timedelta(days=7)
        last_month = now - timedelta(days=30)
        
        recent_events = {
            "last_24h": 0,
            "last_week": 0,
            "last_month": 0
        }
        
        for event in self.audit_events:
            # Count by event type
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
            
            # Count by risk level
            risk_counts[event.risk_level] = risk_counts.get(event.risk_level, 0) + 1
            
            # Count by user
            user_counts[event.user_id] = user_counts.get(event.user_id, 0) + 1
            
            # Count recent events
            event_time = datetime.fromisoformat(event.timestamp)
            if event_time >= last_24h:
                recent_events["last_24h"] += 1
            if event_time >= last_week:
                recent_events["last_week"] += 1
            if event_time >= last_month:
                recent_events["last_month"] += 1
        
        return {
            "total_events": len(self.audit_events),
            "event_type_counts": event_counts,
            "risk_level_counts": risk_counts,
            "active_users": len(user_counts),
            "recent_activity": recent_events,
            "retention_policy": {
                "hot_storage_days": self.hot_storage_days,
                "cold_storage_years": 7  # As specified in compliance config
            }
        }
    
    async def export_audit_log(
        self,
        start_date: datetime,
        end_date: datetime,
        format: str = "json"
    ) -> str:
        """Export audit log for compliance reporting"""
        
        events = await self.get_system_events(
            start_date=start_date,
            end_date=end_date,
            limit=None  # No limit for exports
        )
        
        if format == "json":
            return json.dumps(events, indent=2, default=str)
        elif format == "csv":
            # TODO: Implement CSV export
            raise NotImplementedError("CSV export not yet implemented")
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def _cleanup_expired_logs(self):
        """Clean up logs older than retention period"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=self.hot_storage_days)
        
        original_count = len(self.audit_events)
        
        # Filter out expired events (in production, move to cold storage)
        self.audit_events = [
            event for event in self.audit_events
            if datetime.fromisoformat(event.timestamp) >= cutoff_date
        ]
        
        cleaned_count = original_count - len(self.audit_events)
        
        if cleaned_count > 0:
            logger.info("Audit log cleanup completed", 
                       events_cleaned=cleaned_count,
                       events_remaining=len(self.audit_events))
    
    async def search_events(
        self,
        query: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search audit events by query string"""
        
        matching_events = []
        query_lower = query.lower()
        
        for event in self.audit_events:
            # Search in event data
            event_dict = asdict(event)
            event_json = json.dumps(event_dict, default=str).lower()
            
            if query_lower in event_json:
                # Apply date filters
                event_time = datetime.fromisoformat(event.timestamp)
                if start_date and event_time < start_date:
                    continue
                if end_date and event_time > end_date:
                    continue
                
                matching_events.append(event_dict)
                
                if len(matching_events) >= limit:
                    break
        
        return matching_events
    
    def cleanup(self):
        """Manual cleanup method"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            self._cleanup_started = False
    
    def __del__(self):
        """Cleanup on destruction"""
        # Don't try to cancel tasks in __del__ as it can cause issues
        # Use the cleanup() method explicitly when needed
        pass
