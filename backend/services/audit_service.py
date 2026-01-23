"""
Audit Service
Tenant-aware audit log operations
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_

from backend.database.models import AuditLog, User


class AuditService:
    """
    Service layer for audit log operations.
    All methods enforce tenant isolation.
    """

    def __init__(self, db: Session, tenant_id: uuid.UUID, user_id: Optional[uuid.UUID] = None):
        """
        Initialize service with database session and tenant context.

        Args:
            db: SQLAlchemy session
            tenant_id: Current tenant's UUID
            user_id: Current user's UUID
        """
        self.db = db
        self.tenant_id = tenant_id
        self.user_id = user_id

    def _base_query(self):
        """Base query with tenant filter"""
        return self.db.query(AuditLog).filter(
            AuditLog.tenant_id == self.tenant_id
        )

    def log_action(
        self,
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> AuditLog:
        """
        Create an audit log entry.

        Args:
            action: Action identifier (e.g., 'user.login', 'document.create')
            resource_type: Type of resource affected
            resource_id: ID of resource affected
            details: Additional context
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Created AuditLog instance
        """
        log = AuditLog(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent
        )

        self.db.add(log)
        self.db.flush()

        return log

    def list_logs(
        self,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        user_id: Optional[uuid.UUID] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        page: int = 1,
        per_page: int = 50
    ) -> Tuple[List[AuditLog], int]:
        """
        List audit logs with filtering and pagination.

        Args:
            action: Filter by action
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            user_id: Filter by user
            start_date: Filter from date
            end_date: Filter to date
            page: Page number
            per_page: Items per page

        Returns:
            Tuple of (logs list, total count)
        """
        per_page = min(max(per_page, 1), 100)
        page = max(page, 1)

        query = self._base_query()

        if action:
            query = query.filter(AuditLog.action == action)
        if resource_type:
            query = query.filter(AuditLog.resource_type == resource_type)
        if resource_id:
            query = query.filter(AuditLog.resource_id == resource_id)
        if user_id:
            query = query.filter(AuditLog.user_id == user_id)
        if start_date:
            query = query.filter(AuditLog.created_at >= start_date)
        if end_date:
            query = query.filter(AuditLog.created_at <= end_date)

        total = query.count()

        logs = query.order_by(desc(AuditLog.created_at))\
            .offset((page - 1) * per_page)\
            .limit(per_page)\
            .all()

        return logs, total

    def get_user_activity(
        self,
        user_id: uuid.UUID,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get activity summary for a specific user.

        Args:
            user_id: User UUID
            days: Number of days to look back

        Returns:
            List of activity summaries by action
        """
        since = datetime.utcnow() - timedelta(days=days)

        results = self._base_query().filter(
            AuditLog.user_id == user_id,
            AuditLog.created_at >= since
        ).with_entities(
            AuditLog.action,
            func.count(AuditLog.id).label('count'),
            func.max(AuditLog.created_at).label('last_at')
        ).group_by(AuditLog.action).all()

        return [
            {
                "action": r.action,
                "count": r.count,
                "last_at": r.last_at.isoformat() if r.last_at else None
            }
            for r in results
        ]

    def get_resource_history(
        self,
        resource_type: str,
        resource_id: str,
        limit: int = 50
    ) -> List[AuditLog]:
        """
        Get audit history for a specific resource.

        Args:
            resource_type: Type of resource
            resource_id: Resource ID
            limit: Maximum number of entries

        Returns:
            List of audit log entries
        """
        return self._base_query().filter(
            AuditLog.resource_type == resource_type,
            AuditLog.resource_id == resource_id
        ).order_by(desc(AuditLog.created_at)).limit(limit).all()

    def get_activity_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        Get activity statistics for the tenant.

        Args:
            days: Number of days to analyze

        Returns:
            Dict with activity statistics
        """
        since = datetime.utcnow() - timedelta(days=days)

        base = self._base_query().filter(AuditLog.created_at >= since)

        # Total actions
        total_actions = base.count()

        # By action type
        by_action = dict(
            base.with_entities(
                AuditLog.action, func.count(AuditLog.id)
            ).group_by(AuditLog.action).all()
        )

        # By user
        by_user = base.join(User).with_entities(
            User.email, func.count(AuditLog.id).label('count')
        ).group_by(User.email).order_by(desc('count')).limit(10).all()

        # Daily activity
        daily = base.with_entities(
            func.date(AuditLog.created_at).label('date'),
            func.count(AuditLog.id).label('count')
        ).group_by('date').order_by('date').all()

        return {
            "period_days": days,
            "total_actions": total_actions,
            "by_action": by_action,
            "top_users": [{"email": u.email, "actions": u.count} for u in by_user],
            "daily_activity": [{"date": str(d.date), "count": d.count} for d in daily]
        }

    def cleanup_old_logs(self, days: int = 90) -> int:
        """
        Delete audit logs older than specified days.
        Admin-only operation.

        Args:
            days: Delete logs older than this many days

        Returns:
            Number of logs deleted
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        count = self._base_query().filter(
            AuditLog.created_at < cutoff
        ).delete(synchronize_session=False)

        self.db.flush()

        return count
