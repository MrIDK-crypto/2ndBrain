"""
Connector Service
Tenant-aware connector operations with proper isolation
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.database.models import (
    Connector, ConnectorType, create_audit_log
)


class ConnectorService:
    """
    Service layer for connector operations.
    All methods enforce tenant isolation.
    """

    def __init__(self, db: Session, tenant_id: uuid.UUID, user_id: Optional[uuid.UUID] = None):
        """
        Initialize service with database session and tenant context.

        Args:
            db: SQLAlchemy session
            tenant_id: Current tenant's UUID
            user_id: Current user's UUID (for audit logging)
        """
        self.db = db
        self.tenant_id = tenant_id
        self.user_id = user_id

    def _base_query(self):
        """Base query with tenant filter"""
        return self.db.query(Connector).filter(
            Connector.tenant_id == self.tenant_id,
            Connector.is_active == True
        )

    def get_by_id(self, connector_id: uuid.UUID) -> Optional[Connector]:
        """
        Get a connector by ID.

        Args:
            connector_id: Connector UUID

        Returns:
            Connector if found, None otherwise
        """
        return self._base_query().filter(Connector.id == connector_id).first()

    def get_by_type(self, connector_type: ConnectorType) -> Optional[Connector]:
        """
        Get the connector of a specific type for the tenant.
        Assumes one connector per type per tenant.

        Args:
            connector_type: Type of connector

        Returns:
            Connector if found, None otherwise
        """
        return self._base_query().filter(
            Connector.connector_type == connector_type
        ).first()

    def list_connectors(self) -> List[Connector]:
        """
        List all active connectors for the tenant.

        Returns:
            List of Connector instances
        """
        return self._base_query().order_by(desc(Connector.created_at)).all()

    def create_connector(
        self,
        connector_type: ConnectorType,
        name: Optional[str] = None,
        credentials: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> Connector:
        """
        Create a new connector.

        Args:
            connector_type: Type of connector
            name: User-friendly name
            credentials: OAuth credentials (encrypted in production)
            settings: Connector-specific settings

        Returns:
            Created Connector instance
        """
        connector = Connector(
            tenant_id=self.tenant_id,
            connector_type=connector_type,
            name=name or f"{connector_type.value.title()} Connector",
            credentials=credentials or {},
            settings=settings or {},
            sync_enabled=True,
            is_active=True
        )

        self.db.add(connector)
        self.db.flush()

        # Audit log (don't log credentials)
        create_audit_log(
            session=self.db,
            tenant_id=self.tenant_id,
            action="connector.create",
            user_id=self.user_id,
            resource_type="connector",
            resource_id=str(connector.id),
            details={
                "connector_type": connector_type.value,
                "name": connector.name
            }
        )

        return connector

    def update_credentials(
        self,
        connector_id: uuid.UUID,
        credentials: Dict[str, Any]
    ) -> Optional[Connector]:
        """
        Update connector credentials (e.g., after OAuth refresh).

        Args:
            connector_id: Connector UUID
            credentials: New credentials

        Returns:
            Updated Connector if found, None otherwise
        """
        connector = self.get_by_id(connector_id)
        if not connector:
            return None

        connector.credentials = credentials
        connector.updated_at = datetime.utcnow()
        self.db.flush()

        # Audit log (don't log actual credentials)
        create_audit_log(
            session=self.db,
            tenant_id=self.tenant_id,
            action="connector.credentials_updated",
            user_id=self.user_id,
            resource_type="connector",
            resource_id=str(connector_id),
            details={"connector_type": connector.connector_type.value}
        )

        return connector

    def update_sync_status(
        self,
        connector_id: uuid.UUID,
        status: str,
        error: Optional[str] = None,
        cursor: Optional[str] = None
    ) -> Optional[Connector]:
        """
        Update connector sync status after a sync operation.

        Args:
            connector_id: Connector UUID
            status: Sync status ('success', 'failed', 'in_progress')
            error: Error message if failed
            cursor: Sync cursor for incremental sync

        Returns:
            Updated Connector if found, None otherwise
        """
        connector = self.get_by_id(connector_id)
        if not connector:
            return None

        connector.last_sync_at = datetime.utcnow()
        connector.last_sync_status = status
        connector.last_sync_error = error
        if cursor:
            connector.sync_cursor = cursor

        self.db.flush()

        return connector

    def toggle_sync(self, connector_id: uuid.UUID, enabled: bool) -> Optional[Connector]:
        """
        Enable or disable sync for a connector.

        Args:
            connector_id: Connector UUID
            enabled: Whether sync should be enabled

        Returns:
            Updated Connector if found, None otherwise
        """
        connector = self.get_by_id(connector_id)
        if not connector:
            return None

        connector.sync_enabled = enabled
        self.db.flush()

        # Audit log
        create_audit_log(
            session=self.db,
            tenant_id=self.tenant_id,
            action="connector.sync_toggled",
            user_id=self.user_id,
            resource_type="connector",
            resource_id=str(connector_id),
            details={
                "connector_type": connector.connector_type.value,
                "sync_enabled": enabled
            }
        )

        return connector

    def disconnect(self, connector_id: uuid.UUID) -> bool:
        """
        Disconnect (deactivate) a connector.

        Args:
            connector_id: Connector UUID

        Returns:
            True if disconnected, False if not found
        """
        connector = self.get_by_id(connector_id)
        if not connector:
            return False

        connector.is_active = False
        connector.credentials = {}  # Clear credentials
        connector.sync_enabled = False
        self.db.flush()

        # Audit log
        create_audit_log(
            session=self.db,
            tenant_id=self.tenant_id,
            action="connector.disconnect",
            user_id=self.user_id,
            resource_type="connector",
            resource_id=str(connector_id),
            details={"connector_type": connector.connector_type.value}
        )

        return True

    def get_connectors_for_sync(self) -> List[Connector]:
        """
        Get all connectors that need syncing.

        Returns:
            List of connectors with sync enabled
        """
        return self._base_query().filter(
            Connector.sync_enabled == True
        ).all()

    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get summary of connector statuses.

        Returns:
            Dict with connector status information
        """
        connectors = self.list_connectors()

        return {
            "total": len(connectors),
            "connected": [
                {
                    "type": c.connector_type.value,
                    "name": c.name,
                    "sync_enabled": c.sync_enabled,
                    "last_sync_at": c.last_sync_at.isoformat() if c.last_sync_at else None,
                    "last_sync_status": c.last_sync_status
                }
                for c in connectors
            ],
            "available_types": [t.value for t in ConnectorType],
            "connected_types": [c.connector_type.value for c in connectors]
        }
