"""
2ndBrain Database Models
Multi-tenant schema with proper relationships and constraints
Supports both PostgreSQL (production) and SQLite (development)
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, List

from sqlalchemy import (
    Column, String, Text, Boolean, DateTime, Integer,
    ForeignKey, Enum as SQLEnum, JSON, Index, UniqueConstraint,
    event, TypeDecorator
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship, validates
from passlib.context import CryptContext

from .database import Base


# Cross-database UUID type that works with both PostgreSQL and SQLite
class UUID(TypeDecorator):
    """Platform-independent UUID type.
    Uses PostgreSQL's UUID type when available, otherwise uses String(36).
    """
    impl = String(36)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == 'postgresql':
            return str(value)
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        return uuid.UUID(value) if not isinstance(value, uuid.UUID) else value

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PG_UUID())
        return dialect.type_descriptor(String(36))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserRole(str, Enum):
    """User roles for RBAC"""
    OWNER = "owner"          # Full access, can delete tenant
    ADMIN = "admin"          # Full access except delete tenant
    EDITOR = "editor"        # Can create/edit documents
    VIEWER = "viewer"        # Read-only access


class ConnectorType(str, Enum):
    """Supported connector types"""
    GMAIL = "gmail"
    SLACK = "slack"
    GITHUB = "github"
    UPLOAD = "upload"


class DocumentStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    ARCHIVED = "archived"


class Tenant(Base):
    """
    Tenant/Organization model.
    All data is isolated by tenant_id.
    """
    __tablename__ = "tenants"

    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)

    # Settings stored as JSON for flexibility
    settings = Column(JSON, default=dict)

    # Subscription/limits (for future use)
    plan = Column(String(50), default="free")
    document_limit = Column(Integer, default=100)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Soft delete
    is_active = Column(Boolean, default=True, nullable=False)
    deleted_at = Column(DateTime, nullable=True)

    # Relationships
    users = relationship("User", back_populates="tenant", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="tenant", cascade="all, delete-orphan")
    connectors = relationship("Connector", back_populates="tenant", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="tenant", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Tenant {self.slug}>"

    @validates('slug')
    def validate_slug(self, key, slug):
        """Ensure slug is lowercase and URL-safe"""
        import re
        slug = slug.lower().strip()
        if not re.match(r'^[a-z0-9][a-z0-9-]*[a-z0-9]$', slug) and len(slug) > 2:
            raise ValueError("Slug must be lowercase alphanumeric with hyphens")
        return slug


class User(Base):
    """
    User model with tenant association.
    Supports both email/password and OAuth authentication.
    """
    __tablename__ = "users"

    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)

    # Identity
    email = Column(String(255), nullable=False)
    name = Column(String(255), nullable=True)

    # Authentication
    password_hash = Column(String(255), nullable=True)  # Null for OAuth-only users
    oauth_provider = Column(String(50), nullable=True)  # 'google', 'github', etc.
    oauth_id = Column(String(255), nullable=True)

    # Role and permissions
    role = Column(SQLEnum(UserRole), default=UserRole.VIEWER, nullable=False)

    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    email_verified = Column(Boolean, default=False, nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)

    # Relationships
    tenant = relationship("Tenant", back_populates="users")
    documents = relationship("Document", back_populates="created_by_user")
    audit_logs = relationship("AuditLog", back_populates="user")

    # Constraints
    __table_args__ = (
        UniqueConstraint('tenant_id', 'email', name='uq_user_tenant_email'),
        Index('ix_user_email', 'email'),
    )

    def __repr__(self):
        return f"<User {self.email}>"

    def set_password(self, password: str):
        """Hash and set the user's password"""
        self.password_hash = pwd_context.hash(password)

    def verify_password(self, password: str) -> bool:
        """Verify a password against the hash"""
        if not self.password_hash:
            return False
        return pwd_context.verify(password, self.password_hash)

    def has_permission(self, required_role: UserRole) -> bool:
        """Check if user has at least the required role level"""
        role_hierarchy = {
            UserRole.VIEWER: 0,
            UserRole.EDITOR: 1,
            UserRole.ADMIN: 2,
            UserRole.OWNER: 3
        }
        return role_hierarchy.get(self.role, 0) >= role_hierarchy.get(required_role, 0)


class Document(Base):
    """
    Document model for storing indexed content.
    Each document belongs to a tenant.
    """
    __tablename__ = "documents"

    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    created_by = Column(UUID(), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    # Document info
    title = Column(String(500), nullable=True)
    content = Column(Text, nullable=True)
    content_hash = Column(String(64), nullable=True)  # SHA-256 for deduplication

    # Source information
    source_type = Column(SQLEnum(ConnectorType), nullable=False)
    source_id = Column(String(255), nullable=True)  # External ID (email ID, Slack message ID, etc.)
    source_url = Column(String(2000), nullable=True)

    # Metadata stored as JSON (named doc_metadata to avoid SQLAlchemy reserved name)
    doc_metadata = Column('metadata', JSON, default=dict)

    # Processing status
    status = Column(SQLEnum(DocumentStatus), default=DocumentStatus.PENDING, nullable=False)
    error_message = Column(Text, nullable=True)

    # Classification
    classification = Column(String(50), nullable=True)  # 'work', 'personal', 'uncertain'
    classification_confidence = Column(Integer, nullable=True)  # 0-100

    # Clustering
    cluster_id = Column(String(100), nullable=True)
    cluster_label = Column(String(255), nullable=True)

    # Vector embedding reference (stored in vector DB, not here)
    embedding_id = Column(String(255), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    indexed_at = Column(DateTime, nullable=True)

    # Soft delete
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime, nullable=True)

    # Relationships
    tenant = relationship("Tenant", back_populates="documents")
    created_by_user = relationship("User", back_populates="documents")

    # Indexes for common queries
    __table_args__ = (
        Index('ix_doc_tenant_status', 'tenant_id', 'status'),
        Index('ix_doc_tenant_source', 'tenant_id', 'source_type'),
        Index('ix_doc_content_hash', 'tenant_id', 'content_hash'),
        Index('ix_doc_created_at', 'tenant_id', 'created_at'),
    )

    def __repr__(self):
        return f"<Document {self.id} - {self.title[:30] if self.title else 'Untitled'}>"


class Connector(Base):
    """
    OAuth connector configuration.
    Stores encrypted credentials for external services.
    """
    __tablename__ = "connectors"

    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)

    # Connector info
    connector_type = Column(SQLEnum(ConnectorType), nullable=False)
    name = Column(String(255), nullable=True)  # User-friendly name

    # Credentials (encrypted in production)
    # In production, use a secrets manager or encrypted column
    credentials = Column(JSON, nullable=True)  # access_token, refresh_token, etc.

    # Sync configuration
    sync_enabled = Column(Boolean, default=True, nullable=False)
    last_sync_at = Column(DateTime, nullable=True)
    last_sync_status = Column(String(50), nullable=True)
    last_sync_error = Column(Text, nullable=True)
    sync_cursor = Column(String(500), nullable=True)  # For incremental sync

    # Settings specific to this connector
    settings = Column(JSON, default=dict)  # e.g., which folders to sync

    # Status
    is_active = Column(Boolean, default=True, nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    tenant = relationship("Tenant", back_populates="connectors")

    # Constraints
    __table_args__ = (
        Index('ix_connector_tenant_type', 'tenant_id', 'connector_type'),
    )

    def __repr__(self):
        return f"<Connector {self.connector_type.value} for tenant {self.tenant_id}>"


class AuditLog(Base):
    """
    Audit log for tracking user actions.
    Important for compliance and debugging.
    """
    __tablename__ = "audit_logs"

    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    # Action details
    action = Column(String(100), nullable=False)  # 'document.create', 'user.login', etc.
    resource_type = Column(String(50), nullable=True)  # 'document', 'user', 'connector'
    resource_id = Column(String(255), nullable=True)

    # Additional context
    details = Column(JSON, default=dict)  # Any additional data
    ip_address = Column(String(45), nullable=True)  # IPv4 or IPv6
    user_agent = Column(String(500), nullable=True)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    tenant = relationship("Tenant", back_populates="audit_logs")
    user = relationship("User", back_populates="audit_logs")

    # Index for querying logs
    __table_args__ = (
        Index('ix_audit_tenant_action', 'tenant_id', 'action'),
        Index('ix_audit_tenant_created', 'tenant_id', 'created_at'),
    )

    def __repr__(self):
        return f"<AuditLog {self.action} at {self.created_at}>"


# Helper function to create audit logs
def create_audit_log(
    session,
    tenant_id: uuid.UUID,
    action: str,
    user_id: Optional[uuid.UUID] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    details: Optional[dict] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
) -> AuditLog:
    """
    Helper to create an audit log entry.

    Usage:
        create_audit_log(
            session=db,
            tenant_id=current_tenant.id,
            action="document.create",
            user_id=current_user.id,
            resource_type="document",
            resource_id=str(doc.id),
            details={"title": doc.title}
        )
    """
    log = AuditLog(
        tenant_id=tenant_id,
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details or {},
        ip_address=ip_address,
        user_agent=user_agent
    )
    session.add(log)
    return log
