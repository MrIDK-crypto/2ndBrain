"""Initial schema - tenants, users, documents, connectors, audit_logs

Revision ID: 0001
Revises: None
Create Date: 2026-01-21

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create enum types
    op.execute("CREATE TYPE userrole AS ENUM ('owner', 'admin', 'editor', 'viewer')")
    op.execute("CREATE TYPE connectortype AS ENUM ('gmail', 'slack', 'github', 'upload')")
    op.execute("CREATE TYPE documentstatus AS ENUM ('pending', 'processing', 'indexed', 'failed', 'archived')")

    # Create tenants table
    op.create_table(
        'tenants',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('slug', sa.String(100), unique=True, nullable=False),
        sa.Column('settings', postgresql.JSON, default={}),
        sa.Column('plan', sa.String(50), default='free'),
        sa.Column('document_limit', sa.Integer, default=100),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, onupdate=sa.func.now()),
        sa.Column('is_active', sa.Boolean, default=True, nullable=False),
        sa.Column('deleted_at', sa.DateTime, nullable=True),
    )
    op.create_index('ix_tenants_slug', 'tenants', ['slug'])

    # Create users table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('tenants.id', ondelete='CASCADE'), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('name', sa.String(255), nullable=True),
        sa.Column('password_hash', sa.String(255), nullable=True),
        sa.Column('oauth_provider', sa.String(50), nullable=True),
        sa.Column('oauth_id', sa.String(255), nullable=True),
        sa.Column('role', postgresql.ENUM('owner', 'admin', 'editor', 'viewer', name='userrole', create_type=False), default='viewer', nullable=False),
        sa.Column('is_active', sa.Boolean, default=True, nullable=False),
        sa.Column('email_verified', sa.Boolean, default=False, nullable=False),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, onupdate=sa.func.now()),
        sa.Column('last_login_at', sa.DateTime, nullable=True),
    )
    op.create_index('ix_users_email', 'users', ['email'])
    op.create_unique_constraint('uq_user_tenant_email', 'users', ['tenant_id', 'email'])

    # Create documents table
    op.create_table(
        'documents',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('tenants.id', ondelete='CASCADE'), nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('title', sa.String(500), nullable=True),
        sa.Column('content', sa.Text, nullable=True),
        sa.Column('content_hash', sa.String(64), nullable=True),
        sa.Column('source_type', postgresql.ENUM('gmail', 'slack', 'github', 'upload', name='connectortype', create_type=False), nullable=False),
        sa.Column('source_id', sa.String(255), nullable=True),
        sa.Column('source_url', sa.String(2000), nullable=True),
        sa.Column('metadata', postgresql.JSON, default={}),
        sa.Column('status', postgresql.ENUM('pending', 'processing', 'indexed', 'failed', 'archived', name='documentstatus', create_type=False), default='pending', nullable=False),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('classification', sa.String(50), nullable=True),
        sa.Column('classification_confidence', sa.Integer, nullable=True),
        sa.Column('cluster_id', sa.String(100), nullable=True),
        sa.Column('cluster_label', sa.String(255), nullable=True),
        sa.Column('embedding_id', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, onupdate=sa.func.now()),
        sa.Column('indexed_at', sa.DateTime, nullable=True),
        sa.Column('is_deleted', sa.Boolean, default=False, nullable=False),
        sa.Column('deleted_at', sa.DateTime, nullable=True),
    )
    op.create_index('ix_doc_tenant_status', 'documents', ['tenant_id', 'status'])
    op.create_index('ix_doc_tenant_source', 'documents', ['tenant_id', 'source_type'])
    op.create_index('ix_doc_content_hash', 'documents', ['tenant_id', 'content_hash'])
    op.create_index('ix_doc_created_at', 'documents', ['tenant_id', 'created_at'])

    # Create connectors table
    op.create_table(
        'connectors',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('tenants.id', ondelete='CASCADE'), nullable=False),
        sa.Column('connector_type', postgresql.ENUM('gmail', 'slack', 'github', 'upload', name='connectortype', create_type=False), nullable=False),
        sa.Column('name', sa.String(255), nullable=True),
        sa.Column('credentials', postgresql.JSON, nullable=True),
        sa.Column('sync_enabled', sa.Boolean, default=True, nullable=False),
        sa.Column('last_sync_at', sa.DateTime, nullable=True),
        sa.Column('last_sync_status', sa.String(50), nullable=True),
        sa.Column('last_sync_error', sa.Text, nullable=True),
        sa.Column('sync_cursor', sa.String(500), nullable=True),
        sa.Column('settings', postgresql.JSON, default={}),
        sa.Column('is_active', sa.Boolean, default=True, nullable=False),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, onupdate=sa.func.now()),
    )
    op.create_index('ix_connector_tenant_type', 'connectors', ['tenant_id', 'connector_type'])

    # Create audit_logs table
    op.create_table(
        'audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('tenants.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('resource_type', sa.String(50), nullable=True),
        sa.Column('resource_id', sa.String(255), nullable=True),
        sa.Column('details', postgresql.JSON, default={}),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_audit_tenant_action', 'audit_logs', ['tenant_id', 'action'])
    op.create_index('ix_audit_tenant_created', 'audit_logs', ['tenant_id', 'created_at'])
    op.create_index('ix_audit_created_at', 'audit_logs', ['created_at'])


def downgrade() -> None:
    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table('audit_logs')
    op.drop_table('connectors')
    op.drop_table('documents')
    op.drop_table('users')
    op.drop_table('tenants')

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS documentstatus")
    op.execute("DROP TYPE IF EXISTS connectortype")
    op.execute("DROP TYPE IF EXISTS userrole")
