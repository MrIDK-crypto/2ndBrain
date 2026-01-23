"""
Document Service
Tenant-aware document operations with proper isolation
"""

import hashlib
import uuid
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func

from backend.database.models import (
    Document, DocumentStatus, ConnectorType,
    Tenant, User, UserRole, create_audit_log
)


class DocumentService:
    """
    Service layer for document operations.
    All methods enforce tenant isolation.
    """

    def __init__(self, db: Session, tenant_id: uuid.UUID, user_id: Optional[uuid.UUID] = None):
        """
        Initialize service with database session and tenant context.

        Args:
            db: SQLAlchemy session
            tenant_id: Current tenant's UUID (required for all operations)
            user_id: Current user's UUID (optional, for audit logging)
        """
        self.db = db
        self.tenant_id = tenant_id
        self.user_id = user_id

    def _base_query(self):
        """Base query with tenant filter - always use this"""
        return self.db.query(Document).filter(
            Document.tenant_id == self.tenant_id,
            Document.is_deleted == False
        )

    def get_by_id(self, document_id: uuid.UUID) -> Optional[Document]:
        """
        Get a document by ID within the current tenant.

        Args:
            document_id: Document UUID

        Returns:
            Document if found, None otherwise
        """
        return self._base_query().filter(Document.id == document_id).first()

    def get_by_source_id(self, source_type: ConnectorType, source_id: str) -> Optional[Document]:
        """
        Get a document by its source ID (e.g., Gmail message ID).
        Used for deduplication during sync.

        Args:
            source_type: Type of connector (gmail, slack, etc.)
            source_id: External source identifier

        Returns:
            Document if found, None otherwise
        """
        return self._base_query().filter(
            Document.source_type == source_type,
            Document.source_id == source_id
        ).first()

    def get_by_content_hash(self, content_hash: str) -> Optional[Document]:
        """
        Get a document by content hash.
        Used for content-based deduplication.

        Args:
            content_hash: SHA-256 hash of document content

        Returns:
            Document if found, None otherwise
        """
        return self._base_query().filter(Document.content_hash == content_hash).first()

    def list_documents(
        self,
        status: Optional[DocumentStatus] = None,
        source_type: Optional[ConnectorType] = None,
        classification: Optional[str] = None,
        search_query: Optional[str] = None,
        page: int = 1,
        per_page: int = 50,
        order_by: str = "created_at",
        order_desc: bool = True
    ) -> Tuple[List[Document], int]:
        """
        List documents with filtering and pagination.

        Args:
            status: Filter by document status
            source_type: Filter by source connector type
            classification: Filter by classification
            search_query: Search in title (basic LIKE search)
            page: Page number (1-indexed)
            per_page: Items per page (max 100)
            order_by: Field to order by
            order_desc: Descending order if True

        Returns:
            Tuple of (documents list, total count)
        """
        # Enforce pagination limits
        per_page = min(max(per_page, 1), 100)
        page = max(page, 1)

        query = self._base_query()

        # Apply filters
        if status:
            query = query.filter(Document.status == status)
        if source_type:
            query = query.filter(Document.source_type == source_type)
        if classification:
            query = query.filter(Document.classification == classification)
        if search_query:
            # Basic search - for full-text search, use PostgreSQL FTS or vector search
            query = query.filter(Document.title.ilike(f"%{search_query}%"))

        # Get total count before pagination
        total = query.count()

        # Apply ordering
        order_column = getattr(Document, order_by, Document.created_at)
        if order_desc:
            query = query.order_by(desc(order_column))
        else:
            query = query.order_by(order_column)

        # Apply pagination
        offset = (page - 1) * per_page
        documents = query.offset(offset).limit(per_page).all()

        return documents, total

    def create_document(
        self,
        source_type: ConnectorType,
        title: Optional[str] = None,
        content: Optional[str] = None,
        source_id: Optional[str] = None,
        source_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        status: DocumentStatus = DocumentStatus.PENDING
    ) -> Document:
        """
        Create a new document within the tenant.

        Args:
            source_type: Type of connector that created this document
            title: Document title
            content: Document content
            source_id: External source identifier
            source_url: URL to original source
            metadata: Additional metadata as dict
            status: Initial processing status

        Returns:
            Created Document instance
        """
        # Calculate content hash for deduplication
        content_hash = None
        if content:
            content_hash = hashlib.sha256(content.encode()).hexdigest()

        document = Document(
            tenant_id=self.tenant_id,
            created_by=self.user_id,
            source_type=source_type,
            title=title,
            content=content,
            content_hash=content_hash,
            source_id=source_id,
            source_url=source_url,
            doc_metadata=metadata or {},
            status=status
        )

        self.db.add(document)
        self.db.flush()  # Get the ID without committing

        # Audit log
        create_audit_log(
            session=self.db,
            tenant_id=self.tenant_id,
            action="document.create",
            user_id=self.user_id,
            resource_type="document",
            resource_id=str(document.id),
            details={"title": title, "source_type": source_type.value}
        )

        return document

    def update_document(
        self,
        document_id: uuid.UUID,
        **kwargs
    ) -> Optional[Document]:
        """
        Update a document's fields.

        Args:
            document_id: Document UUID
            **kwargs: Fields to update

        Returns:
            Updated Document if found, None otherwise
        """
        document = self.get_by_id(document_id)
        if not document:
            return None

        # Allowed fields to update
        allowed_fields = {
            'title', 'content', 'status', 'classification',
            'classification_confidence', 'cluster_id', 'cluster_label',
            'embedding_id', 'error_message', 'doc_metadata'
        }

        changes = {}
        for key, value in kwargs.items():
            if key in allowed_fields:
                old_value = getattr(document, key)
                setattr(document, key, value)
                changes[key] = {"old": str(old_value), "new": str(value)}

        # Update content hash if content changed
        if 'content' in kwargs and kwargs['content']:
            document.content_hash = hashlib.sha256(kwargs['content'].encode()).hexdigest()

        # Mark as indexed if status changed to indexed
        if kwargs.get('status') == DocumentStatus.INDEXED:
            document.indexed_at = datetime.utcnow()

        self.db.flush()

        # Audit log
        create_audit_log(
            session=self.db,
            tenant_id=self.tenant_id,
            action="document.update",
            user_id=self.user_id,
            resource_type="document",
            resource_id=str(document_id),
            details={"changes": changes}
        )

        return document

    def soft_delete_document(self, document_id: uuid.UUID) -> bool:
        """
        Soft delete a document (mark as deleted).

        Args:
            document_id: Document UUID

        Returns:
            True if deleted, False if not found
        """
        document = self.get_by_id(document_id)
        if not document:
            return False

        document.is_deleted = True
        document.deleted_at = datetime.utcnow()
        self.db.flush()

        # Audit log
        create_audit_log(
            session=self.db,
            tenant_id=self.tenant_id,
            action="document.delete",
            user_id=self.user_id,
            resource_type="document",
            resource_id=str(document_id),
            details={"title": document.title}
        )

        return True

    def hard_delete_document(self, document_id: uuid.UUID) -> bool:
        """
        Permanently delete a document.
        Use with caution - typically for admin cleanup.

        Args:
            document_id: Document UUID

        Returns:
            True if deleted, False if not found
        """
        document = self.db.query(Document).filter(
            Document.id == document_id,
            Document.tenant_id == self.tenant_id
        ).first()

        if not document:
            return False

        title = document.title
        self.db.delete(document)
        self.db.flush()

        # Audit log
        create_audit_log(
            session=self.db,
            tenant_id=self.tenant_id,
            action="document.hard_delete",
            user_id=self.user_id,
            resource_type="document",
            resource_id=str(document_id),
            details={"title": title}
        )

        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Get document statistics for the tenant.

        Returns:
            Dict with counts by status, source_type, etc.
        """
        base = self.db.query(Document).filter(
            Document.tenant_id == self.tenant_id,
            Document.is_deleted == False
        )

        # Total count
        total = base.count()

        # By status
        by_status = dict(
            base.with_entities(
                Document.status, func.count(Document.id)
            ).group_by(Document.status).all()
        )

        # By source type
        by_source = dict(
            base.with_entities(
                Document.source_type, func.count(Document.id)
            ).group_by(Document.source_type).all()
        )

        # By classification
        by_classification = dict(
            base.filter(Document.classification.isnot(None)).with_entities(
                Document.classification, func.count(Document.id)
            ).group_by(Document.classification).all()
        )

        return {
            "total": total,
            "by_status": {k.value if k else "unknown": v for k, v in by_status.items()},
            "by_source_type": {k.value if k else "unknown": v for k, v in by_source.items()},
            "by_classification": by_classification,
            "indexed_count": by_status.get(DocumentStatus.INDEXED, 0),
            "pending_count": by_status.get(DocumentStatus.PENDING, 0),
            "failed_count": by_status.get(DocumentStatus.FAILED, 0),
        }

    def get_documents_for_indexing(self, batch_size: int = 50) -> List[Document]:
        """
        Get pending documents for indexing.

        Args:
            batch_size: Maximum number of documents to return

        Returns:
            List of pending documents
        """
        return self._base_query().filter(
            Document.status == DocumentStatus.PENDING
        ).order_by(Document.created_at).limit(batch_size).all()

    def bulk_update_status(
        self,
        document_ids: List[uuid.UUID],
        status: DocumentStatus,
        error_message: Optional[str] = None
    ) -> int:
        """
        Bulk update status for multiple documents.

        Args:
            document_ids: List of document UUIDs
            status: New status
            error_message: Error message (for failed status)

        Returns:
            Number of documents updated
        """
        update_data = {"status": status}
        if status == DocumentStatus.INDEXED:
            update_data["indexed_at"] = datetime.utcnow()
        if error_message:
            update_data["error_message"] = error_message

        count = self.db.query(Document).filter(
            Document.id.in_(document_ids),
            Document.tenant_id == self.tenant_id
        ).update(update_data, synchronize_session=False)

        # Audit log
        create_audit_log(
            session=self.db,
            tenant_id=self.tenant_id,
            action="document.bulk_update_status",
            user_id=self.user_id,
            resource_type="document",
            details={
                "count": count,
                "status": status.value,
                "document_ids": [str(d) for d in document_ids[:10]]  # Limit for log size
            }
        )

        return count

    def check_document_limit(self) -> Tuple[bool, int, int]:
        """
        Check if tenant has reached their document limit.

        Returns:
            Tuple of (can_add_more, current_count, limit)
        """
        tenant = self.db.query(Tenant).filter(Tenant.id == self.tenant_id).first()
        if not tenant:
            return False, 0, 0

        current_count = self._base_query().count()
        limit = tenant.document_limit or 100

        return current_count < limit, current_count, limit
