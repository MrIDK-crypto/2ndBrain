"""
Document schemas for request/response validation
"""

from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class DocumentStatusEnum(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    ARCHIVED = "archived"


class SourceTypeEnum(str, Enum):
    """Connector/source types"""
    GMAIL = "gmail"
    SLACK = "slack"
    GITHUB = "github"
    UPLOAD = "upload"


class DocumentCreate(BaseModel):
    """Schema for creating a document"""
    title: Optional[str] = Field(None, max_length=500, description="Document title")
    content: Optional[str] = Field(None, description="Document content")
    source_type: SourceTypeEnum = Field(description="Source connector type")
    source_id: Optional[str] = Field(None, max_length=255, description="External source ID")
    source_url: Optional[str] = Field(None, max_length=2000, description="URL to original")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Project Meeting Notes",
                "content": "Discussed Q1 roadmap...",
                "source_type": "upload",
                "metadata": {"tags": ["meeting", "q1"]}
            }
        }
    )


class DocumentUpdate(BaseModel):
    """Schema for updating a document"""
    title: Optional[str] = Field(None, max_length=500)
    content: Optional[str] = None
    status: Optional[DocumentStatusEnum] = None
    classification: Optional[str] = Field(None, max_length=50)
    metadata: Optional[Dict[str, Any]] = None


class DocumentResponse(BaseModel):
    """Schema for document response"""
    id: UUID
    title: Optional[str]
    content: Optional[str] = Field(None, description="Content (may be truncated)")
    source_type: SourceTypeEnum
    source_id: Optional[str]
    source_url: Optional[str]
    status: DocumentStatusEnum
    classification: Optional[str]
    classification_confidence: Optional[int]
    cluster_id: Optional[str]
    cluster_label: Optional[str]
    metadata: Dict[str, Any] = Field(default_factory=dict, validation_alias='doc_metadata')
    created_at: datetime
    updated_at: Optional[datetime]
    indexed_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class DocumentSummaryResponse(BaseModel):
    """Abbreviated document response for lists"""
    id: UUID
    title: Optional[str]
    source_type: SourceTypeEnum
    status: DocumentStatusEnum
    classification: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    """Paginated document list response"""
    documents: List[DocumentSummaryResponse]
    total: int
    page: int
    per_page: int
    total_pages: int
    has_next: bool
    has_prev: bool

    @classmethod
    def create(cls, documents: list, total: int, page: int, per_page: int):
        total_pages = (total + per_page - 1) // per_page if per_page > 0 else 0
        return cls(
            documents=[DocumentSummaryResponse.model_validate(d) for d in documents],
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )


class DocumentStatsResponse(BaseModel):
    """Document statistics response"""
    total: int
    indexed_count: int
    pending_count: int
    failed_count: int
    by_status: Dict[str, int]
    by_source_type: Dict[str, int]
    by_classification: Dict[str, int]


class DocumentUploadRequest(BaseModel):
    """Schema for file upload"""
    title: Optional[str] = Field(None, max_length=500)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    # File is handled separately via multipart form


class DocumentDecisionRequest(BaseModel):
    """Schema for document classification decision"""
    document_id: UUID
    decision: str = Field(description="Classification decision (work/personal/archive)")
    notes: Optional[str] = Field(None, max_length=1000, description="Optional notes")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "decision": "work",
                "notes": "Important project document"
            }
        }
    )


class BulkStatusUpdateRequest(BaseModel):
    """Schema for bulk status update"""
    document_ids: List[UUID] = Field(min_length=1, max_length=100)
    status: DocumentStatusEnum
    error_message: Optional[str] = None
