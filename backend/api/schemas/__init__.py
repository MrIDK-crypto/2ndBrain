"""
2ndBrain API Schemas
Pydantic models for request validation and response serialization
"""

from .common import (
    PaginationParams,
    PaginatedResponse,
    ErrorResponse,
    SuccessResponse,
)
from .documents import (
    DocumentCreate,
    DocumentUpdate,
    DocumentResponse,
    DocumentListResponse,
    DocumentStatsResponse,
)
from .connectors import (
    ConnectorCreate,
    ConnectorResponse,
    ConnectorListResponse,
    ConnectorStatusResponse,
)
from .search import (
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from .admin import (
    UserResponse,
    UserCreate,
    UserUpdate,
    TenantResponse,
    TenantUpdate,
    AuditLogResponse,
)

__all__ = [
    # Common
    "PaginationParams",
    "PaginatedResponse",
    "ErrorResponse",
    "SuccessResponse",
    # Documents
    "DocumentCreate",
    "DocumentUpdate",
    "DocumentResponse",
    "DocumentListResponse",
    "DocumentStatsResponse",
    # Connectors
    "ConnectorCreate",
    "ConnectorResponse",
    "ConnectorListResponse",
    "ConnectorStatusResponse",
    # Search
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    # Admin
    "UserResponse",
    "UserCreate",
    "UserUpdate",
    "TenantResponse",
    "TenantUpdate",
    "AuditLogResponse",
]
