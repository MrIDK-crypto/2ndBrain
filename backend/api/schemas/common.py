"""
Common schemas used across the API
"""

from typing import Optional, Any, List, Generic, TypeVar
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

T = TypeVar('T')


class PaginationParams(BaseModel):
    """Pagination parameters for list endpoints"""
    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    per_page: int = Field(default=50, ge=1, le=100, description="Items per page")
    order_by: Optional[str] = Field(default=None, description="Field to order by")
    order_desc: bool = Field(default=True, description="Descending order")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response"""
    items: List[T]
    total: int = Field(description="Total number of items")
    page: int = Field(description="Current page number")
    per_page: int = Field(description="Items per page")
    total_pages: int = Field(description="Total number of pages")
    has_next: bool = Field(description="Whether there's a next page")
    has_prev: bool = Field(description="Whether there's a previous page")

    @classmethod
    def create(cls, items: List[T], total: int, page: int, per_page: int):
        """Factory method to create paginated response"""
        total_pages = (total + per_page - 1) // per_page if per_page > 0 else 0
        return cls(
            items=items,
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )


class ErrorResponse(BaseModel):
    """Standard error response format"""
    error: str = Field(description="Error type/code")
    message: str = Field(description="Human-readable error message")
    details: Optional[dict] = Field(default=None, description="Additional error details")
    request_id: Optional[str] = Field(default=None, description="Request ID for support")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "validation_error",
                "message": "Invalid input provided",
                "details": {"field": "email", "issue": "invalid format"},
                "request_id": "req_abc123"
            }
        }
    )


class SuccessResponse(BaseModel):
    """Standard success response"""
    success: bool = Field(default=True)
    message: str = Field(description="Success message")
    data: Optional[Any] = Field(default=None, description="Response data")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(description="Service status")
    version: str = Field(description="API version")
    timestamp: datetime = Field(description="Server timestamp")


class TimestampMixin(BaseModel):
    """Mixin for models with timestamps"""
    created_at: datetime
    updated_at: Optional[datetime] = None
