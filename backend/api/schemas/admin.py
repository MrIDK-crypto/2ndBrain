"""
Admin schemas for user, tenant, and audit management
"""

from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr, ConfigDict
from enum import Enum


class UserRoleEnum(str, Enum):
    """User roles for RBAC"""
    OWNER = "owner"
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"


class UserCreate(BaseModel):
    """Schema for creating a user (admin only)"""
    email: EmailStr
    name: Optional[str] = Field(None, max_length=255)
    password: Optional[str] = Field(None, min_length=8, max_length=128)
    role: UserRoleEnum = Field(default=UserRoleEnum.VIEWER)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "newuser@company.com",
                "name": "New User",
                "role": "editor"
            }
        }
    )


class UserUpdate(BaseModel):
    """Schema for updating a user"""
    name: Optional[str] = Field(None, max_length=255)
    role: Optional[UserRoleEnum] = None
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    """User response schema"""
    id: UUID
    email: str
    name: Optional[str]
    role: UserRoleEnum
    is_active: bool
    email_verified: bool
    created_at: datetime
    last_login_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


class UserListResponse(BaseModel):
    """Paginated user list"""
    users: List[UserResponse]
    total: int
    page: int
    per_page: int


class TenantResponse(BaseModel):
    """Tenant information response"""
    id: UUID
    name: str
    slug: str
    plan: str
    document_limit: int
    settings: Dict[str, Any]
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TenantUpdate(BaseModel):
    """Schema for updating tenant settings"""
    name: Optional[str] = Field(None, max_length=255)
    settings: Optional[Dict[str, Any]] = None


class TenantStatsResponse(BaseModel):
    """Tenant statistics"""
    tenant: TenantResponse
    user_count: int
    document_count: int
    connector_count: int
    document_limit: int
    documents_remaining: int
    storage_used_mb: float


class AuditLogResponse(BaseModel):
    """Audit log entry response"""
    id: UUID
    user_id: Optional[UUID]
    user_email: Optional[str]
    action: str
    resource_type: Optional[str]
    resource_id: Optional[str]
    details: Dict[str, Any]
    ip_address: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AuditLogListResponse(BaseModel):
    """Paginated audit log list"""
    logs: List[AuditLogResponse]
    total: int
    page: int
    per_page: int


class AuditLogFilters(BaseModel):
    """Filters for audit log queries"""
    action: Optional[str] = None
    resource_type: Optional[str] = None
    user_id: Optional[UUID] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    page: int = Field(default=1, ge=1)
    per_page: int = Field(default=50, ge=1, le=100)


class ActivityStatsResponse(BaseModel):
    """Activity statistics response"""
    period_days: int
    total_actions: int
    by_action: Dict[str, int]
    top_users: List[Dict[str, Any]]
    daily_activity: List[Dict[str, Any]]


class InviteUserRequest(BaseModel):
    """Schema for inviting a new user"""
    email: EmailStr
    role: UserRoleEnum = Field(default=UserRoleEnum.VIEWER)
    send_email: bool = Field(default=True, description="Send invitation email")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "colleague@company.com",
                "role": "editor",
                "send_email": True
            }
        }
    )


class PasswordResetRequest(BaseModel):
    """Schema for password reset"""
    current_password: str = Field(min_length=1)
    new_password: str = Field(min_length=8, max_length=128)


class AdminDashboardResponse(BaseModel):
    """Admin dashboard summary"""
    tenant: TenantResponse
    stats: TenantStatsResponse
    recent_activity: List[AuditLogResponse]
    connector_status: Dict[str, Any]
