"""
Connector schemas for request/response validation
"""

from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class ConnectorTypeEnum(str, Enum):
    """Supported connector types"""
    GMAIL = "gmail"
    SLACK = "slack"
    GITHUB = "github"
    UPLOAD = "upload"


class ConnectorCreate(BaseModel):
    """Schema for creating a connector"""
    connector_type: ConnectorTypeEnum
    name: Optional[str] = Field(None, max_length=255)
    settings: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ConnectorResponse(BaseModel):
    """Schema for connector response"""
    id: UUID
    connector_type: ConnectorTypeEnum
    name: Optional[str]
    sync_enabled: bool
    last_sync_at: Optional[datetime]
    last_sync_status: Optional[str]
    settings: Dict[str, Any]
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


class ConnectorListResponse(BaseModel):
    """List of connectors response"""
    connectors: List[ConnectorResponse]
    available_types: List[str]
    connected_types: List[str]


class ConnectorStatusResponse(BaseModel):
    """Connector status summary"""
    total: int
    connected: List[Dict[str, Any]]
    available_types: List[str]
    connected_types: List[str]


class ConnectorSyncRequest(BaseModel):
    """Schema for triggering sync"""
    full_sync: bool = Field(default=False, description="Perform full sync instead of incremental")


class ConnectorSettingsUpdate(BaseModel):
    """Schema for updating connector settings"""
    name: Optional[str] = Field(None, max_length=255)
    sync_enabled: Optional[bool] = None
    settings: Optional[Dict[str, Any]] = None


class OAuthCallbackRequest(BaseModel):
    """Schema for OAuth callback"""
    code: str = Field(description="OAuth authorization code")
    state: Optional[str] = Field(None, description="OAuth state parameter")


class GmailAuthResponse(BaseModel):
    """Response for Gmail OAuth initiation"""
    auth_url: str = Field(description="URL to redirect user for OAuth")
    state: str = Field(description="State parameter for verification")


class ConnectorHealthCheck(BaseModel):
    """Connector health check response"""
    connector_type: ConnectorTypeEnum
    is_connected: bool
    is_healthy: bool
    last_check: datetime
    error: Optional[str] = None
