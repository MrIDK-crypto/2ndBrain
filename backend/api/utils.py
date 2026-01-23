"""
API Utilities
Request validation, error handling, and common helpers
"""

import uuid
import logging
from functools import wraps
from typing import Type, Optional, Any, Callable

from flask import request, jsonify, g
from pydantic import BaseModel, ValidationError

from backend.database.database import get_db
from backend.services import DocumentService, ConnectorService, AuditService

logger = logging.getLogger(__name__)


def validate_request(schema: Type[BaseModel]):
    """
    Decorator to validate request JSON body with Pydantic schema.
    Validated data is available in g.validated_data

    Usage:
        @app.route('/api/documents', methods=['POST'])
        @require_auth
        @validate_request(DocumentCreate)
        def create_document():
            data = g.validated_data
            ...
    """
    def decorator(f: Callable):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                json_data = request.get_json(silent=True) or {}
                g.validated_data = schema.model_validate(json_data)
                return f(*args, **kwargs)
            except ValidationError as e:
                errors = []
                for error in e.errors():
                    errors.append({
                        "field": ".".join(str(x) for x in error["loc"]),
                        "message": error["msg"],
                        "type": error["type"]
                    })
                return jsonify({
                    "error": "validation_error",
                    "message": "Invalid request data",
                    "details": {"errors": errors}
                }), 400
        return wrapper
    return decorator


def validate_query_params(schema: Type[BaseModel]):
    """
    Decorator to validate query parameters with Pydantic schema.
    Validated data is available in g.query_params
    """
    def decorator(f: Callable):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                params = dict(request.args)
                # Convert string values to appropriate types
                g.query_params = schema.model_validate(params)
                return f(*args, **kwargs)
            except ValidationError as e:
                errors = []
                for error in e.errors():
                    errors.append({
                        "field": ".".join(str(x) for x in error["loc"]),
                        "message": error["msg"],
                        "type": error["type"]
                    })
                return jsonify({
                    "error": "validation_error",
                    "message": "Invalid query parameters",
                    "details": {"errors": errors}
                }), 400
        return wrapper
    return decorator


def get_document_service() -> DocumentService:
    """
    Get DocumentService instance for current request.
    Requires authentication context (g.current_user, g.current_tenant).
    """
    if not hasattr(g, 'current_tenant') or not g.current_tenant:
        raise RuntimeError("No tenant context - authentication required")

    # Use the db session from the auth decorator if available
    db = getattr(g, 'db', None) or next(get_db())
    user_id = g.current_user.id if hasattr(g, 'current_user') and g.current_user else None

    return DocumentService(
        db=db,
        tenant_id=g.current_tenant.id,
        user_id=user_id
    )


def get_connector_service() -> ConnectorService:
    """
    Get ConnectorService instance for current request.
    """
    if not hasattr(g, 'current_tenant') or not g.current_tenant:
        raise RuntimeError("No tenant context - authentication required")

    db = getattr(g, 'db', None) or next(get_db())
    user_id = g.current_user.id if hasattr(g, 'current_user') and g.current_user else None

    return ConnectorService(
        db=db,
        tenant_id=g.current_tenant.id,
        user_id=user_id
    )


def get_audit_service() -> AuditService:
    """
    Get AuditService instance for current request.
    """
    if not hasattr(g, 'current_tenant') or not g.current_tenant:
        raise RuntimeError("No tenant context - authentication required")

    db = getattr(g, 'db', None) or next(get_db())
    user_id = g.current_user.id if hasattr(g, 'current_user') and g.current_user else None

    return AuditService(
        db=db,
        tenant_id=g.current_tenant.id,
        user_id=user_id
    )


def error_response(
    error: str,
    message: str,
    status_code: int = 400,
    details: Optional[dict] = None
) -> tuple:
    """
    Create standardized error response.

    Args:
        error: Error type/code
        message: Human-readable message
        status_code: HTTP status code
        details: Additional error details

    Returns:
        Tuple of (response, status_code)
    """
    response = {
        "error": error,
        "message": message
    }
    if details:
        response["details"] = details

    return jsonify(response), status_code


def success_response(
    message: str = "Success",
    data: Optional[Any] = None,
    status_code: int = 200
) -> tuple:
    """
    Create standardized success response.
    """
    response = {
        "success": True,
        "message": message
    }
    if data is not None:
        response["data"] = data

    return jsonify(response), status_code


def parse_uuid(value: str, field_name: str = "id") -> Optional[uuid.UUID]:
    """
    Safely parse a UUID string.

    Args:
        value: UUID string
        field_name: Field name for error message

    Returns:
        UUID if valid, None otherwise
    """
    try:
        return uuid.UUID(value)
    except (ValueError, TypeError):
        return None


def get_client_ip() -> str:
    """Get client IP address, accounting for proxies."""
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    return request.remote_addr or 'unknown'


def get_user_agent() -> str:
    """Get user agent string (truncated for storage)."""
    ua = request.headers.get('User-Agent', 'unknown')
    return ua[:500] if ua else 'unknown'


def log_api_error(error: Exception, context: str = ""):
    """Log API error with context."""
    user_info = ""
    if hasattr(g, 'current_user') and g.current_user:
        # g.current_user is a User model object, not a dict
        email = getattr(g.current_user, 'email', 'unknown')
        user_info = f"user={email}"

    logger.error(
        f"API Error [{context}] {user_info}: {type(error).__name__}: {str(error)}",
        exc_info=True
    )


def commit_or_rollback(db):
    """
    Helper to commit transaction or rollback on error.
    Usage:
        with commit_or_rollback(db):
            # do stuff
    """
    class CommitContext:
        def __init__(self, session):
            self.session = session

        def __enter__(self):
            return self.session

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type:
                self.session.rollback()
                return False
            try:
                self.session.commit()
            except Exception:
                self.session.rollback()
                raise
            return False

    return CommitContext(db)
