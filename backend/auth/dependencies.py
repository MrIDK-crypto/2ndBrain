"""
Authentication Dependencies
Provides decorators and context managers for authentication
Works with both Flask and FastAPI
"""

import logging
from functools import wraps
from typing import Optional, Callable, Any
from dataclasses import dataclass
from uuid import UUID

from flask import request, g, jsonify

from .jwt import verify_token, TokenData
from ..database.models import User, Tenant, UserRole

logger = logging.getLogger(__name__)


@dataclass
class AuthContext:
    """Current authentication context"""
    user: Optional[User] = None
    tenant: Optional[Tenant] = None
    token_data: Optional[TokenData] = None

    @property
    def is_authenticated(self) -> bool:
        return self.user is not None and self.tenant is not None

    @property
    def user_id(self) -> Optional[UUID]:
        return self.user.id if self.user else None

    @property
    def tenant_id(self) -> Optional[UUID]:
        return self.tenant.id if self.tenant else None


def extract_token_from_request() -> Optional[str]:
    """
    Extract JWT token from request.
    Checks Authorization header first, then cookies.
    """
    # Check Authorization header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]

    # Check cookie as fallback
    return request.cookies.get("access_token")


def get_current_user(db_session) -> Optional[AuthContext]:
    """
    Get the current authenticated user from the request.

    Args:
        db_session: SQLAlchemy database session

    Returns:
        AuthContext with user and tenant, or None if not authenticated
    """
    token = extract_token_from_request()
    if not token:
        return None

    token_data = verify_token(token, expected_type="access")
    if not token_data:
        return None

    try:
        # Look up user
        user = db_session.query(User).filter(
            User.id == UUID(token_data.user_id),
            User.is_active == True
        ).first()

        if not user:
            logger.warning(f"User not found: {token_data.user_id}")
            return None

        # Look up tenant
        tenant = db_session.query(Tenant).filter(
            Tenant.id == UUID(token_data.tenant_id),
            Tenant.is_active == True
        ).first()

        if not tenant:
            logger.warning(f"Tenant not found: {token_data.tenant_id}")
            return None

        return AuthContext(
            user=user,
            tenant=tenant,
            token_data=token_data
        )

    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        return None


def get_current_tenant(db_session) -> Optional[Tenant]:
    """
    Get just the current tenant from the request.
    Useful for tenant-scoped operations that don't need full user context.
    """
    auth_context = get_current_user(db_session)
    return auth_context.tenant if auth_context else None


def require_auth(f: Callable) -> Callable:
    """
    Decorator that requires authentication.
    Sets g.auth_context with the current user and tenant.

    Usage:
        @app.route('/api/protected')
        @require_auth
        def protected_endpoint():
            user = g.auth_context.user
            return jsonify({"user": user.email})
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        from ..database.database import get_db_context

        with get_db_context() as db:
            auth_context = get_current_user(db)

            if not auth_context or not auth_context.is_authenticated:
                return jsonify({
                    "error": "Authentication required",
                    "code": "AUTH_REQUIRED"
                }), 401

            # Store in Flask's g object for access in the route
            g.auth_context = auth_context
            g.current_user = auth_context.user
            g.current_tenant = auth_context.tenant
            g.db = db

            return f(*args, **kwargs)

    return decorated


def require_role(required_role: UserRole) -> Callable:
    """
    Decorator that requires a specific role or higher.

    Usage:
        @app.route('/api/admin-only')
        @require_role(UserRole.ADMIN)
        def admin_endpoint():
            return jsonify({"message": "You are an admin"})
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated(*args, **kwargs):
            from ..database.database import get_db_context

            with get_db_context() as db:
                auth_context = get_current_user(db)

                if not auth_context or not auth_context.is_authenticated:
                    return jsonify({
                        "error": "Authentication required",
                        "code": "AUTH_REQUIRED"
                    }), 401

                if not auth_context.user.has_permission(required_role):
                    return jsonify({
                        "error": f"Requires {required_role.value} role or higher",
                        "code": "INSUFFICIENT_PERMISSIONS"
                    }), 403

                g.auth_context = auth_context
                g.current_user = auth_context.user
                g.current_tenant = auth_context.tenant
                g.db = db

                return f(*args, **kwargs)

        return decorated
    return decorator


def optional_auth(f: Callable) -> Callable:
    """
    Decorator that optionally loads auth context if available.
    Route still works without authentication.

    Usage:
        @app.route('/api/public')
        @optional_auth
        def public_endpoint():
            if g.auth_context:
                return jsonify({"user": g.auth_context.user.email})
            return jsonify({"user": None})
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        from ..database.database import get_db_context

        with get_db_context() as db:
            auth_context = get_current_user(db)

            g.auth_context = auth_context
            g.current_user = auth_context.user if auth_context else None
            g.current_tenant = auth_context.tenant if auth_context else None
            g.db = db

            return f(*args, **kwargs)

    return decorated


# Convenience functions for use within routes

def get_request_ip() -> str:
    """Get the client IP address from the request"""
    # Check for proxy headers first
    if request.headers.get("X-Forwarded-For"):
        return request.headers.get("X-Forwarded-For").split(",")[0].strip()
    if request.headers.get("X-Real-IP"):
        return request.headers.get("X-Real-IP")
    return request.remote_addr or "unknown"


def get_request_user_agent() -> str:
    """Get the user agent from the request"""
    return request.headers.get("User-Agent", "")[:500]
