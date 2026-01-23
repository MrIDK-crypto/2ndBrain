"""
Authentication API Routes
Handles user registration, login, and token management
"""

import logging
from datetime import datetime
from typing import Optional

from flask import Blueprint, request, jsonify
from pydantic import BaseModel, EmailStr, Field, validator
import re

from .jwt import create_token_pair, verify_token, create_access_token
from .dependencies import get_request_ip, get_request_user_agent
from ..database.database import get_db_context
from ..database.models import User, Tenant, UserRole, create_audit_log

logger = logging.getLogger(__name__)

# Create Blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')


# ============================================================================
# Pydantic Models for Request Validation
# ============================================================================

class RegisterRequest(BaseModel):
    """User registration request"""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    name: Optional[str] = Field(None, max_length=255)
    tenant_name: str = Field(..., min_length=2, max_length=255)
    tenant_slug: str = Field(..., min_length=2, max_length=100)

    @validator('password')
    def password_strength(cls, v):
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        return v

    @validator('tenant_slug')
    def validate_slug(cls, v):
        v = v.lower().strip()
        if not re.match(r'^[a-z0-9][a-z0-9-]*[a-z0-9]$', v) or len(v) < 3:
            raise ValueError('Slug must be lowercase alphanumeric with hyphens, min 3 chars')
        return v


class LoginRequest(BaseModel):
    """User login request"""
    email: EmailStr
    password: str
    tenant_slug: str


class RefreshRequest(BaseModel):
    """Token refresh request"""
    refresh_token: str


class InviteUserRequest(BaseModel):
    """Invite a user to tenant"""
    email: EmailStr
    name: Optional[str] = None
    role: UserRole = UserRole.VIEWER


# ============================================================================
# Routes
# ============================================================================

@auth_bp.route('/register', methods=['POST'])
def register():
    """
    Register a new user and create their tenant.
    This is for new organizations signing up.
    """
    try:
        data = request.get_json()
        req = RegisterRequest(**data)
    except Exception as e:
        return jsonify({"error": str(e), "code": "VALIDATION_ERROR"}), 400

    with get_db_context() as db:
        # Check if tenant slug already exists
        existing_tenant = db.query(Tenant).filter(
            Tenant.slug == req.tenant_slug
        ).first()

        if existing_tenant:
            return jsonify({
                "error": "Organization slug already taken",
                "code": "TENANT_EXISTS"
            }), 409

        # Check if email already exists in any tenant
        existing_user = db.query(User).filter(
            User.email == req.email
        ).first()

        if existing_user:
            return jsonify({
                "error": "Email already registered",
                "code": "EMAIL_EXISTS"
            }), 409

        # Create tenant
        tenant = Tenant(
            name=req.tenant_name,
            slug=req.tenant_slug,
            plan="free"
        )
        db.add(tenant)
        db.flush()  # Get tenant.id

        # Create user as owner
        user = User(
            tenant_id=tenant.id,
            email=req.email,
            name=req.name,
            role=UserRole.OWNER,
            email_verified=False  # Would send verification email in production
        )
        user.set_password(req.password)
        db.add(user)
        db.flush()  # Get user.id

        # Create audit log
        create_audit_log(
            session=db,
            tenant_id=tenant.id,
            user_id=user.id,
            action="user.register",
            resource_type="user",
            resource_id=str(user.id),
            details={"email": user.email, "tenant": tenant.slug},
            ip_address=get_request_ip(),
            user_agent=get_request_user_agent()
        )

        # Generate tokens
        tokens = create_token_pair(
            user_id=str(user.id),
            tenant_id=str(tenant.id),
            email=user.email,
            role=user.role.value
        )

        db.commit()

        logger.info(f"New user registered: {user.email} in tenant {tenant.slug}")

        return jsonify({
            "success": True,
            "user": {
                "id": str(user.id),
                "email": user.email,
                "name": user.name,
                "role": user.role.value
            },
            "tenant": {
                "id": str(tenant.id),
                "name": tenant.name,
                "slug": tenant.slug
            },
            **tokens
        }), 201


@auth_bp.route('/login', methods=['POST'])
def login():
    """
    Login with email and password.
    Requires tenant_slug to identify which organization.
    """
    try:
        data = request.get_json()
        req = LoginRequest(**data)
    except Exception as e:
        return jsonify({"error": str(e), "code": "VALIDATION_ERROR"}), 400

    with get_db_context() as db:
        # Find tenant
        tenant = db.query(Tenant).filter(
            Tenant.slug == req.tenant_slug,
            Tenant.is_active == True
        ).first()

        if not tenant:
            return jsonify({
                "error": "Organization not found",
                "code": "TENANT_NOT_FOUND"
            }), 404

        # Find user
        user = db.query(User).filter(
            User.tenant_id == tenant.id,
            User.email == req.email,
            User.is_active == True
        ).first()

        if not user or not user.verify_password(req.password):
            # Log failed attempt
            create_audit_log(
                session=db,
                tenant_id=tenant.id,
                action="user.login_failed",
                details={"email": req.email, "reason": "invalid_credentials"},
                ip_address=get_request_ip(),
                user_agent=get_request_user_agent()
            )
            db.commit()

            return jsonify({
                "error": "Invalid email or password",
                "code": "INVALID_CREDENTIALS"
            }), 401

        # Update last login
        user.last_login_at = datetime.utcnow()

        # Create audit log
        create_audit_log(
            session=db,
            tenant_id=tenant.id,
            user_id=user.id,
            action="user.login",
            resource_type="user",
            resource_id=str(user.id),
            ip_address=get_request_ip(),
            user_agent=get_request_user_agent()
        )

        # Generate tokens
        tokens = create_token_pair(
            user_id=str(user.id),
            tenant_id=str(tenant.id),
            email=user.email,
            role=user.role.value
        )

        db.commit()

        logger.info(f"User logged in: {user.email}")

        return jsonify({
            "success": True,
            "user": {
                "id": str(user.id),
                "email": user.email,
                "name": user.name,
                "role": user.role.value
            },
            "tenant": {
                "id": str(tenant.id),
                "name": tenant.name,
                "slug": tenant.slug
            },
            **tokens
        })


@auth_bp.route('/refresh', methods=['POST'])
def refresh():
    """
    Refresh access token using refresh token.
    """
    try:
        data = request.get_json()
        req = RefreshRequest(**data)
    except Exception as e:
        return jsonify({"error": str(e), "code": "VALIDATION_ERROR"}), 400

    # Verify refresh token
    token_data = verify_token(req.refresh_token, expected_type="refresh")
    if not token_data:
        return jsonify({
            "error": "Invalid or expired refresh token",
            "code": "INVALID_TOKEN"
        }), 401

    with get_db_context() as db:
        # Look up user to get current data
        from uuid import UUID
        user = db.query(User).filter(
            User.id == UUID(token_data.user_id),
            User.is_active == True
        ).first()

        if not user:
            return jsonify({
                "error": "User not found or inactive",
                "code": "USER_NOT_FOUND"
            }), 401

        tenant = db.query(Tenant).filter(
            Tenant.id == UUID(token_data.tenant_id),
            Tenant.is_active == True
        ).first()

        if not tenant:
            return jsonify({
                "error": "Organization not found or inactive",
                "code": "TENANT_NOT_FOUND"
            }), 401

        # Generate new tokens
        tokens = create_token_pair(
            user_id=str(user.id),
            tenant_id=str(tenant.id),
            email=user.email,
            role=user.role.value
        )

        return jsonify({
            "success": True,
            **tokens
        })


@auth_bp.route('/me', methods=['GET'])
def get_me():
    """
    Get current user info from token.
    """
    from .dependencies import extract_token_from_request

    token = extract_token_from_request()
    if not token:
        return jsonify({
            "error": "No token provided",
            "code": "NO_TOKEN"
        }), 401

    token_data = verify_token(token, expected_type="access")
    if not token_data:
        return jsonify({
            "error": "Invalid or expired token",
            "code": "INVALID_TOKEN"
        }), 401

    with get_db_context() as db:
        from uuid import UUID
        user = db.query(User).filter(
            User.id == UUID(token_data.user_id)
        ).first()

        if not user:
            return jsonify({
                "error": "User not found",
                "code": "USER_NOT_FOUND"
            }), 404

        tenant = db.query(Tenant).filter(
            Tenant.id == UUID(token_data.tenant_id)
        ).first()

        return jsonify({
            "user": {
                "id": str(user.id),
                "email": user.email,
                "name": user.name,
                "role": user.role.value,
                "email_verified": user.email_verified,
                "created_at": user.created_at.isoformat(),
                "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None
            },
            "tenant": {
                "id": str(tenant.id),
                "name": tenant.name,
                "slug": tenant.slug,
                "plan": tenant.plan
            } if tenant else None
        })


@auth_bp.route('/logout', methods=['POST'])
def logout():
    """
    Logout - client should discard tokens.
    In production, you might blacklist the refresh token.
    """
    # In a stateless JWT setup, logout is handled client-side
    # by discarding the tokens. Server-side, we just log the event.

    from .dependencies import extract_token_from_request

    token = extract_token_from_request()
    if token:
        token_data = verify_token(token, expected_type="access")
        if token_data:
            with get_db_context() as db:
                from uuid import UUID
                create_audit_log(
                    session=db,
                    tenant_id=UUID(token_data.tenant_id),
                    user_id=UUID(token_data.user_id),
                    action="user.logout",
                    ip_address=get_request_ip(),
                    user_agent=get_request_user_agent()
                )
                db.commit()

    return jsonify({"success": True, "message": "Logged out"})


# ============================================================================
# Health check for auth service
# ============================================================================

@auth_bp.route('/health', methods=['GET'])
def health():
    """Auth service health check"""
    from ..database.database import check_db_connection

    db_healthy = check_db_connection()

    return jsonify({
        "service": "auth",
        "status": "healthy" if db_healthy else "degraded",
        "database": "connected" if db_healthy else "disconnected"
    }), 200 if db_healthy else 503
