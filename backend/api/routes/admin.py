"""
Admin API Routes
Protected endpoints for admin operations (user, tenant, audit management)
"""

import uuid
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, g

from backend.auth.dependencies import require_auth, require_role
from backend.database.models import User, Tenant, UserRole
from backend.database.database import get_db
from backend.api.utils import (
    validate_request, error_response, success_response,
    get_audit_service, parse_uuid, log_api_error, commit_or_rollback,
    get_document_service, get_connector_service
)
from backend.api.schemas.admin import (
    UserCreate, UserUpdate, UserResponse, UserListResponse,
    TenantResponse, TenantUpdate, TenantStatsResponse,
    AuditLogResponse, AuditLogListResponse, AuditLogFilters,
    ActivityStatsResponse, InviteUserRequest, PasswordResetRequest
)

logger = logging.getLogger(__name__)

admin_bp = Blueprint('admin', __name__, url_prefix='/api/v1/admin')


# ============================================================================
# User Management
# ============================================================================

@admin_bp.route('/users', methods=['GET'])
@require_auth
@require_role(UserRole.ADMIN)
def list_users():
    """
    List all users in the tenant.
    Admin only.
    """
    try:
        page = max(1, int(request.args.get('page', 1)))
        per_page = min(100, max(1, int(request.args.get('per_page', 50))))

        db = next(get_db())
        tenant_id = g.current_tenant.id

        query = db.query(User).filter(
            User.tenant_id == tenant_id,
            User.is_active == True
        )

        total = query.count()
        users = query.order_by(User.created_at.desc())\
            .offset((page - 1) * per_page)\
            .limit(per_page)\
            .all()

        return jsonify({
            "users": [UserResponse.model_validate(u).model_dump() for u in users],
            "total": total,
            "page": page,
            "per_page": per_page
        })

    except Exception as e:
        log_api_error(e, "list_users")
        return error_response("server_error", "Failed to list users", 500)


@admin_bp.route('/users/<user_id>', methods=['GET'])
@require_auth
@require_role(UserRole.ADMIN)
def get_user(user_id):
    """
    Get a specific user.
    Admin only.
    """
    try:
        user_uuid = parse_uuid(user_id)
        if not user_uuid:
            return error_response("invalid_id", "Invalid user ID format")

        db = next(get_db())
        tenant_id = g.current_tenant.id

        user = db.query(User).filter(
            User.id == user_uuid,
            User.tenant_id == tenant_id
        ).first()

        if not user:
            return error_response("not_found", "User not found", 404)

        return jsonify(UserResponse.model_validate(user).model_dump())

    except Exception as e:
        log_api_error(e, "get_user")
        return error_response("server_error", "Failed to get user", 500)


@admin_bp.route('/users', methods=['POST'])
@require_auth
@require_role(UserRole.ADMIN)
@validate_request(UserCreate)
def create_user():
    """
    Create a new user in the tenant.
    Admin only.
    """
    try:
        data = g.validated_data
        db = next(get_db())
        tenant_id = g.current_tenant.id

        # Check if email already exists in tenant
        existing = db.query(User).filter(
            User.tenant_id == tenant_id,
            User.email == data.email
        ).first()

        if existing:
            return error_response("email_exists", "A user with this email already exists")

        # Only owner can create admins
        current_user_role = g.current_user.role
        if data.role == UserRole.ADMIN and current_user_role != UserRole.OWNER:
            return error_response("forbidden", "Only owners can create admin users", 403)

        # Cannot create owner role
        if data.role == UserRole.OWNER:
            return error_response("forbidden", "Cannot create owner users", 403)

        with commit_or_rollback(db):
            user = User(
                tenant_id=tenant_id,
                email=data.email,
                name=data.name,
                role=data.role
            )
            if data.password:
                user.set_password(data.password)

            db.add(user)
            db.flush()

            # Audit log
            audit_service = get_audit_service()
            audit_service.log_action(
                action="user.create",
                resource_type="user",
                resource_id=str(user.id),
                details={"email": data.email, "role": data.role.value}
            )

        return jsonify({
            "success": True,
            "message": "User created",
            "user": UserResponse.model_validate(user).model_dump()
        }), 201

    except Exception as e:
        log_api_error(e, "create_user")
        return error_response("server_error", "Failed to create user", 500)


@admin_bp.route('/users/<user_id>', methods=['PUT', 'PATCH'])
@require_auth
@require_role(UserRole.ADMIN)
@validate_request(UserUpdate)
def update_user(user_id):
    """
    Update a user.
    Admin only.
    """
    try:
        user_uuid = parse_uuid(user_id)
        if not user_uuid:
            return error_response("invalid_id", "Invalid user ID format")

        data = g.validated_data
        db = next(get_db())
        tenant_id = g.current_tenant.id

        user = db.query(User).filter(
            User.id == user_uuid,
            User.tenant_id == tenant_id
        ).first()

        if not user:
            return error_response("not_found", "User not found", 404)

        # Cannot modify owner
        if user.role == UserRole.OWNER:
            return error_response("forbidden", "Cannot modify owner user", 403)

        # Only owner can set admin role
        current_user_role = g.current_user.role
        if data.role == UserRole.ADMIN and current_user_role != UserRole.OWNER:
            return error_response("forbidden", "Only owners can assign admin role", 403)

        with commit_or_rollback(db):
            if data.name is not None:
                user.name = data.name
            if data.role is not None:
                user.role = data.role
            if data.is_active is not None:
                user.is_active = data.is_active

            # Audit log
            audit_service = get_audit_service()
            audit_service.log_action(
                action="user.update",
                resource_type="user",
                resource_id=str(user_uuid),
                details={"changes": data.model_dump(exclude_none=True)}
            )

        return jsonify({
            "success": True,
            "message": "User updated",
            "user": UserResponse.model_validate(user).model_dump()
        })

    except Exception as e:
        log_api_error(e, "update_user")
        return error_response("server_error", "Failed to update user", 500)


@admin_bp.route('/users/<user_id>', methods=['DELETE'])
@require_auth
@require_role(UserRole.ADMIN)
def deactivate_user(user_id):
    """
    Deactivate a user (soft delete).
    Admin only.
    """
    try:
        user_uuid = parse_uuid(user_id)
        if not user_uuid:
            return error_response("invalid_id", "Invalid user ID format")

        db = next(get_db())
        tenant_id = g.current_tenant.id

        user = db.query(User).filter(
            User.id == user_uuid,
            User.tenant_id == tenant_id
        ).first()

        if not user:
            return error_response("not_found", "User not found", 404)

        # Cannot deactivate owner
        if user.role == UserRole.OWNER:
            return error_response("forbidden", "Cannot deactivate owner", 403)

        # Cannot deactivate self
        if user.id == g.current_user.id:
            return error_response("forbidden", "Cannot deactivate yourself", 403)

        with commit_or_rollback(db):
            user.is_active = False

            audit_service = get_audit_service()
            audit_service.log_action(
                action="user.deactivate",
                resource_type="user",
                resource_id=str(user_uuid),
                details={"email": user.email}
            )

        return success_response("User deactivated")

    except Exception as e:
        log_api_error(e, "deactivate_user")
        return error_response("server_error", "Failed to deactivate user", 500)


# ============================================================================
# Tenant Management
# ============================================================================

@admin_bp.route('/tenant', methods=['GET'])
@require_auth
@require_role(UserRole.ADMIN)
def get_tenant():
    """
    Get current tenant information.
    Admin only.
    """
    try:
        db = next(get_db())
        tenant_id = g.current_tenant.id

        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            return error_response("not_found", "Tenant not found", 404)

        return jsonify(TenantResponse.model_validate(tenant).model_dump())

    except Exception as e:
        log_api_error(e, "get_tenant")
        return error_response("server_error", "Failed to get tenant", 500)


@admin_bp.route('/tenant', methods=['PUT', 'PATCH'])
@require_auth
@require_role(UserRole.OWNER)
@validate_request(TenantUpdate)
def update_tenant():
    """
    Update tenant settings.
    Owner only.
    """
    try:
        data = g.validated_data
        db = next(get_db())
        tenant_id = g.current_tenant.id

        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            return error_response("not_found", "Tenant not found", 404)

        with commit_or_rollback(db):
            if data.name is not None:
                tenant.name = data.name
            if data.settings is not None:
                tenant.settings = data.settings

            audit_service = get_audit_service()
            audit_service.log_action(
                action="tenant.update",
                resource_type="tenant",
                resource_id=str(tenant_id),
                details={"changes": data.model_dump(exclude_none=True)}
            )

        return jsonify({
            "success": True,
            "message": "Tenant updated",
            "tenant": TenantResponse.model_validate(tenant).model_dump()
        })

    except Exception as e:
        log_api_error(e, "update_tenant")
        return error_response("server_error", "Failed to update tenant", 500)


@admin_bp.route('/tenant/stats', methods=['GET'])
@require_auth
@require_role(UserRole.ADMIN)
def get_tenant_stats():
    """
    Get tenant statistics.
    Admin only.
    """
    try:
        db = next(get_db())
        tenant_id = g.current_tenant.id

        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            return error_response("not_found", "Tenant not found", 404)

        # Get counts
        user_count = db.query(User).filter(
            User.tenant_id == tenant_id,
            User.is_active == True
        ).count()

        doc_service = get_document_service()
        doc_stats = doc_service.get_stats()

        conn_service = get_connector_service()
        connector_status = conn_service.get_status_summary()

        return jsonify({
            "tenant": TenantResponse.model_validate(tenant).model_dump(),
            "user_count": user_count,
            "document_count": doc_stats["total"],
            "connector_count": connector_status["total"],
            "document_limit": tenant.document_limit,
            "documents_remaining": max(0, tenant.document_limit - doc_stats["total"]),
            "storage_used_mb": 0  # TODO: Calculate actual storage
        })

    except Exception as e:
        log_api_error(e, "get_tenant_stats")
        return error_response("server_error", "Failed to get stats", 500)


# ============================================================================
# Audit Logs
# ============================================================================

@admin_bp.route('/audit-logs', methods=['GET'])
@require_auth
@require_role(UserRole.ADMIN)
def list_audit_logs():
    """
    List audit logs.
    Admin only.
    """
    try:
        page = max(1, int(request.args.get('page', 1)))
        per_page = min(100, max(1, int(request.args.get('per_page', 50))))
        action = request.args.get('action')
        resource_type = request.args.get('resource_type')
        user_id = request.args.get('user_id')

        user_uuid = parse_uuid(user_id) if user_id else None

        audit_service = get_audit_service()
        logs, total = audit_service.list_logs(
            action=action,
            resource_type=resource_type,
            user_id=user_uuid,
            page=page,
            per_page=per_page
        )

        # Enrich with user email
        db = next(get_db())
        enriched_logs = []
        for log in logs:
            log_dict = {
                "id": log.id,
                "user_id": log.user_id,
                "user_email": None,
                "action": log.action,
                "resource_type": log.resource_type,
                "resource_id": log.resource_id,
                "details": log.details,
                "ip_address": log.ip_address,
                "created_at": log.created_at
            }
            if log.user_id:
                user = db.query(User).filter(User.id == log.user_id).first()
                if user:
                    log_dict["user_email"] = user.email
            enriched_logs.append(log_dict)

        return jsonify({
            "logs": enriched_logs,
            "total": total,
            "page": page,
            "per_page": per_page
        })

    except Exception as e:
        log_api_error(e, "list_audit_logs")
        return error_response("server_error", "Failed to list audit logs", 500)


@admin_bp.route('/audit-logs/activity', methods=['GET'])
@require_auth
@require_role(UserRole.ADMIN)
def get_activity_stats():
    """
    Get activity statistics.
    Admin only.
    """
    try:
        days = min(90, max(1, int(request.args.get('days', 7))))

        audit_service = get_audit_service()
        stats = audit_service.get_activity_stats(days=days)

        return jsonify(stats)

    except Exception as e:
        log_api_error(e, "get_activity_stats")
        return error_response("server_error", "Failed to get activity stats", 500)


# ============================================================================
# Admin Dashboard
# ============================================================================

@admin_bp.route('/dashboard', methods=['GET'])
@require_auth
@require_role(UserRole.ADMIN)
def get_dashboard():
    """
    Get admin dashboard data.
    Admin only.
    """
    try:
        db = next(get_db())
        tenant_id = g.current_tenant.id

        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()

        # User count
        user_count = db.query(User).filter(
            User.tenant_id == tenant_id,
            User.is_active == True
        ).count()

        # Document stats
        doc_service = get_document_service()
        doc_stats = doc_service.get_stats()

        # Connector status
        conn_service = get_connector_service()
        connector_status = conn_service.get_status_summary()

        # Recent activity
        audit_service = get_audit_service()
        recent_logs, _ = audit_service.list_logs(page=1, per_page=10)

        return jsonify({
            "tenant": TenantResponse.model_validate(tenant).model_dump(),
            "stats": {
                "user_count": user_count,
                "document_count": doc_stats["total"],
                "indexed_count": doc_stats["indexed_count"],
                "pending_count": doc_stats["pending_count"],
                "connector_count": connector_status["total"]
            },
            "connector_status": connector_status,
            "recent_activity": [
                {
                    "action": log.action,
                    "created_at": log.created_at.isoformat()
                }
                for log in recent_logs
            ]
        })

    except Exception as e:
        log_api_error(e, "get_dashboard")
        return error_response("server_error", "Failed to get dashboard", 500)
