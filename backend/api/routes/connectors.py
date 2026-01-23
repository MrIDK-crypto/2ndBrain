"""
Connector API Routes
Protected endpoints for connector operations
"""

import logging
from flask import Blueprint, request, jsonify, g

from backend.auth.dependencies import require_auth, require_role
from backend.database.models import UserRole, ConnectorType
from backend.api.utils import (
    validate_request, error_response, success_response,
    get_connector_service, parse_uuid, log_api_error, commit_or_rollback
)
from backend.api.schemas.connectors import (
    ConnectorCreate, ConnectorResponse, ConnectorListResponse,
    ConnectorStatusResponse, ConnectorSettingsUpdate, ConnectorSyncRequest
)
from backend.database.database import get_db

logger = logging.getLogger(__name__)

connectors_bp = Blueprint('connectors', __name__, url_prefix='/api/v1/connectors')


@connectors_bp.route('', methods=['GET'])
@require_auth
def list_connectors():
    """
    List all connectors for the tenant.
    """
    try:
        service = get_connector_service()
        connectors = service.list_connectors()
        status = service.get_status_summary()

        return jsonify({
            "connectors": [
                ConnectorResponse.model_validate(c).model_dump()
                for c in connectors
            ],
            "available_types": status["available_types"],
            "connected_types": status["connected_types"]
        })

    except Exception as e:
        log_api_error(e, "list_connectors")
        return error_response("server_error", "Failed to list connectors", 500)


@connectors_bp.route('/status', methods=['GET'])
@require_auth
def get_connector_status():
    """
    Get status summary of all connectors.
    """
    try:
        service = get_connector_service()
        status = service.get_status_summary()

        return jsonify(status)

    except Exception as e:
        log_api_error(e, "get_connector_status")
        return error_response("server_error", "Failed to get status", 500)


@connectors_bp.route('/<connector_id>', methods=['GET'])
@require_auth
def get_connector(connector_id):
    """
    Get a specific connector by ID.
    """
    try:
        conn_uuid = parse_uuid(connector_id)
        if not conn_uuid:
            return error_response("invalid_id", "Invalid connector ID format")

        service = get_connector_service()
        connector = service.get_by_id(conn_uuid)

        if not connector:
            return error_response("not_found", "Connector not found", 404)

        return jsonify(ConnectorResponse.model_validate(connector).model_dump())

    except Exception as e:
        log_api_error(e, "get_connector")
        return error_response("server_error", "Failed to get connector", 500)


@connectors_bp.route('', methods=['POST'])
@require_auth
@require_role(UserRole.ADMIN)
@validate_request(ConnectorCreate)
def create_connector():
    """
    Create a new connector.
    Admin only - connectors handle OAuth credentials.
    """
    try:
        data = g.validated_data
        service = get_connector_service()

        # Check if connector of this type already exists
        existing = service.get_by_type(ConnectorType(data.connector_type.value))
        if existing:
            return error_response(
                "already_exists",
                f"Connector of type {data.connector_type.value} already exists"
            )

        db = next(get_db())
        with commit_or_rollback(db):
            connector = service.create_connector(
                connector_type=ConnectorType(data.connector_type.value),
                name=data.name,
                settings=data.settings
            )

        return jsonify({
            "success": True,
            "message": "Connector created",
            "connector": ConnectorResponse.model_validate(connector).model_dump()
        }), 201

    except Exception as e:
        log_api_error(e, "create_connector")
        return error_response("server_error", "Failed to create connector", 500)


@connectors_bp.route('/<connector_id>', methods=['PUT', 'PATCH'])
@require_auth
@require_role(UserRole.ADMIN)
@validate_request(ConnectorSettingsUpdate)
def update_connector(connector_id):
    """
    Update connector settings.
    Admin only.
    """
    try:
        conn_uuid = parse_uuid(connector_id)
        if not conn_uuid:
            return error_response("invalid_id", "Invalid connector ID format")

        data = g.validated_data
        service = get_connector_service()

        connector = service.get_by_id(conn_uuid)
        if not connector:
            return error_response("not_found", "Connector not found", 404)

        db = next(get_db())
        with commit_or_rollback(db):
            # Update fields
            if data.name is not None:
                connector.name = data.name
            if data.sync_enabled is not None:
                service.toggle_sync(conn_uuid, data.sync_enabled)
            if data.settings is not None:
                connector.settings = data.settings

        return jsonify({
            "success": True,
            "message": "Connector updated",
            "connector": ConnectorResponse.model_validate(connector).model_dump()
        })

    except Exception as e:
        log_api_error(e, "update_connector")
        return error_response("server_error", "Failed to update connector", 500)


@connectors_bp.route('/<connector_id>/sync', methods=['POST'])
@require_auth
@require_role(UserRole.EDITOR)
@validate_request(ConnectorSyncRequest)
def trigger_sync(connector_id):
    """
    Trigger a sync for a connector.
    Requires editor role or higher.
    """
    try:
        conn_uuid = parse_uuid(connector_id)
        if not conn_uuid:
            return error_response("invalid_id", "Invalid connector ID format")

        data = g.validated_data
        service = get_connector_service()

        connector = service.get_by_id(conn_uuid)
        if not connector:
            return error_response("not_found", "Connector not found", 404)

        if not connector.sync_enabled:
            return error_response("sync_disabled", "Sync is disabled for this connector")

        # TODO: Trigger actual sync job
        # For now, update status to indicate sync was requested
        db = next(get_db())
        with commit_or_rollback(db):
            service.update_sync_status(
                conn_uuid,
                status="requested",
                error=None
            )

        return success_response(
            "Sync triggered",
            {
                "connector_id": str(conn_uuid),
                "full_sync": data.full_sync,
                "message": "Sync job queued - integration pending"
            }
        )

    except Exception as e:
        log_api_error(e, "trigger_sync")
        return error_response("server_error", "Failed to trigger sync", 500)


@connectors_bp.route('/<connector_id>/disconnect', methods=['POST'])
@require_auth
@require_role(UserRole.ADMIN)
def disconnect_connector(connector_id):
    """
    Disconnect (deactivate) a connector.
    Admin only - this clears credentials.
    """
    try:
        conn_uuid = parse_uuid(connector_id)
        if not conn_uuid:
            return error_response("invalid_id", "Invalid connector ID format")

        service = get_connector_service()
        db = next(get_db())

        with commit_or_rollback(db):
            disconnected = service.disconnect(conn_uuid)

        if not disconnected:
            return error_response("not_found", "Connector not found", 404)

        return success_response("Connector disconnected")

    except Exception as e:
        log_api_error(e, "disconnect_connector")
        return error_response("server_error", "Failed to disconnect connector", 500)


# OAuth routes for specific connectors
@connectors_bp.route('/gmail/auth', methods=['GET'])
@require_auth
@require_role(UserRole.ADMIN)
def gmail_oauth_start():
    """
    Initiate Gmail OAuth flow.
    Returns URL to redirect user for authorization.
    """
    try:
        # TODO: Integrate with Gmail OAuth service
        return jsonify({
            "auth_url": None,
            "state": None,
            "message": "Gmail OAuth integration pending"
        })

    except Exception as e:
        log_api_error(e, "gmail_oauth_start")
        return error_response("server_error", "Failed to start OAuth flow", 500)


@connectors_bp.route('/gmail/callback', methods=['GET', 'POST'])
@require_auth
def gmail_oauth_callback():
    """
    Handle Gmail OAuth callback.
    """
    try:
        code = request.args.get('code') or request.json.get('code')
        state = request.args.get('state') or request.json.get('state')

        if not code:
            return error_response("missing_code", "OAuth code required")

        # TODO: Exchange code for tokens and store
        return success_response(
            "Gmail connected",
            {"message": "OAuth callback integration pending"}
        )

    except Exception as e:
        log_api_error(e, "gmail_oauth_callback")
        return error_response("server_error", "OAuth callback failed", 500)


@connectors_bp.route('/slack/auth', methods=['GET'])
@require_auth
@require_role(UserRole.ADMIN)
def slack_oauth_start():
    """
    Initiate Slack OAuth flow.
    """
    try:
        return jsonify({
            "auth_url": None,
            "state": None,
            "message": "Slack OAuth integration pending"
        })

    except Exception as e:
        log_api_error(e, "slack_oauth_start")
        return error_response("server_error", "Failed to start OAuth flow", 500)


@connectors_bp.route('/github/auth', methods=['GET'])
@require_auth
@require_role(UserRole.ADMIN)
def github_oauth_start():
    """
    Initiate GitHub OAuth flow.
    """
    try:
        return jsonify({
            "auth_url": None,
            "state": None,
            "message": "GitHub OAuth integration pending"
        })

    except Exception as e:
        log_api_error(e, "github_oauth_start")
        return error_response("server_error", "Failed to start OAuth flow", 500)
