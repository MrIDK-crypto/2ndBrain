"""
Document API Routes
Protected endpoints for document operations
"""

import uuid
import logging
from flask import Blueprint, request, jsonify, g

from backend.auth.dependencies import require_auth, require_role
from backend.database.models import UserRole, DocumentStatus, ConnectorType
from backend.api.utils import (
    validate_request, error_response, success_response,
    get_document_service, parse_uuid, log_api_error, commit_or_rollback
)
from backend.api.schemas.documents import (
    DocumentCreate, DocumentUpdate, DocumentResponse,
    DocumentListResponse, DocumentStatsResponse,
    DocumentDecisionRequest, BulkStatusUpdateRequest
)
from backend.database.database import get_db

logger = logging.getLogger(__name__)

documents_bp = Blueprint('documents', __name__, url_prefix='/api/v1/documents')


@documents_bp.route('', methods=['GET'])
@require_auth
def list_documents():
    """
    List documents for the current tenant.
    Supports filtering and pagination.
    """
    try:
        # Parse query parameters
        page = max(1, int(request.args.get('page', 1)))
        per_page = min(100, max(1, int(request.args.get('per_page', 50))))
        status = request.args.get('status')
        source_type = request.args.get('source_type')
        classification = request.args.get('classification')
        search = request.args.get('search')

        # Convert enum values
        status_enum = DocumentStatus(status) if status else None
        source_enum = ConnectorType(source_type) if source_type else None

        service = get_document_service()
        documents, total = service.list_documents(
            status=status_enum,
            source_type=source_enum,
            classification=classification,
            search_query=search,
            page=page,
            per_page=per_page
        )

        return jsonify(
            DocumentListResponse.create(documents, total, page, per_page).model_dump()
        )

    except ValueError as e:
        return error_response("invalid_parameter", str(e))
    except Exception as e:
        log_api_error(e, "list_documents")
        return error_response("server_error", "Failed to list documents", 500)


@documents_bp.route('/<document_id>', methods=['GET'])
@require_auth
def get_document(document_id):
    """Get a specific document by ID."""
    try:
        doc_uuid = parse_uuid(document_id)
        if not doc_uuid:
            return error_response("invalid_id", "Invalid document ID format")

        service = get_document_service()
        document = service.get_by_id(doc_uuid)

        if not document:
            return error_response("not_found", "Document not found", 404)

        return jsonify(DocumentResponse.model_validate(document).model_dump())

    except Exception as e:
        log_api_error(e, "get_document")
        return error_response("server_error", "Failed to get document", 500)


@documents_bp.route('', methods=['POST'])
@require_auth
@require_role(UserRole.EDITOR)
@validate_request(DocumentCreate)
def create_document():
    """
    Create a new document.
    Requires editor role or higher.
    """
    try:
        data = g.validated_data
        service = get_document_service()

        # Check document limit
        can_add, current, limit = service.check_document_limit()
        if not can_add:
            return error_response(
                "limit_exceeded",
                f"Document limit reached ({current}/{limit})",
                403
            )

        db = next(get_db())
        with commit_or_rollback(db):
            document = service.create_document(
                source_type=ConnectorType(data.source_type.value),
                title=data.title,
                content=data.content,
                source_id=data.source_id,
                source_url=data.source_url,
                metadata=data.metadata  # Service converts to doc_metadata
            )

        return jsonify({
            "success": True,
            "message": "Document created",
            "document": DocumentResponse.model_validate(document).model_dump()
        }), 201

    except Exception as e:
        log_api_error(e, "create_document")
        return error_response("server_error", "Failed to create document", 500)


@documents_bp.route('/<document_id>', methods=['PUT', 'PATCH'])
@require_auth
@require_role(UserRole.EDITOR)
@validate_request(DocumentUpdate)
def update_document(document_id):
    """
    Update a document.
    Requires editor role or higher.
    """
    try:
        doc_uuid = parse_uuid(document_id)
        if not doc_uuid:
            return error_response("invalid_id", "Invalid document ID format")

        data = g.validated_data
        service = get_document_service()

        # Build update dict from non-None values
        updates = {}
        if data.title is not None:
            updates['title'] = data.title
        if data.content is not None:
            updates['content'] = data.content
        if data.status is not None:
            updates['status'] = DocumentStatus(data.status.value)
        if data.classification is not None:
            updates['classification'] = data.classification
        if data.metadata is not None:
            updates['doc_metadata'] = data.metadata

        if not updates:
            return error_response("no_changes", "No fields to update")

        db = next(get_db())
        with commit_or_rollback(db):
            document = service.update_document(doc_uuid, **updates)

        if not document:
            return error_response("not_found", "Document not found", 404)

        return jsonify({
            "success": True,
            "message": "Document updated",
            "document": DocumentResponse.model_validate(document).model_dump()
        })

    except Exception as e:
        log_api_error(e, "update_document")
        return error_response("server_error", "Failed to update document", 500)


@documents_bp.route('/<document_id>', methods=['DELETE'])
@require_auth
@require_role(UserRole.EDITOR)
def delete_document(document_id):
    """
    Soft delete a document.
    Requires editor role or higher.
    """
    try:
        doc_uuid = parse_uuid(document_id)
        if not doc_uuid:
            return error_response("invalid_id", "Invalid document ID format")

        service = get_document_service()
        db = next(get_db())

        with commit_or_rollback(db):
            deleted = service.soft_delete_document(doc_uuid)

        if not deleted:
            return error_response("not_found", "Document not found", 404)

        return success_response("Document deleted")

    except Exception as e:
        log_api_error(e, "delete_document")
        return error_response("server_error", "Failed to delete document", 500)


@documents_bp.route('/stats', methods=['GET'])
@require_auth
def get_stats():
    """Get document statistics for the tenant."""
    try:
        service = get_document_service()
        stats = service.get_stats()

        return jsonify(DocumentStatsResponse(**stats).model_dump())

    except Exception as e:
        log_api_error(e, "get_document_stats")
        return error_response("server_error", "Failed to get stats", 500)


@documents_bp.route('/review', methods=['GET'])
@require_auth
def get_review_queue():
    """
    Get documents pending review/classification.
    """
    try:
        page = max(1, int(request.args.get('page', 1)))
        per_page = min(100, max(1, int(request.args.get('per_page', 20))))

        service = get_document_service()
        documents, total = service.list_documents(
            status=DocumentStatus.PENDING,
            page=page,
            per_page=per_page
        )

        return jsonify(
            DocumentListResponse.create(documents, total, page, per_page).model_dump()
        )

    except Exception as e:
        log_api_error(e, "get_review_queue")
        return error_response("server_error", "Failed to get review queue", 500)


@documents_bp.route('/decision', methods=['POST'])
@require_auth
@require_role(UserRole.EDITOR)
@validate_request(DocumentDecisionRequest)
def make_decision():
    """
    Make a classification decision on a document.
    """
    try:
        data = g.validated_data
        service = get_document_service()
        db = next(get_db())

        with commit_or_rollback(db):
            document = service.update_document(
                data.document_id,
                classification=data.decision,
                status=DocumentStatus.INDEXED
            )

        if not document:
            return error_response("not_found", "Document not found", 404)

        return success_response(
            "Decision recorded",
            {"document_id": str(data.document_id), "decision": data.decision}
        )

    except Exception as e:
        log_api_error(e, "make_decision")
        return error_response("server_error", "Failed to record decision", 500)


@documents_bp.route('/bulk-status', methods=['POST'])
@require_auth
@require_role(UserRole.ADMIN)
@validate_request(BulkStatusUpdateRequest)
def bulk_update_status():
    """
    Bulk update document status.
    Admin only.
    """
    try:
        data = g.validated_data
        service = get_document_service()
        db = next(get_db())

        with commit_or_rollback(db):
            count = service.bulk_update_status(
                document_ids=data.document_ids,
                status=DocumentStatus(data.status.value),
                error_message=data.error_message
            )

        return success_response(
            f"Updated {count} documents",
            {"updated_count": count}
        )

    except Exception as e:
        log_api_error(e, "bulk_update_status")
        return error_response("server_error", "Failed to bulk update", 500)


@documents_bp.route('/limit', methods=['GET'])
@require_auth
def check_limit():
    """Check document limit for tenant."""
    try:
        service = get_document_service()
        can_add, current, limit = service.check_document_limit()

        return jsonify({
            "can_add_more": can_add,
            "current_count": current,
            "limit": limit,
            "remaining": max(0, limit - current)
        })

    except Exception as e:
        log_api_error(e, "check_limit")
        return error_response("server_error", "Failed to check limit", 500)
