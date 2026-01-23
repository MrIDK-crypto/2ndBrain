"""
API Test Script
Tests the 2ndBrain API endpoints
"""

import os
import sys
sys.path.insert(0, '.')

# Load environment variables FIRST before any imports
from dotenv import load_dotenv
load_dotenv(override=True)

# Verify SQLite is being used
print(f"DATABASE_URL: {os.getenv('DATABASE_URL')}")

import json

# Now initialize database tables
from backend.database.database import init_db, DATABASE_URL
print(f"Loaded DATABASE_URL: {DATABASE_URL}")
init_db()

from backend.api.app import app

# Create test client
client = app.test_client()

def test_endpoint(method, url, data=None, headers=None, expected_status=None):
    """Helper to test an endpoint"""
    headers = headers or {}
    if data:
        headers['Content-Type'] = 'application/json'
        response = getattr(client, method.lower())(url, data=json.dumps(data), headers=headers)
    else:
        response = getattr(client, method.lower())(url, headers=headers)

    status = response.status_code
    try:
        body = response.get_json()
    except:
        body = response.data.decode()[:200]

    status_icon = '✓' if (expected_status is None or status == expected_status) else '✗'
    print(f"{status_icon} {method} {url} -> {status}")
    if status >= 400:
        print(f"  Response: {body}")
    return response

print("=" * 60)
print("2ndBrain API Tests")
print("=" * 60)

# ============================================================================
# Health & Version
# ============================================================================
print("\n--- Health & Version ---")
test_endpoint('GET', '/health', expected_status=200)
test_endpoint('GET', '/api/version', expected_status=200)

# ============================================================================
# Auth endpoints (no auth required)
# ============================================================================
print("\n--- Auth Endpoints ---")

# Register a new tenant/user (password needs uppercase + number)
register_data = {
    "email": "test@example.com",
    "password": "TestPassword123",
    "name": "Test User",
    "tenant_name": "Test Company",
    "tenant_slug": "test-company"
}
response = test_endpoint('POST', '/api/auth/register', data=register_data)

# Try to login
login_data = {
    "email": "test@example.com",
    "password": "TestPassword123",
    "tenant_slug": "test-company"
}
response = test_endpoint('POST', '/api/auth/login', data=login_data)

# Get tokens from login response
try:
    tokens = response.get_json()
    access_token = tokens.get('access_token', '')
    refresh_token = tokens.get('refresh_token', '')
    print(f"  Got access token: {access_token[:20]}..." if access_token else "  No token received")
except:
    access_token = ''
    refresh_token = ''

# ============================================================================
# Protected endpoints without auth (should fail with 401)
# ============================================================================
print("\n--- Protected Endpoints (No Auth - Expect 401) ---")
test_endpoint('GET', '/api/v1/documents', expected_status=401)
test_endpoint('GET', '/api/v1/connectors', expected_status=401)
test_endpoint('POST', '/api/v1/search', data={"query": "test"}, expected_status=401)
test_endpoint('GET', '/api/v1/admin/users', expected_status=401)

# ============================================================================
# Protected endpoints with auth
# ============================================================================
if access_token:
    auth_headers = {'Authorization': f'Bearer {access_token}'}

    print("\n--- Protected Endpoints (With Auth) ---")

    # Documents
    print("\n  Documents:")
    test_endpoint('GET', '/api/v1/documents', headers=auth_headers, expected_status=200)
    test_endpoint('GET', '/api/v1/documents/stats', headers=auth_headers, expected_status=200)
    test_endpoint('GET', '/api/v1/documents/limit', headers=auth_headers, expected_status=200)
    test_endpoint('GET', '/api/v1/documents/review', headers=auth_headers, expected_status=200)

    # Create a document
    doc_data = {
        "title": "Test Document",
        "content": "This is test content",
        "source_type": "upload"
    }
    response = test_endpoint('POST', '/api/v1/documents', data=doc_data, headers=auth_headers, expected_status=201)

    # Connectors
    print("\n  Connectors:")
    test_endpoint('GET', '/api/v1/connectors', headers=auth_headers, expected_status=200)
    test_endpoint('GET', '/api/v1/connectors/status', headers=auth_headers, expected_status=200)

    # Search
    print("\n  Search:")
    search_data = {"query": "test query", "top_k": 5}
    test_endpoint('POST', '/api/v1/search', data=search_data, headers=auth_headers, expected_status=200)

    question_data = {"question": "What is the main topic?"}
    test_endpoint('POST', '/api/v1/search/question', data=question_data, headers=auth_headers, expected_status=200)

    # Admin
    print("\n  Admin:")
    test_endpoint('GET', '/api/v1/admin/users', headers=auth_headers, expected_status=200)
    test_endpoint('GET', '/api/v1/admin/tenant', headers=auth_headers, expected_status=200)
    test_endpoint('GET', '/api/v1/admin/tenant/stats', headers=auth_headers, expected_status=200)
    test_endpoint('GET', '/api/v1/admin/dashboard', headers=auth_headers, expected_status=200)
    test_endpoint('GET', '/api/v1/admin/audit-logs', headers=auth_headers, expected_status=200)

    # Get current user
    print("\n  User Info:")
    test_endpoint('GET', '/api/auth/me', headers=auth_headers, expected_status=200)

    # Refresh token
    if refresh_token:
        print("\n  Token Refresh:")
        test_endpoint('POST', '/api/auth/refresh', data={"refresh_token": refresh_token}, expected_status=200)

else:
    print("\n⚠ Skipping authenticated tests - no token available")
    print("  (This is expected if database is not running)")

print("\n" + "=" * 60)
print("Tests Complete")
print("=" * 60)
