"""
JWT Token Management
Handles creation and verification of access and refresh tokens
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass

from jose import jwt, JWTError, ExpiredSignatureError

logger = logging.getLogger(__name__)

# Configuration from environment
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
if not SECRET_KEY:
    # Generate a warning but allow startup for development
    import secrets
    SECRET_KEY = secrets.token_urlsafe(32)
    logger.warning(
        "JWT_SECRET_KEY not set! Using random key. "
        "This will invalidate all tokens on restart. "
        "Set JWT_SECRET_KEY in production!"
    )

ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))


@dataclass
class TokenData:
    """Decoded token data"""
    user_id: str
    tenant_id: str
    email: str
    role: str
    token_type: str  # 'access' or 'refresh'
    exp: datetime
    iat: datetime


def create_access_token(
    user_id: str,
    tenant_id: str,
    email: str,
    role: str,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.

    Args:
        user_id: User's UUID as string
        tenant_id: Tenant's UUID as string
        email: User's email
        role: User's role (owner, admin, editor, viewer)
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token string
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    now = datetime.utcnow()
    expire = now + expires_delta

    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "email": email,
        "role": role,
        "type": "access",
        "iat": now,
        "exp": expire
    }

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(
    user_id: str,
    tenant_id: str,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT refresh token.
    Refresh tokens have longer expiry and fewer claims.

    Args:
        user_id: User's UUID as string
        tenant_id: Tenant's UUID as string
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT refresh token string
    """
    if expires_delta is None:
        expires_delta = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    now = datetime.utcnow()
    expire = now + expires_delta

    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "type": "refresh",
        "iat": now,
        "exp": expire
    }

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode a JWT token without verification.
    Useful for inspecting token contents.

    Args:
        token: JWT token string

    Returns:
        Decoded payload dict or None if invalid
    """
    try:
        # Decode without verification
        return jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            options={"verify_signature": False}
        )
    except JWTError:
        return None


def verify_token(token: str, expected_type: str = "access") -> Optional[TokenData]:
    """
    Verify and decode a JWT token.

    Args:
        token: JWT token string
        expected_type: Expected token type ('access' or 'refresh')

    Returns:
        TokenData if valid, None otherwise

    Raises:
        None - returns None on any error for security
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        # Verify token type
        token_type = payload.get("type")
        if token_type != expected_type:
            logger.warning(f"Token type mismatch: expected {expected_type}, got {token_type}")
            return None

        # Extract required fields
        user_id = payload.get("sub")
        tenant_id = payload.get("tenant_id")

        if not user_id or not tenant_id:
            logger.warning("Token missing required fields")
            return None

        # Build TokenData
        return TokenData(
            user_id=user_id,
            tenant_id=tenant_id,
            email=payload.get("email", ""),
            role=payload.get("role", "viewer"),
            token_type=token_type,
            exp=datetime.fromtimestamp(payload["exp"]),
            iat=datetime.fromtimestamp(payload["iat"])
        )

    except ExpiredSignatureError:
        logger.debug("Token expired")
        return None
    except JWTError as e:
        logger.warning(f"Token verification failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error verifying token: {e}")
        return None


def create_token_pair(
    user_id: str,
    tenant_id: str,
    email: str,
    role: str
) -> Dict[str, str]:
    """
    Create both access and refresh tokens.

    Args:
        user_id: User's UUID as string
        tenant_id: Tenant's UUID as string
        email: User's email
        role: User's role

    Returns:
        Dict with 'access_token' and 'refresh_token'
    """
    return {
        "access_token": create_access_token(user_id, tenant_id, email, role),
        "refresh_token": create_refresh_token(user_id, tenant_id),
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60  # seconds
    }


def refresh_access_token(refresh_token: str) -> Optional[Dict[str, str]]:
    """
    Use a refresh token to get a new access token.
    Requires looking up the user to get current email/role.

    Args:
        refresh_token: Valid refresh token

    Returns:
        New token pair if valid, None otherwise

    Note: This is a partial implementation. In practice, you'd
    look up the user from the database to get current email/role.
    """
    token_data = verify_token(refresh_token, expected_type="refresh")
    if not token_data:
        return None

    # In a full implementation, you would:
    # 1. Look up the user from the database
    # 2. Check if they're still active
    # 3. Get their current role (might have changed)
    # 4. Create new tokens with fresh data

    # For now, we just return None to indicate this needs DB lookup
    return None
