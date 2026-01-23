"""
2ndBrain Authentication Package
JWT-based authentication with multi-tenant support
"""

from .jwt import (
    create_access_token,
    create_refresh_token,
    verify_token,
    decode_token,
    TokenData
)

from .dependencies import (
    get_current_user,
    get_current_tenant,
    require_role,
    AuthContext
)

__all__ = [
    'create_access_token',
    'create_refresh_token',
    'verify_token',
    'decode_token',
    'TokenData',
    'get_current_user',
    'get_current_tenant',
    'require_role',
    'AuthContext'
]
