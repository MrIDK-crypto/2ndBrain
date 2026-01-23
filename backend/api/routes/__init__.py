"""
2ndBrain API Routes
Modular route organization
"""

from .documents import documents_bp
from .connectors import connectors_bp
from .search import search_bp
from .admin import admin_bp

__all__ = [
    "documents_bp",
    "connectors_bp",
    "search_bp",
    "admin_bp",
]
