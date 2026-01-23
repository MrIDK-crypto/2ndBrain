"""
Database Connection and Session Management
Supports PostgreSQL with connection pooling and SQLite for development
"""

import os
import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import QueuePool, StaticPool

logger = logging.getLogger(__name__)

# Module-level variables that can be reconfigured
_engine = None
_SessionLocal = None


def _get_database_url() -> str:
    """Get database URL from environment"""
    url = os.getenv("DATABASE_URL", "postgresql://localhost:5432/secondbrain")
    # Handle Render's postgres:// vs postgresql:// issue
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


def _create_engine(url: str):
    """Create SQLAlchemy engine for the given URL"""
    is_sqlite = url.startswith("sqlite")

    if is_sqlite:
        # SQLite configuration (no connection pooling, single connection)
        eng = create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=os.getenv("SQL_ECHO", "false").lower() == "true"
        )
        logger.info("Using SQLite database (development mode)")
    else:
        # PostgreSQL configuration with connection pooling
        eng = create_engine(
            url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            echo=os.getenv("SQL_ECHO", "false").lower() == "true"
        )
    return eng


def get_engine():
    """Get or create the database engine"""
    global _engine
    if _engine is None:
        _engine = _create_engine(_get_database_url())
    return _engine


def get_session_factory():
    """Get or create the session factory"""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine()
        )
    return _SessionLocal


# Base class for models (needs to be defined early for imports)
Base = declarative_base()


# Lazily evaluated module-level variables
def __getattr__(name):
    if name == 'DATABASE_URL':
        return _get_database_url()
    elif name == 'IS_SQLITE':
        return _get_database_url().startswith("sqlite")
    elif name == 'engine':
        return get_engine()
    elif name == 'SessionLocal':
        return get_session_factory()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_db() -> Generator[Session, None, None]:
    """
    Dependency that provides a database session.
    Use with FastAPI's Depends() or Flask's context.

    Usage:
        # FastAPI
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()

        # Flask
        with get_db_context() as db:
            items = db.query(Item).all()
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    Use in Flask or synchronous code.

    Usage:
        with get_db_context() as db:
            user = db.query(User).first()
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """
    Initialize database tables.
    Call this on application startup.
    """
    # Import models to register them with Base
    from . import models  # noqa: F401

    url = _get_database_url()
    logger.info(f"Initializing database: {url[:50]}...")
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


def check_db_connection() -> bool:
    """
    Check if database connection is healthy.
    Returns True if connection works, False otherwise.
    """
    try:
        eng = get_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


# Row-Level Security helper for multi-tenancy
def set_tenant_context(session: Session, tenant_id: str):
    """
    Set the current tenant context for RLS policies.
    Call this at the start of each request.

    Note: This requires PostgreSQL RLS policies to be set up.
    """
    try:
        session.execute(
            text("SET app.current_tenant_id = :tenant_id"),
            {"tenant_id": tenant_id}
        )
    except Exception as e:
        logger.warning(f"Could not set tenant context: {e}")


# Event listener to log slow queries (optional, for debugging)
if os.getenv("LOG_SLOW_QUERIES", "false").lower() == "true":
    @event.listens_for(engine, "before_cursor_execute")
    def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        conn.info.setdefault('query_start_time', []).append(
            __import__('time').time()
        )

    @event.listens_for(engine, "after_cursor_execute")
    def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        import time
        total = time.time() - conn.info['query_start_time'].pop()
        if total > 0.5:  # Log queries taking more than 500ms
            logger.warning(f"Slow query ({total:.2f}s): {statement[:100]}...")
