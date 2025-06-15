from contextlib import contextmanager
from logging import getLogger
from typing import Generator, Type

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import DeclarativeMeta


class DatabaseManager:
    """Manages database connections and schema setup"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        self.logger = getLogger(self.__class__.__name__)

    def initialize(self):
        """Initialize database connection and session factory"""
        self.engine = create_engine(
            self.database_url, pool_pre_ping=True, pool_recycle=300, echo=False  # Set to True for SQL debugging
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._ensure_pgvector_extension()

    def create_schema(self, schema_name: str) -> None:
        """Create a new schema for a collection type"""
        with self.engine.connect() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))
            conn.commit()

        self.logger.info(f"Created schema: {schema_name}")

    def create_tables(self, base_class: Type[DeclarativeMeta], schema_name: str) -> None:
        """Create tables for a specific schema"""
        # Ensure all tables in this base class use the specified schema
        for table in base_class.metadata.tables.values():
            table.schema = schema_name
        base_class.metadata.create_all(self.engine, checkfirst=True)
        self.logger.info(f"Created tables for schema: {schema_name}")

    def _ensure_pgvector_extension(self):
        """Ensure pgvector extension is available"""
        with self.engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a database session with automatic cleanup. Usage:

        ```python
        with db_mgr.get_session() as session:
            # do stuff with the Session here
        ```
        """
        session = self.SessionLocal()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
