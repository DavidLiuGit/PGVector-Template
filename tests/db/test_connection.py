import unittest
from unittest.mock import patch, MagicMock, call
from sqlalchemy.orm import declarative_base


from pgvector_template.db.connection import DatabaseManager


class TestDatabaseManager(unittest.TestCase):
    """Unit tests for the DatabaseManager class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        self.test_db_url = "postgresql://user:password@localhost:5432/testdb"
        self.db_manager = DatabaseManager(self.test_db_url)

    @patch("pgvector_template.db.connection.create_engine")
    @patch("pgvector_template.db.connection.sessionmaker")
    def test_initialize(self, mock_sessionmaker, mock_create_engine):
        """Test database initialization"""
        # Setup mocks
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session_local = MagicMock()
        mock_sessionmaker.return_value = mock_session_local

        # Mock the connection for _ensure_pgvector_extension
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        # Call the method under test
        self.db_manager.initialize()

        # Assertions
        mock_create_engine.assert_called_once_with(self.test_db_url, pool_pre_ping=True, pool_recycle=300, echo=False)
        mock_sessionmaker.assert_called_once_with(autocommit=False, autoflush=False, bind=mock_engine)
        mock_conn.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
        self.assertEqual(self.db_manager.engine, mock_engine)
        self.assertEqual(self.db_manager.SessionLocal, mock_session_local)

    @patch("pgvector_template.db.connection.text")
    def test_create_schema(self, mock_text):
        """Test schema creation"""
        # Setup
        schema_name = "test_schema"
        mock_text_instance = MagicMock()
        mock_text.return_value = mock_text_instance
        mock_conn = MagicMock()
        self.db_manager.engine = MagicMock()
        self.db_manager.engine.connect.return_value.__enter__.return_value = mock_conn

        # Call the method under test
        self.db_manager.create_schema(schema_name)

        # Assertions
        mock_text.assert_called_once_with(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
        mock_conn.execute.assert_called_once_with(mock_text_instance)
        mock_conn.commit.assert_called_once()

    @patch("sqlalchemy.sql.schema.MetaData.create_all")
    def test_create_tables_with_base_class(self, mock_create_all):
        """Test table creation with SQLAlchemy Base class"""
        # Setup
        schema_name = "test_schema"
        base_class = declarative_base()
        self.db_manager.engine = MagicMock()

        # Call the method under test
        self.db_manager.create_tables(base_class, schema_name)

        # Assertions
        mock_create_all.assert_called_once_with(self.db_manager.engine, checkfirst=True)

    @patch("pgvector_template.db.connection.text")
    def test_ensure_pgvector_extension(self, mock_text):
        """Test ensuring pgvector extension is available"""
        # Setup
        mock_text_instance = MagicMock()
        mock_text.return_value = mock_text_instance
        mock_conn = MagicMock()
        self.db_manager.engine = MagicMock()
        self.db_manager.engine.connect.return_value.__enter__.return_value = mock_conn

        # Call the method under test
        self.db_manager._ensure_pgvector_extension()

        # Assertions
        mock_text.assert_called_once_with("CREATE EXTENSION IF NOT EXISTS vector")
        mock_conn.execute.assert_called_once_with(mock_text_instance)
        mock_conn.commit.assert_called_once()

    def test_get_session(self):
        """Test session context manager"""
        # Setup
        mock_session = MagicMock()
        self.db_manager.SessionLocal = MagicMock(return_value=mock_session)

        # Call the method under test
        with self.db_manager.get_session() as session:
            # Do something with session
            session.query()

        # Assertions
        self.db_manager.SessionLocal.assert_called_once()
        mock_session.close.assert_called_once()
        mock_session.query.assert_called_once()

    def test_get_session_with_exception(self):
        """Test session context manager with exception handling"""
        # Setup
        mock_session = MagicMock()
        self.db_manager.SessionLocal = MagicMock(return_value=mock_session)

        # Call the method under test
        with self.assertRaises(ValueError):
            with self.db_manager.get_session() as session:
                raise ValueError("Test exception")

        # Assertions
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
