import unittest
from unittest.mock import patch, MagicMock, call

from pgvector_template.db.document_db import DocumentDatabaseManager, TempDocumentDatabaseManager


class TestDocumentDatabaseManager(unittest.TestCase):
    """Unit tests for the DocumentDatabaseManager class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        self.test_db_url = "postgresql://user:password@localhost:5432/testdb"
        self.schema_suffix = "test_docs"

        # Create a mock document class instead of a real SQLAlchemy model
        self.mock_document = MagicMock()
        self.mock_document.__table__ = MagicMock()
        self.document_classes = [self.mock_document]

        self.db_manager = DocumentDatabaseManager(self.test_db_url, self.schema_suffix, self.document_classes)

    def test_init(self):
        """Test initialization of DocumentDatabaseManager"""
        self.assertEqual(self.db_manager.database_url, self.test_db_url)
        self.assertEqual(self.db_manager.schema_name, f"{DocumentDatabaseManager.SCHEMA_PREFIX}{self.schema_suffix}")
        self.assertEqual(self.db_manager.document_classes, self.document_classes)

    @patch.object(DocumentDatabaseManager, "initialize")
    @patch.object(DocumentDatabaseManager, "create_schema")
    @patch.object(DocumentDatabaseManager, "create_tables")
    def test_setup(self, mock_create_tables, mock_create_schema, mock_initialize):
        """Test the setup method"""
        # Call the method under test
        self.db_manager.setup()

        # Assertions
        mock_initialize.assert_called_once()
        mock_create_schema.assert_called_once_with(self.db_manager.schema_name)
        mock_create_tables.assert_called_once_with(self.document_classes[0], self.db_manager.schema_name)

        # Verify schema was set on document class
        self.assertEqual(self.document_classes[0].__table__.schema, self.db_manager.schema_name)


class TestTempDocumentDatabaseManager(unittest.TestCase):
    """Unit tests for the TempDocumentDatabaseManager class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        self.test_db_url = "postgresql://user:password@localhost:5432/testdb"
        self.schema_suffix = "temp_test"

        # Create a mock document class instead of a real SQLAlchemy model
        self.mock_document = MagicMock()
        self.mock_document.__table__ = MagicMock()
        self.document_classes = [self.mock_document]

        self.db_manager = TempDocumentDatabaseManager(self.test_db_url, self.schema_suffix, self.document_classes)

    @patch("uuid.uuid4")
    @patch.object(TempDocumentDatabaseManager, "initialize")
    @patch.object(TempDocumentDatabaseManager, "create_schema")
    @patch.object(TempDocumentDatabaseManager, "create_tables")
    def test_create_temp_schema(self, mock_create_tables, mock_create_schema, mock_initialize, mock_uuid4):
        """Test creation of temporary schema"""
        # Setup mock UUID
        mock_uuid = MagicMock()
        mock_uuid.hex = "abcdef1234567890"
        mock_uuid4.return_value = mock_uuid

        # Call the method under test
        result = self.db_manager.setup()

        # Expected schema name
        expected_schema = f"temp_{DocumentDatabaseManager.SCHEMA_PREFIX}{self.db_manager.schema_name}_abcdef12"

        # Assertions
        self.assertEqual(result, expected_schema)
        mock_initialize.assert_called_once()
        mock_create_schema.assert_called_once_with(expected_schema)
        mock_create_tables.assert_called_once_with(self.document_classes[0], expected_schema)

        # Verify schema was set on document class
        self.assertEqual(self.document_classes[0].__table__.schema, expected_schema)

    @patch("pgvector_template.db.document_db.text")
    def test_cleanup_temp_schema(self, mock_text):
        """Test cleanup of temporary schema"""
        # Setup
        temp_schema = "temp_test_schema"
        mock_text_instance = MagicMock()
        mock_text.return_value = mock_text_instance
        mock_conn = MagicMock()
        self.db_manager.engine = MagicMock()
        self.db_manager.engine.connect.return_value.__enter__.return_value = mock_conn

        # Call the method under test
        self.db_manager.cleanup(temp_schema)

        # Assertions
        mock_text.assert_called_once_with(f"DROP SCHEMA IF EXISTS {temp_schema} CASCADE")
        mock_conn.execute.assert_called_once_with(mock_text_instance)
        mock_conn.commit.assert_called_once()

    def test_cleanup_temp_schema_safety_check(self):
        """Test safety check in cleanup_temp_schema"""
        # Setup
        unsafe_schema = "production_schema"

        # Call the method under test and assert it raises ValueError
        with self.assertRaises(ValueError) as context:
            self.db_manager.cleanup(unsafe_schema)

        self.assertIn("Can only drop schemas with 'temp_' prefix", str(context.exception))


if __name__ == "__main__":
    unittest.main()
