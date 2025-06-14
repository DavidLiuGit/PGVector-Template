"""
Integration tests for TempDocumentDatabaseManager
Tests schema & table creation against an actual Postgres DB instance
"""

import pytest
from sqlalchemy import text, inspect

from pgvector_template.db.document_db import TempDocumentDatabaseManager
from test_document import TestDocument, SecondTestDocument


class TestTempDocumentDatabaseManagerIntegration:
    """
    Integration tests for TempDocumentDatabaseManager.
    To see the tables that this test creates, comment out the "Clean up" step at the end of the relevant test
    """

    @pytest.fixture
    def document_classes(self):
        """Return list of document classes for testing"""
        return [TestDocument]

    @pytest.fixture
    def temp_db_manager(self, database_url, document_classes):
        """Create a temporary document database manager"""
        return TempDocumentDatabaseManager(
            database_url=database_url, schema_suffix="integ_test", document_classes=document_classes
        )

    def test_create_temp_schema(self, temp_db_manager):
        """Test creating a temporary schema with tables"""
        # Create temporary schema
        temp_schema = temp_db_manager.create_temp_schema()

        try:
            # Verify schema exists
            with temp_db_manager.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT schema_name FROM information_schema.schemata WHERE schema_name = :schema"),
                    {"schema": temp_schema},
                )
                assert result.scalar() == temp_schema, f"Schema {temp_schema} was not created"

                # Verify table exists in schema
                inspector = inspect(temp_db_manager.engine)
                tables = inspector.get_table_names(schema=temp_schema)
                assert "test_documents" in tables, f"Table 'test_documents' not found in schema {temp_schema}"

                # Verify table structure
                columns = {col["name"] for col in inspector.get_columns("test_documents", schema=temp_schema)}
                required_columns = {"id", "content", "embedding", "document_metadata"}
                assert required_columns.issubset(columns), f"Missing required columns: {required_columns - columns}"
        finally:
            # Clean up
            temp_db_manager.cleanup_temp_schema(temp_schema)

    def test_cleanup_temp_schema(self, temp_db_manager):
        """Test cleaning up a temporary schema"""
        # Create temporary schema
        temp_schema = temp_db_manager.create_temp_schema()

        # Clean up schema
        temp_db_manager.cleanup_temp_schema(temp_schema)

        # Verify schema no longer exists
        with temp_db_manager.engine.connect() as conn:
            result = conn.execute(
                text("SELECT schema_name FROM information_schema.schemata WHERE schema_name = :schema"),
                {"schema": temp_schema},
            )
            assert result.scalar() is None, f"Schema {temp_schema} still exists after cleanup"

    def test_multiple_document_classes(self, database_url):
        """Test creating schema with multiple document classes"""
        multi_doc_manager = TempDocumentDatabaseManager(
            database_url=database_url,
            schema_suffix="multi_doc_test",
            document_classes=[TestDocument, SecondTestDocument],
        )

        # Create temporary schema
        temp_schema = multi_doc_manager.create_temp_schema()

        try:
            # Verify both tables exist
            inspector = inspect(multi_doc_manager.engine)
            tables = inspector.get_table_names(schema=temp_schema)
            assert "test_documents" in tables, "Table 'test_documents' not found"
            assert "second_test_documents" in tables, "Table 'second_test_documents' not found"
        finally:
            # Clean up
            multi_doc_manager.cleanup_temp_schema(temp_schema)
