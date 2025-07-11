"""
Unit tests for BaseDocument.from_props method
"""

import unittest
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import UUID as PGUUID

from pgvector_template.core.document import BaseDocument, BaseDocumentOptionalProps


class MockUUIDDocument(BaseDocument):
    """Mock document class with UUID corpus_id for testing"""

    __tablename__ = "mock_uuid_doc"
    __abstract__ = False

    corpus_id = Column(PGUUID(as_uuid=True), index=True)


class MockStringDocument(BaseDocument):
    """Mock document class with String corpus_id for testing"""

    __tablename__ = "mock_string_doc"
    __abstract__ = False

    corpus_id = Column(String(64), index=True)


class TestFromProps(unittest.TestCase):
    """Test cases for BaseDocument.from_props method"""

    def setUp(self):
        """Set up test data"""
        self.test_uuid = uuid4()
        self.test_content = "Test content"
        self.test_embedding = [0.1, 0.2, 0.3]
        self.test_metadata = {"key": "value"}

    def test_from_props_with_uuid_column_and_uuid_input(self):
        """Test from_props with UUID column and UUID input"""
        doc = MockUUIDDocument.from_props(
            corpus_id=self.test_uuid,
            chunk_index=0,
            content=self.test_content,
            embedding=self.test_embedding,
            metadata=self.test_metadata,
        )

        self.assertEqual(doc.corpus_id, self.test_uuid)
        self.assertEqual(doc.content, self.test_content)
        self.assertEqual(doc.embedding, self.test_embedding)
        self.assertEqual(doc.document_metadata, self.test_metadata)

    def test_from_props_with_uuid_column_and_string_input(self):
        """Test from_props with UUID column and string input (SQLAlchemy handles conversion)"""
        test_string = "test-string-id"

        doc = MockUUIDDocument.from_props(
            corpus_id=test_string, chunk_index=0, content=self.test_content, embedding=self.test_embedding
        )

        # Should remain as string (SQLAlchemy converts during DB operations)
        self.assertEqual(doc.corpus_id, test_string)

    def test_from_props_with_uuid_column_and_uuid_string_input(self):
        """Test from_props with UUID column and valid UUID string input"""
        uuid_string = str(self.test_uuid)

        doc = MockUUIDDocument.from_props(
            corpus_id=uuid_string, chunk_index=0, content=self.test_content, embedding=self.test_embedding
        )

        # Should remain as string (SQLAlchemy converts during DB operations)
        self.assertEqual(doc.corpus_id, uuid_string)

    def test_from_props_with_string_column_and_string_input(self):
        """Test from_props with String column and string input (kept as-is)"""
        test_string = "2025-01-15"

        doc = MockStringDocument.from_props(
            corpus_id=test_string, chunk_index=0, content=self.test_content, embedding=self.test_embedding
        )

        # Should remain as string
        self.assertEqual(doc.corpus_id, test_string)
        self.assertIsInstance(doc.corpus_id, str)

    def test_from_props_with_string_column_and_uuid_input(self):
        """Test from_props with String column and UUID input"""
        doc = MockStringDocument.from_props(
            corpus_id=self.test_uuid, chunk_index=0, content=self.test_content, embedding=self.test_embedding
        )

        # UUID should be passed through as-is (SQLAlchemy will handle conversion)
        self.assertEqual(doc.corpus_id, self.test_uuid)

    def test_from_props_with_optional_props(self):
        """Test from_props with optional properties"""
        optional_props = BaseDocumentOptionalProps(
            title="Test Title",
            collection="test_collection",
            original_url="https://example.com",
            language="es",
            score=0.8,
            tags=["tag1", "tag2"],
        )

        doc = MockStringDocument.from_props(
            corpus_id="test-id",
            chunk_index=1,
            content=self.test_content,
            embedding=self.test_embedding,
            metadata=self.test_metadata,
            optional_props=optional_props,
        )

        self.assertEqual(doc.title, "Test Title")
        self.assertEqual(doc.collection, "test_collection")
        self.assertEqual(doc.origin_url, "https://example.com")
        self.assertEqual(doc.language, "es")
        self.assertEqual(doc.score, 0.8)
        self.assertEqual(doc.tags, ["tag1", "tag2"])
        self.assertEqual(doc.chunk_index, 1)

    def test_from_props_without_optional_props(self):
        """Test from_props without optional properties (uses defaults)"""
        doc = MockStringDocument.from_props(
            corpus_id="test-id", chunk_index=0, content=self.test_content, embedding=self.test_embedding
        )

        self.assertIsNone(doc.title)
        self.assertIsNone(doc.collection)
        self.assertIsNone(doc.origin_url)
        self.assertEqual(doc.language, "en")  # default
        self.assertIsNone(doc.score)
        self.assertIsNone(doc.tags)


if __name__ == "__main__":
    unittest.main()
