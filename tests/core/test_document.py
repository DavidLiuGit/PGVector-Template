import unittest
from datetime import datetime
from unittest.mock import patch, MagicMock
from uuid import UUID

from pgvector_template.core.document import BaseDocument, BaseDocumentMetadata


class TestBaseDocumentMetadata(unittest.TestCase):
    """Unit tests for the BaseDocumentMetadata class"""

    def test_init(self):
        """Test initialization of BaseDocumentMetadata"""
        metadata = BaseDocumentMetadata(document_type="test_type")
        self.assertEqual(metadata.document_type, "test_type")
        self.assertEqual(metadata.schema_version, "1.0")

    def test_to_dict(self):
        """Test conversion to dictionary"""
        metadata = BaseDocumentMetadata(document_type="test_type", schema_version="2.0")
        result = metadata.to_dict()

        self.assertIsInstance(result, dict)
        self.assertEqual(result["document_type"], "test_type")
        self.assertEqual(result["schema_version"], "2.0")


class TestBaseDocument(unittest.TestCase):
    """Unit tests for the BaseDocument class"""

    @patch("pgvector_template.core.document.declarative_base")
    def test_base_document_attributes(self, mock_declarative_base):
        """Test that BaseDocument has the expected attributes"""
        # We can't directly instantiate BaseDocument as it's abstract,
        # but we can check its class attributes

        # Verify it's abstract
        self.assertTrue(BaseDocument.__abstract__)

        # Check column definitions exist
        self.assertTrue(hasattr(BaseDocument, "id"))
        self.assertTrue(hasattr(BaseDocument, "collection"))
        self.assertTrue(hasattr(BaseDocument, "corpus_id"))
        self.assertTrue(hasattr(BaseDocument, "chunk_index"))
        self.assertTrue(hasattr(BaseDocument, "content"))
        self.assertTrue(hasattr(BaseDocument, "title"))
        self.assertTrue(hasattr(BaseDocument, "metadata"))
        self.assertTrue(hasattr(BaseDocument, "origin_url"))
        self.assertTrue(hasattr(BaseDocument, "language"))
        self.assertTrue(hasattr(BaseDocument, "score"))
        self.assertTrue(hasattr(BaseDocument, "tags"))
        self.assertTrue(hasattr(BaseDocument, "embedding"))
        self.assertTrue(hasattr(BaseDocument, "created_at"))
        self.assertTrue(hasattr(BaseDocument, "updated_at"))
        self.assertTrue(hasattr(BaseDocument, "is_deleted"))


if __name__ == "__main__":
    unittest.main()
