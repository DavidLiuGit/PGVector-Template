"""
Unit tests for document metadata validation
"""

import unittest
from pydantic import ValidationError

from pgvector_template.core.document import BaseDocumentMetadata


class TestBaseDocumentMetadataValidation(unittest.TestCase):
    """Test validation for BaseDocumentMetadata"""

    def test_valid_metadata(self):
        """Test valid metadata passes validation"""
        # Test with all required fields
        metadata = BaseDocumentMetadata(document_type="test_type")
        self.assertEqual(metadata.document_type, "test_type")
        self.assertEqual(metadata.schema_version, "1.0")
        
        # Test with custom schema version
        metadata = BaseDocumentMetadata(document_type="test_type", schema_version="2.0")
        self.assertEqual(metadata.document_type, "test_type")
        self.assertEqual(metadata.schema_version, "2.0")

    def test_missing_required_fields(self):
        """Test validation fails when required fields are missing"""
        # Missing document_type
        with self.assertRaises(ValidationError) as context:
            BaseDocumentMetadata()
        
        self.assertIn("document_type", str(context.exception))
        self.assertIn("field required", str(context.exception).lower())

    def test_model_dump(self):
        """Test model_dump method works correctly"""
        metadata = BaseDocumentMetadata(document_type="test_type", schema_version="2.0")
        result = metadata.model_dump()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["document_type"], "test_type")
        self.assertEqual(result["schema_version"], "2.0")
        
    def test_to_dict(self):
        """Test to_dict method works correctly"""
        metadata = BaseDocumentMetadata(document_type="test_type", schema_version="2.0")
        result = metadata.to_dict()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["document_type"], "test_type")
        self.assertEqual(result["schema_version"], "2.0")


if __name__ == "__main__":
    unittest.main()