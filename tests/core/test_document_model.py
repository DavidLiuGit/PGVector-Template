"""
Unit tests for document models
"""

import unittest
from pydantic import ValidationError

from pgvector_template.core.document import BaseDocumentOptionalProps


class TestDocumentModel(unittest.TestCase):
    """Test cases for document models"""

    def test_base_document_optional_props_defaults(self):
        """Test default values for BaseDocumentOptionalProps"""
        props = BaseDocumentOptionalProps()
        self.assertIsNone(props.title)
        self.assertIsNone(props.collection)
        self.assertIsNone(props.original_url)
        self.assertEqual(props.language, "en")
        self.assertIsNone(props.score)
        self.assertIsNone(props.tags)

    def test_base_document_optional_props_validation(self):
        """Test validation for BaseDocumentOptionalProps"""
        # Test with valid values
        props = BaseDocumentOptionalProps(
            title="Test",
            collection="test-collection",
            original_url="https://example.com",
            language="fr",
            score=0.75,
            tags=["test", "document"],
        )

        self.assertEqual(props.title, "Test")
        self.assertEqual(props.collection, "test-collection")
        self.assertEqual(props.original_url, "https://example.com")
        self.assertEqual(props.language, "fr")
        self.assertEqual(props.score, 0.75)
        self.assertEqual(props.tags, ["test", "document"])

        # Test with invalid language
        with self.assertRaises(ValidationError):
            BaseDocumentOptionalProps(language="invalid")

        # Test with invalid score
        with self.assertRaises(ValidationError):
            BaseDocumentOptionalProps(score=2.0)

        # Test with duplicate tags (should be deduplicated)
        props = BaseDocumentOptionalProps(tags=["tag1", "tag2", "tag1"])
        self.assertEqual(props.tags, ["tag1", "tag2"])


if __name__ == "__main__":
    unittest.main()
