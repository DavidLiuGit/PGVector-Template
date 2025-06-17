"""
Unit tests for document validation
"""

import unittest
from pydantic import ValidationError

from pgvector_template.core.document import BaseDocumentOptionalProps


class TestBaseDocumentOptionalProps(unittest.TestCase):
    """Test validation for BaseDocumentOptionalProps"""

    def test_valid_props(self):
        """Test valid properties pass validation"""
        # Test with all valid properties
        props = BaseDocumentOptionalProps(
            title="Test Document",
            collection="test_collection",
            original_url="https://example.com/doc",
            language="en",
            score=0.95,
            tags=["test", "document", "validation"],
        )

        self.assertEqual(props.title, "Test Document")
        self.assertEqual(props.collection, "test_collection")
        self.assertEqual(props.original_url, "https://example.com/doc")
        self.assertEqual(props.language, "en")
        self.assertEqual(props.score, 0.95)
        self.assertEqual(props.tags, ["test", "document", "validation"])

    def test_collection_length_validation(self):
        """Test collection name length validation"""
        # Test with collection name that's too long (>64 chars)
        with self.assertRaises(ValidationError) as context:
            BaseDocumentOptionalProps(collection="x" * 65)

        self.assertIn("collection", str(context.exception))
        self.assertIn("string_too_long", str(context.exception))

    def test_url_length_validation(self):
        """Test URL length validation"""
        # Test with URL that's too long (>2048 chars)
        with self.assertRaises(ValidationError) as context:
            BaseDocumentOptionalProps(original_url="https://example.com/" + "x" * 2030)

        self.assertIn("original_url", str(context.exception))
        self.assertIn("string_too_long", str(context.exception))

    def test_language_validation(self):
        """Test language code validation"""
        # Valid language codes
        valid_codes = ["en", "es", "zh", "en-US", "fr-CA"]
        for code in valid_codes:
            props = BaseDocumentOptionalProps(language=code)
            self.assertEqual(props.language, code)

        # Invalid language codes
        invalid_codes = ["english", "e", "123", "en_US", "EN"]
        for code in invalid_codes:
            with self.assertRaises(ValidationError) as context:
                BaseDocumentOptionalProps(language=code)
            self.assertIn("language", str(context.exception))
            self.assertIn("pattern", str(context.exception))

    def test_score_range_validation(self):
        """Test score range validation"""
        # Valid scores
        valid_scores = [0.0, 0.5, 1.0]
        for score in valid_scores:
            props = BaseDocumentOptionalProps(score=score)
            self.assertEqual(props.score, score)

        # Invalid scores
        invalid_scores = [-0.1, 1.1, 2.0]
        for score in invalid_scores:
            with self.assertRaises(ValidationError) as context:
                BaseDocumentOptionalProps(score=score)
            self.assertIn("score", str(context.exception))

    def test_tags_validation(self):
        """Test tags validation"""
        # Valid tags
        props = BaseDocumentOptionalProps(tags=["tag1", "tag2", "tag3"])
        self.assertEqual(props.tags, ["tag1", "tag2", "tag3"])

        # Duplicate tags should be removed
        props = BaseDocumentOptionalProps(tags=["tag1", "tag2", "tag1", "tag3"])
        self.assertEqual(props.tags, ["tag1", "tag2", "tag3"])

        # Empty tags should be rejected
        with self.assertRaises(ValidationError) as context:
            BaseDocumentOptionalProps(tags=["tag1", "", "tag3"])
        self.assertIn("tags", str(context.exception))

        # Non-string tags should be rejected
        with self.assertRaises(ValidationError) as context:
            BaseDocumentOptionalProps(tags=["tag1", 123, "tag3"])
        self.assertIn("tags", str(context.exception))


if __name__ == "__main__":
    unittest.main()
