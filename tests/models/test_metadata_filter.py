import unittest
from typing import Any

import pytest
from pydantic import ValidationError

from pgvector_template.core.search import MetadataFilter


class TestMetadataFilter(unittest.TestCase):
    """Test MetadataFilter pydantic model validation."""

    def test_valid_filter_creation(self):
        """Test creating valid MetadataFilter instances."""
        # Test basic equality filter
        filter1 = MetadataFilter(field_name="author", condition="eq", value="John Doe")
        self.assertEqual(filter1.field_name, "author")
        self.assertEqual(filter1.condition, "eq")
        self.assertEqual(filter1.value, "John Doe")

        # Test nested field with dot notation
        filter2 = MetadataFilter(
            field_name="publication_info.journal", condition="contains", value="Science"
        )
        self.assertEqual(filter2.field_name, "publication_info.journal")

        # Test numeric comparisons
        filter3 = MetadataFilter(field_name="year", condition="gte", value=2020)
        self.assertEqual(filter3.value, 2020)

        # Test array operations
        filter4 = MetadataFilter(field_name="tags", condition="in", value=["AI", "ML"])
        self.assertEqual(filter4.value, ["AI", "ML"])

        # Test exists condition
        filter5 = MetadataFilter(field_name="optional_field", condition="exists", value=True)
        self.assertEqual(filter5.condition, "exists")

    def test_all_valid_conditions(self):
        """Test all valid condition values."""
        valid_conditions = ["eq", "gt", "gte", "lt", "lte", "contains", "in", "exists"]

        for condition in valid_conditions:
            filter_obj = MetadataFilter(
                field_name="test_field", condition=condition, value="test_value"
            )
            self.assertEqual(filter_obj.condition, condition)

    def test_invalid_condition(self):
        """Test invalid condition values raise ValidationError."""
        with self.assertRaises(ValidationError) as context:
            MetadataFilter(field_name="test_field", condition="invalid", value="test_value")

        self.assertIn("Input should be", str(context.exception))

    def test_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        # Missing field_name
        with self.assertRaises(ValidationError):
            MetadataFilter(condition="eq", value="test")

        # Missing condition
        with self.assertRaises(ValidationError):
            MetadataFilter(field_name="test", value="test")

        # Missing value
        with self.assertRaises(ValidationError):
            MetadataFilter(field_name="test", condition="eq")

    def test_value_types(self):
        """Test various value types are accepted."""
        # String value
        filter1 = MetadataFilter(field_name="title", condition="eq", value="Test Title")
        self.assertIsInstance(filter1.value, str)

        # Integer value
        filter2 = MetadataFilter(field_name="count", condition="gt", value=42)
        self.assertIsInstance(filter2.value, int)

        # Float value
        filter3 = MetadataFilter(field_name="score", condition="gte", value=3.14)
        self.assertIsInstance(filter3.value, float)

        # Boolean value
        filter4 = MetadataFilter(field_name="published", condition="eq", value=True)
        self.assertIsInstance(filter4.value, bool)

        # List value
        filter5 = MetadataFilter(
            field_name="categories", condition="contains", value=["tech", "science"]
        )
        self.assertIsInstance(filter5.value, list)

        # None value
        filter6 = MetadataFilter(field_name="optional", condition="eq", value=None)
        self.assertIsNone(filter6.value)

    def test_field_name_validation(self):
        """Test field_name accepts various formats."""
        # Simple field name
        filter1 = MetadataFilter(field_name="author", condition="eq", value="test")
        self.assertEqual(filter1.field_name, "author")

        # Nested field with dot notation
        filter2 = MetadataFilter(field_name="metadata.publication.year", condition="eq", value=2023)
        self.assertEqual(filter2.field_name, "metadata.publication.year")

        # Field name with underscores
        filter3 = MetadataFilter(field_name="created_at", condition="gte", value="2023-01-01")
        self.assertEqual(filter3.field_name, "created_at")

    def test_model_serialization(self):
        """Test model can be serialized to dict."""
        filter_obj = MetadataFilter(field_name="author", condition="eq", value="John Doe")
        result = filter_obj.model_dump()

        expected = {"field_name": "author", "condition": "eq", "value": "John Doe"}
        self.assertEqual(result, expected)

    def test_model_from_dict(self):
        """Test model can be created from dict."""
        data = {"field_name": "category", "condition": "in", "value": ["tech", "science"]}
        filter_obj = MetadataFilter(**data)

        self.assertEqual(filter_obj.field_name, "category")
        self.assertEqual(filter_obj.condition, "in")
        self.assertEqual(filter_obj.value, ["tech", "science"])


if __name__ == "__main__":
    unittest.main()
