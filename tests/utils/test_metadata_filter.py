import unittest
from typing import Optional

from pydantic import BaseModel, Field

from pgvector_template.core.document import BaseDocumentMetadata
from pgvector_template.core.search import MetadataFilter
from pgvector_template.utils.metadata_filter import validate_metadata_filter, validate_metadata_filters, validate_condition_compatibility


class TestMetadata(BaseDocumentMetadata):
    """Test metadata class with various field types."""
    author: str
    year: int
    score: float
    published: bool
    tags: list[str]
    optional_field: Optional[str] = None


class PublicationInfo(BaseModel):
    journal: str
    volume: int
    pages: list[int]


class DocumentStats(BaseModel):
    views: int
    rating: float
    featured: bool


class NestedMetadata(BaseDocumentMetadata):
    """Test metadata with nested structure."""
    publication_info: PublicationInfo
    stats: DocumentStats


class TestValidateMetadataFilter(unittest.TestCase):
    """Test validate_metadata_filter function."""

    def test_valid_simple_field_filters(self):
        """Test validation of filters on simple fields."""
        # String field with eq condition
        filter1 = MetadataFilter(field_name="author", condition="eq", value="John Doe")
        validate_metadata_filter(filter1, TestMetadata)  # Should not raise

        # Integer field with comparison conditions
        filter2 = MetadataFilter(field_name="year", condition="gte", value=2020)
        validate_metadata_filter(filter2, TestMetadata)  # Should not raise

        # Float field with comparison
        filter3 = MetadataFilter(field_name="score", condition="lt", value=0.8)
        validate_metadata_filter(filter3, TestMetadata)  # Should not raise

        # Boolean field with eq
        filter4 = MetadataFilter(field_name="published", condition="eq", value=True)
        validate_metadata_filter(filter4, TestMetadata)  # Should not raise

        # List field with contains
        filter5 = MetadataFilter(field_name="tags", condition="contains", value="AI")
        validate_metadata_filter(filter5, TestMetadata)  # Should not raise

    def test_valid_nested_field_filters(self):
        """Test validation of filters on nested fields."""
        # Nested string field
        filter1 = MetadataFilter(field_name="publication_info.journal", condition="eq", value="Nature")
        validate_metadata_filter(filter1, NestedMetadata)  # Should not raise

        # Nested integer field
        filter2 = MetadataFilter(field_name="publication_info.volume", condition="gt", value=10)
        validate_metadata_filter(filter2, NestedMetadata)  # Should not raise

        # Nested list field
        filter3 = MetadataFilter(field_name="publication_info.pages", condition="contains", value=42)
        validate_metadata_filter(filter3, NestedMetadata)  # Should not raise

    def test_nonexistent_field_error(self):
        """Test error when field doesn't exist in schema."""
        filter_obj = MetadataFilter(field_name="nonexistent", condition="eq", value="test")
        
        with self.assertRaises(ValueError) as context:
            validate_metadata_filter(filter_obj, TestMetadata)
        
        self.assertIn("Field 'nonexistent' not found in metadata schema", str(context.exception))

    def test_nonexistent_nested_field_error(self):
        """Test error when nested field doesn't exist."""
        filter_obj = MetadataFilter(field_name="publication_info.nonexistent", condition="eq", value="test")
        
        with self.assertRaises(ValueError) as context:
            validate_metadata_filter(filter_obj, NestedMetadata)
        
        # The actual error message depends on implementation details
        self.assertTrue("nonexistent" in str(context.exception))

    def test_invalid_nested_navigation_error(self):
        """Test error when trying to navigate into non-model field."""
        filter_obj = MetadataFilter(field_name="author.invalid", condition="eq", value="test")
        
        with self.assertRaises(ValueError) as context:
            validate_metadata_filter(filter_obj, TestMetadata)
        
        self.assertIn("Cannot navigate into non-model field 'author'", str(context.exception))

    def test_condition_compatibility_validation(self):
        """Test that condition compatibility is validated."""
        # Invalid condition for string field
        filter_obj = MetadataFilter(field_name="author", condition="gt", value="test")
        
        with self.assertRaises(ValueError) as context:
            validate_metadata_filter(filter_obj, TestMetadata)
        
        self.assertIn("Condition 'gt' not valid for field type str", str(context.exception))


class TestValidateMetadataFilters(unittest.TestCase):
    """Test validate_metadata_filters function."""

    def test_valid_filters_list(self):
        """Test validation of list with all valid filters."""
        filters = [
            MetadataFilter(field_name="author", condition="eq", value="John Doe"),
            MetadataFilter(field_name="year", condition="gte", value=2020),
            MetadataFilter(field_name="tags", condition="contains", value="AI")
        ]
        validate_metadata_filters(filters, TestMetadata)  # Should not raise

    def test_empty_filters_list(self):
        """Test validation of empty filters list."""
        validate_metadata_filters([], TestMetadata)  # Should not raise

    def test_invalid_filter_in_list(self):
        """Test that invalid filter in list raises error."""
        filters = [
            MetadataFilter(field_name="author", condition="eq", value="John Doe"),
            MetadataFilter(field_name="nonexistent", condition="eq", value="test")
        ]
        with self.assertRaises(ValueError) as context:
            validate_metadata_filters(filters, TestMetadata)
        
        self.assertIn("Field 'nonexistent' not found in metadata schema", str(context.exception))


class TestValidateConditionCompatibility(unittest.TestCase):
    """Test validate_condition_compatibility function."""

    def test_string_field_conditions(self):
        """Test valid and invalid conditions for string fields."""
        # Valid conditions
        validate_condition_compatibility(str, "eq")  # Should not raise
        validate_condition_compatibility(str, "exists")  # Should not raise
        
        # Invalid conditions
        with self.assertRaises(ValueError):
            validate_condition_compatibility(str, "gt")
        
        with self.assertRaises(ValueError):
            validate_condition_compatibility(str, "contains")

    def test_integer_field_conditions(self):
        """Test valid and invalid conditions for integer fields."""
        # Valid conditions
        for condition in ["eq", "gt", "gte", "lt", "lte", "exists"]:
            validate_condition_compatibility(int, condition)  # Should not raise
        
        # Invalid conditions
        with self.assertRaises(ValueError):
            validate_condition_compatibility(int, "contains")
        
        with self.assertRaises(ValueError):
            validate_condition_compatibility(int, "in")

    def test_float_field_conditions(self):
        """Test valid and invalid conditions for float fields."""
        # Valid conditions
        for condition in ["eq", "gt", "gte", "lt", "lte", "exists"]:
            validate_condition_compatibility(float, condition)  # Should not raise
        
        # Invalid conditions
        with self.assertRaises(ValueError):
            validate_condition_compatibility(float, "contains")

    def test_boolean_field_conditions(self):
        """Test valid and invalid conditions for boolean fields."""
        # Valid conditions
        validate_condition_compatibility(bool, "eq")  # Should not raise
        validate_condition_compatibility(bool, "exists")  # Should not raise
        
        # Invalid conditions
        with self.assertRaises(ValueError):
            validate_condition_compatibility(bool, "gt")
        
        with self.assertRaises(ValueError):
            validate_condition_compatibility(bool, "contains")

    def test_list_field_conditions(self):
        """Test valid and invalid conditions for list fields."""
        # Valid conditions
        for condition in ["contains", "in", "exists"]:
            validate_condition_compatibility(list, condition)  # Should not raise
        
        # Invalid conditions
        with self.assertRaises(ValueError):
            validate_condition_compatibility(list, "eq")
        
        with self.assertRaises(ValueError):
            validate_condition_compatibility(list, "gt")

    def test_unknown_field_type_defaults(self):
        """Test that unknown field types default to eq and exists conditions."""
        class CustomType:
            pass
        
        # Valid default conditions
        validate_condition_compatibility(CustomType, "eq")  # Should not raise
        validate_condition_compatibility(CustomType, "exists")  # Should not raise
        
        # Invalid conditions
        with self.assertRaises(ValueError):
            validate_condition_compatibility(CustomType, "gt")


if __name__ == "__main__":
    unittest.main()