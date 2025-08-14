import unittest
from typing import Type

from pydantic import Field

from pgvector_template.core import BaseDocumentMetadata
from pgvector_template.models.search import SearchQuery


class CustomDocumentMetadata(BaseDocumentMetadata):
    """Custom metadata for testing field schema injection."""

    author: str = Field(..., description="Document author")
    publication_year: int = Field(..., description="Year of publication")


class TestSearchQueryFieldSchema(unittest.TestCase):
    """Test SearchQuery metadata schema injection via json_schema_extra."""

    def test_metadata_schema_in_field_extra(self):
        """Test that metadata schema is in metadata_filters field json_schema_extra."""
        schema = SearchQuery.model_json_schema()

        # Get metadata_filters field schema
        metadata_filters_field = schema["properties"]["metadata_filters"]

        # Verify metadata_schema is in json_schema_extra
        self.assertIn("metadata_schema", metadata_filters_field)
        metadata_schema = metadata_filters_field["metadata_schema"]

        # Verify it contains BaseDocumentMetadata schema
        self.assertIn("properties", metadata_schema)
        self.assertIn("document_type", metadata_schema["properties"])
        self.assertIn("schema_version", metadata_schema["properties"])

    def test_custom_metadata_schema_injection(self):
        """Test custom metadata schema injection via subclass."""

        class CustomSearchQuery(SearchQuery):
            random_field: str = Field(..., description="Random field")

            metadata_filters: list = Field(
                default=[],
                json_schema_extra={"metadata_schema": CustomDocumentMetadata.model_json_schema()},
            )

        schema = CustomSearchQuery.model_json_schema()

        # Get metadata_filters field schema
        metadata_filters_field = schema["properties"]["metadata_filters"]
        metadata_schema = metadata_filters_field["metadata_schema"]

        # Verify custom fields are present
        self.assertIn("author", metadata_schema["properties"])
        self.assertIn("publication_year", metadata_schema["properties"])

    def test_field_description_includes_schema_reference(self):
        """Test that field description references metadata_schema."""
        schema = SearchQuery.model_json_schema()

        metadata_filters_field = schema["properties"]["metadata_filters"]
        description = metadata_filters_field.get("description", "")

        # Verify description mentions metadata_schema
        self.assertIn("metadata_schema", description)

    def test_search_query_instantiation_works(self):
        """Test that SearchQuery can still be instantiated normally."""
        query = SearchQuery(text="test query", limit=10)

        self.assertEqual(query.text, "test query")
        self.assertEqual(query.limit, 10)
        self.assertEqual(query.metadata_filters, [])


if __name__ == "__main__":
    unittest.main()
