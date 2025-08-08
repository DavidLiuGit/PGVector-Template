import unittest
from unittest.mock import Mock

from sqlalchemy import Column
from pgvector.sqlalchemy import Vector

from pgvector_template.core.search import BaseSearchClient, BaseSearchClientConfig
from pgvector_template.core import BaseDocument, BaseDocumentMetadata
from pgvector_template.models.search import MetadataFilter


class TestMetadata(BaseDocumentMetadata):
    author: str
    year: int
    score: float
    published: bool
    tags: list[str]


class NestedInfo(BaseDocumentMetadata):
    journal: str
    volume: int


class NestedMetadata(BaseDocumentMetadata):
    info: NestedInfo
    category: str


class TestDocument(BaseDocument):
    __abstract__ = False
    __tablename__ = "test_documents"
    embedding = Column(Vector(3))


class TestBuildMetadataFilterWhereCondition(unittest.TestCase):

    def setUp(self):
        self.mock_session = Mock()
        self.config = BaseSearchClientConfig(
            document_cls=TestDocument, document_metadata_cls=TestMetadata
        )
        self.client = BaseSearchClient(self.mock_session, self.config)

    def test_eq_condition_string(self):
        """Test equality condition with string value."""
        filter_obj = MetadataFilter(field_name="author", condition="eq", value="John Doe")
        condition = self.client._build_metadata_filter_where_condition(filter_obj)

        query_str = str(condition.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("->> 'author'", query_str)
        self.assertIn("John Doe", query_str)

    def test_eq_condition_non_string(self):
        """Test equality condition with non-string value."""
        filter_obj = MetadataFilter(field_name="year", condition="eq", value=2023)
        condition = self.client._build_metadata_filter_where_condition(filter_obj)

        query_str = str(condition.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("['year']", query_str)
        self.assertIn("2023", query_str)

    def test_comparison_conditions_string(self):
        """Test comparison conditions with string values."""
        test_cases = [
            ("gt", "2023-01-01"),
            ("gte", "2023-01-01"),
            ("lt", "2023-12-31"),
            ("lte", "2023-12-31"),
        ]

        for condition, value in test_cases:
            with self.subTest(condition=condition):
                filter_obj = MetadataFilter(field_name="author", condition=condition, value=value)
                result = self.client._build_metadata_filter_where_condition(filter_obj)

                query_str = str(result.compile(compile_kwargs={"literal_binds": True}))
                self.assertIn("->> 'author'", query_str)
                self.assertIn(value, query_str)

    def test_comparison_conditions_integer(self):
        """Test comparison conditions with integer values."""
        test_cases = [("gt", 2020), ("gte", 2020), ("lt", 2025), ("lte", 2025)]

        for condition, value in test_cases:
            with self.subTest(condition=condition):
                filter_obj = MetadataFilter(field_name="year", condition=condition, value=value)
                result = self.client._build_metadata_filter_where_condition(filter_obj)

                query_str = str(result.compile(compile_kwargs={"literal_binds": True}))
                self.assertIn("CAST", query_str.upper())
                self.assertIn(str(value), query_str)

    def test_comparison_conditions_float(self):
        """Test comparison conditions with float values."""
        filter_obj = MetadataFilter(field_name="score", condition="gte", value=0.75)
        result = self.client._build_metadata_filter_where_condition(filter_obj)

        query_str = str(result.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("CAST", query_str.upper())
        self.assertIn("0.75", query_str)

    def test_contains_condition(self):
        """Test contains condition for JSONB arrays."""
        filter_obj = MetadataFilter(field_name="tags", condition="contains", value="AI")
        result = self.client._build_metadata_filter_where_condition(filter_obj)

        query_str = str(result.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("tags", query_str)
        self.assertIn("AI", query_str)

    def test_in_condition_single_value(self):
        """Test in condition with single value."""
        filter_obj = MetadataFilter(field_name="author", condition="in", value=["John"])
        result = self.client._build_metadata_filter_where_condition(filter_obj)

        query_str = str(result.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("->> 'author'", query_str)
        self.assertIn("John", query_str)

    def test_in_condition_multiple_values(self):
        """Test in condition with multiple values."""
        filter_obj = MetadataFilter(
            field_name="author", condition="in", value=["Alice", "Bob", "Charlie"]
        )
        result = self.client._build_metadata_filter_where_condition(filter_obj)

        query_str = str(result.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("->> 'author'", query_str)
        self.assertIn("Alice", query_str)
        self.assertIn("Bob", query_str)
        self.assertIn("Charlie", query_str)

    def test_in_condition_mixed_types(self):
        """Test in condition with mixed value types."""
        filter_obj = MetadataFilter(
            field_name="category", condition="in", value=["tech", 123, True]
        )
        result = self.client._build_metadata_filter_where_condition(filter_obj)

        query_str = str(result.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("tech", query_str)
        self.assertIn("123", query_str)
        self.assertIn("True", query_str)

    def test_exists_condition_simple_field(self):
        """Test exists condition for simple field."""
        filter_obj = MetadataFilter(field_name="author", condition="exists", value=True)
        result = self.client._build_metadata_filter_where_condition(filter_obj)

        query_str = str(result.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("document_metadata", query_str)
        self.assertIn("author", query_str)

    def test_exists_condition_nested_field(self):
        """Test exists condition for nested field."""
        nested_config = BaseSearchClientConfig(
            document_cls=TestDocument, document_metadata_cls=NestedMetadata
        )
        nested_client = BaseSearchClient(self.mock_session, nested_config)

        filter_obj = MetadataFilter(field_name="info.journal", condition="exists", value=True)
        result = nested_client._build_metadata_filter_where_condition(filter_obj)

        query_str = str(result.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("document_metadata", query_str)
        self.assertIn("info", query_str)
        self.assertIn("journal", query_str)

    def test_nested_field_navigation(self):
        """Test field navigation for nested structures."""
        nested_config = BaseSearchClientConfig(
            document_cls=TestDocument, document_metadata_cls=NestedMetadata
        )
        nested_client = BaseSearchClient(self.mock_session, nested_config)

        filter_obj = MetadataFilter(field_name="info.journal", condition="eq", value="Nature")
        result = nested_client._build_metadata_filter_where_condition(filter_obj)

        query_str = str(result.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("info", query_str)
        self.assertIn("journal", query_str)
        self.assertIn("Nature", query_str)

    def test_deep_nested_field(self):
        """Test deeply nested field path."""
        nested_config = BaseSearchClientConfig(
            document_cls=TestDocument, document_metadata_cls=NestedMetadata
        )
        nested_client = BaseSearchClient(self.mock_session, nested_config)

        filter_obj = MetadataFilter(field_name="info.volume", condition="gt", value=10)
        result = nested_client._build_metadata_filter_where_condition(filter_obj)

        query_str = str(result.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("info", query_str)
        self.assertIn("volume", query_str)
        self.assertIn("10", query_str)

    def test_unsupported_condition_raises_error(self):
        """Test that unsupported condition raises ValueError."""
        # Create filter with valid condition first, then modify to test error handling
        filter_obj = MetadataFilter(field_name="author", condition="eq", value="test")
        filter_obj.condition = "regex"  # Bypass pydantic validation

        with self.assertRaises(ValueError) as context:
            self.client._build_metadata_filter_where_condition(filter_obj)

        self.assertIn("Unsupported condition: regex", str(context.exception))

    def test_edge_case_empty_string_value(self):
        """Test edge case with empty string value."""
        filter_obj = MetadataFilter(field_name="author", condition="eq", value="")
        result = self.client._build_metadata_filter_where_condition(filter_obj)

        query_str = str(result.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("->> 'author'", query_str)

    def test_edge_case_zero_values(self):
        """Test edge cases with zero values."""
        test_cases = [(0, "eq"), (0.0, "gte"), (0, "lt")]

        for value, condition in test_cases:
            with self.subTest(value=value, condition=condition):
                field_name = "year" if isinstance(value, int) else "score"
                filter_obj = MetadataFilter(field_name=field_name, condition=condition, value=value)
                result = self.client._build_metadata_filter_where_condition(filter_obj)

                query_str = str(result.compile(compile_kwargs={"literal_binds": True}))
                self.assertIn(str(value), query_str)

    def test_edge_case_negative_values(self):
        """Test edge cases with negative values."""
        filter_obj = MetadataFilter(field_name="score", condition="lt", value=-1.5)
        result = self.client._build_metadata_filter_where_condition(filter_obj)

        query_str = str(result.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("-1.5", query_str)

    def test_boolean_value_equality(self):
        """Test equality with boolean values."""
        filter_obj = MetadataFilter(field_name="published", condition="eq", value=True)
        result = self.client._build_metadata_filter_where_condition(filter_obj)

        query_str = str(result.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("published", query_str)


if __name__ == "__main__":
    unittest.main()
