import unittest
from unittest.mock import Mock

from sqlalchemy import select, Column
from pgvector.sqlalchemy import Vector

from pgvector_template.core.embedder import BaseEmbeddingProvider
from pgvector_template.core.search import BaseSearchClient, BaseSearchClientConfig
from pgvector_template.core import BaseDocument, BaseDocumentMetadata
from pgvector_template.models.search import SearchQuery, MetadataFilter


class TestMetadata(BaseDocumentMetadata):
    author: str
    year: int
    score: float
    published: bool
    tags: list[str]


class TestDocument(BaseDocument):
    __abstract__ = False
    __tablename__ = "test_search_documents"

    embedding = Column(Vector(3))


class TestEmbeddingProvider(BaseEmbeddingProvider):
    """Mock embedding provider for testing"""

    def embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def get_dimensions(self) -> int:
        return 3


class TestBaseSearchClient(unittest.TestCase):

    def setUp(self):
        self.mock_session = Mock()
        self.mock_embedding_provider = TestEmbeddingProvider()
        self.config = BaseSearchClientConfig(
            document_cls=TestDocument,
            document_metadata_cls=TestMetadata
        )
        self.client = BaseSearchClient(self.mock_session, self.config)

        # Config with embedding provider for tests that need it
        self.config_with_embedding = BaseSearchClientConfig(
            document_cls=TestDocument,
            embedding_provider=self.mock_embedding_provider,
            document_metadata_cls=TestMetadata
        )
        self.client_with_embedding = BaseSearchClient(self.mock_session, self.config_with_embedding)

    def test_apply_keyword_search_with_keywords(self):
        """Test that keyword search applies ILIKE conditions with OR logic"""
        base_query = select(TestDocument)
        search_query = SearchQuery(keywords=["python", "test"], limit=10)

        result_query = self.client._apply_keyword_search(base_query, search_query)

        query_str = str(result_query.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("content", query_str.lower())
        self.assertIn("python", query_str)
        self.assertIn("test", query_str)
        self.assertIn("OR", query_str)

    def test_apply_semantic_search_with_text(self):
        """Test that semantic search applies cosine distance ordering when text is provided"""
        base_query = select(TestDocument)
        search_query = SearchQuery(text="test query", limit=10)

        result_query = self.client_with_embedding._apply_semantic_search(base_query, search_query)

        query_str = str(result_query.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("ORDER BY", query_str)
        self.assertIn("<=>", query_str)  # cosine-distance operator in PGVector

    def test_apply_semantic_search_no_text(self):
        """Test that query is unchanged when no text is provided"""
        base_query = select(TestDocument)
        search_query = SearchQuery(keywords=["test"], limit=10)

        result_query = self.client_with_embedding._apply_semantic_search(base_query, search_query)
        self.assertEqual(str(base_query), str(result_query))

    def test_apply_metadata_filters_eq_string(self):
        """Test metadata filter with string equality"""
        base_query = select(TestDocument)
        filters = [MetadataFilter(field_name="author", condition="eq", value="John Doe")]
        search_query = SearchQuery(metadata_filters=filters, limit=10)

        result_query = self.client._apply_metadata_filters(base_query, search_query)
        query_str = str(result_query.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("test_search_documents.document_metadata ->> 'author') = 'John Doe'", query_str)

    def test_apply_metadata_filters_numeric_comparison(self):
        """Test metadata filter with numeric comparisons"""
        base_query = select(TestDocument)
        filters = [
            MetadataFilter(field_name="year", condition="gte", value=2020),
            MetadataFilter(field_name="score", condition="lt", value=0.8)
        ]
        search_query = SearchQuery(metadata_filters=filters, limit=10)

        result_query = self.client._apply_metadata_filters(base_query, search_query)
        query_str = str(result_query.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("2020", query_str)
        self.assertIn("0.8", query_str)

    def test_apply_metadata_filters_list_operations(self):
        """Test metadata filter with list contains and in operations"""
        base_query = select(TestDocument)
        filters = [
            MetadataFilter(field_name="tags", condition="contains", value="AI"),
            MetadataFilter(field_name="author", condition="in", value=["Alice", "Bob"])
        ]
        search_query = SearchQuery(metadata_filters=filters, limit=10)

        result_query = self.client._apply_metadata_filters(base_query, search_query)
        query_str = str(result_query.compile())
        self.assertIn("@>", query_str)  # PostgreSQL contains operator
        self.assertIn("IN", query_str)  # IN operator for author filter

    def test_apply_metadata_filters_exists(self):
        """Test metadata filter with exists condition"""
        base_query = select(TestDocument)
        filters = [MetadataFilter(field_name="author", condition="exists", value=True)]
        search_query = SearchQuery(metadata_filters=filters, limit=10)

        result_query = self.client._apply_metadata_filters(base_query, search_query)
        query_str = str(result_query.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("document_metadata", query_str)

    def test_apply_metadata_filters_no_filters(self):
        """Test that query is unchanged when no metadata filters are provided"""
        base_query = select(TestDocument)
        search_query = SearchQuery(keywords=["test"], limit=10)

        result_query = self.client._apply_metadata_filters(base_query, search_query)
        self.assertEqual(str(base_query), str(result_query))


if __name__ == "__main__":
    unittest.main()
