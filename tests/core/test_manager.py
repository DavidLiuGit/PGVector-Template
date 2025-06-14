import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from pgvector_template.core.manager import BaseDocumentManager
from pgvector_template.core.document import BaseDocumentMetadata


class ConcreteDocumentManager(BaseDocumentManager):
    """Concrete implementation of BaseDocumentManager for testing"""

    def create_chunks(self, content, metadata):
        """Implementation of abstract method for testing"""
        return [{"chunk": 1}, {"chunk": 2}]


class TestBaseDocumentManager(unittest.TestCase):
    """Unit tests for the BaseDocumentManager class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        self.session = MagicMock()
        self.schema_name = "test_schema"
        self.manager = ConcreteDocumentManager(self.session, self.schema_name)

    def test_init(self):
        """Test initialization of BaseDocumentManager"""
        self.assertEqual(self.manager.session, self.session)
        self.assertEqual(self.manager.schema_name, self.schema_name)

    def test_get_full_corpus_no_chunks(self):
        """Test get_full_corpus when no chunks are found"""
        # Setup mock query that returns no results
        mock_query = MagicMock()
        self.session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []

        # Call the method under test
        result = self.manager.get_full_corpus("test-corpus-id")

        # Assertions
        self.assertIsNone(result)
        self.session.query.assert_called_once()
        mock_query.filter.assert_called_once()
        mock_query.order_by.assert_called_once()
        mock_query.all.assert_called_once()

    def test_get_full_corpus_with_full_doc(self):
        """Test get_full_corpus when a full document (chunk_index=0) exists"""
        # Setup mock documents
        corpus_id = str(uuid4())
        full_doc = MagicMock()
        full_doc.original_id = corpus_id
        full_doc.chunk_index = 0
        full_doc.content = "Full document content"
        full_doc.metadata = {"key": "value"}
        full_doc.id = str(uuid4())
        full_doc.title = "Full Document"

        chunk1 = MagicMock()
        chunk1.chunk_index = 1
        chunk1.id = str(uuid4())
        chunk1.title = "Chunk 1"

        # Setup mock query that returns our test documents
        mock_query = MagicMock()
        self.session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = [full_doc, chunk1]

        # Call the method under test
        result = self.manager.get_full_corpus(corpus_id)

        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], full_doc.original_id)
        self.assertEqual(result["content"], full_doc.content)
        self.assertEqual(result["metadata"], full_doc.metadata)
        self.assertEqual(len(result["chunks"]), 1)
        self.assertEqual(result["chunks"][0]["id"], chunk1.id)
        self.assertEqual(result["chunks"][0]["index"], chunk1.chunk_index)

    def test_get_full_corpus_reconstruct_from_chunks(self):
        """Test get_full_corpus when reconstructing from chunks"""
        # Setup mock documents (no chunk_index=0)
        corpus_id = str(uuid4())

        chunk1 = MagicMock()
        chunk1.chunk_index = 1
        chunk1.content = "Chunk 1 content"
        chunk1.metadata = {"key": "value"}
        chunk1.id = str(uuid4())
        chunk1.title = "Chunk 1"

        chunk2 = MagicMock()
        chunk2.chunk_index = 2
        chunk2.content = "Chunk 2 content"
        chunk2.id = str(uuid4())
        chunk2.title = "Chunk 2"

        # Setup mock query that returns our test documents
        mock_query = MagicMock()
        self.session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = [chunk1, chunk2]

        # Call the method under test
        result = self.manager.get_full_corpus(corpus_id)

        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], corpus_id)
        self.assertEqual(result["content"], "Chunk 1 content\nChunk 2 content")
        self.assertEqual(result["metadata"], chunk1.metadata)
        self.assertEqual(len(result["chunks"]), 2)

    def test_search_by_metadata_simple_filter(self):
        """Test search_by_metadata with simple filter"""
        # Setup
        filters = {"category": "test"}
        limit = 5
        expected_results = ["result1", "result2"]

        # Mock query chain
        mock_query = MagicMock()
        self.session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.params.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = expected_results

        # Call the method under test
        results = self.manager.search_by_metadata(filters, limit)

        # Assertions
        self.assertEqual(results, expected_results)
        self.session.query.assert_called_once()
        self.assertEqual(mock_query.filter.call_count, 2)  # One for is_deleted, one for the filter
        mock_query.params.assert_called_once()
        mock_query.limit.assert_called_once_with(limit)
        mock_query.all.assert_called_once()

    def test_search_by_metadata_list_filter(self):
        """Test search_by_metadata with list filter"""
        # Setup
        filters = {"tags": ["tag1", "tag2"]}
        expected_results = ["result1"]

        # Mock query chain
        mock_query = MagicMock()
        self.session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.params.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = expected_results

        # Call the method under test
        results = self.manager.search_by_metadata(filters)

        # Assertions
        self.assertEqual(results, expected_results)
        self.session.query.assert_called_once()
        self.assertEqual(mock_query.filter.call_count, 2)  # One for is_deleted, one for the filter
        mock_query.params.assert_called_once()
        mock_query.limit.assert_called_once_with(10)  # Default limit
        mock_query.all.assert_called_once()

    def test_search_by_metadata_nested_filter(self):
        """Test search_by_metadata with nested filter"""
        # Setup
        filters = {"author": {"name": "John Doe"}}
        expected_results = ["result1"]

        # Mock query chain
        mock_query = MagicMock()
        self.session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.params.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = expected_results

        # Call the method under test
        results = self.manager.search_by_metadata(filters)

        # Assertions
        self.assertEqual(results, expected_results)
        self.session.query.assert_called_once()
        self.assertEqual(mock_query.filter.call_count, 2)  # One for is_deleted, one for the filter
        mock_query.params.assert_called_once()
        mock_query.limit.assert_called_once_with(10)  # Default limit
        mock_query.all.assert_called_once()

    def test_create_chunks_abstract(self):
        """Test that create_chunks is implemented in concrete class"""
        metadata = BaseDocumentMetadata(document_type="test")
        result = self.manager.create_chunks("Test content", metadata)
        self.assertEqual(result, [{"chunk": 1}, {"chunk": 2}])


if __name__ == "__main__":
    unittest.main()
