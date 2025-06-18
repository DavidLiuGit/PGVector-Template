import unittest
from unittest.mock import MagicMock, patch
from typing import Type
from uuid import uuid4

from pgvector_template.core.manager import BaseCorpusManager, BaseCorpusManagerConfig
from pgvector_template.core.document import BaseDocument, BaseDocumentMetadata, BaseDocumentOptionalProps
from pgvector_template.core.embedder import BaseEmbeddingProvider


class MockDocument(BaseDocument):
    """Mock document class for testing"""

    __tablename__ = "mock_documents"


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Mock embedding provider for testing"""

    def embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def get_dimensions(self) -> int:
        return 3


class MockDocumentMetadata(BaseDocumentMetadata):
    """Mock document metadata for testing"""

    document_type: str = "mock"

    author: str = "mock author"
    source: str
    chunk_length: int


class MockConfig(BaseCorpusManagerConfig):
    """Mock configuration for testing"""

    schema_name: str = "test_schema"
    document_cls: Type[BaseDocument] = MockDocument
    embedding_provider: BaseEmbeddingProvider = MockEmbeddingProvider
    document_metadata_cls: Type[BaseDocumentMetadata] = MockDocumentMetadata


class ConcreteDocumentManager(BaseCorpusManager):
    """Concrete implementation of BaseDocumentManager for testing"""

    def create_chunks(self, content, metadata):
        """Implementation of abstract method for testing"""
        return [{"chunk": 1}, {"chunk": 2}]


class TestBaseDocumentManager(unittest.TestCase):
    """Unit tests for the BaseDocumentManager class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        self.session = MagicMock()
        self.config = MockConfig()
        self.manager = ConcreteDocumentManager(self.session, self.config)

    def test_init(self):
        """Test initialization of BaseDocumentManager"""
        self.assertEqual(self.manager.session, self.session)
        self.assertEqual(self.manager.schema_name, self.config.schema_name)

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
        full_doc.document_metadata = {"key": "value"}
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
        self.assertEqual(result["metadata"], full_doc.document_metadata)
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
        chunk1.document_metadata = {"key": "value"}
        chunk1.id = str(uuid4())
        chunk1.title = "Chunk 1"

        chunk2 = MagicMock()
        chunk2.chunk_index = 2
        chunk2.content = "Chunk 2 content"
        chunk2.document_metadata = {"key": "value"}
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
        self.assertEqual(result["metadata"], chunk1.document_metadata)
        self.assertEqual(len(result["chunks"]), 2)

    def test_insert_documents_success(self):
        """Test successful insertion of documents"""
        # Setup test data
        corpus_id = uuid4()
        document_contents = ["Document 1", "Document 2", "Document 3"]
        document_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        corpus_metadata = {"source": "test", "author": "tester"}
        optional_props = BaseDocumentOptionalProps(title="Test Document", collection="test_collection")

        # Mock document instances that will be returned by from_props
        mock_docs = [MagicMock() for _ in range(3)]

        # Mock the document creation
        with patch.object(MockDocument, "from_props") as mock_from_props:
            # Configure mock to return our mock documents
            mock_from_props.side_effect = mock_docs

            # Call the method under test
            result = self.manager.insert_documents(
                corpus_id, document_contents, document_embeddings, corpus_metadata, optional_props
            )

            # Assertions
            self.assertEqual(result, 3)  # Should return the number of documents inserted
            self.assertEqual(mock_from_props.call_count, 3)

            # Verify session operations
            self.session.add_all.assert_called_once_with(mock_docs)
            self.session.commit.assert_called_once()

            # Verify the calls to from_props with correct arguments
            calls = mock_from_props.call_args_list
            for i, call in enumerate(calls):
                args, kwargs = call
                self.assertEqual(kwargs["corpus_id"], corpus_id)
                self.assertEqual(kwargs["chunk_index"], i)
                self.assertEqual(kwargs["content"], document_contents[i])
                self.assertEqual(kwargs["embedding"], document_embeddings[i])
                self.assertEqual(kwargs["optional_props"], optional_props)
                # Check that metadata includes both corpus metadata and chunk metadata
                self.assertIn("source", kwargs["metadata"])
                self.assertIn("author", kwargs["metadata"])
                self.assertIn("chunk_length", kwargs["metadata"])

    def test_insert_documents_mismatch_error(self):
        """Test error handling when document contents and embeddings don't match"""
        # Setup test data with mismatched lengths
        corpus_id = uuid4()
        document_contents = ["Document 1", "Document 2"]
        document_embeddings = [[0.1, 0.2, 0.3]]  # Only one embedding for two documents
        corpus_metadata = {"source": "test"}

        # Verify that ValueError is raised
        with self.assertRaises(ValueError) as context:
            self.manager.insert_documents(corpus_id, document_contents, document_embeddings, corpus_metadata)

        self.assertIn("Number of embeddings does not match number of documents", str(context.exception))

    def test_insert_documents_with_extract_chunk_metadata(self):
        """Test that _extract_chunk_metadata is called for each document"""
        # Setup test data
        corpus_id = uuid4()
        document_contents = ["Short doc", "Longer document content"]
        document_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        corpus_metadata = {"source": "test"}

        # Mock document instances
        mock_docs = [MagicMock(), MagicMock()]

        # Mock the _extract_chunk_metadata method
        with patch.object(self.manager, "_extract_chunk_metadata") as mock_extract:
            mock_extract.side_effect = lambda content: {"chunk_length": len(content)}

            # Call the method under test
            with patch.object(MockDocument, "from_props", side_effect=mock_docs):
                self.manager.insert_documents(corpus_id, document_contents, document_embeddings, corpus_metadata)

            # Verify _extract_chunk_metadata was called for each document
            self.assertEqual(mock_extract.call_count, 2)
            mock_extract.assert_any_call("Short doc")
            mock_extract.assert_any_call("Longer document content")

            # Verify session operations
            self.session.add_all.assert_called_once_with(mock_docs)
            self.session.commit.assert_called_once()

    def test_split_corpus(self):
        """Test _split_corpus method with built-in filtering"""
        # Test with content that includes empty chunks
        content = "First chunk\n\n\n   \nSecond chunk"

        # Call the method under test
        result = self.manager._split_corpus(content)

        # Verify that only non-empty chunks are returned
        for chunk in result:
            self.assertTrue(len(chunk.strip()) > 0)

    def test_insert_corpus(self):
        """Test insert_corpus method"""
        # Setup test data
        content = "Test corpus content"
        corpus_metadata = {"source": "test", "author": "tester"}
        optional_props = BaseDocumentOptionalProps(title="Test Corpus")

        # Mock the dependencies
        with (
            patch.object(self.manager, "_split_corpus") as mock_split,
            patch.object(self.config.embedding_provider, "embed_batch") as mock_embed,
            patch.object(self.manager, "insert_documents") as mock_insert,
            patch("pgvector_template.core.manager.uuid4") as mock_uuid4,
        ):
            # Configure mocks
            mock_uuid4.return_value = "test-uuid"
            mock_split.return_value = ["Chunk 1", "Chunk 2"]
            mock_embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
            mock_insert.return_value = 2  # Number of documents inserted

            # Call the method under test
            result = self.manager.insert_corpus(content, corpus_metadata, optional_props)

            # Assertions
            self.assertEqual(result, 2)  # Should return the result from insert_documents
            mock_split.assert_called_once_with(content)
            mock_embed.assert_called_once_with(["Chunk 1", "Chunk 2"])
            mock_insert.assert_called_once_with(
                "test-uuid", ["Chunk 1", "Chunk 2"], [[0.1, 0.2], [0.3, 0.4]], corpus_metadata, optional_props
            )

    def test_insert_corpus_empty_content(self):
        """Test insert_corpus with empty content"""
        # Setup test data
        content = ""
        corpus_metadata = {"source": "test"}
        optional_props = None

        # Mock the dependencies
        with (
            patch.object(self.manager, "_split_corpus") as mock_split,
            patch.object(self.config.embedding_provider, "embed_batch") as mock_embed,
            patch.object(self.manager, "insert_documents") as mock_insert,
        ):
            # Configure mocks
            mock_split.return_value = []  # Empty content results in no chunks
            mock_embed.return_value = []  # No embeddings for empty content
            mock_insert.return_value = 0  # No documents inserted

            # Call the method under test
            result = self.manager.insert_corpus(content, corpus_metadata, optional_props)

            # Assertions
            self.assertEqual(result, 0)
            mock_split.assert_called_once_with(content)
            mock_embed.assert_called_once_with([])
            mock_insert.assert_called_once()

    def test_insert_documents_empty_list(self):
        """Test insert_documents with empty lists"""
        # Setup test data
        corpus_id = uuid4()
        document_contents = []
        document_embeddings = []
        corpus_metadata = {"source": "test"}

        # Call the method under test
        result = self.manager.insert_documents(corpus_id, document_contents, document_embeddings, corpus_metadata)

        # Assertions
        self.assertEqual(result, 0)  # Should return 0 for empty lists

        # Verify that session operations are not called for empty lists
        self.session.add_all.assert_not_called()
        self.session.commit.assert_not_called()

    def test_insert_corpus_embedding_error(self):
        """Test insert_corpus when embedding fails"""
        # Setup test data
        content = "Test content"
        corpus_metadata = {"source": "test"}
        optional_props = None

        # Mock the dependencies
        with (
            patch.object(self.manager, "_split_corpus") as mock_split,
            patch.object(self.config.embedding_provider, "embed_batch") as mock_embed,
        ):
            # Configure mocks
            mock_split.return_value = ["Chunk 1"]
            mock_embed.side_effect = Exception("Embedding failed")

            # Verify that the exception is propagated
            with self.assertRaises(Exception) as context:
                self.manager.insert_corpus(content, corpus_metadata, optional_props)

            self.assertIn("Embedding failed", str(context.exception))

    def test_split_corpus_with_custom_implementation(self):
        """Test _split_corpus with custom implementation"""
        # Create content with some empty chunks that should be filtered out
        content = "Valid chunk 1\n\n   \n\nValid chunk 2\n\n\nValid chunk 3"

        # Call the actual implementation without mocking
        result = self.manager._split_corpus(content)

        # Verify that only valid chunks are returned
        for chunk in result:
            self.assertTrue(len(chunk.strip()) > 0)

        # Test with a custom implementation
        class CustomManager(BaseCorpusManager):
            def _split_corpus(self, content, **kwargs):
                # Split by newlines and only keep chunks with more than 10 characters
                chunks = content.split("\n")
                return [c for c in chunks if len(c.strip()) > 10]

        custom_manager = CustomManager(self.session, self.config)
        custom_result = custom_manager._split_corpus("Short\nLong enough chunk\nToo short")

        # Verify that only chunks with more than 10 characters are included
        for chunk in custom_result:
            self.assertTrue(len(chunk.strip()) > 10)


if __name__ == "__main__":
    unittest.main()
