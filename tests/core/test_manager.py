import unittest
from unittest.mock import MagicMock, patch
from typing import Type
from uuid import uuid4

from pgvector_template.core.manager import BaseCorpusManager, BaseCorpusManagerConfig
from pgvector_template.core.document import (
    BaseDocument,
    BaseDocumentMetadata,
    BaseDocumentOptionalProps,
)
from pgvector_template.core.embedder import BaseEmbeddingProvider


class MockDocument(BaseDocument):
    """Mock document class for testing"""

    __tablename__ = "mock_documents"


class TestEmbeddingProvider(BaseEmbeddingProvider):
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


class TestCorpusManagerConfig(BaseCorpusManagerConfig):
    """Mock configuration for testing"""

    schema_name: str = "test_schema"
    document_cls: Type[BaseDocument] = MockDocument
    embedding_provider: BaseEmbeddingProvider | None = TestEmbeddingProvider()
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
        self.config = TestCorpusManagerConfig()
        self.manager = ConcreteDocumentManager(self.session, self.config)

    def test_init(self):
        """Test initialization of BaseDocumentManager"""
        self.assertEqual(self.manager.session, self.session)
        self.assertEqual(self.manager.config, self.config)

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

    def test_get_full_corpus_reconstruct_from_chunks(self):
        """Test get_full_corpus when reconstructing from chunks"""
        # Setup mock documents (no chunk_index=0)
        corpus_id = str(uuid4())

        chunk1 = MagicMock()
        chunk1.chunk_index = 0
        chunk1.content = "Chunk 1 content\n"
        chunk1.document_metadata = {"key": "value"}
        chunk1.id = str(uuid4())
        chunk1.title = "Chunk 1"

        chunk2 = MagicMock()
        chunk2.chunk_index = 1
        chunk2.content = "Chunk 2 content"
        chunk2.document_metadata = {"key": "value1"}
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
        assert result  # for typing
        self.assertIsNotNone(result)
        self.assertEqual(result.corpus_id, corpus_id)
        self.assertEqual(result.content, "Chunk 1 content\nChunk 2 content")
        self.assertEqual(result.metadata, chunk2.document_metadata)
        self.assertEqual(len(result.documents), 2)

    def test_get_full_corpus_with_chunks(self):
        """Test get_full_corpus with chunks found"""
        corpus_id = "test-corpus-id"

        chunk1 = MagicMock(spec=BaseDocument)
        chunk1.chunk_index = 0
        chunk1.content = "First chunk"
        chunk1.document_metadata = {"key": "value1"}

        chunk2 = MagicMock(spec=BaseDocument)
        chunk2.chunk_index = 1
        chunk2.content = "Second chunk"
        chunk2.document_metadata = {"key": "value2"}

        chunks = [chunk1, chunk2]

        mock_query = MagicMock()
        self.session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = chunks

        with patch.object(self.manager, "_join_documents") as mock_join:
            mock_join.return_value = ("First chunkSecond chunk", {"key": "value2"})

            result = self.manager.get_full_corpus(corpus_id)

            self.assertIsNotNone(result)
            assert result  # for typing
            self.assertEqual(result.corpus_id, corpus_id)
            self.assertEqual(result.content, "First chunkSecond chunk")
            self.assertEqual(result.metadata, {"key": "value2"})
            self.assertEqual(result.documents, chunks)
            mock_join.assert_called_once_with(chunks)

    def test_get_full_corpus_calls_join_documents(self):
        """Test get_full_corpus delegates to _join_documents"""
        corpus_id = "test-corpus-id"
        chunks = [MagicMock(spec=BaseDocument)]

        mock_query = MagicMock()
        self.session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = chunks

        with patch.object(self.manager, "_join_documents") as mock_join:
            mock_join.return_value = ("content", {"meta": "data"})

            result = self.manager.get_full_corpus(corpus_id)

            mock_join.assert_called_once_with(chunks)
            assert result  # for typing
            self.assertEqual(result.content, "content")
            self.assertEqual(result.metadata, {"meta": "data"})

    def test_get_full_corpus_database_error(self):
        """Test get_full_corpus when database query fails"""
        self.session.query.side_effect = Exception("Database connection failed")

        with self.assertRaises(Exception) as context:
            self.manager.get_full_corpus("test-corpus-id")

        self.assertIn("Database connection failed", str(context.exception))

    def test_get_full_corpus_join_documents_error(self):
        """Test get_full_corpus when _join_documents raises exception"""
        chunks = [MagicMock(spec=BaseDocument)]

        mock_query = MagicMock()
        self.session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = chunks

        with patch.object(self.manager, "_join_documents") as mock_join:
            mock_join.side_effect = TypeError("document_metadata is None")

            with self.assertRaises(TypeError) as context:
                self.manager.get_full_corpus("test-corpus-id")

            self.assertIn("document_metadata is None", str(context.exception))

    def test_insert_documents_success(self):
        """Test successful insertion of documents"""
        # Setup test data
        corpus_id = uuid4()
        document_contents = ["Document 1", "Document 2", "Document 3"]
        document_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        corpus_metadata = {"source": "test", "author": "tester"}
        optional_props = BaseDocumentOptionalProps(
            title="Test Document", collection="test_collection"
        )

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
            self.manager.insert_documents(
                corpus_id, document_contents, document_embeddings, corpus_metadata
            )

        self.assertIn(
            "Number of embeddings does not match number of documents", str(context.exception)
        )

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
                self.manager.insert_documents(
                    corpus_id, document_contents, document_embeddings, corpus_metadata
                )

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
                "test-uuid",
                ["Chunk 1", "Chunk 2"],
                [[0.1, 0.2], [0.3, 0.4]],
                corpus_metadata,
                optional_props,
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
        result = self.manager.insert_documents(
            corpus_id, document_contents, document_embeddings, corpus_metadata
        )

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

    def test_infer_corpus_metadata_empty_list(self):
        """Test _infer_corpus_metadata with empty document list"""
        result = self.manager._infer_corpus_metadata([])
        self.assertEqual(result, {})

    def test_infer_corpus_metadata_single_document(self):
        """Test _infer_corpus_metadata with single document"""
        doc = MagicMock(spec=BaseDocument)
        doc.document_metadata = {"key1": "value1", "key2": "value2"}

        result = self.manager._infer_corpus_metadata([doc])
        self.assertEqual(result, {"key1": "value1", "key2": "value2"})

    def test_infer_corpus_metadata_multiple_documents_no_overlap(self):
        """Test _infer_corpus_metadata with multiple documents without key conflicts"""
        doc1 = MagicMock(spec=BaseDocument)
        doc1.document_metadata = {"key1": "value1"}
        doc2 = MagicMock(spec=BaseDocument)
        doc2.document_metadata = {"key2": "value2"}

        result = self.manager._infer_corpus_metadata([doc1, doc2])
        self.assertEqual(result, {"key1": "value1", "key2": "value2"})

    def test_infer_corpus_metadata_multiple_documents_with_overlap(self):
        """Test _infer_corpus_metadata with multiple documents with key conflicts"""
        doc1 = MagicMock(spec=BaseDocument)
        doc1.document_metadata = {"key1": "value1", "shared": "first"}
        doc2 = MagicMock(spec=BaseDocument)
        doc2.document_metadata = {"key2": "value2", "shared": "second"}

        result = self.manager._infer_corpus_metadata([doc1, doc2])
        self.assertEqual(result, {"key1": "value1", "key2": "value2", "shared": "second"})

    def test_infer_corpus_metadata_empty_metadata(self):
        """Test _infer_corpus_metadata with documents having empty metadata"""
        doc1 = MagicMock(spec=BaseDocument)
        doc1.document_metadata = {}
        doc2 = MagicMock(spec=BaseDocument)
        doc2.document_metadata = {"key1": "value1"}

        result = self.manager._infer_corpus_metadata([doc1, doc2])
        self.assertEqual(result, {"key1": "value1"})

    def test_infer_corpus_metadata_none_raises_error(self):
        """Test _infer_corpus_metadata raises error when document_metadata is None"""
        doc = MagicMock(spec=BaseDocument)
        doc.document_metadata = None

        with self.assertRaises(TypeError):
            self.manager._infer_corpus_metadata([doc])

    def test_infer_corpus_metadata_mixed_none_and_valid(self):
        """Test _infer_corpus_metadata with mix of None and valid metadata"""
        doc1 = MagicMock(spec=BaseDocument)
        doc1.document_metadata = {"key1": "value1"}
        doc2 = MagicMock(spec=BaseDocument)
        doc2.document_metadata = None

        with self.assertRaises(TypeError):
            self.manager._infer_corpus_metadata([doc1, doc2])

    def test_infer_corpus_metadata_nested_dict_values(self):
        """Test _infer_corpus_metadata with nested dictionary values"""
        doc1 = MagicMock(spec=BaseDocument)
        doc1.document_metadata = {"nested": {"key1": "value1"}, "simple": "value"}
        doc2 = MagicMock(spec=BaseDocument)
        doc2.document_metadata = {"nested": {"key2": "value2"}, "other": "data"}

        result = self.manager._infer_corpus_metadata([doc1, doc2])
        # Later document's nested dict overwrites the first
        self.assertEqual(result["nested"], {"key2": "value2"})
        self.assertEqual(result["simple"], "value")
        self.assertEqual(result["other"], "data")

    def test_join_documents_empty_list(self):
        """Test _join_documents with empty document list"""
        content, metadata = self.manager._join_documents([])
        self.assertEqual(content, "")
        self.assertEqual(metadata, {})

    def test_join_documents_single_document(self):
        """Test _join_documents with single document"""
        doc = MagicMock(spec=BaseDocument)
        doc.chunk_index = 0
        doc.content = "Single document content"
        doc.document_metadata = {"key": "value"}

        content, metadata = self.manager._join_documents([doc])
        self.assertEqual(content, "Single document content")
        self.assertEqual(metadata, {"key": "value"})

    def test_join_documents_multiple_ordered(self):
        """Test _join_documents with multiple documents in order"""
        doc1 = MagicMock(spec=BaseDocument)
        doc1.chunk_index = 0
        doc1.content = "First chunk"
        doc1.document_metadata = {"chunk": "first"}

        doc2 = MagicMock(spec=BaseDocument)
        doc2.chunk_index = 1
        doc2.content = "Second chunk"
        doc2.document_metadata = {"chunk": "second"}

        content, metadata = self.manager._join_documents([doc1, doc2])
        self.assertEqual(content, "First chunkSecond chunk")
        self.assertEqual(metadata, {"chunk": "second"})

    def test_join_documents_multiple_unordered(self):
        """Test _join_documents with documents in wrong order (should be sorted)"""
        doc1 = MagicMock(spec=BaseDocument)
        doc1.chunk_index = 2
        doc1.content = "Third chunk"
        doc1.document_metadata = {"order": "third"}

        doc2 = MagicMock(spec=BaseDocument)
        doc2.chunk_index = 0
        doc2.content = "First chunk"
        doc2.document_metadata = {"order": "first"}

        doc3 = MagicMock(spec=BaseDocument)
        doc3.chunk_index = 1
        doc3.content = "Second chunk"
        doc3.document_metadata = {"order": "second"}

        content, metadata = self.manager._join_documents([doc1, doc2, doc3])
        self.assertEqual(content, "First chunkSecond chunkThird chunk")
        self.assertEqual(metadata, {"order": "third"})

    def test_join_documents_calls_infer_corpus_metadata(self):
        """Test _join_documents calls _infer_corpus_metadata"""
        doc = MagicMock(spec=BaseDocument)
        doc.chunk_index = 0
        doc.content = "Test content"
        doc.document_metadata = {"key": "value"}

        with patch.object(self.manager, "_infer_corpus_metadata") as mock_infer:
            mock_infer.return_value = {"inferred": "metadata"}

            content, metadata = self.manager._join_documents([doc])

            mock_infer.assert_called_once_with([doc])
            self.assertEqual(metadata, {"inferred": "metadata"})


if __name__ == "__main__":
    unittest.main()
