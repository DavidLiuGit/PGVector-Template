"""
Integration tests for the BaseCorpusManager class
"""

import numpy as np
from textwrap import dedent
from typing import Any, Type

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column

from pgvector_template.core.document import BaseDocument, BaseDocumentMetadata, BaseDocumentOptionalProps
from pgvector_template.core.manager import BaseCorpusManager, BaseCorpusManagerConfig
from pgvector_template.core.embedder import BaseEmbeddingProvider
from pgvector_template.db import TempDocumentDatabaseManager


class TestDocument(BaseDocument):
    """Simple document class for testing"""

    __abstract__ = False
    __tablename__ = "test_paragraph_docs"

    embedding = Column(Vector(384))


class SimpleEmbeddingProvider(BaseEmbeddingProvider):
    """Simple embedding provider for testing"""

    def embed_text(self, text: str) -> list[float]:
        """Generate a simple deterministic embedding based on text length"""
        # Create a simple embedding based on text length (for testing only)
        base = np.ones(384) * (len(text) % 10) / 10
        return base.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts"""
        return [self.embed_text(text) for text in texts]

    def get_dimensions(self) -> int:
        """Return embedding dimensions"""
        return 384


class TestDocumentMetadata(BaseDocumentMetadata):
    """Test document metadata"""

    # corpus
    document_type: str = "paragraphs"
    schema_version: str = "1.0"
    source: str
    author: str

    # document chunks
    chunk_length: int
    word_count: int


class ParagraphCorpusManager(BaseCorpusManager):
    """
    Custom corpus manager that splits text on paragraph breaks.

    This demonstrates how to customize the document splitting behavior.
    """

    @property
    def corpus_delimiter(cls):
        return "\n\n"

    def _split_corpus(self, content: str, **kwargs) -> list[str]:
        """Split content on paragraph breaks (double newlines)"""
        # Skip empty paragraphs and strip whitespace
        paragraphs = [p.strip() for p in content.split(self.corpus_delimiter) if p.strip()]
        return paragraphs

    def _extract_chunk_metadata(self, content: str) -> dict[str, Any]:
        """Extract metadata from chunk content"""
        # Add some basic metadata about the chunk
        return {
            "chunk_length": len(content),
            "word_count": len(content.split()),
        }

    def _join_documents(self, documents: list[BaseDocument]) -> tuple[str, dict[str, Any]]:
        documents.sort(key=lambda d: d.chunk_index)  # type: ignore
        corpus_content = self.corpus_delimiter.join(d.content for d in documents)  # type: ignore
        return corpus_content, self._infer_corpus_metadata(documents)

    def _infer_corpus_metadata(self, documents: list[BaseDocument]) -> dict[str, Any]:
        return super()._infer_corpus_metadata(documents)


class TestCorpusManagerConfig(BaseCorpusManagerConfig):
    """Configuration for test corpus manager"""

    document_cls: Type[BaseDocument] = TestDocument
    embedding_provider: BaseEmbeddingProvider | None = SimpleEmbeddingProvider()
    document_metadata_cls: Type[BaseDocumentMetadata] | None = TestDocumentMetadata


class TestCorpusManagerIntegration:
    """Integration tests for the BaseCorpusManager class"""

    def test_paragraph_corpus_manager(self, database_url: str):
        """Test inserting a corpus with paragraph splitting"""
        # Create a temporary document database manager and schema
        db_manager = TempDocumentDatabaseManager(
            database_url=database_url, schema_suffix="corpus_mgr_test", document_classes=[TestDocument]
        )
        temp_schema = db_manager.setup()

        try:
            # Create a session
            with db_manager.get_session() as session:
                # Create corpus manager config and manager
                config = TestCorpusManagerConfig()
                manager = ParagraphCorpusManager(session, config)

                # Sample multi-paragraph text
                sample_text = dedent(
                    """\
                    This is the first paragraph of the test document.
                    It contains multiple sentences that should be kept together.
                    
                    This is the second paragraph with different content.
                    We want to make sure this is split properly.
                    
                    Finally, this is the third paragraph.
                    It should be treated as a separate chunk.
                    """
                )

                # Create metadata and optional properties
                metadata = {"source": "integration_test", "author": "test_user", "created_at": "2023-01-01"}

                optional_props = BaseDocumentOptionalProps(
                    title="Test Document with Paragraphs",
                    collection="test_collection",
                    tags=["test", "paragraphs", "corpus"],
                )  # type: ignore

                # Insert the corpus
                num_docs = manager.insert_corpus(sample_text, metadata, optional_props)

                # Verify the number of documents inserted (should be 3 paragraphs)
                assert num_docs == 3, f"Expected 3 documents, got {num_docs}"

                # Query the documents to verify they were inserted correctly
                docs = session.query(TestDocument).all()
                assert len(docs) == 3, f"Expected 3 documents in database, got {len(docs)}"

                # Check that all documents have the same corpus_id
                corpus_ids = {doc.corpus_id for doc in docs}
                assert len(corpus_ids) == 1, "All documents should have the same corpus_id"

                # Check that documents have the correct content
                contents = [doc.content.strip() for doc in docs]
                expected_contents = [
                    "This is the first paragraph of the test document",
                    "This is the second paragraph with different content",
                    "Finally, this is the third paragraph",
                ]

                for expected in expected_contents:
                    assert any(expected in content for content in contents), f"Expected content not found: {expected}"

                # Check that metadata was properly combined
                for doc in docs:
                    # Base metadata
                    assert doc.document_metadata["source"] == "integration_test"  # type: ignore
                    assert doc.document_metadata["author"] == "test_user"  # type: ignore

                    # Custom metadata from _extract_chunk_metadata
                    assert "chunk_length" in doc.document_metadata
                    assert "word_count" in doc.document_metadata

                # Check optional properties
                for doc in docs:
                    assert doc.title == "Test Document with Paragraphs"  # type: ignore
                    assert doc.collection == "test_collection"  # type: ignore
                    assert doc.tags == ["test", "paragraphs", "corpus"]  # type: ignore

                # Test retrieving the full corpus
                corpus_id = str(list(corpus_ids)[0])
                full_corpus = manager.get_full_corpus(corpus_id)

                assert full_corpus is not None
                assert full_corpus.corpus_id == corpus_id
                assert len(full_corpus.documents) == 3

                # The content should be joined without delimiters (as per _join_documents)
                expected_combined = "\n\n".join(
                    [
                        "This is the first paragraph of the test document.\nIt contains multiple sentences that should be kept together.",
                        "This is the second paragraph with different content.\nWe want to make sure this is split properly.",
                        "Finally, this is the third paragraph.\nIt should be treated as a separate chunk.",
                    ]
                )
                assert full_corpus.content == expected_combined

                # Check that metadata was inferred correctly (last document's metadata wins)
                assert "source" in full_corpus.metadata
                assert "author" in full_corpus.metadata

        finally:
            # Clean up temp schema
            db_manager.cleanup(temp_schema)

    def test_read_only_corpus_manager(self, database_url: str):
        """Test using corpus manager for read-only operations without embedding provider"""
        # Create a temporary document database manager and schema
        db_manager = TempDocumentDatabaseManager(
            database_url=database_url, schema_suffix="corpus_read_test", document_classes=[TestDocument]
        )
        temp_schema = db_manager.setup()

        try:
            # First, insert some test data using full config
            with db_manager.get_session() as session:
                full_config = TestCorpusManagerConfig()
                full_manager = ParagraphCorpusManager(session, full_config)

                sample_text = "First paragraph.\n\nSecond paragraph."
                metadata = {"source": "test", "author": "user"}
                corpus_id = full_manager.insert_corpus(sample_text, metadata)
                session.commit()

                # Now test read-only operations with minimal config
                read_config = BaseCorpusManagerConfig(document_cls=TestDocument)  # type: ignore
                read_manager = ParagraphCorpusManager(session, read_config)

                # Should be able to read corpus
                docs = session.query(TestDocument).all()
                corpus = read_manager.get_full_corpus(str(docs[0].corpus_id))

                assert corpus is not None
                assert len(corpus.documents) == 2
                assert "First paragraph." in corpus.content
                assert "Second paragraph." in corpus.content

                # Should fail on insert operations
                try:
                    read_manager.insert_corpus("test", {})
                    assert False, "Should have raised ValueError"
                except ValueError as e:
                    assert "embedding_provider must be provided" in str(e)

        finally:
            # Clean up temp schema
            db_manager.cleanup(temp_schema)
