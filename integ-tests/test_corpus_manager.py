"""
Integration tests for the BaseCorpusManager class
"""

import numpy as np
from textwrap import dedent
from typing import Any, Type
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from sqlalchemy.exc import IntegrityError

from pgvector_template.core.document import (
    BaseDocument,
    BaseDocumentMetadata,
    BaseDocumentOptionalProps,
)
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

    def _extract_chunk_metadata(self, content: str, **kwargs) -> dict[str, Any]:
        """Extract metadata from chunk content"""
        # Add some basic metadata about the chunk
        return {
            "chunk_length": len(content),
            "word_count": len(content.split()),
        }

    def _join_documents(
        self, documents: list[BaseDocument], **kwargs
    ) -> tuple[str, dict[str, Any]]:
        documents.sort(key=lambda d: d.chunk_index)  # type: ignore
        corpus_content = self.corpus_delimiter.join(d.content for d in documents)  # type: ignore
        return corpus_content, self._infer_corpus_metadata(documents)

    def _infer_corpus_metadata(self, documents: list[BaseDocument], **kwargs) -> dict[str, Any]:
        return super()._infer_corpus_metadata(documents)


class TestCorpusManagerConfig(BaseCorpusManagerConfig):
    """Configuration for test corpus manager"""

    document_cls: Type[BaseDocument] = TestDocument
    embedding_provider: BaseEmbeddingProvider | None = SimpleEmbeddingProvider()
    document_metadata_cls: Type[BaseDocumentMetadata] = TestDocumentMetadata


class TestCorpusManagerIntegration:
    """Integration tests for the BaseCorpusManager class"""

    def test_paragraph_corpus_manager(self, database_url: str):
        """Test inserting a corpus with paragraph splitting"""
        # Create a temporary document database manager and schema
        db_manager = TempDocumentDatabaseManager(
            database_url=database_url,
            schema_suffix="corpus_mgr_test",
            document_classes=[TestDocument],
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
                metadata = {
                    "source": "integration_test",
                    "author": "test_user",
                    "created_at": "2023-01-01",
                }

                optional_props = BaseDocumentOptionalProps(
                    title="Test Document with Paragraphs",
                    collection="test_collection",
                    tags=["test", "paragraphs", "corpus"],
                )

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
                    assert any(
                        expected in content for content in contents
                    ), f"Expected content not found: {expected}"

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

    def test_insert_corpus_unique_constraint_violation(self, database_url: str):
        """Test that inserting the same corpus twice with update_if_exists=False raises SQLAlchemy exception"""
        db_manager = TempDocumentDatabaseManager(
            database_url=database_url, schema_suffix="unique_test", document_classes=[TestDocument]
        )
        temp_schema = db_manager.setup()

        try:
            with db_manager.get_session() as session:
                config = TestCorpusManagerConfig()
                manager = ParagraphCorpusManager(session, config)

                sample_text = "Test paragraph for unique constraint."
                metadata = {"source": "test", "author": "tester"}
                optional_props = BaseDocumentOptionalProps(collection="test_collection")

                # First insertion should succeed
                reused_uuid = uuid4()
                manager.insert_corpus(
                    sample_text,
                    metadata,
                    optional_props,
                    corpus_id=reused_uuid,
                    update_if_exists=False,
                )

                # Second insertion with same corpus_id should fail due to unique constraint on (collection, corpus_id, chunk_index)
                exception_raised = False
                try:
                    manager.insert_corpus(
                        sample_text,
                        metadata,
                        optional_props,
                        corpus_id=reused_uuid,
                        update_if_exists=False,
                    )
                except (IntegrityError, Exception) as e:
                    exception_raised = True
                    # Verify it's related to unique constraint
                    assert "unique" in str(e).lower() or "duplicate" in str(e).lower()

                assert exception_raised, "Expected database constraint violation exception"

        finally:
            db_manager.cleanup(temp_schema)

    def test_read_only_corpus_manager(self, database_url: str):
        """Test using corpus manager for read-only operations without embedding provider"""
        # Create a temporary document database manager and schema
        db_manager = TempDocumentDatabaseManager(
            database_url=database_url,
            schema_suffix="corpus_read_test",
            document_classes=[TestDocument],
        )
        temp_schema = db_manager.setup()

        try:
            # First, insert some test data using full config
            with db_manager.get_session() as session:
                full_config = TestCorpusManagerConfig()
                full_manager = ParagraphCorpusManager(session, full_config)

                sample_text = "First paragraph.\n\nSecond paragraph."
                metadata = {"source": "test", "author": "user"}
                full_manager.insert_corpus(sample_text, metadata)
                session.commit()

                # Now test read-only operations with minimal config
                read_config = BaseCorpusManagerConfig(document_cls=TestDocument)
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

    def test_insert_corpus_update_if_exists_true(self, database_url: str):
        """Test that inserting the same corpus twice with update_if_exists=True updates existing documents"""
        db_manager = TempDocumentDatabaseManager(
            database_url=database_url, schema_suffix="update_true_test", document_classes=[TestDocument]
        )
        temp_schema = db_manager.setup()

        try:
            with db_manager.get_session() as session:
                config = TestCorpusManagerConfig()
                manager = ParagraphCorpusManager(session, config)

                # Initial corpus
                original_text = "Original paragraph."
                original_metadata = {"source": "test", "author": "original_author"}
                optional_props = BaseDocumentOptionalProps(collection="test_collection")
                corpus_id = uuid4()

                # First insertion
                num_docs = manager.insert_corpus(
                    original_text,
                    original_metadata,
                    optional_props,
                    corpus_id=corpus_id,
                    update_if_exists=True,
                )
                assert num_docs == 1

                # Verify initial state
                docs = session.query(TestDocument).filter(TestDocument.corpus_id == corpus_id).all()
                assert len(docs) == 1
                assert docs[0].content == "Original paragraph."
                assert docs[0].document_metadata["author"] == "original_author"

                # Update with new content and metadata
                updated_text = "Updated paragraph with new content."
                updated_metadata = {"source": "test", "author": "updated_author"}

                num_docs = manager.insert_corpus(
                    updated_text,
                    updated_metadata,
                    optional_props,
                    corpus_id=corpus_id,
                    update_if_exists=True,
                )
                assert num_docs == 1

                # Verify update
                docs = session.query(TestDocument).filter(TestDocument.corpus_id == corpus_id).all()
                assert len(docs) == 1  # Still only one document
                assert docs[0].content == "Updated paragraph with new content."
                assert docs[0].document_metadata["author"] == "updated_author"

        finally:
            db_manager.cleanup(temp_schema)

    def test_insert_documents_update_if_exists_true(self, database_url: str):
        """Test that insert_documents with update_if_exists=True updates existing documents"""
        db_manager = TempDocumentDatabaseManager(
            database_url=database_url, schema_suffix="docs_update_test", document_classes=[TestDocument]
        )
        temp_schema = db_manager.setup()

        try:
            with db_manager.get_session() as session:
                config = TestCorpusManagerConfig()
                manager = ParagraphCorpusManager(session, config)
                corpus_id = uuid4()

                # Initial documents
                original_contents = ["First chunk", "Second chunk"]
                original_embeddings = manager.embedding_provider.embed_batch(original_contents)
                original_metadata = {"source": "test", "author": "original"}
                optional_props = BaseDocumentOptionalProps(collection="test_collection")

                num_docs = manager.insert_documents(
                    corpus_id,
                    original_contents,
                    original_embeddings,
                    original_metadata,
                    optional_props,
                    update_if_exists=True,
                )
                assert num_docs == 2

                # Verify initial state
                docs = session.query(TestDocument).filter(TestDocument.corpus_id == corpus_id).all()
                assert len(docs) == 2
                contents = [doc.content for doc in docs]
                assert "First chunk" in contents
                assert "Second chunk" in contents

                # Update with modified content
                updated_contents = ["Updated first chunk", "Updated second chunk"]
                updated_embeddings = manager.embedding_provider.embed_batch(updated_contents)
                updated_metadata = {"source": "test", "author": "updated"}

                num_docs = manager.insert_documents(
                    corpus_id,
                    updated_contents,
                    updated_embeddings,
                    updated_metadata,
                    optional_props,
                    update_if_exists=True,
                )
                assert num_docs == 2

                # Verify update
                docs = session.query(TestDocument).filter(TestDocument.corpus_id == corpus_id).all()
                assert len(docs) == 2  # Still only two documents
                contents = [doc.content for doc in docs]
                assert "Updated first chunk" in contents
                assert "Updated second chunk" in contents
                for doc in docs:
                    assert doc.document_metadata["author"] == "updated"

        finally:
            db_manager.cleanup(temp_schema)

    def test_insert_documents_update_if_exists_false(self, database_url: str):
        """Test that insert_documents with update_if_exists=False raises constraint violation"""
        db_manager = TempDocumentDatabaseManager(
            database_url=database_url, schema_suffix="docs_no_update_test", document_classes=[TestDocument]
        )
        temp_schema = db_manager.setup()

        try:
            with db_manager.get_session() as session:
                config = TestCorpusManagerConfig()
                manager = ParagraphCorpusManager(session, config)
                corpus_id = uuid4()

                # Initial documents
                contents = ["First chunk", "Second chunk"]
                embeddings = manager.embedding_provider.embed_batch(contents)
                metadata = {"source": "test", "author": "tester"}
                optional_props = BaseDocumentOptionalProps(collection="test_collection")

                # First insertion should succeed
                num_docs = manager.insert_documents(
                    corpus_id,
                    contents,
                    embeddings,
                    metadata,
                    optional_props,
                    update_if_exists=False,
                )
                assert num_docs == 2

                # Second insertion should fail
                exception_raised = False
                try:
                    manager.insert_documents(
                        corpus_id,
                        contents,
                        embeddings,
                        metadata,
                        optional_props,
                        update_if_exists=False,
                    )
                except (IntegrityError, Exception) as e:
                    exception_raised = True
                    assert "unique" in str(e).lower() or "duplicate" in str(e).lower()

                assert exception_raised, "Expected database constraint violation exception"

        finally:
            db_manager.cleanup(temp_schema)

    def test_insert_corpus_different_chunk_counts_update_if_exists_true(self, database_url: str):
        """Test updating corpus with different number of chunks using update_if_exists=True"""
        db_manager = TempDocumentDatabaseManager(
            database_url=database_url, schema_suffix="chunk_count_test", document_classes=[TestDocument]
        )
        temp_schema = db_manager.setup()

        try:
            with db_manager.get_session() as session:
                config = TestCorpusManagerConfig()
                manager = ParagraphCorpusManager(session, config)
                corpus_id = uuid4()

                # Initial corpus with 2 paragraphs
                original_text = "First paragraph.\n\nSecond paragraph."
                metadata = {"source": "test", "author": "tester"}
                optional_props = BaseDocumentOptionalProps(collection="test_collection")

                num_docs = manager.insert_corpus(
                    original_text,
                    metadata,
                    optional_props,
                    corpus_id=corpus_id,
                    update_if_exists=True,
                )
                assert num_docs == 2

                # Verify initial state
                docs = session.query(TestDocument).filter(TestDocument.corpus_id == corpus_id).all()
                assert len(docs) == 2

                # Update with 3 paragraphs
                updated_text = "Updated first.\n\nUpdated second.\n\nNew third paragraph."
                num_docs = manager.insert_corpus(
                    updated_text,
                    metadata,
                    optional_props,
                    corpus_id=corpus_id,
                    update_if_exists=True,
                )
                assert num_docs == 3

                # Verify update - should now have 3 documents (old ones deleted, new ones added)
                docs = session.query(TestDocument).filter(TestDocument.corpus_id == corpus_id).all()
                assert len(docs) == 3, f"Expected 3 documents after update, got {len(docs)}"
                contents = [doc.content for doc in docs]
                assert "Updated first." in contents, f"Expected 'Updated first.' in {contents}"
                assert "Updated second." in contents, f"Expected 'Updated second.' in {contents}"
                assert "New third paragraph." in contents, f"Expected 'New third paragraph.' in {contents}"

        finally:
            db_manager.cleanup(temp_schema)

    def test_insert_corpus_update_deletes_entire_corpus(self, database_url: str):
        """Test that updating a corpus deletes all documents with that corpus_id regardless of collection"""
        db_manager = TempDocumentDatabaseManager(
            database_url=database_url, schema_suffix="corpus_delete_test", document_classes=[TestDocument]
        )
        temp_schema = db_manager.setup()

        try:
            with db_manager.get_session() as session:
                config = TestCorpusManagerConfig()
                manager = ParagraphCorpusManager(session, config)
                corpus_id = uuid4()

                # Insert corpus with collection A
                text_a = "Collection A content."
                metadata = {"source": "test", "author": "tester"}
                props_a = BaseDocumentOptionalProps(collection="collection_a")

                manager.insert_corpus(
                    text_a,
                    metadata,
                    props_a,
                    corpus_id=corpus_id,
                    update_if_exists=True,
                )

                # Verify document exists
                docs = session.query(TestDocument).filter(TestDocument.corpus_id == corpus_id).all()
                assert len(docs) == 1
                assert docs[0].collection == "collection_a"
                assert docs[0].content == "Collection A content."

                # Update same corpus_id with different collection - should replace entirely
                text_b = "Collection B content."
                props_b = BaseDocumentOptionalProps(collection="collection_b")

                manager.insert_corpus(
                    text_b,
                    metadata,
                    props_b,
                    corpus_id=corpus_id,
                    update_if_exists=True,
                )

                # Verify only collection B document exists now
                docs = session.query(TestDocument).filter(TestDocument.corpus_id == corpus_id).all()
                assert len(docs) == 1
                assert docs[0].collection == "collection_b"
                assert docs[0].content == "Collection B content."

                # Verify collection A document was deleted
                docs_a = session.query(TestDocument).filter(
                    TestDocument.corpus_id == corpus_id,
                    TestDocument.collection == "collection_a"
                ).all()
                assert len(docs_a) == 0

        finally:
            db_manager.cleanup(temp_schema)
