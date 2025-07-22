"""
Unit tests for DocumentService
"""

import unittest
from unittest.mock import MagicMock, patch
from typing import Type
from uuid import uuid4

from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column

from pgvector_template.core.document import BaseDocument, BaseDocumentMetadata
from pgvector_template.core.embedder import BaseEmbeddingProvider
from pgvector_template.core.manager import BaseCorpusManager, BaseCorpusManagerConfig
from pgvector_template.core.search import BaseSearchClient, BaseSearchClientConfig
from pgvector_template.service.document_service import DocumentService, DocumentServiceConfig


# Test document classes
class TestDocument(BaseDocument):
    """Test document class"""

    __abstract__ = False
    __tablename__ = "test_document_service"

    embedding = Column(Vector(384))


class TestDocumentMetadata(BaseDocumentMetadata):
    """Test document metadata class"""

    document_type: str = "test"
    source: str = "test"


# Test embedding provider
class TestEmbeddingProvider(BaseEmbeddingProvider):
    """Test embedding provider"""

    def embed_text(self, text: str) -> list[float]:
        return [0.1] * 384

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]

    def get_dimensions(self) -> int:
        return 384


# Test corpus manager
class TestCorpusManager(BaseCorpusManager):
    """Test corpus manager"""

    pass


# Test search client
class TestSearchClient(BaseSearchClient):
    """Test search client"""

    pass


class TestDocumentService(unittest.TestCase):
    """Unit tests for DocumentService"""

    def setUp(self):
        """Set up test fixtures"""
        self.session = MagicMock(spec=Session)
        self.embedding_provider = TestEmbeddingProvider()

    def test_init_with_minimal_config(self):
        """Test initialization with minimal config"""
        # Create config with just the required fields
        config = DocumentServiceConfig(
            document_cls=TestDocument,
        )

        # Initialize service
        service = DocumentService(self.session, config)

        # Verify service was initialized correctly
        self.assertEqual(service.session, self.session)
        self.assertEqual(service.config, config)
        self.assertIsInstance(service.corpus_manager, BaseCorpusManager)
        self.assertIsInstance(service.search_client, BaseSearchClient)

        # Verify corpus manager was initialized with correct config
        self.assertEqual(service.corpus_manager.config.document_cls, TestDocument)
        self.assertIsNone(service.corpus_manager.config.embedding_provider)

        # Verify search client was initialized with correct config
        self.assertEqual(service.search_client.config.document_cls, TestDocument)
        self.assertIsNone(service.search_client.config.embedding_provider)

    def test_init_with_embedding_provider(self):
        """Test initialization with embedding provider"""
        # Create config with embedding provider
        config = DocumentServiceConfig(
            document_cls=TestDocument,
            embedding_provider=self.embedding_provider,
        )

        # Initialize service
        service = DocumentService(self.session, config)

        # Verify embedding provider was assigned to both configs
        self.assertEqual(service.corpus_manager.config.embedding_provider, self.embedding_provider)
        self.assertEqual(service.search_client.config.embedding_provider, self.embedding_provider)

    def test_init_with_full_config(self):
        """Test initialization with full config"""
        # Create corpus manager config
        corpus_config = BaseCorpusManagerConfig(
            document_cls=TestDocument,
            embedding_provider=self.embedding_provider,
            document_metadata_cls=TestDocumentMetadata,
        )

        # Create search client config
        search_config = BaseSearchClientConfig(
            document_cls=TestDocument,
            embedding_provider=self.embedding_provider,
            document_metadata_cls=TestDocumentMetadata,
        )

        # Create service config
        config = DocumentServiceConfig(
            document_cls=TestDocument,
            corpus_manager_cls=TestCorpusManager,
            search_client_cls=TestSearchClient,
            embedding_provider=self.embedding_provider,
            document_metadata_cls=TestDocumentMetadata,
            corpus_manager_cfg=corpus_config,
            search_client_cfg=search_config,
        )

        # Initialize service
        service = DocumentService(self.session, config)

        # Verify service was initialized correctly
        self.assertIsInstance(service.corpus_manager, TestCorpusManager)
        self.assertIsInstance(service.search_client, TestSearchClient)
        self.assertEqual(service.corpus_manager.config, corpus_config)
        self.assertEqual(service.search_client.config, search_config)
        # Configs already have embedding providers, so they should remain unchanged
        self.assertEqual(service.corpus_manager.config.embedding_provider, self.embedding_provider)
        self.assertEqual(service.search_client.config.embedding_provider, self.embedding_provider)
        self.assertEqual(service.corpus_manager.config.document_metadata_cls, TestDocumentMetadata)
        self.assertEqual(service.search_client.config.document_metadata_cls, TestDocumentMetadata)

    def test_embedding_provider_not_overridden(self):
        """Test that existing embedding providers in configs are not overridden"""
        different_provider = TestEmbeddingProvider()
        
        # Create configs with their own embedding providers
        corpus_config = BaseCorpusManagerConfig(
            document_cls=TestDocument,
            embedding_provider=different_provider,
        )
        search_config = BaseSearchClientConfig(
            document_cls=TestDocument,
            embedding_provider=different_provider,
        )

        # Create service config with different embedding provider
        config = DocumentServiceConfig(
            document_cls=TestDocument,
            embedding_provider=self.embedding_provider,
            corpus_manager_cfg=corpus_config,
            search_client_cfg=search_config,
        )

        # Initialize service
        service = DocumentService(self.session, config)

        # Verify original embedding providers were preserved
        self.assertEqual(service.corpus_manager.config.embedding_provider, different_provider)
        self.assertEqual(service.search_client.config.embedding_provider, different_provider)

    # def test_custom_corpus_manager_creation(self):
    #     """Test custom corpus manager creation"""
    #     # Create a custom DocumentService subclass
    #     class CustomDocumentService(DocumentService):
    #         def _create_corpus_manager(self):
    #             # Custom logic to create corpus manager
    #             return TestCorpusManager(self.session, BaseCorpusManagerConfig(
    #                 document_cls=self.config.document_cls,
    #                 embedding_provider=self.embedding_provider
    #             ))

    #     # Create minimal config
    #     config = DocumentServiceConfig(document_cls=TestDocument)

    #     # Initialize custom service
    #     service = CustomDocumentService(self.session, config)

    #     # Verify custom corpus manager was created
    #     self.assertIsInstance(service.corpus_manager, TestCorpusManager)
    #     self.assertEqual(service.corpus_manager.config.document_cls, TestDocument)
    #     self.assertEqual(service.corpus_manager.config.embedding_provider, self.embedding_provider)

    def test_config_validation(self):
        """Test config validation"""
        # Test with invalid document_cls
        with self.assertRaises(ValueError):
            DocumentServiceConfig(document_cls=str)

        # Test with invalid corpus_manager_cls
        with self.assertRaises(ValueError):
            DocumentServiceConfig(document_cls=TestDocument, corpus_manager_cls=str)

        # Test with invalid search_client_cls
        with self.assertRaises(ValueError):
            DocumentServiceConfig(document_cls=TestDocument, search_client_cls=str)

        # Test with invalid document_metadata_cls
        with self.assertRaises(ValueError):
            DocumentServiceConfig(document_cls=TestDocument, document_metadata_cls=str)


if __name__ == "__main__":
    unittest.main()
