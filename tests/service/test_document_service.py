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
from pgvector_template.service.document_service import DocumentService, DocumentServiceConfig


# Test document classes
class TestDocument(BaseDocument):
    """Test document class"""
    __abstract__ = False
    __tablename__ = "test_documents"
    
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
        
        # Verify corpus manager was initialized with correct config
        self.assertEqual(service.corpus_manager.config.document_cls, TestDocument)
        self.assertIsNone(service.corpus_manager.config.embedding_provider)
        
        # Verify search client raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            _ = service.search_client
    
    def test_init_with_full_config(self):
        """Test initialization with full config"""
        # Create corpus manager config
        corpus_config = BaseCorpusManagerConfig(
            document_cls=TestDocument,
            embedding_provider=self.embedding_provider,
            document_metadata_cls=TestDocumentMetadata
        )
        
        # Create service config
        config = DocumentServiceConfig(
            document_cls=TestDocument,
            corpus_manager_cls=TestCorpusManager,
            embedding_provider=self.embedding_provider,
            document_metadata_cls=TestDocumentMetadata,
            corpus_manager_cfg=corpus_config
        )
        
        # Initialize service
        service = DocumentService(self.session, config)
        
        # Verify service was initialized correctly
        self.assertIsInstance(service.corpus_manager, TestCorpusManager)
        self.assertEqual(service.corpus_manager.config, corpus_config)
        self.assertEqual(service.corpus_manager.config.embedding_provider, self.embedding_provider)
        self.assertEqual(service.corpus_manager.config.document_metadata_cls, TestDocumentMetadata)
    
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
        
        # Test with invalid document_metadata_cls
        with self.assertRaises(ValueError):
            DocumentServiceConfig(document_cls=TestDocument, document_metadata_cls=str)


if __name__ == "__main__":
    unittest.main()