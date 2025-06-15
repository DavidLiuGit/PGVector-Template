from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any, Generic, Optional

from sqlalchemy import text

from pgvector_template.core import BaseEmbeddingProvider, RetrievalResult, SearchQuery, BaseDocument
from pgvector_template.types import DocumentType
from sqlalchemy.orm import Session


class BaseSearchClient(ABC, Generic[DocumentType]):
    """Abstract base for document retrieval"""

    def __init__(self, session: Session, embedding_provider: BaseEmbeddingProvider):
        self.session = session
        self.embedding_provider = embedding_provider
        self.logger = getLogger(self.__class__.__name__)

    @abstractmethod
    def search(self, query: SearchQuery) -> list[RetrievalResult]:
        """Main search interface"""
        raise NotImplementedError("Subclasses must implement this method")

    def search_by_metadata(self, filters: dict[str, Any], limit: int = 10) -> list[BaseDocument]:
        """Generic JSON-based metadata search"""
        query = self.session.query(BaseDocument).filter(BaseDocument.is_deleted == False)

        # Apply JSON-based filters
        for key, value in filters.items():
            if isinstance(value, list):
                # Array contains search
                query = query.filter(text(f"metadata->>'{key}' = ANY(:value)")).params(value=value)
            elif isinstance(value, dict):
                # Nested JSON search
                for nested_key, nested_value in value.items():
                    query = query.filter(text(f"metadata->'{key}'->>'{nested_key}' = :value")).params(
                        value=nested_value
                    )
            else:
                # Simple equality
                query = query.filter(text(f"metadata->>'{key}' = :value")).params(value=str(value))

        return query.limit(limit).all()

    @abstractmethod
    def get_full_document(self, original_id: str) -> Optional[dict[str, Any]]:
        """Reconstruct full document from chunks"""
        raise NotImplementedError("Subclasses must implement this method")

    # Template methods with default implementations
    def similarity_search(self, text: str, limit: int = 10) -> list[RetrievalResult]:
        """Vector similarity search"""
        if not text.strip():
            return []

        embedding = self.embedding_provider.embed_text(text)
        return self._vector_search(embedding, limit)

    def keyword_search(self, keywords: list[str], limit: int = 10) -> list[RetrievalResult]:
        """Metadata-based keyword search"""
        return self._metadata_search({"keywords": keywords}, limit)

    def _vector_search(self, embedding: list[float], limit: int) -> list[RetrievalResult]:
        """Internal vector similarity search"""
        # This will be implemented by the template using the DocumentType
        raise NotImplementedError("Subclasses must implement vector search")

    def _metadata_search(self, filters: dict[str, Any], limit: int) -> list[RetrievalResult]:
        """Internal metadata search"""
        raise NotImplementedError("Subclasses must implement metadata search")
