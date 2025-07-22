from dataclasses import dataclass, asdict
from datetime import datetime
from logging import getLogger
from typing import Any, Type, Sequence

from pydantic import BaseModel, Field, model_validator
from sqlalchemy import text, select, or_, and_
from sqlalchemy.sql import Select
from pgvector.sqlalchemy import Vector

from pgvector_template.core import (
    BaseEmbeddingProvider,
    BaseDocument,
    BaseDocumentMetadata,
)
from sqlalchemy.orm import Session


logger = getLogger(__name__)


class SearchQuery(BaseModel):
    """Standardized search query structure. At least 1 search criterion is required."""

    text: str | None = None
    """String to approximate-search (using vector distance) in a semantic search."""
    keywords: list[str] | None = None
    """List of keywords to exact-match in a keyword search."""
    metadata_filters: dict[str, Any] | None = None
    """Strict metadata filters that must be matched."""
    date_range: tuple[datetime, datetime] | None = None
    """Retrieve/limit results based on created_at & updated_at timestamps"""
    limit: int = Field(
        ...,
        ge=1,
    )
    """Maximum number of results to return."""

    @model_validator(mode="after")
    def ensure_criterion(self):
        if not any([self.text, self.keywords, self.metadata_filters, self.date_range]):
            raise ValueError("At least one search criterion is required")
        return self


@dataclass
class RetrievalResult:
    """Standardized result structure for all retrieval operations"""

    document: BaseDocument
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseSearchClientConfig(BaseModel):
    """Config obj for `BaseSearchClient`."""

    document_cls: Type[BaseDocument] = Field(default=BaseDocument)
    """Document class **type** (not an instance). Must be subclass of `BaseDocument`."""
    embedding_provider: BaseEmbeddingProvider | None = Field(default=None)
    """Instance of `BaseEmbeddingProvider` child class. Acts as embedding provider for semantic search."""
    document_metadata_cls: Type[BaseDocumentMetadata] = Field(default=BaseDocumentMetadata)
    """Document metadata class type. Used for metadata search operations."""

    model_config = {"arbitrary_types_allowed": True}


class BaseSearchClient:
    """Minimum-viable implementation of document retrieval for PGVector"""

    @property
    def config(self) -> BaseSearchClientConfig:
        return self._cfg

    @property
    def document_metadata_class(self) -> Type[BaseDocumentMetadata]:
        """Returns the document metadata class, raising an error if it's not set."""
        return self.config.document_metadata_cls

    @property
    def embedding_provider(self) -> BaseEmbeddingProvider:
        """Returns the embedding provider, raising an error if it's not set."""
        if self.config.embedding_provider is None:
            raise ValueError("embedding_provider must be provided in config for this operation")
        return self.config.embedding_provider

    def __init__(
        self,
        session: Session,
        config: BaseSearchClientConfig,
    ):
        self.session = session
        self._cfg = config
        if not self.config.embedding_provider:
            logger.warning(
                "EmbeddingProvider not provided in config. Vector (semantic) search will be unavailable."
            )

    def search(self, query: SearchQuery) -> list[RetrievalResult]:
        """Search for documents based on the provided query.

        Args:
            query: Search query containing text, metadata filters, and pagination.

        Returns:
            List of retrieval results matching the search criteria.
        """
        db_query = select(self.config.document_cls)

        if query.text:
            db_query = self._apply_semantic_search(db_query, query)
        db_query = self._apply_keyword_search(db_query, query)
        # if query.metadata_filters:
        #     db_query = self._apply_metadata_filters(db_query, query)
        db_query = db_query.limit(query.limit)

        # execute query and return results
        results = self.session.scalars(db_query).all()
        return self._convert_to_retrieval_results(results)

    def _apply_semantic_search(self, query: Select, search_query: SearchQuery) -> Select:
        """Apply semantic (vector) search criteria to the query.
        `embedding_provider` must be provided at instantiation, or an `ValueError` will be raised.
        In PGVector, `<=>` operator is used to compare cosine distance. Lower = more similar.

        Args:
            query: The base SQLAlchemy query.
            search_query: The search query containing the text to search for.

        Returns:
            Updated SQLAlchemy query with semantic search applied.
        """
        if not search_query.text:
            return query
        query_embedding = self.embedding_provider.embed_text(search_query.text)
        return query.order_by(self.config.document_cls.embedding.cosine_distance(query_embedding))

    def _apply_keyword_search(self, db_query: Select, search_query: SearchQuery) -> Select:
        """Apply keyword (full-text) search criteria to the query.
        Search against `BaseDocument.content`.
        Args:
            db_query: The base SQLAlchemy query.
            search_query: The search query containing the text to search for.
        Returns:
            Updated SQLAlchemy query with keyword search applied.
        """
        if not search_query.keywords:
            return db_query

        conditions = []
        for keyword in search_query.keywords:
            conditions.append(self.config.document_cls.content.ilike(f"%{keyword}%"))
        return db_query.where(or_(*conditions))

    def _apply_metadata_filters(self, query: Select, search_query: SearchQuery) -> Select:
        """Apply metadata filters to the query.

        Args:
            query: The base SQLAlchemy query.
            search_query: The search query containing metadata filters.

        Returns:
            Updated SQLAlchemy query with metadata filters applied.
        """
        raise NotImplementedError

    def _convert_to_retrieval_results(self, results: Sequence[Any]) -> list[RetrievalResult]:
        """Convert database results to RetrievalResult objects.

        Args:
            results: Raw database results.
            search_query: The original search query.

        Returns:
            List of RetrievalResult objects.
        """
        retrieval_results = []
        for result in results:
            doc = result[0] if isinstance(result, tuple) else result
            retrieval_results.append(RetrievalResult(document=result, score=1.0))
        return retrieval_results
