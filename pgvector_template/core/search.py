from logging import getLogger
from typing import Any, Type, Sequence

from pydantic import BaseModel, Field
from sqlalchemy import select, or_, Integer, Float
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select, ColumnElement

from pgvector_template.core import (
    BaseEmbeddingProvider,
    BaseDocument,
    BaseDocumentMetadata,
)
from pgvector_template.models.search import SearchQuery, MetadataFilter, RetrievalResult
from pgvector_template.utils.metadata_filter import validate_metadata_filters


logger = getLogger(__name__)


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
        if query.metadata_filters:
            db_query = self._apply_metadata_filters(db_query, query)
        db_query = db_query.limit(query.limit)

        # execute query and return results
        results = self.session.scalars(db_query).all()
        return self._convert_to_retrieval_results(results)

    def _apply_semantic_search(self, db_query: Select, search_query: SearchQuery) -> Select:
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
            return db_query
        query_embedding = self.embedding_provider.embed_text(search_query.text)
        return db_query.order_by(
            self.config.document_cls.embedding.cosine_distance(query_embedding)
        )

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

    def _apply_metadata_filters(self, db_query: Select, search_query: SearchQuery) -> Select:
        """
        Apply metadata filters to the query. All condtions in `search_query.metadata_filters`,
        if not None, are ANDed together. Metadata filters are applied against
        `BaseDocument.document_metadata` JSONB field.

        Args:
            query: The base SQLAlchemy query.
            search_query: The `SearchQuery` containing metadata filters.

        Returns:
            Updated SQLAlchemy query with metadata filters applied.
        """
        if not search_query.metadata_filters:
            return db_query

        # strictly speaking, this validation step is optional. You may override this method to disable
        try:
            validate_metadata_filters(search_query.metadata_filters, self.config.document_metadata_cls)
        except ValueError as e:
            logger.warning(f"Metadata filter validation failed: {e}. Query success not guaranteed!")

        conditions = [
            self._build_metadata_filter_where_condition(filter_obj)
            for filter_obj in search_query.metadata_filters
        ]
        return db_query.where(*conditions) if conditions else db_query

    def _build_metadata_filter_where_condition(
        self, filter_obj: MetadataFilter
    ) -> ColumnElement[bool]:
        """Build SQLAlchemy WHERE condition for a metadata filter."""
        field_path = filter_obj.field_name.split(".")
        metadata_col = self.config.document_cls.document_metadata

        # Navigate to field
        field_ref = metadata_col
        for part in field_path:
            field_ref = field_ref[part]

        if filter_obj.condition == "eq":
            return (
                field_ref.astext == filter_obj.value
                if isinstance(filter_obj.value, str)
                else field_ref == filter_obj.value
            )
        elif filter_obj.condition in {"gt", "gte", "lt", "lte"}:
            if isinstance(filter_obj.value, str):
                field_text = field_ref.astext
            else:
                cast_type = Integer if isinstance(filter_obj.value, int) else Float
                field_text = field_ref.astext.cast(cast_type)

            if filter_obj.condition == "gt":
                return field_text > filter_obj.value
            elif filter_obj.condition == "gte":
                return field_text >= filter_obj.value
            elif filter_obj.condition == "lt":
                return field_text < filter_obj.value
            else:  # lte
                return field_text <= filter_obj.value
        elif filter_obj.condition == "contains":
            return field_ref.contains([filter_obj.value])
        elif filter_obj.condition == "in":
            return field_ref.astext.in_([str(v) for v in filter_obj.value])
        elif filter_obj.condition == "exists":
            if len(field_path) == 1:
                return metadata_col.has_key(field_path[0])
            else:
                parent_ref = metadata_col
                for part in field_path[:-1]:
                    parent_ref = parent_ref[part]
                return parent_ref.has_key(field_path[-1])
        else:
            raise ValueError(f"Unsupported condition: {filter_obj.condition}")

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
            retrieval_results.append(RetrievalResult(document=result, score=1.0))
        return retrieval_results
