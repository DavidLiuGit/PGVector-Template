from datetime import datetime
from typing import Any, Type, TypeVar
from uuid import uuid4, UUID as UuidLiteral

from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    Boolean,
    Integer,
    Float,
    Index,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class BaseDocumentOptionalProps(BaseModel):
    """Optional properties for document creation"""

    title: str | None = None
    """Optional title or summary for the document"""
    collection: str | None = Field(default=None, max_length=64)
    """Collection name for grouping documents of the same type"""
    original_url: str | None = Field(default=None, max_length=2048)
    """Optional source URL for the document"""
    language: str | None = Field(default="en", pattern=r"^[a-z]{2}(-[A-Z]{2})?$")
    """Language of the content (ISO 639-1 code), e.g., 'en', 'es', 'zh'"""
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    """Optional score assigned during ingestion (e.g., relevance, confidence)"""
    tags: list[str] | None = None
    """List of tags or keywords for filtering, categorization, or faceted search"""

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v):
        if v is not None:
            # Ensure all tags are strings and not empty
            if not all(isinstance(tag, str) and tag.strip() for tag in v):
                raise ValueError("All tags must be non-empty strings")
            # Remove duplicates while preserving order
            return list(dict.fromkeys(v))
        return v


T = TypeVar("T", bound="BaseDocument")


class BaseDocument(Base):
    """
    Template table for Documents, that works for all collection types.
    Each row represents a single retrievable document (could be chunk or full doc).

    Glossary:
    - `corpus` - a full text document, consisting of 1-or-more documents.
      - `corpus_id` is associated with these entries
    - `document` - a chunk (or entirety) of an corpus. `id` is associated with these chunks
    """

    __abstract__ = True
    __tablename__ = "base_document"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    """Primary key of the Document table. Represents unique ID of a Document"""

    # Hierarchy: original_id groups chunks from same source
    collection = Column(String(64), nullable=True)
    """Collection name. Used for filtering and grouping documents of the same type."""
    corpus_id = Column(UUID(as_uuid=True), index=True)
    """An `corpus` is the original, full text that chunks are a part (or all) of"""
    chunk_index = Column(Integer, default=0)
    """Index of this chunk within an `corpus`. Starts from 0."""

    # Content
    content = Column(Text, nullable=False)
    """String content of the chunk"""
    title = Column(String(500))
    """Optional chunk title/summary"""
    document_metadata = Column(JSONB, nullable=False, default=dict)
    """Flexible metadata as JSON"""
    origin_url = Column(String(2048), nullable=True)
    """Optional source URL"""
    language = Column(String(10), default="en")
    """Language of the content (ISO 639-1 code), e.g., 'en', 'es', 'zh'."""
    score = Column(Float, nullable=True)
    """Optional score assigned during ingestion (e.g., relevance, confidence)."""
    tags = Column(JSONB, nullable=True, default=list)
    """List of tags or keywords for filtering, categorization, or faceted search."""

    # Vector embedding
    embedding = Column(Vector(1024))
    """Embedding vector. 1024 dimensions by default. Adjust as-needed."""

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_deleted = Column(Boolean, default=False, index=True)
    """Entries can be logically marked for deletion before they are permanently deleted."""

    def __init_subclass__(cls, **kwargs):
        """Simply declare `__table_args__` to add more args in child classes."""
        super().__init_subclass__(**kwargs)

        if hasattr(cls, "__tablename__"):
            table_name = cls.__tablename__

            base_table_args = (
                Index(
                    f"{table_name}_metadata_gin_idx", "document_metadata", postgresql_using="gin"
                ),
                cls.get_embedding_index(table_name),
                Index(f"{table_name}_tags_gin_idx", "tags", postgresql_using="gin"),
                Index(f"{table_name}_collection_corpus_idx", "collection", "corpus_id"),
                UniqueConstraint(
                    "collection",
                    "corpus_id",
                    "chunk_index",
                    name=f"{table_name}_collection_corpus_chunk_unique",
                ),
            )

            subclass_args = getattr(cls, "__table_args__", ())
            if isinstance(subclass_args, dict):
                cls.__table_args__ = base_table_args + (subclass_args,)
            else:
                if not isinstance(subclass_args, tuple):
                    subclass_args = (subclass_args,) if subclass_args else ()
                cls.__table_args__ = base_table_args + subclass_args

    @classmethod
    def from_props(
        cls: Type[T],
        corpus_id: UuidLiteral | str,
        chunk_index: int,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any] = {},
        optional_props: BaseDocumentOptionalProps | None = None,
    ) -> T:
        """
        Create a BaseDocument instance from mandatory and optional properties.

        Args:
            corpus_id: UUID or string (max 64 chars) of the corpus this document belongs to
            chunk_index: Index of this chunk within the corpus
            content: Text content of the document
            embedding: Vector embedding of the content
            optional_props: Optional properties for the document

        Returns:
            A new BaseDocument instance of the calling class type
        """
        if optional_props is None:
            optional_props = BaseDocumentOptionalProps()

        # SQLAlchemy handles string-to-UUID conversion automatically with as_uuid=True
        return cls(
            corpus_id=corpus_id,
            chunk_index=chunk_index,
            content=content,
            embedding=embedding,
            title=optional_props.title,
            document_metadata=metadata,
            collection=optional_props.collection,
            origin_url=optional_props.original_url,
            language=optional_props.language,
            score=optional_props.score,
            tags=optional_props.tags,
        )

    @classmethod
    def get_embedding_index(cls, table_name: str) -> Index:
        """Override this method to customize the embedding index."""
        return Index(
            f"{table_name}_embedding_hnsw_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        )


class BaseDocumentMetadata(BaseModel):
    """
    Base metadata structure.
    It is generally expected that every `BaseDocument`'s metadata follows this exact schema,
    without any extraneous properties, or any missing properties, to avoid ambiguity.
    """

    document_type: str = Field(
        ..., description="Description for type of document, e.g. markdown, pdf, etc"
    )
    schema_version: str = Field(default="1.0", description="Schema version for the metadata")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
