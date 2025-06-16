from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, ClassVar, Type, TypeVar, Annotated
from uuid import uuid4, UUID

from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    Boolean,
    Integer,
    Float,
    Index,
    text,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class BaseDocumentOptionalProps(BaseModel):
    """Optional properties for document creation"""

    title: str | None = None
    """Optional title or summary for the document"""
    collection: str | None = None
    """Collection name for grouping documents of the same type"""
    original_url: str | None = None
    """Optional source URL for the document"""
    language: str | None = "en"
    """Language of the content (ISO 639-1 code), e.g., 'en', 'es', 'zh'"""
    score: float | None = None
    """Optional score assigned during ingestion (e.g., relevance, confidence)"""
    tags: list[str] | None = None
    """List of tags or keywords for filtering, categorization, or faceted search"""


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
    is_deleted = Column(Boolean, default=False)
    """Entries can be logically marked for deletion before they are permanently deleted."""

    @classmethod
    def from_props(
        cls: Type[T],
        corpus_id: UUID,
        chunk_index: int,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any] = {},
        optional_props: BaseDocumentOptionalProps | None = None,
    ) -> T:
        """
        Create a BaseDocument instance from mandatory and optional properties.

        Args:
            corpus_id: UUID of the corpus this document belongs to
            chunk_index: Index of this chunk within the corpus
            content: Text content of the document
            embedding: Vector embedding of the content
            optional_props: Optional properties for the document

        Returns:
            A new BaseDocument instance of the calling class type
        """
        if optional_props is None:
            optional_props = BaseDocumentOptionalProps()

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

    # Index("ix_corpus_chunk", "corpus_id", "chunk_index")
    # Index("ix_content_trgm", text("content gin_trgm_ops"), postgresql_using="gin")  # For fuzzy text search
    # Index("ix_metadata_gin", "metadata", postgresql_using="gin")


@dataclass
class BaseDocumentMetadata:
    """Base metadata structure"""

    document_type: str
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Corpus:
    """
    Logical grouping of one or more documents (chunks) belonging to the same original source.

    Typically all documents in a corpus share the same `corpus_id` and are ordered by `chunk_index`.
    """

    corpus_id: UUID
    documents: list[BaseDocument]
    metadata: dict[str, Any]  # e.g. source, tags, etc.
