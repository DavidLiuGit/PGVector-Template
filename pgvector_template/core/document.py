from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any
from uuid import uuid4

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
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector

Base = declarative_base()


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

    Index("ix_corpus_chunk", "corpus_id", "chunk_index")
    Index("ix_content_trgm", text("content gin_trgm_ops"), postgresql_using="gin")  # For fuzzy text search
    Index("ix_metadata_gin", "metadata", postgresql_using="gin")


@dataclass
class BaseDocumentMetadata:
    """Base metadata structure"""

    document_type: str
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
