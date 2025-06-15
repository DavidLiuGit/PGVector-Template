from abc import ABC, abstractmethod
from typing import Any, Optional
from sqlalchemy.orm import Session

from pgvector_template.core.document import BaseDocument, BaseDocumentMetadata


class BaseDocumentManager(ABC):
    """Template class for `Document` management operations"""

    def __init__(self, session: Session, schema_name: str) -> None:
        self.session = session
        self.schema_name = schema_name

    def get_full_corpus(self, corpus_id: str, chunk_delimiter: str = "\n") -> Optional[dict[str, Any]]:
        """Reconstruct full corpus from its individual documents/chunks"""
        chunks = (
            self.session.query(BaseDocument)
            .filter(BaseDocument.corpus_id == corpus_id, BaseDocument.is_deleted == False)
            .order_by(BaseDocument.chunk_index)
            .all()
        )

        if not chunks:
            return None

        # Full document is chunk_index = 0, or reconstruct from chunks
        full_doc = next((c for c in chunks if c.chunk_index == 0), None)
        if full_doc:
            return {
                "id": full_doc.original_id,
                "content": full_doc.content,
                "metadata": full_doc.document_metadata,
                "chunks": [{"id": c.id, "index": c.chunk_index, "title": c.title} for c in chunks if c.chunk_index > 0],
            }

        # Reconstruct from chunks
        reconstructed_content = chunk_delimiter.join([c.content for c in chunks])
        return {
            "id": corpus_id,
            "content": reconstructed_content,
            "metadata": chunks[0].document_metadata,  # Use first chunk's metadata
            "chunks": [{"id": c.id, "index": c.chunk_index, "title": c.title} for c in chunks],
        }

    @abstractmethod
    def create_chunks(self, content: str, metadata: BaseDocumentMetadata) -> list[dict[str, Any]]:
        """Collection-specific chunking logic"""
        raise NotImplementedError("Subclasses must implement this method")
