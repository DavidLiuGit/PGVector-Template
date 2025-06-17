from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any, Optional, Type
from uuid import UUID, uuid4

from pydantic import BaseModel
from sqlalchemy.orm import Session

from pgvector_template.core.document import BaseDocument, BaseDocumentMetadata, BaseDocumentOptionalProps
from pgvector_template.core.embedder import BaseEmbeddingProvider


logger = getLogger(__name__)


class BaseCorpusManagerConfig(BaseModel):
    """Base configuration for `Corpus` & `Document` management operations"""

    schema_name: str
    document_cls: Type[BaseDocument]
    embedding_provider: BaseEmbeddingProvider
    document_metadata: Type[BaseDocumentMetadata]

    model_config = {"arbitrary_types_allowed": True}


class BaseCorpusManager(ABC):
    """
    Template class for `Corpus` & `Document` management operations.
    Each instance should be able to handle multiple collections, of the same Corpus/Document type.
    For example, if the document class is Jira tickets, multiple teams should be able to share the same
    `CorpusManager` implementation, with a slightly different config.
    """

    @property
    def config(self) -> BaseCorpusManagerConfig:
        return self._cfg

    def __init__(
        self,
        session: Session,
        config: BaseCorpusManagerConfig,
    ) -> None:
        self.session = session
        self._cfg = config
        self.schema_name = config.schema_name

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

    def insert_corpus(
        self,
        content: str,
        corpus_metadata: dict[str, Any],
        optional_props: BaseDocumentOptionalProps,
    ) -> int:
        """
        Insert a new `Corpus`, which will be split into 1-or-more `Document`s, depending on its length.
        Each `Document` chunk shall have its own embedding vector, but reference the parent corpus_id.

        Args:
            content: The text content to be inserted as a corpus
            corpus_metadata: Dictionary of metadata associated with the corpus
            optional_props: Optional properties for the documents (title, collection, etc.)

        Returns:
            int: The number of **documents** inserted for the provided corpus
        """
        corpus_id = uuid4()
        document_contents = self._split_corpus(content)
        document_embeddings = self.config.embedding_provider.embed_batch(document_contents)
        return self.insert_documents(corpus_id, document_contents, document_embeddings, corpus_metadata, optional_props)

    def insert_documents(
        self,
        corpus_id: UUID,
        document_contents: list[str],
        document_embeddings: list[list[float]],
        corpus_metadata: dict[str, Any],
        optional_props: BaseDocumentOptionalProps | None = None,
    ) -> int:
        """
        Insert a list of documents (usually from a chunked + embedded corpus).

        Args:
            corpus_id: UUID of the corpus these documents belong to
            document_contents: List of text content for each document
            document_embeddings: List of embedding vectors corresponding to each document
            corpus_metadata: Dictionary of metadata to associate with all documents
            optional_props: Optional properties for the documents (title, collection, etc.)

        Returns:
            int: The number of documents inserted (0 if input lists are empty)

        Raises:
            ValueError: If the length of document_contents doesn't match document_embeddings
        """
        if len(document_contents) != len(document_embeddings):
            raise ValueError("Number of embeddings does not match number of documents")
        if len(document_contents) == 0:
            return 0
        documents_to_insert = []
        for i in range(len(document_contents)):
            chunk_md = self._extract_chunk_metadata(document_contents[i])
            base_metadata = self.config.document_metadata(**(corpus_metadata | chunk_md))
            documents_to_insert.append(
                self.config.document_cls.from_props(
                    corpus_id=corpus_id,
                    chunk_index=i,
                    content=document_contents[i],
                    embedding=document_embeddings[i],
                    metadata=base_metadata.model_dump(),
                    optional_props=optional_props,
                )
            )
        self.session.add_all(documents_to_insert)
        self.session.commit()
        return len(documents_to_insert)

    def _split_corpus(self, content: str, **kwargs) -> list[str]:
        """
        **It is highly recommended to override this method.**
        Split a corpus into chunks.
        """
        if self.__class__ is not BaseCorpusManager:
            logger.warning("Using default _split_corpus. Override this method to improve performance.")
        split_content = [content[i : i + 1000] for i in range(0, len(content), 1000)]
        return [c for c in split_content if len(c.strip()) > 0]

    def _extract_chunk_metadata(self, content: str) -> dict[str, Any]:
        """
        **It is highly recommended to override this method.**
        Extract metadata from a chunk of content, to be appended to corpus metadata
        """
        if self.__class__ is not BaseCorpusManager:
            logger.warning("Using default _extract_chunk_metadata. It is highly recommended to override this method.")
        return {
            "chunk_length": len(content),
        }
