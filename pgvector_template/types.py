from typing import TypeVar

from pgvector_template.core.document import BaseDocument, BaseDocumentMetadata

T = TypeVar("T", bound="BaseDocumentMetadata")
DocumentType = TypeVar("DocumentType", bound="BaseDocument")
