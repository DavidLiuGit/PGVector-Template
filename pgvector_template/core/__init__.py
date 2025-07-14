from pgvector_template.core.document import BaseDocument, BaseDocumentMetadata, BaseDocumentOptionalProps
from pgvector_template.core.embedder import BaseEmbeddingProvider
from pgvector_template.core.manager import BaseCorpusManager, BaseCorpusManagerConfig, Corpus
from pgvector_template.core.retriever import RetrievalResult, SearchQuery
from pgvector_template.core.search import BaseSearchClient


__all__ = [
    ### document
    "BaseDocumentOptionalProps",
    "BaseDocument",
    "BaseDocumentMetadata",
    "Corpus",
    ### embedder
    "BaseEmbeddingProvider",
    ### manager
    "BaseCorpusManager",
    "BaseCorpusManagerConfig",
    ### search
    "RetrievalResult",
    "SearchQuery",
    "BaseSearchClient",
]
