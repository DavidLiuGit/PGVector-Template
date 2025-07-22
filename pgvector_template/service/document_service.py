"""
Document service layer combining corpus management and search capabilities.
"""

from logging import getLogger
from typing import Any, Generic, Type, TypeVar
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from pgvector_template.core.document import BaseDocument, BaseDocumentMetadata
from pgvector_template.core.embedder import BaseEmbeddingProvider
from pgvector_template.core.manager import BaseCorpusManager, BaseCorpusManagerConfig
from pgvector_template.core.search import BaseSearchClient, BaseSearchClientConfig

# Type variable for document type
T = TypeVar("T", bound=BaseDocument)

logger = getLogger(__name__)


class DocumentServiceConfig(BaseModel):
    """Configuration for DocumentService"""

    # Required fields
    document_cls: Type[BaseDocument] = Field(...)
    """Document class **type** (not an instance). Must be subclass of `BaseDocument`."""
    corpus_manager_cls: Type[BaseCorpusManager] = Field(default=BaseCorpusManager)
    """CorpusManager class **type** (not an instance). Must be child of `BaseCorpusManager`."""
    search_client_cls: Type[BaseSearchClient] = Field(default=BaseSearchClient)
    """SearchClient class **type** (not an instance). Must be child of `BaseSearchClient`."""

    # Optional fields with defaults
    embedding_provider: BaseEmbeddingProvider | None = Field(default=None)
    """Embedding provider for insert & vector-search operations."""
    document_metadata_cls: Type[BaseDocumentMetadata] = Field(default=BaseDocumentMetadata)
    """Document metadata schema. Must be child of `BaseDocumentMetadata`."""
    corpus_manager_cfg: BaseCorpusManagerConfig = Field(
        default=BaseCorpusManagerConfig(document_cls=BaseDocument)
    )
    """Instance of `BaseCorpusManagerConfig` or a child. Used to instantiate a CorpusManager."""
    search_client_cfg: BaseSearchClientConfig = Field(
        default=BaseSearchClientConfig(document_cls=BaseDocument)
    )
    """Instance of `BaseSearchClientConfig` or a child. Used to instantiate a SearchClient."""

    def model_post_init(self, _):
        # coerce document_cls onto corpus_manager_cfg & search_client_cfg.document_cls,
        # iff either config is an instance of their respective base config classes (and not a subclass)
        if type(self.corpus_manager_cfg) is BaseCorpusManagerConfig:
            self.corpus_manager_cfg.document_cls = self.document_cls
        if type(self.search_client_cfg) is BaseSearchClientConfig:
            self.search_client_cfg.document_cls = self.document_cls

        # assign embedding_provider to CorpusManager & SearchClient configs
        if not self.corpus_manager_cfg.embedding_provider:
            self.corpus_manager_cfg.embedding_provider = self.embedding_provider
        if not self.search_client_cfg.embedding_provider:
            self.search_client_cfg.embedding_provider = self.embedding_provider

    model_config = {"arbitrary_types_allowed": True}


class DocumentService(Generic[T]):
    """Service layer for document operations combining management and search capabilities"""

    @property
    def config(self) -> DocumentServiceConfig:
        return self._cfg

    @property
    def corpus_manager(self) -> BaseCorpusManager:
        """CorpusManager instance"""
        return self._corpus_manager

    @property
    def search_client(self) -> BaseSearchClient:
        """`SearchClient` instance"""
        return self._search_client

    def __init__(self, session: Session, config: DocumentServiceConfig):
        self.session = session
        self._cfg = config
        self._setup()

    def _setup(self):
        """Initialize CorpusManager and SearchClient"""
        self._corpus_manager = self._create_corpus_manager()
        self._search_client = self._setup_search()

    def _create_corpus_manager(self) -> BaseCorpusManager:
        """Initialize CorpusManager. Override this to provide custom instantiation logic."""
        return self.config.corpus_manager_cls(self.session, self.config.corpus_manager_cfg)

    def _setup_search(self) -> BaseSearchClient:
        """Initialize search client - to be implemented. Override this to provide custom instantiation logic."""
        return self.config.search_client_cls(
            self.session,
            self.config.search_client_cfg,
        )
