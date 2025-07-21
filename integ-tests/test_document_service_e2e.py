"""
Test DocumentService end-to-end, by:

- instantiating DocumentService, and providing real instances of all fields in its config
- insert corpora with CorpusManager
- retrieve documents from the newly inserted corpora, using keyword search
- recover at least 1 original corpus, using CorpusManager.get_full_corpus
"""

from logging import getLogger
import numpy as np
from typing import Any, Type

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column

from pgvector_template.core.document import (
    BaseDocument,
    BaseDocumentMetadata,
    BaseDocumentOptionalProps,
)
from pgvector_template.core.manager import BaseCorpusManager, BaseCorpusManagerConfig
from pgvector_template.core.embedder import BaseEmbeddingProvider
from pgvector_template.core.search import BaseSearchClient, BaseSearchClientConfig, SearchQuery
from pgvector_template.service.document_service import DocumentService, DocumentServiceConfig
from pgvector_template.db import TempDocumentDatabaseManager


logger = getLogger(__name__)


class TestDocumentE2E(BaseDocument):
    """Document class for E2E testing"""

    __abstract__ = False
    __tablename__ = "test_e2e_doc_service"

    embedding = Column(Vector(384))


class SimpleEmbeddingProvider(BaseEmbeddingProvider):
    """Simple deterministic embedding provider for testing"""

    def embed_text(self, text: str) -> list[float]:
        base = np.ones(384) * (len(text) % 10) / 10
        return base.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def get_dimensions(self) -> int:
        return 384


class TestDocumentMetadataE2E(BaseDocumentMetadata):
    """Test document metadata for E2E"""

    document_type: str = "e2e_test"
    schema_version: str = "1.0"
    source: str
    author: str


class TestDocumentServiceE2E:
    """End-to-end tests for DocumentService"""

    def test_document_service_e2e(self, database_url: str):
        """Test complete DocumentService workflow"""
        # Setup database
        db_manager = TempDocumentDatabaseManager(
            database_url=database_url,
            schema_suffix="doc_service_e2e",
            document_classes=[TestDocumentE2E],
        )
        temp_schema = db_manager.setup()

        try:
            with db_manager.get_session() as session:
                # Create real instances for DocumentService config
                embedding_provider = SimpleEmbeddingProvider()

                corpus_manager_cfg = BaseCorpusManagerConfig(
                    document_cls=TestDocumentE2E,
                    embedding_provider=embedding_provider,
                    document_metadata_cls=TestDocumentMetadataE2E,
                )

                search_client_cfg = BaseSearchClientConfig(
                    document_cls=TestDocumentE2E,
                    embedding_provider=embedding_provider,
                    document_metadata_cls=TestDocumentMetadataE2E,
                )

                # Create DocumentService with real config
                service_config = DocumentServiceConfig(
                    document_cls=TestDocumentE2E,
                    corpus_manager_cls=BaseCorpusManager,
                    search_client_cls=BaseSearchClient,
                    embedding_provider=embedding_provider,
                    document_metadata_cls=TestDocumentMetadataE2E,
                    corpus_manager_cfg=corpus_manager_cfg,
                    search_client_cfg=search_client_cfg,
                )

                service = DocumentService(session, service_config)

                # Insert corpora using CorpusManager
                corpus_text_1 = "Machine learning algorithms process data efficiently. Deep learning models require extensive training."
                metadata_1 = {"source": "ml_textbook", "author": "researcher"}
                optional_props_1 = BaseDocumentOptionalProps(
                    title="ML Concepts",
                    collection="ai_docs",
                    tags=["machine_learning", "algorithms"],
                )

                corpus_1_doc_count = service.corpus_manager.insert_corpus(
                    corpus_text_1, metadata_1, optional_props_1
                )
                assert corpus_1_doc_count >= 1, "Should return 1-or-more documents inserted"

                corpus_text_2 = "Database systems store information reliably. SQL queries retrieve specific data records."
                metadata_2 = {"source": "db_manual", "author": "engineer"}
                optional_props_2 = BaseDocumentOptionalProps(
                    title="Database Fundamentals", collection="tech_docs", tags=["database", "sql"]
                )

                corpus_id_2 = service.corpus_manager.insert_corpus(
                    corpus_text_2, metadata_2, optional_props_2
                )

                session.commit()

                # Retrieve documents using keyword search
                search_query = SearchQuery(keywords=["learning", "algorithms"], limit=10)
                search_results = service.search_client.search(search_query)

                # Verify search results
                assert len(search_results) > 0, "Should find documents with keywords"

                found_ml_content = False
                ml_corpus_id = None
                for result in search_results:
                    logger.warning(result.document)
                    logger.warning(result.document.content)
                    if (
                        "learning" in result.document.content
                        or "algorithms" in result.document.content
                    ):
                        found_ml_content = True
                        ml_corpus_id = str(result.document.corpus_id)
                        break

                assert found_ml_content, "Should find ML-related content"
                assert ml_corpus_id is not None, "Should have corpus_id from search results"

                # Recover original corpus using CorpusManager.get_full_corpus
                recovered_corpus_1 = service.corpus_manager.get_full_corpus(ml_corpus_id)

                assert recovered_corpus_1 is not None, "Should recover the first corpus"
                assert recovered_corpus_1.corpus_id == ml_corpus_id
                assert "Machine learning" in recovered_corpus_1.content
                assert "Deep learning" in recovered_corpus_1.content
                assert recovered_corpus_1.metadata["source"] == "ml_textbook"
                assert recovered_corpus_1.metadata["author"] == "researcher"

        finally:
            db_manager.cleanup(temp_schema)
            pass
