"""
Test DocumentService end-to-end, by:

- instantiating DocumentService, and providing real instances of all fields in its config
- insert corpora with CorpusManager
- retrieve documents from the newly inserted corpora, using keyword search
- recover at least 1 original corpus, using CorpusManager.get_full_corpus
"""

from logging import getLogger
import numpy as np

from pgvector.sqlalchemy import Vector
from pydantic import BaseModel
from sqlalchemy import Column

from pgvector_template.core.document import (
    BaseDocument,
    BaseDocumentMetadata,
    BaseDocumentOptionalProps,
)
from pgvector_template.core.manager import BaseCorpusManager, BaseCorpusManagerConfig
from pgvector_template.core.embedder import BaseEmbeddingProvider
from pgvector_template.core.search import BaseSearchClient, BaseSearchClientConfig, SearchQuery
from pgvector_template.models.search import MetadataFilter
from pgvector_template.service.document_service import DocumentService, DocumentServiceConfig
from pgvector_template.db import TempDocumentDatabaseManager


logger = getLogger(__name__)


class TestDocumentE2E(BaseDocument):
    """Document class for E2E testing"""

    __abstract__ = False
    __tablename__ = "test_e2e_doc_service"

    embedding = Column(Vector(16))


class SimpleEmbeddingProvider(BaseEmbeddingProvider):
    """Simple deterministic embedding provider for testing with basic semantic similarity"""

    def embed_text(self, text: str) -> list[float]:
        words = set(text.lower().split())
        embedding = np.zeros(16)

        # Feature 1: ML/AI terms (dimensions 0-3)
        ml_terms = {"machine", "learning", "deep", "algorithms", "models", "training", "neural"}
        ml_score = len(words & ml_terms) / len(words) if words else 0
        embedding[0:4] = ml_score

        # Feature 2: Database terms (dimensions 4-7)
        db_terms = {"database", "sql", "queries", "data", "records", "store", "information"}
        db_score = len(words & db_terms) / len(words) if words else 0
        embedding[4:8] = db_score

        # Feature 3: Technical terms (dimensions 8-11)
        tech_terms = {"systems", "process", "efficiently", "require", "extensive", "specific"}
        tech_score = len(words & tech_terms) / len(words) if words else 0
        embedding[8:12] = tech_score

        # Feature 4: Text length normalized (dimensions 12-15)
        length_feature = min(len(text) / 100, 1.0)
        embedding[12:16] = length_feature

        # Add deterministic randomness based on text hash
        np.random.seed(hash(text) % 2**32)
        noise = np.random.normal(0, 0.01, 16)
        embedding += noise

        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def get_dimensions(self) -> int:
        return 16


class NestedInfo(BaseModel):
    """Nested metadata for testing"""
    version: str = "1.0"
    date: str = "2024-01-01"


class TestDocumentMetadataE2E(BaseDocumentMetadata):
    """Test document metadata for E2E"""

    document_type: str = "e2e_test"
    schema_version: str = "1.0"
    source: str
    author: str
    priority: int = 1
    tags: list[str] = []
    info: NestedInfo = NestedInfo()


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
                metadata_1 = {
                    "source": "ml_textbook", 
                    "author": "researcher", 
                    "priority": 5, 
                    "tags": ["ai", "ml"],
                    "info": {"version": "2.0", "date": "2024-03-01"}
                }
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
                metadata_2 = {"source": "db_manual", "author": "engineer", "priority": 3, "tags": ["database", "sql"]}
                optional_props_2 = BaseDocumentOptionalProps(
                    title="Database Fundamentals", collection="tech_docs", tags=["database", "sql"]
                )

                corpus_id_2 = service.corpus_manager.insert_corpus(
                    corpus_text_2, metadata_2, optional_props_2
                )

                corpus_text_3 = "Web development frameworks simplify application building. Modern tools enhance productivity."
                metadata_3 = {"source": "web_guide", "author": "developer", "priority": 2, "tags": ["web", "frameworks"]}
                optional_props_3 = BaseDocumentOptionalProps(
                    title="Web Development", collection="tech_docs", tags=["web", "development"]
                )

                corpus_id_3 = service.corpus_manager.insert_corpus(
                    corpus_text_3, metadata_3, optional_props_3
                )

                session.commit()

                ##### Test 1: Keyword-only search
                keyword_query = SearchQuery(keywords=["learning", "algorithms"], limit=10)
                keyword_results = service.search_client.search(keyword_query)

                assert len(keyword_results) > 0, "Keyword search should find documents"
                found_ml_keyword = any(
                    "learning" in r.document.content or "algorithms" in r.document.content
                    for r in keyword_results
                )
                assert found_ml_keyword, "Keyword search should find ML content"

                ##### Test 2: Semantic-only search
                semantic_query = SearchQuery(
                    text="artificial intelligence and neural networks", limit=10
                )
                semantic_results = service.search_client.search(semantic_query)

                assert len(semantic_results) > 0, "Semantic search should find documents"
                first_semantic = semantic_results[0]
                assert (
                    "learning" in first_semantic.document.content
                    or "algorithms" in first_semantic.document.content
                ), "Semantic search should prioritize ML content"

                ##### Test 3: Combined keyword + semantic search
                combined_query = SearchQuery(text="neural networks", keywords=["data"], limit=10)
                combined_results = service.search_client.search(combined_query)

                assert len(combined_results) > 0, "Combined search should find documents"
                # Should find documents containing "data" and semantically similar to "neural networks"
                found_data_content = any("data" in r.document.content for r in combined_results)
                assert found_data_content, "Combined search should find content with keyword 'data'"

                ##### Test 4: Metadata filter tests
                # Test equality filter
                eq_filter = MetadataFilter(field_name="author", condition="eq", value="researcher")
                eq_query = SearchQuery(metadata_filters=[eq_filter], limit=10)
                eq_results = service.search_client.search(eq_query)
                assert len(eq_results) > 0, "Should find documents by author"
                assert all(r.document.document_metadata["author"] == "researcher" for r in eq_results)

                # Test greater than filter
                gt_filter = MetadataFilter(field_name="priority", condition="gt", value=3)
                gt_query = SearchQuery(metadata_filters=[gt_filter], limit=10)
                gt_results = service.search_client.search(gt_query)
                assert len(gt_results) > 0, "Should find high priority documents"
                assert all(r.document.document_metadata["priority"] > 3 for r in gt_results)

                # Test contains filter (array contains value)
                contains_filter = MetadataFilter(field_name="tags", condition="contains", value="ai")
                contains_query = SearchQuery(metadata_filters=[contains_filter], limit=10)
                contains_results = service.search_client.search(contains_query)
                assert len(contains_results) > 0, "Should find documents with 'ai' tag"
                assert all("ai" in r.document.document_metadata["tags"] for r in contains_results)

                # Test exists filter
                exists_filter = MetadataFilter(field_name="priority", condition="exists", value=None)
                exists_query = SearchQuery(metadata_filters=[exists_filter], limit=10)
                exists_results = service.search_client.search(exists_query)
                assert len(exists_results) > 0, "Should find documents with priority field"

                # Test string comparison (gt condition with string)
                str_gt_filter = MetadataFilter(field_name="author", condition="gt", value="engineer")
                str_gt_query = SearchQuery(metadata_filters=[str_gt_filter], limit=10)
                str_gt_results = service.search_client.search(str_gt_query)
                assert len(str_gt_results) > 0, "Should find documents with author > 'engineer'"
                assert all(r.document.document_metadata["author"] > "engineer" for r in str_gt_results)

                # Test nested field exists condition
                nested_exists_filter = MetadataFilter(field_name="info.version", condition="exists", value=None)
                nested_exists_query = SearchQuery(metadata_filters=[nested_exists_filter], limit=10)
                nested_exists_results = service.search_client.search(nested_exists_query)
                assert len(nested_exists_results) > 0, "Should find documents with nested info.version field"

                # Test combined metadata filters (AND logic)
                combined_filters = [
                    MetadataFilter(field_name="source", condition="eq", value="db_manual"),
                    MetadataFilter(field_name="priority", condition="gte", value=3)
                ]
                combined_meta_query = SearchQuery(metadata_filters=combined_filters, limit=10)
                combined_meta_results = service.search_client.search(combined_meta_query)
                assert len(combined_meta_results) > 0, "Should find documents matching both filters"
                for result in combined_meta_results:
                    assert result.document.document_metadata["source"] == "db_manual"
                    assert result.document.document_metadata["priority"] >= 3

                ##### Corpus recovery test
                ml_corpus_id = str(keyword_results[0].document.corpus_id)
                recovered_corpus = service.corpus_manager.get_full_corpus(ml_corpus_id)

                assert recovered_corpus is not None, "Should recover corpus"
                assert "Machine learning" in recovered_corpus.content
                assert recovered_corpus.metadata["source"] == "ml_textbook"

        finally:
            db_manager.cleanup(temp_schema)
            pass
