"""
Integration tests for inserting documents into the database
"""

import uuid
import numpy as np
from datetime import datetime, timedelta

from pgvector.sqlalchemy import Vector

from pgvector_template.db import TempDocumentDatabaseManager
from test_document import TestDocument


class TestDocumentInsertionIntegration:
    """Integration tests for document insertion"""

    def test_document_insertion(self, database_url: str):
        """Test inserting documents into the database"""
        # Create a temporary document database manager and schema
        db_manager = TempDocumentDatabaseManager(
            database_url=database_url, schema_suffix="doc_insert_test", document_classes=[TestDocument]
        )
        temp_schema = db_manager.setup()

        try:
            # Create a single test document with minimal required fields
            # Let the DB handle UUID assignment and timestamps
            single_doc = TestDocument(
                corpus_id=uuid.uuid4(),
                content="This is a single test document",
                title="Single Test Document",
                document_metadata={"source": "integration_test", "version": "1.0"},
                embedding=np.random.rand(1024).astype(np.float32),
            )

            # Create multiple test documents
            batch_docs = []
            corpus_id = uuid.uuid4()
            for i in range(3):
                doc = TestDocument(
                    corpus_id=corpus_id,
                    chunk_index=i,
                    content=f"This is batch test document {i}",
                    title=f"Batch Test Document {i}",
                    document_metadata={"index": i, "batch": True},
                    embedding=np.random.rand(1024).astype(np.float32),
                )
                batch_docs.append(doc)

            # Insert all documents
            with db_manager.get_session() as session:
                # Insert single document
                session.add(single_doc)
                session.flush()  # Flush to get the ID

                # Verify single document was inserted with auto-generated fields
                assert single_doc.id is not None  # UUID should be auto-generated
                assert single_doc.created_at is not None
                assert single_doc.updated_at is not None

                # Timestamps should be recent (within the last minute)
                now = datetime.utcnow()
                assert now - timedelta(minutes=1) <= single_doc.created_at <= now
                assert now - timedelta(minutes=1) <= single_doc.updated_at <= now

                # Insert batch documents
                session.add_all(batch_docs)
                session.commit()

                # Verify all documents were inserted
                total_count = session.query(TestDocument).count()
                assert total_count == 4  # 1 single + 3 batch

                # Verify single document content
                retrieved_single = (
                    session.query(TestDocument).filter(TestDocument.title == "Single Test Document").first()
                )
                assert retrieved_single is not None
                assert retrieved_single.content == "This is a single test document"
                assert retrieved_single.document_metadata["source"] == "integration_test"

                # Verify batch documents content
                for i in range(3):
                    doc = session.query(TestDocument).filter(TestDocument.title == f"Batch Test Document {i}").first()
                    assert doc is not None
                    assert doc.content == f"This is batch test document {i}"
                    assert doc.document_metadata["index"] == i
                    assert doc.document_metadata["batch"] is True

                    # Verify auto-generated fields
                    assert doc.id is not None
                    assert doc.created_at is not None
                    assert doc.updated_at is not None

        finally:
            # Clean up temp schema
            db_manager.cleanup(temp_schema)
            pass
