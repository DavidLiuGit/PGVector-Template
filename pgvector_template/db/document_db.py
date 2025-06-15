from typing import List, Type

from sqlalchemy import text

from pgvector_template.core.document import BaseDocument
from pgvector_template.db.connection import DatabaseManager


class DocumentDatabaseManager(DatabaseManager):
    """
    Specialized database manager for document collections with streamlined setup

    Usage example:
    ```python
    db = DocumentDatabaseManager(
        database_url="postgresql://user:pass@localhost/db",
        schema_suffix="fpga_operations",
        document_classes=[TextDocument, ImageDocument, AudioDocument]
    )
    db.setup()
    ```
    """

    SCHEMA_PREFIX = "knowledge_base_"

    def __init__(self, database_url: str, schema_suffix: str, document_classes: List[Type[BaseDocument]]):
        """
        Initialize a document-oriented database manager

        Args:
            database_url: PostgreSQL connection string
            schema_suffix: Suffix for schema name (will be prefixed with SCHEMA_PREFIX)
            document_classes: List of concrete subclasses of BaseDocument
        """
        super().__init__(database_url)
        self.schema_suffix = schema_suffix
        self.schema_name = f"{self.SCHEMA_PREFIX}{schema_suffix}"
        self.document_classes = document_classes

    def setup(self) -> None:
        """One-step setup: initialize connection, create schema and tables for all document classes"""
        self.initialize()
        self.create_schema(self.schema_name)

        for doc_class in self.document_classes:
            # Set schema for each document class
            doc_class.__table__.schema = self.schema_name
            self.create_tables(doc_class, self.schema_name)

        self.logger.info(
            f"Document database setup complete for {self.schema_name} with {len(self.document_classes)} tables"
        )


class TempDocumentDatabaseManager(DocumentDatabaseManager):
    def setup(self) -> str:
        """
        Create a temporary schema with a unique name for testing
        Format: `temp_knowledge_base_<schema_suffix>_<uuid_snippet>`
        """
        from uuid import uuid4

        temp_schema_name = f"temp_{self.schema_name}_{uuid4().hex[:8]}"

        self.initialize()
        self.create_schema(temp_schema_name)

        for doc_class in self.document_classes:
            # Create a copy of the table with the temporary schema
            doc_class.__table__.schema = temp_schema_name
            self.create_tables(doc_class, temp_schema_name)

        self.logger.info(f"Created temporary schema: {temp_schema_name}")
        return temp_schema_name

    def cleanup(self, schema_name: str) -> None:
        """Drop a temporary schema and all its objects"""
        if not schema_name.startswith("temp_"):
            raise ValueError("Can only drop schemas with 'temp_' prefix for safety")

        with self.engine.connect() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
            conn.commit()

        self.logger.info(f"Dropped temporary schema: {schema_name}")
