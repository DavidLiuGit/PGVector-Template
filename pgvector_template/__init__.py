# Import key classes for easier access
from pgvector_template.db.connection import DatabaseManager
from pgvector_template.db.document_db import DocumentDatabaseManager

# Re-export at the top level
__all__ = ["DatabaseManager", "DocumentDatabaseManager"]