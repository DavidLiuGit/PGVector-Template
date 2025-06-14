"""
Test document class for integration tests
"""

from sqlalchemy import Column, Integer

from pgvector_template.core.document import BaseDocument


class TestDocument(BaseDocument):
    """Simple document class for testing"""

    __abstract__ = False
    __tablename__ = "test_documents"


class SecondTestDocument(BaseDocument):
    """Create a second test document class"""

    __abstract__ = False
    __tablename__ = "second_test_documents"

    new_custom_column = Column(Integer, default=69)
