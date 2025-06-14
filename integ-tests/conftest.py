"""
Configuration for integration tests
"""

import os
import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Fixture for database URL - will be implemented by the user
@pytest.fixture
def database_url():
    """Get database URL from environment variables"""
    username = os.getenv("TEST_PGVECTOR_USERNAME")
    password = os.getenv("TEST_PGVECTOR_PASSWORD")
    host = os.getenv("TEST_PGVECTOR_HOST")
    port = os.getenv("TEST_PGVECTOR_PORT")
    db = os.getenv("TEST_PGVECTOR_DB")
    db_url = f"postgresql+psycopg://{username}:{password}@{host}:{port}/{db}"
    if not db_url:
        pytest.skip("TEST_DATABASE_URL environment variable not set")
    return db_url
