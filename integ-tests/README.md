# Integration Tests

This directory contains integration tests that run against a real PostgreSQL database with pgvector extension.

## Setup

1. Create a `.env` file in the project root with your test database credentials:

```
TEST_DATABASE_URL=postgresql://username:password@localhost:5432/testdb
```

2. Make sure your test database has the pgvector extension installed:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Running Tests

Run the integration tests with:

```bash
pytest integ-tests/ -v
```

## Test Structure

- `conftest.py` - Contains pytest fixtures and environment setup
- `test_document.py` - Defines test document classes
- `test_document_db_integ.py` - Tests for TempDocumentDatabaseManager