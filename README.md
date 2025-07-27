# PGVector-Template

A flexible, production-ready template library for building Retrieval-Augmented Generation (RAG) applications using PostgreSQL with PGVector extensions.

## Overview

PGVector-Template provides a robust foundation for implementing vector-based document storage and retrieval systems. It offers a clean abstraction layer over PostgreSQL's PGVector extension, making it easy to build scalable RAG applications with proper document management, metadata handling, and efficient vector search capabilities.

## Key Features

- **Flexible Document Model**: Abstract base classes for customizable document schemas
- **Vector Search**: Optimized HNSW indexing for fast similarity search
- **Metadata Management**: JSON-based flexible metadata with GIN indexing
- **Collection Support**: Organize documents into logical collections
- **Chunk Management**: Handle long content (refer to as **corpus**) by chunking it into smaller **documents**. Handle recovering the original corpus given its id
- **Database Abstraction**: Clean SQLAlchemy-based database layer, with an API to create schemas
- **Type Safety**: Full Pydantic validation and type hints
- **Production Ready**: Comprehensive testing and error handling

## Architecture

The library is organized into several key components:

- **Core**: Document models, embedders, search functionality
- **Database**: Connection management and document database operations
- **Service**: High-level document service layer
- **Types**: Shared type definitions and schemas

## Installation

```bash
pip install pgvector-template
```

Or add `pgvector-template` to your dependencies

### Prerequisites

- Python 3.11+
- To execute tests: PostgreSQL with PGVector extension

## Configuration

### Database Setup

1. Install PostgreSQL with PGVector extension
2. Create your database and enable the vector extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

3. Set up your connection string in environment variables or pass directly to `DatabaseManager`

### Environment Variables

For integration tests, create a `.env` file

```bash
cp integ-tests/.env.example integ-tests/.env
```

Specify envvars directly in the .env file. It is loaded automatically for integ tests.

```bash
DATABASE_URL=postgresql://user:password@localhost:5432/test_db
```

## API Reference

### Core Classes

- `BaseDocument`: Abstract document model with vector embedding support
  - refer to table schema for explanation of the fields
- `BaseDocumentOptionalProps`: Optional properties for document creation
- `DatabaseManager`: Database connection and session management
- `DocumentDatabaseManager`: High-level document operations

### Key Methods

- `BaseDocument.from_props()`: Create document instances from properties
- `DocumentDatabaseManager.insert_document()`: Store documents
- `DocumentDatabaseManager.search_similar()`: Vector similarity search

## Testing

Install dependencies (preferably in a virtualenv) before running tests:
```bash
pip install -e .[test]
```

### Unit Tests
```bash
python -m unittest
```

### Integration Tests

Integration tests require a PostgreSQL database with PGVector extension. Set up your test database and configure the connection in `integ-tests/.env`:

```bash
python -m unittest discover -s integ-tests
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the test suite
5. Submit a pull request

### Development Setup

```bash
pip install -e .[dev,test]
black .  # Format code
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Links

- [GitHub Repository](https://github.com/DavidLiuGit/PGVector-Template)
- [PGVector Documentation](https://github.com/pgvector/pgvector)
