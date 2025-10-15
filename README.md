# PGVector-Template

A flexible, production-ready template library for building Retrieval-Augmented Generation (RAG) applications using PostgreSQL with PGVector extensions.

## Overview

PGVector-Template provides a robust foundation for implementing vector-based document storage and retrieval systems. It offers a clean abstraction layer over PostgreSQL's PGVector extension, making it easy to build scalable RAG applications with proper document management, metadata handling, and efficient vector search capabilities.

## Quick Start

```python
from pgvector_template import DatabaseManager, DocumentDatabaseManager
from pgvector_template.core.document import BaseDocument
from sqlalchemy import create_engine
from uuid import uuid4

# 1. Define your document model
class MyDocument(BaseDocument):
    __tablename__ = "my_documents"

# 2. Set up database connection
engine = create_engine("postgresql://user:pass@localhost/mydb")
db_manager = DatabaseManager(engine)
db_manager.create_tables([MyDocument])

# 3. Create document manager
doc_manager = DocumentDatabaseManager(db_manager)

# 4. Insert a document
with db_manager.get_session() as session:
    doc = MyDocument.from_props(
        corpus_id=uuid4(),
        chunk_index=0,
        content="Your document content here",
        embedding=[0.1, 0.2, 0.3, ...],  # Your embedding vector
    )
    doc_manager.insert_document(session, doc)

# 5. Search similar documents
with db_manager.get_session() as session:
    results = doc_manager.search_similar(
        session, MyDocument, query_embedding=[0.1, 0.2, 0.3, ...], limit=5
    )
```

## Key Concepts

**Corpus vs Document**: Understanding the hierarchy is essential:
- **Corpus**: A complete source document (e.g., a full PDF, article, or book)
- **Document**: A chunk or segment of a corpus that fits within embedding limits
- **Collection**: A logical grouping of related corpora (e.g., "legal_docs", "user_manuals")

Example: A 50-page PDF (corpus) might be split into 200 documents (chunks), all sharing the same `corpus_id` but with different `chunk_index` values.

## Key Features

- **Flexible Document Model**: Abstract base classes for customizable document schemas
- **Vector Search**: Optimized HNSW indexing for fast similarity search
- **Metadata Management**: JSON-based flexible metadata with GIN indexing
- **Collection Support**: Organize documents into logical collections
- **Chunk Management**: Handle long content by chunking into retrievable documents
- **Database Abstraction**: Clean SQLAlchemy-based database layer with schema creation API
- **Type Safety**: Full Pydantic validation and type hints
- **Production Ready**: Comprehensive testing and error handling

## Architecture

The library is organized into several key components:

- **Core**: Document models, embedders, search functionality
- **Database**: Connection management and document database operations
- **Service**: High-level document service layer
- **Types**: Shared type definitions and schemas

## Installation

### Basic Installation

```bash
pip install pgvector-template
```

### With Database Driver

For production use, you'll also need a PostgreSQL driver:

```bash
# For binary driver (recommended)
pip install pgvector-template psycopg[binary]

# Or for source driver
pip install pgvector-template psycopg
```

### Prerequisites

- Python 3.11+
- PostgreSQL 12+ with PGVector extension
- For development: Additional test dependencies

## Configuration

### Database Setup

1. **Install PostgreSQL with PGVector extension**
2. **Create your database and enable the vector extension:**

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

3. **Set up your connection:**

```python
from sqlalchemy import create_engine
from pgvector_template import DatabaseManager

# Option 1: Direct connection string
engine = create_engine("postgresql://user:password@localhost:5432/mydb")
db_manager = DatabaseManager(engine)

# Option 2: From environment variable
import os
engine = create_engine(os.getenv("DATABASE_URL"))
db_manager = DatabaseManager(engine)
```

### Production Configuration

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    "postgresql://user:password@localhost:5432/mydb",
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)
```

### Environment Variables

```bash
# Development
DATABASE_URL=postgresql://user:password@localhost:5432/dev_db
```

## Usage Examples

### 1. Define Your Document Model

```python
from pgvector_template.core.document import BaseDocument, BaseDocumentMetadata
from pydantic import Field

class MyDocumentMetadata(BaseDocumentMetadata):
    source_type: str = Field(..., description="Type of source document")
    author: str = Field(default="unknown", description="Document author")

class MyDocument(BaseDocument):
    __tablename__ = "my_documents"
```

### 2. Insert Documents

```python
from pgvector_template.core.document import BaseDocumentOptionalProps
from uuid import uuid4

# Create document with metadata
metadata = MyDocumentMetadata(source_type="pdf", author="John Doe")
optional_props = BaseDocumentOptionalProps(
    title="Chapter 1: Introduction",
    collection="textbooks",
    language="en",
    tags=["education", "intro"]
)

doc = MyDocument.from_props(
    corpus_id=uuid4(),
    chunk_index=0,
    content="This is the document content...",
    embedding=your_embedding_vector,  # list[float] with 1024 dimensions
    metadata=metadata.to_dict(),
    optional_props=optional_props
)

with db_manager.get_session() as session:
    doc_manager.insert_document(session, doc)
```

### 3. Search Documents

```python
# Basic similarity search
with db_manager.get_session() as session:
    results = doc_manager.search_similar(
        session=session,
        document_cls=MyDocument,
        query_embedding=query_vector,
        limit=10
    )

# Search with filters
from pgvector_template.models.search import MetadataFilter

filters = [
    MetadataFilter(key="source_type", value="pdf"),
    MetadataFilter(key="author", value="John Doe")
]

results = doc_manager.search_similar(
    session=session,
    document_cls=MyDocument,
    query_embedding=query_vector,
    limit=10,
    metadata_filters=filters,
    collection="textbooks"
)
```

### 4. Manage Collections

```python
# Get all documents in a collection
with db_manager.get_session() as session:
    docs = doc_manager.get_documents_by_collection(
        session, MyDocument, "textbooks"
    )

# Get all chunks from a corpus
    corpus_docs = doc_manager.get_documents_by_corpus_id(
        session, MyDocument, corpus_id
    )
```

## Concept Reference

### Core Classes

- **`BaseDocument`**: Abstract SQLAlchemy model for documents with vector embeddings
- **`BaseDocumentOptionalProps`**: Pydantic model for optional document properties
- **`BaseDocumentMetadata`**: Base schema for structured document metadata
- **`DatabaseManager`**: Database connection and session management
- **`DocumentDatabaseManager`**: High-level document CRUD operations

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
