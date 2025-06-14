from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# class RetrievalStrategy(Enum):
#     """Supported retrieval strategies"""
#     SIMILARITY = "similarity"      # Pure vector similarity
#     KEYWORD = "keyword"           # Metadata-based keyword search
#     HYBRID = "hybrid"             # Combined similarity + keyword
#     TEMPORAL = "temporal"         # Time-based search (if applicable)


@dataclass
class RetrievalResult:
    """Standardized result structure for all retrieval operations"""

    document_id: str
    original_id: str
    chunk_index: int
    content: str
    title: Optional[str]
    metadata: dict[str, Any]
    similarity_score: Optional[float] = None
    keyword_matches: Optional[list[str]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchQuery:
    """Standardized search query structure"""

    text: Optional[str] = None
    keywords: Optional[list[str]] = None
    metadata_filters: Optional[dict[str, Any]] = None
    date_range: Optional[tuple[datetime, datetime]] = None
    limit: int = 10
    # strategy: RetrievalStrategy = RetrievalStrategy.HYBRID

    def validate(self) -> bool:
        """Ensure query has at least one search criterion"""
        return any([self.text, self.keywords, self.metadata_filters, self.date_range])
