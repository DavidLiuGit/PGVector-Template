from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pgvector_template.core import BaseDocument


class MetadataFilter(BaseModel):
    """
    An object acting as a filter for an arbitrary `Metadata` dictionary/map/object.
    """
    
    field_name: str
    """Field path in metadata. Use dot notation for nested fields (e.g., 'publication_info.journal')"""
    condition: Literal["eq", "gt", "gte", "lt", "lte", "contains", "in", "exists"]
    """
    Comparison operator: 
        - eq=equal
        - gt/gte=greater than/equal
        - lt/lte=less than/equal
        - contains=array contains values (accepts array)
        - in=value in array
        - exists=field exists
    """
    value: Any
    """Value to compare against. Type should match field type (str, int, float, bool, list)"""

    model_config = ConfigDict(use_attribute_docstrings=True)


class SearchQuery(BaseModel):
    """Standardized search query structure. At least 1 search criterion is required."""

    text: str | None = None
    """String to match against using in a semantic search, i.e. using vector distance."""
    keywords: list[str] = []
    """List of keywords to **exact-match** in a keyword search."""
    metadata_filters: list[MetadataFilter] = []
    """List of metadata filter conditions that must be matched."""
    date_range: tuple[datetime, datetime] | None = None
    """Retrieve/limit results based on created_at & updated_at timestamps (i.e. database operations)"""
    limit: int = Field(
        ...,
        ge=1,
    )
    """Maximum number of results to return."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    @model_validator(mode="after")
    def ensure_criterion(self):
        if not any([self.text, self.keywords, self.metadata_filters, self.date_range]):
            raise ValueError("At least one search criterion is required")
        return self


@dataclass
class RetrievalResult:
    """Standardized result structure for all retrieval operations"""

    document: BaseDocument
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
