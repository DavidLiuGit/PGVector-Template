from typing import Type

from pgvector_template.core.document import BaseDocumentMetadata
from pgvector_template.models.search import MetadataFilter


def validate_metadata_filters(
    filter_obj_list: list[MetadataFilter], metadata_cls: Type[BaseDocumentMetadata]
) -> None:
    """Validate a `list[MetadataFilter]` against schema and condition compatibility.

    Args:
        filter_obj_list (list[MetadataFilter]): _description_
        metadata_cls (Type[BaseDocumentMetadata]): _description_
    """
    for metadata_filter_obj in filter_obj_list:
        validate_metadata_filter(metadata_filter_obj, metadata_cls)


def validate_metadata_filter(
    filter_obj: MetadataFilter, metadata_cls: Type[BaseDocumentMetadata]
) -> None:
    """Validate metadata filter against schema and condition compatibility.

    Note: This validates the schema structure but cannot guarantee runtime data conformity.
    JSONB field access like field_ref[part] will succeed even if the actual data doesn't match the schema.

    Args:
        filter_obj: The metadata filter to validate
        metadata_cls: The metadata class to validate against

    Raises:
        ValueError: If field doesn't exist in schema or condition is incompatible with field type
    """
    field_path = filter_obj.field_name.split(".")
    current_field_info = metadata_cls.model_fields

    # Navigate nested structure
    for i, part in enumerate(field_path):
        if part not in current_field_info:
            raise ValueError(f"Field '{filter_obj.field_name}' not found in metadata schema")

        field_info = current_field_info[part]
        field_type = field_info.annotation

        # Handle nested models
        if not field_type:
            raise ValueError(f"Field '{filter_obj.field_name}' not found in metadata schema")
        elif hasattr(field_type, "model_fields"):
            current_field_info = field_type.model_fields
        elif i < len(field_path) - 1:
            raise ValueError(
                f"Cannot navigate into non-model field '{part}' in path '{filter_obj.field_name}'"
            )

    # Validate condition compatibility with final field type
    validate_condition_compatibility(field_type, filter_obj.condition)


def validate_condition_compatibility(field_type: Type, condition: str) -> None:
    """Validate that condition is compatible with field type."""
    # Extract base type from Optional/Union types
    origin = getattr(field_type, "__origin__", None)
    if origin is not None:
        args = getattr(field_type, "__args__", ())
        if origin is list:
            field_type = list
        elif len(args) > 0:
            field_type = args[0]  # First non-None type

    valid_conditions = {
        str: {"eq", "gt", "gte", "lt", "lte", "in", "exists"},
        int: {"eq", "gt", "gte", "lt", "lte", "exists"},
        float: {"eq", "gt", "gte", "lt", "lte", "exists"},
        bool: {"eq", "exists"},
        list: {"contains", "in", "exists"},
    }

    allowed = valid_conditions.get(field_type, {"eq", "exists"})
    if condition not in allowed:
        raise ValueError(f"Condition '{condition}' not valid for field type {field_type.__name__}")
