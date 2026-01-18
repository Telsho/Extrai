from typing import Any, Dict, Set, Optional, List
from sqlmodel import SQLModel
from sqlalchemy.inspection import inspect
from sqlalchemy.orm.collections import InstrumentedList


def serialize_sqlmodel_with_relationships(
    obj: SQLModel, seen: Optional[Set[int]] = None
) -> Dict[str, Any]:
    """
    Recursively serializes a SQLModel instance, including its loaded relationships.
    Uses model_dump(mode='json') to handle basic types (including Decimal -> str/float).

    Args:
        obj: The SQLModel instance to serialize.
        seen: A set of object IDs visited in the current recursion stack to prevent infinite loops.

    Returns:
        A dictionary representation of the SQLModel instance, including relationships.
    """
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        # Prevent infinite recursion for circular references
        return {}

    seen.add(obj_id)

    # 1. Dump basic fields (handles Decimals, Datetimes, etc. via Pydantic serialization)
    data = obj.model_dump(mode="json")

    # 2. Inspect for relationships
    # We rely on SQLModel's internal metadata which is consistent for SQLModel instances
    relationships = getattr(obj, "__sqlmodel_relationships__", {})

    for key in relationships.keys():
        value = getattr(obj, key, None)

        if value is None:
            continue

        if isinstance(value, (list, InstrumentedList)):
            data[key] = [
                serialize_sqlmodel_with_relationships(item, seen)
                if isinstance(item, SQLModel)
                else item
                for item in value
            ]
        elif isinstance(value, SQLModel):
            data[key] = serialize_sqlmodel_with_relationships(value, seen)

    return data


def make_json_serializable(obj: Any) -> Any:
    """
    Recursively converts objects to JSON-serializable formats.
    Handles Decimals by converting to float.
    """
    from decimal import Decimal

    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    return obj
