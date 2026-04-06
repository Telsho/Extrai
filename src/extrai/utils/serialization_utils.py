from typing import Any

from sqlalchemy.orm.collections import InstrumentedList
from sqlmodel import SQLModel


def serialize_sqlmodel_with_relationships(
    obj: SQLModel, seen: set[int] | None = None
) -> dict[str, Any]:
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


def resolve_step_param(
    param: str | list[str], step_index: int = 0, total_steps: int = 1
) -> str:
    """
    Resolves a parameter that can be a single string or a list of strings
    to the specific string for the current step.

    Args:
        param: The parameter value (str or list[str])
        step_index: The current step index (0-based)
        total_steps: The total number of steps in the process

    Returns:
        The string value for the current step.

    Raises:
        ValueError: If list length does not match requirements.
    """
    if isinstance(param, str):
        return param

    if not isinstance(param, list):
        return str(param) if param is not None else ""

    if not param:
        return ""

    if len(param) == 1:
        return param[0]

    if len(param) != total_steps:
        raise ValueError(
            f"Parameter list has {len(param)} elements, but process has {total_steps} steps. "
            "Pass a single string, a 1-element list, or a list matching the number of steps."
        )

    if step_index < 0 or step_index >= len(param):
        raise ValueError(
            f"Step index {step_index} out of bounds for parameter list of length {len(param)}"
        )

    return param[step_index]
