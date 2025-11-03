import json
from typing import Any, Dict, Type, Optional, Union

from extrai.core.analytics_collector import WorkflowAnalyticsCollector
from sqlmodel import SQLModel
from pydantic import ValidationError as PydanticValidationError

from extrai.core.errors import (
    LLMOutputParseError,
    LLMOutputValidationError,
)
from extrai.utils.json_validation_utils import is_json_valid


def _filter_special_fields_for_validation(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Removes fields that are part of an extended schema (e.g., for relationship handling
    or temporary IDs) but not part of the core SQLModel definition for validation.
    This includes fields like '_temp_id', '_type', and fields ending with '_ref_id' or '_ref_ids'.
    """
    if not isinstance(data, dict):
        # This case should ideally be caught before calling this filter,
        # but as a safeguard:
        return data
    return {
        k: v
        for k, v in data.items()
        if k not in ["_temp_id", "_type"]
        and not k.endswith("_ref_id")
        and not k.endswith("_ref_ids")
    }


def _unwrap_llm_output(data: Any) -> Any:
    """
    Unwraps nested data from LLM JSON outputs.
    It searches for a primary data payload, which could be a list or a dictionary,
    within common wrapping structures like `{"result": [...]}` or `{"data": [...]}`.
    """
    if isinstance(data, dict):
        # Prioritize keys that are likely to contain the main payload.
        for key in ["result", "data", "results", "entities"]:
            if key in data:
                return data[key]

        # If no priority key is found, and there's only one key,
        # return the value associated with that key.
        if len(data) == 1:
            return next(iter(data.values()))

    # If the data is not a dictionary or no specific unwrapping rule applies,
    # return the data as is.
    return data


def process_and_validate_llm_output(
    raw_llm_content: Optional[str],
    model_schema_map: Dict[str, Type[SQLModel]],
    revision_info_for_error: str = "LLM Output",
    analytics_collector: Optional[WorkflowAnalyticsCollector] = None,
) -> list[Dict[str, Any]]:
    """
    Parses raw LLM JSON content, unwraps structures, and validates a list of objects
    against a map of SQLModel schemas.

    Args:
        raw_llm_content: The raw string content from the LLM.
        model_schema_map: A dictionary mapping a '_type' string to a SQLModel class.
        revision_info_for_error: String for error messages.

    Returns:
        A list of validated dictionaries.

    Raises:
        LLMOutputParseError: If parsing fails or the unwrapped data is not a list/dict.
        LLMOutputValidationError: If any object in the list fails validation.
    """
    if not raw_llm_content:
        raise LLMOutputParseError(
            f"{revision_info_for_error}: LLM output is empty.", raw_llm_content
        )

    try:
        parsed_json = json.loads(raw_llm_content)
    except json.JSONDecodeError as e:
        raise LLMOutputParseError(
            f"{revision_info_for_error}: JSON parsing failed. Error: {e}.",
            raw_llm_content,
            e,
        )

    unwrapped_data = _unwrap_llm_output(parsed_json)
    data_list = unwrapped_data if isinstance(unwrapped_data, list) else [unwrapped_data]

    validated_objects = []
    for item in data_list:
        if not isinstance(item, dict):
            raise LLMOutputValidationError(
                f"{revision_info_for_error}: Item in list is not a dictionary.", item
            )

        type_key = item.get("_type")
        if not type_key:
            raise LLMOutputValidationError(
                f"{revision_info_for_error}: Missing '_type' key in object.", item
            )

        model_class = model_schema_map.get(type_key)
        if not model_class:
            raise LLMOutputValidationError(
                f"{revision_info_for_error}: Type '{type_key}' not in schema map.",
                item,
            )

        try:
            filtered_data = _filter_special_fields_for_validation(item.copy())
            model_class.model_validate(filtered_data)
            validated_objects.append(item)
        except PydanticValidationError as e:
            errors = json.dumps(e.errors())
            msg = f"{revision_info_for_error}: Validation failed for '{type_key}'. Errors: {errors}"
            if analytics_collector:
                analytics_collector.record_llm_output_validation_error(
                    {"error": msg, "invalid_object": item}
                )
            raise LLMOutputValidationError(msg, item, e)
        except Exception as e:
            msg = f"{revision_info_for_error}: Unexpected validation error for '{type_key}'. Error: {e}"
            if analytics_collector:
                analytics_collector.record_llm_output_validation_error(
                    {"error": msg, "invalid_object": item}
                )
            raise LLMOutputValidationError(msg, item, e)

    return validated_objects


def process_and_validate_raw_json(
    raw_llm_content: str,
    revision_info_for_error: str,
    target_json_schema: Optional[Dict[str, Any]] = None,
) -> Union[Dict[str, Any], list[Dict[str, Any]]]:
    """
    Parses, unwraps, and validates raw JSON content against a schema.

    Args:
        raw_llm_content: The raw string from the LLM.
        revision_info_for_error: A string for error reporting.
        target_json_schema: An optional JSON schema for validation.

    Returns:
        The validated dictionary or list of dictionaries.

    Raises:
        LLMOutputParseError: If the content is empty, not valid JSON, or not a dictionary or list.
        LLMOutputValidationError: If the content fails schema validation.
    """
    if not raw_llm_content:
        raise LLMOutputParseError(
            message=f"{revision_info_for_error}: LLM returned empty content.",
            raw_content=raw_llm_content,
        )

    try:
        parsed_json = json.loads(raw_llm_content)
    except json.JSONDecodeError as e:
        raise LLMOutputParseError(
            message=f"{revision_info_for_error}: Failed to parse LLM output as JSON. Error: {e}.",
            raw_content=raw_llm_content,
            original_exception=e,
        )

    unwrapped_data = _unwrap_llm_output(parsed_json)

    if not isinstance(unwrapped_data, (dict, list)):
        raise LLMOutputParseError(
            message=f"{revision_info_for_error}: Expected a dictionary or list for validation, but got {type(unwrapped_data).__name__}.",
            raw_content=raw_llm_content,
        )

    if target_json_schema and not is_json_valid(unwrapped_data, target_json_schema):
        raise LLMOutputValidationError(
            message=f"{revision_info_for_error}: LLM output failed validation against the target JSON schema.",
            parsed_json=unwrapped_data,
        )

    return unwrapped_data
