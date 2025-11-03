import jsonschema
from typing import Any, Dict


def is_json_valid(
    json_data_to_validate: Any, json_schema_definition: Dict[str, Any]
) -> bool:
    """
    Validates JSON data against a JSON schema.

    Args:
        json_data_to_validate: The Python object (e.g., dict, list) to validate.
                               This should be the result of json.loads() if the input was a string.
        json_schema_definition: The JSON schema as a Python dictionary.

    Returns:
        True if the JSON data is valid against the schema, False otherwise.
    """
    try:
        jsonschema.validate(
            instance=json_data_to_validate, schema=json_schema_definition
        )
        return True
    except jsonschema.exceptions.ValidationError:
        # print(f"JSON Validation Error: {ve.message}") # Optional: for debugging
        return False
    except jsonschema.exceptions.SchemaError:
        # This indicates the schema itself is invalid, which should ideally be caught earlier.
        # print(f"Invalid JSON Schema: {se.message}") # Optional: for debugging
        return False
    except Exception:
        # Catch any other unexpected errors during validation
        # print(f"An unexpected error occurred during JSON validation: {e}") # Optional: for debugging
        return False
