import json
import pytest
from typing import Optional, List, Any, Dict

from sqlmodel import SQLModel, Field

from extrai.utils.llm_output_processing import (
    _filter_special_fields_for_validation,
    _unwrap_llm_output,
    process_and_validate_llm_output,
    process_and_validate_raw_json,
)
from extrai.core.errors import (
    LLMOutputParseError,
    LLMOutputValidationError,
)

# --- Dummy SQLModels for testing ---


class SimpleModel(SQLModel):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    value: Optional[int] = None


class ArticleData(SQLModel):
    id: int
    title: str
    summary: str
    key_topics: Optional[Dict[str, str]] = None
    categories: Optional[Dict[str, str]] = None
    raw_text: Optional[str] = None
    source_filename: Optional[str] = None


class ModelWithRefs(SQLModel):
    name: str
    item_ref_id: Optional[int] = None
    items_ref_ids: Optional[List[int]] = None


# --- Helper for creating JSON strings ---
def make_json_str(data: Any) -> str:
    return json.dumps(data)


# --- Tests for _unwrap_llm_output ---


@pytest.mark.parametrize(
    "input_data, expected_output",
    [
        # No wrapping
        ({"name": "test"}, "test"),  # Single key dict is unwrapped
        ([1, 2, 3], [1, 2, 3]),
        ("a string", "a string"),
        (None, None),
        # Wrapped in 'result'
        ({"result": {"name": "test"}}, {"name": "test"}),
        ({"result": [1, 2, 3]}, [1, 2, 3]),
        # Wrapped in other priority keys
        ({"data": {"name": "test"}}, {"name": "test"}),
        ({"entities": [1, 2, 3]}, [1, 2, 3]),
        # Single dynamic key
        ({"DynamicKey": {"name": "test"}}, {"name": "test"}),
        # Priority key takes precedence over single key
        (
            {"result": {"DynamicKey": {"name": "test"}}},
            {"DynamicKey": {"name": "test"}},
        ),
        # No unwrapping if multiple non-priority keys
        (
            {"Key1": {"v": 1}, "Key2": {"v": 2}},
            {"Key1": {"v": 1}, "Key2": {"v": 2}},
        ),
        # Unwrapping a non-dict from a single key
        ({"DynamicKey": "not_a_dict"}, "not_a_dict"),
    ],
)
def test_unwrap_llm_output(input_data, expected_output):
    assert _unwrap_llm_output(input_data) == expected_output


# --- Tests for _filter_special_fields_for_validation ---


@pytest.mark.parametrize(
    "input_data, expected_output",
    [
        ({}, {}),
        (
            {
                "_temp_id": "t1",
                "_type": "Dummy",
                "item_ref_id": 1,
                "items_ref_ids": [2, 3],
                "another_ref_id": 4,
            },
            {},
        ),
        ({"name": "Test", "value": 10}, {"name": "Test", "value": 10}),
        (
            {
                "_temp_id": "t1",
                "name": "Test",
                "_type": "Dummy",
                "value": 10,
                "item_ref_id": 1,
                "related_items_ref_ids": [5],
            },
            {"name": "Test", "value": 10},
        ),
        (None, None),
        ([], []),
        ("a string", "a string"),
        (123, 123),
    ],
)
def test_filter_special_fields(input_data, expected_output):
    assert _filter_special_fields_for_validation(input_data) == expected_output


# --- Tests for process_and_validate_llm_output ---


# --- Successful Cases ---
@pytest.mark.parametrize(
    "raw_content_str, model_map, expected_output",
    [
        # Single object in a list
        (
            make_json_str([{"_type": "SimpleModel", "name": "Test1", "value": 100}]),
            {"SimpleModel": SimpleModel},
            [{"_type": "SimpleModel", "name": "Test1", "value": 100}],
        ),
        # Single object, unwrapped from 'result'
        (
            make_json_str(
                {"result": {"_type": "SimpleModel", "name": "Test2", "value": 200}}
            ),
            {"SimpleModel": SimpleModel},
            [{"_type": "SimpleModel", "name": "Test2", "value": 200}],
        ),
        # Multiple objects
        (
            make_json_str(
                [
                    {"_type": "SimpleModel", "name": "Test1", "value": 100},
                    {"_type": "ModelWithRefs", "name": "Ref1"},
                ]
            ),
            {"SimpleModel": SimpleModel, "ModelWithRefs": ModelWithRefs},
            [
                {"_type": "SimpleModel", "name": "Test1", "value": 100},
                {"_type": "ModelWithRefs", "name": "Ref1"},
            ],
        ),
        # Special fields are preserved in the output
        (
            make_json_str(
                [
                    {
                        "_type": "SimpleModel",
                        "_temp_id": "t1",
                        "name": "Test3",
                        "value": 300,
                        "item_ref_id": 1,
                    }
                ]
            ),
            {"SimpleModel": SimpleModel},
            [
                {
                    "_type": "SimpleModel",
                    "_temp_id": "t1",
                    "name": "Test3",
                    "value": 300,
                    "item_ref_id": 1,
                }
            ],
        ),
    ],
)
def test_process_and_validate_llm_output_success_cases(
    raw_content_str, model_map, expected_output
):
    assert (
        process_and_validate_llm_output(raw_content_str, model_map) == expected_output
    )


# --- Parsing Errors (LLMOutputParseError) ---
@pytest.mark.parametrize(
    "raw_content, error_msg_snippet",
    [
        ('{"name": "Test", "value": 10', "JSON parsing failed"),
        ("", "LLM output is empty"),
        (None, "LLM output is empty"),
    ],
)
def test_process_and_validate_llm_output_parse_errors(raw_content, error_msg_snippet):
    with pytest.raises(LLMOutputParseError) as excinfo:
        process_and_validate_llm_output(raw_content, {"SimpleModel": SimpleModel})
    assert error_msg_snippet in str(excinfo.value)
    assert excinfo.value.raw_content == raw_content


# --- Validation Errors (LLMOutputValidationError) ---
@pytest.mark.parametrize(
    "data, model_map, error_msg_snippet",
    [
        # Not a dict
        ([1, 2, 3], {"SimpleModel": SimpleModel}, "Item in list is not a dictionary"),
        # Missing _type
        (
            [{"name": "Test"}],
            {"SimpleModel": SimpleModel},
            "Missing '_type' key in object",
        ),
        # _type not in map
        (
            [{"_type": "UnknownModel", "name": "Test"}],
            {"SimpleModel": SimpleModel},
            "Type 'UnknownModel' not in schema map",
        ),
        # Pydantic validation failure
        (
            [{"_type": "SimpleModel", "value": 100}],
            {"SimpleModel": SimpleModel},
            "Validation failed for 'SimpleModel'",
        ),
    ],
)
def test_process_and_validate_llm_output_validation_errors(
    data, model_map, error_msg_snippet
):
    raw_content = make_json_str(data)
    with pytest.raises(LLMOutputValidationError) as excinfo:
        process_and_validate_llm_output(raw_content, model_map)
    assert error_msg_snippet in str(excinfo.value)


# --- Error Message Content ---
@pytest.mark.parametrize(
    "error_type, raw_content, model_map",
    [
        (LLMOutputParseError, "invalid_json", {"SimpleModel": SimpleModel}),
        (
            LLMOutputValidationError,
            make_json_str([{"_type": "SimpleModel", "value": 123}]),
            {"SimpleModel": SimpleModel},
        ),
    ],
)
def test_errors_include_revision_info(error_type, raw_content, model_map):
    revision_info = "TestRevision:123, Attempt:1"
    with pytest.raises(error_type) as excinfo:
        process_and_validate_llm_output(
            raw_content, model_map, revision_info_for_error=revision_info
        )
    assert revision_info in str(excinfo.value)


def test_process_unexpected_error_during_validation():
    """
    Tests that an unexpected error during validation is handled correctly and
    that the analytics_collector is called if provided.
    """

    class MockAnalyticsCollector:
        def __init__(self):
            self.errors = []

        def record_llm_output_validation_error(self, error_info):
            self.errors.append(error_info)

    class MockModelRaisesError(SQLModel):
        _type: str = "MockModel"
        name: str

        @classmethod
        def model_validate(
            cls, obj, *, strict=None, from_attributes=None, context=None
        ):
            raise RuntimeError("Simulated unexpected validation error")

    analytics_collector = MockAnalyticsCollector()
    raw_content = make_json_str([{"_type": "MockModel", "name": "TestData"}])
    model_map = {"MockModel": MockModelRaisesError}

    with pytest.raises(LLMOutputValidationError) as excinfo:
        process_and_validate_llm_output(
            raw_content, model_map, analytics_collector=analytics_collector
        )

    # Verify the exception details
    assert "Unexpected validation error for 'MockModel'" in str(excinfo.value)
    assert isinstance(excinfo.value.validation_error, RuntimeError)

    # Verify the analytics collector was called
    assert len(analytics_collector.errors) == 1
    assert (
        "Unexpected validation error for 'MockModel'"
        in analytics_collector.errors[0]["error"]
    )


def test_process_and_validate_llm_output_pydantic_error_with_analytics():
    # This test is to ensure coverage of the analytics_collector call inside the
    # PydanticValidationError exception block.
    class MockAnalyticsCollector:
        def __init__(self):
            self.errors = []

        def record_llm_output_validation_error(self, error_info):
            self.errors.append(error_info)

    analytics_collector = MockAnalyticsCollector()
    raw_content = make_json_str([{"_type": "SimpleModel", "value": "not_an_int"}])
    model_map = {"SimpleModel": SimpleModel}

    with pytest.raises(LLMOutputValidationError):
        process_and_validate_llm_output(
            raw_content, model_map, analytics_collector=analytics_collector
        )

    assert len(analytics_collector.errors) == 1
    assert (
        "Validation failed for 'SimpleModel'" in analytics_collector.errors[0]["error"]
    )


# --- Tests for process_and_validate_raw_json ---


def test_process_and_validate_raw_json_success():
    """Tests successful parsing and unwrapping of valid JSON."""
    raw_content = make_json_str({"result": {"key": "value"}})
    result = process_and_validate_raw_json(raw_content, "test_success")
    assert result == {"key": "value"}


def test_process_and_validate_raw_json_success_with_schema():
    """Tests successful validation against a JSON schema."""
    raw_content = make_json_str({"key": "value", "count": 5})
    schema = {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["key", "count"],
    }
    result = process_and_validate_raw_json(
        raw_content, "test_schema_success", target_json_schema=schema
    )
    assert result == {"key": "value", "count": 5}


@pytest.mark.parametrize("raw_content", [None, ""])
def test_process_and_validate_raw_json_empty_content(raw_content):
    """Tests that empty or None content raises LLMOutputParseError."""
    with pytest.raises(LLMOutputParseError, match="LLM returned empty content"):
        process_and_validate_raw_json(raw_content, "test_empty")


def test_process_and_validate_raw_json_invalid_json():
    """Tests that malformed JSON raises LLMOutputParseError."""
    with pytest.raises(LLMOutputParseError, match="Failed to parse LLM output as JSON"):
        process_and_validate_raw_json("{'key': 'value'}", "test_invalid_json")


def test_process_and_validate_raw_json_not_dict_or_list():
    """Tests that unwrapped data that is not a dict or list raises LLMOutputParseError."""
    raw_content = make_json_str("this is just a string")
    with pytest.raises(
        LLMOutputParseError, match="Expected a dictionary or list for validation"
    ):
        process_and_validate_raw_json(raw_content, "test_not_dict_or_list")


def test_process_and_validate_raw_json_schema_validation_fails():
    """Tests that a schema validation failure raises LLMOutputValidationError."""
    # Adding a second key to prevent _unwrap_llm_output from returning just the value
    raw_content = make_json_str(
        {"key": 123, "another_key": "value"}
    )  # 'key' should be a string
    schema = {"type": "object", "properties": {"key": {"type": "string"}}}
    with pytest.raises(
        LLMOutputValidationError,
        match="failed validation against the target JSON schema",
    ):
        process_and_validate_raw_json(
            raw_content, "test_schema_fail", target_json_schema=schema
        )
