import unittest
from extrai.utils.json_validation_utils import is_json_valid
from typing import Any, Dict
import unittest.mock


class TestJsonValidationUtils(unittest.TestCase):
    def test_valid_simple_object(self):
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "city": {"type": "string"},
            },
            "required": ["name", "age"],
        }
        data = {"name": "Alice", "age": 30}
        self.assertTrue(is_json_valid(data, schema))

        data_with_optional = {"name": "Bob", "age": 25, "city": "New York"}
        self.assertTrue(is_json_valid(data_with_optional, schema))

    def test_invalid_data_missing_required_field(self):
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
            "required": ["name", "age"],
        }
        data = {"name": "Charlie"}  # Missing 'age'
        self.assertFalse(is_json_valid(data, schema))

    def test_invalid_data_wrong_type(self):
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
            "required": ["name", "age"],
        }
        data = {"name": "David", "age": "thirty"}  # 'age' is string, not number
        self.assertFalse(is_json_valid(data, schema))

    def test_invalid_data_additional_properties_not_allowed(self):
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        }
        data = {"name": "Eve", "extra_field": "unexpected"}
        self.assertFalse(is_json_valid(data, schema))

    def test_valid_data_additional_properties_allowed_by_default(self):
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            # additionalProperties is true by default
        }
        data = {"name": "Frank", "extra_field": "allowed"}
        self.assertTrue(is_json_valid(data, schema))

    def test_invalid_schema_definition_typo(self):
        # Schema itself is invalid
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "name": {"type": "strng"}  # Typo in type
            },
        }
        data = {"name": "Grace"}
        self.assertFalse(is_json_valid(data, schema))  # Should catch SchemaError

    def test_invalid_schema_not_a_dict(self):
        schema_list: Any = [{"type": "string"}]  # Schema must be a dict
        data = "Hello"
        self.assertFalse(is_json_valid(data, schema_list))

        schema_none: Any = None  # Schema cannot be None
        self.assertFalse(is_json_valid(data, schema_none))

    def test_empty_schema_validates_any_json(self):
        # An empty schema {} is valid and accepts any valid JSON.
        schema: Dict[str, Any] = {}
        self.assertTrue(is_json_valid({"key": "value"}, schema))
        self.assertTrue(is_json_valid([1, 2, 3], schema))
        self.assertTrue(is_json_valid("a string", schema))
        self.assertTrue(is_json_valid(123, schema))
        self.assertTrue(is_json_valid(True, schema))
        self.assertTrue(is_json_valid(None, schema))
        self.assertTrue(is_json_valid({}, schema))
        self.assertTrue(is_json_valid([], schema))

    def test_valid_array_data(self):
        schema: Dict[str, Any] = {"type": "array", "items": {"type": "integer"}}
        data = [1, 2, 3, 4]
        self.assertTrue(is_json_valid(data, schema))

    def test_invalid_array_data_wrong_item_type(self):
        schema: Dict[str, Any] = {"type": "array", "items": {"type": "integer"}}
        data = [1, 2, "three", 4]  # "three" is not an integer
        self.assertFalse(is_json_valid(data, schema))

    def test_valid_empty_array(self):
        schema: Dict[str, Any] = {"type": "array", "items": {"type": "string"}}
        data: list = []
        self.assertTrue(is_json_valid(data, schema))

    def test_valid_primitive_types(self):
        schema_string: Dict[str, Any] = {"type": "string"}
        self.assertTrue(is_json_valid("hello", schema_string))
        self.assertFalse(is_json_valid(123, schema_string))

        schema_number: Dict[str, Any] = {"type": "number"}
        self.assertTrue(is_json_valid(123.45, schema_number))
        self.assertTrue(is_json_valid(100, schema_number))
        self.assertFalse(is_json_valid("not a number", schema_number))

        schema_integer: Dict[str, Any] = {"type": "integer"}
        self.assertTrue(is_json_valid(123, schema_integer))
        self.assertFalse(is_json_valid(123.45, schema_integer))  # float is not integer

        schema_boolean: Dict[str, Any] = {"type": "boolean"}
        self.assertTrue(is_json_valid(True, schema_boolean))
        self.assertTrue(is_json_valid(False, schema_boolean))
        self.assertFalse(is_json_valid("true", schema_boolean))

        schema_null: Dict[str, Any] = {"type": "null"}
        self.assertTrue(is_json_valid(None, schema_null))
        self.assertFalse(is_json_valid(0, schema_null))
        self.assertFalse(is_json_valid("", schema_null))

    def test_enum_validation(self):
        schema: Dict[str, Any] = {"type": "string", "enum": ["red", "amber", "green"]}
        self.assertTrue(is_json_valid("red", schema))
        self.assertFalse(is_json_valid("blue", schema))

    def test_nested_object_validation(self):
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "username": {"type": "string"},
                    },
                    "required": ["id", "username"],
                },
                "status": {"type": "string"},
            },
            "required": ["user"],
        }
        valid_data = {"user": {"id": 1, "username": "testuser"}, "status": "active"}
        self.assertTrue(is_json_valid(valid_data, schema))

        invalid_data_nested_missing = {"user": {"id": 2}}  # username missing
        self.assertFalse(is_json_valid(invalid_data_nested_missing, schema))

        invalid_data_nested_wrong_type = {
            "user": {"id": "3", "username": "another"}
        }  # id is string
        self.assertFalse(is_json_valid(invalid_data_nested_wrong_type, schema))

    def test_data_is_none(self):
        schema: Dict[str, Any] = {"type": "null"}
        self.assertTrue(is_json_valid(None, schema))

        schema_string: Dict[str, Any] = {"type": "string"}
        self.assertFalse(is_json_valid(None, schema_string))  # None is not a string

        schema_object: Dict[str, Any] = {
            "type": "object",
            "properties": {"key": {"type": "string"}},
        }
        self.assertFalse(is_json_valid(None, schema_object))  # None is not an object

    def test_unexpected_exception_during_validation(self):
        # This test is a bit tricky as we want to simulate an unexpected error
        # within jsonschema.validate that isn't a ValidationError or SchemaError.
        # We can mock jsonschema.validate to raise a generic Exception.

        # We need to import jsonschema here or ensure it's imported where is_json_valid is defined
        # for the mock to correctly target it. Assuming jsonschema is imported in json_validation_utils.

        mock_jsonschema_validate = unittest.mock.patch(
            "extrai.utils.json_validation_utils.jsonschema.validate"
        ).start()

        def mock_validate_raises_exception(*args, **kwargs):
            raise Exception("Simulated unexpected error")

        mock_jsonschema_validate.side_effect = mock_validate_raises_exception

        schema = {"type": "string"}
        data = "test"
        self.assertFalse(is_json_valid(data, schema))

        # Stop the mock
        unittest.mock.patch.stopall()


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
