import pytest
import datetime
import enum
from typing import (
    List,
    Dict,
    Optional,
    Union,
    Any,
    Set,
    Tuple,
)
from extrai.utils.type_mapping import (
    get_python_type_str_from_pydantic_annotation,
    map_sql_type_to_llm_type,
    _handle_list_type,
    _handle_dict_type,
    _handle_union_type,
    _handle_generic_or_unknown_type,
    _process_union_types,
)

# --- Test Data ---


class MyEnum(enum.Enum):
    A = 1
    B = 2


class CustomType:
    pass


class SecretStr:
    pass


# --- Tests for get_python_type_str_from_pydantic_annotation ---


def test_get_python_type_str_base_types():
    assert get_python_type_str_from_pydantic_annotation(int) == "int"
    assert get_python_type_str_from_pydantic_annotation(str) == "str"
    assert get_python_type_str_from_pydantic_annotation(bool) == "bool"
    assert get_python_type_str_from_pydantic_annotation(float) == "float"
    assert get_python_type_str_from_pydantic_annotation(datetime.date) == "date"
    assert get_python_type_str_from_pydantic_annotation(datetime.datetime) == "datetime"
    assert get_python_type_str_from_pydantic_annotation(bytes) == "bytes"
    assert get_python_type_str_from_pydantic_annotation(Any) == "any"
    assert get_python_type_str_from_pydantic_annotation(type(None)) == "none"


def test_get_python_type_str_complex_types():
    # List
    assert get_python_type_str_from_pydantic_annotation(List[int]) == "list[int]"
    assert get_python_type_str_from_pydantic_annotation(list) == "list"

    # Dict
    assert (
        get_python_type_str_from_pydantic_annotation(Dict[str, int]) == "dict[str,int]"
    )
    assert get_python_type_str_from_pydantic_annotation(dict) == "dict"

    # Optional
    assert get_python_type_str_from_pydantic_annotation(Optional[int]) == "int"
    assert get_python_type_str_from_pydantic_annotation(Optional[str]) == "str"

    # Union
    # Note: Union[int, str] order is not guaranteed in string representation across versions/implementations strictly,
    # but the implementation sorts them.
    assert (
        get_python_type_str_from_pydantic_annotation(Union[int, str])
        == "union[int,str]"
    )
    assert (
        get_python_type_str_from_pydantic_annotation(Union[str, int])
        == "union[int,str]"
    )
    assert (
        get_python_type_str_from_pydantic_annotation(Union[int, None]) == "int"
    )  # Same as Optional

    # Nested
    assert (
        get_python_type_str_from_pydantic_annotation(List[Dict[str, Any]])
        == "list[dict[str,any]]"
    )


def test_get_python_type_str_enum():
    assert get_python_type_str_from_pydantic_annotation(MyEnum) == "enum"


def test_get_python_type_str_custom_and_fallback():
    # SecretStr simulation (by name)
    # The code checks hasattr(annotation, "__name__") and name_lower == "secretstr"
    # To test this we can pass the class itself if it matches, or mock it.
    # Actually SecretStr is usually pydantic.SecretStr.
    # Let's create a dummy class with that name.

    class SecretStr:
        pass

    assert get_python_type_str_from_pydantic_annotation(SecretStr) == "str"

    # Custom Type
    assert get_python_type_str_from_pydantic_annotation(CustomType) == "customtype"

    # Fallback with typing.
    # The fallback code does: str(annotation).lower().replace("typing.", "")
    # and handles "~" prefix.
    # We can pass something that doesn't match other rules.
    assert get_python_type_str_from_pydantic_annotation("JustAString") == "justastring"

    # Test ForwardRef style string (starts with ~)
    assert get_python_type_str_from_pydantic_annotation("~ForwardRef") == "forwardref"


def test_process_union_types_edge_case():
    # Test _process_union_types with empty args
    # This is hard to trigger via get_python_type_str because Union[] is invalid syntax usually,
    # but we can call the helper directly.
    assert _process_union_types([], lambda x: x) == "union"

    # Test with duplicates and sorting
    args = [int, int, str]
    # recurse_func needs to return the string rep
    recurse = lambda x: x.__name__ if hasattr(x, "__name__") else str(x)
    # Actually the function expects string return from recurse_func
    # Let's use a simple mock
    mock_recurse = lambda x: str(x)

    # Test deduplication and sorting: 'a', 'b', 'a' -> 'a', 'b'
    assert _process_union_types(["b", "a", "a"], lambda x: x) == "union[a,b]"

    # Test single element after processing
    assert _process_union_types(["a", "a"], lambda x: x) == "a"

    # Test None filtering ("none" string)
    assert _process_union_types(["a", "none"], lambda x: x) == "a"


# --- Tests for map_sql_type_to_llm_type and helpers ---


def test_map_sql_type_simple():
    assert map_sql_type_to_llm_type("INTEGER", "int") == "integer"
    assert map_sql_type_to_llm_type("VARCHAR", "str") == "string"
    assert map_sql_type_to_llm_type("BOOLEAN", "bool") == "boolean"
    assert map_sql_type_to_llm_type("FLOAT", "float") == "number (float/decimal)"
    assert map_sql_type_to_llm_type("DATE", "date") == "string (date format)"
    assert (
        map_sql_type_to_llm_type("DATETIME", "datetime") == "string (datetime format)"
    )
    assert map_sql_type_to_llm_type("BLOB", "bytes") == "string (base64 encoded)"
    assert map_sql_type_to_llm_type("ENUM", "enum") == "string (enum)"
    assert map_sql_type_to_llm_type("ANY", "any") == "any"
    assert map_sql_type_to_llm_type("NONE", "none") == "null"


def test_handle_list_type():
    assert _handle_list_type("list[int]") == "array[integer]"
    assert _handle_list_type("list[str]") == "array[string]"
    assert _handle_list_type("notalist") is None

    # Integration via main function
    assert map_sql_type_to_llm_type("", "list[int]") == "array[integer]"


def test_handle_dict_type():
    assert _handle_dict_type("dict[str,int]") == "object[string,integer]"
    assert _handle_dict_type("dict[str, str]") == "object[string,string]"  # spacing
    assert _handle_dict_type("notadict") is None

    # Test ValueError handling (malformed dict string)
    # The code splits by ",", 1. If no comma, it raises ValueError and returns "object"
    assert _handle_dict_type("dict[int]") == "object"

    # Integration via main function
    assert map_sql_type_to_llm_type("", "dict[str,int]") == "object[string,integer]"


def test_handle_union_type():
    assert (
        _handle_union_type("union[int,str]") == "union[integer,string]"
    )  # sorted: integer, string -> integer, string?
    # int->integer, str->string. sorted(['integer', 'string']) -> ['integer', 'string']

    assert _handle_union_type("union[str,int]") == "union[integer,string]"

    # Single type in union
    assert _handle_union_type("union[int]") == "integer"

    # Empty parts -> "any"
    assert _handle_union_type("union[]") == "any"
    assert _handle_union_type("union[ ]") == "any"

    assert _handle_union_type("notaunion") is None

    # Integration
    assert map_sql_type_to_llm_type("", "union[int,str]") == "union[integer,string]"


def test_handle_generic_or_unknown_type():
    # list
    assert _handle_generic_or_unknown_type("list", "") == "array"
    # list with text in sql -> None (fallback)
    assert _handle_generic_or_unknown_type("list", "text[]") is None

    # dict
    assert _handle_generic_or_unknown_type("dict", "") == "object"

    # unknown
    assert _handle_generic_or_unknown_type("unknown_stuff", "json") == "object"
    assert _handle_generic_or_unknown_type("unknown_stuff", "array") == "array"
    assert _handle_generic_or_unknown_type("unknown_stuff", "other") == "string"

    assert _handle_generic_or_unknown_type("other", "") is None


def test_sql_keyword_fallback():
    # Only if python type not handled above
    assert map_sql_type_to_llm_type("int", "other") == "integer"
    assert map_sql_type_to_llm_type("text", "other") == "string"
    assert map_sql_type_to_llm_type("json", "other") == "object"
    assert map_sql_type_to_llm_type("array", "other") == "array"


def test_final_fallback():
    assert map_sql_type_to_llm_type("nomatch", "nomatch") == "string"


def test_map_sql_type_generic_integration():
    # Trigger _handle_generic_or_unknown_type via map_sql_type_to_llm_type
    # "list" -> "array" (if sql type is not text)
    assert map_sql_type_to_llm_type("", "list") == "array"

    # "unknown" with "array" in sql -> "array"
    assert map_sql_type_to_llm_type("ARRAY", "unknown_type") == "array"


# --- Additional Coverage for Origin Handlers ---


def test_origin_handler_list_variations():
    # Test list vs List origin
    assert get_python_type_str_from_pydantic_annotation(List[int]) == "list[int]"
    assert get_python_type_str_from_pydantic_annotation(list) == "list"
    # To test 'args' presence check for list/List, we relied on List[int] vs list.


def test_origin_handler_dict_variations():
    # Test dict vs Dict origin
    assert (
        get_python_type_str_from_pydantic_annotation(Dict[str, int]) == "dict[str,int]"
    )
    assert get_python_type_str_from_pydantic_annotation(dict) == "dict"


def test_origin_handler_optional_none():
    # Optional handler: if args[0] is type(None) -> "none"
    # Optional[None] is basically NoneType
    assert get_python_type_str_from_pydantic_annotation(Optional[type(None)]) == "none"
