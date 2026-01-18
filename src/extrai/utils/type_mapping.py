import datetime
import enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    get_args,
    get_origin,
    Union as TypingUnion,
)


def _process_union_types(args, recurse_func):
    """Helper to process Union types, filtering and sorting."""
    if not args:
        return "union"
    union_types_str = [recurse_func(arg) for arg in args]
    processed_union_types = sorted(set(t for t in union_types_str if t != "none"))
    if len(processed_union_types) == 1:
        return processed_union_types[0]
    return f"union[{','.join(processed_union_types)}]"


# Handler registry for different type origins
ORIGIN_HANDLERS = {
    Optional: lambda args, r: r(args[0])
    if args and args[0] is not type(None)
    else "none",
    list: lambda args, r: f"list[{','.join([r(arg) for arg in args])}]"
    if args
    else "list",
    List: lambda args, r: f"list[{','.join([r(arg) for arg in args])}]"
    if args
    else "list",
    dict: lambda args, r: f"dict[{r(args[0])},{r(args[1])}]"
    if args and len(args) == 2
    else "dict",
    Dict: lambda args, r: f"dict[{r(args[0])},{r(args[1])}]"
    if args and len(args) == 2
    else "dict",
    TypingUnion: _process_union_types,
}

# Data-driven approach for base types
BASE_TYPE_MAP = {
    int: "int",
    str: "str",
    bool: "bool",
    float: "float",
    datetime.date: "date",
    datetime.datetime: "datetime",
    bytes: "bytes",
    Any: "any",
    type(None): "none",
}


def get_python_type_str_from_pydantic_annotation(annotation: Any) -> str:
    """Helper function to get a simplified string from Pydantic/SQLModel annotations."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in ORIGIN_HANDLERS:
        return ORIGIN_HANDLERS[origin](
            args, get_python_type_str_from_pydantic_annotation
        )

    if annotation in BASE_TYPE_MAP:
        return BASE_TYPE_MAP[annotation]

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return "enum"

    if hasattr(annotation, "__name__"):
        name_lower = annotation.__name__.lower()
        if name_lower == "secretstr":
            return "str"
        return name_lower

    # Fallback
    cleaned_annotation_str = str(annotation).lower().replace("typing.", "")
    if cleaned_annotation_str.startswith("~"):
        cleaned_annotation_str = cleaned_annotation_str[1:]
    return cleaned_annotation_str


# --- Data-driven mappings for type conversion ---
SIMPLE_PYTHON_TYPE_MAP = {
    "int": "integer",
    "str": "string",
    "bool": "boolean",
    "float": "number (float/decimal)",
    "date": "string (date format)",
    "datetime": "string (datetime format)",
    "bytes": "string (base64 encoded)",
    "enum": "string (enum)",
    "any": "any",
    "none": "null",
}

SQL_TYPE_KEYWORDS = [
    ("int", "integer"),
    ("char", "string"),
    ("text", "string"),
    ("clob", "string"),
    ("bool", "boolean"),
    ("date", "string (date/datetime format)"),
    ("time", "string (date/datetime format)"),
    ("numeric", "number (float/decimal)"),
    ("decimal", "number (float/decimal)"),
    ("float", "number (float/decimal)"),
    ("double", "number (float/decimal)"),
    ("json", "object"),
    ("array", "array"),
]


# --- Handlers for complex and generic types ---
def _handle_list_type(python_type_lower: str) -> Optional[str]:
    """Handles list[...] and array[...] type mappings."""
    if python_type_lower.startswith("list[") and python_type_lower.endswith("]"):
        inner_type_str = python_type_lower[5:-1]
        mapped_inner_type = map_sql_type_to_llm_type("", inner_type_str)
        return f"array[{mapped_inner_type}]"
    return None


def _handle_dict_type(python_type_lower: str) -> Optional[str]:
    """Handles dict[...] and object[...] type mappings."""
    if python_type_lower.startswith("dict[") and python_type_lower.endswith("]"):
        inner_types_str = python_type_lower[5:-1]
        try:
            key_type_str, value_type_str = inner_types_str.split(",", 1)
            mapped_key_type = map_sql_type_to_llm_type("", key_type_str.strip())
            mapped_value_type = map_sql_type_to_llm_type("", value_type_str.strip())
            return f"object[{mapped_key_type},{mapped_value_type}]"
        except ValueError:
            return "object"
    return None


def _handle_union_type(python_type_lower: str) -> Optional[str]:
    """Handles union[...] type mappings."""
    if python_type_lower.startswith("union[") and python_type_lower.endswith("]"):
        inner_types_str = python_type_lower[6:-1]
        union_parts = [p.strip() for p in inner_types_str.split(",") if p.strip()]
        mapped_parts = sorted(
            set(map_sql_type_to_llm_type("", part) for part in union_parts)
        )
        if not mapped_parts:
            return "any"
        return (
            mapped_parts[0]
            if len(mapped_parts) == 1
            else f"union[{','.join(mapped_parts)}]"
        )
    return None


def _handle_generic_or_unknown_type(
    python_type_lower: str, sql_type_lower: str
) -> Optional[str]:
    """Handles ambiguous types like plain 'list' or 'dict' and unknown types."""
    if python_type_lower == "list":
        if "text" in sql_type_lower:  # Let the SQL keyword mapping handle this case
            return None

        return "array"

    if python_type_lower == "dict":
        return "object"

    if python_type_lower.startswith("unknown"):
        if "json" in sql_type_lower:
            return "object"
        if "array" in sql_type_lower:
            return "array"
        return "string"
    return None


def map_sql_type_to_llm_type(sql_type_str: str, python_type_str: str) -> str:
    """
    Maps SQL/Python types to simpler LLM-friendly type strings using a dispatcher pattern.
    """
    sql_type_lower = str(sql_type_str).lower()
    python_type_lower = str(python_type_str).lower()

    # 1. Handle complex Python types first
    for handler in [_handle_list_type, _handle_dict_type, _handle_union_type]:
        result = handler(python_type_lower)
        if result:
            return result

    # 2. Look up in the simple Python type map
    if python_type_lower in SIMPLE_PYTHON_TYPE_MAP:
        return SIMPLE_PYTHON_TYPE_MAP[python_type_lower]

    # 3. Handle generic or unknown types, which have precedence over broad SQL keywords
    result = _handle_generic_or_unknown_type(python_type_lower, sql_type_lower)
    if result:
        return result

    # 4. Search through SQL type keywords as a fallback
    for keyword, llm_type in SQL_TYPE_KEYWORDS:
        if keyword in sql_type_lower:
            return llm_type

    # 5. Final fallback if no other rule matched
    return "string"
