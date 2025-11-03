# extrai/core/schema_inspector.py

import json
from sqlalchemy import inspect, Column, Table
from sqlalchemy.orm import RelationshipProperty
from sqlalchemy.exc import NoInspectionAvailable
from sqlalchemy.schema import UniqueConstraint, PrimaryKeyConstraint
import enum
import datetime
from typing import Any, Dict, Type, Set, Optional, List, get_origin, get_args, Tuple
from sqlmodel import SQLModel


from typing import Union as TypingUnion


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


# Helper function to get a simplified string from Pydantic/SQLModel annotations
def _get_python_type_str_from_pydantic_annotation(annotation: Any) -> str:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in ORIGIN_HANDLERS:
        return ORIGIN_HANDLERS[origin](
            args, _get_python_type_str_from_pydantic_annotation
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
        mapped_inner_type = _map_sql_type_to_llm_type("", inner_type_str)
        return f"array[{mapped_inner_type}]"
    return None


def _handle_dict_type(python_type_lower: str) -> Optional[str]:
    """Handles dict[...] and object[...] type mappings."""
    if python_type_lower.startswith("dict[") and python_type_lower.endswith("]"):
        inner_types_str = python_type_lower[5:-1]
        try:
            key_type_str, value_type_str = inner_types_str.split(",", 1)
            mapped_key_type = _map_sql_type_to_llm_type("", key_type_str.strip())
            mapped_value_type = _map_sql_type_to_llm_type("", value_type_str.strip())
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
            set(_map_sql_type_to_llm_type("", part) for part in union_parts)
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


def _map_sql_type_to_llm_type(sql_type_str: str, python_type_str: str) -> str:
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


def _is_column_unique(column_obj: Column) -> bool:
    """Checks if a column has a unique constraint."""
    if column_obj.unique:
        return True
    if column_obj.table is not None:
        for constraint in column_obj.table.constraints:
            if isinstance(constraint, (UniqueConstraint, PrimaryKeyConstraint)):
                if column_obj.name in constraint.columns:
                    return True
    return False


def _get_python_type_from_column(column_obj: Column) -> str:
    """Safely extracts the Python type name from a column object."""
    try:
        return column_obj.type.python_type.__name__
    except NotImplementedError:
        return "unknown_not_implemented"
    except AttributeError:
        return "unknown_no_python_type_attr"
    except Exception:
        return "unknown_error_accessing_type"


def _build_column_info(
    column_obj: Column, is_unique: bool, python_type_name: str
) -> Dict[str, Any]:
    """Builds the column information dictionary."""
    col_info = {
        "type": str(column_obj.type),
        "python_type": python_type_name,
        "primary_key": column_obj.primary_key,
        "nullable": column_obj.nullable,
        "unique": is_unique,
        "foreign_key_to": None,
        "comment": column_obj.comment,
        "info_dict": column_obj.info,
    }
    if column_obj.foreign_keys:
        fk_constraint_obj = next(iter(column_obj.foreign_keys))
        col_info["foreign_key_to"] = str(fk_constraint_obj.column)
    return col_info


def _get_columns_from_inspector(inspector) -> Dict[str, Any]:
    """Extracts all column properties from a SQLAlchemy inspector."""
    columns_info = {}
    for col_attr in inspector.column_attrs:
        if not isinstance(col_attr.expression, Column):
            continue
        column_obj = col_attr.expression
        is_unique = _is_column_unique(column_obj)
        python_type_name = _get_python_type_from_column(column_obj)
        columns_info[col_attr.key] = _build_column_info(
            column_obj, is_unique, python_type_name
        )
    return columns_info


def _get_fks_from_secondary_table(rel_prop: RelationshipProperty) -> Set[str]:
    """Handles relationships that use a secondary table."""
    involved_fk_columns: Set[str] = set()
    if rel_prop.secondary is not None:
        for fk_constraint in rel_prop.secondary.foreign_key_constraints:
            for col in fk_constraint.columns:
                involved_fk_columns.add(str(col))
    return involved_fk_columns


def _get_fks_from_synchronize_pairs(rel_prop: RelationshipProperty) -> Set[str]:
    """Handles relationships that use synchronize_pairs."""
    involved_fk_columns: Set[str] = set()
    if rel_prop.synchronize_pairs:
        for local_join_col, remote_join_col in rel_prop.synchronize_pairs:
            if hasattr(local_join_col, "foreign_keys") and local_join_col.foreign_keys:
                involved_fk_columns.add(str(local_join_col))
            if (
                hasattr(remote_join_col, "foreign_keys")
                and remote_join_col.foreign_keys
            ):
                involved_fk_columns.add(str(remote_join_col))
    return involved_fk_columns


def _get_fks_from_direct_foreign_keys(rel_prop: RelationshipProperty) -> Set[str]:
    """Handles relationships that have direct foreign_keys."""
    involved_fk_columns: Set[str] = set()
    if hasattr(rel_prop, "foreign_keys") and rel_prop.foreign_keys is not None:
        for fk_col in rel_prop.foreign_keys:
            involved_fk_columns.add(str(fk_col))
    return involved_fk_columns


def _get_involved_foreign_keys(rel_prop: RelationshipProperty) -> Set[str]:
    """
    Finds all foreign key columns involved in a relationship by dispatching to helper functions.
    """
    if rel_prop.secondary is not None:
        return _get_fks_from_secondary_table(rel_prop)

    if rel_prop.synchronize_pairs:
        return _get_fks_from_synchronize_pairs(rel_prop)

    if hasattr(rel_prop, "foreign_keys") and rel_prop.foreign_keys is not None:
        return _get_fks_from_direct_foreign_keys(rel_prop)

    return set()


def _build_relationship_info(
    rel_prop: RelationshipProperty,
    involved_fk_columns: Set[str],
    recursion_path_tracker: Set[Type[Any]],
) -> Dict[str, Any]:
    """Builds the relationship information dictionary, including recursion."""
    related_model_class = rel_prop.mapper.class_
    return {
        "type": rel_prop.direction.name,
        "uselist": rel_prop.uselist,
        "related_model_name": related_model_class.__name__,
        "secondary_table_name": rel_prop.secondary.name
        if rel_prop.secondary is not None
        else None,
        "local_columns": [str(c) for c in rel_prop.local_columns],
        "remote_columns_in_join": [str(pair[1]) for pair in rel_prop.local_remote_pairs]
        if rel_prop.local_remote_pairs
        else [],
        "foreign_key_constraints_involved": sorted(involved_fk_columns),
        "back_populates": rel_prop.back_populates,
        "info_dict": rel_prop.info,
        "nested_schema": _inspect_sqlalchemy_model_recursive(
            related_model_class, recursion_path_tracker
        ),
    }


def _get_relationships_from_inspector(
    inspector, recursion_path_tracker: Set[Type[Any]]
) -> Dict[str, Any]:
    """Extracts all relationship properties from a SQLAlchemy inspector."""
    relationships_info = {}
    for name, rel_prop in inspector.relationships.items():
        if isinstance(rel_prop, RelationshipProperty):
            involved_fk_columns = _get_involved_foreign_keys(rel_prop)
            relationships_info[name] = _build_relationship_info(
                rel_prop, involved_fk_columns, recursion_path_tracker
            )
    return relationships_info


def _inspect_sqlalchemy_model_recursive(
    model_class: Type[Any], recursion_path_tracker: Set[Type[Any]]
) -> Dict[str, Any]:
    """
    Internal recursive function to introspect a SQLAlchemy model class,
    including column comments, info dictionaries, and handling recursion.

    Args:
        model_class: The SQLAlchemy model class to inspect.
        recursion_path_tracker: A set used to track visited models in the current
                                recursion path to prevent infinite loops.

    Returns:
        A dictionary containing the schema information of the model.
        If recursion is detected for a model already in the path, a
        simplified dictionary indicating recursion is returned.
        If the model cannot be inspected, an error dictionary is returned.
    """
    try:
        inspector = inspect(model_class)
    except NoInspectionAvailable:
        return {
            "error": f"Could not get an inspector for {model_class}. It might not be a valid SQLAlchemy mapped class."
        }

    if inspector is None:
        return {"error": f"Inspector is None for {model_class}."}

    table_obj = inspector.selectable
    table_info_dict = (
        getattr(table_obj, "info", None) if isinstance(table_obj, Table) else None
    )
    table_comment = (
        getattr(table_obj, "comment", None) if isinstance(table_obj, Table) else None
    )

    table_name_str = getattr(model_class, "__tablename__", model_class.__name__.lower())
    if hasattr(table_obj, "name") and table_obj.name:
        table_name_str = table_obj.name

    if model_class in recursion_path_tracker:
        return {
            "table_name": table_name_str,
            "model_name": model_class.__name__,
            "recursion_detected_for_type": model_class.__name__,
            "info_dict": table_info_dict,
            "comment": table_comment,
            "description_note": "Schema for this model is detailed elsewhere in the current path.",
        }

    recursion_path_tracker.add(model_class)

    schema_info: Dict[str, Any] = {
        "table_name": table_name_str,
        "model_name": model_class.__name__,
        "info_dict": table_info_dict,
        "comment": table_comment,
        "columns": _get_columns_from_inspector(inspector),
        "relationships": _get_relationships_from_inspector(
            inspector, recursion_path_tracker
        ),
    }

    recursion_path_tracker.remove(model_class)
    return schema_info


def inspect_sqlalchemy_model(model_class: Type[Any]) -> Dict[str, Any]:
    """
    Public wrapper function to start the SQLAlchemy model introspection.
    Initializes an empty set for tracking the recursion path.

    Args:
        model_class: The SQLAlchemy model class to inspect.

    Returns:
        A dictionary containing the schema information of the model,
        including nested schemas, comments, and info dictionaries.
    """
    return _inspect_sqlalchemy_model_recursive(model_class, set())


def _collect_all_sqla_models_recursively(
    current_model_class: Type[Any],
    all_discovered_models: List[Type[Any]],
    recursion_guard: Set[Type[Any]],
) -> None:
    """
    Recursively collects all unique SQLAlchemy model classes related to current_model_class.
    This function populates the `all_discovered_models` list, preserving order.

    Args:
        current_model_class: The SQLAlchemy model class currently being processed.
        all_discovered_models: A list to store all unique model classes found.
        recursion_guard: A set to track models visited in the current recursion
                         path to prevent infinite loops.
    """
    if current_model_class in recursion_guard:
        return
    recursion_guard.add(current_model_class)

    # Add the model if it's not already in the list to preserve order and uniqueness
    if current_model_class not in all_discovered_models:
        all_discovered_models.append(current_model_class)

    try:
        inspector = inspect(current_model_class)
    except NoInspectionAvailable:
        recursion_guard.remove(current_model_class)
        return

    if inspector is None:
        recursion_guard.remove(current_model_class)
        return

    for rel_prop in inspector.relationships:
        related_sqla_model_class = rel_prop.mapper.class_
        if related_sqla_model_class not in recursion_guard:
            _collect_all_sqla_models_recursively(
                related_sqla_model_class, all_discovered_models, recursion_guard
            )
    recursion_guard.remove(current_model_class)


def _get_prioritized_description(
    *,
    custom_desc: Optional[str] = None,
    pydantic_desc: Optional[str] = None,
    info_dict: Optional[Dict[str, Any]] = None,
    comment: Optional[str] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Centralized helper to determine the best description from multiple sources.
    Priority: custom -> pydantic -> info_dict['description'] -> comment.
    Also extracts any other key-value pairs from the info_dict.
    """
    description = None
    if custom_desc:
        description = custom_desc
    elif pydantic_desc:
        description = pydantic_desc

    other_info_from_dict = {}
    if isinstance(info_dict, dict):
        info_desc = info_dict.get("description")
        if info_desc and not description:
            description = info_desc
        other_info_from_dict = {
            k: v for k, v in info_dict.items() if k != "description"
        }

    if not description and comment:
        description = comment

    return description, other_info_from_dict


def _process_column_for_llm_schema(
    col_name: str,
    col_data: Dict[str, Any],
    pydantic_fields: Dict[str, Any],
    custom_descs: Dict[str, str],
    model_name: str,
) -> Tuple[str, str]:
    """Processes a single column to generate its LLM schema representation."""
    python_type_for_mapping = str(col_data.get("python_type", ""))
    pydantic_field_description = None

    if col_name in pydantic_fields:
        field_pydantic_info = pydantic_fields[col_name]
        if field_pydantic_info.annotation:
            pydantic_derived_type_str = _get_python_type_str_from_pydantic_annotation(
                field_pydantic_info.annotation
            )
            if (
                pydantic_derived_type_str
                and not pydantic_derived_type_str.startswith("unknown")
                and pydantic_derived_type_str != "any"
            ):
                python_type_for_mapping = pydantic_derived_type_str

        if field_pydantic_info.description:
            pydantic_field_description = field_pydantic_info.description

    llm_type = _map_sql_type_to_llm_type(
        str(col_data.get("type", "")),
        python_type_for_mapping,
    )

    description, other_info = _get_prioritized_description(
        custom_desc=custom_descs.get(col_name),
        pydantic_desc=pydantic_field_description,
        info_dict=col_data.get("info_dict"),
        comment=col_data.get("comment"),
    )

    if not description:
        description = f"Field '{col_name}' of type {llm_type} for {model_name}."

    additional_info_items_str = ""
    if other_info:
        try:
            additional_info_items_str = f" (Info: {json.dumps(other_info)})"
        except TypeError:
            additional_info_items_str = f" (Info: {str(other_info)})"

    final_description = f"{description}{additional_info_items_str}"
    formatted_string = f"{llm_type} // {final_description.strip()}"

    return col_name, formatted_string


def _process_relationship_for_llm_schema(
    rel_name: str, rel_data: Dict[str, Any], custom_descs: Dict[str, str]
) -> Optional[Tuple[str, str]]:
    """Processes a single relationship to generate its LLM schema representation."""
    related_model_name = rel_data.get("related_model_name", "UnknownRelatedModel")

    temp_ref_field_name_single = f"{rel_name}_ref_id"
    temp_ref_field_name_list = f"{rel_name}_ref_ids"

    custom_desc_lookup = (
        custom_descs.get(rel_name)
        or custom_descs.get(temp_ref_field_name_single)
        or custom_descs.get(temp_ref_field_name_list)
    )

    description, other_info = _get_prioritized_description(
        custom_desc=custom_desc_lookup,
        info_dict=rel_data.get("info_dict"),
    )

    additional_info_items_str = ""
    if other_info:
        try:
            additional_info_items_str = f" (Info: {json.dumps(other_info)})"
        except TypeError:
            additional_info_items_str = f" (Info: {str(other_info)})"

    ref_field_name_for_llm = ""
    field_type_for_llm = ""
    default_desc = ""

    if rel_data.get("uselist") is True:
        ref_field_name_for_llm = temp_ref_field_name_list
        field_type_for_llm = "array of strings (temporary IDs)"
        default_desc = f"A list of _temp_ids for related {related_model_name} entities in '{rel_name}'."
    elif rel_data.get("uselist") is False:
        ref_field_name_for_llm = temp_ref_field_name_single
        field_type_for_llm = "string (temporary ID)"
        default_desc = (
            f"The _temp_id of the related {related_model_name} for '{rel_name}'."
        )

    if not ref_field_name_for_llm:
        return None

    final_description = description or default_desc
    full_description = f"{final_description}{additional_info_items_str}"

    formatted_string = f"{field_type_for_llm} // {full_description.strip()}"

    return ref_field_name_for_llm, formatted_string


def _generate_model_level_description(
    model_name: str, raw_schema: Dict[str, Any], custom_descs: Dict[str, str]
) -> str:
    """Generates the complete model-level description block."""
    description, other_info = _get_prioritized_description(
        custom_desc=custom_descs.get("_model_description"),
        info_dict=raw_schema.get("info_dict"),
        comment=raw_schema.get("comment"),
    )

    if not description:
        description = f"Represents a {model_name} entity."

    model_additional_info = ""
    if other_info:
        try:
            model_additional_info = f" (Info: {json.dumps(other_info)})"
        except TypeError:
            model_additional_info = f" (Info: {str(other_info)})"

    final_model_description_base = f"{description}{model_additional_info}"
    final_model_overall_description = (
        f"{final_model_description_base.strip()} "
        f"When processing a {model_name}, the LLM should assign a unique '_temp_id' "
        f"to each instance and use '{model_name}' as its '_type' field in the output 'entities' list."
    )
    return final_model_overall_description


def generate_llm_schema_from_models(
    initial_model_classes: List[Type[SQLModel]],
    custom_field_descriptions: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    """
    Generates an LLM-friendly schema representation for a list of SQLAlchemy models.
    It starts with `initial_model_classes`, discovers all related SQLAlchemy models
    recursively, and includes them in the generated schema.
    The schema utilizes comments and info dictionaries from the models and allows
    for custom descriptions to override or augment default ones.

    Args:
        initial_model_classes: A list of SQLAlchemy model classes to serve as
                               starting points for schema generation.
        custom_field_descriptions: An optional dictionary to provide custom
                                   descriptions for models or their fields.
                                   Format: `{"ModelName": {"field_name": "desc", "_model_description": "model_desc"}}`

    Returns:
        A JSON string representing the LLM-friendly schema for all discovered models.
    """
    if custom_field_descriptions is None:
        custom_field_descriptions = {}

    all_sqla_models_to_document: List[Type[Any]] = []
    for root_model_class in initial_model_classes:
        _collect_all_sqla_models_recursively(
            root_model_class, all_sqla_models_to_document, set()
        )

    llm_schema_map = {}

    for model_class in all_sqla_models_to_document:
        model_name = model_class.__name__
        raw_schema = inspect_sqlalchemy_model(model_class)

        if raw_schema.get("error"):
            print(
                f"Warning: Could not inspect model {model_name} for LLM schema generation. Error: {raw_schema['error']}"
            )
            continue

        model_custom_descs = custom_field_descriptions.get(model_name, {})

        # Get pydantic model fields if applicable
        pydantic_model_fields = {}
        if hasattr(model_class, "model_fields") and issubclass(model_class, SQLModel):
            pydantic_model_fields = model_class.model_fields

        fields_info = {}
        for col_name, col_data in raw_schema.get("columns", {}).items():
            processed_col_name, formatted_col_string = _process_column_for_llm_schema(
                col_name,
                col_data,
                pydantic_model_fields,
                model_custom_descs,
                model_name,
            )
            fields_info[processed_col_name] = formatted_col_string

        for rel_name, rel_data in raw_schema.get("relationships", {}).items():
            processed_rel = _process_relationship_for_llm_schema(
                rel_name, rel_data, model_custom_descs
            )
            if processed_rel:
                field_name, formatted_string = processed_rel
                fields_info[field_name] = formatted_string

        final_model_overall_description = _generate_model_level_description(
            model_name, raw_schema, model_custom_descs
        )

        llm_schema_map[model_name] = {
            "description": final_model_overall_description,
            "fields": fields_info,
            "notes_for_llm": (
                f"For {model_name}: Ensure all fields conform to their types. "
                "Relationship fields (like '{rel_name}_ref_id' or '{rel_name}_ref_ids') "
                "must use the _temp_ids of corresponding related entities defined in this response. "
                "Omit optional fields if no information is found."
            ),
        }
    return json.dumps(llm_schema_map, indent=2)


def discover_sqlmodels_from_root(
    root_sqlmodel_class: Type[SQLModel],
) -> List[Type[SQLModel]]:
    """
    Discovers all unique SQLModel classes starting from a root SQLModel class,
    by recursively inspecting SQLAlchemy relationships, preserving discovery order.

    Args:
        root_sqlmodel_class: The primary SQLModel class to start discovery from.

    Returns:
        A list of all unique SQLModel classes discovered (including the root),
        in the order they were found. Returns an empty list if the
        root_sqlmodel_class is not a valid SQLModel or if no classes can be inspected.
    """
    if not root_sqlmodel_class or not issubclass(root_sqlmodel_class, SQLModel):
        print(f"Warning: {root_sqlmodel_class} is not a valid SQLModel class.")
        return []

    all_discovered_models: List[Type[SQLModel]] = []
    try:
        _collect_all_sqla_models_recursively(
            current_model_class=root_sqlmodel_class,
            all_discovered_models=all_discovered_models,  # type: ignore[arg-type]
            recursion_guard=set(),
        )
    except Exception as e:
        print(
            f"Error during SQLModel discovery starting from {root_sqlmodel_class.__name__}: {e}"
        )
        return []

    return all_discovered_models
