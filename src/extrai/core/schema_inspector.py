import json
import logging
import enum
from typing import Type, List, Optional, Any, Dict, Set, Tuple
from sqlalchemy import inspect, Column, Table
from sqlalchemy.orm import RelationshipProperty
from sqlalchemy.exc import NoInspectionAvailable
from sqlalchemy.schema import UniqueConstraint, PrimaryKeyConstraint
from sqlmodel import SQLModel

from extrai.utils.type_mapping import (
    map_sql_type_to_llm_type,
    get_python_type_str_from_pydantic_annotation,
)


class SchemaInspector:
    """Helper class to inspect SQLAlchemy models and generate LLM schemas."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def _is_column_unique(self, column_obj: Column) -> bool:
        """Checks if a column has a unique constraint."""
        if column_obj.unique:
            return True
        if column_obj.table is not None:
            for constraint in column_obj.table.constraints:
                if isinstance(constraint, (UniqueConstraint, PrimaryKeyConstraint)):
                    if column_obj.name in constraint.columns:
                        return True
        return False

    def _get_python_type_from_column(self, column_obj: Column) -> str:
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
        self, column_obj: Column, is_unique: bool, python_type_name: str
    ) -> Dict[str, Any]:
        """Builds the column information dictionary."""
        enum_values = None
        # Handle SQLAlchemy Enum types (both class-based and string-based)
        if hasattr(column_obj.type, "enum_class") and column_obj.type.enum_class:
            if isinstance(column_obj.type.enum_class, type) and issubclass(
                column_obj.type.enum_class, enum.Enum
            ):
                enum_values = [e.value for e in column_obj.type.enum_class]
        elif hasattr(column_obj.type, "enums") and column_obj.type.enums:
            enum_values = list(column_obj.type.enums)

        col_info = {
            "type": str(column_obj.type),
            "python_type": python_type_name,
            "primary_key": column_obj.primary_key,
            "nullable": column_obj.nullable,
            "unique": is_unique,
            "foreign_key_to": None,
            "comment": column_obj.comment,
            "info_dict": column_obj.info,
            "enum_values": enum_values,
        }
        if column_obj.foreign_keys:
            fk_constraint_obj = next(iter(column_obj.foreign_keys))
            col_info["foreign_key_to"] = str(fk_constraint_obj.column)
        return col_info

    def _get_columns_from_inspector(self, inspector) -> Dict[str, Any]:
        """Extracts all column properties from a SQLAlchemy inspector."""
        columns_info = {}
        for col_attr in inspector.column_attrs:
            if not isinstance(col_attr.expression, Column):
                continue
            column_obj = col_attr.expression
            is_unique = self._is_column_unique(column_obj)
            python_type_name = self._get_python_type_from_column(column_obj)
            columns_info[col_attr.key] = self._build_column_info(
                column_obj, is_unique, python_type_name
            )
        return columns_info

    def _get_fks_from_secondary_table(self, rel_prop: RelationshipProperty) -> Set[str]:
        """Handles relationships that use a secondary table."""
        involved_fk_columns: Set[str] = set()
        if rel_prop.secondary is not None:
            for fk_constraint in rel_prop.secondary.foreign_key_constraints:
                for col in fk_constraint.columns:
                    involved_fk_columns.add(str(col))
        return involved_fk_columns

    def _get_fks_from_synchronize_pairs(self, rel_prop: RelationshipProperty) -> Set[str]:
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

    def _get_fks_from_direct_foreign_keys(self, rel_prop: RelationshipProperty) -> Set[str]:
        """Handles relationships that have direct foreign_keys."""
        involved_fk_columns: Set[str] = set()
        if hasattr(rel_prop, "foreign_keys") and rel_prop.foreign_keys is not None:
            for fk_col in rel_prop.foreign_keys:
                involved_fk_columns.add(str(fk_col))
        return involved_fk_columns

    def _get_involved_foreign_keys(self, rel_prop: RelationshipProperty) -> Set[str]:
        """
        Finds all foreign key columns involved in a relationship by dispatching to helper functions.
        """
        if rel_prop.secondary is not None:
            return self._get_fks_from_secondary_table(rel_prop)

        if rel_prop.synchronize_pairs:
            return self._get_fks_from_synchronize_pairs(rel_prop)

        if hasattr(rel_prop, "foreign_keys") and rel_prop.foreign_keys is not None:
            return self._get_fks_from_direct_foreign_keys(rel_prop)

        return set()

    def _build_relationship_info(
        self,
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
            "nested_schema": self._inspect_sqlalchemy_model_recursive(
                related_model_class, recursion_path_tracker
            ),
        }

    def _get_relationships_from_inspector(
        self, inspector, recursion_path_tracker: Set[Type[Any]]
    ) -> Dict[str, Any]:
        """Extracts all relationship properties from a SQLAlchemy inspector."""
        relationships_info = {}
        for name, rel_prop in inspector.relationships.items():
            if isinstance(rel_prop, RelationshipProperty):
                involved_fk_columns = self._get_involved_foreign_keys(rel_prop)
                relationships_info[name] = self._build_relationship_info(
                    rel_prop, involved_fk_columns, recursion_path_tracker
                )
        return relationships_info

    def _inspect_sqlalchemy_model_recursive(
        self, model_class: Type[Any], recursion_path_tracker: Set[Type[Any]]
    ) -> Dict[str, Any]:
        """
        Internal recursive function to introspect a SQLAlchemy model class.
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
            "columns": self._get_columns_from_inspector(inspector),
            "relationships": self._get_relationships_from_inspector(
                inspector, recursion_path_tracker
            ),
        }

        recursion_path_tracker.remove(model_class)
        return schema_info

    def inspect_sqlalchemy_model(self, model_class: Type[Any]) -> Dict[str, Any]:
        """
        Public wrapper function to start the SQLAlchemy model introspection.
        """
        return self._inspect_sqlalchemy_model_recursive(model_class, set())

    def _collect_all_sqla_models_recursively(
        self,
        current_model_class: Type[Any],
        all_discovered_models: List[Type[Any]],
        recursion_guard: Set[Type[Any]],
    ) -> None:
        """
        Recursively collects all unique SQLAlchemy model classes related to current_model_class.
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
                self._collect_all_sqla_models_recursively(
                    related_sqla_model_class, all_discovered_models, recursion_guard
                )
        recursion_guard.remove(current_model_class)

    def _get_prioritized_description(
        self,
        *,
        custom_desc: Optional[str] = None,
        pydantic_desc: Optional[str] = None,
        info_dict: Optional[Dict[str, Any]] = None,
        comment: Optional[str] = None,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Centralized helper to determine the best description from multiple sources.
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
        self,
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
                pydantic_derived_type_str = get_python_type_str_from_pydantic_annotation(
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

        llm_type = map_sql_type_to_llm_type(
            str(col_data.get("type", "")),
            python_type_for_mapping,
        )

        description, other_info = self._get_prioritized_description(
            custom_desc=custom_descs.get(col_name),
            pydantic_desc=pydantic_field_description,
            info_dict=col_data.get("info_dict"),
            comment=col_data.get("comment"),
        )

        if not description:
            description = f"Field '{col_name}' of type {llm_type} for {model_name}."

        if col_data.get("enum_values"):
            description += (
                f" Authorized values: {', '.join(map(str, col_data['enum_values']))}."
            )

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
        self,
        rel_name: str,
        rel_data: Dict[str, Any],
        custom_descs: Dict[str, str]
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

        description, other_info = self._get_prioritized_description(
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
        self,
        model_name: str, raw_schema: Dict[str, Any], custom_descs: Dict[str, str]
    ) -> str:
        """Generates the complete model-level description block."""
        description, other_info = self._get_prioritized_description(
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
        self,
        initial_model_classes: List[Type[SQLModel]],
        custom_field_descriptions: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> str:
        """
        Generates an LLM-friendly schema representation for a list of SQLAlchemy models.
        """
        if custom_field_descriptions is None:
            custom_field_descriptions = {}

        all_sqla_models_to_document: List[Type[Any]] = []
        for root_model_class in initial_model_classes:
            self._collect_all_sqla_models_recursively(
                root_model_class, all_sqla_models_to_document, set()
            )

        llm_schema_map = {}

        for model_class in all_sqla_models_to_document:
            model_name = model_class.__name__
            raw_schema = self.inspect_sqlalchemy_model(model_class)

            if raw_schema.get("error"):
                self.logger.warning(
                    f"Could not inspect model {model_name} for LLM schema generation. Error: {raw_schema['error']}"
                )
                continue

            model_custom_descs = custom_field_descriptions.get(model_name, {})

            # Get pydantic model fields if applicable
            pydantic_model_fields = {}
            if hasattr(model_class, "model_fields") and issubclass(model_class, SQLModel):
                pydantic_model_fields = model_class.model_fields

            fields_info = {}
            for col_name, col_data in raw_schema.get("columns", {}).items():
                processed_col_name, formatted_col_string = self._process_column_for_llm_schema(
                    col_name,
                    col_data,
                    pydantic_model_fields,
                    model_custom_descs,
                    model_name,
                )
                fields_info[processed_col_name] = formatted_col_string

            for rel_name, rel_data in raw_schema.get("relationships", {}).items():
                processed_rel = self._process_relationship_for_llm_schema(
                    rel_name, rel_data, model_custom_descs
                )
                if processed_rel:
                    field_name, formatted_string = processed_rel
                    fields_info[field_name] = formatted_string

            final_model_overall_description = self._generate_model_level_description(
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
        self,
        root_sqlmodel_class: Type[SQLModel],
    ) -> List[Type[SQLModel]]:
        """
        Discovers all unique SQLModel classes starting from a root SQLModel class.
        """
        if not root_sqlmodel_class or not issubclass(root_sqlmodel_class, SQLModel):
            self.logger.warning(f"{root_sqlmodel_class} is not a valid SQLModel class.")
            return []

        all_discovered_models: List[Type[SQLModel]] = []
        try:
            self._collect_all_sqla_models_recursively(
                current_model_class=root_sqlmodel_class,
                all_discovered_models=all_discovered_models,  # type: ignore[arg-type]
                recursion_guard=set(),
            )
        except Exception as e:
            self.logger.error(
                f"Error during SQLModel discovery starting from {root_sqlmodel_class.__name__}: {e}"
            )
            return []

        return all_discovered_models
