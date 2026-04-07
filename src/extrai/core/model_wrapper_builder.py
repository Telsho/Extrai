from typing import Any, Optional

from pydantic import BaseModel, Field, create_model
from sqlalchemy import inspect
from sqlalchemy.orm import RelationshipProperty
from sqlmodel import SQLModel


class ModelWrapperBuilder:
    """
    Utility class to convert SQLModel classes (with Relationship fields)
    into pure Pydantic models suitable for structured LLM output (e.g. OpenAI).
    Replaces Relationships with nested models.
    """

    def __init__(self):
        self._generated_models: dict[type[SQLModel], type[BaseModel]] = {}

    def generate_wrapper_model(
        self, root_sqlmodel: type[SQLModel], include_relationships: bool = True
    ) -> type[BaseModel]:
        """
        Generates a Pydantic wrapper model for the given root SQLModel.
        This wrapper creates a hierarchy of Pydantic models where relationships
        are replaced by nested lists or single instances of the related Pydantic model.

        It also wraps the result in a container to ensure we capture a list of the root entities.

        Args:
            root_sqlmodel: The root SQLModel class.
            include_relationships: If False, relationships will be excluded from the schema.
                                   Useful for hierarchical extraction steps.
        """
        self._generated_models = {}

        pydantic_model = self._create_pydantic_model_recursive(
            root_sqlmodel, include_relationships
        )

        # Let's define a wrapper that has a field `entities` which is a list of the root model.
        wrapper_name = f"{root_sqlmodel.__name__}ExtractionResult"

        wrapper_model = create_model(
            wrapper_name,
            entities=(
                list[pydantic_model],
                Field(
                    description=f"List of extracted {root_sqlmodel.__name__} entities."
                ),
            ),
        )

        return wrapper_model

    def _enrich_field_description(self, field_info: Any) -> Any:
        """
        Appends validation constraints to the field description to help the LLM.
        """
        import copy

        new_field_info = copy.copy(field_info)

        # Using .metadata to access constraints from Pydantic v2
        constraints = []
        if hasattr(new_field_info, "metadata"):
            for item in new_field_info.metadata:
                if hasattr(item, "max_length"):
                    constraints.append(f"max_length={item.max_length}")
                if hasattr(item, "min_length"):
                    constraints.append(f"min_length={item.min_length}")
                # Looking for numeric constraints
                if hasattr(item, "ge"):
                    constraints.append(f"min_value={item.ge}")
                if hasattr(item, "le"):
                    constraints.append(f"max_value={item.le}")
                if hasattr(item, "gt"):
                    constraints.append(f"greater_than={item.gt}")
                if hasattr(item, "lt"):
                    constraints.append(f"less_than={item.lt}")

        # Fallback for older Pydantic or direct attributes if metadata is not used
        if (
            hasattr(new_field_info, "max_length")
            and new_field_info.max_length is not None
        ):
            if f"max_length={new_field_info.max_length}" not in constraints:
                constraints.append(f"max_length={new_field_info.max_length}")

        if constraints:
            constraint_str = "Constraints: " + ", ".join(constraints)
            if new_field_info.description:
                new_field_info.description = (
                    f"{new_field_info.description} ({constraint_str})"
                )
            else:
                new_field_info.description = constraint_str

        return new_field_info

    def _create_pydantic_model_recursive(
        self, sql_model: type[SQLModel], include_relationships: bool = True
    ) -> type[BaseModel]:
        if sql_model in self._generated_models:
            return self._generated_models[sql_model]

        model_name = f"{sql_model.__name__}Structure"

        fields = {}
        inspector = inspect(sql_model)

        for name, field_info in sql_model.model_fields.items():
            # Check if it is a relationship field, if so, we will handle it later
            if name in inspector.relationships:
                continue

            # Enrich description with constraints
            enriched_field_info = self._enrich_field_description(field_info)
            fields[name] = (field_info.annotation, enriched_field_info)

        relationships = {}
        if include_relationships:
            inspector = inspect(sql_model)

            for rel in inspector.relationships:
                if isinstance(rel, RelationshipProperty):
                    target_model = rel.mapper.class_

                    if rel.direction.name == "MANYTOONE":
                        # Skip child->parent links to enforce hierarchy
                        continue

                    # Recurse
                    nested_model = self._create_pydantic_model_recursive(
                        target_model, include_relationships
                    )

                    if rel.uselist:
                        # List[NestedModel]
                        field_type = list[nested_model]
                        field_desc = f"List of {target_model.__name__} items."
                    else:
                        # NestedModel (Optional?)
                        field_type = Optional[nested_model]
                        field_desc = f"Related {target_model.__name__} item."

                    relationships[rel.key] = (
                        field_type,
                        Field(default=None, description=field_desc),
                    )

        # Merge fields
        all_fields = {**fields, **relationships}

        # Create the model
        model = create_model(model_name, **all_fields)

        self._generated_models[sql_model] = model
        return model
