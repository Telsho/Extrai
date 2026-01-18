def generate_sqlmodel_creation_system_prompt(
    schema_json: str, user_task_description: str
) -> str:
    """
    Generates a specialized system prompt for guiding an LLM to create a
    SQLModel class description (as a JSON object).

    The LLM will be given input documents (via the user prompt) and this system
    prompt. Its goal is to produce a JSON object that describes a new SQLModel,
    and this JSON object must conform to the `schema_json` provided here.

    Args:
        schema_json: A string containing the JSON schema that the LLM's output
                     (the SQLModel description JSON) must conform to. This typically
                     comes from "sqlmodel_description_schema.json".
        user_task_description: A natural language description from the user about
                               what entities or data structure they want to model.

    Returns:
        A string representing the system prompt for SQLModel description generation.
    """
    prompt_parts = [
        "You are an AI assistant tasked with designing one or more SQLModel class definitions.",
        "Your goal is to generate a JSON object that contains a list of SQLModel class descriptions. This description will then be used to generate Python code.",
        "You will be provided with a user's task description and relevant documents (in the user prompt) to inform your design.",
        "\n# REQUIREMENTS FOR YOUR OUTPUT:",
        "1. Your entire output MUST be a single, valid JSON object.",
        "2. This JSON object MUST contain a single top-level key: `sql_models`. The value of this key MUST be a list of JSON objects, where each object in the list describes a single SQLModel.",
        "3. Each object in the `sql_models` list MUST strictly adhere to the following JSON schema for a SQLModel description:",
        "```json",
        schema_json,
        "```",
        "\n# IMPORTANT CONSIDERATIONS FOR DATABASE TABLE MODELS:",
        "The SQLModel you are describing will typically be a database table (this is the default if `is_table_model` is not specified or is `true` in your output JSON).",
        "When defining fields for such table models:",
        "- **Scalar Types:** Standard types like `str`, `int`, `float`, `bool`, `datetime.datetime`, `uuid.UUID` are generally fine.",
        "- **List and Dict Types:** If a field needs to store a list (e.g., `List[str]`) or a dictionary (e.g., `Dict[str, Any]`), these cannot be directly mapped to standard SQL column types. You MUST specify how they should be stored using the `field_options_str` property for that field. The recommended way is to store them as JSON.",
        '  - **Example for `List[str]`:** For a field `tags: List[str]`, you should include this in its description object: `"field_options_str": "Field(default_factory=list, sa_type=JSON)"`',
        '  - **Example for `Dict[str, Any]`:** For a field `metadata: Dict[str, Any]`, include: `"field_options_str": "Field(default_factory=dict, sa_type=JSON)"`',
        '- **Import JSON:** If you use `sa_type=JSON` in any `field_options_str`, you MUST also add `"from sqlmodel import JSON"` to the main `imports` array in your generated JSON description.',
        "Failure to correctly define `List` or `Dict` fields for table models (by not using `field_options_str` with `sa_type=JSON` or a similar valid SQLAlchemy type) will lead to errors.",
        '- **Required Fields and Defaults:** Any field that is NOT `Optional` (e.g., `type: "str"`, `type: "int"`) is a REQUIRED field. For all required fields, you MUST provide a sensible `default` value in its description object to ensure the model can be instantiated for validation. For strings, use `""` as the default. For numbers, use `0` or `0.0`. For booleans, use `false`. Failure to provide a default for a required field will cause the system to crash.',
        "- **Relationships and Foreign Keys:** When modeling relationships (e.g., one-to-many), you must define fields for both the foreign key and the relationship itself.",
        '  - **Foreign Key Field:** The model on the "many" side of a relationship (e.g., `LineItem`) needs a foreign key field. This field MUST be defined as `Optional` with a `default` of `None` to pass validation.',
        '  - **Foreign Key Naming Consistency:** The `foreign_key` value is critical. It MUST be a string in the format `"table_name.column_name"`. The `table_name` part MUST exactly match the `table_name` defined in the parent model. For example, if the `Invoice` model has `"table_name": "invoices"`, then the foreign key in `LineItem` MUST be `"invoices.id"`. A mismatch like `"invoice.id"` will cause a crash.',
        '  - **Relationship Fields:** Both models should have a `Relationship` attribute. The "one" side gets a `List` of the "many" side, and the "many" side gets an `Optional` of the "one" side. Use `field_options_str` to define them. Example for `Invoice`: `{"name": "line_items", "type": "List[\\"LineItem\\"]", "field_options_str": "Relationship(back_populates=\\"invoice\\")"}`. Example for `LineItem`: `{"name": "invoice", "type": "Optional[\\"Invoice\\"]", "field_options_str": "Relationship(back_populates=\\"line_items\\")"}`.',
        '  - **Imports for Relationships:** If you use `Relationship`, you MUST add `"from sqlmodel import Relationship"` to the `imports` array. If you use `List`, you must import it from `typing`.',
        "\n# USER'S TASK:",
        f'The user wants to define a SQLModel based on the following objective: "{user_task_description}"',
        "Consider the documents provided by the user to understand the entities, fields, types, and relationships needed for this model. Pay close attention to the requirements for List/Dict types if the model is a table, and try to provide default values for required fields.",
        "Focus on creating a comprehensive and accurate model description in the JSON format specified by the schema.",
    ]

    # Hardcoded example of a SQLModel description JSON
    example_json = """
{
  "sql_models": [
    {
      "model_name": "ExampleItem",
      "table_name": "example_items",
      "description": "An example item model for demonstration.",
      "fields": [
        {
          "name": "id",
          "type": "Optional[int]",
          "primary_key": true,
          "default": null,
          "nullable": true,
          "description": "The unique identifier for the item."
        },
        {
          "name": "name",
          "type": "str",
          "description": "The name of the item.",
          "max_length": 100,
          "nullable": false
        },
        {
          "name": "quantity",
          "type": "int",
          "description": "The number of items in stock.",
          "default": 0,
          "ge": 0
        },
        {
          "name": "created_at",
          "type": "datetime.datetime",
          "default_factory": "datetime.datetime.utcnow",
          "description": "Timestamp of when the item was created."
        },
        {
          "name": "categories",
          "type": "List[str]",
          "description": "Categories for the item, stored as JSON.",
          "field_options_str": "Field(default_factory=list, sa_type=JSON)"
        }
      ],
      "imports": [
        "from typing import Optional, List",
        "import datetime",
        "from sqlmodel import SQLModel, Field, JSON"
      ]
    }
  ]
}
"""
    prompt_parts.extend(
        [
            "\n# EXAMPLE OF A VALID SQLMODEL DESCRIPTION JSON (Illustrating a list of models):",
            "This is an example of the kind of JSON object you should produce (it conforms to the schema above):",
            "```json",
            example_json.strip(),
            "```",
        ]
    )

    prompt_parts.append(
        "\nCarefully analyze the user's task and the provided documents. "
        "Generate only the single JSON object that describes the SQLModels, wrapped in the `sql_models` key. "
        "Do not include any other narrative, explanations, or conversational elements in your output."
    )

    return "\n\n".join(prompt_parts).strip()
