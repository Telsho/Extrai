def generate_system_prompt(
    schema_json: str,
    extraction_example_json: str = "",
    custom_extraction_process: str = "",
    custom_extraction_guidelines: str = "",
    custom_final_checklist: str = "",
    custom_context: str = "",
) -> str:
    """
    Generates a generic system prompt for guiding an LLM to extract information
    from text and structure it according to a provided JSON schema.

    Args:
        schema_json: A string containing the JSON schema for the target data structure.
        extraction_example_json: An optional string containing an example of a JSON
                                 object that conforms to the schema.
        custom_extraction_process: Optional custom instructions for the extraction process.
        custom_extraction_guidelines: Optional custom guidelines for extraction.
        custom_final_checklist: Optional custom final checklist for the LLM.
        custom_context: Optional custom contextual information to be included in the prompt.

    Returns:
        A string representing the system prompt.
    """

    default_extraction_process = """\
# EXTRACTION PROCESS
Follow this step-by-step process meticulously:
1.  **Understand the Goal:** Your primary objective is to extract information from the provided text and structure it precisely according to the JSON schema.
2.  **Full Text Analysis:** Read and comprehend the entirety of the provided document(s) before initiating extraction. This helps in understanding context and relationships.
3.  **Schema Adherence:** The provided JSON schema is your definitive guide. All extracted data must conform to this schema in terms of structure, field names, and data types.
4.  **Identify Relevant Data:** Locate all data points within the text that correspond to the fields defined in the JSON schema.
5.  **Map Data to Schema:** Carefully assign the identified data to the correct fields in the schema.
6.  **Handle Ambiguity and Missing Information:**
    * If information for a field is ambiguous, use your reasoning capabilities to determine the most plausible interpretation based on the context.
    * If information for an optional field is not present, omit the field or use `null` if the schema allows.
    * For required fields, if information is genuinely missing and cannot be inferred, this is a critical issue. However, strive to find or infer it. If the schema defines a default, consider that.
7.  **Prioritize Explicit Information:** Base your extraction on information explicitly stated in the text. Avoid making assumptions unless absolutely necessary and clearly justifiable by the context.
8.  **Synthesize from Multiple Documents:** If multiple documents are provided, synthesize the information comprehensively. If conflicting information arises, prioritize what appears to be the most current, official, or reliable source. Note any significant discrepancies if the output format allows, but the primary goal is a single coherent JSON.
9.  **Data Type Conformance:** Strictly adhere to the data types specified in the JSON schema (e.g., string, number, boolean, array, object). Numbers should be formatted as numbers (e.g., `123`, `12.34`), not strings containing numbers (e.g., `"123"`). Booleans should be `true` or `false`.
10. **Nested Structures and Relationships:**
    * For nested objects or arrays, ensure your JSON output accurately reflects the hierarchical structure defined in the schema.
    * If the schema implies relationships between different entities (e.g., using foreign keys or requiring linking), ensure these are correctly represented.
    * If temporary identifiers are needed to link entities within the JSON output (e.g., for items that will later become related records in a database), generate unique and descriptive temporary IDs (e.g., "_temp_id_entityName_XYZ123"). Use these temporary IDs consistently for all references within the current JSON output.
"""

    default_extraction_guidelines = """\
# IMPORTANT EXTRACTION GUIDELINES
- **Output Format:** Your entire output must be a single, valid JSON object. Do not include any explanatory text, comments, apologies, or any other content before or after the JSON object.
- **Output Structure Mandate:** Your response MUST be a single JSON object. This object MUST have a single top-level key named "result". The value of this "result" key MUST be the JSON object that conforms to the provided JSON schema. Example: `{"result": {your_schema_compliant_object_here}}`. Do NOT use any other top-level keys. Do NOT return the schema-compliant object directly as the root.
- **Field Names:** Use the exact field names (case-sensitive) as specified in the JSON schema for the object under the "result" key.
- **Structured Elements:** Pay close attention to structured elements within the text, such as tables, lists, headings, and emphasized text, as they often contain key information.
- **Dates and Times:** Unless the schema specifies a different format, use ISO 8601 format for dates (YYYY-MM-DD) and date-times (YYYY-MM-DDTHH:MM:SSZ).
- **Enumerations (Enums):** If a field in the schema is an enumeration with a predefined set of allowed values, ensure that the extracted value is one of those permitted values.
- **Null Values:** Use `null` for optional fields where data is not available or not applicable, provided the schema allows for null values for that field. Do not use strings like "N/A", "Not available", or empty strings "" unless the schema explicitly defines such string literals as valid values.
- **String Values:** Ensure all string values in the JSON are correctly escaped (e.g., quotes within strings).
- **Meticulousness:** Accuracy is paramount. Double-check your extracted data against the source text and the schema before finalizing your output.
"""

    default_final_checklist = """\
# FINAL CHECK BEFORE SUBMISSION
1.  **Valid JSON?** Is the entire output a single, syntactically correct JSON object?
2.  **Output Structure Correct?** Does the output JSON object have a single top-level key named "result"?
3.  **Schema Conformity?** Does the JSON object under the "result" key strictly adhere to all aspects of the provided JSON schema (all required fields present, correct data types for all values, correct structure for nested objects and arrays)?
4.  **Field Name Accuracy?** Are all field names within the object under the "result" key exactly as specified in the schema (case-sensitive)?
5.  **Relationship Integrity?** If temporary IDs or other linking mechanisms were required within the object under the "result" key, are they used correctly and consistently?
6.  **Null Handling?** Are `null` values used appropriately for missing optional data, according to schema constraints?
7.  **No Extraneous Text?** Is there absolutely no text or characters outside of the main JSON object itself?
"""

    # Use custom instructions if provided, otherwise use defaults
    extraction_process = custom_extraction_process or default_extraction_process
    extraction_guidelines = (
        custom_extraction_guidelines or default_extraction_guidelines
    )
    final_checklist = custom_final_checklist or default_final_checklist

    prompt_parts = [
        "You are an advanced AI specializing in data extraction and structuring. Your task is to analyze user-provided text and transform the relevant information into a structured JSON object, strictly adhering to the provided JSON schema.",
        "You must focus on precision, accuracy, and complete adherence to the schema.",
        "\n# JSON SCHEMA TO ADHERE TO:",
        "```json",
        schema_json,
        "```",
    ]

    if custom_context:  # New block
        prompt_parts.append("\n# ADDITIONAL CONTEXT:")
        prompt_parts.append(custom_context)

    prompt_parts.extend([f"\n{extraction_process}", f"\n{extraction_guidelines}"])

    if extraction_example_json:
        prompt_parts.append("\n# EXAMPLE OF EXTRACTION:")
        prompt_parts.append(
            "## CONCEPTUAL INPUT TEXT (This is illustrative; your actual input text will be different):"
        )
        prompt_parts.append(
            "\"Imagine a piece of text that contains details about an entity or event. For instance, if the schema is about a 'Book', the text might say: 'The Great Novel, written by Jane Author in 2023, has 300 pages and is published by World Publishers. ISBN: 978-0123456789.'\""
        )
        prompt_parts.append(
            "## EXAMPLE EXTRACTED JSON (This JSON conforms to the schema based on the conceptual text above):"
        )
        prompt_parts.append("```json")

        if extraction_example_json.strip().startswith(
            "{"
        ) and extraction_example_json.strip().endswith("}"):
            prompt_parts.append(f'{{\n  "result": {extraction_example_json}\n}}')
        else:
            prompt_parts.append(extraction_example_json)
        prompt_parts.append("```")

    prompt_parts.append(f"\n{final_checklist}")
    prompt_parts.append(
        "\nProceed with the extraction based on the user's documents. Your response MUST be only the single, valid JSON object. Do not include any other narrative, explanations, or conversational elements in your output."
    )

    return "\n\n".join(prompt_parts).strip()


def generate_user_prompt_for_docs(
    documents: list[str], custom_context: str = ""
) -> str:
    """
    Generates a simple user prompt containing the documents for extraction.

    Args:
        documents: A list of strings, where each string is a document
                   or a piece of text for extraction.
        custom_context: Optional custom contextual information to be included in the prompt.

    Returns:
        A string representing the user prompt with the documents.
    """
    separator = "\n\n---END OF DOCUMENT---\n\n---START OF NEW DOCUMENT---\n\n"
    combined_documents = separator.join(documents)

    prompt = """
Please extract information from the following document(s) strictly according to the schema and instructions previously provided (in the system prompt).
"""
    if custom_context:
        prompt += f"\n{custom_context}\n"

    prompt += f"""
# DOCUMENT(S) FOR EXTRACTION:

{combined_documents}

---
Remember: Your output must be only a single, valid JSON object.
""".strip()
    return prompt


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


def generate_prompt_for_example_json_generation(
    target_model_schema_str: str, root_model_name: str
) -> str:
    """
    Generates a system prompt for guiding an LLM to create a single, valid
    example JSON object based on a provided schema.

    Args:
        target_model_schema_str: A string containing the JSON schema for which
                                 an example is to be generated.
        root_model_name: The name of the root model/entity this schema represents
                         (e.g., "Product", "User"). Used for context in the prompt.

    Returns:
        A string representing the system prompt for example JSON generation.
    """
    prompt_parts = [
        "You are an AI assistant tasked with generating a sample JSON object.",
        f"The goal is to create a single, valid JSON object that conforms to the provided schema for a model named '{root_model_name}' and its related models.",
        "This sample will be used as a few-shot example for another LLM task, so it needs to be accurate and representative.",
        "\n# JSON SCHEMA TO ADHERE TO:",
        "```json",
        target_model_schema_str,
        "```",
        "\n# INSTRUCTIONS FOR YOUR OUTPUT:",
        "1.  **Output Content:** Your entire output MUST be a single, valid JSON object.",
        "2.  **Output Structure:** Your output MUST be a single JSON object with a top-level key named 'entities'. The value of 'entities' MUST be a list of JSON objects, where each object represents a single data entity.",
        "3.  **No Extra Text:** Do NOT include any explanatory text, comments, apologies, markdown formatting (like ```json), or any other content before or after the JSON object.",
        "4.  **Schema Compliance:** Strictly adhere to all field names (case-sensitive), data types (string, number, boolean, array, object), and structural requirements defined in the schema for each entity in the 'entities' list.",
        "5.  **Entity Metadata:** Each object inside the 'entities' list MUST include two metadata fields:",
        '    *   `_type`: This field\'s value MUST be a string matching the name of the model it represents (e.g., "Product", "ProductSpecs").',
        '    *   `_temp_id`: This field\'s value MUST be a unique temporary string identifier for that specific entity instance (e.g., "product_example_001", "spec_example_001"). Use these IDs in the `_ref_id` or `_ref_ids` fields to link entities.',
        "6.  **Simplicity and Clarity:** The generated example should be simple and illustrative. Populate all other fields (defined in the schema) with plausible, concise, and representative data. Avoid overly complex or lengthy values unless the schema demands it.",
        f"7.  **Completeness and Relationships:** Your 'entities' list should contain an instance of the root model (`{root_model_name}`) and at least one instance of each of its related models as described in the schema. For example, if generating an example for a 'Product' that has 'ProductSpecs', the 'entities' list should contain at least one 'Product' object and one 'ProductSpecs' object, linked together using their `_temp_id`s in the appropriate `_ref_id` or `_ref_ids` field.",
        f"\nConsider the schema for '{root_model_name}' and its related models. Generate a representative set of linked entities in the format `{{\"entities\": [...]}}`.",
        "Proceed with generating the JSON object.",
    ]
    return "\n\n".join(prompt_parts).strip()
