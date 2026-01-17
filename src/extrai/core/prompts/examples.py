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
