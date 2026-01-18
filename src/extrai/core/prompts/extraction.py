import json
from typing import Optional, List, Dict, Any


def generate_system_prompt(
    schema_json: str,
    extraction_example_json: str = "",
    custom_extraction_process: str = "",
    custom_extraction_guidelines: str = "",
    custom_final_checklist: str = "",
    custom_context: str = "",
    expected_entity_descriptions: Optional[List[str]] = None,
    previous_entities: Optional[List[Dict[str, Any]]] = None,
    target_model_name: Optional[str] = None,
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
        expected_entity_descriptions: Optional list of descriptions for the entities to be extracted.
        previous_entities: List of previously extracted entities for hierarchical linking.
        target_model_name: Name of the specific model to extract (for hierarchical steps).

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
    * If temporary identifiers are needed to link entities within the JSON output, generate unique and descriptive temporary IDs based on the entity's key attributes.
11. **ID and Temporary ID Generation Directives:**
    *   **Explicit IDs:** If the text contains an explicit identifier for an entity (e.g., "ID: 123", "Code: A-55"), use it for the `id` field if the schema has one.
    *   **Temporary IDs:** When generating temporary IDs for linking entities (e.g., for `temp_id`, `_id`, or foreign keys):
        *   **Format:** You MUST use the format `[entity_type]_[key_attribute]` in `snake_case`. E.g., `user_john_doe`, `order_12345`.
        *   **Determinism:** Do NOT use random strings (like UUIDs) or simple counters (like `item_1`) unless there is absolutely no distinguishing attribute. Random values make consistency checking impossible.
        *   **Sanitization:** Convert to lowercase and replace spaces/special characters with underscores.
        *   **Consistency:** If the same entity appears multiple times, it MUST have the identical temporary ID every time.
"""

    default_extraction_guidelines = """\
# IMPORTANT EXTRACTION GUIDELINES
- **Ordering:** Maintain the order of items as they appear in the source text when populating arrays.
- **Output Format:** Your entire output must be a single, valid JSON object. Do not include any other explanatory text, comments, apologies, or any other content before or after the JSON object.
- **Output Structure Mandate:** Your response MUST be a single JSON object. This object MUST have a single top-level key named "result". The value of this "result" key MUST be the JSON object that conforms to the provided JSON schema. Example: `{"result": {your_schema_compliant_object_here}}`. Do NOT use any other top-level keys. Do NOT return the schema-compliant object directly as the root.
- **Field Names:** Use the exact field names (case-sensitive) as specified in the JSON schema for the object under the "result" key.
- **Structured Elements:** Pay close attention to structured elements within the text, such as tables, lists, headings, and emphasized text, as they often contain key information.
- **Dates and Times:** Unless the schema specifies a different format, use ISO 8601 format for dates (YYYY-MM-DD) and date-times (YYYY-MM-DDTHH:MM:SSZ).
- **Enumerations (Enums):** If a field in the schema is an enumeration with a predefined set of allowed values, ensure that the extracted value is one of those permitted values.
- **Null Values:** Use `null` for optional fields where data is not available or not applicable, provided the schema allows for null values for that field. Do not use strings like "N/A", "Not available", or empty strings "" unless the schema explicitly defines such string literals as valid values.
- **String Values:** Ensure all string values in the JSON are correctly escaped (e.g., quotes within strings).
- **Foreign Key Fields:** If a model has a required foreign key field (e.g., `object_id`) and you are establishing the relationship using a temporary ID field (e.g., `airline_ref_id`), you MUST provide a placeholder value (e.g., `0`) for the foreign key field if the schema requires it. This ensures the JSON remains valid against the schema constraints.
- **ID Consistency:** Ensure that `id` and `temp_id` values are consistent throughout the JSON. If you refer to `user_john_doe` in one place, do not refer to them as `user_john` elsewhere. Avoid generating random UUIDs or hashes for IDs unless explicitly instructed. Prefer human-readable, content-derived IDs for temporary linking.
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

    if target_model_name:
        prompt_parts.append("\n# TARGET ENTITY")
        prompt_parts.append(
            f"Your task is to extract **only** entities of type '{target_model_name}'. "
            "Do not extract other entity types in this step."
        )

    if expected_entity_descriptions:
        prompt_parts.append("\n# EXPECTED ENTITIES & ORDER:")
        prompt_parts.append(
            "You MUST extract entities matching the following descriptions, in this exact order:"
        )
        for i, desc in enumerate(expected_entity_descriptions, 1):
            prompt_parts.append(f"{i}. {desc}")
        prompt_parts.append(
            f"\nYou must extract EXACTLY {len(expected_entity_descriptions)} items/entities corresponding to these descriptions."
        )

    if custom_context:
        prompt_parts.append("\n# ADDITIONAL CONTEXT:")
        prompt_parts.append(custom_context)

    if previous_entities:
        entities_json = json.dumps(previous_entities, indent=2)
        prompt_parts.append("\n# PREVIOUSLY EXTRACTED ENTITIES:")
        prompt_parts.append(entities_json)
        prompt_parts.append(
            "\nIMPORTANT: Use the 'id' values from the entities above to populate foreign key fields "
            "(e.g. 'recipe_id') in the new entities you extract. Ensure correct linking."
        )

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
