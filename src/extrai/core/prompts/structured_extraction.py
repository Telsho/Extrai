import json
from typing import Any


def generate_structured_system_prompt(
    custom_extraction_process: str = "",
    custom_extraction_guidelines: str = "",
    custom_context: str = "",
    extraction_example_json: str = "",
    expected_entity_descriptions: list[dict] | None = None,
    previous_entities: list[dict[str, Any]] | None = None,
    target_model_name: str | None = None,
) -> str:
    """
    Generates a system prompt tailored for structured output extraction.
    Simplified instructions as the structure is enforced by the API.

    Args:
        custom_extraction_process: Optional custom instructions for the extraction process.
        custom_extraction_guidelines: Optional custom guidelines for extraction.
        custom_context: Optional custom contextual information.
        extraction_example_json: Optional example JSON string.
        expected_entity_descriptions: Optional list of descriptions for the entities to be extracted.
        previous_entities: List of previously extracted entities for hierarchical linking.
        target_model_name: Name of the specific model to extract (for hierarchical steps).

    Returns:
        A string representing the system prompt.
    """

    default_instructions = """\
# EXTRACTION INSTRUCTIONS
You are an expert data extraction AI. Your goal is to extract structured data from the provided text.

1.  **Analyze the Text:** Read the provided documents carefully.
2.  **Extract Entities:** Identify all entities that match the requested structure.
3.  **Accuracy:** Ensure all extracted data is accurate and supported by the text.
4.  **Inference:** If a field is missing but can be reasonably inferred from context, you may do so. Otherwise, leave it as null/None.
5.  **Relationships:** Capture relationships by nesting entities as defined in the structure.
6.  **IDs:** If the schema contains an `id` field, you MUST populate it. If the `id` is an integer, start counting from 1. If it is a string, create a deterministic ID based on the content.
"""

    parts = [default_instructions]

    if target_model_name:
        parts.append("# TARGET ENTITY")
        parts.append(
            f"Your task is to extract **only** entities of type '{target_model_name}'. "
            "Do not extract other entity types in this step."
        )

    if expected_entity_descriptions is not None:
        parts.append("# EXPECTED ENTITIES & ORDER")
        if len(expected_entity_descriptions) == 0:
            parts.append(
                "Based on the counting phase, there are NO entities of this type to extract. "
                "You MUST return an empty array/list. Extract exactly 0 entities."
            )
        else:
            parts.append(
                "You MUST extract entities matching the following descriptions, in this exact order:"
            )
            for i, entity_dict in enumerate(expected_entity_descriptions, 1):
                model = entity_dict.get("model", "Unknown")
                desc = entity_dict.get("description", "")
                related_ids = entity_dict.get("related_ids", [])
                related_str = (
                    f" | Related IDs: {', '.join(related_ids)}" if related_ids else ""
                )
                parts.append(f"{i}. [Model: {model}] Description: {desc}{related_str}")

            parts.append(
                f"\nYou must extract EXACTLY {len(expected_entity_descriptions)} items/entities corresponding to these descriptions."
            )

    # Assemble comprehensive custom instructions
    instructions_parts = []
    if custom_extraction_process:
        instructions_parts.append(custom_extraction_process)

    if custom_context:
        instructions_parts.append(f"CONTEXT:\n{custom_context}")

    if previous_entities:
        entities_json = json.dumps(previous_entities, indent=2)
        instructions_parts.append(
            f"PREVIOUSLY EXTRACTED ENTITIES:\n{entities_json}\n\n"
            "IMPORTANT: Use the 'id' values from the entities above to populate foreign key fields "
            "(e.g. 'recipe_id') in the new entities you extract. Ensure correct linking."
        )

    if custom_extraction_guidelines:
        instructions_parts.append(f"GUIDELINES:\n{custom_extraction_guidelines}")

    if extraction_example_json:
        instructions_parts.append(f"EXAMPLE REFERENCE:\n{extraction_example_json}")

    if instructions_parts:
        parts.append("# CUSTOM INSTRUCTIONS")
        parts.append("\n\n".join(instructions_parts))

    return "\n\n".join(parts)
