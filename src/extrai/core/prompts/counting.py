import json
from typing import Any


def generate_entity_counting_system_prompt(
    model_names: list[str],
    schema_json: str = None,
    custom_counting_context: str = "",
    previous_entities: list[dict[str, Any]] | None = None,
    examples: str = "",
    conflicting_revisions: list[dict[str, Any]] | None = None,
) -> str:
    """
    Generates a system prompt for counting entities in the provided documents.

    Args:
        model_names: A list of names of the models/entities to count.
        schema_json: A string containing the JSON schema for the models.
                     This helps the LLM understand the structure of the entities to count.
        custom_counting_context: Optional custom context to guide the counting phase.
        previous_entities: Optional list of previously extracted entities for context.
        examples: Optional string containing examples of the entities to count.
        conflicting_revisions: Optional list of previous conflicting counting attempts to merge.

    Returns:
        A string representing the system prompt for entity counting.
    """
    model_list_str = ", ".join(model_names)
    prompt = f"""
You are an expert data analyst. Your task is to analyze the provided documents and count the occurrences of specific entities.

You need to count the following entities: {model_list_str}.
"""

    if custom_counting_context:
        prompt += f"""
# CUSTOM CONTEXT:
{custom_counting_context}
"""

    if previous_entities:
        entities_json = json.dumps(previous_entities, indent=2)
        prompt += f"""
# PREVIOUSLY EXTRACTED ENTITIES:
{entities_json}

IMPORTANT: If the entities you are counting are related to any of the previously extracted entities above, you MUST specify the unique ID (or temp_id) of that related entity in your description string. This ensures correct linking in subsequent steps. Therefore take a good look at the previously extracted entities and ensure they are linked correctly.
Do not hesitate to add details to help identify those links. Also note that it's possible that there are no objects to extract!
"""

    prompt += f"""
# ENTITY DEFINITIONS:
To help you identify these entities correctly, here are their schema definitions:
```json
{schema_json}
```
"""

    if examples:
        prompt += f"""
# EXAMPLES:
Here are some examples of the objects that will be extracted on the next step. Your goal is to facilitate the extraction of these objects in the future:
{examples}
"""

    if conflicting_revisions:
        revisions_json = json.dumps(conflicting_revisions, indent=2)
        prompt += f"""
# MERGE REQUIRED:
Previous extraction attempts returned conflicting results. Here are the conflicting revisions:
{revisions_json}

Your task is to cross-reference these previous attempts with the text and provide the final, comprehensive, and correct list of entities, resolving any discrepancies.
"""

    prompt += """
# OUTPUT INSTRUCTIONS:
1.  **Output Format:** Your output must be a single JSON object with a `counted_entities` array.
2.  **Array Items:** Each item in the array must be an object containing:
    - `model`: the exact name of the entity model
    - `temp_id`: a unique temporary string identifier for this specific entity instance
    - `related_ids`: a list of string identifiers (temp_id or actual id) of any related entities
    - `description`: a detailed description of the entity found
3.  **Order:** The order of the entities in the list should generally match the order of appearance in the document.
4.  **Relational Detail:** If an entity relates to a previously extracted entity (e.g., a child entity belonging to a parent), you MUST include the ID of that parent entity in the `related_ids` list and optionally in the `description`.

Proceed with identifying and describing the entities in the user's documents.
""".strip()
    return prompt


def generate_entity_counting_user_prompt(documents: list[str]) -> str:
    """
    Generates a user prompt containing the documents for entity counting.

    Args:
        documents: A list of strings, where each string is a document.

    Returns:
        A string representing the user prompt.
    """
    separator = "\n\n---END OF DOCUMENT---\n\n---START OF NEW DOCUMENT---\n\n"
    combined_documents = separator.join(documents)

    prompt = f"""
Please count the entities in the following document(s) according to the instructions in the system prompt.

# DOCUMENT(S) TO ANALYZE:

{combined_documents}

---
Remember: Your output must match the structured format requested (an object with a `counted_entities` array).
""".strip()
    return prompt
