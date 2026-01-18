import json
from typing import List, Dict, Any, Optional

def generate_entity_counting_system_prompt(
    model_names: list[str], 
    schema_json: str = None,
    custom_counting_context: str = "",
    previous_entities: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Generates a system prompt for counting entities in the provided documents.

    Args:
        model_names: A list of names of the models/entities to count.
        schema_json: A string containing the JSON schema for the models.
                     This helps the LLM understand the structure of the entities to count.
        custom_counting_context: Optional custom context to guide the counting phase.
        previous_entities: Optional list of previously extracted entities for context.

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

IMPORTANT: If the entities you are counting are related to any of the previously extracted entities above, you MUST specify the unique ID (or temp_id) of that related entity in your description string. This ensures correct linking in subsequent steps.
"""

    prompt += f"""
# ENTITY DEFINITIONS:
To help you identify these entities correctly, here are their schema definitions:
```json
{schema_json}
```
"""

    prompt += """
# OUTPUT INSTRUCTIONS:
1.  **Output Format:** Your output must be a single, valid JSON object.
2.  **Keys:** The JSON object keys must be the exact names of the entities provided above.
3.  **Values:** The values must be a list of strings, where each string is a description of the entity found.
4.  **Order:** The order of the descriptions in the list must match the order of appearance in the document.
5.  **Relational Detail:** If an entity relates to a previously extracted entity (e.g., a child entity belonging to a parent), your description MUST include the ID of that parent entity from the provided context.
6.  **No Extra Text:** Do NOT include any explanations, markdown formatting, or text outside the JSON object.

Example Output:
{{
  "Invoice": [
      "Invoice #123 from ABC Corp with a value of 50euros",
      "Invoice #456 from XYZ Inc with a value of 506euros",
      "Invoice #789 from Foo Bar with a value of 30euros"
  ],
  "LineItem": [
      "Item A - Widget linked to Invoice ID: invoice_123",
      "Item B - Gadget linked to Invoice ID: invoice_123",
      "Item C - Doohickey linked to Invoice ID: invoice_456",
  ]
}}

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
Remember: Your output must be only a single, valid JSON object mapping entity names to counts.
""".strip()
    return prompt
