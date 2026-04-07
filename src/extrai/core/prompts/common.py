def generate_user_prompt_for_docs(
    documents: list[str], custom_context: str = ""
) -> str:
    """
    Generates a generic user prompt containing the documents for extraction.
    Used by both standard and structured extraction flows.

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
Please extract information from the following document(s).
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
