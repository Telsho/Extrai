from extrai.core.prompts.structured_extraction import (
    generate_structured_system_prompt,
)
from extrai.core.prompts.common import generate_user_prompt_for_docs


def test_generate_structured_system_prompt():
    prompt = generate_structured_system_prompt()
    assert "# EXTRACTION INSTRUCTIONS" in prompt
    assert "Extract Entities" in prompt
    
    custom = "Pay attention to dates."
    prompt_custom = generate_structured_system_prompt(custom_extraction_process=custom)
    assert custom in prompt_custom

def test_generate_user_prompt_for_docs():
    docs = ["Doc 1 content", "Doc 2 content"]
    prompt = generate_user_prompt_for_docs(docs)
    assert "Doc 1 content" in prompt
    assert "Doc 2 content" in prompt
    assert "---END OF DOCUMENT---" in prompt
