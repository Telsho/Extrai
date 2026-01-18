import unittest
import json
from extrai.core.prompts.common import generate_user_prompt_for_docs
from extrai.core.prompts.extraction import generate_system_prompt


class TestExtractionPrompts(unittest.TestCase):
    def setUp(self):
        """Set up common test data."""
        self.sample_schema_dict = {
            "type": "object",
            "title": "TestEntity",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "value": {"type": "number"},
            },
            "required": ["id", "name"],
        }
        self.sample_schema_json_str = json.dumps(self.sample_schema_dict, indent=2)

        self.sample_extraction_example_dict = {
            "id": "test001",
            "name": "Test Name",
            "value": 123.45,
        }
        self.sample_extraction_example_json_str = json.dumps(
            self.sample_extraction_example_dict, indent=2
        )

    def test_generate_system_prompt_basic(self):
        """Test system prompt with only schema."""
        prompt = generate_system_prompt(schema_json=self.sample_schema_json_str)
        self.assertIn("You are an advanced AI specializing in data extraction", prompt)
        self.assertIn("# JSON SCHEMA TO ADHERE TO:", prompt)
        self.assertIn(self.sample_schema_json_str, prompt)
        self.assertIn("# EXTRACTION PROCESS", prompt)  # Default process
        self.assertIn("# IMPORTANT EXTRACTION GUIDELINES", prompt)  # Default guidelines
        self.assertIn("# FINAL CHECK BEFORE SUBMISSION", prompt)  # Default checklist
        self.assertNotIn("# EXAMPLE OF EXTRACTION:", prompt)

    def test_generate_system_prompt_with_example(self):
        """Test system prompt with schema and example."""
        prompt = generate_system_prompt(
            schema_json=self.sample_schema_json_str,
            extraction_example_json=self.sample_extraction_example_json_str,
        )
        self.assertIn(self.sample_schema_json_str, prompt)
        self.assertIn("# EXAMPLE OF EXTRACTION:", prompt)
        self.assertIn("## CONCEPTUAL INPUT TEXT", prompt)
        self.assertIn(self.sample_extraction_example_json_str, prompt)

    def test_generate_system_prompt_with_custom_process(self):
        """Test system prompt with custom extraction process."""
        custom_process = "# MY CUSTOM PROCESS\n1. Do this first.\n2. Then do that."
        prompt = generate_system_prompt(
            schema_json=self.sample_schema_json_str,
            custom_extraction_process=custom_process,
        )
        self.assertIn(custom_process, prompt)
        self.assertNotIn(
            "Follow this step-by-step process meticulously:", prompt
        )  # Default process heading
        self.assertIn(
            "# IMPORTANT EXTRACTION GUIDELINES", prompt
        )  # Default guidelines should still be there
        self.assertIn("# FINAL CHECK BEFORE SUBMISSION", prompt)  # Default checklist

    def test_generate_system_prompt_with_custom_guidelines(self):
        """Test system prompt with custom extraction guidelines."""
        custom_guidelines = "# MY CUSTOM GUIDELINES\n- Be awesome.\n- Be accurate."
        prompt = generate_system_prompt(
            schema_json=self.sample_schema_json_str,
            custom_extraction_guidelines=custom_guidelines,
        )
        self.assertIn(custom_guidelines, prompt)
        self.assertNotIn(
            "- **Output Format:** Your entire output must be a single, valid JSON object.",
            prompt,
        )  # Default guideline
        self.assertIn("# EXTRACTION PROCESS", prompt)  # Default process
        self.assertIn("# FINAL CHECK BEFORE SUBMISSION", prompt)  # Default checklist

    def test_generate_system_prompt_with_custom_checklist(self):
        """Test system prompt with custom final checklist."""
        custom_checklist = "# MY CUSTOM CHECKLIST\n1. Did I do well?\n2. Yes."
        prompt = generate_system_prompt(
            schema_json=self.sample_schema_json_str,
            custom_final_checklist=custom_checklist,
        )
        self.assertIn(custom_checklist, prompt)
        self.assertNotIn("1.  **Valid JSON?**", prompt)  # Default checklist item
        self.assertIn("# EXTRACTION PROCESS", prompt)  # Default process
        self.assertIn("# IMPORTANT EXTRACTION GUIDELINES", prompt)  # Default guidelines

    def test_generate_system_prompt_with_custom_context(self):
        """Test system prompt with custom_context."""
        custom_context_content = "This is some important external context for the LLM."
        prompt = generate_system_prompt(
            schema_json=self.sample_schema_json_str,
            custom_context=custom_context_content,
        )
        self.assertIn("# ADDITIONAL CONTEXT:", prompt)
        self.assertIn(custom_context_content, prompt)
        self.assertIn(
            self.sample_schema_json_str, prompt
        )  # Ensure schema is still there
        self.assertIn(
            "# EXTRACTION PROCESS", prompt
        )  # Default process should still be there

        # Test that it's not included if empty
        prompt_no_custom_context = generate_system_prompt(
            schema_json=self.sample_schema_json_str, custom_context=""
        )
        self.assertNotIn("# ADDITIONAL CONTEXT:", prompt_no_custom_context)
        self.assertNotIn(custom_context_content, prompt_no_custom_context)

    def test_generate_system_prompt_all_custom(self):
        """Test system prompt with all custom sections."""
        custom_process = "# CUSTOM PROCESS V2"
        custom_guidelines = "# CUSTOM GUIDELINES V2"
        custom_checklist = "# CUSTOM CHECKLIST V2"
        custom_context_content = "All custom context here for the all_custom test."
        prompt = generate_system_prompt(
            schema_json=self.sample_schema_json_str,
            extraction_example_json=self.sample_extraction_example_json_str,
            custom_extraction_process=custom_process,
            custom_extraction_guidelines=custom_guidelines,
            custom_final_checklist=custom_checklist,
            custom_context=custom_context_content,
        )
        self.assertIn(custom_process, prompt)
        self.assertIn(custom_guidelines, prompt)
        self.assertIn(custom_checklist, prompt)
        self.assertIn("# ADDITIONAL CONTEXT:", prompt)
        self.assertIn(custom_context_content, prompt)
        self.assertIn(self.sample_schema_json_str, prompt)
        self.assertIn(self.sample_extraction_example_json_str, prompt)
        self.assertNotIn("Follow this step-by-step process meticulously:", prompt)
        self.assertNotIn(
            "- **Output Format:** Your entire output must be a single, valid JSON object.",
            prompt,
        )
        self.assertNotIn("1.  **Valid JSON?**", prompt)

    def test_generate_user_prompt_single_document(self):
        """Test user prompt with a single document."""
        doc1 = "This is the first document."
        prompt = generate_user_prompt_for_docs([doc1])
        self.assertIn(
            "Please extract information from the following document(s).", prompt
        )
        self.assertIn("# DOCUMENT(S) FOR EXTRACTION:", prompt)
        self.assertIn(doc1, prompt)
        self.assertNotIn("---END OF DOCUMENT---", prompt)  # No separator for single doc
        self.assertIn(
            "Remember: Your output must be only a single, valid JSON object.", prompt
        )

    def test_generate_user_prompt_multiple_documents(self):
        """Test user prompt with multiple documents."""
        doc1 = "Document one content."
        doc2 = "Document two content here."
        doc3 = "And a third document."
        separator = "\n\n---END OF DOCUMENT---\n\n---START OF NEW DOCUMENT---\n\n"
        prompt = generate_user_prompt_for_docs([doc1, doc2, doc3])
        self.assertIn(doc1, prompt)
        self.assertIn(doc2, prompt)
        self.assertIn(doc3, prompt)
        self.assertEqual(prompt.count(separator), 2)  # Two separators for three docs
        self.assertIn("# DOCUMENT(S) FOR EXTRACTION:", prompt)

    def test_generate_user_prompt_empty_documents_list(self):
        """Test user prompt with an empty list of documents."""
        prompt = generate_user_prompt_for_docs([])
        self.assertIn(
            "Please extract information from the following document(s).",
            prompt,
        )
        self.assertIn("# DOCUMENT(S) FOR EXTRACTION:", prompt)

        # Check that the space between "EXTRACTION:" and "---" is just newlines (or empty)
        extraction_header_end_index = prompt.find("# DOCUMENT(S) FOR EXTRACTION:")
        if extraction_header_end_index != -1:
            extraction_header_end_index += len("# DOCUMENT(S) FOR EXTRACTION:")

        reminder_start_index = prompt.find(
            "---"
        )  # Assuming "---" is the start of the reminder section

        if (
            extraction_header_end_index != -1
            and reminder_start_index != -1
            and extraction_header_end_index < reminder_start_index
        ):
            content_between = prompt[extraction_header_end_index:reminder_start_index]
            self.assertEqual(
                content_between.strip(),
                "",
                f"Content between extraction header and reminder should be whitespace only, but was: '{content_between}'",
            )
        elif extraction_header_end_index == -1:
            self.fail("Could not find '# DOCUMENT(S) FOR EXTRACTION:' marker.")
        elif reminder_start_index == -1:
            self.fail("Could not find '---' marker for reminder section.")
        else:  # Markers found but order is wrong or overlap
            self.fail(
                f"Problem with marker positions: extraction_header_end_index={extraction_header_end_index}, reminder_start_index={reminder_start_index}"
            )

        self.assertIn(
            "Remember: Your output must be only a single, valid JSON object.", prompt
        )
        self.assertNotIn(
            "---END OF DOCUMENT---", prompt
        )  # Still important for empty list

    def test_generate_user_prompt_documents_with_special_chars(self):
        """Test user prompt with documents containing special JSON characters."""
        doc1 = 'This document has "quotes" and {curly braces}.'
        doc2 = "Another one with a backslash \\ and newlines \n in theory."
        prompt = generate_user_prompt_for_docs([doc1, doc2])
        self.assertIn(doc1, prompt)
        self.assertIn(doc2, prompt)
        self.assertIn("---END OF DOCUMENT---", prompt)

    def test_generate_system_prompt_with_non_json_example_string(self):
        """
        Test system prompt with an extraction_example_json that is a non-empty,
        non-JSON string. This should cover line 108 and the 'else' branch
        of the inner conditional (line 119 in prompt_builder.py).
        """
        non_json_example_str = "This is a raw example string, not a JSON object."
        prompt = generate_system_prompt(
            schema_json=self.sample_schema_json_str,
            extraction_example_json=non_json_example_str,
        )

        # Check that line 108's content ("# EXAMPLE OF EXTRACTION:") is present
        self.assertIn("# EXAMPLE OF EXTRACTION:", prompt)

        # Check that the non_json_example_str itself is used (from line 119)
        self.assertIn(non_json_example_str, prompt)

        # Check that it's NOT wrapped with {"result": ...} (line 117 should not be hit)
        # Constructing the exact f-string format for the negative assertion
        wrapped_example_check = f'{{\n  "result": {non_json_example_str}\n}}'
        self.assertNotIn(wrapped_example_check, prompt)

        # Also check for other parts of the example section to be sure they are still there
        self.assertIn("## CONCEPTUAL INPUT TEXT", prompt)
        self.assertIn("## EXAMPLE EXTRACTED JSON", prompt)
        # Ensure the ```json block markers are present around the example
        # The example string is non_json_example_str
        # So we expect "```json\n\n" + non_json_example_str + "\n\n```" (joined by \n\n)
        # More robustly, check that the example string is between ```json and ```
        # Find the start of the example section text
        example_section_header_idx = prompt.find("## EXAMPLE EXTRACTED JSON")
        self.assertNotEqual(
            example_section_header_idx != -1, "Example JSON header not found"
        )

        # Find ```json after this header
        json_block_start_marker = "```json"
        json_block_start_idx = prompt.find(
            json_block_start_marker, example_section_header_idx
        )
        self.assertNotEqual(
            json_block_start_idx != -1,
            "```json start marker not found after example header",
        )

        # Find the example string after the ```json marker
        example_str_idx = prompt.find(
            non_json_example_str, json_block_start_idx + len(json_block_start_marker)
        )
        self.assertNotEqual(
            example_str_idx != -1,
            "Non-JSON example string not found after ```json marker",
        )

        # Find ``` end marker after the example string
        json_block_end_marker = "```"
        json_block_end_idx = prompt.find(
            json_block_end_marker, example_str_idx + len(non_json_example_str)
        )
        self.assertNotEqual(
            json_block_end_idx != -1,
            "``` end marker not found after non-JSON example string",
        )

    def test_generate_system_prompt_includes_ordering_and_id_rules(self):
        """Test that the system prompt includes instructions for ordering and semantic IDs."""
        prompt = generate_system_prompt(schema_json=self.sample_schema_json_str)

        # Check for ordering instruction
        self.assertIn(
            "Maintain the order of items as they appear in the source text",
            prompt,
            "Prompt missing instruction about preserving order",
        )

        # Check for semantic ID instruction
        self.assertIn(
            "based on the entity's key attributes",
            prompt,
            "Prompt missing instruction about semantic IDs",
        )
        self.assertIn(
            "E.g., `user_john_doe`", prompt, "Prompt missing example of semantic IDs"
        )

    def test_generate_user_prompt_with_custom_context(self):
        """Test user prompt with custom context."""
        doc1 = "This is a document."
        custom_ctx = "Pay attention to X."
        prompt = generate_user_prompt_for_docs([doc1], custom_context=custom_ctx)
        self.assertIn(custom_ctx, prompt)
        self.assertIn(doc1, prompt)


if __name__ == "__main__":
    unittest.main()
