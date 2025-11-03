import unittest
import json
from extrai.core.prompt_builder import (
    generate_system_prompt,
    generate_user_prompt_for_docs,
    generate_sqlmodel_creation_system_prompt,
    generate_prompt_for_example_json_generation,  # New function to be tested
)


class TestPromptBuilder(unittest.TestCase):
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
            "Please extract information from the following document(s)", prompt
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
            "Please extract information from the following document(s) strictly according to the schema and instructions previously provided (in the system prompt).",
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

    def test_generate_sqlmodel_creation_system_prompt(self):
        """Test the specialized system prompt for SQLModel creation."""
        sample_sqlmodel_schema_str = json.dumps(
            {
                "title": "SQLModelDescription",
                "type": "object",
                "properties": {
                    "model_name": {"type": "string"},
                    "fields": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["model_name", "fields"],
            },
            indent=2,
        )
        user_task_desc = "Create a model for tracking library books, including title, author, and ISBN."

        prompt = generate_sqlmodel_creation_system_prompt(
            schema_json=sample_sqlmodel_schema_str, user_task_description=user_task_desc
        )

        self.assertIn(
            "You are an AI assistant tasked with designing one or more SQLModel class definitions.",
            prompt,
        )

        self.assertIn(
            "3. Each object in the `sql_models` list MUST strictly adhere to the following JSON schema for a SQLModel description:",
            prompt,
        )
        # Correctly check for the schema block - note the double newlines from how prompt_parts are joined
        self.assertIn(f"```json\n\n{sample_sqlmodel_schema_str}\n\n```", prompt)

        # Check for the new "IMPORTANT CONSIDERATIONS" section
        self.assertIn("# IMPORTANT CONSIDERATIONS FOR DATABASE TABLE MODELS:", prompt)
        self.assertIn(
            'field_options_str": "Field(default_factory=list, sa_type=JSON)"', prompt
        )  # Example for List
        self.assertIn(
            'add `"from sqlmodel import JSON"` to the main `imports` array', prompt
        )
        self.assertIn(
            "MUST provide a sensible `default` value", prompt
        )  # Instruction for defaults

        # Check for the user task description section
        self.assertIn("# USER'S TASK:", prompt)
        self.assertIn(
            f'The user wants to define a SQLModel based on the following objective: "{user_task_desc}"',
            prompt,
        )
        self.assertIn(
            "Pay close attention to the requirements for List/Dict types if the model is a table, and try to provide default values for required fields.",
            prompt,
        )

        # Check for the hardcoded example section
        self.assertIn(
            "# EXAMPLE OF A VALID SQLMODEL DESCRIPTION JSON (Illustrating a list of models):",
            prompt,
        )
        # Check for a snippet from the hardcoded example to ensure it's present
        self.assertIn('"model_name": "ExampleItem"', prompt)
        self.assertIn('"table_name": "example_items"', prompt)
        self.assertIn("Timestamp of when the item was created.", prompt)
        self.assertIn('"name": "categories"', prompt)  # Part of the new example
        self.assertIn(
            '"field_options_str": "Field(default_factory=list, sa_type=JSON)"', prompt
        )  # Part of the new example
        self.assertIn(
            '"from sqlmodel import SQLModel, Field, JSON"', prompt
        )  # Import in the new example

    def test_generate_sqlmodel_creation_system_prompt_structure_and_hardcoded_example(
        self,
    ):
        """
        Tests the overall structure and presence of the hardcoded example in
        the SQLModel creation prompt.
        """
        sample_schema_for_description_str = json.dumps(
            {
                "title": "SQLModelDesc",
                "type": "object",
                "properties": {"model_name": {"type": "string"}},
                "required": ["model_name"],
            },
            indent=2,
        )
        user_task = "Define a product model."

        prompt = generate_sqlmodel_creation_system_prompt(
            schema_json=sample_schema_for_description_str,
            user_task_description=user_task,
        )

        # General structure checks
        self.assertTrue(
            prompt.startswith(
                "You are an AI assistant tasked with designing one or more SQLModel class definitions."
            )
        )
        self.assertIn("# REQUIREMENTS FOR YOUR OUTPUT:", prompt)
        self.assertIn(
            "3. Each object in the `sql_models` list MUST strictly adhere to the following JSON schema for a SQLModel description:",
            prompt,
        )
        self.assertIn(
            sample_schema_for_description_str, prompt
        )  # Check if the passed schema is there
        self.assertIn("# USER'S TASK:", prompt)
        self.assertIn(user_task, prompt)
        self.assertIn(
            "# IMPORTANT CONSIDERATIONS FOR DATABASE TABLE MODELS:", prompt
        )  # New section check
        self.assertIn(
            "# EXAMPLE OF A VALID SQLMODEL DESCRIPTION JSON (Illustrating a list of models):",
            prompt,
        )
        self.assertTrue(
            prompt.endswith(
                "Do not include any other narrative, explanations, or conversational elements in your output."
            )
        )

        # Specific checks for the hardcoded example's content (which now includes List[str] example)
        self.assertIn('"model_name": "ExampleItem"', prompt)
        self.assertIn('"table_name": "example_items"', prompt)
        self.assertIn('primary_key": true', prompt)
        self.assertIn("datetime.datetime.utcnow", prompt)
        self.assertIn('"name": "categories"', prompt)
        self.assertIn(
            '"field_options_str": "Field(default_factory=list, sa_type=JSON)"', prompt
        )
        self.assertIn('"from sqlmodel import SQLModel, Field, JSON"', prompt)

        # Ensure the example JSON block is correctly formatted
        example_intro_text = "This is an example of the kind of JSON object you should produce (it conforms to the schema above):"
        self.assertIn(example_intro_text, prompt)

        # Extract the example JSON part to validate it
        try:
            # Find the start of the example JSON block
            json_block_marker = "```json\n\n"  # Note the double newline
            # Find the specific example block after the intro text
            example_intro_end_idx = prompt.find(example_intro_text) + len(
                example_intro_text
            )
            json_code_block_start_idx = prompt.find(
                json_block_marker, example_intro_end_idx
            )

            if json_code_block_start_idx == -1:
                self.fail(
                    f"Could not find the start of the example JSON code block ('{json_block_marker}')."
                )

            # Move past the marker itself
            actual_json_start_idx = json_code_block_start_idx + len(json_block_marker)

            # Find the end of this specific JSON code block (which is \n\n```)
            json_code_block_end_marker = "\n\n```"
            json_code_block_end_idx = prompt.find(
                json_code_block_end_marker, actual_json_start_idx
            )
            if json_code_block_end_idx == -1:
                self.fail(
                    f"Could not find the end of the example JSON code block ('{json_code_block_end_marker}')."
                )

            example_json_str_from_prompt = prompt[
                actual_json_start_idx:json_code_block_end_idx
            ].strip()

            # Validate that this extracted string is valid JSON
            json.loads(example_json_str_from_prompt)
        except json.JSONDecodeError as e:
            self.fail(
                f"Hardcoded example JSON in prompt is not valid JSON: {e}\nExtracted JSON string:\n'{example_json_str_from_prompt}'"
            )
        except Exception as e:  # Catch other potential errors during extraction
            self.fail(
                f"Failed to extract or validate the hardcoded example JSON from prompt: {e}"
            )

    def test_generate_prompt_for_example_json_generation(self):
        """Test the prompt generation for creating an example JSON."""
        root_model_name = "SampleOutputModel"

        prompt = generate_prompt_for_example_json_generation(
            target_model_schema_str=self.sample_schema_json_str,
            root_model_name=root_model_name,
        )

        # General instructions
        self.assertIn(
            "You are an AI assistant tasked with generating a sample JSON object.",
            prompt,
        )
        self.assertIn(
            f"The goal is to create a single, valid JSON object that conforms to the provided schema for a model named '{root_model_name}' and its related models.",
            prompt,
        )
        self.assertIn("# JSON SCHEMA TO ADHERE TO:", prompt)
        self.assertIn(self.sample_schema_json_str, prompt)
        self.assertIn(
            "Your output MUST be a single JSON object with a top-level key named 'entities'.",
            prompt,
        )
        self.assertIn(
            "Each object inside the 'entities' list MUST include two metadata fields:",
            prompt,
        )
        self.assertIn(
            "`_type`: This field's value MUST be a string matching the name of the model it represents",
            prompt,
        )
        self.assertIn(
            "`_temp_id`: This field's value MUST be a unique temporary string identifier for that specific entity instance",
            prompt,
        )
        self.assertIn(
            "Your 'entities' list should contain an instance of the root model", prompt
        )
        self.assertIn("at least one instance of each of its related models", prompt)

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


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
