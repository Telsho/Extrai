import unittest
import json
from extrai.core.prompts.sqlmodel import generate_sqlmodel_creation_system_prompt

class TestSQLModelPrompts(unittest.TestCase):
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

if __name__ == "__main__":
    unittest.main()
