import unittest
import json
from extrai.core.prompts.examples import generate_prompt_for_example_json_generation


class TestExamplePrompts(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
