import unittest
from extrai.core.prompts.counting import (
    generate_entity_counting_system_prompt,
    generate_entity_counting_user_prompt,
)


class TestCountingPrompts(unittest.TestCase):
    def test_generate_entity_counting_system_prompt(self):
        prompt = generate_entity_counting_system_prompt(
            model_names=["TestModel"], schema_json="{}"
        )
        self.assertIn("You are an expert data analyst", prompt)

    def test_generate_entity_counting_user_prompt(self):
        docs = ["doc1", "doc2"]
        prompt = generate_entity_counting_user_prompt(docs)
        self.assertIn("doc1", prompt)
        self.assertIn("doc2", prompt)


if __name__ == "__main__":
    unittest.main()
