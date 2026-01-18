import unittest
from extrai.core.prompt_builder import (
    PromptBuilder,
    generate_system_prompt,
    generate_user_prompt_for_docs,
    generate_sqlmodel_creation_system_prompt,
    generate_prompt_for_example_json_generation,
    generate_entity_counting_system_prompt,
    generate_entity_counting_user_prompt,
)


class TestPromptBuilderFacade(unittest.TestCase):
    def test_facade_exports(self):
        """Verify that the facade module exports the expected functions and class."""
        # Just checking if they are callable is enough for a facade test
        # deeper logic is tested in tests/core/prompts/*
        self.assertTrue(callable(PromptBuilder))
        self.assertTrue(callable(generate_system_prompt))
        self.assertTrue(callable(generate_user_prompt_for_docs))
        self.assertTrue(callable(generate_sqlmodel_creation_system_prompt))
        self.assertTrue(callable(generate_prompt_for_example_json_generation))
        self.assertTrue(callable(generate_entity_counting_system_prompt))
        self.assertTrue(callable(generate_entity_counting_user_prompt))


if __name__ == "__main__":
    unittest.main()
