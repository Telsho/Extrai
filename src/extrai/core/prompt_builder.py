"""
This module serves as a facade for the prompt generation logic, which has been
modularized into the `extrai.core.prompts` package.
"""

import logging

from extrai.core.model_registry import ModelRegistry
from extrai.core.prompts.common import generate_user_prompt_for_docs
from extrai.core.prompts.counting import (
    generate_entity_counting_system_prompt,
    generate_entity_counting_user_prompt,
)
from extrai.core.prompts.examples import (
    generate_prompt_for_example_json_generation,
)
from extrai.core.prompts.extraction import (
    generate_system_prompt,
)
from extrai.core.prompts.sqlmodel import (
    generate_sqlmodel_creation_system_prompt,
)
from extrai.core.prompts.structured_extraction import (
    generate_structured_system_prompt,
)


class PromptBuilder:
    """
    Facade class for generating prompts, maintaining compatibility with
    pipeline components that expect an object instance.
    """

    def __init__(
        self, model_registry: ModelRegistry, logger: logging.Logger | None = None
    ):
        self.model_registry = model_registry
        self.logger = logger or logging.getLogger(__name__)

    def build_prompts(
        self,
        input_strings: list[str],
        schema_json: str,
        extraction_example_json: str = "",
        custom_extraction_process: str = "",
        custom_extraction_guidelines: str = "",
        custom_final_checklist: str = "",
        custom_context: str = "",
        expected_entity_descriptions: list[dict] | None = None,
        previous_entities: list[dict] | None = None,
        target_model_name: str | None = None,
    ) -> tuple[str, str]:
        """
        Builds system and user prompts for extraction.
        """
        system_prompt = generate_system_prompt(
            schema_json=schema_json,
            extraction_example_json=extraction_example_json,
            custom_extraction_process=custom_extraction_process,
            custom_extraction_guidelines=custom_extraction_guidelines,
            custom_final_checklist=custom_final_checklist,
            custom_context=custom_context,
            expected_entity_descriptions=expected_entity_descriptions,
            previous_entities=previous_entities,
            target_model_name=target_model_name,
        )

        user_prompt = generate_user_prompt_for_docs(input_strings)

        return system_prompt, user_prompt

    def build_structured_prompts(
        self,
        input_strings: list[str],
        custom_extraction_process: str = "",
        custom_extraction_guidelines: str = "",
        custom_context: str = "",
        extraction_example_json: str = "",
        expected_entity_descriptions: list[dict] | None = None,
        previous_entities: list[dict] | None = None,
        target_model_name: str | None = None,
    ) -> tuple[str, str]:
        """
        Builds prompts for structured extraction.
        """
        system_prompt = generate_structured_system_prompt(
            custom_extraction_process=custom_extraction_process,
            custom_extraction_guidelines=custom_extraction_guidelines,
            custom_context=custom_context,
            extraction_example_json=extraction_example_json,
            expected_entity_descriptions=expected_entity_descriptions,
            previous_entities=previous_entities,
            target_model_name=target_model_name,
        )

        user_prompt = generate_user_prompt_for_docs(input_strings)

        return system_prompt, user_prompt


__all__ = [
    "PromptBuilder",
    "generate_system_prompt",
    "generate_user_prompt_for_docs",
    "generate_sqlmodel_creation_system_prompt",
    "generate_entity_counting_system_prompt",
    "generate_entity_counting_user_prompt",
    "generate_prompt_for_example_json_generation",
]
