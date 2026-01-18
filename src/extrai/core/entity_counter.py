import logging
from typing import List, Dict, Any, Optional
from pydantic import create_model

from .model_registry import ModelRegistry
from .extraction_config import ExtractionConfig
from .prompt_builder import (
    generate_entity_counting_system_prompt,
    generate_entity_counting_user_prompt,
)


class EntityCounter:
    """Counts entities in input documents using LLM."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        llm_client,
        config: ExtractionConfig,
        analytics_collector,
        logger: logging.Logger,
    ):
        self.model_registry = model_registry
        self.llm_client = llm_client
        self.config = config
        self.analytics_collector = analytics_collector
        self.logger = logger

    def prepare_counting_prompts(
        self,
        input_strings: List[str],
        model_names: List[str],
        custom_counting_context: str = "",
        previous_entities: Optional[List[Dict[str, Any]]] = None,
    ):
        """Prepares prompts for batch counting."""
        # Generate schema for models
        schema_json = self.model_registry.get_schema_for_models(model_names)

        # Build prompts
        system_prompt = generate_entity_counting_system_prompt(
            model_names, schema_json, custom_counting_context, previous_entities
        )
        user_prompt = generate_entity_counting_user_prompt(input_strings)

        return system_prompt, user_prompt

    def validate_counts(
        self, raw_counts: Dict[str, Any], model_names: List[str]
    ) -> Dict[str, List[str]]:
        """Validates raw counting results against dynamic model."""
        fields = {name: (List[str], ...) for name in model_names}
        EntityCountModel = create_model("EntityCountModel", **fields)
        try:
            validated = EntityCountModel(**raw_counts)
            return validated.model_dump()
        except Exception as e:
            self.logger.warning(f"Count validation failed: {e}")
            return {}

    async def count_entities(
        self,
        input_strings: List[str],
        model_names: List[str],
        custom_counting_context: str = "",
        previous_entities: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, List[str]]:
        """Performs entity counting for specified models."""
        self.logger.info(f"Counting entities for: {model_names}")

        system_prompt, user_prompt = self.prepare_counting_prompts(
            input_strings, model_names, custom_counting_context, previous_entities
        )

        # Create validation model
        fields = {name: (List[str], ...) for name in model_names}
        EntityCountModel = create_model("EntityCountModel", **fields)

        # Call LLM
        try:
            # Get next client (assuming llm_client is list or has rotation)
            if isinstance(self.llm_client, list):
                client = self.llm_client[0]
            else:
                client = self.llm_client

            result = await client.generate_and_validate_raw_json_output(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                target_json_schema=None,
                num_revisions=1,
                max_validation_retries_per_revision=self.config.max_validation_retries_per_revision,
                attempt_unwrap=False,
            )

            # Process result
            if isinstance(result, list) and result:
                result = result[0]

            if isinstance(result, dict):
                validated = EntityCountModel(**result)
                counts = validated.model_dump()
                self.logger.info(f"Entity counts: {counts}")
                return counts

            self.logger.warning("Entity counting returned invalid result")
            return {}

        except Exception as e:
            self.logger.error(f"Entity counting failed: {e}")
            return {}
