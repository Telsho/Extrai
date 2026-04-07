import logging
from typing import Any

from pydantic import BaseModel, Field

from .counting_consensus import CountingConsensus
from .extraction_config import ExtractionConfig
from .model_registry import ModelRegistry
from .prompt_builder import (
    generate_entity_counting_system_prompt,
    generate_entity_counting_user_prompt,
)


class CountedEntity(BaseModel):
    model: str
    temp_id: str
    related_ids: list[str] = Field(default_factory=list)
    description: str


class EntityCountResult(BaseModel):
    counted_entities: list[CountedEntity] = Field(default_factory=list)


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
        self.counting_consensus = CountingConsensus(
            config=self.config,
            llm_client=self.llm_client,
            logger=self.logger,
        )

    def prepare_counting_prompts(
        self,
        input_strings: list[str],
        model_names: list[str],
        custom_counting_context: str = "",
        previous_entities: list[dict[str, Any]] | None = None,
        examples: str = "",
    ):
        """Prepares prompts for batch counting."""
        # Generate schema for models
        schema_json = self.model_registry.get_schema_for_models(model_names)

        # Build prompts
        system_prompt = generate_entity_counting_system_prompt(
            model_names,
            schema_json,
            custom_counting_context,
            previous_entities,
            examples,
        )
        user_prompt = generate_entity_counting_user_prompt(input_strings)

        return system_prompt, user_prompt

    def validate_counts(
        self, raw_counts: dict[str, Any], model_names: list[str]
    ) -> dict[str, list[str]]:
        """Validates raw counting results against static model."""
        try:
            validated = EntityCountResult(**raw_counts)
            return validated.model_dump()
        except Exception as e:
            self.logger.warning(f"Count validation failed: {e}")
            return {}

    def get_counting_model(self, model_names: list[str]):
        """Creates a Pydantic model for entity counting."""
        return EntityCountResult

    async def count_entities(
        self,
        input_strings: list[str],
        model_names: list[str],
        custom_counting_context: str = "",
        previous_entities: list[dict[str, Any]] | None = None,
        examples: str = "",
    ) -> list[dict[str, Any]]:
        """Performs entity counting for specified models using consensus."""
        self.logger.info(f"Counting entities for: {model_names}")

        system_prompt, user_prompt = self.prepare_counting_prompts(
            input_strings,
            model_names,
            custom_counting_context,
            previous_entities,
            examples,
        )

        target_json_schema = (
            EntityCountResult.model_json_schema()
            if self.config.use_structured_output
            else None
        )

        client = self.llm_client
        if isinstance(client, list):
            client = client[0]

        try:
            # Execute multiple revisions natively
            revisions = await client.generate_and_validate_raw_json_output(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                target_json_schema=target_json_schema,
                num_revisions=self.config.num_counting_revisions,
                max_validation_retries_per_revision=self.config.max_validation_retries_per_revision,
                attempt_unwrap=False,
            )

            # Revisions should be a list of dictionaries if successful
            if not isinstance(revisions, list):
                if isinstance(revisions, dict):
                    revisions = [revisions]
                else:
                    self.logger.warning(
                        "Entity counting returned invalid result format"
                    )
                    return []

            # Achieve consensus
            consensus_result = await self.counting_consensus.achieve_consensus(
                revisions=revisions,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                target_json_schema=target_json_schema,
            )

            return consensus_result

        except Exception as e:
            self.logger.error(f"Entity counting failed: {e}")
            return []
