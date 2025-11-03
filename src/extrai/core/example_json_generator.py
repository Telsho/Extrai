import json
import logging
from typing import Optional, Dict, Any, Type

from sqlmodel import SQLModel
from extrai.core.base_llm_client import BaseLLMClient
from extrai.core.prompt_builder import (
    generate_prompt_for_example_json_generation,
)
from extrai.core.analytics_collector import (
    WorkflowAnalyticsCollector,
)
from extrai.core.errors import (
    ExampleGenerationError,
    LLMAPICallError,
    LLMOutputParseError,
    LLMOutputValidationError,
    ConfigurationError,
)
from .schema_inspector import (
    generate_llm_schema_from_models,
    discover_sqlmodels_from_root,
)


class ExampleJSONGenerator:
    """
    Handles the generation of an example JSON string based on a given schema,
    using an LLM.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        output_model: Type[SQLModel],
        analytics_collector: Optional[WorkflowAnalyticsCollector] = None,
        max_validation_retries_per_revision: int = 1,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        if not logger:
            self.logger.setLevel(logging.WARNING)

        if not llm_client:
            raise ConfigurationError("llm_client must be provided.")
        if not output_model:
            raise ConfigurationError("output_model must be provided.")
        if not issubclass(output_model, SQLModel):
            raise ConfigurationError(
                f"output_model ({output_model.__name__}) must be a subclass of SQLModel."
            )
        if max_validation_retries_per_revision < 1:
            raise ConfigurationError(
                "max_validation_retries_per_revision must be at least 1."
            )

        self.llm_client = llm_client
        self.llm_client.logger = self.logger
        self.output_model = output_model
        self.analytics_collector = analytics_collector
        self.max_validation_retries_per_revision = max_validation_retries_per_revision

        # Derive schema and root model name from the SQLModel
        try:
            # Discover all related models starting from the root model
            all_models = discover_sqlmodels_from_root(output_model)

            # Generate the comprehensive schema for the LLM, which includes all related models
            # to guide the LLM in creating a nested example.
            self.target_json_schema_for_llm_str = generate_llm_schema_from_models(
                initial_model_classes=all_models
            )

            # The schema for basic validation by the LLM client needs to match the new
            # expected output format: `{"entities": [...]}`.
            self.target_json_schema_dict: Dict[str, Any] = {
                "type": "object",
                "properties": {
                    "entities": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["entities"],
            }
        except Exception as e:
            raise ConfigurationError(
                f"Failed to derive JSON schema from output_model {output_model.__name__}: {e}"
            ) from e
        self.root_model_name: str = output_model.__name__

    async def generate_example(self) -> str:
        """
        Generates a sample JSON string conforming to the target_json_schema.

        Returns:
            A JSON string representing the generated example.

        Raises:
            ExampleGenerationError: If any step in the generation process fails.
        """
        self.logger.info("Attempting to generate example JSON...")
        system_prompt = generate_prompt_for_example_json_generation(
            target_model_schema_str=self.target_json_schema_for_llm_str,
            root_model_name=self.root_model_name,
        )

        user_prompt = "Please generate a sample JSON object based on the schema and instructions provided in the system prompt."

        try:
            # Discover all related models to build the schema map for validation.
            all_models = discover_sqlmodels_from_root(self.output_model)
            model_schema_map = {model.__name__: model for model in all_models}

            validated_revisions = await self.llm_client.generate_json_revisions(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                num_revisions=1,
                model_schema_map=model_schema_map,
                max_validation_retries_per_revision=self.max_validation_retries_per_revision,
                analytics_collector=self.analytics_collector,
            )

            if not validated_revisions:
                raise ExampleGenerationError(
                    "LLM client returned no valid example JSON revisions."
                )

            # The final output for the few-shot example should be the full entities list.
            return json.dumps(validated_revisions)

        except (
            LLMAPICallError,
            LLMOutputParseError,
            LLMOutputValidationError,
        ) as llm_err:
            self.logger.error(
                f"Failed to generate example JSON due to LLM client error: {llm_err}"
            )
            raise ExampleGenerationError(
                f"Failed to generate example JSON due to LLM client error: {llm_err}",
                original_exception=llm_err,
            ) from llm_err
        except TypeError as e:
            self.logger.error(f"Failed to serialize the generated example JSON: {e}")
            raise ExampleGenerationError(
                f"Failed to serialize the generated example JSON: {e}",
                original_exception=e,
            ) from e
        except ExampleGenerationError:
            raise
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred during example JSON generation: {e}"
            )
            raise ExampleGenerationError(
                f"An unexpected error occurred during example JSON generation: {e}",
                original_exception=e,
            ) from e
