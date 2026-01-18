import json
import logging
from typing import List, Optional, Union, Callable
from sqlmodel import SQLModel

from .model_registry import ModelRegistry
from .example_json_generator import ExampleJSONGenerator, ExampleGenerationError
from .analytics_collector import WorkflowAnalyticsCollector
from .errors import WorkflowError
from .base_llm_client import BaseLLMClient
from extrai.utils.serialization_utils import serialize_sqlmodel_with_relationships


class ExtractionContextPreparer:
    """
    Helper class to prepare context for extraction, including example generation.
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        analytics_collector: WorkflowAnalyticsCollector,
        max_retries: int,
        logger: logging.Logger,
    ):
        self.model_registry = model_registry
        self.analytics_collector = analytics_collector
        self.max_retries = max_retries
        self.logger = logger

    async def prepare_example(
        self,
        extraction_example_json: str,
        extraction_example_object: Optional[Union[SQLModel, List[SQLModel]]],
        client_provider: Callable[[], BaseLLMClient],
    ) -> str:
        """
        Prepares or auto-generates extraction example.

        Priority:
        1. Use provided extraction_example_json if available
        2. Serialize extraction_example_object if provided
        3. Auto-generate example using LLM
        """
        # If JSON provided, use it directly
        if extraction_example_json:
            self.logger.info("Using provided extraction example JSON")
            return extraction_example_json

        # If object provided, serialize it
        if extraction_example_object:
            self.logger.info("Serializing extraction example object")
            return self._serialize_example_object(extraction_example_object)

        # Auto-generate
        self.logger.info("No example provided, auto-generating...")
        return await self._auto_generate_example(client_provider)

    def _serialize_example_object(self, obj: Union[SQLModel, List[SQLModel]]) -> str:
        """Serializes SQLModel object(s) to JSON."""
        objects = obj if isinstance(obj, list) else [obj]
        serialized = []

        for o in objects:
            if isinstance(o, SQLModel):
                serialized.append(serialize_sqlmodel_with_relationships(o))
            else:
                self.logger.warning(
                    f"Skipping non-SQLModel object in example: {type(o)}"
                )

        if not serialized:
            self.logger.warning("No valid SQLModel objects to serialize")
            return ""

        return json.dumps(serialized, default=str, indent=2)

    async def _auto_generate_example(
        self, client_provider: Callable[[], BaseLLMClient]
    ) -> str:
        """Auto-generates an extraction example using LLM."""
        try:
            generator = ExampleJSONGenerator(
                llm_client=client_provider(),
                output_model=self.model_registry.root_model,
                analytics_collector=self.analytics_collector,
                max_validation_retries_per_revision=self.max_retries,
                logger=self.logger,
            )

            self.logger.info(
                f"Auto-generating extraction example for "
                f"{self.model_registry.root_model.__name__}..."
            )

            example = await generator.generate_example()

            self.analytics_collector.record_custom_event(
                "example_json_auto_generation_success"
            )
            self.logger.info("Successfully auto-generated extraction example")

            return example

        except ExampleGenerationError as e:
            self.analytics_collector.record_custom_event(
                "example_json_auto_generation_failure"
            )
            raise WorkflowError(
                f"Failed to auto-generate extraction example: {e}"
            ) from e
        except Exception as e:
            self.analytics_collector.record_custom_event(
                "example_json_auto_generation_unexpected_failure"
            )
            raise WorkflowError(
                f"Unexpected error during extraction example auto-generation: {e}"
            ) from e
