import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import json
from sqlmodel import SQLModel, Field

from extrai.core.batch_pipeline import BatchPipeline
from extrai.core.model_registry import ModelRegistry
from extrai.core.extraction_config import ExtractionConfig
from extrai.core.base_llm_client import BaseLLMClient
from extrai.core.batch_models import BatchJobContext


class Recipe(SQLModel):
    name: str
    ingredients: list[str] = Field(default_factory=list)
    prep_time: int


class TestBatchPipelineStructured(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_model_registry = MagicMock(spec=ModelRegistry)
        self.mock_model_registry.root_model = Recipe
        self.mock_model_registry.models = [Recipe]
        self.mock_model_registry.llm_schema_json = "{}"
        self.mock_model_registry.model_map = {"Recipe": Recipe}

        self.mock_client = MagicMock(spec=BaseLLMClient)
        self.mock_client.temperature = 0.0

        # Configure for Structured Output
        self.mock_config = MagicMock(spec=ExtractionConfig)
        self.mock_config.consensus_threshold = 0.6
        self.mock_config.conflict_resolver = None
        self.mock_config.num_llm_revisions = 1
        self.mock_config.max_validation_retries_per_revision = 1
        self.mock_config.use_structured_output = True  # ENABLED
        self.mock_config.use_hierarchical_extraction = False

        self.mock_analytics = MagicMock()
        self.mock_session = MagicMock()
        self.mock_logger = MagicMock()

        with (
            patch("extrai.core.batch_pipeline.ClientRotator") as MockClientRotator,
            patch(
                "extrai.core.batch_pipeline.ExtractionContextPreparer"
            ) as MockContextPreparer,
            patch("extrai.core.batch_pipeline.PromptBuilder") as MockBuilder,
            patch("extrai.core.batch_pipeline.EntityCounter") as MockCounter,
            patch("extrai.core.batch_pipeline.JSONConsensus") as MockConsensus,
            patch("extrai.core.batch_pipeline.ModelWrapperBuilder"),
        ):
            self.pipeline = BatchPipeline(
                self.mock_model_registry,
                self.mock_client,
                self.mock_config,
                self.mock_analytics,
                self.mock_logger,
            )
            self.pipeline.client_rotator = MockClientRotator.return_value
            self.pipeline.context_preparer = MockContextPreparer.return_value
            self.pipeline.prompt_builder = MockBuilder.return_value
            self.pipeline.entity_counter = MockCounter.return_value
            self.pipeline.consensus = MockConsensus.return_value

    async def test_retrieve_and_validate_results_missing_type(self):
        """
        Test that validation FAILS when _type is missing in structured output mode
        (before the fix is applied).
        """
        mock_context = BatchJobContext(
            current_batch_id="prov_1",
            config={
                "use_structured_output": True,
                "schema_json": {},
            },
        )

        mock_client = self.pipeline.client_rotator.get_next_client.return_value

        # Simulating structured output which does NOT contain _type
        structured_response = {
            "entities": [
                {"name": "Pancake", "ingredients": ["flour", "milk"], "prep_time": 10}
            ]
        }

        mock_client.retrieve_batch_results = AsyncMock(
            return_value=json.dumps(structured_response)
        )

        # We need to simulate how extract_content_from_batch_response behaves.
        # Assuming it returns the inner JSON string or dict.
        # In the original code it calls `process_and_validate_llm_output`.

        # For this test, we mock extract_content_from_batch_response to return the JSON string of entities wrapper
        # The real client implementation varies, but let's assume it returns the raw JSON string
        mock_client.extract_content_from_batch_response.return_value = json.dumps(
            structured_response
        )

        # We expect this to fail because we haven't fixed the code yet,
        # and process_and_validate_llm_output will look for _type.

        # NOTE: process_and_validate_llm_output is imported in batch_pipeline.
        # We shouldn't patch it if we want to test the failure integration,
        # but process_and_validate_llm_output raises LLMOutputValidationError.
        # BatchPipeline catches Exception and logs it, returning empty list if validation fails.

        # However, looking at _retrieve_and_validate_results:
        # It logs warning on validation failure.

        # To assert failure, we can check that the returned list is empty
        # OR we can mock process_and_validate_llm_output to see what it was called with
        # OR we can let it run and see if it returns valid objects.

        # Since we want to prove it fails validation, we should let the real process_and_validate_llm_output run.
        # But `process_and_validate_llm_output` requires `Recipe` (SQLModel) to be in `model_schema_map`.
        # We set that up in setUp.

        results = await self.pipeline._retrieve_and_validate_results(mock_context)

        # With the fix, we expect results to be validated and returned
        # Since _type is injected, it should be present in the result
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 1)
        item = results[0][0]
        self.assertEqual(item["name"], "Pancake")
        self.assertEqual(item["_type"], "Recipe")


if __name__ == "__main__":
    unittest.main()
