import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import json
from extrai.core.batch_pipeline import BatchPipeline, BatchJobStatus
from extrai.core.model_registry import ModelRegistry
from extrai.core.extraction_config import ExtractionConfig
from extrai.core.base_llm_client import BaseLLMClient
from extrai.core.batch_models import BatchJobContext


class TestBatchPipeline(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_model_registry = MagicMock(spec=ModelRegistry)
        self.mock_model_registry.root_model = MagicMock()
        self.mock_model_registry.root_model.__name__ = "RootModel"
        self.mock_model_registry.models = [self.mock_model_registry.root_model]
        self.mock_model_registry.llm_schema_json = "{}"
        self.mock_model_registry.model_map = {
            "RootModel": self.mock_model_registry.root_model
        }

        self.mock_client = MagicMock(spec=BaseLLMClient)
        self.mock_client.temperature = 0.0
        self.mock_config = MagicMock(spec=ExtractionConfig)
        self.mock_config.consensus_threshold = 0.6
        self.mock_config.conflict_resolver = None
        self.mock_config.num_llm_revisions = 1
        self.mock_config.max_validation_retries_per_revision = 1
        self.mock_config.use_structured_output = False
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
            # We need to access the instances created inside, so we'll mock them on the pipeline instance
            self.pipeline.client_rotator = MockClientRotator.return_value
            self.pipeline.context_preparer = MockContextPreparer.return_value
            self.pipeline.prompt_builder = MockBuilder.return_value
            self.pipeline.entity_counter = MockCounter.return_value
            self.pipeline.consensus = MockConsensus.return_value

    async def test_submit_batch_success(self):
        self.pipeline.prompt_builder.build_prompts.return_value = ("sys", "user")

        mock_batch_job = MagicMock()
        mock_batch_job.id = "provider_id_123"

        mock_client_instance = self.pipeline.client_rotator.get_next_client.return_value
        mock_client_instance.create_batch_job = AsyncMock(return_value=mock_batch_job)

        self.pipeline.context_preparer.prepare_example = AsyncMock(return_value="")
        self.pipeline._count_if_needed = AsyncMock(return_value=None)

        root_id = await self.pipeline.submit_batch(self.mock_session, ["doc"])

        self.assertIsInstance(root_id, str)
        self.mock_session.add.assert_called_once()
        self.mock_session.commit.assert_called_once()

        added_context = self.mock_session.add.call_args[0][0]
        self.assertIsInstance(added_context, BatchJobContext)
        self.assertEqual(added_context.current_batch_id, "provider_id_123")
        self.assertEqual(added_context.status, BatchJobStatus.SUBMITTED)

    async def test_retrieve_and_validate_results(self):
        mock_context = BatchJobContext(current_batch_id="prov_1")

        mock_client = self.pipeline.client_rotator.get_next_client.return_value
        mock_client.retrieve_batch_results = AsyncMock(
            return_value='{"key": "value"}\n{"key": "value2"}'
        )
        mock_client.extract_content_from_batch_response.side_effect = [
            '{"_type": "RootModel", "id": 1}',
            '{"_type": "RootModel", "id": 2}',
        ]

        with patch(
            "extrai.core.batch_pipeline.process_and_validate_llm_output"
        ) as mock_validate:
            mock_validate.side_effect = [[{"id": 1}], [{"id": 2}]]
            results = await self.pipeline._retrieve_and_validate_results(mock_context)

            # normalize_json_revisions wraps each revision in a list.
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0], [{"id": 1}])
            self.assertEqual(results[1], [{"id": 2}])


if __name__ == "__main__":
    unittest.main()
