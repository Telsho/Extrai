import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import json
from extrai.core.batch_pipeline import BatchPipeline, BatchJobStatus
from extrai.core.model_registry import ModelRegistry
from extrai.core.extraction_config import ExtractionConfig
from extrai.core.base_llm_client import BaseLLMClient
from extrai.core.batch_models import BatchJobContext


class TestBatchPipelineCounting(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_model_registry = MagicMock(spec=ModelRegistry)
        self.mock_model_registry.root_model = MagicMock()
        self.mock_model_registry.root_model.__name__ = "RootModel"
        self.mock_model_registry.models = [self.mock_model_registry.root_model]
        self.mock_model_registry.llm_schema_json = "{}"
        self.mock_model_registry.model_map = {
            "RootModel": self.mock_model_registry.root_model
        }
        self.mock_model_registry.get_all_model_names.return_value = ["RootModel"]

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
            self.pipeline.client_rotator = MockClientRotator.return_value
            self.pipeline.context_preparer = MockContextPreparer.return_value
            self.pipeline.prompt_builder = MockBuilder.return_value
            self.pipeline.entity_counter = MockCounter.return_value
            self.pipeline.consensus = MockConsensus.return_value

    async def test_submit_batch_counting(self):
        # Setup mocks
        self.pipeline.entity_counter.prepare_counting_prompts.return_value = (
            "count_sys",
            "count_user",
        )
        self.pipeline.context_preparer.prepare_example = AsyncMock(return_value="")

        mock_batch_job = MagicMock()
        mock_batch_job.id = "counting_batch_id"

        mock_client_instance = self.pipeline.client_rotator.get_next_client.return_value
        mock_client_instance.create_batch_job = AsyncMock(return_value=mock_batch_job)

        # Test submit
        root_id = await self.pipeline.submit_batch(
            self.mock_session, ["doc"], count_entities=True
        )

        # Verify
        self.assertIsInstance(root_id, str)
        self.mock_session.add.assert_called_once()
        added_context = self.mock_session.add.call_args[0][0]

        self.assertEqual(added_context.current_batch_id, "counting_batch_id")
        self.assertEqual(added_context.status, BatchJobStatus.COUNTING_SUBMITTED)

        config = added_context.config
        self.assertTrue(config["count_entities"])

    async def test_process_batch_counting_transition(self):
        # Mock Context
        context = BatchJobContext(
            root_batch_id="root_1",
            current_batch_id="counting_batch_id",
            status=BatchJobStatus.COUNTING_SUBMITTED,
            input_strings=["doc"],
            config={"count_entities": True, "custom_extraction_process": "proc"},
        )
        self.mock_session.get.return_value = context

        # Mock get_status to return COUNTING_READY_TO_PROCESS
        # We patch retrieve_batch_job to return completed status
        mock_client_instance = self.pipeline.client_rotator.get_next_client.return_value
        mock_provider_job = MagicMock()
        mock_provider_job.status = "completed"
        mock_client_instance.retrieve_batch_job = AsyncMock(
            return_value=mock_provider_job
        )

        # Mock counting results
        counting_results_file = '{"id": "line1"}'
        mock_client_instance.retrieve_batch_results = AsyncMock(
            return_value=counting_results_file
        )
        mock_client_instance.extract_content_from_batch_response.return_value = (
            '{"RootModel": ["desc1"]}'
        )

        self.pipeline.entity_counter.validate_counts.return_value = {
            "RootModel": ["desc1"]
        }

        # Mock extraction batch submission
        mock_extraction_job = MagicMock()
        mock_extraction_job.id = "extraction_batch_id"

        # Ensure build_prompts returns expected tuple
        self.pipeline.prompt_builder.build_prompts.return_value = ("sys", "user")

        mock_client_instance.create_batch_job = AsyncMock(
            return_value=mock_extraction_job
        )

        # Test process
        result = await self.pipeline.process_batch("root_1", self.mock_session)

        # Verify transition
        self.assertEqual(result.status, BatchJobStatus.PROCESSING)
        self.assertEqual(result.message, "Transitioned from counting to extraction")

        # Verify context updated
        self.assertEqual(context.status, BatchJobStatus.SUBMITTED)
        self.assertEqual(context.current_batch_id, "extraction_batch_id")

        # Verify config updated with descriptions
        config = context.config
        self.assertIn("expected_entity_descriptions", config)
        self.assertEqual(config["expected_entity_descriptions"], ["[RootModel] desc1"])


if __name__ == "__main__":
    unittest.main()
